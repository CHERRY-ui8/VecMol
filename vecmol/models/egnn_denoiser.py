import torch
import torch.nn as nn
from torch_scatter import scatter
from torch_scatter import scatter
from torch_geometric.nn import MessagePassing, radius_graph
from vecmol.models.encoder import create_grid_coords


class EGNNDenoiserLayer(MessagePassing):
    def __init__(self, in_channels, hidden_channels, out_channels, cutoff=None, radius=None, time_emb_dim=None):
        super().__init__(aggr='mean')
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.cutoff = cutoff
        self.radius = radius
        self.time_emb_dim = time_emb_dim

        # Edge MLP with optional time embedding
        if time_emb_dim is not None:
            self.edge_mlp = nn.Sequential(
                nn.Linear(2 * in_channels + 1 + hidden_channels, hidden_channels),
                nn.SiLU(),
                nn.Linear(hidden_channels, hidden_channels),
                nn.SiLU(),
            )
        else:
            self.edge_mlp = nn.Sequential(
                nn.Linear(2 * in_channels + 1, hidden_channels),
                nn.SiLU(),
                nn.Linear(hidden_channels, hidden_channels),
                nn.SiLU(),
            )
        
        # Node feature update MLP
        self.node_mlp = nn.Sequential(
            nn.Linear(in_channels + hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, out_channels)
        )
        
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x, h, edge_index, t_emb=None):
        h = self.norm(h)
        
        row, col = edge_index
        
        # Bounds check
        max_idx = h.size(0)
        if not ((row < max_idx) & (col < max_idx)).all():
            raise ValueError(f"Edge indices out of bounds: max_idx={max_idx}, "
                           f"row_max={row.max().item()}, col_max={col.max().item()}")
        
        rel = x[col] - x[row]  # [E, 3] row -> col
        dist = torch.norm(rel, dim=-1, keepdim=True)  # [E, 1]
        
        h_i = h[row]
        h_j = h[col]
        
        if t_emb is not None and self.time_emb_dim is not None:
            t_emb_i = t_emb[row]  # [E, hidden_dim]
            edge_input = torch.cat([h_i, h_j, dist, t_emb_i], dim=-1)
        else:
            edge_input = torch.cat([h_i, h_j, dist], dim=-1)
        
        m_ij = self.edge_mlp(edge_input)  # [E, hidden_channels]
        
        if self.cutoff is not None:
            PI = torch.pi
            C = 0.5 * (torch.cos(dist.squeeze(-1) * PI / self.cutoff) + 1.0)
            C = C * (dist.squeeze(-1) <= self.cutoff) * (dist.squeeze(-1) >= 0.0)
            m_ij = m_ij * C.view(-1, 1)
        
        # Aggregate edge messages to target nodes (col)
        m_aggr = scatter(m_ij, edge_index[1], dim=0, dim_size=h.size(0), reduce='mean')
        h_delta = self.node_mlp(torch.cat([h, m_aggr], dim=-1))
        h = h + h_delta
        
        return h

    def message(self, message_j):  
        return message_j


class GNNDenoiser(nn.Module):
    def __init__(self, code_dim=1024, hidden_dim=128, num_layers=4, 
                 cutoff=None, radius=None, dropout=0.1, grid_size=8, 
                 anchor_spacing=2.0, use_radius_graph=True, device=None,
                 time_emb_dim=None):
        super().__init__()
        self.code_dim = code_dim
        self.hidden_dim = hidden_dim
        self.num_blocks = num_layers
        self.cutoff = cutoff
        self.radius = radius
        self.grid_size = grid_size
        self.anchor_spacing = anchor_spacing
        self.use_radius_graph = use_radius_graph
        self.time_emb_dim = time_emb_dim
        
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.input_projection = nn.Linear(code_dim, hidden_dim)
        
        if time_emb_dim is not None:
            self.time_projection = nn.Linear(time_emb_dim, hidden_dim)
        
        # GNN Layers (now integrated with residual blocks)
        self.blocks = nn.ModuleList([
            EGNNDenoiserLayer(
                in_channels=hidden_dim,
                hidden_channels=hidden_dim,
                out_channels=hidden_dim,
                radius=radius,
                time_emb_dim=time_emb_dim
            ) for _ in range(num_layers)
        ])
        
        self.output_projection = nn.Linear(hidden_dim, code_dim)
        
        # Pre-create grid coords in __init__
        self.register_buffer('grid_coords', 
                           create_grid_coords(
                               batch_size=1, 
                               grid_size=self.grid_size, 
                               device='cpu', 
                               anchor_spacing=self.anchor_spacing
                           ).squeeze(0))  # [n_grid, 3]
        
    def forward(self, y, t=None):
        # Input: 3D [batch_size, n_grid, code_dim] or 4D
        if y.dim() == 4:
            y = y.squeeze(1)
        elif y.dim() == 3:
            pass
        else:
            raise ValueError(f"Expected 3D or 4D input, got {y.dim()}D: {y.shape}")
        
        batch_size = y.size(0)
        n_grid = y.size(1)
        n_grid_actual = self.grid_size ** 3
        
        if n_grid != n_grid_actual:
            raise ValueError(
                f"Input grid size mismatch: expected n_grid={n_grid_actual} (grid_size={self.grid_size}^3), "
                f"but got n_grid={n_grid} from input shape {y.shape}. "
                f"Please ensure the input tensor has the correct second dimension."
            )
        
        grid_coords = self.grid_coords.to(self.device)
        grid_coords = grid_coords.unsqueeze(0).expand(batch_size, -1, -1)
        grid_coords = grid_coords.reshape(-1, 3)
        grid_batch = torch.arange(batch_size, device=self.device).repeat_interleave(n_grid)
        edge_index = radius_graph(
            x=grid_coords,
            r=self.radius,
            batch=grid_batch
        )
        
        h = self.input_projection(y)
        h = h.reshape(-1, self.hidden_dim)
        t_emb_broadcast = None
        if t is not None and self.time_emb_dim is not None:
            from vecmol.models.ddpm import get_time_embedding
            t_emb = get_time_embedding(t, self.time_emb_dim)  # [B, time_emb_dim]
            t_emb_proj = self.time_projection(t_emb)  # [B, hidden_dim]
            t_emb_broadcast = t_emb_proj.unsqueeze(1).expand(-1, n_grid_actual, -1)  # [B, n_grid, hidden_dim]
            t_emb_broadcast = t_emb_broadcast.reshape(-1, self.hidden_dim)  # [B*n_grid, hidden_dim]
        
        for block in self.blocks:
            h = block(grid_coords, h, edge_index, t_emb_broadcast)
        h = h.view(batch_size, n_grid, self.hidden_dim)
        output = self.output_projection(h)
        
        return output
