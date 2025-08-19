import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, knn_graph, knn, radius
from funcmol.models.encoder import create_grid_coords
import math


class EGNNDenoiserLayer(MessagePassing):
    def __init__(self, in_channels, hidden_channels, out_channels, k_neighbors=8, cutoff=None, radius=None):
        super().__init__(aggr='mean')
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.k_neighbors = k_neighbors
        self.cutoff = cutoff
        self.radius = radius

        # 边特征MLP
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * in_channels + 1, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
        )
        
        # 节点特征更新MLP
        self.node_mlp = nn.Sequential(
            nn.Linear(in_channels + hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, out_channels)
        )

    def forward(self, x, h, edge_index):
        row, col = edge_index
        rel = x[row] - x[col]
        dist = torch.norm(rel, dim=-1, keepdim=True)
        
        h_i = h[row]
        h_j = h[col]
        edge_input = torch.cat([h_i, h_j, dist], dim=-1)
        m_ij = self.edge_mlp(edge_input)
        
        if self.cutoff is not None:
            PI = torch.pi
            C = 0.5 * (torch.cos(dist.squeeze(-1) * PI / self.cutoff) + 1.0)
            C = C * (dist.squeeze(-1) <= self.cutoff) * (dist.squeeze(-1) >= 0.0)
            m_ij = m_ij * C.view(-1, 1)
        
        m_aggr = self.propagate(edge_index, x=x, message=m_ij, size=(x.size(0), x.size(0)))
        h_delta = self.node_mlp(torch.cat([h, m_aggr], dim=-1))
        h_new = h + h_delta
        return h_new

    def message(self, message):
        return message


class GNNResBlock(nn.Module):
    def __init__(self, code_dim, hidden_dim, k_neighbors=8, cutoff=None, radius=None, dropout=0.1):
        super().__init__()
        self.egnn_layer = EGNNDenoiserLayer(
            in_channels=code_dim,
            hidden_channels=hidden_dim,
            out_channels=code_dim,
            k_neighbors=k_neighbors,
            cutoff=cutoff,
            radius=radius
        )
        
        self.mlp = nn.Sequential(
            nn.Linear(code_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, code_dim)
        )
        
        self.norm1 = nn.LayerNorm(code_dim)
        self.norm2 = nn.LayerNorm(code_dim)

    def forward(self, x, h, edge_index):
        h_residual = h
        h = self.norm1(h)
        h = self.egnn_layer(x, h, edge_index)
        h = h_residual + h
        
        h_residual = h
        h = self.norm2(h)
        h = self.mlp(h)
        h = h_residual + h
        
        return h


class GNNDenoiser(nn.Module):
    def __init__(self, code_dim=1024, hidden_dim=2048, num_blocks=4, k_neighbors=8, 
                 cutoff=None, radius=None, dropout=0.1, grid_size=8, 
                 anchor_spacing=2.0, use_radius_graph=True, device=None):
        super().__init__()
        self.code_dim = code_dim
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        self.k_neighbors = k_neighbors
        self.cutoff = cutoff
        self.radius = radius
        self.grid_size = grid_size
        self.anchor_spacing = anchor_spacing
        self.use_radius_graph = use_radius_graph
        
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        # 创建网格坐标
        self.register_buffer('grid_points', 
                           create_grid_coords(batch_size=1, grid_size=self.grid_size, 
                                            device=device, anchor_spacing=anchor_spacing).squeeze(0))

        # 输入投影层
        self.input_projection = nn.Linear(code_dim, hidden_dim)
        
        # GNN ResBlocks
        self.blocks = nn.ModuleList([
            GNNResBlock(
                code_dim=hidden_dim,
                hidden_dim=hidden_dim,
                k_neighbors=k_neighbors,
                cutoff=cutoff,
                radius=radius,
                dropout=dropout
            ) for _ in range(num_blocks)
        ])
        
        # 输出投影层
        self.output_projection = nn.Linear(hidden_dim, code_dim)
        
        self._init_weights()
        
        # 将整个模型移动到指定设备
        self.to(device)

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _build_graph(self, batch_size):
        device = self.grid_points.device
        n_grid = self.grid_points.size(0)
        
        grid_coords = self.grid_points.unsqueeze(0).expand(batch_size, -1, -1)
        grid_coords = grid_coords.reshape(-1, 3)
        
        grid_batch = torch.arange(batch_size, device=device).repeat_interleave(n_grid)
        
        if self.use_radius_graph and self.radius is not None:
            edge_index = radius(
                x=grid_coords,
                y=grid_coords,
                r=self.radius,
                batch_x=grid_batch,
                batch_y=grid_batch,
                max_num_neighbors=self.k_neighbors
            )
        else:
            edge_index = knn_graph(
                x=grid_coords,
                k=self.k_neighbors,
                batch=grid_batch,
                loop=False
            )
        
        return edge_index, grid_coords

    def forward(self, y):
        # 处理输入维度
        if y.dim() == 4:  # (batch_size, 1, n_grid, code_dim)
            y = y.squeeze(1)  # (batch_size, n_grid, code_dim)
        
        batch_size = y.size(0)
        n_grid = y.size(1)  # 使用实际的grid数量，而不是计算值
        
        edge_index, grid_coords = self._build_graph(batch_size)
        
        # 直接对每个grid point进行投影
        h = self.input_projection(y)  # (batch_size, n_grid, hidden_dim)
        h = h.reshape(-1, self.hidden_dim)  # (batch_size * n_grid, hidden_dim)
        
        for block in self.blocks:
            h = block(grid_coords, h, edge_index)
        
        h = h.view(batch_size, n_grid, self.hidden_dim)  # (batch_size, n_grid, hidden_dim)
        
        # 对每个grid point进行输出投影
        output = self.output_projection(h)  # (batch_size, n_grid, code_dim)
        return output
