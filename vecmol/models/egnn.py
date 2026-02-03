import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, knn_graph, knn, radius
from vecmol.models.encoder import create_grid_coords
import math

# torch.compile compatibility
try:
    from torch._dynamo import disable
except ImportError:
    def disable(fn):
        return fn

class EGNNLayer(MessagePassing):
    def __init__(self, in_channels, hidden_channels, out_channels, radius=2.0, out_x_dim=1, cutoff=None, k_neighbors=32):
        """
            EGNN equivariant layer: edge input h_i, h_j, ||x_i-x_j||; scalar-weighted direction for coord update; cutoff for distance filter; radius/k_neighbors for graph.
        """
        super().__init__(aggr='mean')
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.radius = radius
        self.out_x_dim = out_x_dim
        self.cutoff = cutoff
        self.k_neighbors = k_neighbors

        # Edge MLP: [h_i, h_j, ||x_i-x_j||] -> hidden_channels
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * in_channels + 1, hidden_channels),  # 2*512+1=1025 -> 512
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
        )
        # Coord MLP: hidden_channels -> scalar
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_channels, out_x_dim),
        )
        # Node feature update
        self.node_mlp = nn.Sequential(
            nn.Linear(in_channels + hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, out_channels)
        )

    def forward(self, x, h, edge_index):
        """
            x: [N, 3] coords; h: [N, in_channels] node features; edge_index: [2, E]. Returns h_new, x_new.
        """
        row, col = edge_index
        rel = x[row] - x[col]  # [E, 3]
        dist = torch.norm(rel, dim=-1, keepdim=True)  # [E, 1]
        
        h_i = h[row]
        h_j = h[col]
        edge_input = torch.cat([h_i, h_j, dist], dim=-1)  # [E, 2*in_channels+1]
        m_ij = self.edge_mlp(edge_input)  # [E, hidden_channels]
        
        # Apply cutoff filter
        if self.cutoff is not None:
            PI = torch.pi
            C = 0.5 * (torch.cos(dist.squeeze(-1) * PI / self.cutoff) + 1.0)
            C = C * (dist.squeeze(-1) <= self.cutoff) * (dist.squeeze(-1) >= 0.0)
            m_ij = m_ij * C.view(-1, 1)
        
        # Coord update: scalar * direction
        coord_coef = self.coord_mlp(m_ij)  # [E, out_x_dim]
        direction = rel / (dist + 1e-8)  # unit vector [E, 3]
        if self.out_x_dim == 1:
            coord_message = coord_coef * direction  # [E, 3]
            delta_x = self.propagate(edge_index, x=x, message=coord_message, size=(x.size(0), x.size(0)))  # [N, 3]
            x_new = x + delta_x
        else:
            coord_message = coord_coef[..., None] * direction[:, None, :]  # [E, out_x_dim, 3]
            x = x[:, None, :]  # [N, 1, 3]
            delta_x = self.propagate(edge_index, x=None, message=coord_message.view(len(row), -1), size=(x.size(0), x.size(0)))
            delta_x = delta_x.view(x.size(0), self.out_x_dim, 3)
            x_new = x + delta_x
        
        # Node feature update
        m_aggr = self.propagate(edge_index, x=x, message=m_ij, size=(x.size(0), x.size(0)))
        h_delta = self.node_mlp(torch.cat([h, m_aggr], dim=-1))
        h_new = h + h_delta
        return h_new, x_new

    def message(self, message):
        return message

class EGNNVectorField(nn.Module):
    def __init__(self, 
                 grid_size: int = 8,
                 hidden_dim: int = 128,
                 num_layers: int = 3,
                 radius: float = 2.0,
                 n_atom_types: int = 5,
                 code_dim: int = 128,
                 cutoff: float = None,
                 anchor_spacing: float = 2.0,
                 k_neighbors: int = 32):
        """
        Initialize the EGNN Vector Field model.
        Each query point predicts a 3D vector for every atom type.

        Args:
            grid_size (int): Size of the grid for G_L (L in the paper)
            hidden_dim (int): Dimension of hidden layers
            num_layers (int): Number of EGNN layers
            radius (float): Radius for building graph connections (replaces k_neighbors)
            n_atom_types (int): Number of atom types
            code_dim (int): Dimension of the latent code and node features
            cutoff (float, optional): Cutoff distance for gradient decay. If None, uses radius.
                                     When cutoff=radius, gradient decays to 0 at radius distance.
            anchor_spacing (float): Spacing between anchor points
            device (torch.device, optional): The device to run the model on.
        """
        super().__init__()
        self.grid_size = grid_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.radius = radius
        self.n_atom_types = n_atom_types
        self.code_dim = code_dim
        self.cutoff = cutoff if cutoff is not None else radius
        self.k_neighbors = k_neighbors

        # batch_size=1: store one grid coords copy
        self.register_buffer('grid_points',
                create_grid_coords(
                    batch_size=1, grid_size=self.grid_size, device='cpu', anchor_spacing=anchor_spacing
                ).squeeze(0)
            )
        # self.grid_features = nn.Parameter(...)  # TODO: init?

        # type embedding
        self.type_embed = nn.Embedding(self.n_atom_types, self.code_dim)

        # Create EGNN layers
        self.layers = nn.ModuleList([
            EGNNLayer(
                in_channels=code_dim,
                hidden_channels=hidden_dim,
                out_channels=code_dim,
                radius=radius,
                cutoff=cutoff,
                k_neighbors=k_neighbors
            ) for _ in range(num_layers)
        ])
        
        # Field prediction layer
        self.field_layer = EGNNLayer(
            in_channels=code_dim,
            hidden_channels=hidden_dim,
            out_channels=code_dim,
            radius=radius,
            out_x_dim=n_atom_types,
            cutoff=cutoff,
            k_neighbors=k_neighbors
        )
        
    def forward(self, query_points, codes):
        """
        Args:
            query_points: [B, N, 3] sampled query positions
            codes: [B, G, C] latent features for each anchor grid (G = grid_size³)

        Returns:
            vector_field: [B, N, T, 3] vector field at each query point for each atom type
        """
        batch_size = query_points.size(0)
        n_points = query_points.size(1)
        device = query_points.device
        grid_points = self.grid_points.to(device)  # [grid_size**3, 3]
        n_grid = grid_points.size(0)

        # Flatten query_points immediately
        query_points = query_points.reshape(-1, 3).float()  # [B * N, 3]        
        # grid_coords = grid_points.unsqueeze(0).expand(batch_size, -1, -1).reshape(-1, 3)  # [B * grid_size**3, 3]
        grid_coords = grid_points.repeat(batch_size, 1)  # [B * grid_size**3, 3]
        n_points_total = query_points.size(0)
        
        # 1. Init node features
        query_features = torch.zeros(n_points_total, self.code_dim, device=device)
        grid_features = codes.reshape(-1, self.code_dim)  # [B * grid_size**3, code_dim]
        
        # 3. Merge nodes
        combined_features = torch.cat([query_features, grid_features], dim=0)
        combined_coords = torch.cat([query_points, grid_coords], dim=0)

        # 5. Build edges (knn)
        query_batch = torch.arange(batch_size, device=device).repeat_interleave(n_points)
        grid_batch = torch.arange(batch_size, device=device).repeat_interleave(n_grid)

        # knn for query -> grid edges
        # edge_grid_query = radius(
        #     x=grid_coords,
        #     y=query_points,
        #     r=self.radius,
        #     batch_x=grid_batch,
        #     batch_y=query_batch
        # ) # [2, E]

        edge_grid_query = knn(
            x=grid_coords,
            y=query_points,
            k=self.k_neighbors,
            batch_x=grid_batch,
            batch_y=query_batch
        ) # [2, E] where E = k_neighbors * n_points_total

        edge_grid_query[1] += n_points_total # bias
        
        edge_grid_query = torch.stack([edge_grid_query[1], edge_grid_query[0]], dim=0)  # swap edge direction
        
        # 7. EGNN message passing
        h = combined_features
        x = combined_coords
        for layer in self.layers:
            h, x = layer(x, h, edge_grid_query)

        # 8. Predict vector field
        
        _, predicted_sources = self.field_layer(x, h, edge_grid_query)  # [total_nodes, n_atom_types, 3]
        predicted_sources = predicted_sources[:n_points_total]  # keep only query points
        predicted_sources = predicted_sources.view(batch_size, n_points, self.n_atom_types, 3)

        # Compute residual vectors relative to query points
        query_points_reshaped = query_points.view(batch_size, n_points, 3)
        vector_field = predicted_sources - query_points_reshaped[:, :, None, :]  # [B, N, n_atom_types, 3]

        return vector_field