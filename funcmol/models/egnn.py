import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

class EGNNLayer(MessagePassing):
    def __init__(self, in_channels, hidden_channels, out_channels, k_neighbors=8):
        super(EGNNLayer, self).__init__(aggr='mean')
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.k_neighbors = k_neighbors
        
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * in_channels + 1, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(in_channels + hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, out_channels)
        )
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, 1)
        )

    def forward(self, x, pos, edge_index, batch_size, num_points):
        edge_index, _ = add_self_loops(edge_index)
        row, col = edge_index
        edge_attr = torch.cat([x[row], x[col], 
                             torch.norm(pos[row] - pos[col], dim=1, keepdim=True)], dim=1)
        out = self.propagate(edge_index, x=x, pos=pos, edge_attr=edge_attr)
        out = self.node_mlp(torch.cat([x, out], dim=1))
        coord_update = self.update_coords(x, pos, edge_index, batch_size, num_points)
        return out, coord_update

    def message(self, x_j, pos_i, pos_j, edge_attr):
        msg = self.edge_mlp(edge_attr)
        return msg

    def update_coords(self, h, pos, edge_index, batch_size, num_points):
        row, col = edge_index
        mask = row != col
        row, col = row[mask], col[mask]
        edge_attr = torch.cat([h[row], h[col], 
                             torch.norm(pos[row] - pos[col], dim=1, keepdim=True)], dim=1)
        msg = self.edge_mlp(edge_attr)
        coord_update = self.coord_mlp(msg) * (pos[col] - pos[row])
        num_edges = coord_update.size(0)
        total_nodes = batch_size * num_points
        actual_k_neighbors = min(self.k_neighbors, num_edges // total_nodes)
        target_edges = total_nodes * actual_k_neighbors
        # print(f"num_edges={num_edges}, actual_k_neighbors={actual_k_neighbors}, target_edges={target_edges}")
        if num_edges > target_edges:
            coord_update = coord_update[:target_edges]
        elif num_edges < target_edges:
            padding = torch.zeros(target_edges - num_edges, 3, device=coord_update.device)
            coord_update = torch.cat([coord_update, padding], dim=0)
        coord_update = coord_update.view(batch_size, num_points, actual_k_neighbors, -1)
        coord_update = coord_update.mean(dim=2)
        return coord_update.reshape(-1, 3)

class EGNNVectorField(nn.Module):
    def __init__(self, 
                 grid_size: int = 8,
                 feature_dim: int = 64,
                 hidden_dim: int = 128,
                 num_layers: int = 4,
                 k_neighbors: int = 8,
                 n_atom_types: int = 5):
        super().__init__()
        self.grid_size = grid_size
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.k_neighbors = k_neighbors
        self.n_atom_types = n_atom_types
        
        self.register_buffer('grid_points', self._create_grid_points())
        self.anchor_features = nn.Parameter(
            torch.randn(grid_size**3, feature_dim)
        )
        
        self.layers = nn.ModuleList([
            EGNNLayer(
                in_channels=feature_dim if i == 0 else hidden_dim,
                hidden_channels=hidden_dim,
                out_channels=hidden_dim if i < num_layers-1 else self.n_atom_types * 3,
                k_neighbors=k_neighbors
            ) for i in range(num_layers)
        ])
    
    def _create_grid_points(self):
        points = torch.meshgrid(
            torch.linspace(-1, 1, self.grid_size),
            torch.linspace(-1, 1, self.grid_size),
            torch.linspace(-1, 1, self.grid_size)
        )
        points = torch.stack(points, dim=-1).reshape(-1, 3)
        return points
    
    def build_knn_graph(self, pos):
        batch_size, n_points, _ = pos.shape
        pos = pos.contiguous()
        pos_flat = pos.view(batch_size * n_points, 3)
        unique_pos, inverse_indices = torch.unique(pos_flat, dim=0, return_inverse=True)
        num_unique = unique_pos.size(0)
        # print(f"unique_points={num_unique}, total_points={batch_size * n_points}")
        if num_unique < batch_size * n_points:
            # print(f"Warning: Found {batch_size * n_points - num_unique} duplicate points")
            pass
        dist = torch.cdist(unique_pos, unique_pos)
        if torch.isnan(dist).any() or torch.isinf(dist).any():
            raise ValueError("Distance matrix contains NaN or Inf values")
        mask = torch.eye(num_unique, device=pos.device).bool()
        dist = dist.masked_fill(mask, float('inf'))
        _, topk_indices = torch.topk(dist, min(self.k_neighbors, num_unique-1), dim=1, largest=False)
        row = torch.arange(num_unique, device=pos.device).view(-1, 1).repeat(1, min(self.k_neighbors, num_unique-1))
        col = topk_indices
        edge_index = torch.stack([row, col], dim=0).reshape(2, -1)
        edge_index = torch.unique(edge_index, dim=1)
        # print(f"edge_index.shape={edge_index.shape}")
        return edge_index
    
    def forward(self, query_points):
        batch_size, num_points, _ = query_points.shape
        if torch.isnan(query_points).any() or torch.isinf(query_points).any():
            raise ValueError("query_points contains NaN or Inf values")
        # print(f"query_points.shape={query_points.shape}")
        grid_points = self.grid_points.repeat(batch_size, 1).reshape(batch_size, self.grid_size**3, 3)
        anchor_features = self.anchor_features.repeat(batch_size, 1).reshape(batch_size, self.grid_size**3, self.feature_dim)
        all_points = torch.cat([query_points, grid_points], dim=1)
        edge_index = self.build_knn_graph(all_points)
        query_features = torch.zeros(batch_size * num_points, self.feature_dim, device=query_points.device)
        all_features = torch.cat([query_features, anchor_features.view(-1, self.feature_dim)], dim=0)
        query_points_flat = all_points.reshape(-1, 3)
        h = all_features
        total_points = num_points + self.grid_size**3
        for layer in self.layers:
            h, coord_update = layer(h, query_points_flat, edge_index, batch_size, total_points)
            query_points_flat = query_points_flat + coord_update
        h = h.reshape(batch_size, total_points, -1)
        h_query = h[:, :num_points, :]
        return torch.tanh(h_query)