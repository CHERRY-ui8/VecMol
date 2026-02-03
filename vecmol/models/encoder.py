import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, knn_graph, knn, radius
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import math
import numpy as np


class GaussianSmearing(nn.Module):
    """Gaussian distance expansion for edge features."""
    
    def __init__(self, start=0.0, stop=10.0, num_gaussians=50, type_='linear'):
        super().__init__()
        self.start = start
        self.stop = stop
        if type_ == 'exp':
            offset = torch.exp(torch.linspace(start=np.log(start+1), end=np.log(stop+1), steps=num_gaussians)) - 1
        elif type_ == 'linear':
            offset = torch.linspace(start=start, end=stop, steps=num_gaussians)
        else:
            raise NotImplementedError('type_ must be either exp or linear')
        diff = torch.diff(offset)
        diff = torch.cat([diff[:1], diff])
        coeff = -0.5 / (diff**2)
        self.register_buffer('coeff', coeff)
        self.register_buffer('offset', offset)

    def forward(self, dist):
        """
        Args:
            dist: Tensor of shape [E] or [E, 1] containing distances
            
        Returns:
            Tensor of shape [E, num_gaussians] containing Gaussian expanded features
        """
        if dist.dim() == 2:
            dist = dist.squeeze(-1)
        
        # Ensure buffers are on the same device as the input tensor
        offset = self.offset.to(dist.device)
        coeff = self.coeff.to(dist.device)
        
        dist = dist.clamp_min(self.start)
        dist = dist.clamp_max(self.stop)
        dist = dist.view(-1, 1) - offset.view(1, -1)
        return torch.exp(coeff * torch.pow(dist, 2))


class CrossGraphEncoder(nn.Module):
    def __init__(self, n_atom_types, grid_size, code_dim, hidden_dim=128, num_layers=4, k_neighbors=32, atom_k_neighbors=8, 
                 dist_version='new', cutoff=5.0, additional_edge_feat=0, edge_dim=128, anchor_spacing=1.5):
        super().__init__()
        self.n_atom_types = n_atom_types
        self.grid_size = grid_size
        self.code_dim = code_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.k_neighbors = k_neighbors  # atom-grid connections
        self.atom_k_neighbors = atom_k_neighbors  # atom-atom connections
        self.dist_version = dist_version
        self.cutoff = cutoff
        self.additional_edge_feat = additional_edge_feat
        self.edge_dim = edge_dim
        self.anchor_spacing = anchor_spacing

        # Register grid coords as buffer (not trained)
        grid_coords = create_grid_coords(1, self.grid_size,
                        device="cpu", anchor_spacing=self.anchor_spacing).squeeze(0)  # [n_grid, 3]
        self.register_buffer('grid_coords', grid_coords)

        # GNN layers
        self.layers = nn.ModuleList([
            MessagePassingGNN(n_atom_types, code_dim, hidden_dim, edge_dim, dist_version, cutoff)
            for _ in range(num_layers)
        ])

    def forward(self, data):
        """
        data: torch_geometric.data.Batch object
              - data.pos: [N_total_atoms, 3], atom coordinates
              - data.x: [N_total_atoms], atom types
              - data.batch: [N_total_atoms], batch index for each atom
        """
        atom_coords = data.pos
        atoms_channel = data.x  # atom types
        atom_batch_idx = data.batch
        
        device = atom_coords.device
        B = data.num_graphs
        N_total_atoms = data.num_nodes
        n_grid = self.grid_size ** 3

        # 1. Atom type one-hot, pad to code_dim
        if atoms_channel.numel() > 0:
            assert atoms_channel.min() >= 0, f"Negative values in atoms_channel: {atoms_channel.min()}"
            assert atoms_channel.max() < self.n_atom_types, f"atoms_channel max {atoms_channel.max()} >= n_atom_types {self.n_atom_types}"
        
        atom_feat = F.one_hot(atoms_channel.long(), num_classes=self.n_atom_types).float()
        if self.n_atom_types < self.code_dim:
            padding = torch.zeros(N_total_atoms, self.code_dim - self.n_atom_types, device=device)
            atom_feat = torch.cat([atom_feat, padding], dim=1)
        else:
            atom_feat = atom_feat[:, :self.code_dim]

        # 2. Grid coords
        grid_coords_flat = self.grid_coords.to(device).repeat(B, 1)  # [B*n_grid, 3]
        # 3. Init grid codes to 0
        grid_codes = torch.zeros(B * n_grid, self.code_dim, device=device)
        # 4. Concatenate all nodes; grid batch index
        grid_batch_idx = torch.arange(B, device=device).repeat_interleave(n_grid)

        # Concatenate all nodes
        node_feats = torch.cat([atom_feat, grid_codes], dim=0)
        node_pos = torch.cat([atom_coords, grid_coords_flat], dim=0)

        # 5. Build two graphs: 5.1 atom-atom only
        atom_edge_index = knn_graph(
            x=atom_coords,
            k=self.atom_k_neighbors,
            batch=atom_batch_idx,
            loop=False
        )
        
        # 5.2 Atom-grid: knn from grid to atoms (grid as query so each grid has edges)
        grid_to_atom_edges = knn(
            x=atom_coords,            # source points  (atom)
            y=grid_coords_flat,       # target points (grid)
            k=self.k_neighbors,
            batch_x=atom_batch_idx,
            batch_y=grid_batch_idx
        )  # [2, E] E = k_neighbors * N_total_atoms

        # Debug
        # print(f"DEBUG: grid_to_atom_edges.shape: {grid_to_atom_edges.shape}")
        # print(f"DEBUG: grid_to_atom_edges[0].max(): {grid_to_atom_edges[0].max()}, grid_to_atom_edges[1].max(): {grid_to_atom_edges[1].max()}")
        # print(f"DEBUG: N_total_atoms: {N_total_atoms}, B*n_grid: {B*n_grid}")
        # print(f"DEBUG: atom_coords.shape: {atom_coords.shape}, grid_coords_flat.shape: {grid_coords_flat.shape}")
        
        # Shift grid indices by N_total_atoms; [0]=grid, [1]=atom
        grid_to_atom_edges[0] += N_total_atoms
        grid_to_atom_edges = torch.stack([grid_to_atom_edges[1], grid_to_atom_edges[0]], dim=0)  # swap direction
        
        # print(f"DEBUG: After correction - grid_to_atom_edges[0].max(): {grid_to_atom_edges[0].max()}, grid_to_atom_edges[1].max(): {grid_to_atom_edges[1].max()}")
        # print(f"DEBUG: atom_edge_index.shape: {atom_edge_index.shape}")
        # print(f"DEBUG: atom_edge_index[0].max(): {atom_edge_index[0].max()}, atom_edge_index[1].max(): {atom_edge_index[1].max()}")
                
        # Merge all edges
        edge_index = torch.cat([atom_edge_index, grid_to_atom_edges], dim=1)
        
        # print(f"DEBUG: Final edge_index.shape: {edge_index.shape}")
        # print(f"DEBUG: Final edge_index[0].max(): {edge_index[0].max()}, edge_index[1].max(): {edge_index[1].max()}")
        # print(f"DEBUG: node_pos.shape: {node_pos.shape}")

        # 6. GNN message passing
        h = node_feats
        
        for layer in self.layers:
            h = layer(h, node_pos, edge_index)

        # 7. Take grid part and reshape to [B, n_grid, code_dim]
        grid_h = h[N_total_atoms:].reshape(B, n_grid, self.code_dim)
        return grid_h  # [B, n_grid, code_dim]
    

class MessagePassingGNN(MessagePassing):
    def __init__(self, atom_feat_dim, code_dim, hidden_dim, edge_dim, dist_version, cutoff=5.0):
        super().__init__(aggr='mean')
        self.code_dim = code_dim
        self.hidden_dim = hidden_dim
        self.edge_dim = edge_dim
        self.dist_version = dist_version
        self.cutoff = cutoff
        
        # Distance expansion
        if dist_version == 'new':
            self.distance_expansion = GaussianSmearing(start=0.0, stop=cutoff, num_gaussians=20, type_='exp')
            self.edge_emb = nn.Linear(20, edge_dim)
            self.use_gaussian_smearing = True
        elif dist_version == 'old':
            self.distance_expansion = GaussianSmearing(start=0.0, stop=cutoff, num_gaussians=edge_dim, type_='exp')
            self.edge_emb = nn.Linear(edge_dim, edge_dim)
            self.use_gaussian_smearing = True
        elif dist_version is None:
            # Backward compat: no GaussianSmearing
            self.distance_expansion = None
            self.edge_emb = None
            self.use_gaussian_smearing = False
        else:
            raise NotImplementedError('dist_version notimplemented')
        
        # Match MLP input dim to actual input
        if self.use_gaussian_smearing:
            self.mlp = nn.Sequential(
                nn.Linear(2*code_dim + edge_dim, hidden_dim, bias=True),
                nn.ReLU(),
                nn.Linear(hidden_dim, code_dim, bias=True)
            )
        else:
            # Backward compat: original dim
            self.mlp = nn.Sequential(
                nn.Linear(2*code_dim + 1, hidden_dim, bias=True),
                nn.ReLU(),
                nn.Linear(hidden_dim, code_dim, bias=True)
            )
        self.layernorm = nn.LayerNorm(code_dim)
        
        # Ensure requires_grad=True for all params
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, x, pos, edge_index):
        # x: [N, code_dim], pos: [N, 3]
        row, col = edge_index
        
        # Debug
        if torch.any(row >= pos.size(0)) or torch.any(col >= pos.size(0)):
            print(f"ERROR: Index out of bounds!")
            print(f"pos.size(0): {pos.size(0)}")
            print(f"row.max(): {row.max()}, col.max(): {col.max()}")
            print(f"row.min(): {row.min()}, col.min(): {col.min()}")
            print(f"edge_index.shape: {edge_index.shape}")
            print(f"x.shape: {x.shape}")
            raise ValueError("Index out of bounds in edge_index")
        
        rel = pos[row] - pos[col]  # [E, 3]
        dist = torch.norm(rel, dim=-1, keepdim=True)  # [E, 1]
                
        x = x.float()
        rel = rel.float()
        dist = dist.float()

        # Distance expansion and edge embedding
        if self.use_gaussian_smearing:
            dist_expanded = self.distance_expansion(dist)  # [E, num_gaussians]
            # Ensure edge_emb on correct device
            if self.edge_emb.weight.device != dist_expanded.device:
                self.edge_emb = self.edge_emb.to(dist_expanded.device)
            edge_features = self.edge_emb(dist_expanded)  # [E, edge_dim]
            msg_input = torch.cat([x[row], x[col], edge_features], dim=-1)  # [E, 2*code_dim+edge_dim]
        else:
            # Backward compat: use distance directly
            msg_input = torch.cat([x[row], x[col], dist], dim=-1)  # [E, 2*code_dim+1]
        
        # Ensure mlp on correct device
        if self.mlp[0].weight.device != msg_input.device:
            self.mlp = self.mlp.to(msg_input.device)
        msg = self.mlp(msg_input)  # [E, code_dim]
        
        # Message passing with size for aggregation
        aggr = self.propagate(edge_index, x=x, message=msg, size=(x.size(0), x.size(0)))  # [N, code_dim]
        x = x + aggr
        # Ensure layernorm on correct device
        if self.layernorm.weight.device != x.device:
            self.layernorm = self.layernorm.to(x.device)
        x = self.layernorm(x)
        return x
    
    def message(self, message):
        """Message function for MessagePassing"""
        return message

def create_grid_coords(batch_size, grid_size, device=None, anchor_spacing=1.5):
    """Create anchor grid coordinates for a given grid size.
    
    Args:
        batch_size: Number of batches
        grid_size: Size of the anchor grid (will create grid_size^3 anchor points)
        device: Optional device to place the tensor on. If None, uses the default device.
        anchor_spacing: Distance between anchor points in Angstroms (default: 2.0)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif not isinstance(device, torch.device):
        device = torch.device(device)
        
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    
    # Calculate the total span for anchor grid in Angstroms
    # For anchor grid, we want to cover a reasonable molecular space
    total_span = (grid_size - 1) * anchor_spacing
    half_span = total_span / 2
    
    # Create anchor grid points in real space (Angstroms)
    grid_1d = torch.linspace(-half_span, half_span, grid_size, device=device)
    mesh = torch.meshgrid(grid_1d, grid_1d, grid_1d, indexing='ij')
    coords = torch.stack(mesh, dim=-1).reshape(-1, 3)  # [n_grid, 3]
    coords = coords.unsqueeze(0).expand(batch_size, -1, -1)  # [B, n_grid, 3]
    return coords