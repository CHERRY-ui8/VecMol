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
        
        dist = dist.clamp_min(self.start)
        dist = dist.clamp_max(self.stop)
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class CrossGraphEncoder(nn.Module):
    def __init__(self, n_atom_types, grid_size, code_dim, hidden_dim=128, num_layers=4, k_neighbors=32, atom_k_neighbors=8, 
                 dist_version='new', cutoff=5.0, additional_edge_feat=0, edge_dim=128, anchor_spacing=1.5):
        super().__init__()
        self.n_atom_types = n_atom_types
        self.grid_size = grid_size
        self.code_dim = code_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.k_neighbors = k_neighbors  # 原子 atom 和网格 grid 之间的连接数
        self.atom_k_neighbors = atom_k_neighbors  # 原子 atom 内部的连接数
        self.dist_version = dist_version
        self.cutoff = cutoff
        self.additional_edge_feat = additional_edge_feat
        self.edge_dim = edge_dim
        self.anchor_spacing = anchor_spacing  # 锚点间距

        # 注册grid坐标作为buffer（不需要训练）
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
        B = data.num_graphs # 一个batch中包含的分子数量
        N_total_atoms = data.num_nodes
        
        n_grid = self.grid_size ** 3

        # 1. 原子类型 one-hot，并填充到code_dim维度
        # 验证值范围
        if atoms_channel.numel() > 0:
            assert atoms_channel.min() >= 0, f"Negative values in atoms_channel: {atoms_channel.min()}"
            assert atoms_channel.max() < self.n_atom_types, f"atoms_channel max {atoms_channel.max()} >= n_atom_types {self.n_atom_types}"
        
        # 创建one-hot编码并填充到code_dim维度
        atom_feat = F.one_hot(atoms_channel.long(), num_classes=self.n_atom_types).float()  # [N_total_atoms, n_atom_types]
        if self.n_atom_types < self.code_dim:
            # 如果code_dim更大，用0填充剩余维度
            padding = torch.zeros(N_total_atoms, self.code_dim - self.n_atom_types, device=device)
            atom_feat = torch.cat([atom_feat, padding], dim=1)  # [N_total_atoms, code_dim]
        else:
            # 如果code_dim更小，截断多余维度（不建议，最好保证code_dim >= n_atom_types）
            atom_feat = atom_feat[:, :self.code_dim]

        # 2. 构造 grid 坐标
        grid_coords_flat = self.grid_coords.to(device).repeat(B, 1)  # [B*n_grid, 3]

        # 3. 初始化 grid codes 为0
        grid_codes = torch.zeros(B * n_grid, self.code_dim, device=device)  # [B*n_grid, code_dim]

        # 4. 拼接所有节点
        # 创建 grid 的 batch 索引
        grid_batch_idx = torch.arange(B, device=device).repeat_interleave(n_grid)  # [B*n_grid]

        # 拼接所有节点
        node_feats = torch.cat([atom_feat, grid_codes], dim=0)  # [(N_total_atoms + B*n_grid), code_dim]
        node_pos = torch.cat([atom_coords, grid_coords_flat], dim=0)  # [(N_total_atoms + B*n_grid), 3]

        # 5. 构建两个分离的图
        # 5.1 原子内部连接图（只连接原子之间）
        atom_edge_index = knn_graph(
            x=atom_coords,
            k=self.atom_k_neighbors,
            batch=atom_batch_idx,
            loop=False
        )
        
        # 5.2 原子-网格连接图
        # 使用knn构建原子到网格的连接
        # 在这里找的是 grid 的邻居，因为要更新的是 grid 的特征，而且这样每个 grid 都一定会有连边（不需要给所有grid加自环边了）
        grid_to_atom_edges = knn(
            x=atom_coords,            # source points  (atom)
            y=grid_coords_flat,       # target points (grid)
            k=self.k_neighbors,
            batch_x=atom_batch_idx,
            batch_y=grid_batch_idx
        )  # [2, E] 其中 E = k_neighbors * N_total_atoms (22240 = 32 * 695)
        
        # 添加调试信息
        # print(f"DEBUG: grid_to_atom_edges.shape: {grid_to_atom_edges.shape}")
        # print(f"DEBUG: grid_to_atom_edges[0].max(): {grid_to_atom_edges[0].max()}, grid_to_atom_edges[1].max(): {grid_to_atom_edges[1].max()}")
        # print(f"DEBUG: N_total_atoms: {N_total_atoms}, B*n_grid: {B*n_grid}")
        # print(f"DEBUG: atom_coords.shape: {atom_coords.shape}, grid_coords_flat.shape: {grid_coords_flat.shape}")
        
        # 修正边索引：网格节点索引需要加上N_total_atoms的偏移量
        # grid_to_atom_edges[0] 是网格索引，需要加上偏移量
        # grid_to_atom_edges[1] 是原子索引，保持不变
        grid_to_atom_edges[0] += N_total_atoms
        # 交换边的方向
        grid_to_atom_edges = torch.stack([grid_to_atom_edges[1], grid_to_atom_edges[0]], dim=0)
        
        # print(f"DEBUG: After correction - grid_to_atom_edges[0].max(): {grid_to_atom_edges[0].max()}, grid_to_atom_edges[1].max(): {grid_to_atom_edges[1].max()}")
        # print(f"DEBUG: atom_edge_index.shape: {atom_edge_index.shape}")
        # print(f"DEBUG: atom_edge_index[0].max(): {atom_edge_index[0].max()}, atom_edge_index[1].max(): {atom_edge_index[1].max()}")
                
        # 合并所有边
        edge_index = torch.cat([atom_edge_index, grid_to_atom_edges], dim=1)
        
        # print(f"DEBUG: Final edge_index.shape: {edge_index.shape}")
        # print(f"DEBUG: Final edge_index[0].max(): {edge_index[0].max()}, edge_index[1].max(): {edge_index[1].max()}")
        # print(f"DEBUG: node_pos.shape: {node_pos.shape}")

        # 6. GNN消息传递
        h = node_feats
        
        for layer in self.layers:
            h = layer(h, node_pos, edge_index)

        # 7. 只取 grid 部分并重塑为 [B, n_grid, code_dim]
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
        
        # 距离扩展
        if dist_version == 'new':
            self.distance_expansion = GaussianSmearing(start=0.0, stop=cutoff, num_gaussians=20, type_='exp')
            self.edge_emb = nn.Linear(20, edge_dim)
            self.use_gaussian_smearing = True
        elif dist_version == 'old':
            self.distance_expansion = GaussianSmearing(start=0.0, stop=cutoff, num_gaussians=edge_dim, type_='exp')
            self.edge_emb = nn.Linear(edge_dim, edge_dim)
            self.use_gaussian_smearing = True
        elif dist_version is None:
            # 向后兼容：不使用GaussianSmearing
            self.distance_expansion = None
            self.edge_emb = None
            self.use_gaussian_smearing = False
        else:
            raise NotImplementedError('dist_version notimplemented')
        
        # 修改MLP的输入维度，使其匹配实际输入
        if self.use_gaussian_smearing:
            self.mlp = nn.Sequential(
                nn.Linear(2*code_dim + edge_dim, hidden_dim, bias=True),
                nn.ReLU(),
                nn.Linear(hidden_dim, code_dim, bias=True)
            )
        else:
            # 向后兼容：使用原始维度
            self.mlp = nn.Sequential(
                nn.Linear(2*code_dim + 1, hidden_dim, bias=True),
                nn.ReLU(),
                nn.Linear(hidden_dim, code_dim, bias=True)
            )
        self.layernorm = nn.LayerNorm(code_dim)
        
        # 确保所有参数都设置了requires_grad=True
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, x, pos, edge_index):
        # x: [N, code_dim], pos: [N, 3]
        row, col = edge_index
        
        # 添加调试信息
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
                
        # 确保数据类型正确
        x = x.float()  # 确保x是float类型
        rel = rel.float()  # 确保rel是float类型
        dist = dist.float()  # 确保dist是float类型
        
        # 距离扩展和边嵌入
        if self.use_gaussian_smearing:
            dist_expanded = self.distance_expansion(dist)  # [E, num_gaussians]
            edge_features = self.edge_emb(dist_expanded)  # [E, edge_dim]
            msg_input = torch.cat([x[row], x[col], edge_features], dim=-1)  # [E, 2*code_dim+edge_dim]
        else:
            # 向后兼容：直接使用距离
            msg_input = torch.cat([x[row], x[col], dist], dim=-1)  # [E, 2*code_dim+1]
        
        msg = self.mlp(msg_input)  # [E, code_dim]
        
        # 使用propagate方法进行消息传递
        # 使用更精确的size参数：源节点和目标节点都是所有节点
        # 这样可以确保所有节点都参与聚合，同时避免维度不匹配
        aggr = self.propagate(edge_index, x=x, message=msg, size=(x.size(0), x.size(0)))  # [N, code_dim]
        
        # 残差连接
        x = x + aggr  # 残差连接
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


def analyze_distribution(dist_tensor, save_plot=False, plot_path="dist_distribution.png"):
    """
    Analyze distance distribution and provide cutoff suggestions
    
    Args:
        dist_tensor: Distance tensor [E] or [E, 1]
        save_plot: Whether to save distribution plot
        plot_path: Path to save the plot
    
    Returns:
        dict: Dictionary containing distribution statistics
    """
    import matplotlib.pyplot as plt
    
    # Ensure dist is 1D tensor
    if dist_tensor.dim() == 2:
        dist_tensor = dist_tensor.squeeze(-1)
    
    dist_np = dist_tensor.detach().cpu().numpy()
    
    # Basic statistics
    stats = {
        'min': float(dist_np.min()),
        'max': float(dist_np.max()),
        'mean': float(dist_np.mean()),
        'median': float(np.median(dist_np)),
        'std': float(dist_np.std()),
        'total_edges': len(dist_np)
    }
    
    # Percentile statistics
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    for p in percentiles:
        stats[f'p{p}'] = float(np.percentile(dist_np, p))
    
    # Edge ratio under different cutoffs
    cutoffs = [2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0]
    for cutoff in cutoffs:
        ratio = np.mean(dist_np < cutoff)
        stats[f'ratio_cutoff_{cutoff}'] = ratio
    
    # Print statistics
    print("=" * 50)
    print("Distance Distribution Analysis Report")
    print("=" * 50)
    print(f"Total edges: {stats['total_edges']}")
    print(f"Distance range: {stats['min']:.4f} - {stats['max']:.4f} Å")
    print(f"Mean: {stats['mean']:.4f} Å")
    print(f"Median: {stats['median']:.4f} Å")
    print(f"Standard deviation: {stats['std']:.4f} Å")
    print()
    
    print("Percentile statistics:")
    for p in percentiles:
        print(f"  {p}% percentile: {stats[f'p{p}']:.4f} Å")
    print()
    
    print("Edge ratio under different cutoff values:")
    for cutoff in cutoffs:
        ratio = stats[f'ratio_cutoff_{cutoff}']
        print(f"  Distance<{cutoff}Å: {ratio:.3f} ({ratio*100:.1f}%)")
    print()
    
    # Provide cutoff suggestions
    print("Cutoff suggestions:")
    if stats['p90'] < 5.0:
        print(f"  Recommended cutoff: {stats['p90']:.2f}Å (90% percentile)")
    elif stats['p95'] < 8.0:
        print(f"  Recommended cutoff: {stats['p95']:.2f}Å (95% percentile)")
    else:
        print(f"  Recommended cutoff: {stats['p99']:.2f}Å (99% percentile)")
    
    # Warning if distance distribution is too scattered
    if stats['std'] > stats['mean'] * 0.5:
        print("  Warning: Distance distribution is quite scattered, consider checking data or adjusting parameters")
    
    print("=" * 50)
    
    # Plot distribution
    if save_plot:
        plt.figure(figsize=(12, 8))
        
        # Main distribution plot
        plt.subplot(2, 2, 1)
        plt.hist(dist_np, bins=100, alpha=0.7, edgecolor='black')
        plt.xlabel('Distance (Å)')
        plt.ylabel('Frequency')
        plt.title('Distance Distribution Histogram')
        plt.axvline(stats['mean'], color='red', linestyle='--', label=f'Mean: {stats["mean"]:.2f}Å')
        plt.axvline(stats['median'], color='orange', linestyle='--', label=f'Median: {stats["median"]:.2f}Å')
        plt.legend()
        
        # Cumulative distribution plot
        plt.subplot(2, 2, 2)
        sorted_dist = np.sort(dist_np)
        cumulative = np.arange(1, len(sorted_dist) + 1) / len(sorted_dist)
        plt.plot(sorted_dist, cumulative)
        plt.xlabel('Distance (Å)')
        plt.ylabel('Cumulative Probability')
        plt.title('Cumulative Distribution Function')
        
        # Edge ratio under different cutoffs
        plt.subplot(2, 2, 3)
        ratios = [stats[f'ratio_cutoff_{c}'] for c in cutoffs]
        plt.bar(range(len(cutoffs)), ratios)
        plt.xlabel('Cutoff Value (Å)')
        plt.ylabel('Edge Ratio')
        plt.title('Edge Ratio Under Different Cutoffs')
        plt.xticks(range(len(cutoffs)), cutoffs)
        
        # Box plot
        plt.subplot(2, 2, 4)
        plt.boxplot(dist_np)
        plt.ylabel('Distance (Å)')
        plt.title('Distance Distribution Box Plot')
        
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Distribution plot saved to: {plot_path}")
    
    return stats