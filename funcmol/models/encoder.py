import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, knn_graph, knn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader


class CrossGraphEncoder(nn.Module):
    def __init__(self, n_atom_types, grid_size, code_dim, hidden_dim=128, num_layers=4, k_neighbors=32, atom_k_neighbors=8):
        super().__init__()
        self.n_atom_types = n_atom_types
        self.grid_size = grid_size
        self.code_dim = code_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.k_neighbors = k_neighbors  # 原子 atom 和网格 grid 之间的连接数
        self.atom_k_neighbors = atom_k_neighbors  # 原子 atom 内部的连接数

        # 注册grid坐标作为buffer（不需要训练）
        grid_coords = create_grid_coords(1, self.grid_size).squeeze(0)  # [n_grid, 3]
        self.register_buffer('grid_coords', grid_coords)

        # GNN layers
        self.layers = nn.ModuleList([
            MessagePassingGNN(n_atom_types, code_dim, hidden_dim)
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

        # 5. 构建两个分离的图 TODO
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
        
        # 修正边索引：网格节点索引需要加上N_total_atoms的偏移量
        # grid_to_atom_edges[0] 是网格索引，需要加上偏移量
        # grid_to_atom_edges[1] 是原子索引，保持不变
        grid_to_atom_edges[0] += N_total_atoms
        # 交换边的方向
        grid_to_atom_edges = torch.stack([
            grid_to_atom_edges[1],
            grid_to_atom_edges[0]
        ], dim=0)
                
        # 合并所有边
        edge_index = torch.cat([atom_edge_index, grid_to_atom_edges], dim=1)
        
        # 6. GNN消息传递
        h = node_feats
        
        for layer in self.layers:
            h = layer(h, node_pos, edge_index)

        # 7. 只取 grid 部分并重塑为 [B, n_grid, code_dim]
        grid_h = h[N_total_atoms:].reshape(B, n_grid, self.code_dim)
        return grid_h  # [B, n_grid, code_dim]
    

class MessagePassingGNN(MessagePassing):
    def __init__(self, atom_feat_dim, code_dim, hidden_dim):
        super().__init__(aggr='mean')
        self.code_dim = code_dim
        self.hidden_dim = hidden_dim
        # 修改MLP的输入维度，使其匹配实际输入
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
        rel = pos[row] - pos[col]  # [E, 3]
        dist = torch.norm(rel, dim=-1, keepdim=True)  # [E, 1]
        
        # 确保数据类型正确
        x = x.float()  # 确保x是float类型
        rel = rel.float()  # 确保rel是float类型
        dist = dist.float()  # 确保dist是float类型
        
        msg_input = torch.cat([x[row], x[col], dist], dim=-1)  # [E, 2*code_dim+1]
        
        msg = self.mlp(msg_input)  # [E, code_dim]
        aggr = self.propagate(edge_index, x=x, message=msg)  # [N, code_dim]
        x = x + aggr  # 残差连接
        x = self.layernorm(x)
        return x
    
    def message(self, message):
        """Message function for MessagePassing"""
        return message

def create_grid_coords(batch_size, grid_size, device=None):
    """Create grid coordinates for a given grid size.
    
    Args:
        batch_size: Number of batches
        grid_size: Size of the grid (will create grid_size^3 points)
        device: Optional device to place the tensor on. If None, uses the default device.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif not isinstance(device, torch.device):
        device = torch.device(device)
        
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
        
    grid_1d = torch.linspace(-1, 1, grid_size, device=device)
    mesh = torch.meshgrid(grid_1d, grid_1d, grid_1d, indexing='ij')
    coords = torch.stack(mesh, dim=-1).reshape(-1, 3)  # [n_grid, 3]
    coords = coords.unsqueeze(0).expand(batch_size, -1, -1)  # [B, n_grid, 3]
    return coords


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels,
        level_channels=[32, 64, 128],
        bottleneck_channel=1024,
        smaller=False
    ):
        super(Encoder, self).__init__()
        self.enc_blocks = nn.ModuleList()
        for i in range(len(level_channels)):
            in_ch = in_channels if i == 0 else level_channels[i - 1]
            out_ch = level_channels[i]
            self.enc_blocks.append(
                Conv3DBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    bottleneck=False,
                    smaller=smaller
                )
            )
        self.bottleNeck = Conv3DBlock(
            in_channels=out_ch,
            out_channels=bottleneck_channel,
            bottleneck=True,
            smaller=smaller
        )
        self.fc = nn.Linear(bottleneck_channel, bottleneck_channel)

    def forward(self, voxels):
        # encoder
        out = voxels
        for block in self.enc_blocks:
            out, _ = block(out)
        out, _ = self.bottleNeck(out)

        # pooling
        out = torch.nn.functional.avg_pool3d(out, out.size()[2:])
        out = out.squeeze()
        out = self.fc(out)

        return out