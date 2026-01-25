"""Steerable E(3) Equivariant Encoder for Neural Field
使用SEGNN将原子信息编码到网格锚点
"""

import torch
import torch.nn as nn
from torch_geometric.nn import knn_graph, knn
from torch_scatter import scatter
from e3nn.o3 import Irreps, spherical_harmonics
import torch.nn.functional as F

from funcmol.models.segnn import SEGNN
from funcmol.models.encoder import create_grid_coords


class SteerableEncoder(nn.Module):
    """
    使用Steerable E(3) GNN将原子信息编码到网格锚点
    
    架构流程:
    1. 原子类型 → irreps特征 (使用one-hot + 嵌入)
    2. 构建图: atom-atom + atom-anchor 边
    3. 计算边属性: spherical_harmonics(relative_pos)
    4. SEGNN消息传递 (多层)
    5. 提取anchor节点特征作为codes
    """
    
    def __init__(self, 
                 n_atom_types: int,
                 grid_size: int,
                 code_dim: int,
                 hidden_irreps: str = "128x0e + 64x1o + 32x2e",
                 num_layers: int = 4,
                 lmax_attr: int = 2,
                 norm: str = "instance",
                 k_neighbors: int = 24,
                 atom_k_neighbors: int = 8,
                 cutoff: float = 8.0,
                 anchor_spacing: float = 1.5,
                 **kwargs):
        """
        Args:
            n_atom_types: 原子类型数量
            grid_size: 网格大小 (每维)
            code_dim: 输出code的维度
            hidden_irreps: 隐藏层的irreps配置
            num_layers: SEGNN层数
            lmax_attr: 边属性球谐函数最高阶数
            norm: 归一化类型 ("instance", "batch", 或 None)
            k_neighbors: atom-anchor连接数
            atom_k_neighbors: atom-atom连接数
            cutoff: 距离截断
            anchor_spacing: 锚点间距
        """
        super().__init__()
        self.n_atom_types = n_atom_types
        self.grid_size = grid_size
        self.code_dim = code_dim
        self.hidden_irreps = Irreps(hidden_irreps)
        self.num_layers = num_layers
        self.lmax_attr = lmax_attr
        self.k_neighbors = k_neighbors
        self.atom_k_neighbors = atom_k_neighbors
        self.cutoff = cutoff
        self.anchor_spacing = anchor_spacing
        
        # 配置irreps
        # 输入：原子类型作为标量特征
        self.input_irreps = Irreps(f"{n_atom_types}x0e")
        # 输出：codes也是标量
        self.output_irreps = Irreps(f"{code_dim}x0e")
        # 边和节点属性
        self.edge_attr_irreps = Irreps.spherical_harmonics(lmax_attr)
        self.node_attr_irreps = Irreps.spherical_harmonics(lmax_attr)
        
        # 注册网格坐标
        grid_coords = create_grid_coords(1, self.grid_size,
                        device="cpu", anchor_spacing=self.anchor_spacing).squeeze(0)
        self.register_buffer('grid_coords', grid_coords)
        
        # 创建SEGNN
        self.segnn = SEGNN(
            input_irreps=self.input_irreps,
            hidden_irreps=self.hidden_irreps,
            output_irreps=self.output_irreps,
            edge_attr_irreps=self.edge_attr_irreps,
            node_attr_irreps=self.node_attr_irreps,
            num_layers=num_layers,
            norm=norm,
            task="node",  # 节点级别任务
        )
        
    def forward(self, data):
        """
        Args:
            data: torch_geometric.data.Batch object
                  - data.pos: [N_total_atoms, 3], atom coordinates
                  - data.x: [N_total_atoms], atom types
                  - data.batch: [N_total_atoms], batch index for each atom
                  
        Returns:
            grid_codes: [B, n_grid, code_dim] 网格锚点的latent codes
        """
        atom_coords = data.pos
        atoms_channel = data.x
        atom_batch_idx = data.batch
        
        device = atom_coords.device
        # 处理单个Data对象（不是Batch）
        B = data.num_graphs if hasattr(data, 'num_graphs') else 1
        N_total_atoms = data.num_nodes
        n_grid = self.grid_size ** 3
        
        # 1. 原子特征：one-hot编码
        atom_feat = F.one_hot(atoms_channel.long(), num_classes=self.n_atom_types).float()
        
        # 2. 构造网格坐标
        grid_coords_flat = self.grid_coords.to(device).repeat(B, 1)  # [B*n_grid, 3]
        
        # 3. 初始化网格特征为0
        grid_feat = torch.zeros(B * n_grid, self.n_atom_types, device=device)
        
        # 4. 拼接所有节点
        grid_batch_idx = torch.arange(B, device=device).repeat_interleave(n_grid)
        
        node_feats = torch.cat([atom_feat, grid_feat], dim=0)  # [(N + B*n_grid), n_atom_types]
        node_pos = torch.cat([atom_coords, grid_coords_flat], dim=0)  # [(N + B*n_grid), 3]
        node_batch = torch.cat([atom_batch_idx, grid_batch_idx], dim=0)  # [(N + B*n_grid)]
        
        # 5. 构建边
        # 5.1 原子内部连接
        atom_edge_index = knn_graph(
            x=atom_coords,
            k=self.atom_k_neighbors,
            batch=atom_batch_idx,
            loop=False
        )
        
        # 5.2 原子-网格连接
        grid_to_atom_edges = knn(
            x=atom_coords,
            y=grid_coords_flat,
            k=self.k_neighbors,
            batch_x=atom_batch_idx,
            batch_y=grid_batch_idx
        )
        
        # 修正索引
        grid_to_atom_edges[0] += N_total_atoms
        grid_to_atom_edges = torch.stack([grid_to_atom_edges[1], grid_to_atom_edges[0]], dim=0)
        
        # 合并所有边
        edge_index = torch.cat([atom_edge_index, grid_to_atom_edges], dim=1)
        
        # 6. 计算边属性和节点属性（球谐函数）
        rel_pos = node_pos[edge_index[0]] - node_pos[edge_index[1]]
        edge_dist = torch.norm(rel_pos, dim=-1, keepdim=True)
        
        # 应用cutoff
        if self.cutoff is not None:
            PI = torch.pi
            cutoff_mask = (edge_dist.squeeze(-1) <= self.cutoff) & (edge_dist.squeeze(-1) >= 0.0)
            # 只保留在cutoff内的边
            valid_edges = cutoff_mask.nonzero().squeeze(-1)
            edge_index = edge_index[:, valid_edges]
            rel_pos = rel_pos[valid_edges]
            
        # 计算球谐函数作为边属性
        edge_attr = spherical_harmonics(
            self.edge_attr_irreps,
            rel_pos,
            normalize=True,
            normalization='integral'
        )
        
        # 聚合边属性得到节点属性
        node_attr = scatter(edge_attr, edge_index[1], dim=0, 
                           dim_size=node_pos.size(0), reduce="mean")
        
        # 确保所有节点都有属性（处理孤立节点）
        if node_attr.size(0) < node_pos.size(0):
            pad_size = node_pos.size(0) - node_attr.size(0)
            padding = torch.zeros(pad_size, node_attr.size(1), device=device)
            node_attr = torch.cat([node_attr, padding], dim=0)
        
        # 确保第一个分量为1（trivial irrep）
        node_attr[:, 0] = 1.0
        
        # 7. 创建PyG Data对象
        from torch_geometric.data import Data
        graph = Data(
            x=node_feats,
            pos=node_pos,
            edge_index=edge_index,
            edge_attr=edge_attr,
            node_attr=node_attr,
            batch=node_batch
        )
        
        # 8. SEGNN前向传播
        output = self.segnn(graph)  # [(N + B*n_grid), code_dim]
        
        # 9. 提取网格节点的输出
        grid_output = output[N_total_atoms:].reshape(B, n_grid, self.code_dim)
        
        return grid_output
