"""Steerable E(3) Equivariant Decoder for Neural Field
使用SEGNN将anchor codes解码为向量场
"""

import torch
import torch.nn as nn
from torch_geometric.nn import knn
from torch_scatter import scatter
from e3nn.o3 import Irreps, spherical_harmonics

from funcmol.models.segnn import SEGNN
from funcmol.models.encoder import create_grid_coords


class SteerableDecoder(nn.Module):
    """
    使用Steerable E(3) GNN将anchor codes解码为向量场
    
    架构流程:
    1. 输入: query点坐标 + anchor的codes
    2. 构建query→anchor的边
    3. 计算边属性 (spherical harmonics)
    4. SEGNN消息传递
    5. 输出层: 预测向量场 (irreps包含向量部分)
    """
    
    def __init__(self,
                 grid_size: int,
                 code_dim: int,
                 n_atom_types: int,
                 hidden_irreps: str = "128x0e + 64x1o + 32x2e",
                 num_layers: int = 5,
                 lmax_attr: int = 2,
                 norm: str = "instance",
                 k_neighbors: int = 32,
                 cutoff: float = 8.0,
                 anchor_spacing: float = 1.5,
                 **kwargs):
        """
        Args:
            grid_size: 网格大小
            code_dim: code维度
            n_atom_types: 原子类型数量
            hidden_irreps: 隐藏层irreps配置
            num_layers: SEGNN层数
            lmax_attr: 边属性球谐函数最高阶数
            norm: 归一化类型
            k_neighbors: query-anchor连接数
            cutoff: 距离截断
            anchor_spacing: 锚点间距
        """
        super().__init__()
        self.grid_size = grid_size
        self.code_dim = code_dim
        self.n_atom_types = n_atom_types
        self.hidden_irreps = Irreps(hidden_irreps)
        self.num_layers = num_layers
        self.lmax_attr = lmax_attr
        self.k_neighbors = k_neighbors
        self.cutoff = cutoff
        
        # 配置irreps
        # 输入：codes作为标量特征
        self.input_irreps = Irreps(f"{code_dim}x0e")
        # 隐藏层包含标量和向量
        self.hidden_irreps = Irreps(hidden_irreps)
        # 输出：每个原子类型一个向量 (1o表示奇宇称的向量)
        self.output_irreps = Irreps(f"{n_atom_types}x1o")
        # 边和节点属性
        self.edge_attr_irreps = Irreps.spherical_harmonics(lmax_attr)
        self.node_attr_irreps = Irreps.spherical_harmonics(lmax_attr)
        
        # 注册网格坐标
        grid_coords = create_grid_coords(1, self.grid_size,
                        device="cpu", anchor_spacing=anchor_spacing).squeeze(0)
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
    
    def forward(self, query_points, codes):
        """
        Args:
            query_points: [B, N, 3] 查询点坐标
            codes: [B, grid_size^3, code_dim] 锚点的latent codes
            
        Returns:
            vector_field: [B, N, n_atom_types, 3] 每个查询点对每种原子类型的向量场
        """
        batch_size = query_points.size(0)
        n_points = query_points.size(1)
        device = query_points.device
        
        grid_points = self.grid_coords.to(device)
        n_grid = grid_points.size(0)
        
        # 1. 展平输入
        query_points_flat = query_points.reshape(-1, 3)  # [B*N, 3]
        grid_coords_flat = grid_points.repeat(batch_size, 1)  # [B*n_grid, 3]
        codes_flat = codes.reshape(-1, self.code_dim)  # [B*n_grid, code_dim]
        
        # 2. 初始化query特征为0
        query_feat = torch.zeros(batch_size * n_points, self.code_dim, device=device)
        
        # 3. 拼接所有节点
        query_batch = torch.arange(batch_size, device=device).repeat_interleave(n_points)
        grid_batch = torch.arange(batch_size, device=device).repeat_interleave(n_grid)
        
        node_feats = torch.cat([query_feat, codes_flat], dim=0)  # [(B*N + B*n_grid), code_dim]
        node_pos = torch.cat([query_points_flat, grid_coords_flat], dim=0)  # [(B*N + B*n_grid), 3]
        node_batch = torch.cat([query_batch, grid_batch], dim=0)  # [(B*N + B*n_grid)]
        
        # 4. 构建query→anchor的边
        # 使用knn：每个query连接最近的k个anchor
        edge_anchor_to_query = knn(
            x=grid_coords_flat,  # source: anchors
            y=query_points_flat,  # target: queries
            k=self.k_neighbors,
            batch_x=grid_batch,
            batch_y=query_batch
        )  # [2, E]
        
        # 调整索引
        edge_anchor_to_query[0] += batch_size * n_points  # anchor索引偏移
        # 交换方向: [query, anchor]
        edge_index = torch.stack([edge_anchor_to_query[1], edge_anchor_to_query[0]], dim=0)
        
        # 5. 计算边属性和节点属性
        rel_pos = node_pos[edge_index[0]] - node_pos[edge_index[1]]
        edge_dist = torch.norm(rel_pos, dim=-1, keepdim=True)
        
        # 应用cutoff
        if self.cutoff is not None:
            PI = torch.pi
            cutoff_mask = (edge_dist.squeeze(-1) <= self.cutoff) & (edge_dist.squeeze(-1) >= 0.0)
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
        
        # 确保所有节点都有属性
        if node_attr.size(0) < node_pos.size(0):
            pad_size = node_pos.size(0) - node_attr.size(0)
            padding = torch.zeros(pad_size, node_attr.size(1), device=device)
            node_attr = torch.cat([node_attr, padding], dim=0)
        
        # 确保第一个分量为1
        node_attr[:, 0] = 1.0
        
        # 6. 创建PyG Data对象
        from torch_geometric.data import Data
        graph = Data(
            x=node_feats,
            pos=node_pos,
            edge_index=edge_index,
            edge_attr=edge_attr,
            node_attr=node_attr,
            batch=node_batch
        )
        
        # 7. SEGNN前向传播
        output = self.segnn(graph)  # [(B*N + B*n_grid), n_atom_types*3]
        
        # 8. 提取query节点的输出
        query_output = output[:batch_size * n_points]  # [B*N, n_atom_types*3]
        
        # 9. 重塑为向量场格式
        # output_irreps是n_atom_types个1o (向量)，每个向量3维
        vector_field = query_output.reshape(batch_size, n_points, self.n_atom_types, 3)
        
        return vector_field
