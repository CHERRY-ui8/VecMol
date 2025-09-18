import torch
import torch.nn as nn
from torch_scatter import scatter
from torch_scatter import scatter
from torch_geometric.nn import MessagePassing, radius_graph
from funcmol.models.encoder import create_grid_coords


class EGNNDenoiserLayer(MessagePassing):
    def __init__(self, in_channels, hidden_channels, out_channels, cutoff=None, radius=None):
        super().__init__(aggr='mean')
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
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
        
        # 添加LayerNorm层
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x, h, edge_index):
        # LayerNorm + EGNN层
        h = self.norm(h)
        
        row, col = edge_index
        
        # 安全检查：确保索引在有效范围内
        max_idx = h.size(0)
        if not ((row < max_idx) & (col < max_idx)).all():
            raise ValueError(f"Edge indices out of bounds: max_idx={max_idx}, "
                           f"row_max={row.max().item()}, col_max={col.max().item()}")
        
        # 计算距离和方向 - 修复为标准EGNN实现
        rel = x[col] - x[row]  # [E, 3] 从row指向col
        dist = torch.norm(rel, dim=-1, keepdim=True)  # [E, 1]
        
        # 构造message输入 - 标准EGNN实现
        h_i = h[row]  # 源节点特征
        h_j = h[col]  # 目标节点特征
        edge_input = torch.cat([h_i, h_j, dist], dim=-1)  # [E, 2*in_channels+1]
        m_ij = self.edge_mlp(edge_input)  # [E, hidden_channels]
        
        if self.cutoff is not None:
            PI = torch.pi
            C = 0.5 * (torch.cos(dist.squeeze(-1) * PI / self.cutoff) + 1.0)
            C = C * (dist.squeeze(-1) <= self.cutoff) * (dist.squeeze(-1) >= 0.0)
            m_ij = m_ij * C.view(-1, 1)
        
        # 使用 torch_scatter 来聚合边消息到目标节点 (col)
        m_aggr = scatter(m_ij, edge_index[1], dim=0, dim_size=h.size(0), reduce='mean')
        h_delta = self.node_mlp(torch.cat([h, m_aggr], dim=-1))
        h = h + h_delta  # 残差连接
        
        return h

    def message(self, message_j):  
        return message_j


class GNNDenoiser(nn.Module):
    def __init__(self, code_dim=1024, hidden_dim=128, num_layers=4, 
                 cutoff=None, radius=None, dropout=0.1, grid_size=8, 
                 anchor_spacing=2.0, use_radius_graph=True, device=None):
        super().__init__()
        self.code_dim = code_dim
        self.hidden_dim = hidden_dim
        self.num_blocks = num_layers
        self.cutoff = cutoff
        self.radius = radius
        self.grid_size = grid_size
        self.anchor_spacing = anchor_spacing
        self.use_radius_graph = use_radius_graph
        
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        # 输入投影层
        self.input_projection = nn.Linear(code_dim, hidden_dim)
        
        # GNN Layers (now integrated with residual blocks)
        self.blocks = nn.ModuleList([
            EGNNDenoiserLayer(
                in_channels=hidden_dim,
                hidden_channels=hidden_dim,
                out_channels=hidden_dim,
                radius=radius
            ) for _ in range(num_layers)
        ])
        
        # 输出投影层
        self.output_projection = nn.Linear(hidden_dim, code_dim)
        
        # 预创建网格坐标（在__init__中）
        self.register_buffer('grid_coords', 
                           create_grid_coords(
                               batch_size=1, 
                               grid_size=self.grid_size, 
                               device='cpu', 
                               anchor_spacing=self.anchor_spacing
                           ).squeeze(0))  # [n_grid, 3]
        
    def forward(self, y):
        # 处理输入维度 - 现在只支持3D输入 [batch_size, n_grid, code_dim]
        if y.dim() == 4:  # (batch_size, 1, n_grid, code_dim)
            y = y.squeeze(1)  # (batch_size, n_grid, code_dim)
        elif y.dim() == 3:  # (batch_size, n_grid, code_dim)
            pass
        else:
            raise ValueError(f"Expected 3D or 4D input, got {y.dim()}D: {y.shape}")
        
        batch_size = y.size(0)
        n_grid = y.size(1)
        
        # 构建图 - 直接在这里实现
        # 使用预创建的网格坐标
        n_grid_actual = self.grid_size ** 3
        grid_coords = self.grid_coords.to(self.device)  # [n_grid, 3]
        
        # 为每个batch复制网格坐标
        grid_coords = grid_coords.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, n_grid, 3]
        grid_coords = grid_coords.reshape(-1, 3)  # [batch_size * n_grid, 3]
        
        # 创建batch索引
        grid_batch = torch.arange(batch_size, device=self.device).repeat_interleave(n_grid_actual)
        
        # 使用 radius_graph 构建图
        edge_index = radius_graph(
            x=grid_coords,
            r=self.radius,
            batch=grid_batch
        )
        
        # 构建图 - 直接在这里实现
        # 使用预创建的网格坐标
        n_grid_actual = self.grid_size ** 3
        grid_coords = self.grid_coords.to(self.device)  # [n_grid, 3]
        
        # 为每个batch复制网格坐标
        grid_coords = grid_coords.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, n_grid, 3]
        grid_coords = grid_coords.reshape(-1, 3)  # [batch_size * n_grid, 3]
        
        # 创建batch索引
        grid_batch = torch.arange(batch_size, device=self.device).repeat_interleave(n_grid_actual)
        
        # 使用 radius_graph 构建图
        edge_index = radius_graph(
            x=grid_coords,
            r=self.radius,
            batch=grid_batch
        )
        
        # 输入投影
        h = self.input_projection(y)  # (batch_size, n_grid, hidden_dim)
        h = h.reshape(-1, self.hidden_dim)  # (batch_size * n_grid, hidden_dim)
        
        # 通过GNN块
        for block in self.blocks:
            h = block(grid_coords, h, edge_index)
        
        # 重塑回原始维度
        h = h.view(batch_size, n_grid, self.hidden_dim)
        
        # 输出投影
        output = self.output_projection(h)
        
        return output
