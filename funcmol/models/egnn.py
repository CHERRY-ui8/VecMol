import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, knn_graph, knn, radius
from funcmol.models.encoder import create_grid_coords
import math

# 添加torch.compile兼容性处理
try:
    from torch._dynamo import disable
except ImportError:
    def disable(fn):
        return fn

class EGNNLayer(MessagePassing):
    def __init__(self, in_channels, hidden_channels, out_channels, radius=2.0, out_x_dim=1, cutoff=None):
        """
            EGNN等变层实现：
            - 只用h_i, h_j, ||x_i-x_j||作为边特征输入
            - 用message的标量加权方向向量更新坐标
            - 不用EGNN论文里的a_ij（在这个模型里无化学键信息）
            - 添加cutoff参数进行距离过滤
            - 使用radius而不是k_neighbors来构建图
        """
        super().__init__(aggr='mean')
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.radius = radius
        self.out_x_dim = out_x_dim
        self.cutoff = cutoff

        # 用于message计算的MLP，输入[h_i, h_j, ||x_i-x_j||]，输出hidden_channels
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * in_channels + 1, hidden_channels),  # 2*512+1=1025 -> 512
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
        )
        # 用于坐标更新的MLP，输入hidden_channels，输出1（标量）
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_channels, out_x_dim),
        )
        # 节点特征更新
        self.node_mlp = nn.Sequential(
            nn.Linear(in_channels + hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, out_channels)
        )

    def forward(self, x, h, edge_index):
        """
            x: [N, 3] 坐标
            h: [N, in_channels] 节点特征
            edge_index: [2, E]
            返回: h_new, x_new
        """
        row, col = edge_index
        # 计算距离和方向
        rel = x[row] - x[col]  # [E, 3]
        dist = torch.norm(rel, dim=-1, keepdim=True)  # [E, 1]
        
        # 构造message输入
        h_i = h[row]
        h_j = h[col] # h_j 应该恒等于0，但是不是。发现h_j的从第501个开始不是0
        edge_input = torch.cat([h_i, h_j, dist], dim=-1)  # [E, 2*in_channels+1]
        m_ij = self.edge_mlp(edge_input)  # [E, hidden_channels]
        
        # 应用cutoff过滤
        if self.cutoff is not None:
            PI = torch.pi
            C = 0.5 * (torch.cos(dist.squeeze(-1) * PI / self.cutoff) + 1.0)
            C = C * (dist.squeeze(-1) <= self.cutoff) * (dist.squeeze(-1) >= 0.0)
            m_ij = m_ij * C.view(-1, 1)
        
        # 坐标更新：用message生成标量，乘以方向
        coord_coef = self.coord_mlp(m_ij)  # [E, out_x_dim]
        direction = rel / (dist + 1e-8)  # 单位向量 [E, 3]
        # 聚合到每个节点
        if self.out_x_dim == 1:
            coord_message = coord_coef * direction  # [E, 3]
            # 添加size参数确保输出维度正确，size应该是元组
            delta_x = self.propagate(edge_index, x=x, message=coord_message, size=(x.size(0), x.size(0)))  # [N, 3]
            x_new = x + delta_x  # 残差连接
        else:
            coord_message = coord_coef[..., None] * direction[:, None, :]  # [E, out_x_dim, 3]
            x = x[:, None, :]  # [N, 1, 3]
            # 添加size参数确保输出维度正确，size应该是元组
            delta_x = self.propagate(edge_index, x=None, message=coord_message.view(len(row), -1), size=(x.size(0), x.size(0)))  # [N, out_x_dim * 3]
            delta_x = delta_x.view(x.size(0), self.out_x_dim, 3)  # [N, out_x_dim, 3]
            x_new = x + delta_x  # 残差连接, [N, out_x_dim, 3]
        
        # 节点特征更新
        # 添加size参数确保输出维度正确，size应该是元组
        m_aggr = self.propagate(edge_index, x=x, message=m_ij, size=(x.size(0), x.size(0)))  # [N, hidden_channels]
        h_delta = self.node_mlp(torch.cat([h, m_aggr], dim=-1))
        h_new = h + h_delta  # 残差连接
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
                 device=None):
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
        # 如果cutoff为None，则使用radius作为cutoff
        self.cutoff = cutoff if cutoff is not None else radius

        # Create learnable grid points and features for G_L
        # 指定batch_size=1：只存储一份网格坐标，而不是为每个可能的 batch_size 都存储一份
        self.register_buffer('grid_points', create_grid_coords(batch_size=1, grid_size=self.grid_size, device=device, anchor_spacing=anchor_spacing).squeeze(0))
        self.grid_features = nn.Parameter(torch.randn(self.grid_size**3, self.code_dim, requires_grad=True) / math.sqrt(self.code_dim)) # TODO: 是否需要初始化？

        # type embedding
        self.type_embed = nn.Embedding(self.n_atom_types, self.code_dim)

        # Create EGNN layers
        self.layers = nn.ModuleList([
            EGNNLayer(
                in_channels=code_dim,
                hidden_channels=hidden_dim,
                out_channels=code_dim,
                radius=radius,
                cutoff=cutoff
            ) for _ in range(num_layers)
        ])
        
        # 基准场预测层
        self.field_layer = EGNNLayer(
            in_channels=code_dim,
            hidden_channels=hidden_dim,
            out_channels=code_dim,
            radius=radius,
            out_x_dim=n_atom_types,
            cutoff=cutoff
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
        grid_coords = grid_points.unsqueeze(0).expand(batch_size, -1, -1).reshape(-1, 3)  # [B * grid_size**3, 3]
        n_points_total = query_points.size(0)
        
        # 1. 初始化节点特征
        query_features = torch.zeros(n_points_total, self.code_dim, device=device)
        grid_features = codes.reshape(-1, self.code_dim)  # [B * grid_size**3, code_dim]
        
        # 3. 合并节点
        combined_features = torch.cat([query_features, grid_features], dim=0)  # [B*N + B*grid_size**3, code_dim]
        combined_coords = torch.cat([query_points, grid_coords], dim=0)  # [B*N + B*grid_size**3, 3]

        # 5. 构建边 - 使用radius而不是knn
        query_batch = torch.arange(batch_size, device=device).repeat_interleave(n_points)
        grid_batch = torch.arange(batch_size, device=device).repeat_interleave(n_grid)

        # 确保所有输入都在正确的设备上
        grid_coords = grid_coords.to(device)
        query_points = query_points.to(device)

        # 使用radius构建 query -> grid edges，确保每个query point都能从指定半径内的grid获取信息
        edge_grid_query = radius(
            x=grid_coords,
            y=query_points,
            r=self.radius,
            batch_x=grid_batch,
            batch_y=query_batch
        ) # [2, E] where E is variable depending on radius

        edge_grid_query[1] += n_points_total # bias
        
        edge_grid_query = torch.stack([edge_grid_query[1], edge_grid_query[0]], dim=0) # 交换边的方向
        
        # 7. 逐层EGNN消息传递
        h = combined_features
        x = combined_coords
        for layer in self.layers:
            h, x = layer(x, h, edge_grid_query) # [total_nodes, code_dim], [total_nodes, 3]

        # 8. 预测矢量场
        
        _, predicted_sources = self.field_layer(x, h, edge_grid_query)  # [total_nodes, n_atom_types, 3]
        predicted_sources = predicted_sources[:n_points_total]  # keep only query points
        predicted_sources = predicted_sources.view(batch_size, n_points, self.n_atom_types, 3)

        # Compute residual vectors relative to query points
        query_points_reshaped = query_points.view(batch_size, n_points, 3)
        vector_field = predicted_sources - query_points_reshaped[:, :, None, :]  # [B, N, n_atom_types, 3]

        return vector_field