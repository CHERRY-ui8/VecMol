import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, knn_graph, knn
from funcmol.models.encoder import create_grid_coords
import math

# 添加torch.compile兼容性处理
try:
    from torch._dynamo import disable
except ImportError:
    def disable(fn):
        return fn

class EGNNLayer(MessagePassing):
    def __init__(self, in_channels, hidden_channels, out_channels, k_neighbors=8, out_x_dim=1):
        """
            EGNN等变层实现：
            - 只用h_i, h_j, ||x_i-x_j||作为边特征输入
            - 用message的标量加权方向向量更新坐标
            - 不用EGNN论文里的a_ij（在这个模型里无化学键信息）
        """
        super().__init__(aggr='mean')
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.k_neighbors = k_neighbors
        self.out_x_dim = out_x_dim

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
        h_j = h[col]
        edge_input = torch.cat([h_i, h_j, dist], dim=-1)  # [E, 2*in_channels+1]
        m_ij = self.edge_mlp(edge_input)  # [E, hidden_channels]
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
                 k_neighbors: int = 8,
                 n_atom_types: int = 5,
                 code_dim: int = 128,
                 device=None):
        """
        Initialize the EGNN Vector Field model.
        Each query point predicts a 3D vector for every atom type.

        Args:
            grid_size (int): Size of the grid for G_L (L in the paper)
            hidden_dim (int): Dimension of hidden layers
            num_layers (int): Number of EGNN layers
            k_neighbors (int): Number of nearest neighbors for KNN graph
            n_atom_types (int): Number of atom types
            code_dim (int): Dimension of the latent code and node features
            device (torch.device, optional): The device to run the model on.
        """
        super().__init__()
        self.grid_size = grid_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.k_neighbors = k_neighbors
        self.n_atom_types = n_atom_types
        self.code_dim = code_dim

        # Create learnable grid points and features for G_L
        self.register_buffer('grid_points', create_grid_coords(batch_size=1, grid_size=self.grid_size, device=device).squeeze(0))
        self.grid_features = nn.Parameter(torch.randn(self.grid_size**3, self.code_dim, requires_grad=True) / math.sqrt(self.code_dim)) # TODO: 是否需要初始化？

        # type embedding
        self.type_embed = nn.Embedding(self.n_atom_types, self.code_dim)

        # Create EGNN layers
        self.layers = nn.ModuleList([
            EGNNLayer(
                in_channels=code_dim,
                hidden_channels=hidden_dim,
                out_channels=code_dim,
                k_neighbors=k_neighbors
            ) for _ in range(num_layers)
        ])
        
        # 基准场预测层
        self.field_layer = EGNNLayer(
            in_channels=code_dim,
            hidden_channels=hidden_dim,
            out_channels=code_dim,
            k_neighbors=k_neighbors,
            out_x_dim=n_atom_types
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

        # 1. 初始化节点特征
        # 查询点特征初始化为0
        # [B, n_points, code_dim]
        query_features = torch.zeros(batch_size, n_points, self.code_dim, device=device)
        
        # 锚点特征为codes
        # [B, n_grid, code_dim]
        grid_features = codes

        # 2. 初始化节点坐标
        # 锚点坐标: [1, n_grid, 3] -> [batch_size, n_grid, 3]
        grid_coords = grid_points.unsqueeze(0).expand(batch_size, -1, -1)

        # 3. 合并节点，为每个分子构建一个图 TODO:
        # [B, n_points + n_grid, code_dim]
        combined_features = torch.cat([query_features, grid_features], dim=1)
        # [B, n_points + n_grid, 3]
        combined_coords = torch.cat([query_points, grid_coords], dim=1)
        total_nodes = n_points + n_grid

        # 4. 展平成适合PyG的格式
        h = combined_features.reshape(-1, self.code_dim)  # [B * total_nodes, code_dim]
        x = combined_coords.reshape(-1, 3)                  # [B * total_nodes, 3]

        # 5. 构建边 - 修改逻辑：query_points之间不连边，grid_points之间可以连边
            # 使用 knn 构建部分图，只保留 grid -> query 的连边
            # 分别构造 batch index
        query_coords_flat = query_points.reshape(-1, 3).float()       # [B * n_query, 3]
        grid_coords_flat = grid_coords.reshape(-1, 3).float()     # [B * n_grid, 3]
        query_batch = torch.arange(batch_size, device=device).repeat_interleave(n_points)
        grid_batch = torch.arange(batch_size, device=device).repeat_interleave(n_grid)

        # 只构建 grid -> query edges (查询点从网格点获取信息)
        edge_grid_query = knn(
            x=query_coords_flat,
            y=grid_coords_flat,
            k=self.k_neighbors, 
            batch_x=query_batch,
            batch_y=grid_batch
        ) # [2, E=k_neighbors*n_points]

        edge_grid_query[0] += len(query_coords_flat) # 添加bias，匹配knn的输出
                    
        # 7. 逐层EGNN消息传递
        for layer in self.layers:
            h, x = layer(x, h, edge_grid_query) # h: [B * total_nodes, code_dim], x: [B * total_nodes, 3]
        
        # TODO: x-->node_coords
        # Use original coordinates for final field prediction
        node_coords = torch.cat([query_points, grid_coords], dim=1).reshape(-1, 3)


        # 8. 预测矢量场 
        # _, predicted_sources = self.field_layer(x, h, edge_grid_query)  # [B * total_nodes, n_atom_types, 3]
        # predicted_sources = predicted_sources.view(batch_size, total_nodes, self.n_atom_types, 3)
        # predicted_sources = predicted_sources[:, :n_points, :, :]  # 只取查询点部分
        # vector_field = predicted_sources - query_points[:,:,None,:]  # (B, n_points, n_atom_types, 3）
        # Predict vector field separately for each atom type
        # 如果没有 early type-awareness 的模型，容易出现折中解（所有类型都预测个平均场）。
        # 如果一个 query point 对不同 atom type 的向量场是完全不同的（方向、长度、聚集），
        # 而模型前几层是共享的，那么它容易出现折中解（所有类型都预测个平均场）。
        # 模型在训练时会更难对多样化结构建模。
        _, predicted_sources = self.field_layer(node_coords, h, edge_grid_query)  # [B*(N+G), T, 3]
        predicted_sources = predicted_sources.view(batch_size, total_nodes, self.n_atom_types, 3)
        predicted_sources = predicted_sources[:, :n_points, :, :]  # keep only query points

        # Compute residual vectors relative to query points
        vector_field = predicted_sources - query_points[:, :, None, :]  # [B, N, T, 3]

        return vector_field