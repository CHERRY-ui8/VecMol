import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, knn_graph, knn
from funcmol.models.encoder import create_grid_coords

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

    def forward(self, x, h, edge_index, is_fixed=None, use_knn_graph=True): #TODO: 添加is_fixed参数
        """
            x: [N, 3] 坐标
            h: [N, in_channels] 节点特征
            edge_index: [2, E]
            is_fixed: [N] 布尔张量，True表示该节点不更新
            use_knn_graph: bool，是否使用knn_graph构建的完整图
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
            # 根据是否使用knn_graph决定是否添加size参数
            if not use_knn_graph:
                delta_x = self.propagate(edge_index, x=x, message=coord_message, size=(x.size(0), x.size(0)))  # [N, 3]
            else:
                delta_x = self.propagate(edge_index, x=x, message=coord_message)  # [N, 3]
            # 如果节点被固定，则不更新坐标
            if is_fixed is not None:
                delta_x = delta_x * (~is_fixed).float().unsqueeze(-1)
            x_new = x + delta_x  # 残差连接
        else:
            coord_message = coord_coef[..., None] * direction[:, None, :]  # [E, out_x_dim, 3]
            x = x[:, None, :]  # [N, 1, 3]
            # 根据是否使用knn_graph决定是否添加size参数
            if not use_knn_graph:
                delta_x = self.propagate(edge_index, x=None, message=coord_message.view(len(row), -1), size=(x.size(0), x.size(0)))  # [N, out_x_dim * 3]
            else:
                delta_x = self.propagate(edge_index, x=None, message=coord_message.view(len(row), -1))  # [N, out_x_dim * 3]
            delta_x = delta_x.view(x.size(0), self.out_x_dim, 3)  # [N, out_x_dim, 3]
            # 如果节点被固定，则不更新坐标
            if is_fixed is not None:
                delta_x = delta_x * (~is_fixed).float().unsqueeze(-1).unsqueeze(-1)
            x_new = x + delta_x  # 残差连接, [N, out_x_dim, 3]
        
        # 节点特征更新
        # 根据是否使用knn_graph决定是否添加size参数
        if not use_knn_graph:
            m_aggr = self.propagate(edge_index, x=x, message=m_ij, size=(x.size(0), x.size(0)))  # [N, hidden_channels]
        else:
            m_aggr = self.propagate(edge_index, x=x, message=m_ij)  # [N, hidden_channels]
        h_delta = self.node_mlp(torch.cat([h, m_aggr], dim=-1))
        # 如果节点被固定，则不更新特征
        if is_fixed is not None:
            h_delta = h_delta * (~is_fixed).float().unsqueeze(-1)
        h_new = h + h_delta  # 残差连接
        return h_new, x_new

    def message(self, message):
        return message

class EGNNVectorField(nn.Module):
    def __init__(self, 
                 grid_size: int = 8,
                 hidden_dim: int = 512,
                 num_layers: int = 3,
                 k_neighbors: int = 8,
                 n_atom_types: int = 5,
                 code_dim: int = 512,
                 device=None,
                 use_knn_graph: bool = True):  # 添加配置选项，可以选择使用knn_graph或knn
        """
        Initialize the EGNN Vector Field model.

        Args:
            grid_size (int): Size of the grid for G_L (L in the paper)
            hidden_dim (int): Dimension of hidden layers
            num_layers (int): Number of EGNN layers
            k_neighbors (int): Number of nearest neighbors for KNN graph
            n_atom_types (int): Number of atom types
            code_dim (int): Dimension of the latent code and node features
            device (torch.device, optional): The device to run the model on.
            use_knn_graph (bool): Whether to use knn_graph (slower but more stable) or knn (faster)
        """
        super().__init__()
        self.grid_size = grid_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.k_neighbors = k_neighbors
        self.n_atom_types = n_atom_types
        self.code_dim = code_dim
        self.use_knn_graph = use_knn_graph

        # Create learnable grid points and features for G_L
        self.register_buffer('grid_points', create_grid_coords(device, 1, self.grid_size).squeeze(0))
        self.grid_features = nn.Parameter(torch.randn(grid_size**3, code_dim, requires_grad=True))

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
        # self.field_layer = nn.Sequential(
        #     nn.Linear(code_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, n_atom_types)  # 3D vector for each atom type
        # )
        self.field_layer = EGNNLayer(
            in_channels=code_dim,
            hidden_channels=hidden_dim,
            out_channels=code_dim,
            k_neighbors=k_neighbors,
            out_x_dim=n_atom_types
        )
        
    def forward(self, query_points, codes=None):
        """
        输入：
            query_points: [batch_size, n_points, 3]，空间中采样的查询点
            codes: [batch_size, grid_size**3, feature_dim]，每个分子的latent grid特征
        输出：
            vector_field: [batch_size, n_points, n_atom_types, 3]，每个查询点的矢量场
        """
        batch_size = query_points.size(0)
        n_points = query_points.size(1)
        device = query_points.device
        grid_points = self.grid_points.to(device)  # [grid_size**3, 3]
        n_grid = grid_points.size(0)

        # 1. 初始化节点特征
        # 查询点特征初始化为0
        query_features = torch.zeros(batch_size, n_points, self.code_dim, device=device)
        
        # 锚点特征为codes
        # assert codes is not None and codes.size(0) == batch_size, "Codes must be provided with matching batch size"
        grid_features = codes

        # 2. 初始化节点坐标
        # 锚点坐标: [1, n_grid, 3] -> [batch_size, n_grid, 3]
        grid_coords = grid_points.unsqueeze(0).expand(batch_size, -1, -1)

        # 3. 合并节点，为每个分子构建一个图
        # [B, n_points + n_grid, C]
        node_features = torch.cat([query_features, grid_features], dim=1) # TODO：在这里，要注意拼接之后的编号问题，因为拼接前的编号是0-n_points-1，拼接后的编号是0-n_points+n_grid-1，会有一个bias
        # [B, n_points + n_grid, 3]
        node_coords = torch.cat([query_points, grid_coords], dim=1)
        total_nodes = n_points + n_grid

        # 4. 展平成适合PyG的格式
        node_features = node_features.reshape(-1, self.code_dim)  # [B * total_nodes, code_dim]
        node_coords = node_coords.reshape(-1, 3)                  # [B * total_nodes, 3]
        batch_index = torch.arange(batch_size, device=device).repeat_interleave(total_nodes)

        # 5. 构建边
        if self.use_knn_graph:
            # 使用 knn_graph 构建完整的图（更稳定但较慢）
            edge_index = knn_graph(
                x=node_coords,
                k=self.k_neighbors,
                batch=batch_index,
                loop=False
            )
        else:
            # 使用 knn 构建部分图
            # 分别构造 batch index
            query_coords = query_points.reshape(-1, 3).float()       # [B * n_query, 3]
            grid_coords_all = grid_coords.reshape(-1, 3).float()     # [B * n_grid, 3]
            query_batch = torch.arange(batch_size, device=device).repeat_interleave(n_points)
            grid_batch = torch.arange(batch_size, device=device).repeat_interleave(n_grid)

            # query <-> query edges (双向)
            edge_query_query = knn(x=query_coords, y=query_coords, k=self.k_neighbors, batch_x=query_batch, batch_y=query_batch)
            # 添加反向边
            edge_query_query_rev = edge_query_query.flip(0)
            edge_query_query = torch.cat([edge_query_query, edge_query_query_rev], dim=1)

            # query -> grid edges (查询点从网格点获取信息)
            edge_query_grid = knn(x=grid_coords_all, y=query_coords, k=self.k_neighbors, batch_x=grid_batch, batch_y=query_batch)
            edge_query_grid = edge_query_grid.flip(0)  # 保证方向是 query -> grid

            # 合并边
            edge_index = torch.cat([edge_query_query, edge_query_grid], dim=1)
        
        # 6. 构造is_fixed参数，标记网格点为固定节点
        is_fixed = torch.zeros(batch_size * total_nodes, dtype=torch.bool, device=device)
        # 网格点（后半部分）标记为固定
        for b in range(batch_size):
            start_idx = b * total_nodes + n_points
            end_idx = (b + 1) * total_nodes
            is_fixed[start_idx:end_idx] = True

        # 7. 逐层EGNN消息传递
        h, x = node_features, node_coords
        for layer in self.layers:
            h, x = layer(x, h, edge_index, is_fixed, self.use_knn_graph) # h: [B * total_nodes, code_dim], x: [B * total_nodes, 3]

        # 8. 预测矢量场
        _, predicted_sources = self.field_layer(x, h, edge_index, is_fixed, self.use_knn_graph)  # [B * total_nodes, n_atom_types, 3]
        predicted_sources = predicted_sources.view(batch_size, total_nodes, self.n_atom_types, 3)
        predicted_sources = predicted_sources[:, :n_points, :, :]  # 只取查询点部分
        vector_field = predicted_sources - query_points[:,:,None,:]  # (B, n_points, n_atom_types, 3）
        
        return vector_field