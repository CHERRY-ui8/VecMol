import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, knn_graph

class EGNNLayer(MessagePassing):
    def __init__(self, in_channels, hidden_channels, out_channels, k_neighbors=8):
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

        # 用于message计算的MLP，输入[h_i, h_j, ||x_i-x_j||]，输出hidden_channels
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * in_channels + 1, hidden_channels),  # 2*512+1=1025 -> 512
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
        )
        # 用于坐标更新的MLP，输入hidden_channels，输出1（标量）
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_channels, 1),
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
        coord_coef = self.coord_mlp(m_ij)  # [E, 1]
        direction = rel / (dist + 1e-8)  # 单位向量 [E, 3]
        coord_message = coord_coef * direction  # [E, 3]
        # 聚合到每个节点
        delta_x = self.propagate(edge_index, x=x, message=coord_message)  # [N, 3]
        x_new = x + delta_x  # 残差连接
        # 节点特征更新
        m_aggr = self.propagate(edge_index, x=x, message=m_ij)  # [N, hidden_channels]
        
        # 打印调试信息
        # print(f"h shape: {h.shape}, m_aggr shape: {m_aggr.shape}")
        # print(f"h device: {h.device}, m_aggr device: {m_aggr.device}")
        
        # 确保h和m_aggr在同一个设备上
        if h.device != m_aggr.device:
            m_aggr = m_aggr.to(h.device)
        
        # 确保h和m_aggr的batch维度匹配
        if h.size(0) != m_aggr.size(0):
            print(f"Warning: h size {h.size(0)} != m_aggr size {m_aggr.size(0)}")
            # 重新计算m_aggr
            m_aggr = self.propagate(edge_index, x=x, message=m_ij, size=(h.size(0), h.size(0)))  # [N, hidden_channels]
        
        h_delta = self.node_mlp(torch.cat([h, m_aggr], dim=-1))
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
                 code_dim: int = 512):
        """
        Initialize the EGNN Vector Field model.

        Args:
            grid_size (int): Size of the grid for G_L (L in the paper)
            hidden_dim (int): Dimension of hidden layers
            num_layers (int): Number of EGNN layers
            k_neighbors (int): Number of nearest neighbors for KNN graph
            n_atom_types (int): Number of atom types
            code_dim (int): Dimension of the latent code and node features
        """
        super().__init__()
        self.grid_size = grid_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.k_neighbors = k_neighbors
        self.n_atom_types = n_atom_types
        self.code_dim = code_dim

        # Create learnable grid points and features for G_L
        self.register_buffer('grid_points', self._create_grid_points())
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
        self.base_field_layer = nn.Sequential(
            nn.Linear(code_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_atom_types * 3)  # 3D vector for each atom type
        )
        
        # 差值预测层
        self.delta_field_layer = nn.Sequential(
            nn.Linear(code_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_atom_types * 3)  # 3D vector for each atom type
        )

    def _create_grid_points(self):
        """Create grid points for G_L"""
        grid_points = []
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                for z in range(self.grid_size):
                    # Normalize coordinates to [-1, 1]
                    point = torch.tensor([
                        2.0 * x / (self.grid_size - 1) - 1.0,
                        2.0 * y / (self.grid_size - 1) - 1.0,
                        2.0 * z / (self.grid_size - 1) - 1.0
                    ])
                    grid_points.append(point)
        return torch.stack(grid_points)

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
        # 查询点特征初始化为0，使用与codes相同的维度
        query_features = torch.zeros(batch_size, n_points, self.code_dim, device=device)
        
        # 锚点特征为codes（latent grid），如果codes为None则用self.grid_features
        if codes is not None:
            # 确保codes的batch_size与query_points匹配
            if codes.size(0) == 1:
                grid_features = codes.expand(batch_size, -1, -1)  # [batch_size, n_grid, code_dim]
            else:
                assert codes.size(0) == batch_size
                grid_features = codes
        else:
            grid_features = self.grid_features.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, n_grid, code_dim]

        # 2. 初始化节点坐标
        # 查询点坐标: [batch_size, n_points, 3]
        # 锚点坐标: [1, n_grid, 3] -> [batch_size, n_grid, 3]
        grid_coords = grid_points.unsqueeze(0).expand(batch_size, -1, -1)

        # 3. 合并节点特征和坐标
        # 节点顺序：[查询点, 锚点]
        node_features = torch.cat([query_features, grid_features], dim=1)  # [batch_size, n_points + n_grid, code_dim]
        node_coords = torch.cat([query_points, grid_coords], dim=1)        # [batch_size, n_points + n_grid, 3]
        total_nodes = n_points + n_grid

        # 4. 展平成单batch图
        node_features = node_features.reshape(-1, self.code_dim)  # [batch_size * total_nodes, code_dim]
        node_coords = node_coords.reshape(-1, 3)  # [batch_size * total_nodes, 3]
        # batch索引
        batch_index = torch.arange(batch_size, device=device).repeat_interleave(total_nodes)

        # 5. 用knn_graph建边（只连查询点到锚点）
        edge_index = knn_graph(
            x=node_coords,
            k=self.k_neighbors,
            batch=batch_index,
            loop=False
        )

        # 6. 逐层EGNN消息传递（每层都要同步更新特征和坐标）
        h, x = node_features, node_coords
        for layer in self.layers:
            h, x = layer(x, h, edge_index)
            # 清理不需要的中间变量
            torch.cuda.empty_cache()

        # 7. 取出查询点部分的特征
        h = h.view(batch_size, total_nodes, -1)
        h_query = h[:, :n_points, :]
        x = x.view(batch_size, total_nodes, 3)
        x_query = x[:, :n_points, :]

        # 8. 预测基准场和差值场
        base_field = self.base_field_layer(h_query)
        delta_field = self.delta_field_layer(h_query)
        
        # 9. 合并基准场和差值场，并计算相对于查询点的位移
        vector_field = base_field + delta_field
        vector_field = vector_field.view(batch_size, n_points, self.n_atom_types, 3)
        
        # 10. 计算最终的向量场（相对于查询点的位移）
        vector_field = vector_field - query_points.unsqueeze(2) # ⚠️需要保证输出的是差量
        
        # 清理不需要的中间变量
        del h, x, node_features, node_coords, edge_index
        torch.cuda.empty_cache()
        
        return vector_field