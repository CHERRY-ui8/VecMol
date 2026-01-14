"""
梯度场计算模块：计算梯度场（GNF的核心功能）
"""
import torch
from typing import Dict


class GradientFieldComputer:
    """梯度场计算器"""
    
    def __init__(
        self,
        sigma_params: Dict[int, float],
        sigma: float,
        gradient_field_method: str = "softmax",
        temperature: float = 1.0,
        logsumexp_eps: float = 1e-8,
        inverse_square_strength: float = 1.0,
        gradient_clip_threshold: float = 0.3,
        sig_sf: float = 0.1,
        sig_mag: float = 0.45,
    ):
        """
        初始化梯度场计算器
        
        Args:
            sigma_params: 每个原子类型的sigma参数字典
            sigma: 默认sigma值
            gradient_field_method: 梯度场计算方法
            temperature: softmax温度参数
            logsumexp_eps: logsumexp方法的数值稳定性参数
            inverse_square_strength: 距离平方反比方法的强度参数
            gradient_clip_threshold: 梯度模长截断阈值
            sig_sf: softmax field的sigma参数
            sig_mag: magnitude的sigma参数
        """
        self.sigma_params = sigma_params
        self.sigma = sigma
        self.gradient_field_method = gradient_field_method
        self.temperature = temperature
        self.logsumexp_eps = logsumexp_eps
        self.inverse_square_strength = inverse_square_strength
        self.gradient_clip_threshold = gradient_clip_threshold
        self.sig_sf = sig_sf
        self.sig_mag = sig_mag
    
    def compute_gradient_field_matrix(
        self, 
        coords: torch.Tensor, 
        atom_types: torch.Tensor, 
        query_points: torch.Tensor, 
        n_atom_types: int
    ) -> torch.Tensor:
        """
        完全矩阵化计算梯度场，避免所有原子类型循环。
        
        Args:
            coords: 有效原子坐标 [n_valid_atoms, 3]
            atom_types: 有效原子类型 [n_valid_atoms]
            query_points: 查询点 [n_points, 3]
            n_atom_types: 原子类型数量
            
        Returns:
            vector_field: [n_points, n_atom_types, 3]
        """
        n_points, _ = query_points.shape
        device = coords.device
        
        # 计算所有原子到所有查询点的距离和方向向量
        coords_exp = coords.unsqueeze(1)  # [n_atoms, 1, 3]
        q_exp = query_points.unsqueeze(0)  # [1, n_points, 3]
        diff = coords_exp - q_exp  # [n_atoms, n_points, 3]
        dist_sq = torch.sum(diff ** 2, dim=-1, keepdim=True)  # [n_atoms, n_points, 1]
        
        # 创建原子类型掩码矩阵 [n_atoms, n_atom_types]
        atom_type_mask = (atom_types.unsqueeze(1) == torch.arange(n_atom_types, device=device).unsqueeze(0))  # [n_atoms, n_atom_types]
        
        # 创建sigma参数矩阵 [n_atom_types]
        sigma_values = torch.tensor([self.sigma_params.get(t, self.sigma) for t in range(n_atom_types)], device=device, dtype=torch.float32)  # [n_atom_types]
        
        # 初始化结果张量
        vector_field = torch.zeros(n_points, n_atom_types, 3, device=device)
        
        # 优化：为向量化准备原子类型索引
        # 创建原子类型到索引的映射 [n_atoms]，用于 scatter 操作
        atom_type_indices = atom_types  # [n_atoms]，值在 [0, n_atom_types-1]
        
        # 根据梯度场计算方法进行完全矩阵化计算
        if self.gradient_field_method == "gaussian":
            # 向量化版本：一次性处理所有原子类型
            # 扩展 sigma_values 以匹配每个原子 [n_atoms, 1, 1]
            sigma_per_atom = sigma_values[atom_type_indices].unsqueeze(1).unsqueeze(2)  # [n_atoms, 1, 1]
            
            # 为所有原子计算高斯权重 [n_atoms, n_points, 3]
            gaussian_weights = torch.exp(-dist_sq / (2 * sigma_per_atom ** 2)) / (sigma_per_atom ** 2)
            individual_gradients = diff * gaussian_weights  # [n_atoms, n_points, 3]
            
            # 使用 index_add 按原子类型聚合：对每个查询点和原子类型，累加对应原子的梯度
            # 重塑为 [n_atoms * n_points, 3] 以便使用 index_add
            n_atoms = coords.size(0)
            individual_gradients_flat = individual_gradients.view(n_atoms * n_points, 3)  # [n_atoms * n_points, 3]
            
            # 为每个 (atom, point) 对创建索引：point_idx * n_atom_types + atom_type
            point_indices = torch.arange(n_points, device=device).unsqueeze(0).expand(n_atoms, -1)  # [n_atoms, n_points]
            point_indices_flat = point_indices.flatten()  # [n_atoms * n_points]
            atom_type_indices_flat = atom_type_indices.unsqueeze(1).expand(-1, n_points).flatten()  # [n_atoms * n_points]
            
            # 计算在 vector_field 中的线性索引
            linear_indices = point_indices_flat * n_atom_types + atom_type_indices_flat  # [n_atoms * n_points]
            
            # 使用 index_add 聚合
            vector_field_flat = vector_field.view(n_points * n_atom_types, 3)  # [n_points * n_atom_types, 3]
            vector_field_flat.index_add_(0, linear_indices, individual_gradients_flat)
            vector_field = vector_field_flat.view(n_points, n_atom_types, 3)
            
        elif self.gradient_field_method == "softmax":
            # 优化版本：批量处理，减少 Python 循环开销
            # 计算距离 [n_atoms, n_points]
            distances = torch.sqrt(dist_sq.squeeze(-1))  # [n_atoms, n_points]
            
            # 为每个原子类型批量计算
            for t in range(n_atom_types):
                type_mask = atom_type_mask[:, t]  # [n_atoms]
                if type_mask.sum() > 0:
                    type_distances = distances[type_mask]  # [n_type_atoms, n_points]
                    type_diff = diff[type_mask]  # [n_type_atoms, n_points, 3]
                    
                    # 批量计算 softmax 权重 [n_type_atoms, n_points]
                    weights = torch.softmax(-type_distances / self.temperature, dim=0)  # [n_type_atoms, n_points]
                    weights = weights.unsqueeze(-1)  # [n_type_atoms, n_points, 1]
                    weighted_gradients = type_diff * weights  # [n_type_atoms, n_points, 3]
                    type_gradients = torch.sum(weighted_gradients, dim=0)  # [n_points, 3]
                    
                    # 批量应用梯度模长截断
                    if self.gradient_clip_threshold > 0:
                        gradient_magnitudes = torch.norm(type_gradients, dim=-1, keepdim=True)  # [n_points, 1]
                        clip_mask = gradient_magnitudes > self.gradient_clip_threshold
                        if clip_mask.any():
                            type_gradients = torch.where(
                                clip_mask,
                                type_gradients * self.gradient_clip_threshold / gradient_magnitudes,
                                type_gradients
                            )
                    vector_field[:, t, :] = type_gradients
                    
        elif self.gradient_field_method == "sfnorm":
            # 计算距离 [n_atoms, n_points]
            distances = torch.sqrt(dist_sq.squeeze(-1))  # [n_atoms, n_points]
            
            for t in range(n_atom_types):
                type_mask = atom_type_mask[:, t]  # [n_atoms]
                if type_mask.sum() > 0:
                    type_distances = distances[type_mask]  # [n_type_atoms, n_points]
                    type_diff = diff[type_mask]  # [n_type_atoms, n_points, 3]
                    
                    weights = torch.softmax(-type_distances / self.temperature, dim=0)  # [n_type_atoms, n_points]
                    weights = weights.unsqueeze(-1)  # [n_type_atoms, n_points, 1]
                    diff_norm = type_diff / (torch.norm(type_diff, dim=-1, keepdim=True) + 1e-8)
                    weighted_gradients = diff_norm * weights  # [n_type_atoms, n_points, 3]
                    vector_field[:, t, :] = torch.sum(weighted_gradients, dim=0)  # [n_points, 3]
                    
        elif self.gradient_field_method == "logsumexp":
            scale = 0.1  # 可配置的缩放参数
            for t in range(n_atom_types):
                type_mask = atom_type_mask[:, t]  # [n_atoms]
                if type_mask.sum() > 0:
                    sigma = sigma_values[t]
                    type_diff = diff[type_mask]  # [n_type_atoms, n_points, 3]
                    type_dist_sq = dist_sq[type_mask]  # [n_type_atoms, n_points, 1]
                    
                    individual_gradients = type_diff * torch.exp(-type_dist_sq / (2 * sigma ** 2)) / (sigma ** 2)
                    gradient_magnitudes = torch.norm(individual_gradients, dim=2, keepdim=True)  # [n_type_atoms, n_points, 1]
                    gradient_directions = individual_gradients / (gradient_magnitudes + self.logsumexp_eps)  # [n_type_atoms, n_points, 3]
                    log_sum_exp = torch.logsumexp(gradient_magnitudes, dim=0, keepdim=True)  # [1, n_points, 1]
                    vector_field[:, t, :] = scale * torch.sum(gradient_directions, dim=0) * log_sum_exp.squeeze(0)  # [n_points, 3]
                    
        elif self.gradient_field_method == "inverse_square":
            for t in range(n_atom_types):
                type_mask = atom_type_mask[:, t]  # [n_atoms]
                if type_mask.sum() > 0:
                    type_diff = diff[type_mask]  # [n_type_atoms, n_points, 3]
                    type_dist_sq = dist_sq[type_mask]  # [n_type_atoms, n_points, 1]
                    
                    distances = torch.sqrt(type_dist_sq.squeeze(-1) + 1e-8)  # [n_type_atoms, n_points]
                    inverse_square_weights = 1.0 / (distances ** 2 + 1e-8)  # [n_type_atoms, n_points]
                    weighted_weights = inverse_square_weights * self.inverse_square_strength  # [n_type_atoms, n_points]
                    weighted_weights = weighted_weights.unsqueeze(-1)  # [n_type_atoms, n_points, 1]
                    weighted_gradients = type_diff * weighted_weights  # [n_type_atoms, n_points, 3]
                    vector_field[:, t, :] = torch.sum(weighted_gradients, dim=0)  # [n_points, 3]
                    
        elif self.gradient_field_method == "tanh":
            for t in range(n_atom_types):
                type_mask = atom_type_mask[:, t]  # [n_atoms]
                if type_mask.sum() > 0:
                    type_diff = diff[type_mask]  # [n_type_atoms, n_points, 3]
                    type_dist_sq = dist_sq[type_mask]  # [n_type_atoms, n_points, 1]
                    
                    distances = torch.sqrt(type_dist_sq.squeeze(-1))  # [n_type_atoms, n_points]
                    w_softmax = torch.softmax(-distances / self.sig_sf, dim=0)  # [n_type_atoms, n_points]
                    w_mag = torch.tanh(distances / self.sig_mag)  # [n_type_atoms, n_points]
                    diff_normed = type_diff / (torch.norm(type_diff, dim=-1, keepdim=True) + 1e-8)  # [n_type_atoms, n_points, 3]
                    weighted_gradients = diff_normed * w_softmax.unsqueeze(-1) * w_mag.unsqueeze(-1)  # [n_type_atoms, n_points, 3]
                    vector_field[:, t, :] = torch.sum(weighted_gradients, dim=0)  # [n_points, 3]
                    
        elif self.gradient_field_method == "gaussian_mag":
            for t in range(n_atom_types):
                type_mask = atom_type_mask[:, t]  # [n_atoms]
                if type_mask.sum() > 0:
                    type_diff = diff[type_mask]  # [n_type_atoms, n_points, 3]
                    type_dist_sq = dist_sq[type_mask]  # [n_type_atoms, n_points, 1]
                    
                    distances = torch.sqrt(type_dist_sq.squeeze(-1))  # [n_type_atoms, n_points]
                    w_softmax = torch.softmax(-distances / self.sig_sf, dim=0)  # [n_type_atoms, n_points]
                    w_mag = torch.exp(-distances**2 / (2 * self.sig_mag**2)) * distances  # [n_type_atoms, n_points]
                    diff_normed = type_diff / (torch.norm(type_diff, dim=-1, keepdim=True) + 1e-8)  # [n_type_atoms, n_points, 3]
                    weighted_gradients = diff_normed * w_softmax.unsqueeze(-1) * w_mag.unsqueeze(-1)  # [n_type_atoms, n_points, 3]
                    vector_field[:, t, :] = torch.sum(weighted_gradients, dim=0)  # [n_points, 3]
                    
        elif self.gradient_field_method == "distance":
            for t in range(n_atom_types):
                type_mask = atom_type_mask[:, t]  # [n_atoms]
                if type_mask.sum() > 0:
                    type_diff = diff[type_mask]  # [n_type_atoms, n_points, 3]
                    type_dist_sq = dist_sq[type_mask]  # [n_type_atoms, n_points, 1]
                    
                    distances = torch.sqrt(type_dist_sq.squeeze(-1))  # [n_type_atoms, n_points]
                    w_softmax = torch.softmax(-distances / self.sig_sf, dim=0)  # [n_type_atoms, n_points]
                    w_mag = torch.clamp(distances, min=0, max=1)  # [n_type_atoms, n_points]
                    diff_normed = type_diff / (torch.norm(type_diff, dim=-1, keepdim=True) + 1e-8)  # [n_type_atoms, n_points, 3]
                    weighted_gradients = diff_normed * w_softmax.unsqueeze(-1) * w_mag.unsqueeze(-1)  # [n_type_atoms, n_points, 3]
                    vector_field[:, t, :] = torch.sum(weighted_gradients, dim=0)  # [n_points, 3]
        
        return vector_field

    def compute_field_variance(
        self, 
        points: torch.Tensor, 
        field_values: torch.Tensor, 
        k_neighbors: int, 
        per_type_independent: bool = True
    ) -> torch.Tensor:
        """
        计算每个点周围field的变化率（使用方差作为变化率）。
        
        Args:
            points: 点坐标 [n_points, 3]
            field_values: field值 [n_points, n_atom_types, 3] 或 [n_points, 3]
            k_neighbors: 用于计算方差的最近邻数量
            per_type_independent: 如果为True，每个原子类型独立计算方差（基于该类型的field值找最近邻）
                                 如果为False，使用空间距离找最近邻（旧的行为）
            
        Returns:
            variances: 每个点的field方差 
                - 如果field_values是3D: [n_points, n_atom_types] (为每个原子类型分别计算)
                - 如果field_values是2D: [n_points] (单一field的方差)
        """
        n_points = points.size(0)
        device = points.device
        
        # 如果field_values是3D的（包含原子类型维度），需要为每个原子类型分别计算
        if field_values.dim() == 3:
            # field_values: [n_points, n_atom_types, 3]
            n_atom_types = field_values.size(1)
            variances = torch.zeros(n_points, n_atom_types, device=device)
            
            if per_type_independent:
                # 优化：批量处理所有原子类型，减少 Python 循环开销
                # 为所有原子类型同时计算 field 模长
                field_magnitudes_all = torch.norm(field_values, dim=2)  # [n_points, n_atom_types]
                
                # 对每个原子类型分别计算（因为每个类型的最近邻可能不同）
                for t in range(n_atom_types):
                    field_t = field_values[:, t, :]  # [n_points, 3]
                    
                    # 基于该类型的field向量计算距离矩阵（用于找最近邻）
                    # 使用field向量的欧氏距离作为距离度量
                    field_dist_matrix = torch.cdist(field_t, field_t)  # [n_points, n_points]
                    
                    # 找到每个点的k个最近邻（基于field向量距离，排除自身）
                    _, k_nearest_indices = torch.topk(field_dist_matrix, k=k_neighbors + 1, dim=1, largest=False)  # [n_points, k+1]
                    k_nearest_indices = k_nearest_indices[:, 1:]  # 排除自身 [n_points, k]
                    
                    # 使用预计算的模长
                    field_magnitudes = field_magnitudes_all[:, t]  # [n_points]
                    
                    # 对每个点，计算其k个最近邻的field模长
                    neighbor_magnitudes = field_magnitudes[k_nearest_indices]  # [n_points, k]
                    
                    # 计算方差
                    mean_magnitudes = torch.mean(neighbor_magnitudes, dim=1)  # [n_points]
                    variances[:, t] = torch.mean((neighbor_magnitudes - mean_magnitudes.unsqueeze(1)) ** 2, dim=1)  # [n_points]
            else:
                # 旧的行为：使用空间距离找最近邻
                # 计算点之间的距离矩阵
                dist_matrix = torch.cdist(points, points)  # [n_points, n_points]
                
                # 找到每个点的k个最近邻（排除自身）
                _, k_nearest_indices = torch.topk(dist_matrix, k=k_neighbors + 1, dim=1, largest=False)  # [n_points, k+1]
                k_nearest_indices = k_nearest_indices[:, 1:]  # 排除自身 [n_points, k]
                
                # 优化：批量计算所有原子类型的 field 模长
                field_magnitudes_all = torch.norm(field_values, dim=2)  # [n_points, n_atom_types]
                
                for t in range(n_atom_types):
                    # 使用预计算的模长
                    field_magnitudes = field_magnitudes_all[:, t]  # [n_points]
                    
                    # 对每个点，计算其k个最近邻的field模长
                    neighbor_magnitudes = field_magnitudes[k_nearest_indices]  # [n_points, k]
                    
                    # 计算方差
                    mean_magnitudes = torch.mean(neighbor_magnitudes, dim=1)  # [n_points]
                    variances[:, t] = torch.mean((neighbor_magnitudes - mean_magnitudes.unsqueeze(1)) ** 2, dim=1)  # [n_points]
            
            # 返回每个原子类型的方差 [n_points, n_atom_types]，不再求平均
            return variances
        else:
            # field_values: [n_points, 3]
            # 对于2D情况，使用空间距离
            dist_matrix = torch.cdist(points, points)  # [n_points, n_points]
            _, k_nearest_indices = torch.topk(dist_matrix, k=k_neighbors + 1, dim=1, largest=False)  # [n_points, k+1]
            k_nearest_indices = k_nearest_indices[:, 1:]  # 排除自身 [n_points, k]
            
            field_magnitudes = torch.norm(field_values, dim=1)  # [n_points]
            neighbor_magnitudes = field_magnitudes[k_nearest_indices]  # [n_points, k]
            mean_magnitudes = torch.mean(neighbor_magnitudes, dim=1)  # [n_points]
            variances = torch.mean((neighbor_magnitudes - mean_magnitudes.unsqueeze(1)) ** 2, dim=1)  # [n_points]
            return variances

