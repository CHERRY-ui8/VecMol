import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass
from pathlib import Path
from sklearn.cluster import DBSCAN
import torch.nn.functional as F
from funcmol.utils.constants import PADDING_INDEX, BOND_LENGTHS_PM, DEFAULT_BOND_LENGTH_THRESHOLD, ELEMENTS_HASH_INV
import gc


@dataclass
class ClusteringIterationRecord:
    """记录单次聚类迭代的信息"""
    iteration: int  # 迭代轮数
    eps: float  # 当前eps阈值
    min_samples: int  # 当前min_samples阈值
    new_atoms_coords: np.ndarray  # [N, 3] 本轮新聚类的原子坐标
    new_atoms_types: np.ndarray  # [N] 本轮新聚类的原子类型
    n_clusters_found: int  # 本轮找到的簇数量
    n_atoms_clustered: int  # 本轮聚类的原子数量
    n_noise_points: int  # 本轮噪声点数量
    bond_validation_passed: bool  # 是否通过键长检查（第一轮为True）


@dataclass
class ClusteringHistory:
    """记录整个聚类过程的历史"""
    atom_type: int  # 当前处理的原子类型
    iterations: List[ClusteringIterationRecord]  # 所有迭代的记录
    total_atoms: int  # 最终聚类的原子总数


class GNFConverter(nn.Module):
    """
    Converter between molecular structures and Gradient Neural Fields (GNF).
    
    The GNF is defined as r = f(X,z) where:
    - X is the set of atomic coordinates
    - z is a query point in 3D space
    - r is the gradient at point z
    """
    
    def __init__(self,
                sigma: float,
                n_query_points: int,
                n_iter: int,
                step_size: float,
                eps: float,  # DBSCAN的邻域半径参数
                min_samples: int,  # DBSCAN的最小样本数参数
                sigma_ratios: Dict[str, float],
                gradient_field_method: str = "softmax",  # 梯度场计算方法: "gaussian", "softmax", "logsumexp", "inverse_square", "sigmoid", "gaussian_mag", "distance"
                temperature: float = 1.0,  # softmax温度参数，控制分布尖锐程度
                logsumexp_eps: float = 1e-8,  # logsumexp方法的数值稳定性参数
                inverse_square_strength: float = 1.0,  # 距离平方反比方法的强度参数
                gradient_clip_threshold: float = 0.3,  # 梯度模长截断阈值
                sig_sf: float = 0.1,  # softmax field的sigma参数
                sig_mag: float = 0.45,  # magnitude的sigma参数
                gradient_sampling_candidate_multiplier: int = 3,  # 梯度采样候选点倍数
                field_variance_k_neighbors: int = 10,  # 计算field方差时使用的最近邻数量
                field_variance_weight: float = 1.0,  # field方差在采样概率中的权重（作为softmax温度参数）
                n_atom_types: int = 5,  # 原子类型数量，默认为5以保持向后兼容
                enable_early_stopping: bool = True,  # 是否启用早停机制
                convergence_threshold: float = 1e-6,  # 收敛阈值（梯度模长变化）
                min_iterations: int = 50,  # 最少迭代次数（早停前必须达到的最小迭代数）
                n_query_points_per_type: Optional[Dict[str, int]] = None,  # 每个原子类型的query_points数，如果为None则使用统一的n_query_points
                enable_autoregressive_clustering: bool = False,  # 是否启用自回归聚类
                initial_min_samples: Optional[int] = None,  # 初始min_samples（默认使用self.min_samples）
                initial_eps: Optional[float] = None,  # 初始eps（默认使用self.eps）
                eps_decay_factor: float = 0.8,  # 每轮eps衰减因子
                min_samples_decay_factor: float = 0.7,  # 每轮min_samples衰减因子
                min_eps: float = 0.1,  # eps下限
                min_min_samples: int = 2,  # min_samples下限
                max_clustering_iterations: int = 10,  # 最大迭代轮数
                bond_length_tolerance: float = 0.4,  # 键长合理性检查的容差（单位：Å），在标准键长基础上增加的容差
                enable_clustering_history: bool = False):  # 是否记录聚类历史
        super().__init__()
        self.sigma = sigma
        self.n_query_points = n_query_points
        self.n_iter = n_iter
        self.step_size = step_size
        self.eps = eps
        self.min_samples = min_samples
        self.sigma_ratios = sigma_ratios
        self.gradient_field_method = gradient_field_method  # 保存梯度场计算方法
        self.temperature = temperature  # 保存temperature参数
        self.logsumexp_eps = logsumexp_eps  # 保存logsumexp_eps参数
        self.inverse_square_strength = inverse_square_strength  # 保存inverse_square_strength参数
        self.gradient_clip_threshold = gradient_clip_threshold  # 保存梯度模长截断阈值
        self.sig_sf = sig_sf  # 保存softmax field的sigma参数
        self.sig_mag = sig_mag  # 保存magnitude的sigma参数
        self.gradient_sampling_candidate_multiplier = gradient_sampling_candidate_multiplier  # 保存梯度采样候选点倍数
        self.field_variance_k_neighbors = field_variance_k_neighbors  # 计算field方差时使用的最近邻数量
        self.field_variance_weight = field_variance_weight  # field方差在采样概率中的权重（作为softmax温度参数）
        self.n_atom_types = n_atom_types  # 保存原子类型数量
        self.enable_early_stopping = enable_early_stopping  # 是否启用早停机制
        self.convergence_threshold = convergence_threshold  # 收敛阈值
        self.min_iterations = min_iterations  # 最少迭代次数
        
        # 自回归聚类相关参数
        self.enable_autoregressive_clustering = enable_autoregressive_clustering
        self.initial_min_samples = initial_min_samples
        self.initial_eps = initial_eps
        self.eps_decay_factor = eps_decay_factor
        self.min_samples_decay_factor = min_samples_decay_factor
        self.min_eps = min_eps
        self.min_min_samples = min_min_samples
        self.max_clustering_iterations = max_clustering_iterations
        self.bond_length_tolerance = bond_length_tolerance
        self.enable_clustering_history = enable_clustering_history
        
        # 为不同类型的原子设置不同的 sigma 参数
        # We model hydrogen explicitly and consider 5 chemical elements for QM9 (C, H, O, N, F), 
        # 6 for CREMP (C, H, O, N, F, S) and 8 for GEOM-drugs (C, H, O, N, F, S, Cl and Br)
        # 原子类型索引映射：0=C, 1=H, 2=O, 3=N, 4=F, 5=S, 6=Cl, 7=Br
        atom_type_mapping = {0: 'C', 1: 'H', 2: 'O', 3: 'N', 4: 'F', 5: 'S', 6: 'Cl', 7: 'Br'}
        self.sigma_params = {}
        for atom_idx in range(n_atom_types):
            atom_symbol = atom_type_mapping.get(atom_idx, f'Type{atom_idx}')
            ratio = self.sigma_ratios.get(atom_symbol, 1.0)  # 默认比例为1.0
            self.sigma_params[atom_idx] = sigma * ratio
        
        # 为不同类型的原子设置不同的 query_points 数
        # 如果提供了 n_query_points_per_type，则使用它；否则所有原子类型使用统一的 n_query_points
        self.n_query_points_per_type = {}
        if n_query_points_per_type is not None:
            for atom_idx in range(n_atom_types):
                atom_symbol = atom_type_mapping.get(atom_idx, f'Type{atom_idx}')
                # 如果配置中指定了该原子类型的query_points数，则使用它；否则使用统一的n_query_points
                self.n_query_points_per_type[atom_idx] = n_query_points_per_type.get(atom_symbol, n_query_points)
        else:
            # 如果未提供 n_query_points_per_type，所有原子类型使用统一的 n_query_points
            for atom_idx in range(n_atom_types):
                self.n_query_points_per_type[atom_idx] = n_query_points
    
    def forward(self, coords: torch.Tensor, atom_types: torch.Tensor, 
                query_points: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the GNF converter.
        """
        return self.mol2gnf(coords, atom_types, query_points)
    
    def mol2gnf(self, coords: torch.Tensor, atom_types: torch.Tensor, 
                query_points: torch.Tensor) -> torch.Tensor:
        """
        将分子坐标和原子类型转换为GNF（梯度神经场）在查询点的向量场。
        梯度场定义为指向原子位置的向量场，使得在梯度上升时点会向原子位置移动。
        
        Args:
            coords: 原子坐标，shape为 [batch, n_atoms, 3] 或 [n_atoms, 3]
            atom_types: 原子类型，shape为 [batch, n_atoms] 或 [n_atoms]
            query_points: 查询点，shape为 [batch, n_points, 3] 或 [n_points, 3]
        
        Returns:
            vector_field: shape为 [batch, n_points, n_atom_types, 3] 的张量，
                其中 n_atom_types 通常为5（C、H、O、N、F），每个原子类型对应一个通道。
                始终保留batch维度。
        """
        # 兼容batch维度
        if coords.dim() == 2:
            coords = coords.unsqueeze(0)  # [1, N, 3]
        if atom_types.dim() == 1:
            atom_types = atom_types.unsqueeze(0)  # [1, N]
        if query_points.dim() == 2:
            query_points = query_points.unsqueeze(0)  # [1, M, 3]
            
        n_atom_types = self.n_atom_types  # 使用初始化时传入的n_atom_types
        batch_size, n_points, _ = query_points.shape
        device = query_points.device  # 使用输入张量的设备，而不是converter的设备
        vector_field = torch.zeros(batch_size, n_points, n_atom_types, 3, device=device)
        
        for b in range(batch_size):
            # 创建一维掩码
            mask = (atom_types[b] != PADDING_INDEX)  # [n_atoms]
            valid_coords = coords[b][mask]  # [n_valid_atoms, 3]
            valid_types = atom_types[b][mask].long()  # [n_valid_atoms]
            
            if valid_coords.size(0) == 0:
                continue
                
            # 矩阵化计算：一次性处理所有原子类型
            vector_field[b] = self._compute_gradient_field_matrix(
                valid_coords, valid_types, query_points[b], n_atom_types
            )
        return vector_field

    def _compute_gradient_field_matrix(self, coords: torch.Tensor, atom_types: torch.Tensor, 
                                     query_points: torch.Tensor, n_atom_types: int) -> torch.Tensor:
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
        sigma_values = torch.tensor([self.sigma_params.get(t, self.sigma) for t in range(n_atom_types)], device=device)  # [n_atom_types]
        
        # 初始化结果张量
        vector_field = torch.zeros(n_points, n_atom_types, 3, device=device)
        
        # 根据梯度场计算方法进行完全矩阵化计算
        if self.gradient_field_method == "gaussian":
            # 为每个原子类型分别计算高斯权重（保持与原始实现一致）
            for t in range(n_atom_types):
                type_mask = atom_type_mask[:, t]  # [n_atoms]
                if type_mask.sum() > 0:
                    sigma = sigma_values[t]
                    type_diff = diff[type_mask]  # [n_type_atoms, n_points, 3]
                    type_dist_sq = dist_sq[type_mask]  # [n_type_atoms, n_points, 1]
                    
                    individual_gradients = type_diff * torch.exp(-type_dist_sq / (2 * sigma ** 2)) / (sigma ** 2)
                    vector_field[:, t, :] = torch.sum(individual_gradients, dim=0)  # [n_points, 3]
            
        elif self.gradient_field_method == "softmax":
            # 计算距离 [n_atoms, n_points]
            distances = torch.sqrt(dist_sq.squeeze(-1))  # [n_atoms, n_points]
            
            # 为每个原子类型分别计算softmax权重
            for t in range(n_atom_types):
                type_mask = atom_type_mask[:, t]  # [n_atoms]
                if type_mask.sum() > 0:
                    type_distances = distances[type_mask]  # [n_type_atoms, n_points]
                    type_diff = diff[type_mask]  # [n_type_atoms, n_points, 3]
                    
                    # 计算softmax权重
                    weights = torch.softmax(-type_distances / self.temperature, dim=0)  # [n_type_atoms, n_points]
                    weights = weights.unsqueeze(-1)  # [n_type_atoms, n_points, 1]
                    weighted_gradients = type_diff * weights  # [n_type_atoms, n_points, 3]
                    type_gradients = torch.sum(weighted_gradients, dim=0)  # [n_points, 3]
                    
                    # 应用梯度模长截断
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

    def _compute_field_variance(self, points: torch.Tensor, field_values: torch.Tensor, 
                               k_neighbors: int, per_type_independent: bool = True) -> torch.Tensor:
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
                # 每个原子类型独立处理：基于该类型的field向量找最近邻
                for t in range(n_atom_types):
                    field_t = field_values[:, t, :]  # [n_points, 3]
                    
                    # 基于该类型的field向量计算距离矩阵（用于找最近邻）
                    # 使用field向量的欧氏距离作为距离度量
                    field_dist_matrix = torch.cdist(field_t, field_t)  # [n_points, n_points]
                    
                    # 找到每个点的k个最近邻（基于field向量距离，排除自身）
                    _, k_nearest_indices = torch.topk(field_dist_matrix, k=k_neighbors + 1, dim=1, largest=False)  # [n_points, k+1]
                    k_nearest_indices = k_nearest_indices[:, 1:]  # 排除自身 [n_points, k]
                    
                    # 计算field的模长
                    field_magnitudes = torch.norm(field_t, dim=1)  # [n_points]
                    
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
                
                for t in range(n_atom_types):
                    field_t = field_values[:, t, :]  # [n_points, 3]
                    # 计算field的模长
                    field_magnitudes = torch.norm(field_t, dim=1)  # [n_points]
                    
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

    def _process_atom_types_matrix(self, current_codes: torch.Tensor, n_atom_types: int, 
                                 device: torch.device, 
                                 decoder: nn.Module,
                                 iteration_callback: Optional[callable] = None,
                                 element_existence: Optional[torch.Tensor] = None,
                                 gradient_ascent_callback: Optional[callable] = None) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[ClusteringHistory]]:
        """
        完全矩阵化处理所有原子类型，避免所有循环。
        使用图结构思想：构建同种原子内部连边的图，一次性处理所有原子类型。
        
        Args:
            current_codes: 当前batch的编码 [1, code_dim]
            n_atom_types: 原子类型数量
            device: 设备
            decoder: 解码器模型
            iteration_callback: 可选的回调函数
            element_existence: 可选的元素存在性向量 [n_atom_types]，如果提供，只处理存在的元素类型
            
        Returns:
            (coords_list, types_list, histories): 坐标列表、类型列表和聚类历史列表
        """
        # 如果提供了element_existence，只处理存在的元素类型
        if element_existence is not None:
            # element_existence: [n_atom_types] 或 [1, n_atom_types]
            if element_existence.dim() == 2:
                element_existence = element_existence.squeeze(0)  # [n_atom_types]
            
            # 将概率转换为二进制（阈值0.5）
            element_mask = (element_existence > 0.5).float()
            # 获取存在的元素类型索引
            existing_types = torch.nonzero(element_mask, as_tuple=False).squeeze(-1).tolist()
            
            if len(existing_types) == 0:
                # 如果没有元素存在，返回空列表
                return [], [], []
        else:
            # 如果没有提供element_existence，处理所有类型
            existing_types = list(range(n_atom_types))
        
        # 获取每个原子类型的 query_points 数（只考虑存在的类型）
        query_points_per_type = [self.n_query_points_per_type.get(t, self.n_query_points) for t in existing_types]
        max_query_points = max(query_points_per_type) if query_points_per_type else self.n_query_points
        
        # 1. 初始化采样点 - 为所有原子类型一次性采样
        init_min, init_max = -7.0, 7.0
        n_candidates = max_query_points * self.gradient_sampling_candidate_multiplier
        candidate_points = torch.rand(n_candidates, 3, device=device) * (init_max - init_min) + init_min
        
        # 计算候选点的梯度场强度
        candidate_batch = candidate_points.unsqueeze(0)  # [1, n_candidates, 3]
        
        try:
            # 使用torch.no_grad()包装候选点采样
            with torch.no_grad():
                candidate_field = decoder(candidate_batch, current_codes)
            # 移除强制内存清理，让PyTorch自动管理
        except Exception as e:
            raise e
        
        # 2. 完全矩阵化采样 - 一次性处理所有原子类型
        # 计算每个候选点的field方差（为每个原子类型分别计算），基于方差采样
        # per_type_independent=True: 每个原子类型独立计算方差（基于该类型的field值找最近邻）
        field_variances_per_type = self._compute_field_variance(
            candidate_points, candidate_field[0], 
            self.field_variance_k_neighbors, 
            per_type_independent=True
        )  # [n_candidates, n_atom_types] 或 [n_candidates]
        
        # 为每个原子类型采样点 - 矩阵化操作
        all_sampled_points = []
        all_atom_types = []
        
        # 初始化结果列表
        coords_list = []
        types_list = []
        all_clustering_histories = []  # 收集所有原子类型的聚类历史
        
        # 初始化全局参考点（用于跨原子类型的键长检查）
        all_reference_points = []
        all_reference_types = []
        
        # 记录每个原子类型的起始索引，用于后续分离结果
        type_start_indices = []
        current_start_idx = 0
        
        # 只处理存在的元素类型
        for type_idx, t in enumerate(existing_types):
            n_query_points_t = query_points_per_type[t]  # 当前原子类型的 query_points 数
            
            # 获取当前原子类型的field方差
            if field_variances_per_type.dim() == 2:
                # [n_candidates, n_atom_types] - 为每个原子类型分别计算了方差
                field_variances = field_variances_per_type[:, t]  # [n_candidates]
            else:
                # [n_candidates] - 单一field的方差（向后兼容）
                field_variances = field_variances_per_type  # [n_candidates]
            
            # ===== 仅根据field变化率（方差）进行采样，完全忽略梯度场强度（模长） =====
            # 归一化方差（避免数值不稳定）
            variance_min = field_variances.min()
            variance_max = field_variances.max()
            
            if variance_max > variance_min:
                # 方差越大，采样概率越高（field变化率大的地方采样密度高）
                # 归一化到[0, 1]区间
                normalized_variances = (field_variances - variance_min) / (variance_max - variance_min + 1e-8)
            else:
                # 如果所有方差相同或为0，归一化后所有值都是0
                # 经过softmax后，所有值相等，得到均匀分布
                normalized_variances = torch.zeros_like(field_variances)
                        
            # field_variance_weight作为温度参数，控制采样分布的尖锐程度
            # 值越小，分布越尖锐（更倾向于选择方差大的点）
            # 值越大，分布越平缓（更接近均匀分布）
            probabilities = torch.softmax(normalized_variances / (self.field_variance_weight + 1e-8), dim=0)
            
            # 从候选点中采样 n_query_points_t 个点
            # 如果候选点数量少于需要的点数，使用 replacement=True
            replacement = n_query_points_t > n_candidates
            sampled_indices = torch.multinomial(probabilities, n_query_points_t, replacement=replacement)
            z = candidate_points[sampled_indices]  # [n_query_points_t, 3]
            
            all_sampled_points.append(z)
            all_atom_types.append(torch.full((n_query_points_t,), t, dtype=torch.long, device=device))
            
            # 记录起始索引
            type_start_indices.append(current_start_idx)
            current_start_idx += n_query_points_t
        
        # 合并所有原子类型的采样点进行批量梯度上升
        if all_sampled_points:
            # 将所有采样点合并 [total_points, 3]
            combined_points = torch.cat(all_sampled_points, dim=0)
            combined_types = torch.cat(all_atom_types, dim=0)  # [total_points]
            
            # 合并两个回调
            combined_callback = None
            if iteration_callback is not None or gradient_ascent_callback is not None:
                def combined(iter_idx, current_points, atom_types):
                    if iteration_callback is not None:
                        iteration_callback(iter_idx, current_points, atom_types)
                    if gradient_ascent_callback is not None:
                        gradient_ascent_callback(iter_idx, current_points, atom_types)
                combined_callback = combined
            
            # 批量梯度上升
            final_points = self._batch_gradient_ascent(
                combined_points, combined_types, current_codes, device, decoder,
                iteration_callback=combined_callback
            )
            
            # 按原子类型分离结果并进行聚类（只处理存在的类型）
            for type_idx, t in enumerate(existing_types):
                start_idx = type_start_indices[type_idx]
                end_idx = start_idx + query_points_per_type[type_idx]
                type_points = final_points[start_idx:end_idx]  # [n_query_points_t, 3]
                
                # 聚类/合并
                z_np = type_points.detach().cpu().numpy()
                merged_points, history = self._merge_points(
                    z_np, 
                    atom_type=t,
                    reference_points=np.array(all_reference_points) if len(all_reference_points) > 0 else None,
                    reference_types=np.array(all_reference_types) if len(all_reference_types) > 0 else None,
                    record_history=self.enable_clustering_history
                )
                
                if len(merged_points) > 0:
                    merged_tensor = torch.from_numpy(merged_points).to(device)
                    coords_list.append(merged_tensor)
                    types_list.append(torch.full((len(merged_points),), t, dtype=torch.long, device=device))
                    
                    # 更新全局参考点（用于后续原子类型的键长检查）
                    all_reference_points.extend(merged_points.tolist())
                    all_reference_types.extend([t] * len(merged_points))
                
                if history is not None:
                    all_clustering_histories.append(history)
        
        return coords_list, types_list, all_clustering_histories

    def _batch_gradient_ascent(self, points: torch.Tensor, atom_types: torch.Tensor,
                              current_codes: torch.Tensor, device: torch.device,
                              decoder: nn.Module,
                              iteration_callback: Optional[callable] = None) -> torch.Tensor:
        """
        批量梯度上升，对所有原子类型的点同时进行梯度上升。
        支持自适应停止：当梯度变化很小时提前停止。
        
        Args:
            points: 采样点 [n_total_points, 3]
            atom_types: 原子类型 [n_total_points]
            current_codes: 当前编码 [1, code_dim]
            device: 设备
            iteration_callback: 可选的回调函数，在每次迭代时调用，参数为 (iteration_idx, current_points, atom_types)
            
        Returns:
            final_points: 最终点位置 [n_total_points, 3]
        """
        z = points.clone()
        prev_grad_norm = None
        
        for iter_idx in range(self.n_iter):  # tqdm
            z_batch = z.unsqueeze(0)  # [1, n_total_points, 3]
            
            try:                
                # 使用torch.no_grad()包装，这是最关键的优化
                with torch.no_grad():
                    current_field = decoder(z_batch, current_codes)  # [1, n_total_points, n_atom_types, 3]
                
                # 为每个点选择对应原子类型的梯度
                # 使用高级索引选择梯度
                point_indices = torch.arange(z.size(0), device=device)  # [n_total_points]
                type_indices = atom_types  # [n_total_points]
                grad = current_field[0, point_indices, type_indices, :]  # [n_total_points, 3]
                
                # 检查梯度是否包含NaN/Inf
                if torch.isnan(grad).any() or torch.isinf(grad).any():
                    break
                
                # 使用原子类型特定的sigma调整步长
                sigma_ratios = torch.tensor([self.sigma_params.get(t.item(), self.sigma) / self.sigma 
                                           for t in atom_types], device=device)  # [n_total_points]
                adjusted_step_sizes = self.step_size * sigma_ratios.unsqueeze(-1)  # [n_total_points, 1]
                
                # 计算当前梯度的模长
                current_grad_norm = torch.norm(grad, dim=-1).mean().item()
                
                # 检查收敛条件（仅在启用早停时）
                if self.enable_early_stopping:
                    if iter_idx >= self.min_iterations and prev_grad_norm is not None:
                        grad_change = abs(current_grad_norm - prev_grad_norm)
                        if grad_change < self.convergence_threshold:
                            # 在停止前调用一次回调
                            if iteration_callback is not None:
                                iteration_callback(iter_idx, z.clone(), atom_types)
                            break
                
                prev_grad_norm = current_grad_norm
                
                # 更新采样点位置
                z = z + adjusted_step_sizes * grad
                
                # 调用迭代回调（如果提供）
                if iteration_callback is not None:
                    iteration_callback(iter_idx, z.clone(), atom_types)
                
            except (RuntimeError, ValueError, IndexError):
                # 发生错误时也清理内存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()
                break
        
        # 移除强制内存清理，让PyTorch自动管理
        
        return z

    def gnf2mol(self, decoder: nn.Module, codes: torch.Tensor,
                _atom_types: Optional[torch.Tensor] = None,
                save_interval: Optional[int] = None,
                visualization_callback: Optional[callable] = None,
                predictor: Optional[nn.Module] = None,
                save_clustering_history: bool = False,  # 是否保存聚类历史
                clustering_history_dir: Optional[str] = None,  # 保存目录
                save_gradient_ascent_sdf: bool = False,  # 是否保存梯度上升SDF
                gradient_ascent_sdf_dir: Optional[str] = None,  # 保存目录
                gradient_ascent_sdf_interval: int = 100) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:  # 保存间隔
        """
        直接用梯度场重建分子坐标。
        Args:
            decoder: 解码器模型，用于动态计算向量场
            codes: [batch, grid_size**3, code_dim]  # 编码器的输出
            _atom_types: 可选的原子类型（当前未使用，保留用于接口兼容性）
            save_interval: 可选的保存间隔，用于可视化。如果提供，会在每 save_interval 步调用 visualization_callback
            visualization_callback: 可选的可视化回调函数，参数为 (iteration_idx, all_points_dict, batch_idx)
                                   其中 all_points_dict 是一个字典，键为原子类型索引，值为该类型的所有点 [n_points, 3]
            predictor: 可选的元素存在性预测器，如果提供，将用于过滤不存在的元素类型
        Returns:
            (final_coords, final_types)
        """
        device = codes.device
        batch_size = codes.size(0)
        n_atom_types = self.n_atom_types

        all_coords = []
        all_types = []
        # 对每个batch分别处理
        for b in range(batch_size):
            # 检查索引边界
            if b >= codes.size(0):
                break
            
            # 检查当前batch的codes
            current_codes = codes[b:b+1]
            
            if current_codes.numel() == 0:
                continue
            
            if torch.isnan(current_codes).any() or torch.isinf(current_codes).any():
                continue
            
            # 如果提供了predictor，预测元素存在性
            element_existence = None
            if predictor is not None:
                with torch.no_grad():
                    element_existence = predictor(current_codes)  # [1, n_atom_types]
            
            # 如果提供了可视化回调，创建一个迭代回调函数
            iteration_callback = None
            if save_interval is not None and visualization_callback is not None:
                def create_iteration_callback(batch_idx, n_atom_types, save_interval_val, n_iter_val):
                    def callback(iter_idx, current_points, atom_types):
                        # 只在指定间隔时调用可视化回调
                        if iter_idx % save_interval_val == 0 or iter_idx == n_iter_val - 1:
                            # 按原子类型分离点
                            all_points_dict = {}
                            for t in range(n_atom_types):
                                type_mask = (atom_types == t)
                                if type_mask.any():
                                    all_points_dict[t] = current_points[type_mask].clone()
                                else:
                                    all_points_dict[t] = torch.empty((0, 3), device=current_points.device)
                            visualization_callback(iter_idx, all_points_dict, batch_idx)
                    return callback
                
                iteration_callback = create_iteration_callback(b, n_atom_types, save_interval, self.n_iter)
            
            # 创建梯度上升SDF保存回调
            gradient_ascent_callback = None
            if save_gradient_ascent_sdf and gradient_ascent_sdf_dir:
                from pathlib import Path
                sdf_dir = Path(gradient_ascent_sdf_dir)
                sdf_dir.mkdir(parents=True, exist_ok=True)
                
                def create_gradient_ascent_sdf_callback(batch_idx, n_atom_types, interval):
                    def callback(iter_idx, current_points, atom_types):
                        # 只在指定间隔时保存
                        if iter_idx % interval == 0 or iter_idx == self.n_iter - 1:
                            # 按原子类型分组
                            for t in range(n_atom_types):
                                type_mask = (atom_types == t)
                                if not type_mask.any():
                                    continue
                                
                                type_points = current_points[type_mask].cpu().numpy()
                                
                                # 直接保存所有query_points，不进行聚类
                                if len(type_points) > 0:
                                    try:
                                        from funcmol.sample_fm import xyz_to_sdf
                                        elements = [ELEMENTS_HASH_INV.get(i, f"Type{i}") for i in range(n_atom_types)]
                                        atom_symbol = elements[t] if t < len(elements) else f"Type{t}"
                                        
                                        # 创建SDF字符串，保存所有query_points
                                        query_types = np.full(len(type_points), t)
                                        sdf_str = xyz_to_sdf(type_points, query_types, elements)
                                        
                                        # 构建标题，包含query_points数量信息
                                        title = f"Gradient Ascent Iter {iter_idx}, Type {atom_symbol}, {len(type_points)} query_points"
                                        
                                        # 保存SDF文件
                                        sdf_file = sdf_dir / f"batch_{batch_idx}_type_{atom_symbol}_iter_{iter_idx:04d}.sdf"
                                        with open(sdf_file, 'w') as f:
                                            f.write(title + "\n\n")
                                            f.write(sdf_str)
                                    except Exception as e:
                                        print(f"Warning: Failed to save gradient ascent SDF: {e}")
                    
                    return callback
                
                gradient_ascent_callback = create_gradient_ascent_sdf_callback(b, n_atom_types, gradient_ascent_sdf_interval)
            
            # 矩阵化处理所有原子类型（传入element_existence以过滤不存在的元素）
            coords_list, types_list, histories = self._process_atom_types_matrix(
                current_codes, n_atom_types, device=device, decoder=decoder,
                iteration_callback=iteration_callback,
                element_existence=element_existence,
                gradient_ascent_callback=gradient_ascent_callback
            )
            
            # 保存聚类历史
            if save_clustering_history and clustering_history_dir and histories:
                self._save_clustering_history(
                    histories, 
                    clustering_history_dir,
                    batch_idx=b
                )
            
            # 合并所有类型
            if coords_list:
                all_coords.append(torch.cat(coords_list, dim=0))
                all_types.append(torch.cat(types_list, dim=0))
            else:
                all_coords.append(torch.empty(0, 3, device=device))
                all_types.append(torch.empty(0, dtype=torch.long, device=device))
        
        # pad到batch最大长度
        max_atoms = max([c.size(0) for c in all_coords]) if all_coords else 0
        final_coords = torch.stack([F.pad(c, (0,0,0,max_atoms-c.size(0))) if c.size(0)<max_atoms else c for c in all_coords], dim=0)
        final_types = torch.stack([F.pad(t, (0,max_atoms-t.size(0)), value=-1) if t.size(0)<max_atoms else t for t in all_types], dim=0)
        return final_coords, final_types # [batch, n_atoms, 3]



    def _get_bond_length_threshold(self, atom1_type: int, atom2_type: int) -> float:
        """
        根据两个原子类型返回合理的键长阈值（单位：Å）
        
        Args:
            atom1_type: 第一个原子类型索引
            atom2_type: 第二个原子类型索引
            
        Returns:
            键长阈值（单位：Å）
        """
        atom1_symbol = ELEMENTS_HASH_INV.get(atom1_type, None)
        atom2_symbol = ELEMENTS_HASH_INV.get(atom2_type, None)
        
        if atom1_symbol is None or atom2_symbol is None:
            return DEFAULT_BOND_LENGTH_THRESHOLD
        
        # 尝试获取键长数据（双向查找）
        bond_length_pm = None
        if atom1_symbol in BOND_LENGTHS_PM and atom2_symbol in BOND_LENGTHS_PM[atom1_symbol]:
            bond_length_pm = BOND_LENGTHS_PM[atom1_symbol][atom2_symbol]
        elif atom2_symbol in BOND_LENGTHS_PM and atom1_symbol in BOND_LENGTHS_PM[atom2_symbol]:
            bond_length_pm = BOND_LENGTHS_PM[atom2_symbol][atom1_symbol]
        
        if bond_length_pm is not None:
            # 转换为Å并加上容差
            return (bond_length_pm / 100.0) + self.bond_length_tolerance
        else:
            return DEFAULT_BOND_LENGTH_THRESHOLD
    
    def _check_bond_length_validity(
        self, 
        new_point: np.ndarray, 
        new_atom_type: int,
        reference_points: np.ndarray, 
        reference_types: np.ndarray
    ) -> bool:
        """
        检查新点与所有参考点的键长是否合理
        
        Args:
            new_point: 新点坐标 [3]
            new_atom_type: 新点原子类型
            reference_points: 参考点坐标 [M, 3]
            reference_types: 参考点原子类型 [M]
            
        Returns:
            如果存在至少一个参考点使得键长在合理范围内，返回True；否则返回False
        """
        if len(reference_points) == 0:
            return True  # 没有参考点，第一轮聚类，直接通过
        
        # 计算新点与所有参考点的距离
        distances = np.sqrt(((reference_points - new_point[None, :]) ** 2).sum(axis=1))
        
        # 检查是否至少有一个参考点使得键长合理
        for i, (ref_point, ref_type) in enumerate(zip(reference_points, reference_types)):
            distance = distances[i]
            threshold = self._get_bond_length_threshold(new_atom_type, ref_type)
            if distance < threshold:
                return True  # 找到至少一个合理的键长
        
        return False  # 没有找到合理的键长
    
    def _merge_points(
        self, 
        points: np.ndarray, 
        atom_type: Optional[int] = None,
        reference_points: Optional[np.ndarray] = None,
        reference_types: Optional[np.ndarray] = None,
        record_history: bool = False
    ) -> Tuple[np.ndarray, Optional[ClusteringHistory]]:
        """
        Use DBSCAN to merge close points and determine atom centers.
        Supports autoregressive clustering with bond length validation.
        
        Args:
            points: 点坐标 [N, 3]
            atom_type: 原子类型索引（可选）
            reference_points: 参考点坐标 [M, 3]（可选，用于键长检查）
            reference_types: 参考点原子类型 [M]（可选）
            record_history: 是否记录聚类历史
            
        Returns:
            merged_points: 合并后的点坐标
            history: 聚类历史记录（如果record_history=True）
        """
        if len(points.shape) == 3:
            points = points.reshape(-1, 3)
        elif len(points.shape) != 2 or points.shape[1] != 3:
            raise ValueError(f"Expected points shape (n, 3), got {points.shape}")

        if len(points) == 0:
            return points, None

        # 如果未启用自回归聚类，使用原有逻辑
        if not self.enable_autoregressive_clustering:
            clustering = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(points)
            labels = clustering.labels_  # -1表示噪声点

            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            print(f"[DBSCAN] Total points: {len(points)}, Clusters found: {n_clusters}, Noise points: {(labels == -1).sum()}")

            # 计算每个簇的中心
            merged = []
            for label in set(labels):
                if label == -1:
                    continue  # 跳过噪声点
                cluster_points = points[labels == label]
                center = np.mean(cluster_points, axis=0)
                merged.append(center)

            return np.array(merged), None
        
        # 自回归聚类逻辑
        current_eps = self.initial_eps if self.initial_eps is not None else self.eps
        current_min_samples = self.initial_min_samples if self.initial_min_samples is not None else self.min_samples
        all_merged_points = []  # 累积已合并的点
        remaining_points = points.copy()  # 剩余未处理的点
        iteration = 0
        
        history = None
        if record_history:
            history = ClusteringHistory(atom_type=atom_type if atom_type is not None else -1, iterations=[], total_atoms=0)
        
        while iteration < self.max_clustering_iterations:
            if len(remaining_points) == 0:
                break
            
            # 对remaining_points执行DBSCAN
            clustering = DBSCAN(eps=current_eps, min_samples=current_min_samples).fit(remaining_points)
            labels = clustering.labels_  # -1表示噪声点
            
            # 收集所有簇
            clusters_info = []  # [(label, points), ...]
            for label in set(labels):
                if label == -1:
                    continue
                cluster_points = remaining_points[labels == label]
                clusters_info.append((label, cluster_points))
            
            # 处理所有簇
            new_merged_this_iteration = []
            new_atoms_coords = []
            new_atoms_types = []
            n_clusters_found = len(clusters_info)
            n_noise_points = (labels == -1).sum()
            
            for label, cluster_points in clusters_info:
                # 计算簇中心
                center = np.mean(cluster_points, axis=0)
                
                # 键长检查（如果有参考点）
                bond_validation_passed = True
                if reference_points is not None and len(reference_points) > 0:
                    bond_validation_passed = self._check_bond_length_validity(
                        center, atom_type if atom_type is not None else -1,
                        reference_points, reference_types
                    )
                
                if bond_validation_passed:
                    # 通过检查，加入结果
                    new_merged_this_iteration.append(center)
                    new_atoms_coords.append(center)
                    if atom_type is not None:
                        new_atoms_types.append(atom_type)
                    all_merged_points.append(center)
                    
                    # 从remaining_points中移除该簇的点
                    mask = labels != label
                    remaining_points = remaining_points[mask]
                    labels = labels[mask]
            
            # 记录历史
            if record_history and history is not None:
                record = ClusteringIterationRecord(
                    iteration=iteration,
                    eps=current_eps,
                    min_samples=current_min_samples,
                    new_atoms_coords=np.array(new_atoms_coords) if len(new_atoms_coords) > 0 else np.empty((0, 3)),
                    new_atoms_types=np.array(new_atoms_types) if len(new_atoms_types) > 0 else np.empty((0,), dtype=np.int64),
                    n_clusters_found=n_clusters_found,
                    n_atoms_clustered=len(new_merged_this_iteration),
                    n_noise_points=n_noise_points,
                    bond_validation_passed=True  # 简化，实际每个簇都有检查
                )
                history.iterations.append(record)
            
            # 更新参数
            current_eps = max(current_eps * self.eps_decay_factor, self.min_eps)
            current_min_samples = max(int(current_min_samples * self.min_samples_decay_factor), self.min_min_samples)
            
            # 检查终止条件
            if len(new_merged_this_iteration) == 0:
                break
            
            iteration += 1
        
        if history is not None:
            history.total_atoms = len(all_merged_points)
        
        result = np.array(all_merged_points) if len(all_merged_points) > 0 else np.empty((0, 3))
        return result, history

    def compute_reconstruction_metrics(
        self,
        recon_coords: torch.Tensor,
        recon_types: torch.Tensor,
        gt_coords: torch.Tensor,
        gt_types: torch.Tensor
    ) -> Dict[str, float]:
        """
        计算重建分子与真实分子之间的评估指标。
        
        Args:
            recon_coords: 重建的分子坐标 [batch_size, n_atoms, 3]
            recon_types: 重建的原子类型 [batch_size, n_atoms]
            gt_coords: 真实的分子坐标 [batch_size, n_atoms, 3]
            gt_types: 真实的原子类型 [batch_size, n_atoms]
            
        Returns:
            Dict[str, float]: 包含各种评估指标的字典
        """
        from funcmol.utils.utils_nf import compute_rmsd
        
        batch_size = recon_coords.size(0)
        metrics = {
            'avg_rmsd': 0.0,
            'min_rmsd': float('inf'),
            'max_rmsd': 0.0,
            'successful_reconstructions': 0
        }
        
        rmsd_values = []
        
        for b in range(batch_size):
            # 过滤掉填充的原子
            gt_mask = gt_types[b] != PADDING_INDEX
            recon_mask = recon_types[b] != -1  # 假设-1是填充值
            
            if gt_mask.sum() > 0 and recon_mask.sum() > 0:
                gt_valid_coords = gt_coords[b, gt_mask]
                recon_valid_coords = recon_coords[b, recon_mask]
                
                # 计算RMSD
                rmsd = compute_rmsd(gt_valid_coords, recon_valid_coords)
                rmsd_value = rmsd.item() if hasattr(rmsd, 'item') else float(rmsd)
                rmsd_values.append(rmsd_value)
                
                metrics['avg_rmsd'] += rmsd_value
                metrics['min_rmsd'] = min(metrics['min_rmsd'], rmsd_value)
                metrics['max_rmsd'] = max(metrics['max_rmsd'], rmsd_value)
                metrics['successful_reconstructions'] += 1
        
        if metrics['successful_reconstructions'] > 0:
            metrics['avg_rmsd'] /= metrics['successful_reconstructions']
            metrics['rmsd_std'] = np.std(rmsd_values) if rmsd_values else 0.0
        else:
            metrics['avg_rmsd'] = float('inf')
            metrics['rmsd_std'] = 0.0
        
        return metrics
    
    def _save_clustering_history(
        self,
        histories: List[ClusteringHistory],
        output_dir: str,
        batch_idx: int = 0,
        elements: Optional[List[str]] = None
    ) -> None:
        """
        保存聚类历史为SDF文件（每轮一个分子）和文本文件（详细记录）
        
        Args:
            histories: 聚类历史列表
            output_dir: 输出目录
            batch_idx: batch索引
            elements: 原子类型符号列表，如 ["C", "H", "O", "N", "F"]
        """
        from pathlib import Path
        
        if elements is None:
            # 默认元素列表
            elements = [ELEMENTS_HASH_INV.get(i, f"Type{i}") for i in range(self.n_atom_types)]
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 为每个原子类型保存
        for history in histories:
            atom_symbol = elements[history.atom_type] if history.atom_type < len(elements) else f"Type{history.atom_type}"
            
            # 1. 保存为SDF文件（每轮一个分子）
            sdf_path = output_path / f"batch_{batch_idx}_type_{atom_symbol}_clustering_history.sdf"
            self._save_clustering_history_sdf(history, sdf_path, elements)
            
            # 2. 保存为文本文件（详细记录）
            txt_path = output_path / f"batch_{batch_idx}_type_{atom_symbol}_clustering_history.txt"
            self._save_clustering_history_txt(history, txt_path, elements)
    
    def _save_clustering_history_sdf(
        self,
        history: ClusteringHistory,
        output_path: Path,
        elements: List[str]
    ) -> None:
        """保存聚类历史为SDF文件，每轮一个分子"""
        try:
            from funcmol.sample_fm import xyz_to_sdf
            
            sdf_strings = []
            for record in history.iterations:
                if len(record.new_atoms_coords) > 0:
                    sdf_str = xyz_to_sdf(
                        record.new_atoms_coords,
                        record.new_atoms_types,
                        elements
                    )
                    # 修改SDF标题，包含迭代信息
                    lines = sdf_str.split('\n')
                    lines[0] = f"Clustering Iteration {record.iteration}, eps={record.eps:.4f}, min_samples={record.min_samples}, atoms={record.n_atoms_clustered}"
                    sdf_strings.append('\n'.join(lines))
            
            # 写入文件
            if sdf_strings:
                with open(output_path, 'w') as f:
                    f.write(''.join(sdf_strings))
        except Exception as e:
            print(f"Warning: Failed to save clustering history SDF: {e}")
    
    def _save_clustering_history_txt(
        self,
        history: ClusteringHistory,
        output_path: Path,
        elements: List[str]
    ) -> None:
        """保存聚类历史为文本文件，包含详细信息"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("自回归聚类过程详细记录\n")
                f.write("=" * 80 + "\n\n")
                
                atom_symbol = elements[history.atom_type] if history.atom_type < len(elements) else f"Type{history.atom_type}"
                f.write(f"原子类型: {atom_symbol} (索引: {history.atom_type})\n")
                f.write(f"总迭代轮数: {len(history.iterations)}\n")
                f.write(f"最终原子总数: {history.total_atoms}\n")
                f.write("-" * 80 + "\n\n")
                
                for record in history.iterations:
                    f.write(f"迭代轮数: {record.iteration}\n")
                    f.write(f"  阈值: eps={record.eps:.4f}, min_samples={record.min_samples}\n")
                    f.write(f"  找到簇数: {record.n_clusters_found}\n")
                    f.write(f"  聚类原子数: {record.n_atoms_clustered}\n")
                    f.write(f"  噪声点数: {record.n_noise_points}\n")
                    f.write(f"  键长检查通过: {record.bond_validation_passed}\n")
                    
                    if len(record.new_atoms_coords) > 0:
                        f.write(f"  新聚类原子坐标:\n")
                        for i, (coord, atom_type) in enumerate(zip(record.new_atoms_coords, record.new_atoms_types)):
                            atom_sym = elements[atom_type] if atom_type < len(elements) else f"Type{atom_type}"
                            f.write(f"    {i+1}. {atom_sym}: ({coord[0]:.4f}, {coord[1]:.4f}, {coord[2]:.4f})\n")
                    else:
                        f.write(f"  本轮无新原子被聚类\n")
                    f.write("\n")
                
                f.write("\n" + "=" * 80 + "\n\n")
        except Exception as e:
            print(f"Warning: Failed to save clustering history text: {e}")
    