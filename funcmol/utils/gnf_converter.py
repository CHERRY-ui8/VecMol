import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, List, Dict
from sklearn.cluster import DBSCAN
import torch.nn.functional as F
from funcmol.utils.constants import PADDING_INDEX
import gc

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
                gradient_sampling_candidate_multiplier: int = 10,  # 梯度采样候选点倍数
                gradient_sampling_temperature: float = 0.1,  # 梯度采样温度参数
                n_atom_types: int = 5):  # 原子类型数量，默认为5以保持向后兼容
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
        self.gradient_sampling_temperature = gradient_sampling_temperature  # 保存梯度采样温度参数
        self.n_atom_types = n_atom_types  # 保存原子类型数量
        
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

    def _process_atom_types_matrix(self, current_codes: torch.Tensor, n_atom_types: int, 
                                 n_query_points: int, device: torch.device, 
                                 decoder: nn.Module, fabric: object = None) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        完全矩阵化处理所有原子类型，避免所有循环。
        使用图结构思想：构建同种原子内部连边的图，一次性处理所有原子类型。
        
        Args:
            current_codes: 当前batch的编码 [1, code_dim]
            n_atom_types: 原子类型数量
            n_query_points: 查询点数量
            device: 设备
            fabric: 日志对象
            
        Returns:
            (coords_list, types_list): 坐标和类型列表
        """
        # 1. 初始化采样点 - 为所有原子类型一次性采样
        init_min, init_max = -7.0, 7.0
        n_candidates = n_query_points * self.gradient_sampling_candidate_multiplier
        candidate_points = torch.rand(n_candidates, 3, device=device) * (init_max - init_min) + init_min
        
        # 计算候选点的梯度场强度
        candidate_batch = candidate_points.unsqueeze(0)  # [1, n_candidates, 3]
        
        try:
            # 使用torch.no_grad()包装候选点采样
            with torch.no_grad():
                candidate_field = decoder(candidate_batch, current_codes)
            # 移除强制内存清理，让PyTorch自动管理
        except Exception as e:
            if fabric:
                fabric.print(f">> ERROR in decoder call: {e}")
            raise e
        
        # 2. 完全矩阵化采样 - 一次性处理所有原子类型
        # 计算所有原子类型的梯度场强度 [n_candidates, n_atom_types]
        grad_magnitudes = torch.norm(candidate_field[0], dim=-1)  # [n_candidates, n_atom_types]
        
        # 为每个原子类型采样点 - 矩阵化操作
        all_sampled_points = []
        all_atom_types = []
        
        # 初始化结果列表
        coords_list = []
        types_list = []
        
        # TODO：使用矩阵化采样，避免循环
        for t in range(n_atom_types):
            candidate_grad = candidate_field[0, :, t, :]  # [n_candidates, 3]
            
            # 计算梯度场强度（模长）
            grad_magnitudes = torch.norm(candidate_grad, dim=1)  # [n_candidates]
            
            # 根据梯度场强度进行加权采样
            probabilities = torch.softmax(grad_magnitudes / self.gradient_sampling_temperature, dim=0)
            
            # 从候选点中采样n_query_points个点
            sampled_indices = torch.multinomial(probabilities, n_query_points, replacement=False)
            z = candidate_points[sampled_indices]  # [n_query_points, 3]
            
            all_sampled_points.append(z)
            all_atom_types.append(torch.full((n_query_points,), t, dtype=torch.long, device=device))
        
        # 合并所有原子类型的采样点进行批量梯度上升
        if all_sampled_points:
            # 将所有采样点合并 [n_atom_types * n_query_points, 3]
            combined_points = torch.cat(all_sampled_points, dim=0)
            combined_types = torch.cat(all_atom_types, dim=0)  # [n_atom_types * n_query_points]
            
            # 批量梯度上升
            final_points = self._batch_gradient_ascent(
                combined_points, combined_types, current_codes, device, decoder, fabric
            )
            
            # 按原子类型分离结果并进行聚类
            for t in range(n_atom_types):
                start_idx = t * n_query_points
                end_idx = (t + 1) * n_query_points
                type_points = final_points[start_idx:end_idx]  # [n_query_points, 3]
                
                # 聚类/合并
                z_np = type_points.detach().cpu().numpy()
                merged_points = self._merge_points(z_np)
                if len(merged_points) > 0:
                    coords_list.append(torch.from_numpy(merged_points).to(device))
                    types_list.append(torch.full((len(merged_points),), t, dtype=torch.long, device=device))
        
        return coords_list, types_list

    def _batch_gradient_ascent(self, points: torch.Tensor, atom_types: torch.Tensor,
                              current_codes: torch.Tensor, device: torch.device,
                              decoder: nn.Module, fabric: object = None) -> torch.Tensor:
        """
        批量梯度上升，对所有原子类型的点同时进行梯度上升。
        支持自适应停止：当梯度变化很小时提前停止。
        
        Args:
            points: 采样点 [n_total_points, 3]
            atom_types: 原子类型 [n_total_points]
            current_codes: 当前编码 [1, code_dim]
            device: 设备
            fabric: 日志对象
            
        Returns:
            final_points: 最终点位置 [n_total_points, 3]
        """
        z = points.clone()
        prev_grad_norm = None
        convergence_threshold = 1e-6  # 收敛阈值
        min_iterations = 50  # 最少迭代次数
        
        for iter_idx in range(self.n_iter):
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
                
                # 检查收敛条件（在最少迭代次数之后）
                if iter_idx >= min_iterations and prev_grad_norm is not None:
                    grad_change = abs(current_grad_norm - prev_grad_norm)
                    if grad_change < convergence_threshold:
                        break
                
                prev_grad_norm = current_grad_norm
                
                # 更新采样点位置
                z = z + adjusted_step_sizes * grad
                
                # 完全移除内存清理以避免干扰梯度上升过程
                # 让PyTorch自动管理内存
                if fabric and iter_idx % 50 == 0:
                    allocated = torch.cuda.memory_allocated() / 1024**3
                    reserved = torch.cuda.memory_reserved() / 1024**3
                    fabric.print(f">>     Memory status at iteration {iter_idx}: Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB")
                
            except (RuntimeError, ValueError, IndexError) as e:
                if fabric:
                    fabric.print(f">> ERROR in batch gradient ascent iteration {iter_idx}: {e}")
                # 发生错误时也清理内存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()
                break
        
        # 移除强制内存清理，让PyTorch自动管理
        
        return z

    def gnf2mol(self, decoder: nn.Module, codes: torch.Tensor,
                atom_types: Optional[torch.Tensor] = None,
                fabric: object = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        直接用梯度场重建分子坐标。
        Args:
            decoder: 解码器模型，用于动态计算向量场
            codes: [batch, grid_size**3, code_dim]  # 编码器的输出
            atom_types: 可选的原子类型
        Returns:
            (final_coords, final_types)
        """
        device = codes.device
        batch_size = codes.size(0)
        n_atom_types = self.n_atom_types
        n_query_points = self.n_query_points

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
            
            # 矩阵化处理所有原子类型
            coords_list, types_list = self._process_atom_types_matrix(
                current_codes, n_atom_types, n_query_points, device, decoder, fabric
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



    def _merge_points(self, points: np.ndarray) -> np.ndarray:
        """Use DBSCAN to merge close points and determine atom centers."""
        if len(points.shape) == 3:
            points = points.reshape(-1, 3)
        elif len(points.shape) != 2 or points.shape[1] != 3:
            raise ValueError(f"Expected points shape (n, 3), got {points.shape}")

        if len(points) == 0:
            return points

        # 使用类的DBSCAN参数
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

        return np.array(merged)

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
    