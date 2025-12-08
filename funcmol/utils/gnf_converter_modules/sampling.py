"""
采样模块：采样点生成和选择
"""
import torch
import torch.nn as nn
import numpy as np
import time
from typing import Optional, Tuple, List, Dict
from sklearn.cluster import DBSCAN
from funcmol.utils.gnf_converter_modules.dataclasses import ClusteringHistory


class SamplingProcessor:
    """采样处理器"""
    
    def __init__(
        self,
        gradient_field_computer,
        gradient_ascent_optimizer,
        clustering_processor,
        n_query_points: int,
        n_query_points_per_type: Dict[int, int],
        gradient_sampling_candidate_multiplier: int = 3,
        field_variance_k_neighbors: int = 10,
        field_variance_weight: float = 1.0,
        eps: float = 0.5,
        min_min_samples: int = 2,
        enable_clustering_history: bool = False,
    ):
        """
        初始化采样处理器
        
        Args:
            gradient_field_computer: GradientFieldComputer实例
            gradient_ascent_optimizer: GradientAscentOptimizer实例
            clustering_processor: ClusteringProcessor实例
            n_query_points: 默认query_points数
            n_query_points_per_type: 每个原子类型的query_points数字典
            gradient_sampling_candidate_multiplier: 梯度采样候选点倍数
            field_variance_k_neighbors: 计算field方差时使用的最近邻数量
            field_variance_weight: field方差在采样概率中的权重
            eps: DBSCAN的eps参数（用于初步聚类）
            min_min_samples: min_samples下限（用于初步聚类）
            enable_clustering_history: 是否记录聚类历史
        """
        self.gradient_field_computer = gradient_field_computer
        self.gradient_ascent_optimizer = gradient_ascent_optimizer
        self.clustering_processor = clustering_processor
        self.n_query_points = n_query_points
        self.n_query_points_per_type = n_query_points_per_type
        self.gradient_sampling_candidate_multiplier = gradient_sampling_candidate_multiplier
        self.field_variance_k_neighbors = field_variance_k_neighbors
        self.field_variance_weight = field_variance_weight
        self.eps = eps
        self.min_min_samples = min_min_samples
        self.enable_clustering_history = enable_clustering_history
    
    def process_atom_types_matrix(
        self,
        current_codes: torch.Tensor,
        n_atom_types: int,
        device: torch.device,
        decoder: nn.Module,
        iteration_callback: Optional[callable] = None,
        element_existence: Optional[torch.Tensor] = None,
        gradient_ascent_callback: Optional[callable] = None,
        enable_timing: bool = False
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[ClusteringHistory]]:
        """
        完全矩阵化处理所有原子类型，避免所有循环。
        使用图结构思想：构建同种原子内部连边的图，一次性处理所有原子类型。
        支持批量处理：如果current_codes是[B, grid_size**3, code_dim]，则批量处理B个样本。
        
        Args:
            current_codes: 当前batch的编码 [1, grid_size**3, code_dim] 或 [B, grid_size**3, code_dim]（批量模式）
            n_atom_types: 原子类型数量
            device: 设备
            decoder: 解码器模型
            iteration_callback: 可选的回调函数，签名: (iter_idx, current_points, atom_types, batch_idx=None)
            element_existence: 可选的元素存在性向量 [n_atom_types] 或 [B, n_atom_types]（批量模式），如果提供，只处理存在的元素类型
            gradient_ascent_callback: 梯度上升回调，签名: (iter_idx, current_points, atom_types, batch_idx=None)
            enable_timing: 是否启用时间统计
            
        Returns:
            (coords_list, types_list, histories): 坐标列表、类型列表和聚类历史列表
            如果是批量模式，返回的列表包含B个子列表，每个子列表对应一个batch的结果
        """
        # 判断是否是批量模式
        # current_codes的形状是[B, grid_size**3, code_dim]
        if current_codes.dim() == 3:
            # 3维张量：[B, grid_size**3, code_dim]
            is_batch_mode = current_codes.size(0) > 1
            B = current_codes.size(0)
        elif current_codes.dim() == 2:
            # 2维张量：[grid_size**3, code_dim] 或 [1, grid_size**3, code_dim]被squeeze了
            # 这种情况应该很少见，但为了兼容性处理
            is_batch_mode = False
            B = 1
            current_codes = current_codes.unsqueeze(0)  # [1, grid_size**3, code_dim]
        else:
            raise ValueError(f"Unexpected current_codes shape: {current_codes.shape}, expected [B, grid_size**3, code_dim]")
        
        # 处理element_existence：为每个batch确定存在的元素类型
        existing_types_per_batch = []
        if element_existence is not None:
            # element_existence: [n_atom_types] 或 [1, n_atom_types] 或 [B, n_atom_types]
            if element_existence.dim() == 1:
                # [n_atom_types] - 所有batch共享
                element_existence = element_existence.unsqueeze(0).expand(B, -1)  # [B, n_atom_types]
            elif element_existence.dim() == 2 and element_existence.size(0) == 1:
                # [1, n_atom_types] - 扩展到所有batch
                element_existence = element_existence.expand(B, -1)  # [B, n_atom_types]
            # 现在element_existence是[B, n_atom_types]
            
            for b in range(B):
                element_mask = (element_existence[b] > 0.5).float()
                existing_types = torch.nonzero(element_mask, as_tuple=False).squeeze(-1).tolist()
                if len(existing_types) == 0:
                    existing_types_per_batch.append([])
                else:
                    existing_types_per_batch.append(existing_types)
        else:
            # 如果没有提供element_existence，所有batch处理所有类型
            existing_types = list(range(n_atom_types))
            existing_types_per_batch = [existing_types] * B
        
        # 检查是否有batch没有任何元素存在
        if all(len(types) == 0 for types in existing_types_per_batch):
            # 所有batch都没有元素存在，返回空列表
            if is_batch_mode:
                return [[] for _ in range(B)], [[] for _ in range(B)], [[] for _ in range(B)]
            else:
                return [], [], []
        
        # 获取所有batch的并集元素类型（用于确定采样点数）
        all_existing_types = set()
        for types in existing_types_per_batch:
            all_existing_types.update(types)
        all_existing_types = sorted(list(all_existing_types))
        
        # 获取每个原子类型的 query_points 数（使用所有batch的并集）
        query_points_per_type = [self.n_query_points_per_type.get(t, self.n_query_points) for t in all_existing_types]
        max_query_points = max(query_points_per_type) if query_points_per_type else self.n_query_points
        
        # 时间记录
        timing_info = {} if enable_timing else None
        
        # 1. 初始化采样点 - 为所有原子类型一次性采样
        t_init_start = time.perf_counter() if enable_timing else None
        init_min, init_max = -7.0, 7.0
        n_candidates = max_query_points * self.gradient_sampling_candidate_multiplier
        candidate_points = torch.rand(n_candidates, 3, device=device) * (init_max - init_min) + init_min
        if enable_timing:
            timing_info['init_sampling'] = time.perf_counter() - t_init_start
        
        # 计算候选点的梯度场强度 - 批量处理所有batch
        candidate_batch = candidate_points.unsqueeze(0).expand(B, -1, -1)  # [B, n_candidates, 3]
        
        t_field_start = time.perf_counter() if enable_timing else None
        try:
            # 使用torch.no_grad()包装候选点采样 - 批量调用decoder
            with torch.no_grad():
                candidate_field = decoder(candidate_batch, current_codes)  # [B, n_candidates, n_atom_types, 3]
        except Exception as e:
            raise e
        if enable_timing:
            timing_info['candidate_field'] = time.perf_counter() - t_field_start
        
        # 2. 完全矩阵化采样 - 为每个batch分别处理（因为element_existence可能不同）
        # 计算每个候选点的field方差（为每个原子类型分别计算），基于方差采样
        t_variance_start = time.perf_counter() if enable_timing else None
        # 为每个batch分别计算方差
        field_variances_per_batch = []
        for b in range(B):
            field_variances_per_type = self.gradient_field_computer.compute_field_variance(
                candidate_points, candidate_field[b], 
                self.field_variance_k_neighbors, 
                per_type_independent=True
            )  # [n_candidates, n_atom_types] 或 [n_candidates]
            field_variances_per_batch.append(field_variances_per_type)
        if enable_timing:
            timing_info['field_variance'] = time.perf_counter() - t_variance_start
        
        # 为每个batch分别采样点（因为element_existence可能不同）
        t_sampling_start = time.perf_counter() if enable_timing else None
        
        # 为每个batch存储采样点和类型
        all_batch_sampled_points = []  # List[List[Tensor]] - 每个batch一个列表
        all_batch_atom_types = []  # List[List[Tensor]] - 每个batch一个列表
        all_batch_type_start_indices = []  # List[List[int]] - 每个batch的类型起始索引
        
        for b in range(B):
            existing_types = existing_types_per_batch[b]
            if len(existing_types) == 0:
                # 这个batch没有元素存在，跳过
                all_batch_sampled_points.append([])
                all_batch_atom_types.append([])
                all_batch_type_start_indices.append([])
                continue
            
            # 获取这个batch的query_points数
            query_points_per_type_b = [self.n_query_points_per_type.get(t, self.n_query_points) for t in existing_types]
            
            all_sampled_points_b = []
            all_atom_types_b = []
            type_start_indices_b = []
            current_start_idx = 0
            
            # 获取这个batch的方差
            field_variances_per_type = field_variances_per_batch[b]
            
            # 为每个存在的元素类型采样点
            for type_idx, t in enumerate(existing_types):
                n_query_points_t = query_points_per_type_b[type_idx]
                
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
                probabilities = torch.softmax(normalized_variances / (self.field_variance_weight + 1e-8), dim=0)
                
                # 从候选点中采样 n_query_points_t 个点
                replacement = n_query_points_t > n_candidates
                sampled_indices = torch.multinomial(probabilities, n_query_points_t, replacement=replacement)
                z = candidate_points[sampled_indices]  # [n_query_points_t, 3]
                
                all_sampled_points_b.append(z)
                all_atom_types_b.append(torch.full((n_query_points_t,), t, dtype=torch.long, device=device))
                
                # 记录起始索引
                type_start_indices_b.append(current_start_idx)
                current_start_idx += n_query_points_t
            
            all_batch_sampled_points.append(all_sampled_points_b)
            all_batch_atom_types.append(all_atom_types_b)
            all_batch_type_start_indices.append(type_start_indices_b)
        
        if enable_timing:
            timing_info['point_sampling'] = time.perf_counter() - t_sampling_start
            timing_info.setdefault('gradient_ascent', 0.0)
            timing_info.setdefault('clustering', 0.0)
        
        # 批量梯度上升：将所有batch的点合并，批量处理
        # 首先检查所有batch是否有采样点
        has_points = any(len(points) > 0 for points in all_batch_sampled_points)
        
        if has_points:
            # 合并所有batch的采样点
            if is_batch_mode:
                # 批量模式：将所有batch的点合并成 [B, total_points, 3]
                # 需要确保所有batch的点数相同（用户已确认）
                combined_points_list = []
                combined_types_list = []
                
                # 首先检查所有batch的点数是否相同
                total_points_per_batch = []
                for b in range(B):
                    if len(all_batch_sampled_points[b]) > 0:
                        total_points_b = sum(p.size(0) for p in all_batch_sampled_points[b])
                        total_points_per_batch.append(total_points_b)
                    else:
                        total_points_per_batch.append(0)
                
                # 验证所有batch的点数相同
                if len(set(total_points_per_batch)) > 1:
                    raise ValueError(f"批量模式下，所有batch的点数必须相同，但得到: {total_points_per_batch}")
                
                total_points = total_points_per_batch[0] if total_points_per_batch else 0
                
                for b in range(B):
                    if len(all_batch_sampled_points[b]) > 0:
                        combined_points_b = torch.cat(all_batch_sampled_points[b], dim=0)  # [total_points, 3]
                        combined_types_b = torch.cat(all_batch_atom_types[b], dim=0)  # [total_points]
                    else:
                        # 这个batch没有点，创建空tensor（但需要与其他batch维度匹配）
                        # 这种情况不应该发生（因为用户说所有batch点数相同），但为了安全
                        combined_points_b = torch.empty(0, 3, device=device)
                        combined_types_b = torch.empty(0, dtype=torch.long, device=device)
                    
                    combined_points_list.append(combined_points_b)
                    combined_types_list.append(combined_types_b)
                
                # 堆叠成批量格式 [B, total_points, 3]
                # 所有batch的点数相同，可以安全stack
                combined_points = torch.stack(combined_points_list, dim=0)  # [B, total_points, 3]
                combined_types = torch.stack(combined_types_list, dim=0)  # [B, total_points]
            else:
                # 单样本模式（向后兼容）
                combined_points = torch.cat(all_batch_sampled_points[0], dim=0)  # [total_points, 3]
                combined_types = torch.cat(all_batch_atom_types[0], dim=0)  # [total_points]
            
            # 合并两个回调
            combined_callback = None
            if iteration_callback is not None or gradient_ascent_callback is not None:
                def combined(iter_idx, current_points, atom_types, batch_idx=None):
                    def safe_call(callback, iter_idx, points, types, batch_idx_val=None):
                        """安全调用回调函数，兼容支持和不支持batch_idx的情况"""
                        if callback is None:
                            return
                        # 先尝试传递batch_idx（支持batch_idx_inner参数名）
                        try:
                            callback(iter_idx, points, types, batch_idx_inner=batch_idx_val)
                        except TypeError:
                            # 如果不支持batch_idx_inner，尝试batch_idx
                            try:
                                callback(iter_idx, points, types, batch_idx=batch_idx_val)
                            except TypeError:
                                # 如果不支持batch_idx，只传递基本参数
                                callback(iter_idx, points, types)
                    
                    if batch_idx is None:
                        # 批量模式：current_points是[B, n_points, 3]
                        if iteration_callback is not None:
                            # 直接传递整个批次，批量回调函数会自己处理
                            try:
                                iteration_callback(iter_idx, current_points, atom_types, batch_idx=None)
                            except TypeError:
                                # 如果批量回调失败，尝试逐个调用（向后兼容）
                                B_cb = current_points.size(0)
                                for b in range(B_cb):
                                    safe_call(iteration_callback, iter_idx, current_points[b], atom_types[b], batch_idx_val=b)
                        
                        if gradient_ascent_callback is not None:
                            # 直接传递整个批次，批量回调函数会自己处理
                            try:
                                gradient_ascent_callback(iter_idx, current_points, atom_types, batch_idx=None)
                            except TypeError:
                                # 如果批量回调失败，尝试逐个调用（向后兼容）
                                B_cb = current_points.size(0)
                                for b in range(B_cb):
                                    safe_call(gradient_ascent_callback, iter_idx, current_points[b], atom_types[b], batch_idx_val=b)
                    else:
                        # 单样本模式：current_points是[n_points, 3]
                        if iteration_callback is not None:
                            safe_call(iteration_callback, iter_idx, current_points, atom_types, batch_idx_val=batch_idx)
                        if gradient_ascent_callback is not None:
                            safe_call(gradient_ascent_callback, iter_idx, current_points, atom_types, batch_idx_val=batch_idx)
                combined_callback = combined
            
            # 批量梯度上升
            t_gradient_start = time.perf_counter() if enable_timing else None
            final_points = self.gradient_ascent_optimizer.batch_gradient_ascent(
                combined_points, combined_types, current_codes, device, decoder,
                iteration_callback=combined_callback,
                enable_timing=enable_timing
            )
            if enable_timing:
                timing_info['gradient_ascent'] = time.perf_counter() - t_gradient_start
            
            # 按batch和原子类型分离结果并进行聚类
            t_clustering_start = time.perf_counter() if enable_timing else None
            
            # 为每个batch分别处理聚类
            all_batch_coords = []
            all_batch_types = []
            all_batch_histories = []
            
            for b in range(B):
                existing_types = existing_types_per_batch[b]
                if len(existing_types) == 0:
                    # 这个batch没有元素存在
                    all_batch_coords.append([])
                    all_batch_types.append([])
                    all_batch_histories.append([])
                    continue
                
                # 获取这个batch的最终点
                if is_batch_mode:
                    final_points_b = final_points[b]  # [total_points, 3]
                else:
                    final_points_b = final_points  # [total_points, 3]
                
                # 获取这个batch的类型起始索引和query_points数
                type_start_indices = all_batch_type_start_indices[b]
                query_points_per_type_b = [self.n_query_points_per_type.get(t, self.n_query_points) for t in existing_types]
                
                coords_list_b = []
                types_list_b = []
                all_clustering_histories_b = []
                
                # 初始化全局参考点（用于跨原子类型的键长检查）
                all_reference_points = []
                all_reference_types = []
                
                # 按原子类型分离结果并进行聚类
                for type_idx, t in enumerate(existing_types):
                    # 在处理当前原子类型之前，预先对后续原子类型进行初始聚类，找出所有簇作为参考点
                    if type_idx < len(existing_types) - 1:  # 不是最后一个原子类型
                        preliminary_points = []
                        preliminary_types = []
                        for future_type_idx in range(type_idx + 1, len(existing_types)):
                            future_t = existing_types[future_type_idx]
                            future_start_idx = type_start_indices[future_type_idx]
                            future_end_idx = future_start_idx + query_points_per_type_b[future_type_idx]
                            future_type_points = final_points_b[future_start_idx:future_end_idx]
                            future_z_np = future_type_points.detach().cpu().numpy()
                            
                            # 进行初始聚类，找出所有簇（不进行键长检查）
                            if len(future_z_np) > 0:
                                initial_clustering = DBSCAN(eps=self.eps, min_samples=self.min_min_samples).fit(future_z_np)
                                initial_labels = initial_clustering.labels_
                                for label in set(initial_labels):
                                    if label == -1:
                                        continue  # 跳过噪声点
                                    cluster_points = future_z_np[initial_labels == label]
                                    center = np.mean(cluster_points, axis=0)
                                    preliminary_points.append(center)
                                    preliminary_types.append(future_t)
                        
                        # 将后续原子类型的初步簇添加到参考点中（用于当前原子类型的键长检查）
                        if len(preliminary_points) > 0:
                            all_reference_points.extend(preliminary_points)
                            all_reference_types.extend(preliminary_types)
                    
                    start_idx = type_start_indices[type_idx]
                    end_idx = start_idx + query_points_per_type_b[type_idx]
                    type_points = final_points_b[start_idx:end_idx]  # [n_query_points_t, 3]
                    
                    # 聚类/合并
                    z_np = type_points.detach().cpu().numpy()
                    merged_points, history = self.clustering_processor.merge_points(
                        z_np, 
                        atom_type=t,
                        reference_points=np.array(all_reference_points) if len(all_reference_points) > 0 else None,
                        reference_types=np.array(all_reference_types) if len(all_reference_types) > 0 else None,
                        record_history=self.enable_clustering_history
                    )
                    
                    if len(merged_points) > 0:
                        merged_tensor = torch.from_numpy(merged_points).to(device)
                        coords_list_b.append(merged_tensor)
                        types_list_b.append(torch.full((len(merged_points),), t, dtype=torch.long, device=device))
                        
                        # 更新全局参考点（用于后续原子类型的键长检查）
                        all_reference_points.extend(merged_points.tolist())
                        all_reference_types.extend([t] * len(merged_points))
                    
                    if history is not None:
                        all_clustering_histories_b.append(history)
                
                all_batch_coords.append(coords_list_b)
                all_batch_types.append(types_list_b)
                all_batch_histories.append(all_clustering_histories_b)
            
            if enable_timing:
                timing_info['clustering'] = time.perf_counter() - t_clustering_start
            
            # 根据批量模式返回结果
            if is_batch_mode:
                # 批量模式：返回每个batch的结果列表
                return all_batch_coords, all_batch_types, all_batch_histories
            else:
                # 单样本模式：返回单个batch的结果
                return all_batch_coords[0], all_batch_types[0], all_batch_histories[0]
        else:
            # 没有采样点
            if is_batch_mode:
                return [[] for _ in range(B)], [[] for _ in range(B)], [[] for _ in range(B)]
            else:
                return [], [], []
        
        # 打印详细时间信息
        if enable_timing and timing_info:
            total_time = sum(timing_info.values())
            print(f"[详细时间] init_sampling={timing_info.get('init_sampling', 0):.3f}s, "
                  f"candidate_field={timing_info.get('candidate_field', 0):.3f}s, "
                  f"field_variance={timing_info.get('field_variance', 0):.3f}s, "
                  f"point_sampling={timing_info.get('point_sampling', 0):.3f}s, "
                  f"gradient_ascent={timing_info.get('gradient_ascent', 0):.3f}s, "
                  f"clustering={timing_info.get('clustering', 0):.3f}s, "
                  f"total={total_time:.3f}s")
        
        return coords_list, types_list, all_clustering_histories

