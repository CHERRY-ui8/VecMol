"""
采样模块：采样点生成和选择
"""
import torch
import torch.nn as nn
import numpy as np
import time
from typing import Optional, Tuple, List, Dict
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
        
        Args:
            current_codes: 当前batch的编码 [1, code_dim]
            n_atom_types: 原子类型数量
            device: 设备
            decoder: 解码器模型
            iteration_callback: 可选的回调函数
            element_existence: 可选的元素存在性向量 [n_atom_types]，如果提供，只处理存在的元素类型
            gradient_ascent_callback: 梯度上升回调
            enable_timing: 是否启用时间统计
            
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
        
        # 时间记录
        timing_info = {} if enable_timing else None
        
        # 1. 初始化采样点 - 为所有原子类型一次性采样
        t_init_start = time.perf_counter() if enable_timing else None
        init_min, init_max = -7.0, 7.0
        n_candidates = max_query_points * self.gradient_sampling_candidate_multiplier
        candidate_points = torch.rand(n_candidates, 3, device=device) * (init_max - init_min) + init_min
        if enable_timing:
            timing_info['init_sampling'] = time.perf_counter() - t_init_start
        
        # 计算候选点的梯度场强度
        candidate_batch = candidate_points.unsqueeze(0)  # [1, n_candidates, 3]
        
        t_field_start = time.perf_counter() if enable_timing else None
        try:
            # 使用torch.no_grad()包装候选点采样
            with torch.no_grad():
                candidate_field = decoder(candidate_batch, current_codes)
        except Exception as e:
            raise e
        if enable_timing:
            timing_info['candidate_field'] = time.perf_counter() - t_field_start
        
        # 2. 完全矩阵化采样 - 一次性处理所有原子类型
        # 计算每个候选点的field方差（为每个原子类型分别计算），基于方差采样
        t_variance_start = time.perf_counter() if enable_timing else None
        field_variances_per_type = self.gradient_field_computer.compute_field_variance(
            candidate_points, candidate_field[0], 
            self.field_variance_k_neighbors, 
            per_type_independent=True
        )  # [n_candidates, n_atom_types] 或 [n_candidates]
        if enable_timing:
            timing_info['field_variance'] = time.perf_counter() - t_variance_start
        
        # 为每个原子类型采样点 - 矩阵化操作
        t_sampling_start = time.perf_counter() if enable_timing else None
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
            n_query_points_t = query_points_per_type[type_idx]  # 当前原子类型的 query_points 数
            
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
            
            all_sampled_points.append(z)
            all_atom_types.append(torch.full((n_query_points_t,), t, dtype=torch.long, device=device))
            
            # 记录起始索引
            type_start_indices.append(current_start_idx)
            current_start_idx += n_query_points_t
        
        if enable_timing:
            timing_info['point_sampling'] = time.perf_counter() - t_sampling_start
            timing_info.setdefault('gradient_ascent', 0.0)
            timing_info.setdefault('clustering', 0.0)
        
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
            t_gradient_start = time.perf_counter() if enable_timing else None
            final_points = self.gradient_ascent_optimizer.batch_gradient_ascent(
                combined_points, combined_types, current_codes, device, decoder,
                iteration_callback=combined_callback,
                enable_timing=enable_timing
            )
            if enable_timing:
                timing_info['gradient_ascent'] = time.perf_counter() - t_gradient_start
            
            # 按原子类型分离结果并进行聚类（只处理存在的类型）
            t_clustering_start = time.perf_counter() if enable_timing else None
            
            # 使用跨原子类型iteration循环
            # 1. 先为每个原子类型准备簇
            type_clusters_data = {}  # {atom_type: (clusters_with_tags, iteration_thresholds)}
            type_points_dict = {}  # {atom_type: points}
            
            for type_idx, t in enumerate(existing_types):
                start_idx = type_start_indices[type_idx]
                end_idx = start_idx + query_points_per_type[type_idx]
                type_points = final_points[start_idx:end_idx]  # [n_query_points_t, 3]
                z_np = type_points.detach().cpu().numpy()
                type_points_dict[t] = z_np
                
                # 准备簇
                clusters_with_tags, iteration_thresholds = self.clustering_processor._prepare_clusters(z_np)
                type_clusters_data[t] = (clusters_with_tags, iteration_thresholds)
            
            # 2. 外层循环iteration，内层循环原子类型
            max_iterations = self.clustering_processor.max_clustering_iterations
            type_pending_clusters = {t: [] for t in existing_types}  # {atom_type: pending_clusters}
            type_all_merged_points = {t: [] for t in existing_types}  # {atom_type: [centers, ...]}
            
            # 初始化全局参考点（使用numpy数组以便修改）
            global_ref_points = np.array(all_reference_points) if len(all_reference_points) > 0 else np.empty((0, 3))
            global_ref_types = np.array(all_reference_types) if len(all_reference_types) > 0 else np.empty((0,), dtype=np.int64)
            
            for current_iteration in range(max_iterations):
                # 内层循环：处理所有原子类型
                for type_idx, t in enumerate(existing_types):
                    clusters_with_tags, iteration_thresholds = type_clusters_data[t]
                    pending_clusters = type_pending_clusters[t]
                    
                    # 获取当前原子类型已通过的原子数量
                    current_type_atom_count = len(type_all_merged_points[t])
                    
                    # 处理当前iteration
                    new_atoms, new_pending, stats, updated_ref_points, updated_ref_types = \
                        self.clustering_processor.merge_points_single_iteration(
                            clusters_with_tags, iteration_thresholds, current_iteration,
                            atom_type=t,
                            reference_points=global_ref_points if len(global_ref_points) > 0 else None,
                            reference_types=global_ref_types if len(global_ref_types) > 0 else None,
                            pending_clusters=pending_clusters,
                            current_type_atom_count=current_type_atom_count
                        )
                    
                    # 更新pending簇
                    type_pending_clusters[t] = new_pending
                    
                    # 收集通过检查的原子
                    if len(new_atoms) > 0:
                        type_all_merged_points[t].extend(new_atoms.tolist())
                        # 更新全局参考点
                        global_ref_points = updated_ref_points
                        global_ref_types = updated_ref_types
                    
                    # 输出调试信息
                    if self.clustering_processor.debug_bond_validation:
                        from collections import Counter
                        from funcmol.utils.constants import ELEMENTS_HASH_INV
                        atom_symbol = ELEMENTS_HASH_INV.get(t, f"Type{t}") if t is not None else "Unknown"
                        print(f"\n{'='*60}")
                        print(f"[Iteration {current_iteration}] 原子类型: {atom_symbol}, min_samples={stats['current_min_samples']}")
                        print(f"  处理簇数: {stats['n_clusters_found']} (新簇: {stats['n_clusters_found'] - len(new_pending)}, 待重试: {len(new_pending)})")
                        print(f"  结果: ✓通过={stats['n_passed']}, ✗拒绝={stats['n_rejected']}")
                        if stats['n_rejected'] > 0:
                            reason_str = ", ".join([f"{k}={v}" for k, v in stats['rejection_reasons'].items() if v > 0])
                            print(f"  拒绝原因: {reason_str}")
                        if len(global_ref_points) > 0:
                            type_counts = Counter([ELEMENTS_HASH_INV.get(int(tt), f"Type{int(tt)}") for tt in global_ref_types])
                            print(f"  当前参考点: {len(global_ref_points)} 个, 类型分布: {dict(type_counts)}")
                        print(f"{'='*60}")
            
            # 3. 收集所有结果
            for type_idx, t in enumerate(existing_types):
                merged_points = np.array(type_all_merged_points[t]) if len(type_all_merged_points[t]) > 0 else np.empty((0, 3))
                if len(merged_points) > 0:
                    merged_tensor = torch.from_numpy(merged_points).to(device)
                    coords_list.append(merged_tensor)
                    types_list.append(torch.full((len(merged_points),), t, dtype=torch.long, device=device))
            
            if enable_timing:
                timing_info['clustering'] = time.perf_counter() - t_clustering_start
        
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

