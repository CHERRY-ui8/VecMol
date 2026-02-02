import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, List, Dict, Union
from pathlib import Path
from sklearn.cluster import DBSCAN
import torch.nn.functional as F
from funcmol.utils.constants import PADDING_INDEX, ELEMENTS_HASH_INV
import gc
import time

# 导入拆解后的模块
from funcmol.utils.gnf_converter_modules import (
    ClusteringIterationRecord,
    ClusteringHistory,
    BondValidator,
    ConnectivityAnalyzer,
    ClusteringHistorySaver,
    ReconstructionMetrics,
    GradientFieldComputer,
    GradientAscentOptimizer,
    ClusteringProcessor,
    SamplingProcessor,
)


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
                gradient_field_method: str = "softmax",  # 梯度场计算方法: "gaussian", "softmax", "logsumexp", "inverse_square", "sigmoid", "gaussian_mag", "distance", "gaussian_hole"
                temperature: float = 1.0,  # softmax温度参数，控制分布尖锐程度
                logsumexp_eps: float = 1e-8,  # logsumexp方法的数值稳定性参数
                inverse_square_strength: float = 1.0,  # 距离平方反比方法的强度参数
                gradient_clip_threshold: float = 0.3,  # 梯度模长截断阈值
                sig_sf: float = 0.1,  # softmax field的sigma参数
                sig_mag: float = 0.45,  # magnitude的sigma参数
                gaussian_hole_clip: float = 0.8,  # gaussian_hole方法中距离裁剪的上限值
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
                min_samples_decay_factor: float = 0.7,  # 每轮min_samples衰减因子
                min_min_samples: int = 2,  # min_samples下限
                max_clustering_iterations: int = 10,  # 最大迭代轮数
                bond_length_tolerance: float = 0.4,  # 键长合理性检查的上限容差（单位：Å），在标准键长基础上增加的容差
                bond_length_lower_tolerance: float = 0.2,  # 键长合理性检查的下限容差（单位：Å），从标准键长中减去的容差
                enable_clustering_history: bool = False,  # 是否记录聚类历史
                debug_bond_validation: bool = False,  # 是否输出键长检查的调试信息
                gradient_batch_size: Optional[int] = None,  # 梯度计算时的批次大小，None表示一次性处理所有点
                n_initial_atoms_no_bond_check: int = 3,  # 前N个原子不受键长限制（无论上限还是下限）
                enable_bond_validation: bool = True,  # 是否启用键长检查，默认True
                sampling_range_min: float = -7.0,  # 撒点范围的最小值（单位：Å）
                sampling_range_max: float = 7.0):  # 撒点范围的最大值（单位：Å）
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
        self.gaussian_hole_clip = gaussian_hole_clip  # 保存gaussian_hole方法中距离裁剪的上限值
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
        self.min_samples_decay_factor = min_samples_decay_factor
        self.min_min_samples = min_min_samples
        self.max_clustering_iterations = max_clustering_iterations
        self.bond_length_tolerance = bond_length_tolerance
        self.bond_length_lower_tolerance = bond_length_lower_tolerance
        self.enable_clustering_history = enable_clustering_history
        self.debug_bond_validation = debug_bond_validation
        self.gradient_batch_size = gradient_batch_size
        self.n_initial_atoms_no_bond_check = n_initial_atoms_no_bond_check
        self.enable_bond_validation = enable_bond_validation
        self.sampling_range_min = sampling_range_min
        self.sampling_range_max = sampling_range_max
        
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
        
        # 初始化拆解后的模块
        self.bond_validator = BondValidator(
            bond_length_tolerance=bond_length_tolerance,
            bond_length_lower_tolerance=bond_length_lower_tolerance,
            debug=debug_bond_validation
        )
        self.connectivity_analyzer = ConnectivityAnalyzer(bond_validator=self.bond_validator)
        self.history_saver = ClusteringHistorySaver(n_atom_types=n_atom_types)
        self.metrics_computer = ReconstructionMetrics()
        
        # 初始化梯度场计算模块
        self.gradient_field_computer = GradientFieldComputer(
            sigma_params=self.sigma_params,
            sigma=sigma,
            gradient_field_method=gradient_field_method,
            temperature=temperature,
            logsumexp_eps=logsumexp_eps,
            inverse_square_strength=inverse_square_strength,
            gradient_clip_threshold=gradient_clip_threshold,
            sig_sf=sig_sf,
            sig_mag=sig_mag,
            gaussian_hole_clip=gaussian_hole_clip,
        )
        
        # 初始化梯度上升模块
        self.gradient_ascent_optimizer = GradientAscentOptimizer(
            n_iter=n_iter,
            step_size=step_size,
            sigma_params=self.sigma_params,
            sigma=sigma,
            enable_early_stopping=enable_early_stopping,
            convergence_threshold=convergence_threshold,
            min_iterations=min_iterations,
            gradient_batch_size=gradient_batch_size,
        )
        
        # 初始化聚类模块
        self.clustering_processor = ClusteringProcessor(
            bond_validator=self.bond_validator,
            eps=eps,
            min_samples=min_samples,
            enable_autoregressive_clustering=enable_autoregressive_clustering,
            initial_min_samples=initial_min_samples,
            min_samples_decay_factor=min_samples_decay_factor,
            min_min_samples=min_min_samples,
            max_clustering_iterations=max_clustering_iterations,
            debug_bond_validation=debug_bond_validation,
            n_initial_atoms_no_bond_check=n_initial_atoms_no_bond_check,
            enable_bond_validation=enable_bond_validation,
        )
        
        # 初始化采样模块
        self.sampling_processor = SamplingProcessor(
            gradient_field_computer=self.gradient_field_computer,
            gradient_ascent_optimizer=self.gradient_ascent_optimizer,
            clustering_processor=self.clustering_processor,
            n_query_points=n_query_points,
            n_query_points_per_type=self.n_query_points_per_type,
            gradient_sampling_candidate_multiplier=gradient_sampling_candidate_multiplier,
            field_variance_k_neighbors=field_variance_k_neighbors,
            field_variance_weight=field_variance_weight,
            eps=eps,
            min_min_samples=min_min_samples,
            enable_clustering_history=enable_clustering_history,
            sampling_range_min=sampling_range_min,
            sampling_range_max=sampling_range_max,
        )
    
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
        
        优化版本：支持高效的批量计算，减少Python循环开销。
        
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
        
        # 优化：使用列表推导和向量化操作减少循环开销
        # 对于batch_size > 1的情况，批量处理可以进一步提升性能
        if batch_size == 1:
            # 单样本情况：直接处理，避免不必要的循环开销
            mask = (atom_types[0] != PADDING_INDEX)  # [n_atoms]
            valid_coords = coords[0][mask]  # [n_valid_atoms, 3]
            valid_types = atom_types[0][mask].long()  # [n_valid_atoms]
            
            if valid_coords.size(0) > 0:
                vector_field[0] = self.gradient_field_computer.compute_gradient_field_matrix(
                    valid_coords, valid_types, query_points[0], n_atom_types
                )
        else:
            # 批量处理：虽然每个样本的原子数可能不同，但我们可以优化循环
            # 使用列表存储有效数据，然后批量处理（如果可能）
            valid_data = []
            valid_indices = []
            
            for b in range(batch_size):
                mask = (atom_types[b] != PADDING_INDEX)  # [n_atoms]
                valid_coords = coords[b][mask]  # [n_valid_atoms, 3]
                valid_types = atom_types[b][mask].long()  # [n_valid_atoms]
                
                if valid_coords.size(0) > 0:
                    valid_data.append((valid_coords, valid_types, query_points[b]))
                    valid_indices.append(b)
            
            # 批量计算：逐个处理（因为原子数量不同，难以完全向量化）
            # 但我们可以使用更高效的方式
            for idx, (valid_coords, valid_types, q_points) in zip(valid_indices, valid_data):
                vector_field[idx] = self.gradient_field_computer.compute_gradient_field_matrix(
                    valid_coords, valid_types, q_points, n_atom_types
                )
        
        return vector_field


    def gnf2mol(self, decoder: nn.Module, codes: torch.Tensor,
                _atom_types: Optional[torch.Tensor] = None,
                save_interval: Optional[int] = None,
                visualization_callback: Optional[callable] = None,
                predictor: Optional[nn.Module] = None,
                save_clustering_history: bool = False,  # 是否保存聚类历史
                clustering_history_dir: Optional[str] = None,  # 保存目录
                save_gradient_ascent_sdf: bool = False,  # 是否保存梯度上升SDF
                gradient_ascent_sdf_dir: Optional[str] = None,  # 保存目录
                gradient_ascent_sdf_interval: int = 100,  # 保存间隔
                sample_id: Optional[Union[int, str]] = None,  # 样本标识符，用于文件命名
                enable_timing: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
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
            enable_timing: 如果为 True，将在控制台打印每个 batch 的主要阶段耗时以及总耗时（用于简单性能分析）
        Returns:
            (final_coords, final_types)
        """
        t_global_start = time.perf_counter() if enable_timing else None

        device = codes.device
        batch_size = codes.size(0)
        n_atom_types = self.n_atom_types

        all_coords = []
        all_types = []
        
        # 对于批次化的codes，我们可以尝试并行处理
        # 但由于聚类、连通分支选择等步骤的复杂性，我们仍然逐个处理每个分子
        # 不过，process_atom_types_matrix内部已经支持批次化的decoder调用，可以实现部分并行化
        
        # 对每个batch分别处理
        for b in range(batch_size):
            t_batch_start = time.perf_counter() if enable_timing else None

            # 确定文件标识符：优先使用sample_id，否则使用batch索引
            file_identifier = sample_id if sample_id is not None else b
            
            # 检查索引边界
            if b >= codes.size(0):
                break
            
            # 检查当前batch的codes
            # 注意：虽然我们逐个处理每个分子，但process_atom_types_matrix内部已经支持批次化的decoder调用
            current_codes = codes[b:b+1]
            
            if current_codes.numel() == 0:
                continue
            
            if torch.isnan(current_codes).any() or torch.isinf(current_codes).any():
                continue
            
            # 如果提供了predictor，预测元素存在性
            t_predictor_start = time.perf_counter() if enable_timing else None
            element_existence = None
            if predictor is not None:
                with torch.no_grad():
                    # predictor现在输出logits，需要应用sigmoid得到概率
                    logits = predictor(current_codes)  # [1, n_atom_types]
                    element_existence = torch.sigmoid(logits)  # [1, n_atom_types]
            t_predictor_end = time.perf_counter() if enable_timing else None
            
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
                
                def create_gradient_ascent_sdf_callback(file_id, n_atom_types, interval):
                    def callback(iter_idx, current_points, atom_types):
                        # 只在指定间隔时保存
                        if iter_idx % interval == 0 or iter_idx == self.n_iter - 1:
                            try:
                                from funcmol.sample_fm import xyz_to_sdf
                                elements = [ELEMENTS_HASH_INV.get(i, f"Type{i}") for i in range(n_atom_types)]
                                
                                # 收集所有类型的query_points
                                all_points = []
                                all_types = []
                                type_counts = {}
                                
                                for t in range(n_atom_types):
                                    type_mask = (atom_types == t)
                                    if not type_mask.any():
                                        continue
                                    
                                    type_points = current_points[type_mask].cpu().numpy()
                                    all_points.append(type_points)
                                    all_types.append(np.full(len(type_points), t))
                                    
                                    atom_symbol = elements[t] if t < len(elements) else f"Type{t}"
                                    type_counts[atom_symbol] = len(type_points)
                                
                                # 合并所有类型的点
                                if len(all_points) > 0:
                                    combined_points = np.vstack(all_points)
                                    combined_types = np.concatenate(all_types)
                                    
                                    # 创建SDF字符串，包含所有类型的query_points
                                    sdf_str = xyz_to_sdf(combined_points, combined_types, elements)
                                    
                                    # SDF标题包含所有类型的信息
                                    lines = sdf_str.split('\n')
                                    type_info = ", ".join([f"{symbol}:{count}" for symbol, count in type_counts.items()])
                                    title = f"Gradient Ascent Iter {iter_idx}, Total {len(combined_points)} query_points ({type_info})"
                                    # SDF格式要求标题行必须是80字符，不足用空格填充
                                    title = title[:80].ljust(80)
                                    lines[0] = title
                                    
                                    # 保存SDF文件（所有类型合并到一个文件），使用有意义的标识符
                                    if isinstance(file_id, int):
                                        sdf_file = sdf_dir / f"sample_{file_id:04d}_iter_{iter_idx:04d}.sdf"
                                    else:
                                        sdf_file = sdf_dir / f"{file_id}_iter_{iter_idx:04d}.sdf"
                                    with open(sdf_file, 'w') as f:
                                        f.write('\n'.join(lines))
                            except Exception as e:
                                print(f"Warning: Failed to save gradient ascent SDF: {e}")
                    
                    return callback
                
                gradient_ascent_callback = create_gradient_ascent_sdf_callback(file_identifier, n_atom_types, gradient_ascent_sdf_interval)
            
            # 矩阵化处理所有原子类型（传入element_existence以过滤不存在的元素）
            t_process_start = time.perf_counter() if enable_timing else None
            coords_list, types_list, histories = self.sampling_processor.process_atom_types_matrix(
                current_codes, n_atom_types, device=device, decoder=decoder,
                iteration_callback=iteration_callback,
                element_existence=element_existence,
                gradient_ascent_callback=gradient_ascent_callback,
                enable_timing=enable_timing
            )
            t_process_end = time.perf_counter() if enable_timing else None
            
            # 保存聚类历史
            t_history_start = time.perf_counter() if enable_timing else None
            if save_clustering_history and clustering_history_dir and histories:
                self.history_saver.save_clustering_history(
                    histories, 
                    clustering_history_dir,
                    batch_idx=file_identifier
                )
            t_history_end = time.perf_counter() if enable_timing else None
            
            # 合并所有类型
            t_merge_start = time.perf_counter() if enable_timing else None
            # 初始化连通分支相关的时间变量，避免在else分支中未定义
            t_connected_start = None
            t_connected_end = None
            if coords_list:
                merged_coords = torch.cat(coords_list, dim=0)
                merged_types = torch.cat(types_list, dim=0)
                
                # 选择最大连通分支（仅在启用键长检查时执行，因为连通性判断基于键长阈值）
                # 如果禁用了键长检查，保留所有原子，不进行连通分支过滤
                if self.enable_bond_validation:
                    t_connected_start = time.perf_counter() if enable_timing else None
                    filtered_coords, filtered_types = self.connectivity_analyzer.select_largest_connected_component(
                        merged_coords, merged_types
                    )
                    t_connected_end = time.perf_counter() if enable_timing else None
                else:
                    # 禁用键长检查时，只过滤掉填充原子（-1），保留所有有效原子
                    valid_mask = (merged_types != PADDING_INDEX) & (merged_types != -1)
                    if valid_mask.any():
                        filtered_coords = merged_coords[valid_mask]
                        filtered_types = merged_types[valid_mask]
                    else:
                        filtered_coords = torch.empty(0, 3, device=device)
                        filtered_types = torch.empty(0, dtype=torch.long, device=device)
                    t_connected_start = None
                    t_connected_end = None
                
                all_coords.append(filtered_coords)
                all_types.append(filtered_types)
            else:
                all_coords.append(torch.empty(0, 3, device=device))
                all_types.append(torch.empty(0, dtype=torch.long, device=device))
            t_merge_end = time.perf_counter() if enable_timing else None

            if enable_timing:
                t_batch_end = time.perf_counter()
                predictor_time = (t_predictor_end - t_predictor_start) if (t_predictor_start is not None and t_predictor_end is not None) else 0.0
                process_time = (t_process_end - t_process_start) if (t_process_start is not None and t_process_end is not None) else 0.0
                history_time = (t_history_end - t_history_start) if (t_history_start is not None and t_history_end is not None) else 0.0
                merge_time = (t_merge_end - t_merge_start) if (t_merge_start is not None and t_merge_end is not None) else 0.0
                connected_time = (t_connected_end - t_connected_start) if (t_connected_start is not None and t_connected_end is not None) else 0.0
                total_batch_time = t_batch_end - t_batch_start if t_batch_start is not None else 0.0
                print(
                    f"[GNFConverter.gnf2mol] Batch {file_identifier} timing: "
                    f"predictor={predictor_time:.3f}s, "
                    f"process={process_time:.3f}s, "
                    f"merge={merge_time:.3f}s, "
                    f"connected_component={connected_time:.3f}s, "
                    f"history_io={history_time:.3f}s, "
                    f"total={total_batch_time:.3f}s"
                )
        
        # pad到batch最大长度
        max_atoms = max([c.size(0) for c in all_coords]) if all_coords else 0
        final_coords = torch.stack([F.pad(c, (0,0,0,max_atoms-c.size(0))) if c.size(0)<max_atoms else c for c in all_coords], dim=0)
        final_types = torch.stack([F.pad(t, (0,max_atoms-t.size(0)), value=-1) if t.size(0)<max_atoms else t for t in all_types], dim=0)

        if enable_timing and t_global_start is not None:
            t_global_end = time.perf_counter()
            print(f"[GNFConverter.gnf2mol] Total time for all batches: {t_global_end - t_global_start:.3f}s")

        return final_coords, final_types # [batch, n_atoms, 3]



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
        return self.metrics_computer.compute_reconstruction_metrics(
            recon_coords, recon_types, gt_coords, gt_types
        )
    