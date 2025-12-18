"""
聚类模块：DBSCAN聚类和点合并
"""
import numpy as np
from typing import Optional, Tuple
from sklearn.cluster import DBSCAN
from funcmol.utils.gnf_converter_modules.dataclasses import ClusteringIterationRecord, ClusteringHistory
from funcmol.utils.constants import ELEMENTS_HASH_INV


class ClusteringProcessor:
    """聚类处理器"""
    
    def __init__(
        self,
        bond_validator,
        eps: float,
        min_samples: int,
        enable_autoregressive_clustering: bool = False,
        initial_min_samples: Optional[int] = None,
        min_samples_decay_factor: float = 0.7,
        min_min_samples: int = 2,
        max_clustering_iterations: int = 10,
        debug_bond_validation: bool = False,
        n_initial_atoms_no_bond_check: int = 3,
    ):
        """
        初始化聚类处理器
        
        Args:
            bond_validator: BondValidator实例
            eps: DBSCAN的邻域半径参数
            min_samples: DBSCAN的最小样本数参数
            enable_autoregressive_clustering: 是否启用自回归聚类
            initial_min_samples: 初始min_samples
            min_samples_decay_factor: 每轮min_samples衰减因子
            min_min_samples: min_samples下限
            max_clustering_iterations: 最大迭代轮数
            debug_bond_validation: 是否输出键长检查的调试信息
            n_initial_atoms_no_bond_check: 前N个原子不受键长限制（无论上限还是下限）
        """
        self.bond_validator = bond_validator
        self.eps = eps
        self.min_samples = min_samples
        self.enable_autoregressive_clustering = enable_autoregressive_clustering
        self.initial_min_samples = initial_min_samples
        self.min_samples_decay_factor = min_samples_decay_factor
        self.min_min_samples = min_min_samples
        self.max_clustering_iterations = max_clustering_iterations
        self.debug_bond_validation = debug_bond_validation
        self.n_initial_atoms_no_bond_check = n_initial_atoms_no_bond_check
    
    def _prepare_clusters(self, points: np.ndarray) -> Tuple[list, list]:
        """
        预先找出所有簇并分配iteration tag
        
        Returns:
            clusters_with_tags: [(center, iteration_tag, n_samples, cluster_points), ...]
            iteration_thresholds: [(iteration, min_samples), ...]
        """
        if len(points) == 0:
            # 计算iteration_thresholds（即使没有簇也需要）
            if self.enable_autoregressive_clustering:
                initial_min_samples = self.initial_min_samples if self.initial_min_samples is not None else self.min_samples
                current_min_samples = initial_min_samples
                iteration_thresholds = []
                iteration = 0
                while iteration < self.max_clustering_iterations:
                    iteration_thresholds.append((iteration, current_min_samples))
                    if current_min_samples <= self.min_min_samples:
                        break
                    current_min_samples = max(int(current_min_samples * self.min_samples_decay_factor), self.min_min_samples)
                    iteration += 1
            else:
                # 非自回归模式：只使用一次迭代
                iteration_thresholds = [(0, self.min_samples)]
            return [], iteration_thresholds
        
        # 1. 根据是否启用自回归聚类选择初始聚类参数
        if self.enable_autoregressive_clustering:
            # 自回归模式：使用min_min_samples找出所有可能的簇
            initial_min_samples = self.initial_min_samples if self.initial_min_samples is not None else self.min_samples
            clustering_min_samples = self.min_min_samples
        else:
            # 非自回归模式：直接使用min_samples进行单次聚类
            initial_min_samples = self.min_samples
            clustering_min_samples = self.min_samples
        
        initial_clustering = DBSCAN(eps=self.eps, min_samples=clustering_min_samples).fit(points)
        initial_labels = initial_clustering.labels_
        
        # 2. 收集所有簇及其samples数量
        all_clusters = []  # [(center, n_samples, cluster_points), ...]
        for label in set(initial_labels):
            if label == -1:
                continue  # 跳过噪声点
            cluster_points = points[initial_labels == label]
            center = np.mean(cluster_points, axis=0)
            n_samples = len(cluster_points)
            all_clusters.append((center, n_samples, cluster_points))
        
        if len(all_clusters) == 0:
            # 计算iteration_thresholds（即使没有簇也需要）
            if self.enable_autoregressive_clustering:
                current_min_samples = initial_min_samples
                iteration_thresholds = []
                iteration = 0
                while iteration < self.max_clustering_iterations:
                    iteration_thresholds.append((iteration, current_min_samples))
                    if current_min_samples <= self.min_min_samples:
                        break
                    current_min_samples = max(int(current_min_samples * self.min_samples_decay_factor), self.min_min_samples)
                    iteration += 1
            else:
                # 非自回归模式：只使用一次迭代
                iteration_thresholds = [(0, self.min_samples)]
            return [], iteration_thresholds
        
        # 2.5. 合并距离太近的簇（距离 < eps），避免重复原子
        merged_clusters = []
        used_indices = set()
        
        for i, (center_i, n_samples_i, cluster_points_i) in enumerate(all_clusters):
            if i in used_indices:
                continue
            
            merged_center = center_i.copy()
            merged_points = cluster_points_i.copy()
            merged_n_samples = n_samples_i
            
            for j, (center_j, _, cluster_points_j) in enumerate(all_clusters):
                if j <= i or j in used_indices:
                    continue
                
                distance = np.sqrt(((center_i - center_j) ** 2).sum())
                if distance < self.eps:
                    merged_points = np.vstack([merged_points, cluster_points_j])
                    merged_center = np.mean(merged_points, axis=0)
                    merged_n_samples = len(merged_points)
                    used_indices.add(j)
            
            used_indices.add(i)
            merged_clusters.append((merged_center, merged_n_samples, merged_points))
        
        all_clusters = merged_clusters
        
        # 3. 计算min_samples的衰减序列
        if self.enable_autoregressive_clustering:
            # 自回归模式：计算衰减序列
            current_min_samples = initial_min_samples
            iteration_thresholds = []  # [(iteration, min_samples), ...]
            iteration = 0
            while iteration < self.max_clustering_iterations:
                iteration_thresholds.append((iteration, current_min_samples))
                if current_min_samples <= self.min_min_samples:
                    break
                current_min_samples = max(int(current_min_samples * self.min_samples_decay_factor), self.min_min_samples)
                iteration += 1
        else:
            # 非自回归模式：只使用一次迭代，使用min_samples
            iteration_thresholds = [(0, self.min_samples)]
        
        # 4. 为每个簇分配iteration tag
        clusters_with_tags = []  # [(center, iteration_tag, n_samples, cluster_points), ...]
        for center, n_samples, cluster_points in all_clusters:
            iteration_tag = 0
            for iter_idx, min_samples_threshold in iteration_thresholds:
                if n_samples >= min_samples_threshold:
                    iteration_tag = iter_idx
                else:
                    break
            clusters_with_tags.append((center, iteration_tag, n_samples, cluster_points))
        
        clusters_with_tags.sort(key=lambda x: x[1])
        return clusters_with_tags, iteration_thresholds
    
    def merge_points_single_iteration(
        self,
        clusters_with_tags: list,
        iteration_thresholds: list,
        current_iteration: int,
        atom_type: Optional[int] = None,
        reference_points: Optional[np.ndarray] = None,
        reference_types: Optional[np.ndarray] = None,
        pending_clusters: Optional[list] = None,
        current_type_atom_count: int = 0,
    ) -> Tuple[np.ndarray, list, dict, np.ndarray, np.ndarray]:
        """
        处理单个iteration的聚类
        
        Args:
            clusters_with_tags: 所有簇及其iteration tag
            iteration_thresholds: iteration阈值列表
            current_iteration: 当前处理的iteration
            atom_type: 原子类型
            reference_points: 参考点坐标
            reference_types: 参考点类型
            pending_clusters: 待处理的簇列表 [(center, n_samples, cluster_points, original_iteration_tag), ...]
            current_type_atom_count: 当前原子类型已通过的原子数量（用于判断是否需要键长检查）
            
        Returns:
            new_atoms: 本轮通过检查的原子坐标
            new_pending_clusters: 新的pending簇列表
            stats: 统计信息字典
            updated_reference_points: 更新后的参考点坐标
            updated_reference_types: 更新后的参考点类型
        """
        if pending_clusters is None:
            pending_clusters = []
        
        # 获取当前iteration的min_samples
        current_min_samples = iteration_thresholds[current_iteration][1] if current_iteration < len(iteration_thresholds) else self.min_min_samples
        
        # 收集当前iteration需要处理的簇
        clusters_to_process = []
        
        # 添加原本属于当前iteration的簇
        for center, iteration_tag, n_samples, cluster_points in clusters_with_tags:
            if iteration_tag == current_iteration:
                clusters_to_process.append((center, n_samples, cluster_points, current_iteration))
        
        # 添加pending的簇
        clusters_to_process.extend(pending_clusters)
        new_pending_clusters = []
        
        # 从传入的参考点开始（复制以避免修改原数组）
        current_reference_points = reference_points.copy() if reference_points is not None and len(reference_points) > 0 else None
        current_reference_types = reference_types.copy() if reference_types is not None and len(reference_types) > 0 else None
        
        new_atoms = []
        n_passed = 0
        n_rejected = 0
        rejection_reasons = {"下限": 0, "上限": 0, "重复": 0}
        
        # 处理所有簇
        for center, n_samples, cluster_points, original_iteration_tag in clusters_to_process:
            bond_validation_passed = True
            rejection_reason = None
            
            # 如果当前原子类型已通过的原子数量小于阈值，前N个原子不受键长限制
            if current_type_atom_count < self.n_initial_atoms_no_bond_check:
                bond_validation_passed = True
            elif current_reference_points is not None and len(current_reference_points) > 0:
                if current_iteration == 0:
                    bond_validation_passed, rejection_reason = self.bond_validator.check_bond_length_lower_only(
                        center, atom_type if atom_type is not None else -1,
                        current_reference_points, current_reference_types,
                        debug=False
                    )
                else:
                    bond_validation_passed, rejection_reason = self.bond_validator.check_bond_length_validity(
                        center, atom_type if atom_type is not None else -1,
                        current_reference_points, current_reference_types,
                        debug=False
                    )
            else:
                bond_validation_passed = True
            
            if bond_validation_passed:
                new_atoms.append(center)
                n_passed += 1
                
                # 更新参考点
                if current_reference_points is None:
                    current_reference_points = np.array([center])
                    current_reference_types = np.array([atom_type if atom_type is not None else -1])
                else:
                    distances_to_existing = np.sqrt(((current_reference_points - center[None, :]) ** 2).sum(axis=1))
                    if len(distances_to_existing) == 0 or distances_to_existing.min() > 1e-6:
                        current_reference_points = np.vstack([current_reference_points, center])
                        current_reference_types = np.append(current_reference_types, atom_type if atom_type is not None else -1)
            else:
                new_pending_clusters.append((center, n_samples, cluster_points, original_iteration_tag))
                n_rejected += 1
                if rejection_reason:
                    rejection_reasons[rejection_reason] = rejection_reasons.get(rejection_reason, 0) + 1
        
        stats = {
            'n_clusters_found': len(clusters_to_process),
            'n_passed': n_passed,
            'n_rejected': n_rejected,
            'rejection_reasons': rejection_reasons,
            'current_min_samples': current_min_samples
        }
        
        updated_reference_points = current_reference_points if current_reference_points is not None else np.empty((0, 3))
        updated_reference_types = current_reference_types if current_reference_types is not None else np.empty((0,), dtype=np.int64)
        
        return (np.array(new_atoms) if len(new_atoms) > 0 else np.empty((0, 3)), 
                new_pending_clusters, stats, updated_reference_points, updated_reference_types)
    
    def merge_points(
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

        # 使用自回归聚类逻辑：预先找出所有满足min_min_samples的簇，并分配iteration tag
        # 1. 根据是否启用自回归聚类选择初始聚类参数
        if self.enable_autoregressive_clustering:
            # 自回归模式：使用min_min_samples找出所有可能的簇
            initial_min_samples = self.initial_min_samples if self.initial_min_samples is not None else self.min_samples
            clustering_min_samples = self.min_min_samples
        else:
            # 非自回归模式：直接使用min_samples进行单次聚类
            initial_min_samples = self.min_samples
            clustering_min_samples = self.min_samples
        
        initial_clustering = DBSCAN(eps=self.eps, min_samples=clustering_min_samples).fit(points)
        initial_labels = initial_clustering.labels_
        
        # 2. 收集所有簇及其samples数量
        all_clusters = []  # [(center, n_samples, cluster_points), ...]
        for label in set(initial_labels):
            if label == -1:
                continue  # 跳过噪声点
            cluster_points = points[initial_labels == label]
            center = np.mean(cluster_points, axis=0)
            n_samples = len(cluster_points)
            all_clusters.append((center, n_samples, cluster_points))
        
        if len(all_clusters) == 0:
            # 没有找到任何簇
            return np.empty((0, 3)), None
        
        # 2.5. 合并距离太近的簇（距离 < eps），避免重复原子
        # 如果两个簇的中心距离小于eps，它们应该代表同一个原子，需要合并
        n_clusters_before_merge = len(all_clusters)
        merged_clusters = []
        used_indices = set()
        
        for i, (center_i, n_samples_i, cluster_points_i) in enumerate(all_clusters):
            if i in used_indices:
                continue
            
            # 找到所有与当前簇距离 < eps 的簇
            merged_center = center_i.copy()
            merged_points = cluster_points_i.copy()
            merged_n_samples = n_samples_i
            merged_indices = [i]
            
            for j, (center_j, _, cluster_points_j) in enumerate(all_clusters):
                if j <= i or j in used_indices:
                    continue
                
                distance = np.sqrt(((center_i - center_j) ** 2).sum())
                if distance < self.eps:
                    # 合并簇：合并所有点，重新计算中心
                    merged_points = np.vstack([merged_points, cluster_points_j])
                    merged_center = np.mean(merged_points, axis=0)
                    merged_n_samples = len(merged_points)
                    merged_indices.append(j)
                    used_indices.add(j)
            
            used_indices.add(i)
            merged_clusters.append((merged_center, merged_n_samples, merged_points))
        
        all_clusters = merged_clusters
        n_clusters_after_merge = len(all_clusters)
        
        if self.debug_bond_validation and n_clusters_before_merge != n_clusters_after_merge:
            atom_symbol = ELEMENTS_HASH_INV.get(atom_type, f"Type{atom_type}") if atom_type is not None else "Unknown"
            print(f"[簇合并] 原子类型: {atom_symbol}, 合并前: {n_clusters_before_merge} 个簇, 合并后: {n_clusters_after_merge} 个簇 (合并了 {n_clusters_before_merge - n_clusters_after_merge} 个距离<eps的簇)")
        
        if len(all_clusters) == 0:
            # 合并后没有簇了（不应该发生，但保险起见）
            return np.empty((0, 3)), None
        
        # 3. 根据samples数量计算每个簇应该在第几个iteration被聚类出来
        # 计算min_samples的衰减序列，确定每个samples数量对应的iteration
        if self.enable_autoregressive_clustering:
            # 自回归模式：计算衰减序列
            current_min_samples = initial_min_samples
            iteration_thresholds = []  # [(iteration, min_samples), ...]
            iteration = 0
            while iteration < self.max_clustering_iterations:
                iteration_thresholds.append((iteration, current_min_samples))
                if current_min_samples <= self.min_min_samples:
                    break
                current_min_samples = max(int(current_min_samples * self.min_samples_decay_factor), self.min_min_samples)
                iteration += 1
        else:
            # 非自回归模式：只使用一次迭代，使用min_samples
            iteration_thresholds = [(0, self.min_samples)]
        
        # 4. 为每个簇分配iteration tag
        # samples数量越少，需要的min_samples越小，对应的iteration越大
        clusters_with_tags = []  # [(center, iteration_tag, n_samples, cluster_points), ...]
        for center, n_samples, cluster_points in all_clusters:
            # 找到满足n_samples >= min_samples的最大iteration（即最晚的iteration）
            iteration_tag = 0
            for iter_idx, min_samples_threshold in iteration_thresholds:
                if n_samples >= min_samples_threshold:
                    iteration_tag = iter_idx
                else:
                    break
            clusters_with_tags.append((center, iteration_tag, n_samples, cluster_points))
        
        # 5. 按iteration tag排序（iteration越小越先处理）
        clusters_with_tags.sort(key=lambda x: x[1])
        
        # 6. 按iteration顺序逐步聚类，同时考虑键长检查
        all_merged_points = []
        current_reference_points = reference_points.copy() if reference_points is not None and len(reference_points) > 0 else None
        current_reference_types = reference_types.copy() if reference_types is not None and len(reference_types) > 0 else None
        
        history = None
        if record_history:
            history = ClusteringHistory(atom_type=atom_type if atom_type is not None else -1, iterations=[], total_atoms=0)
        
        current_iteration = 0
        clusters_by_iteration = {}  # {iteration: [(center, n_samples, cluster_points), ...]}
        pending_clusters = []  # 待处理的簇列表，包含之前未通过键长检查的簇 [(center, n_samples, cluster_points, original_iteration_tag), ...]
        
        # 按iteration分组
        for center, iteration_tag, n_samples, cluster_points in clusters_with_tags:
            if iteration_tag not in clusters_by_iteration:
                clusters_by_iteration[iteration_tag] = []
            clusters_by_iteration[iteration_tag].append((center, n_samples, cluster_points))
        
        # 按iteration顺序处理
        while current_iteration < self.max_clustering_iterations:
            # 收集当前iteration需要处理的簇（包括原本属于当前iteration的簇和pending的簇）
            clusters_to_process = []
            
            # 添加原本属于当前iteration的簇
            if current_iteration in clusters_by_iteration:
                for center, n_samples, cluster_points in clusters_by_iteration[current_iteration]:
                    clusters_to_process.append((center, n_samples, cluster_points, current_iteration))
            
            # 添加pending的簇（之前未通过键长检查的簇）
            clusters_to_process.extend(pending_clusters)
            pending_clusters = []  # 清空pending列表，重新检查
            
            if len(clusters_to_process) == 0:
                # 如果当前iteration没有簇，仍然继续迭代（不早停，确保完成max_clustering_iterations次迭代）
                # 记录空迭代的历史
                if record_history and history is not None:
                    current_min_samples = iteration_thresholds[current_iteration][1] if current_iteration < len(iteration_thresholds) else self.min_min_samples
                    record = ClusteringIterationRecord(
                        iteration=current_iteration,
                        eps=self.eps,
                        min_samples=current_min_samples,
                        new_atoms_coords=np.empty((0, 3)),
                        new_atoms_types=np.empty((0,), dtype=np.int64),
                        n_clusters_found=0,
                        n_atoms_clustered=0,
                        n_noise_points=0,
                        bond_validation_passed=True
                    )
                    history.iterations.append(record)
                current_iteration += 1
                continue
            
            current_min_samples = iteration_thresholds[current_iteration][1] if current_iteration < len(iteration_thresholds) else self.min_min_samples
            
            new_merged_this_iteration = []
            new_atoms_coords = []
            new_atoms_types = []
            n_clusters_found = len(clusters_to_process)
            n_noise_points = 0  # 在预先规划模式下，噪声点已经在初始聚类时被过滤
            
            # 统计信息
            n_passed = 0
            n_rejected = 0
            rejection_reasons = {"下限": 0, "上限": 0, "重复": 0}
            
            # 处理当前iteration的所有簇
            for center, n_samples, cluster_points, original_iteration_tag in clusters_to_process:
                # 键长检查
                bond_validation_passed = True
                rejection_reason = None
                if current_reference_points is not None and len(current_reference_points) > 0:
                    # 有参考点时，根据是否为第一轮决定检查方式
                    if current_iteration == 0:
                        # 第一轮：只检查下限（防止同一类型原子太近），不检查上限
                        bond_validation_passed, rejection_reason = self.bond_validator.check_bond_length_lower_only(
                            center, atom_type if atom_type is not None else -1,
                            current_reference_points, current_reference_types,
                            debug=False  # 不在每个簇检查时输出，统一在最后输出汇总
                        )
                    else:
                        # 后续轮次：检查下限和上限（确保有连接）
                        bond_validation_passed, rejection_reason = self.bond_validator.check_bond_length_validity(
                            center, atom_type if atom_type is not None else -1,
                            current_reference_points, current_reference_types,
                            debug=False  # 不在每个簇检查时输出，统一在最后输出汇总
                        )
                else:
                    # 第一轮，且还没有任何参考点（第一个簇），直接通过
                    bond_validation_passed = True
                
                if bond_validation_passed:
                    # 通过检查，加入结果
                    new_merged_this_iteration.append(center)
                    new_atoms_coords.append(center)
                    if atom_type is not None:
                        new_atoms_types.append(atom_type)
                    all_merged_points.append(center)
                    n_passed += 1
                    
                    # 只有通过键长检查的簇才添加到参考点（用于后续iteration的键长检查）
                    # pending 的簇不作为参考点，因为它们本身还没有通过键长检查
                    if current_reference_points is None:
                        current_reference_points = np.array([center])
                        current_reference_types = np.array([atom_type if atom_type is not None else -1])
                    else:
                        # 检查是否已经存在（避免重复添加）
                        distances_to_existing = np.sqrt(((current_reference_points - center[None, :]) ** 2).sum(axis=1))
                        if len(distances_to_existing) == 0 or distances_to_existing.min() > 1e-6:  # 容差：1e-6 Å
                            current_reference_points = np.vstack([current_reference_points, center])
                            current_reference_types = np.append(current_reference_types, atom_type if atom_type is not None else -1)
                else:
                    # 键长检查失败，加入pending列表，在下一轮重新检查
                    pending_clusters.append((center, n_samples, cluster_points, original_iteration_tag))
                    n_rejected += 1
                    if rejection_reason:
                        rejection_reasons[rejection_reason] = rejection_reasons.get(rejection_reason, 0) + 1
            
            # 输出iteration级别的汇总信息
            if self.debug_bond_validation:
                atom_symbol = ELEMENTS_HASH_INV.get(atom_type, f"Type{atom_type}") if atom_type is not None else "Unknown"
                print(f"\n{'='*60}")
                print(f"[Iteration {current_iteration}] 原子类型: {atom_symbol}, min_samples={current_min_samples}")
                print(f"  处理簇数: {n_clusters_found} (新簇: {n_clusters_found - len(pending_clusters)}, 待重试: {len(pending_clusters)})")
                print(f"  结果: ✓通过={n_passed}, ✗拒绝={n_rejected}")
                if n_rejected > 0:
                    reason_str = ", ".join([f"{k}={v}" for k, v in rejection_reasons.items() if v > 0])
                    print(f"  拒绝原因: {reason_str}")
                if current_reference_points is not None and len(current_reference_points) > 0:
                    from collections import Counter
                    type_counts = Counter([ELEMENTS_HASH_INV.get(t, f"Type{t}") for t in current_reference_types])
                    print(f"  当前参考点: {len(current_reference_points)} 个, 类型分布: {dict(type_counts)}")
                print(f"{'='*60}")
            
            # 记录历史
            if record_history and history is not None:
                record = ClusteringIterationRecord(
                    iteration=current_iteration,
                    eps=self.eps,
                    min_samples=current_min_samples,
                    new_atoms_coords=np.array(new_atoms_coords) if len(new_atoms_coords) > 0 else np.empty((0, 3)),
                    new_atoms_types=np.array(new_atoms_types) if len(new_atoms_types) > 0 else np.empty((0,), dtype=np.int64),
                    n_clusters_found=n_clusters_found,
                    n_atoms_clustered=len(new_merged_this_iteration),
                    n_noise_points=n_noise_points,
                    bond_validation_passed=True
                )
                history.iterations.append(record)
            
            # 移除早停机制：无论如何都完成max_clustering_iterations次迭代
            # 即使某一轮没有新的原子被聚类，且没有pending的簇，也继续迭代
            if self.debug_bond_validation:
                if len(new_merged_this_iteration) == 0 and len(pending_clusters) == 0:
                    print(f"[自回归聚类] 迭代轮数 {current_iteration}: 没有新原子被聚类，且没有pending簇，但继续迭代直到完成max_clustering_iterations")
            
            current_iteration += 1
        
        if history is not None:
            history.total_atoms = len(all_merged_points)
        
        result = np.array(all_merged_points) if len(all_merged_points) > 0 else np.empty((0, 3))
        return result, history

