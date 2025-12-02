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
        
        # 改进的自回归聚类逻辑：预先找出所有满足min_min_samples的簇，并分配iteration tag
        # 1. 首先用min_min_samples找出所有可能的簇
        initial_min_samples = self.initial_min_samples if self.initial_min_samples is not None else self.min_samples
        initial_clustering = DBSCAN(eps=self.eps, min_samples=self.min_min_samples).fit(points)
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
        
        # 3. 根据samples数量计算每个簇应该在第几个iteration被聚类出来
        # 计算min_samples的衰减序列，确定每个samples数量对应的iteration
        current_min_samples = initial_min_samples
        iteration_thresholds = []  # [(iteration, min_samples), ...]
        iteration = 0
        while iteration < self.max_clustering_iterations:
            iteration_thresholds.append((iteration, current_min_samples))
            if current_min_samples <= self.min_min_samples:
                break
            current_min_samples = max(int(current_min_samples * self.min_samples_decay_factor), self.min_min_samples)
            iteration += 1
        
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
                # 如果当前iteration没有簇，检查是否还有后续iteration
                if current_iteration >= max(clusters_by_iteration.keys()) if clusters_by_iteration else 0:
                    break
                current_iteration += 1
                continue
            
            current_min_samples = iteration_thresholds[current_iteration][1] if current_iteration < len(iteration_thresholds) else self.min_min_samples
            
            new_merged_this_iteration = []
            new_atoms_coords = []
            new_atoms_types = []
            n_clusters_found = len(clusters_to_process)
            n_noise_points = 0  # 在预先规划模式下，噪声点已经在初始聚类时被过滤
            
            # 处理当前iteration的所有簇
            for center, n_samples, cluster_points, original_iteration_tag in clusters_to_process:
                # 键长检查（如果有参考点）
                bond_validation_passed = True
                if current_reference_points is not None and len(current_reference_points) > 0:
                    # 添加调试信息：显示当前有多少参考点
                    if self.debug_bond_validation:
                        atom_symbol = ELEMENTS_HASH_INV.get(atom_type, f"Type{atom_type}") if atom_type is not None else "Unknown"
                        print(f"[键长检查] 原子类型: {atom_symbol}, 簇中心: {center}, 当前有 {len(current_reference_points)} 个参考点")
                        # 显示参考点的类型分布
                        from collections import Counter
                        type_counts = Counter([ELEMENTS_HASH_INV.get(t, f"Type{t}") for t in current_reference_types])
                        print(f"  参考点类型分布: {dict(type_counts)}")
                    
                    bond_validation_passed = self.bond_validator.check_bond_length_validity(
                        center, atom_type if atom_type is not None else -1,
                        current_reference_points, current_reference_types,
                        debug=self.debug_bond_validation
                    )
                
                if bond_validation_passed:
                    # 通过检查，加入结果
                    new_merged_this_iteration.append(center)
                    new_atoms_coords.append(center)
                    if atom_type is not None:
                        new_atoms_types.append(atom_type)
                    all_merged_points.append(center)
                    
                    # 只有通过键长检查的簇才添加到参考点（用于后续iteration的键长检查）
                    # pending 的簇不应该被作为参考点，因为它们本身还没有通过键长检查
                    if current_reference_points is None:
                        current_reference_points = np.array([center])
                        current_reference_types = np.array([atom_type if atom_type is not None else -1])
                    else:
                        # 检查是否已经存在（避免重复添加）
                        distances_to_existing = np.sqrt(((current_reference_points - center[None, :]) ** 2).sum(axis=1))
                        if len(distances_to_existing) == 0 or distances_to_existing.min() > 1e-6:  # 容差：1e-6 Å
                            current_reference_points = np.vstack([current_reference_points, center])
                            current_reference_types = np.append(current_reference_types, atom_type if atom_type is not None else -1)
                            if self.debug_bond_validation:
                                atom_symbol = ELEMENTS_HASH_INV.get(atom_type, f"Type{atom_type}") if atom_type is not None else "Unknown"
                                print(f"[添加到参考点] 原子类型: {atom_symbol}, 簇中心: {center}")
                else:
                    # 键长检查失败，加入pending列表，在下一轮重新检查
                    # 注意：pending 的簇不会被添加到参考点中，因为它们本身还没有通过键长检查
                    pending_clusters.append((center, n_samples, cluster_points, original_iteration_tag))
                    if self.debug_bond_validation:
                        atom_symbol = ELEMENTS_HASH_INV.get(atom_type, f"Type{atom_type}") if atom_type is not None else "Unknown"
                        print(f"[簇被拒绝，待下一轮] 原子类型: {atom_symbol}, 簇大小: {n_samples}, 簇中心: {center}, 原始iteration_tag: {original_iteration_tag}, 当前iteration: {current_iteration}")
            
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
            
            # 检查终止条件：如果某一轮没有新的原子被聚类，且没有pending的簇，就停止
            if len(new_merged_this_iteration) == 0 and len(pending_clusters) == 0:
                if self.debug_bond_validation:
                    print(f"[自回归聚类早停] 迭代轮数 {current_iteration}: 没有新原子被聚类，且没有pending簇，停止循环")
                break
            
            current_iteration += 1
        
        if history is not None:
            history.total_atoms = len(all_merged_points)
        
        result = np.array(all_merged_points) if len(all_merged_points) > 0 else np.empty((0, 3))
        return result, history

