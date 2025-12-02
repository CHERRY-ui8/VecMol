"""
数据类模块：定义GNF转换器中使用的数据结构
"""
from dataclasses import dataclass
from typing import List
import numpy as np


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

