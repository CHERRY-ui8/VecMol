"""
分子结构评估模块
包含距离分析、空间分布分析等功能
"""

import torch
import numpy as np
from typing import Tuple


def compute_min_distances(coords):
    """
    计算分子中所有原子对的最小距离
    
    Args:
        coords: [N, 3] 原子坐标
        
    Returns:
        tuple: (min_distances, all_distances)
            - min_distances: 每个原子到最近邻原子的距离 [N]
            - all_distances: 所有原子对的距离（上三角矩阵）
    """
    n_atoms = coords.shape[0]
    if n_atoms < 2:
        return torch.tensor([]), torch.tensor([])
    
    # 计算所有原子对的距离
    distances = torch.cdist(coords, coords, p=2)  # [N, N]
    
    # 将对角线设为无穷大（自己到自己的距离）
    distances.fill_diagonal_(float('inf'))
    
    # 找到每个原子到最近邻的距离
    min_distances, _ = torch.min(distances, dim=1)
    
    # 获取上三角矩阵（避免重复计算）
    triu_mask = torch.triu(torch.ones(n_atoms, n_atoms, dtype=torch.bool), diagonal=1)
    all_distances = distances[triu_mask]
    
    return min_distances, all_distances


def analyze_spatial_distribution(positions):
    """
    分析分子的空间分布
    
    Args:
        positions: [N, 3] 原子坐标
    
    Returns:
        dict: 包含空间分布统计信息的字典
    """
    coord_min = positions.min(dim=0)[0]
    coord_max = positions.max(dim=0)[0]
    coord_range = coord_max - coord_min  # [3]
    coord_center = positions.mean(dim=0)  # [3]
    coord_span = torch.cdist(positions, positions, p=2).max().item()  # 最大原子对距离
    
    return {
        'coord_range': coord_range.cpu().numpy(),
        'coord_center': coord_center.cpu().numpy(),
        'coord_span': coord_span
    }


def analyze_molecular_span(positions):
    """
    分析分子跨度（最大原子对距离）
    
    Args:
        positions: [N, 3] 原子坐标
    
    Returns:
        float: 分子跨度（最大原子对距离，单位：Å）
    """
    return torch.cdist(positions, positions, p=2).max().item()

