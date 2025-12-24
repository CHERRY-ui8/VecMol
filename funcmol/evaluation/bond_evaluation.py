"""
键相关评估模块
包含键判断、键长分析、缺失键检测、连通性分析等功能
"""

import torch
import numpy as np
from typing import Tuple, Optional, List, Dict

from funcmol.evaluation.utils_evaluation import (
    bonds1, bonds2, bonds3,
    margin1, margin2, margin3,
    atom_decoder_dict
)


def get_bond_order(atom1, atom2, distance, check_exists=False, 
                   margin1_val=None, margin2_val=None, margin3_val=None):
    """
    判断键类型（单键/双键/三键）
    
    Args:
        atom1: 原子1类型（字符串，如 'C', 'H'）
        atom2: 原子2类型（字符串）
        distance: 原子间距离（单位：Å）
        check_exists: 是否检查原子对是否存在标准键长
        margin1_val: margin1值（pm单位），默认使用全局margin1
        margin2_val: margin2值（pm单位），默认使用全局margin2
        margin3_val: margin3值（pm单位），默认使用全局margin3
    
    Returns:
        int: 键类型 (0=无键, 1=单键, 2=双键, 3=三键)
    """
    if margin1_val is None:
        margin1_val = margin1
    if margin2_val is None:
        margin2_val = margin2
    if margin3_val is None:
        margin3_val = margin3
    
    distance_pm = 100 * distance  # 转换为pm单位
    
    # 检查原子对是否存在标准键长
    if check_exists:
        if atom1 not in bonds1:
            return 0
        if atom2 not in bonds1[atom1]:
            return 0
    
    # 首先检查是否在单键范围内
    if distance_pm < bonds1[atom1][atom2] + margin1_val:
        # 检查是否在双键范围内
        if atom1 in bonds2 and atom2 in bonds2[atom1]:
            thr_bond2 = bonds2[atom1][atom2] + margin2_val
            if distance_pm < thr_bond2:
                # 检查是否在三键范围内
                if atom1 in bonds3 and atom2 in bonds3[atom1]:
                    thr_bond3 = bonds3[atom1][atom2] + margin3_val
                    if distance_pm < thr_bond3:
                        return 3  # 三键
                return 2  # 双键
        return 1  # 单键
    return 0  # 无键


def get_expected_bond_order(atom1, atom2, distance, 
                            margin1_val=None, margin2_val=None, margin3_val=None):
    """
    根据距离判断期望的键类型（用于检测缺失键和键类型不匹配）
    
    按照从高到低的优先级检查：三键 -> 双键 -> 单键
    
    Args:
        atom1: 原子1类型（字符串）
        atom2: 原子2类型（字符串）
        distance: 原子间距离（单位：Å）
        margin1_val: margin1值（pm单位），默认使用全局margin1
        margin2_val: margin2值（pm单位），默认使用全局margin2
        margin3_val: margin3值（pm单位），默认使用全局margin3
    
    Returns:
        tuple: (expected_order, standard_dist, threshold)
            - expected_order: 期望的键类型 (0/1/2/3)
            - standard_dist: 对应的标准键长（单位：Å）
            - threshold: 判断阈值（单位：Å）
    """
    if margin1_val is None:
        margin1_val = margin1
    if margin2_val is None:
        margin2_val = margin2
    if margin3_val is None:
        margin3_val = margin3
    
    distance_pm = 100 * distance  # 转换为pm单位
    
    # 按优先级检查：三键 -> 双键 -> 单键
    # 检查三键
    if atom1 in bonds3 and atom2 in bonds3[atom1]:
        standard_dist_pm = bonds3[atom1][atom2]
        threshold_pm = standard_dist_pm + margin3_val
        if distance_pm < threshold_pm:
            return 3, standard_dist_pm / 100.0, threshold_pm / 100.0
    
    # 检查双键
    if atom1 in bonds2 and atom2 in bonds2[atom1]:
        standard_dist_pm = bonds2[atom1][atom2]
        threshold_pm = standard_dist_pm + margin2_val
        if distance_pm < threshold_pm:
            return 2, standard_dist_pm / 100.0, threshold_pm / 100.0
    
    # 检查单键
    if atom1 in bonds1 and atom2 in bonds1[atom1]:
        standard_dist_pm = bonds1[atom1][atom2]
        threshold_pm = standard_dist_pm + margin1_val
        if distance_pm < threshold_pm:
            return 1, standard_dist_pm / 100.0, threshold_pm / 100.0
    
    return 0, None, None


def build_xae_molecule(positions, atom_types, dataset_info, atom_decoder):
    """
    构建分子键矩阵
    
    Args:
        positions: [N, 3] 原子坐标
        atom_types: [N] 原子类型索引
        dataset_info: 数据集信息字典
        atom_decoder: 原子类型解码器列表
    
    Returns:
        tuple: (X, A, E)
            - X: 原子类型 [N]
            - A: 邻接矩阵 [N, N] (bool)
            - E: 键类型矩阵 [N, N] (int)
    """
    n = positions.shape[0]
    X = atom_types
    A = torch.zeros((n, n), dtype=torch.bool)
    E = torch.zeros((n, n), dtype=torch.int)
    
    pos = positions.unsqueeze(0)
    dists = torch.cdist(pos, pos, p=2).squeeze(0)
    
    for i in range(n):
        for j in range(i):
            pair = sorted([atom_types[i], atom_types[j]])
            if dataset_info['name'] == 'qm9':
                order = get_bond_order(
                    atom_decoder[pair[0]], 
                    atom_decoder[pair[1]], 
                    dists[i, j]
                )
            elif dataset_info['name'] == 'geom':
                # 对于geom数据集，使用limit_bonds_to_one
                order = get_bond_order(
                    atom_decoder[pair[0]], 
                    atom_decoder[pair[1]], 
                    dists[i, j],
                    check_exists=True
                )
                if order > 1:
                    order = 1  # 限制为单键
            
            if order > 0:
                A[i, j] = 1
                E[i, j] = order
                E[j, i] = order
    
    return X, A, E


def check_connectivity(bond_types):
    """
    检查分子的连通性（使用DFS）
    
    Args:
        bond_types: [N, N] 键类型矩阵
    
    Returns:
        tuple: (num_components, is_connected)
            - num_components: 连通分量数
            - is_connected: 是否连通（连通分量数==1）
    """
    n_atoms = bond_types.shape[0]
    if n_atoms == 0:
        return 0, False
    
    # 检查是否有边
    has_edges = (bond_types > 0).any()
    if not has_edges:
        # 没有边，每个原子都是独立的连通分量
        return n_atoms, False
    
    # 使用DFS计算连通分量
    visited = torch.zeros(n_atoms, dtype=torch.bool)
    num_components = 0
    
    def dfs(node):
        """深度优先搜索"""
        visited[node] = True
        neighbors = torch.nonzero(bond_types[node] > 0, as_tuple=False).squeeze(-1)
        for neighbor in neighbors:
            if neighbor.item() != node and not visited[neighbor.item()]:
                dfs(neighbor.item())
    
    # 遍历所有未访问的节点
    for i in range(n_atoms):
        if not visited[i]:
            dfs(i)
            num_components += 1
    
    is_connected = (num_components == 1)
    return num_components, is_connected


def check_connectivity_with_labels(bond_types):
    """
    检查分子的连通性并返回每个原子所属的连通分量ID
    
    Args:
        bond_types: [N, N] 键类型矩阵
        
    Returns:
        tuple: (num_components, component_ids)
            - num_components: 连通分量数
            - component_ids: 每个原子所属的连通分量ID [N]
    """
    n_atoms = bond_types.shape[0]
    if n_atoms == 0:
        return 0, torch.zeros(0, dtype=torch.long)
    
    # 检查是否有边
    has_edges = (bond_types > 0).any()
    if not has_edges:
        # 没有边，每个原子都是独立的连通分量
        component_ids = torch.arange(n_atoms, dtype=torch.long)
        return n_atoms, component_ids
    
    # 使用DFS计算连通分量并标记
    visited = torch.zeros(n_atoms, dtype=torch.bool)
    component_ids = torch.zeros(n_atoms, dtype=torch.long)
    current_component = 0
    
    def dfs_label(node, comp_id):
        """深度优先搜索并标记连通分量"""
        visited[node] = True
        component_ids[node] = comp_id
        neighbors = torch.nonzero(bond_types[node] > 0, as_tuple=False).squeeze(-1)
        for neighbor in neighbors:
            if neighbor.item() != node and not visited[neighbor.item()]:
                dfs_label(neighbor.item(), comp_id)
    
    # 遍历所有未访问的节点
    for i in range(n_atoms):
        if not visited[i]:
            dfs_label(i, current_component)
            current_component += 1
    
    num_components = current_component
    return num_components, component_ids


def compute_missing_bond_deviations_strict(positions, atom_types, bond_types, atom_decoder, dataset_info, 
                                            strict_margin1=15, strict_margin2=10, strict_margin3=6):
    """
    计算应该形成键但未形成键的原子对的距离偏差（使用严格标准，考虑所有键类型）
    
    优化版本：使用 get_expected_bond_order() 来判断期望的键类型，不仅检测单键缺失，
    还检测双键/三键缺失，以及键类型不匹配的情况。
    
    Args:
        positions: [N, 3] 原子坐标
        atom_types: [N] 原子类型索引
        bond_types: [N, N] 键类型矩阵
        atom_decoder: 原子类型解码器列表
        dataset_info: 数据集信息字典
        strict_margin1: 严格margin1值（pm单位），默认15pm
        strict_margin2: 严格margin2值（pm单位），默认10pm
        strict_margin3: 严格margin3值（pm单位），默认6pm
    
    Returns:
        missing_bonds: List of dicts with keys: pair, actual_dist, standard_dist, deviation_pct, 
                      expected_order, actual_order, is_type_mismatch
    """
    n = positions.shape[0]
    dists = torch.cdist(positions, positions, p=2)
    missing_bonds = []
    
    # 计算连通分量，用于检测跨分量的缺失键
    num_components, component_ids = check_connectivity_with_labels(bond_types)
    
    for i in range(n):
        for j in range(i):
            atom1_str = atom_decoder[atom_types[i].item()]
            atom2_str = atom_decoder[atom_types[j].item()]
            
            # 检查标准键长（需要按字母顺序排序）
            pair = sorted([atom1_str, atom2_str])
            atom1_key = pair[0]
            atom2_key = pair[1]
            
            # 使用优化后的函数判断期望的键类型
            actual_dist = dists[i, j].item()
            expected_order, standard_dist, threshold = get_expected_bond_order(
                atom1_key, atom2_key, actual_dist,
                margin1_val=strict_margin1,
                margin2_val=strict_margin2,
                margin3_val=strict_margin3
            )
            
            if expected_order > 0:  # 应该形成某种类型的键
                actual_order = bond_types[i, j].item()
                is_cross_component = (component_ids[i] != component_ids[j])
                has_bond = (actual_order > 0)
                deviation_pct = (actual_dist - standard_dist) / standard_dist * 100
                
                # 检查是否应该形成键但未形成，或键类型不匹配
                is_missing = (not has_bond) or is_cross_component
                is_type_mismatch = (has_bond and actual_order != expected_order)
                
                # 情况1：没有键或跨分量
                if is_missing:
                    missing_bonds.append({
                        'pair': (i, j),
                        'actual_dist': actual_dist,
                        'standard_dist': standard_dist,
                        'deviation_pct': deviation_pct,
                        'atom1': atom1_str,
                        'atom2': atom2_str,
                        'expected_order': expected_order,
                        'actual_order': actual_order,
                        'is_cross_component': is_cross_component,
                        'is_type_mismatch': False,
                        'has_bond': has_bond
                    })
                # 情况2：键类型不匹配（例如应该形成双键但只形成了单键）
                elif is_type_mismatch:
                    missing_bonds.append({
                        'pair': (i, j),
                        'actual_dist': actual_dist,
                        'standard_dist': standard_dist,
                        'deviation_pct': deviation_pct,
                        'atom1': atom1_str,
                        'atom2': atom2_str,
                        'expected_order': expected_order,
                        'actual_order': actual_order,
                        'is_cross_component': False,
                        'is_type_mismatch': True,
                        'has_bond': True
                    })
                # 情况3：有键但距离偏离标准值超过5%（结构不理想）
                elif has_bond and abs(deviation_pct) > 5.0:
                    # 使用更宽松的阈值来检测偏离标准值的键
                    relaxed_threshold = threshold * 1.2  # 放宽20%
                    if actual_dist < relaxed_threshold:
                        missing_bonds.append({
                            'pair': (i, j),
                            'actual_dist': actual_dist,
                            'standard_dist': standard_dist,
                            'deviation_pct': deviation_pct,
                            'atom1': atom1_str,
                            'atom2': atom2_str,
                            'expected_order': expected_order,
                            'actual_order': actual_order,
                            'is_cross_component': False,
                            'is_type_mismatch': False,
                            'has_bond': True
                        })
    
    return missing_bonds


def compute_bond_type_mismatches(positions, atom_types, bond_types, atom_decoder, dataset_info,
                                  margin1_val=None, margin2_val=None, margin3_val=None):
    """
    检测键类型不匹配的情况（实际键类型与期望键类型不一致）
    
    Args:
        positions: [N, 3] 原子坐标
        atom_types: [N] 原子类型索引
        bond_types: [N, N] 键类型矩阵
        atom_decoder: 原子类型解码器列表
        dataset_info: 数据集信息字典
        margin1_val: margin1值（pm单位），默认使用全局margin1
        margin2_val: margin2值（pm单位），默认使用全局margin2
        margin3_val: margin3值（pm单位），默认使用全局margin3
    
    Returns:
        mismatches: List of dicts with keys: pair, actual_dist, expected_order, actual_order, deviation_pct
    """
    if margin1_val is None:
        margin1_val = margin1
    if margin2_val is None:
        margin2_val = margin2
    if margin3_val is None:
        margin3_val = margin3
    
    n = positions.shape[0]
    dists = torch.cdist(positions, positions, p=2)
    mismatches = []
    
    for i in range(n):
        for j in range(i):
            if bond_types[i, j] > 0:  # 当前判断为有键
                atom1_str = atom_decoder[atom_types[i].item()]
                atom2_str = atom_decoder[atom_types[j].item()]
                pair = sorted([atom1_str, atom2_str])
                atom1_key = pair[0]
                atom2_key = pair[1]
                
                # 判断期望的键类型
                actual_dist = dists[i, j].item()
                expected_order, standard_dist, _ = get_expected_bond_order(
                    atom1_key, atom2_key, actual_dist,
                    margin1_val=margin1_val,
                    margin2_val=margin2_val,
                    margin3_val=margin3_val
                )
                
                actual_order = bond_types[i, j].item()
                
                # 如果期望键类型与实际键类型不匹配
                if expected_order > 0 and actual_order != expected_order:
                    deviation_pct = (actual_dist - standard_dist) / standard_dist * 100
                    mismatches.append({
                        'pair': (i, j),
                        'actual_dist': actual_dist,
                        'standard_dist': standard_dist,
                        'deviation_pct': deviation_pct,
                        'atom1': atom1_str,
                        'atom2': atom2_str,
                        'expected_order': expected_order,
                        'actual_order': actual_order
                    })
    
    return mismatches


def compute_excessive_bond_deviations(positions, atom_types, bond_types, atom_decoder, dataset_info, 
                                       margin1_val=None, margin2_val=None, margin3_val=None):
    """
    计算不应该形成键但形成键的原子对（距离过远但仍被判断为有键）
    
    Args:
        positions: [N, 3] 原子坐标
        atom_types: [N] 原子类型索引
        bond_types: [N, N] 键类型矩阵
        atom_decoder: 原子类型解码器列表
        dataset_info: 数据集信息字典
        margin1_val: margin1值（pm单位），默认使用全局margin1
        margin2_val: margin2值（pm单位），默认使用全局margin2
        margin3_val: margin3值（pm单位），默认使用全局margin3
    
    Returns:
        excessive_bonds: List of dicts with keys: pair, actual_dist, standard_dist, deviation_pct, bond_order
    """
    if margin1_val is None:
        margin1_val = margin1
    if margin2_val is None:
        margin2_val = margin2
    if margin3_val is None:
        margin3_val = margin3
    
    n = positions.shape[0]
    dists = torch.cdist(positions, positions, p=2)
    excessive_bonds = []
    
    for i in range(n):
        for j in range(i):
            if bond_types[i, j] > 0:  # 当前判断为有键
                atom1_str = atom_decoder[atom_types[i].item()]
                atom2_str = atom_decoder[atom_types[j].item()]
                pair = sorted([atom1_str, atom2_str])
                atom1_key = pair[0]
                atom2_key = pair[1]
                
                # 判断期望的键类型
                actual_dist = dists[i, j].item()
                expected_order, standard_dist, threshold = get_expected_bond_order(
                    atom1_key, atom2_key, actual_dist,
                    margin1_val=margin1_val,
                    margin2_val=margin2_val,
                    margin3_val=margin3_val
                )
                
                # 如果距离超过应该形成键的范围
                if expected_order == 0 or actual_dist > threshold:
                    # 使用单键标准作为参考
                    if atom1_key in bonds1 and atom2_key in bonds1[atom1_key]:
                        ref_standard_dist = bonds1[atom1_key][atom2_key] / 100.0
                        deviation_pct = (actual_dist - ref_standard_dist) / ref_standard_dist * 100
                        excessive_bonds.append({
                            'pair': (i, j),
                            'actual_dist': actual_dist,
                            'standard_dist': ref_standard_dist,
                            'deviation_pct': deviation_pct,
                            'atom1': atom1_str,
                            'atom2': atom2_str,
                            'bond_order': bond_types[i, j].item()
                        })
    
    return excessive_bonds


def compute_connectivity_continuity_score(positions, atom_types, bond_types, atom_decoder, dataset_info,
                                          strict_margin1=15, strict_margin2=10, strict_margin3=6):
    """
    计算综合连续性连通性分数
    
    使用更严格的标准来判断"应该形成键"，考虑所有键类型（单/双/三键），
    这样可以检测出即使使用宽松margin判断为有键，但距离仍然偏离标准值的情况。
    
    Args:
        positions: [N, 3] 原子坐标
        atom_types: [N] 原子类型索引
        bond_types: [N, N] 键类型矩阵
        atom_decoder: 原子类型解码器列表
        dataset_info: 数据集信息字典
        strict_margin1: 严格margin1值（pm单位），默认15pm
        strict_margin2: 严格margin2值（pm单位），默认10pm
        strict_margin3: 严格margin3值（pm单位），默认6pm
    
    Returns:
        dict: 包含连续性指标的字典
    """
    n = positions.shape[0]
    dists = torch.cdist(positions, positions, p=2)
    
    missing_bonds = []
    all_potential_bonds = []  # 所有应该形成键的原子对（无论是否已形成键）
    type_mismatches = []  # 键类型不匹配的情况
    
    for i in range(n):
        for j in range(i):
            atom1_str = atom_decoder[atom_types[i].item()]
            atom2_str = atom_decoder[atom_types[j].item()]
            pair = sorted([atom1_str, atom2_str])
            atom1_key = pair[0]
            atom2_key = pair[1]
            
            # 使用优化后的函数判断期望的键类型
            actual_dist = dists[i, j].item()
            expected_order, standard_dist, threshold = get_expected_bond_order(
                atom1_key, atom2_key, actual_dist,
                margin1_val=strict_margin1,
                margin2_val=strict_margin2,
                margin3_val=strict_margin3
            )
            
            if expected_order > 0:  # 应该形成某种类型的键
                deviation_pct = (actual_dist - standard_dist) / standard_dist * 100
                actual_order = bond_types[i, j].item()
                
                all_potential_bonds.append({
                    'pair': (i, j),
                    'actual_dist': actual_dist,
                    'standard_dist': standard_dist,
                    'deviation_pct': deviation_pct,
                    'expected_order': expected_order,
                    'actual_order': actual_order,
                    'has_bond': actual_order > 0
                })
                
                # 检查缺失键
                if actual_order == 0:
                    missing_bonds.append({
                        'pair': (i, j),
                        'actual_dist': actual_dist,
                        'standard_dist': standard_dist,
                        'deviation_pct': deviation_pct,
                        'expected_order': expected_order,
                        'actual_order': 0
                    })
                # 检查键类型不匹配
                elif actual_order != expected_order:
                    type_mismatches.append({
                        'pair': (i, j),
                        'actual_dist': actual_dist,
                        'standard_dist': standard_dist,
                        'deviation_pct': deviation_pct,
                        'expected_order': expected_order,
                        'actual_order': actual_order
                    })
    
    if len(all_potential_bonds) == 0:
        return {
            'mean_deviation_pct': 0.0,
            'max_deviation_pct': 0.0,
            'missing_bond_count': 0,
            'missing_bond_ratio': 0.0,
            'continuity_score': 1.0,
            'overall_mean_deviation_pct': 0.0,
            'type_mismatch_count': 0,
            'type_mismatch_ratio': 0.0
        }
    
    # 计算所有应该形成键的原子对的偏差（包括已形成键的）
    all_deviations = [bond['deviation_pct'] for bond in all_potential_bonds]
    overall_mean_deviation = np.mean(all_deviations)
    overall_max_deviation = np.max(all_deviations)
    
    # 计算缺失键的统计信息
    if len(missing_bonds) > 0:
        missing_deviations = [bond['deviation_pct'] for bond in missing_bonds]
        mean_deviation = np.mean(missing_deviations)
        max_deviation = np.max(missing_deviations)
    else:
        mean_deviation = 0.0
        max_deviation = 0.0
    
    missing_bond_ratio = len(missing_bonds) / len(all_potential_bonds) if len(all_potential_bonds) > 0 else 0.0
    type_mismatch_ratio = len(type_mismatches) / len(all_potential_bonds) if len(all_potential_bonds) > 0 else 0.0
    
    # 计算连续性分数（综合考虑所有应该形成键的原子对的偏差、缺失键比例、键类型不匹配比例）
    # 使用整体平均偏差的归一化版本
    # 假设最大合理偏差为30%，超过30%认为严重偏离
    normalized_deviation = min(abs(overall_mean_deviation) / 30.0, 1.0)
    
    # 综合考虑偏差、缺失键比例、键类型不匹配比例
    continuity_score = 1.0 - (0.4 * normalized_deviation + 0.3 * missing_bond_ratio + 0.3 * type_mismatch_ratio)
    continuity_score = max(0.0, continuity_score)  # 确保分数在[0, 1]范围内
    
    return {
        'mean_deviation_pct': mean_deviation,
        'max_deviation_pct': max_deviation,
        'missing_bond_count': len(missing_bonds),
        'missing_bond_ratio': missing_bond_ratio,
        'continuity_score': continuity_score,
        'total_potential_bonds': len(all_potential_bonds),
        'overall_mean_deviation_pct': overall_mean_deviation,
        'overall_max_deviation_pct': overall_max_deviation,
        'type_mismatch_count': len(type_mismatches),
        'type_mismatch_ratio': type_mismatch_ratio
    }


def analyze_bonds(molecule_dir, output_dir=None, strict_margin1=15, strict_margin2=10, strict_margin3=6):
    """
    分析分子的键和连通性
    
    Args:
        molecule_dir: 包含 .npz 文件的目录
        output_dir: 输出目录（可选）
        strict_margin1: 严格margin1值（pm单位），默认15pm
        strict_margin2: 严格margin2值（pm单位），默认10pm
        strict_margin3: 严格margin3值（pm单位），默认6pm
    
    Returns:
        dict: 包含键和连通性分析结果的字典
    """
    from pathlib import Path
    from tqdm import tqdm
    from funcmol.evaluation.utils_evaluation import load_molecules_from_npz, atom_decoder_dict, margin1
    from funcmol.evaluation.structure_evaluation import compute_min_distances
    
    molecule_dir = Path(molecule_dir)
    npz_files = sorted(molecule_dir.glob("generated_*.npz"))
    
    print(f"找到 {len(npz_files)} 个 .npz 分子文件")
    
    atom_decoder = atom_decoder_dict['qm9_with_h']
    dataset_info = {'name': 'qm9'}
    
    # 存储统计数据
    num_components_list = []
    is_connected_list = []
    bond_lengths = []
    
    # 连续性指标统计
    all_missing_bond_deviations = []
    all_continuity_scores = []
    all_missing_bond_ratios = []
    all_mean_deviations = []
    all_max_deviations = []
    all_overall_mean_deviations = []
    all_overall_max_deviations = []
    all_type_mismatch_counts = []
    all_type_mismatch_ratios = []
    
    print("分析分子键和连通性...")
    for npz_file in tqdm(npz_files, desc="处理分子"):
        try:
            # 加载分子
            data = np.load(npz_file)
            coords = data['coords']  # (N, 3)
            types = data['types']    # (N,)
            
            # 转换为torch张量
            positions = torch.tensor(coords, dtype=torch.float32)
            atom_types = torch.tensor(types, dtype=torch.long)
            
            # 过滤掉填充的原子
            valid_mask = atom_types != -1
            if not valid_mask.any():
                continue
            
            positions = positions[valid_mask]
            atom_types = atom_types[valid_mask]
            
            # 构建键类型矩阵
            _, _, bond_types = build_xae_molecule(
                positions=positions,
                atom_types=atom_types,
                dataset_info=dataset_info,
                atom_decoder=atom_decoder
            )
            
            # 检查连通性
            num_components, is_connected = check_connectivity(bond_types)
            num_components_list.append(num_components)
            is_connected_list.append(is_connected)
            
            # 计算连续性指标（使用严格标准）
            continuity_metrics = compute_connectivity_continuity_score(
                positions, atom_types, bond_types, atom_decoder, dataset_info,
                strict_margin1=strict_margin1,
                strict_margin2=strict_margin2,
                strict_margin3=strict_margin3
            )
            all_continuity_scores.append(continuity_metrics['continuity_score'])
            all_missing_bond_ratios.append(continuity_metrics['missing_bond_ratio'])
            all_mean_deviations.append(continuity_metrics['mean_deviation_pct'])
            all_max_deviations.append(continuity_metrics['max_deviation_pct'])
            all_overall_mean_deviations.append(continuity_metrics['overall_mean_deviation_pct'])
            all_overall_max_deviations.append(continuity_metrics['overall_max_deviation_pct'])
            all_type_mismatch_counts.append(continuity_metrics['type_mismatch_count'])
            all_type_mismatch_ratios.append(continuity_metrics['type_mismatch_ratio'])
            
            # 收集缺失键的偏差
            missing_bonds = compute_missing_bond_deviations_strict(
                positions, atom_types, bond_types, atom_decoder, dataset_info,
                strict_margin1=strict_margin1,
                strict_margin2=strict_margin2,
                strict_margin3=strict_margin3
            )
            if missing_bonds:
                missing_deviations = [bond['deviation_pct'] for bond in missing_bonds]
                all_missing_bond_deviations.extend(missing_deviations)
            
            # 计算键长（只计算有键的原子对）
            distances = torch.cdist(positions, positions, p=2)
            triu_mask = torch.triu(torch.ones_like(bond_types, dtype=torch.bool), diagonal=1)
            bond_mask = (bond_types > 0) & triu_mask
            
            if bond_mask.any():
                bond_distances = distances[bond_mask]
                bond_lengths.extend(bond_distances.cpu().numpy())
            
        except Exception as e:
            print(f"\n处理文件 {npz_file} 时出错: {e}")
            continue
    
    # 转换为numpy数组
    num_components_list = np.array(num_components_list)
    is_connected_list = np.array(is_connected_list)
    bond_lengths = np.array(bond_lengths) if bond_lengths else np.array([])
    
    # 连续性指标数组
    all_continuity_scores = np.array(all_continuity_scores) if all_continuity_scores else np.array([])
    all_missing_bond_ratios = np.array(all_missing_bond_ratios) if all_missing_bond_ratios else np.array([])
    all_mean_deviations = np.array(all_mean_deviations) if all_mean_deviations else np.array([])
    all_max_deviations = np.array(all_max_deviations) if all_max_deviations else np.array([])
    all_missing_bond_deviations = np.array(all_missing_bond_deviations) if all_missing_bond_deviations else np.array([])
    all_overall_mean_deviations = np.array(all_overall_mean_deviations) if all_overall_mean_deviations else np.array([])
    all_overall_max_deviations = np.array(all_overall_max_deviations) if all_overall_max_deviations else np.array([])
    all_type_mismatch_counts = np.array(all_type_mismatch_counts) if all_type_mismatch_counts else np.array([])
    all_type_mismatch_ratios = np.array(all_type_mismatch_ratios) if all_type_mismatch_ratios else np.array([])
    
    # 打印统计结果
    print("\n" + "="*60)
    print("键和连通性分析结果")
    print("="*60)
    
    print(f"\n🔗 键长统计:")
    if len(bond_lengths) > 0:
        print(f"  总键数: {len(bond_lengths)}")
        print(f"  平均键长: {bond_lengths.mean():.4f} Å")
        print(f"  中位数键长: {np.median(bond_lengths):.4f} Å")
        print(f"  键长范围: {bond_lengths.min():.4f} - {bond_lengths.max():.4f} Å")
    else:
        print("  没有检测到任何键！")
    
    print(f"\n🌐 分子连通性统计:")
    print(f"  连通分子数（连通分量=1）: {is_connected_list.sum()}")
    print(f"  非连通分子数（连通分量>1）: {(~is_connected_list).sum()}")
    print(f"  连通分子比例: {is_connected_list.sum() / len(is_connected_list) * 100:.2f}%")
    print(f"  平均连通分量数: {num_components_list.mean():.2f}")
    print(f"  最大连通分量数: {num_components_list.max()}")
    
    print(f"\n🔗 连续性连通性指标 (使用严格标准: margin1={strict_margin1}pm, margin2={strict_margin2}pm, margin3={strict_margin3}pm):")
    if len(all_continuity_scores) > 0:
        print(f"  平均连续性分数: {all_continuity_scores.mean():.4f} (1.0=完美连通)")
        print(f"  中位数连续性分数: {np.median(all_continuity_scores):.4f}")
        print(f"  连续性分数范围: {all_continuity_scores.min():.4f} - {all_continuity_scores.max():.4f}")
        print(f"  说明: 分数基于所有应该形成键的原子对的整体偏差、缺失键比例和键类型不匹配比例计算")
        
        print(f"\n  整体偏差统计（所有应该形成键的原子对，无论是否已形成键）:")
        if len(all_overall_mean_deviations) > 0:
            print(f"    平均整体偏差百分比: {all_overall_mean_deviations.mean():.4f}%")
            print(f"    中位数整体偏差百分比: {np.median(all_overall_mean_deviations):.4f}%")
        
        print(f"\n  缺失键统计（应该形成键但未形成键的原子对）:")
        if len(all_missing_bond_ratios) > 0:
            print(f"    平均缺失键比例: {all_missing_bond_ratios.mean():.4f} ({all_missing_bond_ratios.mean()*100:.2f}%)")
            print(f"    中位数缺失键比例: {np.median(all_missing_bond_ratios):.4f} ({np.median(all_missing_bond_ratios)*100:.2f}%)")
        
        if len(all_missing_bond_deviations) > 0:
            print(f"    所有缺失键的偏差分布:")
            print(f"      总缺失键数: {len(all_missing_bond_deviations)}")
            print(f"      平均偏差: {all_missing_bond_deviations.mean():.4f}%")
            print(f"      中位数偏差: {np.median(all_missing_bond_deviations):.4f}%")
        
        print(f"\n  键类型不匹配统计:")
        if len(all_type_mismatch_counts) > 0:
            print(f"    平均键类型不匹配数: {all_type_mismatch_counts.mean():.2f}")
            print(f"    平均键类型不匹配比例: {all_type_mismatch_ratios.mean():.4f} ({all_type_mismatch_ratios.mean()*100:.2f}%)")
    
    return {
        'bond_lengths': bond_lengths,
        'num_components': num_components_list,
        'is_connected': is_connected_list,
        'continuity_scores': all_continuity_scores,
        'missing_bond_ratios': all_missing_bond_ratios,
        'mean_deviations': all_mean_deviations,
        'max_deviations': all_max_deviations,
        'missing_bond_deviations': all_missing_bond_deviations,
        'overall_mean_deviations': all_overall_mean_deviations,
        'overall_max_deviations': all_overall_max_deviations,
        'type_mismatch_counts': all_type_mismatch_counts,
        'type_mismatch_ratios': all_type_mismatch_ratios
    }

