"""
键相关评估模块
包含键判断、键长分析、缺失键检测、连通性分析等功能
"""

import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

from funcmol.evaluation.utils_evaluation import (
    bonds1, bonds2, bonds3,
    margin1, margin2, margin3,
    atom_decoder_dict
)


def get_all_possible_bond_orders(atom1, atom2, distance, 
                                 margin1_val=None, margin2_val=None, margin3_val=None):
    """
    获取所有在margin内的可能键类型（不按优先级排序）
    
    Args:
        atom1: 原子1类型（字符串）
        atom2: 原子2类型（字符串）
        distance: 原子间距离（单位：Å）
        margin1_val: margin1值（pm单位），默认使用全局margin1
        margin2_val: margin2值（pm单位），默认使用全局margin2
        margin3_val: margin3值（pm单位），默认使用全局margin3
    
    Returns:
        list: 所有可能的键类型列表，每个元素为 (bond_order, standard_dist_pm, relative_deviation)
    """
    if margin1_val is None:
        margin1_val = margin1
    if margin2_val is None:
        margin2_val = margin2
    if margin3_val is None:
        margin3_val = margin3
    
    distance_pm = 100 * distance  # 转换为pm单位
    possible_bonds = []
    
    # 检查三键
    if atom1 in bonds3 and atom2 in bonds3[atom1]:
        standard_bond3_pm = bonds3[atom1][atom2]
        threshold3_pm = standard_bond3_pm + margin3_val
        if distance_pm < threshold3_pm:
            relative_deviation = abs(distance_pm - standard_bond3_pm) / standard_bond3_pm
            possible_bonds.append((3, standard_bond3_pm, relative_deviation))
    
    # 检查双键
    if atom1 in bonds2 and atom2 in bonds2[atom1]:
        standard_bond2_pm = bonds2[atom1][atom2]
        threshold2_pm = standard_bond2_pm + margin2_val
        if distance_pm < threshold2_pm:
            relative_deviation = abs(distance_pm - standard_bond2_pm) / standard_bond2_pm
            possible_bonds.append((2, standard_bond2_pm, relative_deviation))
    
    # 检查单键
    if atom1 in bonds1 and atom2 in bonds1[atom1]:
        standard_bond1_pm = bonds1[atom1][atom2]
        threshold1_pm = standard_bond1_pm + margin1_val
        if distance_pm < threshold1_pm:
            relative_deviation = abs(distance_pm - standard_bond1_pm) / standard_bond1_pm
            possible_bonds.append((1, standard_bond1_pm, relative_deviation))
    
    return possible_bonds


def get_bond_order(atom1, atom2, distance, check_exists=False, 
                   margin1_val=None, margin2_val=None, margin3_val=None):
    """
    判断键类型（单键/双键/三键）
    
    使用"最接近标准键长"原则：选择距离最接近标准键长的键类型
    
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
    
    # 使用"最接近标准键长"原则（方案1：相对偏差）
    # 收集所有可能的键类型及其标准键长
    candidate_bonds = []
    
    # 检查三键
    if atom1 in bonds3 and atom2 in bonds3[atom1]:
        standard_bond3_pm = bonds3[atom1][atom2]
        threshold3_pm = standard_bond3_pm + margin3_val
        if distance_pm < threshold3_pm:
            # 使用相对偏差而非绝对偏差
            relative_deviation = abs(distance_pm - standard_bond3_pm) / standard_bond3_pm
            candidate_bonds.append((3, standard_bond3_pm, relative_deviation))
    
    # 检查双键
    if atom1 in bonds2 and atom2 in bonds2[atom1]:
        standard_bond2_pm = bonds2[atom1][atom2]
        threshold2_pm = standard_bond2_pm + margin2_val
        if distance_pm < threshold2_pm:
            # 使用相对偏差而非绝对偏差
            relative_deviation = abs(distance_pm - standard_bond2_pm) / standard_bond2_pm
            candidate_bonds.append((2, standard_bond2_pm, relative_deviation))
    
    # 检查单键
    if atom1 in bonds1 and atom2 in bonds1[atom1]:
        standard_bond1_pm = bonds1[atom1][atom2]
        threshold1_pm = standard_bond1_pm + margin1_val
        if distance_pm < threshold1_pm:
            # 使用相对偏差而非绝对偏差
            relative_deviation = abs(distance_pm - standard_bond1_pm) / standard_bond1_pm
            candidate_bonds.append((1, standard_bond1_pm, relative_deviation))
    
    # 如果没有候选键，返回无键
    if not candidate_bonds:
        return 0
    
    # 选择相对偏差最小的键类型（最接近标准键长）
    candidate_bonds.sort(key=lambda x: x[2])  # 按相对偏差排序
    return candidate_bonds[0][0]  # 返回键类型


def get_expected_bond_order(atom1, atom2, distance, 
                            margin1_val=None, margin2_val=None, margin3_val=None):
    """
    根据距离判断期望的键类型（用于检测缺失键和键类型不匹配）
    
    使用"最接近标准键长"原则：选择距离最接近标准键长的键类型
    
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
    
    # 使用"最接近标准键长"原则（方案1：相对偏差）
    # 收集所有可能的键类型及其标准键长
    candidate_bonds = []
    
    # 检查三键
    if atom1 in bonds3 and atom2 in bonds3[atom1]:
        standard_bond3_pm = bonds3[atom1][atom2]
        threshold3_pm = standard_bond3_pm + margin3_val
        if distance_pm < threshold3_pm:
            # 使用相对偏差而非绝对偏差
            relative_deviation = abs(distance_pm - standard_bond3_pm) / standard_bond3_pm
            candidate_bonds.append((3, standard_bond3_pm, relative_deviation, threshold3_pm))
    
    # 检查双键
    if atom1 in bonds2 and atom2 in bonds2[atom1]:
        standard_bond2_pm = bonds2[atom1][atom2]
        threshold2_pm = standard_bond2_pm + margin2_val
        if distance_pm < threshold2_pm:
            # 使用相对偏差而非绝对偏差
            relative_deviation = abs(distance_pm - standard_bond2_pm) / standard_bond2_pm
            candidate_bonds.append((2, standard_bond2_pm, relative_deviation, threshold2_pm))
    
    # 检查单键
    if atom1 in bonds1 and atom2 in bonds1[atom1]:
        standard_bond1_pm = bonds1[atom1][atom2]
        threshold1_pm = standard_bond1_pm + margin1_val
        if distance_pm < threshold1_pm:
            # 使用相对偏差而非绝对偏差
            relative_deviation = abs(distance_pm - standard_bond1_pm) / standard_bond1_pm
            candidate_bonds.append((1, standard_bond1_pm, relative_deviation, threshold1_pm))
    
    # 如果没有候选键，返回无键
    if not candidate_bonds:
        return 0, None, None
    
    # 选择相对偏差最小的键类型（最接近标准键长）
    candidate_bonds.sort(key=lambda x: x[2])  # 按相对偏差排序
    best_order, best_standard_pm, _, best_threshold_pm = candidate_bonds[0]
    
    return best_order, best_standard_pm / 100.0, best_threshold_pm / 100.0


def optimize_bonds_for_stability(positions, atom_types, atom_decoder, charges,
                                 margin1_val=None, margin2_val=None, margin3_val=None,
                                 max_iterations=10):
    """
    全局优化键组合，选择使稳定原子比例最大的组合
    
    使用迭代改进算法：
    1. 初始：使用"最接近标准键长"原则选择键
    2. 迭代改进：对于每个可能的键，尝试改变它的类型，看是否能提高稳定性
    3. 重复直到收敛或达到最大迭代次数
    
    Args:
        positions: [N, 3] 原子坐标
        atom_types: [N] 原子类型索引
        atom_decoder: 原子类型解码器列表
        charges: [N] 原子电荷
        margin1_val: margin1值（pm单位）
        margin2_val: margin2值（pm单位）
        margin3_val: margin3值（pm单位）
        max_iterations: 最大迭代次数
    
    Returns:
        torch.Tensor: 优化后的键类型矩阵 [N, N]
    """
    from funcmol.analysis.rdkit_functions import allowed_bonds
    
    if margin1_val is None:
        margin1_val = margin1
    if margin2_val is None:
        margin2_val = margin2
    if margin3_val is None:
        margin3_val = margin3
    
    n = positions.shape[0]
    device = positions.device
    
    # 计算所有原子对的距离
    pos = positions.unsqueeze(0)
    dists = torch.cdist(pos, pos, p=2).squeeze(0)
    
    # 找出所有可能的键类型（在margin内的）
    possible_bonds = {}  # {(i, j): [(order, std_dist, rel_dev), ...]}
    for i in range(n):
        for j in range(i):
            atom1 = atom_decoder[atom_types[i].item()]
            atom2 = atom_decoder[atom_types[j].item()]
            distance = dists[i, j].item()
            bonds = get_all_possible_bond_orders(
                atom1, atom2, distance, margin1_val, margin2_val, margin3_val
            )
            if bonds:
                possible_bonds[(i, j)] = bonds
                possible_bonds[(j, i)] = bonds  # 对称
    
    # 初始化：使用"最接近标准键长"原则
    bond_types = torch.zeros((n, n), dtype=torch.int, device=device)
    for (i, j), bonds in possible_bonds.items():
        if i < j:  # 只处理上三角
            # 选择相对偏差最小的
            bonds_sorted = sorted(bonds, key=lambda x: x[2])
            best_order = bonds_sorted[0][0]
            bond_types[i, j] = best_order
            bond_types[j, i] = best_order
    
    def calculate_stability_score(bond_matrix):
        """计算稳定性分数（稳定原子的比例）"""
        edge_types = bond_matrix.clone()
        edge_types[edge_types == 4] = 1.5
        edge_types[edge_types < 0] = 0
        valencies = torch.sum(edge_types, dim=-1).long()
        
        stable_count = 0
        for i, (atom_type, valency, charge) in enumerate(zip(atom_types, valencies, charges)):
            atom_type = atom_type.item()
            valency = valency.item()
            charge = charge.item()
            possible_bonds_list = allowed_bonds[atom_decoder[atom_type]]
            
            if type(possible_bonds_list) == int:
                is_stable = possible_bonds_list == valency
            elif type(possible_bonds_list) == dict:
                expected_bonds = possible_bonds_list.get(charge, possible_bonds_list.get(0))
                if type(expected_bonds) == int:
                    is_stable = expected_bonds == valency
                else:
                    is_stable = valency in expected_bonds
            else:
                is_stable = valency in possible_bonds_list
            
            if is_stable:
                stable_count += 1
        
        return stable_count / n if n > 0 else 0.0
    
    # 迭代改进
    current_score = calculate_stability_score(bond_types)
    
    for _ in range(max_iterations):
        improved = False
        
        # 尝试改进每个可能的键
        for (i, j), bonds in possible_bonds.items():
            if i >= j:  # 只处理上三角
                continue
            
            if len(bonds) <= 1:
                continue  # 只有一个选择，无法改进
            
            # 尝试每个可能的键类型
            best_order = bond_types[i, j].item()
            best_score = current_score
            
            for order, _, _ in bonds:
                if order == best_order:
                    continue
                
                # 尝试这个键类型
                bond_types[i, j] = order
                bond_types[j, i] = order
                new_score = calculate_stability_score(bond_types)
                
                if new_score > best_score:
                    best_score = new_score
                    best_order = order
                    improved = True
            
            # 恢复最佳选择
            bond_types[i, j] = best_order
            bond_types[j, i] = best_order
            current_score = best_score
        
        if not improved:
            break  # 没有改进，收敛
    
    return bond_types


def build_xae_molecule(positions, atom_types, dataset_info, atom_decoder, 
                       margin1_val=None, margin2_val=None, margin3_val=None,
                       use_global_optimization=True, charges=None, 
                       use_iterative_improvement=True, max_iterations=10):
    """
    构建分子键矩阵
    
    Args:
        positions: [N, 3] 原子坐标
        atom_types: [N] 原子类型索引
        dataset_info: 数据集信息字典
        atom_decoder: 原子类型解码器列表
        margin1_val: margin1值（pm单位），默认使用全局margin1
        margin2_val: margin2值（pm单位），默认使用全局margin2
        margin3_val: margin3值（pm单位），默认使用全局margin3
        use_global_optimization: 是否使用全局优化
            - True: 小分子(n<=12)使用回溯穷尽搜索，大分子使用迭代改进或贪心
            - False: 使用简单的"最接近标准键长"方法（贪心）
        charges: [N] 原子电荷
        use_iterative_improvement: 对于大分子(n>12)，是否使用迭代改进（默认True）
            - True: 使用迭代改进（从贪心解开始，逐步改进）
            - False: 使用纯贪心算法
        max_iterations: 迭代改进的最大迭代次数（默认10）
    
    Returns:
        tuple: (X, A, E)
            - X: 原子类型 [N]
            - A: 邻接矩阵 [N, N] (bool)
            - E: 键类型矩阵 [N, N] (int)
    """
    n = positions.shape[0]
    X = atom_types
    device = positions.device
    
    # 默认使用全局优化（穷尽所有可能的键组合，找到稳定原子数最多的组合）
    # 如果use_global_optimization=False，则使用简单的"最接近标准键长"方法
    if use_global_optimization is False:
        # 使用简单的"最接近标准键长"方法（不进行全局优化）
        A = torch.zeros((n, n), dtype=torch.bool, device=device)
        E = torch.zeros((n, n), dtype=torch.int, device=device)
        
        pos = positions.unsqueeze(0)
        dists = torch.cdist(pos, pos, p=2).squeeze(0)
        
        for i in range(n):
            for j in range(i):
                atom1_str = atom_decoder[atom_types[i].item()]
                atom2_str = atom_decoder[atom_types[j].item()]
                if dataset_info['name'] == 'qm9':
                    order = get_bond_order(
                        atom1_str, 
                        atom2_str, 
                        dists[i, j],
                        margin1_val=margin1_val,
                        margin2_val=margin2_val,
                        margin3_val=margin3_val
                    )
                elif dataset_info['name'] == 'geom':
                    order = get_bond_order(
                        atom1_str, 
                        atom2_str, 
                        dists[i, j],
                        check_exists=True,
                        margin1_val=margin1_val,
                        margin2_val=margin2_val,
                        margin3_val=margin3_val
                    )
                    if order > 1:
                        order = 1
                
                if order > 0:
                    A[i, j] = 1
                    A[j, i] = 1
                    E[i, j] = order
                    E[j, i] = order
        
        return X, A, E
    
    # 使用全局优化（默认行为）
    A = torch.zeros((n, n), dtype=torch.bool, device=device)
    E = torch.zeros((n, n), dtype=torch.int, device=device)
    
    pos = positions.unsqueeze(0)
    dists = torch.cdist(pos, pos, p=2).squeeze(0)
    
    # 使用全局优化：穷尽所有可能的键组合，找到稳定原子数最多的组合
    from funcmol.analysis.rdkit_functions import allowed_bonds
    
    # 第一步：找出所有可能的键（距离在margin内的所有键类型）
    # possible_bonds_dict: {(i, j): [(order, std, dev), ...]}
    possible_bonds_dict = {}
    for i in range(n):
        for j in range(i):
            # 获取原子类型字符串
            atom1_str = atom_decoder[atom_types[i].item()]
            atom2_str = atom_decoder[atom_types[j].item()]
            if dataset_info['name'] == 'qm9':
                possible_bonds = get_all_possible_bond_orders(
                    atom1_str, 
                    atom2_str, 
                    dists[i, j],
                    margin1_val=margin1_val,
                    margin2_val=margin2_val,
                    margin3_val=margin3_val
                )
            elif dataset_info['name'] == 'geom':
                # 对于geom数据集，使用limit_bonds_to_one
                possible_bonds = get_all_possible_bond_orders(
                    atom1_str, 
                    atom2_str, 
                    dists[i, j],
                    margin1_val=margin1_val,
                    margin2_val=margin2_val,
                    margin3_val=margin3_val
                )
                # 限制为单键
                possible_bonds = [(1, std, dev) for order, std, dev in possible_bonds if order == 1]
            
            if possible_bonds:
                # 存储所有可能的键类型（包括0=无键）
                possible_bonds_dict[(i, j)] = possible_bonds
    
    # 第二步：全局优化，穷尽所有可能的键组合，找到稳定原子数最多的组合
    # 使用递归回溯或动态规划来搜索所有可能的组合
    def calculate_stability_score(bond_matrix):
        """计算稳定性分数（稳定原子数）"""
        edge_types = bond_matrix.clone()
        edge_types[edge_types == 4] = 1.5
        edge_types[edge_types < 0] = 0
        valencies = torch.sum(edge_types, dim=-1).long()
        
        stable_count = 0
        for i, (atom_type, valency, charge) in enumerate(zip(atom_types, valencies, charges if charges is not None else torch.zeros(n, dtype=torch.long))):
            atom_type = atom_type.item()
            valency = valency.item()
            charge = charge.item()
            possible_bonds_list = allowed_bonds[atom_decoder[atom_type]]
            
            if type(possible_bonds_list) == int:
                is_stable = possible_bonds_list == valency
            elif type(possible_bonds_list) == dict:
                expected_bonds = possible_bonds_list.get(charge, possible_bonds_list.get(0))
                if type(expected_bonds) == int:
                    is_stable = expected_bonds == valency
                else:
                    is_stable = valency in expected_bonds
            else:
                is_stable = valency in possible_bonds_list
            
            if is_stable:
                stable_count += 1
        
        return stable_count
    
    # 使用回溯算法搜索所有可能的键组合，找到稳定原子数最多的组合
    # 为了效率，只对原子数较少的分子进行穷尽搜索，对于大分子使用改进的贪心算法
    if n <= 12:  # 对于小分子，使用穷尽搜索（降低阈值以提高效率）
        best_bond_matrix = None
        best_stable_count = -1
        
        # 获取每个原子的最大价（用于剪枝）
        max_valencies = []
        for i in range(n):
            atom_str = atom_decoder[atom_types[i].item()]
            charge = charges[i].item() if charges is not None else 0
            possible_bonds_list = allowed_bonds[atom_str]
            if type(possible_bonds_list) == int:
                max_val = possible_bonds_list
            elif type(possible_bonds_list) == dict:
                expected_bonds = possible_bonds_list.get(charge, possible_bonds_list.get(0))
                if type(expected_bonds) == int:
                    max_val = expected_bonds
                else:
                    max_val = max(expected_bonds) if expected_bonds else 4
            else:
                max_val = max(possible_bonds_list) if possible_bonds_list else 4
            max_valencies.append(max_val)
        
        def backtrack(bond_matrix, pair_idx, pairs_list, used_valencies):
            nonlocal best_bond_matrix, best_stable_count
            
            if pair_idx == len(pairs_list):
                # 所有原子对都已处理，计算稳定性
                stable_count = calculate_stability_score(bond_matrix)
                if stable_count > best_stable_count:
                    best_stable_count = stable_count
                    best_bond_matrix = bond_matrix.clone()
                return
            
            i, j = pairs_list[pair_idx]
            possible_bonds = possible_bonds_dict.get((i, j), [])
            
            # 尝试无键
            bond_matrix[i, j] = 0
            bond_matrix[j, i] = 0
            backtrack(bond_matrix, pair_idx + 1, pairs_list, used_valencies)
            
            # 尝试所有可能的键类型
            for order, _, _ in possible_bonds:
                # 剪枝：如果添加这个键会导致原子价超出范围，跳过
                if used_valencies[i] + order > max_valencies[i] or used_valencies[j] + order > max_valencies[j]:
                    continue
                
                bond_matrix[i, j] = order
                bond_matrix[j, i] = order
                used_valencies[i] += order
                used_valencies[j] += order
                backtrack(bond_matrix, pair_idx + 1, pairs_list, used_valencies)
                # 回溯
                used_valencies[i] -= order
                used_valencies[j] -= order
        
        pairs_list = list(possible_bonds_dict.keys())
        temp_bond_matrix = torch.zeros((n, n), dtype=torch.int, device=device)
        used_valencies = [0] * n
        backtrack(temp_bond_matrix, 0, pairs_list, used_valencies)
        
        if best_bond_matrix is not None:
            E = best_bond_matrix
            A = (E > 0)
        else:
            # 如果没有找到，使用贪心算法作为后备
            E = temp_bond_matrix
            A = (E > 0)
    else:
        # 对于大分子，使用迭代改进或贪心算法
        if use_iterative_improvement:
            # 使用迭代改进：从贪心解开始，逐步改进
            # 第一步：使用贪心算法获得初始解
            candidate_bonds = []
            for (i, j), possible_bonds in possible_bonds_dict.items():
                if possible_bonds:
                    possible_bonds.sort(key=lambda x: x[2])  # 按相对偏差排序
                    best_order, _, _ = possible_bonds[0]
                    candidate_bonds.append((i, j, best_order))
            
            # 获取每个原子的最大价
            max_valencies = []
            for i in range(n):
                atom_str = atom_decoder[atom_types[i].item()]
                charge = charges[i].item() if charges is not None else 0
                possible_bonds_list = allowed_bonds[atom_str]
                if type(possible_bonds_list) == int:
                    max_val = possible_bonds_list
                elif type(possible_bonds_list) == dict:
                    expected_bonds = possible_bonds_list.get(charge, possible_bonds_list.get(0))
                    if type(expected_bonds) == int:
                        max_val = expected_bonds
                    else:
                        max_val = max(expected_bonds) if expected_bonds else 4
                else:
                    max_val = max(possible_bonds_list) if possible_bonds_list else 4
                max_valencies.append(max_val)
            
            # 按相对偏差排序候选键
            candidate_bonds_with_dev = []
            for i, j, order in candidate_bonds:
                possible_bonds = possible_bonds_dict[(i, j)]
                dev = next((d for o, _, d in possible_bonds if o == order), 1.0)
                candidate_bonds_with_dev.append((i, j, order, dev))
            candidate_bonds_with_dev.sort(key=lambda x: x[3])
            
            # 为每个原子跟踪已使用的价
            used_valencies = [0] * n
            
            # 初始解：选择键，确保每个原子的总价不超过最大价
            for i, j, order, _ in candidate_bonds_with_dev:
                if used_valencies[i] + order <= max_valencies[i] and used_valencies[j] + order <= max_valencies[j]:
                    E[i, j] = order
                    E[j, i] = order
                    A[i, j] = 1
                    A[j, i] = 1
                    used_valencies[i] += order
                    used_valencies[j] += order
            
            # 第二步：迭代改进
            # 计算稳定性分数的函数（返回稳定原子数）
            def calculate_stability_score(bond_matrix):
                edge_types = bond_matrix.clone()
                edge_types[edge_types == 4] = 1.5
                edge_types[edge_types < 0] = 0
                valencies = torch.sum(edge_types, dim=-1).long()
                
                stable_count = 0
                for i, (atom_type, valency, charge) in enumerate(zip(atom_types, valencies, charges if charges is not None else torch.zeros(n, dtype=torch.long))):
                    atom_type = atom_type.item()
                    valency = valency.item()
                    charge = charge.item()
                    possible_bonds_list = allowed_bonds[atom_decoder[atom_type]]
                    
                    if type(possible_bonds_list) == int:
                        is_stable = possible_bonds_list == valency
                    elif type(possible_bonds_list) == dict:
                        expected_bonds = possible_bonds_list.get(charge, possible_bonds_list.get(0))
                        if type(expected_bonds) == int:
                            is_stable = expected_bonds == valency
                        else:
                            is_stable = valency in expected_bonds
                    else:
                        is_stable = valency in possible_bonds_list
                    
                    if is_stable:
                        stable_count += 1
                
                return stable_count
            
            # 迭代改进循环
            current_score = calculate_stability_score(E)
            initial_score = current_score
            
            for iteration in range(max_iterations):
                improved = False
                
                # 尝试改进每个可能的键
                for (i, j), bonds in possible_bonds_dict.items():
                    if i >= j:  # 只处理上三角
                        continue
                    
                    if len(bonds) <= 1:
                        continue  # 只有一个选择，无法改进
                    
                    # 尝试每个可能的键类型
                    current_order = E[i, j].item()
                    best_order = current_order
                    best_score = current_score
                    
                    for order, _, _ in bonds:
                        if order == current_order:
                            continue
                        
                        # 检查原子价约束
                        # 计算当前原子价（考虑所有键，包括当前键）
                        edge_types = E.clone()
                        edge_types[edge_types == 4] = 1.5
                        edge_types[edge_types < 0] = 0
                        current_val_i = torch.sum(edge_types[i]).long().item()
                        current_val_j = torch.sum(edge_types[j]).long().item()
                        
                        # 计算改变键类型后的原子价
                        # 注意：需要先减去当前键的贡献，再加上新键的贡献
                        new_val_i = current_val_i - current_order + order
                        new_val_j = current_val_j - current_order + order
                        
                        # 检查是否超出最大价
                        if new_val_i > max_valencies[i] or new_val_j > max_valencies[j]:
                            continue
                        
                        # 尝试这个键类型
                        E[i, j] = order
                        E[j, i] = order
                        new_score = calculate_stability_score(E)
                        
                        if new_score > best_score:
                            best_score = new_score
                            best_order = order
                            improved = True
                        else:
                            # 恢复原来的键类型
                            E[i, j] = current_order
                            E[j, i] = current_order
                    
                    # 应用最佳选择（如果改进了）
                    if best_order != current_order:
                        E[i, j] = best_order
                        E[j, i] = best_order
                        A[i, j] = (best_order > 0)
                        A[j, i] = (best_order > 0)
                        current_score = best_score
                
                if not improved:
                    break  # 没有改进，收敛
            
            # 最终更新邻接矩阵
            A = (E > 0)
            
            # 更新邻接矩阵
            A = (E > 0)
        else:
            # 使用纯贪心算法（按相对偏差排序，选择最接近标准键长的键）
            candidate_bonds = []
            for (i, j), possible_bonds in possible_bonds_dict.items():
                if possible_bonds:
                    possible_bonds.sort(key=lambda x: x[2])  # 按相对偏差排序
                    best_order, _, _ = possible_bonds[0]
                    candidate_bonds.append((i, j, best_order))
            
            # 获取每个原子的最大价
            max_valencies = []
            for i in range(n):
                atom_str = atom_decoder[atom_types[i].item()]
                charge = charges[i].item() if charges is not None else 0
                possible_bonds_list = allowed_bonds[atom_str]
                if type(possible_bonds_list) == int:
                    max_val = possible_bonds_list
                elif type(possible_bonds_list) == dict:
                    expected_bonds = possible_bonds_list.get(charge, possible_bonds_list.get(0))
                    if type(expected_bonds) == int:
                        max_val = expected_bonds
                    else:
                        max_val = max(expected_bonds) if expected_bonds else 4
                else:
                    max_val = max(possible_bonds_list) if possible_bonds_list else 4
                max_valencies.append(max_val)
            
            # 按相对偏差排序候选键
            candidate_bonds_with_dev = []
            for i, j, order in candidate_bonds:
                possible_bonds = possible_bonds_dict[(i, j)]
                dev = next((d for o, _, d in possible_bonds if o == order), 1.0)
                candidate_bonds_with_dev.append((i, j, order, dev))
            candidate_bonds_with_dev.sort(key=lambda x: x[3])
            
            # 为每个原子跟踪已使用的价
            used_valencies = [0] * n
            
            # 选择键，确保每个原子的总价不超过最大价
            for i, j, order, _ in candidate_bonds_with_dev:
                if used_valencies[i] + order <= max_valencies[i] and used_valencies[j] + order <= max_valencies[j]:
                    E[i, j] = order
                    E[j, i] = order
                    A[i, j] = 1
                    A[j, i] = 1
                    used_valencies[i] += order
                    used_valencies[j] += order
    
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
        # 如果只有一个原子，它被认为是连通的（单个原子是一个完整的分子）
        is_connected = (n_atoms == 1)
        return n_atoms, is_connected
    
    # 使用DFS计算连通分量
    visited = torch.zeros(n_atoms, dtype=torch.bool)
    num_components = 0
    
    def dfs(node):
        """深度优先搜索"""
        visited[node] = True
        neighbors = torch.nonzero(bond_types[node] > 0, as_tuple=False).squeeze(-1)
        for neighbor in neighbors:
            neighbor_idx = neighbor.item()
            if neighbor_idx != node and not visited[neighbor_idx]:
                dfs(neighbor_idx)
    
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
            neighbor_idx = neighbor.item()
            if neighbor_idx != node and not visited[neighbor_idx]:
                dfs_label(neighbor_idx, comp_id)
    
    # 遍历所有未访问的节点
    for i in range(n_atoms):
        if not visited[i]:
            dfs_label(i, current_component)
            current_component += 1
    
    num_components = current_component
    return num_components, component_ids


def compute_missing_bond_deviations_strict(positions, atom_types, bond_types, atom_decoder, dataset_info, 
                                            strict_margin1, strict_margin2, strict_margin3):
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
        strict_margin1: 严格margin1值（pm单位）
        strict_margin2: 严格margin2值（pm单位）
        strict_margin3: 严格margin3值（pm单位）
    
    Returns:
        missing_bonds: List of dicts with keys: pair, actual_dist, standard_dist, deviation_pct, 
                      expected_order, actual_order, is_type_mismatch
    """
    n = positions.shape[0]
    dists = torch.cdist(positions, positions, p=2)
    missing_bonds = []
    
    # 计算连通分量，用于检测跨分量的缺失键
    _, component_ids = check_connectivity_with_labels(bond_types)
    
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


def _analyze_bonds_with_standard(positions, atom_types, atom_decoder, dataset_info,
                                  margin1_val, margin2_val, margin3_val):
    """
    使用指定标准分析单个分子的键和连通性
    
    Args:
        positions: [N, 3] 原子坐标
        atom_types: [N] 原子类型索引
        atom_decoder: 原子解码器列表
        dataset_info: 数据集信息字典
        margin1_val: margin1值（pm单位）
        margin2_val: margin2值（pm单位）
        margin3_val: margin3值（pm单位）
    
    Returns:
        tuple: (num_components, is_connected, missing_bond_deviations)
    """
    # 构建键矩阵
    _, _, bond_types = build_xae_molecule(
        positions=positions,
        atom_types=atom_types,
        dataset_info=dataset_info,
        atom_decoder=atom_decoder,
        margin1_val=margin1_val,
        margin2_val=margin2_val,
        margin3_val=margin3_val
    )
    
    # 计算连通性
    num_components, is_connected = check_connectivity(bond_types)
    
    # 计算缺失键偏差
    missing_bonds = compute_missing_bond_deviations_strict(
        positions, atom_types, bond_types, atom_decoder, dataset_info,
        strict_margin1=margin1_val,
        strict_margin2=margin2_val,
        strict_margin3=margin3_val
    )
    
    missing_deviations = [bond['deviation_pct'] for bond in missing_bonds] if missing_bonds else []
    
    return num_components, is_connected, missing_deviations


def _print_standard_results(standard_name, margin1, margin2, margin3,
                           num_components, is_connected, missing_deviations):
    """
    打印单个标准的分析结果
    
    Args:
        standard_name: 标准名称（如 '严格标准'）
        margin1/2/3: margin值
        num_components: 连通分量数数组
        is_connected: 连通性布尔数组
        missing_deviations: 缺失键偏差列表
    """
    print(f"\n🔗 {standard_name} (margin1={margin1}pm, margin2={margin2}pm, margin3={margin3}pm):")
    print(f"  连通性:")
    print(f"    连通分子数: {is_connected.sum()}")
    print(f"    非连通分子数: {(~is_connected).sum()}")
    print(f"    连通分子比例: {is_connected.sum() / len(is_connected) * 100:.2f}%")
    print(f"    平均连通分量数: {num_components.mean():.2f}")
    print(f"    最大连通分量数: {num_components.max()}")
    if len(missing_deviations) > 0:
        print(f"  缺失键偏差:")
        print(f"    总缺失键数: {len(missing_deviations)}")
        print(f"    平均偏差: {np.mean(missing_deviations):.4f}%")
        print(f"    中位数偏差: {np.median(missing_deviations):.4f}%")
    else:
        print(f"  缺失键偏差: 无缺失键")


def analyze_bonds(molecule_dir,
                 strict_margin1, strict_margin2, strict_margin3,
                 medium_margin1, medium_margin2, medium_margin3,
                 relaxed_margin1, relaxed_margin2, relaxed_margin3,
                 output_dir=None,
                 use_sdf_bonds=True):
    """
    分析分子的键和连通性
    
    Args:
        molecule_dir: 包含 .npz 文件的目录
        output_dir: 输出目录（可选）
        strict_margin1/2/3: 严格标准的margin值（pm单位）
        medium_margin1/2/3: 中等标准的margin值（pm单位）
        relaxed_margin1/2/3: 宽松标准的margin值（pm单位）
        use_sdf_bonds: 是否使用 SDF 文件中的键信息（默认 True）
                      - True: 使用 SDF 文件中的键（如果存在），只分析一次
                      - False: 使用三种不同的 margin 值重新构建键并分析
    
    Returns:
        dict: 包含键和连通性分析结果的字典
    """
    from funcmol.evaluation.quality_evaluation import _extract_bonds_from_sdf
    import re
    
    molecule_dir = Path(molecule_dir)
    npz_files = sorted(molecule_dir.glob("generated_*.npz"))
    
    if use_sdf_bonds:
        print(f"找到 {len(npz_files)} 个 .npz 分子文件（将优先使用 SDF 文件中的键信息）")
    else:
        print(f"找到 {len(npz_files)} 个 .npz 分子文件（使用三种标准分析）")
    
    atom_decoder = atom_decoder_dict['qm9_with_h']
    dataset_info = {'name': 'qm9'}
    
    # 存储键长统计数据（使用relaxed margin构建的键矩阵，用于键长统计）
    bond_lengths = []
    
    # 三种标准的缺失键偏差统计
    all_missing_bond_deviations_strict = []
    all_missing_bond_deviations_medium = []
    all_missing_bond_deviations_relaxed = []
    
    # 三种标准的连通性统计
    num_components_strict = []
    is_connected_strict = []
    num_components_medium = []
    is_connected_medium = []
    num_components_relaxed = []
    is_connected_relaxed = []
    
    if use_sdf_bonds:
        print("分析分子键和连通性（使用 SDF 文件中的键）...")
    else:
        print("分析分子键和连通性（使用三种标准）...")
    
    sdf_count = 0
    fallback_count = 0
    
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
            
            distances = torch.cdist(positions, positions, p=2)
            
            if use_sdf_bonds:
                # 优先尝试从 SDF 文件读取键信息
                bond_types_sdf = None
                npz_stem = npz_file.stem
                match = re.search(r'generated_(\d+)_tanh', npz_stem)
                if match:
                    index = match.group(1)
                    sdf_file = molecule_dir / f"genmol_{index}.sdf"
                    
                    if sdf_file.exists():
                        try:
                            with open(sdf_file, 'r', encoding='utf-8') as f:
                                sdf_string = f.read()
                            bond_types_sdf = _extract_bonds_from_sdf(sdf_string, len(atom_types))
                            sdf_count += 1
                        except Exception:
                            bond_types_sdf = None
                
                # 如果找到 SDF 键，使用它进行分析
                if bond_types_sdf is not None:
                    # 计算连通性
                    num_components, is_connected = check_connectivity(bond_types_sdf)
                    num_components_strict.append(num_components)
                    is_connected_strict.append(is_connected)
                    # 使用 relaxed margin 计算缺失键偏差（用于统计）
                    missing_bonds = compute_missing_bond_deviations_strict(
                        positions, atom_types, bond_types_sdf, atom_decoder, dataset_info,
                        strict_margin1=relaxed_margin1,
                        strict_margin2=relaxed_margin2,
                        strict_margin3=relaxed_margin3
                    )
                    missing_deviations = [bond['deviation_pct'] for bond in missing_bonds] if missing_bonds else []
                    all_missing_bond_deviations_strict.extend(missing_deviations)
                    # 三种标准都使用相同的结果
                    num_components_medium.append(num_components)
                    is_connected_medium.append(is_connected)
                    all_missing_bond_deviations_medium.extend(missing_deviations)
                    num_components_relaxed.append(num_components)
                    is_connected_relaxed.append(is_connected)
                    all_missing_bond_deviations_relaxed.extend(missing_deviations)
                    
                    # 计算键长
                    triu_mask = torch.triu(torch.ones_like(bond_types_sdf, dtype=torch.bool), diagonal=1)
                    bond_mask = (bond_types_sdf > 0) & triu_mask
                    if bond_mask.any():
                        bond_distances = distances[bond_mask]
                        bond_lengths.extend(bond_distances.cpu().numpy())
                else:
                    # 回退到距离方法
                    fallback_count += 1
                    num_comp_strict, is_conn_strict, missing_devs_strict = _analyze_bonds_with_standard(
                        positions, atom_types, atom_decoder, dataset_info,
                        strict_margin1, strict_margin2, strict_margin3
                    )
                    num_components_strict.append(num_comp_strict)
                    is_connected_strict.append(is_conn_strict)
                    all_missing_bond_deviations_strict.extend(missing_devs_strict)
                    num_components_medium.append(num_comp_strict)
                    is_connected_medium.append(is_conn_strict)
                    all_missing_bond_deviations_medium.extend(missing_devs_strict)
                    num_components_relaxed.append(num_comp_strict)
                    is_connected_relaxed.append(is_conn_strict)
                    all_missing_bond_deviations_relaxed.extend(missing_devs_strict)
                    
                    _, _, bond_types_fallback = build_xae_molecule(
                        positions=positions,
                        atom_types=atom_types,
                        dataset_info=dataset_info,
                        atom_decoder=atom_decoder,
                        margin1_val=relaxed_margin1,
                        margin2_val=relaxed_margin2,
                        margin3_val=relaxed_margin3
                    )
                    triu_mask = torch.triu(torch.ones_like(bond_types_fallback, dtype=torch.bool), diagonal=1)
                    bond_mask = (bond_types_fallback > 0) & triu_mask
                    if bond_mask.any():
                        bond_distances = distances[bond_mask]
                        bond_lengths.extend(bond_distances.cpu().numpy())
            else:
                # 使用三种标准分析
                # 严格标准：分析键和连通性
                num_comp_strict, is_conn_strict, missing_devs_strict = _analyze_bonds_with_standard(
                    positions, atom_types, atom_decoder, dataset_info,
                    strict_margin1, strict_margin2, strict_margin3
                )
                num_components_strict.append(num_comp_strict)
                is_connected_strict.append(is_conn_strict)
                all_missing_bond_deviations_strict.extend(missing_devs_strict)
                
                # 中等标准：分析键和连通性
                num_comp_medium, is_conn_medium, missing_devs_medium = _analyze_bonds_with_standard(
                    positions, atom_types, atom_decoder, dataset_info,
                    medium_margin1, medium_margin2, medium_margin3
                )
                num_components_medium.append(num_comp_medium)
                is_connected_medium.append(is_conn_medium)
                all_missing_bond_deviations_medium.extend(missing_devs_medium)
                
                # 宽松标准：分析键和连通性
                num_comp_relaxed, is_conn_relaxed, missing_devs_relaxed = _analyze_bonds_with_standard(
                    positions, atom_types, atom_decoder, dataset_info,
                    relaxed_margin1, relaxed_margin2, relaxed_margin3
                )
                num_components_relaxed.append(num_comp_relaxed)
                is_connected_relaxed.append(is_conn_relaxed)
                all_missing_bond_deviations_relaxed.extend(missing_devs_relaxed)
                
                # 计算键长（使用relaxed标准构建的键矩阵，用于键长统计）
                _, _, bond_types_relaxed = build_xae_molecule(
                    positions=positions,
                    atom_types=atom_types,
                    dataset_info=dataset_info,
                    atom_decoder=atom_decoder,
                    margin1_val=relaxed_margin1,
                    margin2_val=relaxed_margin2,
                    margin3_val=relaxed_margin3
                )
                triu_mask = torch.triu(torch.ones_like(bond_types_relaxed, dtype=torch.bool), diagonal=1)
                bond_mask = (bond_types_relaxed > 0) & triu_mask
                
                if bond_mask.any():
                    bond_distances = distances[bond_mask]
                    bond_lengths.extend(bond_distances.cpu().numpy())
            
        except Exception as e:
            print(f"\n处理文件 {npz_file} 时出错: {e}")
            continue
    
    # 转换为numpy数组
    bond_lengths = np.array(bond_lengths) if bond_lengths else np.array([])
    
    num_components_strict = np.array(num_components_strict)
    is_connected_strict = np.array(is_connected_strict)
    num_components_medium = np.array(num_components_medium)
    is_connected_medium = np.array(is_connected_medium)
    num_components_relaxed = np.array(num_components_relaxed)
    is_connected_relaxed = np.array(is_connected_relaxed)
    
    # 三种标准的缺失键偏差数组
    all_missing_bond_deviations_strict = np.array(all_missing_bond_deviations_strict) if all_missing_bond_deviations_strict else np.array([])
    all_missing_bond_deviations_medium = np.array(all_missing_bond_deviations_medium) if all_missing_bond_deviations_medium else np.array([])
    all_missing_bond_deviations_relaxed = np.array(all_missing_bond_deviations_relaxed) if all_missing_bond_deviations_relaxed else np.array([])
    
    # 打印统计结果
    print("\n" + "="*60)
    print("键和连通性分析结果")
    print("="*60)
    
    if use_sdf_bonds:
        print(f"✅ 使用 SDF 文件中的键信息（来自 OpenBabel）")
        print(f"  - 从 SDF 文件读取键信息: {sdf_count} 个")
        print(f"  - 使用距离方法推断键: {fallback_count} 个")
        # 使用 SDF 时，三种标准结果相同，只打印一次
        _print_standard_results("SDF 键（来自 OpenBabel）", relaxed_margin1, relaxed_margin2, relaxed_margin3,
                               num_components_strict, is_connected_strict, all_missing_bond_deviations_strict)
    else:
        # 打印三种标准的结果
        _print_standard_results("严格标准", strict_margin1, strict_margin2, strict_margin3,
                               num_components_strict, is_connected_strict, all_missing_bond_deviations_strict)
        _print_standard_results("中等标准", medium_margin1, medium_margin2, medium_margin3,
                               num_components_medium, is_connected_medium, all_missing_bond_deviations_medium)
        _print_standard_results("宽松标准", relaxed_margin1, relaxed_margin2, relaxed_margin3,
                               num_components_relaxed, is_connected_relaxed, all_missing_bond_deviations_relaxed)
    
    # 键长统计（使用relaxed标准）
    print(f"\n🔗 键长统计（使用relaxed标准）:")
    if len(bond_lengths) > 0:
        print(f"  总键数: {len(bond_lengths)}")
        print(f"  平均键长: {bond_lengths.mean():.4f} Å")
        print(f"  中位数键长: {np.median(bond_lengths):.4f} Å")
        print(f"  键长范围: {bond_lengths.min():.4f} - {bond_lengths.max():.4f} Å")
    else:
        print("  没有检测到任何键！")
    
    print(f"\n说明: 缺失键是指根据原子类型和距离判断应该形成键，但实际键矩阵中未识别出的原子对")
    
    return {
        'bond_lengths': bond_lengths,
        'strict': {
            'num_components': num_components_strict,
            'is_connected': is_connected_strict,
            'missing_bond_deviations': all_missing_bond_deviations_strict
        },
        'medium': {
            'num_components': num_components_medium,
            'is_connected': is_connected_medium,
            'missing_bond_deviations': all_missing_bond_deviations_medium
        },
        'relaxed': {
            'num_components': num_components_relaxed,
            'is_connected': is_connected_relaxed,
            'missing_bond_deviations': all_missing_bond_deviations_relaxed
        }
    }
