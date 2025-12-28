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


def build_xae_molecule(positions, atom_types, dataset_info, atom_decoder, 
                       margin1_val=None, margin2_val=None, margin3_val=None):
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
                    dists[i, j],
                    margin1_val=margin1_val,
                    margin2_val=margin2_val,
                    margin3_val=margin3_val
                )
            elif dataset_info['name'] == 'geom':
                # 对于geom数据集，使用limit_bonds_to_one
                order = get_bond_order(
                    atom_decoder[pair[0]], 
                    atom_decoder[pair[1]], 
                    dists[i, j],
                    check_exists=True,
                    margin1_val=margin1_val,
                    margin2_val=margin2_val,
                    margin3_val=margin3_val
                )
                if order > 1:
                    order = 1  # 限制为单键
            
            if order > 0:
                A[i, j] = 1
                A[j, i] = 1  # 确保邻接矩阵对称
                E[i, j] = order
                E[j, i] = order  # 确保键类型矩阵对称
    
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
                 output_dir=None):
    """
    分析分子的键和连通性（使用三种标准：strict, medium, relaxed）
    
    Args:
        molecule_dir: 包含 .npz 文件的目录
        output_dir: 输出目录（可选）
        strict_margin1/2/3: 严格标准的margin值（pm单位）
        medium_margin1/2/3: 中等标准的margin值（pm单位）
        relaxed_margin1/2/3: 宽松标准的margin值（pm单位）
    
    Returns:
        dict: 包含键和连通性分析结果的字典（包含三种标准的结果）
    """
    molecule_dir = Path(molecule_dir)
    npz_files = sorted(molecule_dir.glob("generated_*.npz"))
    
    print(f"找到 {len(npz_files)} 个 .npz 分子文件")
    
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
    
    print("分析分子键和连通性（使用三种标准）...")
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

