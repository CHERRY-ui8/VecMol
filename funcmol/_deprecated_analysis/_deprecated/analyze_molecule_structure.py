"""
分析生成分子的结构问题：
1. 最近原子距离分布
2. 分子连通性（是否形成完整graph）
3. 键长分布
"""

import sys
from pathlib import Path
import torch
import numpy as np
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from funcmol.analysis.baselines_evaluation import (
    atom_decoder_dict,
    build_xae_molecule,
    bonds1,
    margin1
)


def compute_min_distances(coords):
    """
    计算分子中所有原子对的最小距离
    
    Args:
        coords: [N, 3] 原子坐标
        
    Returns:
        min_distances: 每个原子到最近邻原子的距离 [N]
        all_distances: 所有原子对的距离（上三角矩阵）
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


def check_connectivity(bond_types):
    """
    检查分子的连通性（使用简单的DFS）
    
    Args:
        bond_types: [N, N] 键类型矩阵
        
    Returns:
        num_components: 连通分量数
        is_connected: 是否连通（连通分量数==1）
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
        # 找到所有与当前节点相连的节点
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


def compute_missing_bond_deviations(positions, atom_types, bond_types, atom_decoder, dataset_info, margin1_val=40):
    """
    计算应该形成键但未形成键的原子对的距离偏差
    
    Args:
        positions: [N, 3] 原子坐标
        atom_types: [N] 原子类型索引
        bond_types: [N, N] 键类型矩阵
        atom_decoder: 原子类型解码器列表
        dataset_info: 数据集信息字典
        margin1_val: margin1 值（pm单位），
    
    Returns:
        missing_bonds: List of dicts with keys: pair, actual_dist, standard_dist, deviation_pct
    """
    n = positions.shape[0]
    dists = torch.cdist(positions, positions, p=2)
    missing_bonds = []
    
    for i in range(n):
        for j in range(i):
            if bond_types[i, j] == 0:  # 当前判断为无键
                atom1_str = atom_decoder[atom_types[i].item()]
                atom2_str = atom_decoder[atom_types[j].item()]
                
                # 检查标准键长（需要按字母顺序排序）
                pair = sorted([atom1_str, atom2_str])
                atom1_key = pair[0]
                atom2_key = pair[1]
                
                # 检查标准键长
                if atom1_key in bonds1 and atom2_key in bonds1[atom1_key]:
                    standard_dist_pm = bonds1[atom1_key][atom2_key]
                    standard_dist = standard_dist_pm / 100.0  # 转换为Å
                    threshold = (standard_dist_pm + margin1_val) / 100.0  # 转换为Å
                    actual_dist = dists[i, j].item()
                    
                    # 如果距离在应该形成键的范围内
                    if actual_dist < threshold:
                        deviation_pct = (actual_dist - standard_dist) / standard_dist * 100
                        missing_bonds.append({
                            'pair': (i, j),
                            'actual_dist': actual_dist,
                            'standard_dist': standard_dist,
                            'deviation_pct': deviation_pct,
                            'atom1': atom1_str,
                            'atom2': atom2_str
                        })
    
    return missing_bonds


def compute_missing_bond_deviations_strict(positions, atom_types, bond_types, atom_decoder, dataset_info, strict_margin=15):
    """
    计算应该形成键但未形成键的原子对的距离偏差（使用严格标准）
    
    检查两种情况：
    1. 距离在严格阈值内，但没有键（bond_types[i, j] == 0）
    2. 距离在严格阈值内，有键，但属于不同的连通分量（跨分量断裂）
    
    Args:
        positions: [N, 3] 原子坐标
        atom_types: [N] 原子类型索引
        bond_types: [N, N] 键类型矩阵
        atom_decoder: 原子类型解码器列表
        dataset_info: 数据集信息字典
        strict_margin: 严格margin值（pm单位），默认15pm
    
    Returns:
        missing_bonds: List of dicts with keys: pair, actual_dist, standard_dist, deviation_pct
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
            
            # 检查标准键长
            if atom1_key in bonds1 and atom2_key in bonds1[atom1_key]:
                standard_dist_pm = bonds1[atom1_key][atom2_key]
                standard_dist = standard_dist_pm / 100.0  # 转换为Å
                strict_threshold = (standard_dist_pm + strict_margin) / 100.0  # 严格阈值
                # 使用更宽松的阈值来检测偏离标准值的键（标准键长 + 30pm）
                relaxed_threshold = (standard_dist_pm + 30) / 100.0
                actual_dist = dists[i, j].item()
                
                # 检查两种情况：
                # 1. 距离在严格阈值内（应该形成键）
                # 2. 距离在宽松阈值内但有键，且偏离标准值超过5%（结构不理想）
                is_cross_component = (component_ids[i] != component_ids[j])
                has_bond = (bond_types[i, j] > 0)
                deviation_pct = (actual_dist - standard_dist) / standard_dist * 100
                significant_deviation = abs(deviation_pct) > 5.0
                
                # 情况1：距离在严格阈值内，应该形成键
                if actual_dist < strict_threshold:
                    if bond_types[i, j] == 0 or is_cross_component:
                        missing_bonds.append({
                            'pair': (i, j),
                            'actual_dist': actual_dist,
                            'standard_dist': standard_dist,
                            'deviation_pct': deviation_pct,
                            'atom1': atom1_str,
                            'atom2': atom2_str,
                            'is_cross_component': is_cross_component,
                            'has_bond': has_bond
                        })
                # 情况2：距离在宽松阈值内，有键但偏离标准值（结构不理想）
                elif actual_dist < relaxed_threshold and has_bond and significant_deviation:
                    missing_bonds.append({
                        'pair': (i, j),
                        'actual_dist': actual_dist,
                        'standard_dist': standard_dist,
                        'deviation_pct': deviation_pct,
                        'atom1': atom1_str,
                        'atom2': atom2_str,
                        'is_cross_component': is_cross_component,
                        'has_bond': has_bond
                    })
    
    return missing_bonds


def check_connectivity_with_labels(bond_types):
    """
    检查分子的连通性并返回每个原子所属的连通分量ID
    
    Args:
        bond_types: [N, N] 键类型矩阵
        
    Returns:
        num_components: 连通分量数
        component_ids: 每个原子所属的连通分量ID [N]
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


def compute_excessive_bond_deviations(positions, atom_types, bond_types, atom_decoder, dataset_info, margin1_val=40):
    """
    计算不应该形成键但形成键的原子对（距离过远但仍被判断为有键）
    
    Args:
        positions: [N, 3] 原子坐标
        atom_types: [N] 原子类型索引
        bond_types: [N, N] 键类型矩阵
        atom_decoder: 原子类型解码器列表
        dataset_info: 数据集信息字典
        margin1_val: margin1 值（pm单位），
    
    Returns:
        excessive_bonds: List of dicts with keys: pair, actual_dist, standard_dist, deviation_pct
    """
    n = positions.shape[0]
    dists = torch.cdist(positions, positions, p=2)
    excessive_bonds = []
    
    for i in range(n):
        for j in range(i):
            if bond_types[i, j] > 0:  # 当前判断为有键
                atom1_str = atom_decoder[atom_types[i].item()]
                atom2_str = atom_decoder[atom_types[j].item()]
                
                # 检查标准键长（需要按字母顺序排序）
                pair = sorted([atom1_str, atom2_str])
                atom1_key = pair[0]
                atom2_key = pair[1]
                
                # 检查标准键长
                if atom1_key in bonds1 and atom2_key in bonds1[atom1_key]:
                    standard_dist_pm = bonds1[atom1_key][atom2_key]
                    standard_dist = standard_dist_pm / 100.0  # 转换为Å
                    threshold = (standard_dist_pm + margin1_val) / 100.0  # 转换为Å
                    actual_dist = dists[i, j].item()
                    
                    # 如果距离超过应该形成键的范围
                    if actual_dist > threshold:
                        deviation_pct = (actual_dist - standard_dist) / standard_dist * 100
                        excessive_bonds.append({
                            'pair': (i, j),
                            'actual_dist': actual_dist,
                            'standard_dist': standard_dist,
                            'deviation_pct': deviation_pct,
                            'atom1': atom1_str,
                            'atom2': atom2_str,
                            'bond_order': bond_types[i, j].item()
                        })
    
    return excessive_bonds


def compute_connectivity_continuity_score(positions, atom_types, bond_types, atom_decoder, dataset_info, margin1_val=40):
    """
    计算综合连续性连通性分数
    
    使用更严格的标准（标准键长 + 15pm）来判断"应该形成键"，
    这样可以检测出即使使用宽松margin判断为有键，但距离仍然偏离标准值的情况。
    
    Args:
        positions: [N, 3] 原子坐标
        atom_types: [N] 原子类型索引
        bond_types: [N, N] 键类型矩阵
        atom_decoder: 原子类型解码器列表
        dataset_info: 数据集信息字典
        margin1_val: margin1 值（pm单位），用于键判断，但连续性评估使用更严格标准
    
    Returns:
        dict: 包含连续性指标的字典
    """
    n = positions.shape[0]
    dists = torch.cdist(positions, positions, p=2)
    
    # 使用更严格的标准来判断"应该形成键"：标准键长 + 15pm
    # 这样可以检测出距离偏离标准值的情况，即使它们被宽松的margin判断为有键
    strict_margin = 15  # 使用较小的margin来定义"应该形成键"的标准
    
    missing_bonds = []
    all_potential_bonds = []  # 所有应该形成键的原子对（无论是否已形成键）
    
    for i in range(n):
        for j in range(i):
            atom1_str = atom_decoder[atom_types[i].item()]
            atom2_str = atom_decoder[atom_types[j].item()]
            pair = sorted([atom1_str, atom2_str])
            atom1_key = pair[0]
            atom2_key = pair[1]
            
            if atom1_key in bonds1 and atom2_key in bonds1[atom1_key]:
                standard_dist_pm = bonds1[atom1_key][atom2_key]
                standard_dist = standard_dist_pm / 100.0  # 转换为Å
                strict_threshold = (standard_dist_pm + strict_margin) / 100.0  # 严格阈值
                actual_dist = dists[i, j].item()
                
                # 如果距离在"应该形成键"的范围内（使用严格标准）
                if actual_dist < strict_threshold:
                    deviation_pct = (actual_dist - standard_dist) / standard_dist * 100
                    all_potential_bonds.append({
                        'pair': (i, j),
                        'actual_dist': actual_dist,
                        'standard_dist': standard_dist,
                        'deviation_pct': deviation_pct,
                        'has_bond': bond_types[i, j] > 0
                    })
                    
                    # 如果当前判断为无键，记录为缺失键
                    if bond_types[i, j] == 0:
                        missing_bonds.append({
                            'pair': (i, j),
                            'actual_dist': actual_dist,
                            'standard_dist': standard_dist,
                            'deviation_pct': deviation_pct,
                            'atom1': atom1_str,
                            'atom2': atom2_str
                        })
    
    if len(all_potential_bonds) == 0:
        return {
            'mean_deviation_pct': 0.0,
            'max_deviation_pct': 0.0,
            'missing_bond_count': 0,
            'missing_bond_ratio': 0.0,
            'continuity_score': 1.0,
            'overall_mean_deviation_pct': 0.0
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
    
    # 计算连续性分数（综合考虑所有应该形成键的原子对的偏差）
    # 使用整体平均偏差的归一化版本
    # 假设最大合理偏差为30%，超过30%认为严重偏离
    normalized_deviation = min(abs(overall_mean_deviation) / 30.0, 1.0)
    continuity_score = 1.0 - normalized_deviation
    
    return {
        'mean_deviation_pct': mean_deviation,
        'max_deviation_pct': max_deviation,
        'missing_bond_count': len(missing_bonds),
        'missing_bond_ratio': missing_bond_ratio,
        'continuity_score': continuity_score,
        'total_potential_bonds': len(all_potential_bonds),
        'overall_mean_deviation_pct': overall_mean_deviation,
        'overall_max_deviation_pct': overall_max_deviation
    }


def analyze_molecules(molecule_dir, output_dir=None):
    """
    分析分子结构
    
    Args:
        molecule_dir: 包含 .npz 文件的目录
        output_dir: 输出目录（可选）
    """
    molecule_dir = Path(molecule_dir)
    npz_files = sorted(molecule_dir.glob("generated_*.npz"))
    
    print(f"找到 {len(npz_files)} 个 .npz 分子文件")
    
    atom_decoder = atom_decoder_dict['qm9_with_h']
    dataset_info = {'name': 'qm9'}
    
    # 存储统计数据
    all_min_distances = []
    all_pair_distances = []
    num_components_list = []
    is_connected_list = []
    bond_lengths = []
    num_atoms_list = []
    
    # 空间分布统计
    coord_ranges = []  # 每个分子的坐标范围 (max - min)
    coord_centers = []  # 每个分子的中心坐标
    coord_spans = []  # 每个分子的跨度（最大距离）
    large_gap_ratios = []  # 每个分子中距离>3Å的原子对比例
    
    # 连续性指标统计
    all_missing_bond_deviations = []  # 所有缺失键的偏差百分比
    all_continuity_scores = []  # 所有分子的连续性分数
    all_missing_bond_ratios = []  # 所有分子的缺失键比例
    all_mean_deviations = []  # 所有分子的平均偏差（仅缺失键）
    all_max_deviations = []  # 所有分子的最大偏差（仅缺失键）
    all_overall_mean_deviations = []  # 所有分子的整体平均偏差（所有应该形成键的原子对）
    all_overall_max_deviations = []  # 所有分子的整体最大偏差
    
    print("分析分子结构...")
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
            num_atoms = len(positions)
            num_atoms_list.append(num_atoms)
            
            # 计算空间分布统计
            coord_min = positions.min(dim=0)[0]
            coord_max = positions.max(dim=0)[0]
            coord_range = coord_max - coord_min  # [3]
            coord_center = positions.mean(dim=0)  # [3]
            coord_span = torch.cdist(positions, positions, p=2).max().item()  # 最大原子对距离
            
            coord_ranges.append(coord_range.cpu().numpy())
            coord_centers.append(coord_center.cpu().numpy())
            coord_spans.append(coord_span)
            
            # 计算最近距离
            min_dists, pair_dists = compute_min_distances(positions)
            if len(min_dists) > 0:
                all_min_distances.extend(min_dists.cpu().numpy())
            if len(pair_dists) > 0:
                all_pair_distances.extend(pair_dists.cpu().numpy())
                # 计算距离>3Å的原子对比例
                large_gap_ratio = (pair_dists > 3.0).sum().item() / len(pair_dists) * 100
                large_gap_ratios.append(large_gap_ratio)
            
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
            
            # 计算连续性指标（内部使用严格标准15pm来判断"应该形成键"）
            continuity_metrics = compute_connectivity_continuity_score(
                positions, atom_types, bond_types, atom_decoder, dataset_info, margin1_val=margin1
            )
            all_continuity_scores.append(continuity_metrics['continuity_score'])
            all_missing_bond_ratios.append(continuity_metrics['missing_bond_ratio'])
            all_mean_deviations.append(continuity_metrics['mean_deviation_pct'])
            all_max_deviations.append(continuity_metrics['max_deviation_pct'])
            all_overall_mean_deviations.append(continuity_metrics['overall_mean_deviation_pct'])
            all_overall_max_deviations.append(continuity_metrics['overall_max_deviation_pct'])
            
            # 重新计算缺失键（使用与连续性评估相同的严格标准15pm）
            # 这样可以确保数据一致性
            missing_bonds = compute_missing_bond_deviations_strict(
                positions, atom_types, bond_types, atom_decoder, dataset_info, strict_margin=15
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
    all_min_distances = np.array(all_min_distances)
    all_pair_distances = np.array(all_pair_distances)
    num_components_list = np.array(num_components_list)
    is_connected_list = np.array(is_connected_list)
    bond_lengths = np.array(bond_lengths) if bond_lengths else np.array([])
    num_atoms_list = np.array(num_atoms_list)
    coord_ranges = np.array(coord_ranges)  # [N_mols, 3]
    coord_centers = np.array(coord_centers)  # [N_mols, 3]
    coord_spans = np.array(coord_spans)
    large_gap_ratios = np.array(large_gap_ratios) if large_gap_ratios else np.array([])
    
    # 连续性指标数组
    all_continuity_scores = np.array(all_continuity_scores) if all_continuity_scores else np.array([])
    all_missing_bond_ratios = np.array(all_missing_bond_ratios) if all_missing_bond_ratios else np.array([])
    all_mean_deviations = np.array(all_mean_deviations) if all_mean_deviations else np.array([])
    all_max_deviations = np.array(all_max_deviations) if all_max_deviations else np.array([])
    all_missing_bond_deviations = np.array(all_missing_bond_deviations) if all_missing_bond_deviations else np.array([])
    all_overall_mean_deviations = np.array(all_overall_mean_deviations) if all_overall_mean_deviations else np.array([])
    all_overall_max_deviations = np.array(all_overall_max_deviations) if all_overall_max_deviations else np.array([])
    
    # 打印统计结果
    print("\n" + "="*60)
    print("分子结构分析结果")
    print("="*60)
    
    print(f"\n📊 基本统计:")
    print(f"  总分子数: {len(npz_files)}")
    print(f"  平均原子数: {num_atoms_list.mean():.2f}")
    print(f"  原子数范围: {num_atoms_list.min()} - {num_atoms_list.max()}")
    
    print(f"\n📏 最近原子距离统计:")
    if len(all_min_distances) > 0:
        print(f"  平均最近距离: {all_min_distances.mean():.4f} Å")
        print(f"  中位数最近距离: {np.median(all_min_distances):.4f} Å")
        print(f"  最小最近距离: {all_min_distances.min():.4f} Å")
        print(f"  最大最近距离: {all_min_distances.max():.4f} Å")
        print(f"  标准差: {all_min_distances.std():.4f} Å")
        
        # 统计距离过大的原子比例
        large_dist_threshold = 3.0  # 超过3Å认为距离过大
        large_dist_ratio = (all_min_distances > large_dist_threshold).sum() / len(all_min_distances) * 100
        print(f"  最近距离 > {large_dist_threshold}Å 的原子比例: {large_dist_ratio:.2f}%")
        
        # 统计距离过小的原子比例（可能是重叠）
        small_dist_threshold = 0.5  # 小于0.5Å认为距离过小
        small_dist_ratio = (all_min_distances < small_dist_threshold).sum() / len(all_min_distances) * 100
        print(f"  最近距离 < {small_dist_threshold}Å 的原子比例: {small_dist_ratio:.2f}%")
    
    print(f"\n🔗 键长统计:")
    if len(bond_lengths) > 0:
        print(f"  总键数: {len(bond_lengths)}")
        print(f"  平均键长: {bond_lengths.mean():.4f} Å")
        print(f"  中位数键长: {np.median(bond_lengths):.4f} Å")
        print(f"  键长范围: {bond_lengths.min():.4f} - {bond_lengths.max():.4f} Å")
        
        # 统计异常键长
        normal_bond_range = (0.7, 2.0)  # 正常键长范围
        normal_bond_ratio = ((bond_lengths >= normal_bond_range[0]) & 
                            (bond_lengths <= normal_bond_range[1])).sum() / len(bond_lengths) * 100
        print(f"  正常键长 ({normal_bond_range[0]}-{normal_bond_range[1]}Å) 比例: {normal_bond_ratio:.2f}%")
    else:
        print("  没有检测到任何键！")
    
    print(f"\n🌐 分子连通性统计:")
    print(f"  连通分子数（连通分量=1）: {is_connected_list.sum()}")
    print(f"  非连通分子数（连通分量>1）: {(~is_connected_list).sum()}")
    print(f"  连通分子比例: {is_connected_list.sum() / len(is_connected_list) * 100:.2f}%")
    print(f"  平均连通分量数: {num_components_list.mean():.2f}")
    print(f"  最大连通分量数: {num_components_list.max()}")
    print(f"  连通分量数分布:")
    unique, counts = np.unique(num_components_list, return_counts=True)
    for comp, count in zip(unique, counts):
        print(f"    {comp} 个连通分量: {count} 个分子 ({count/len(num_components_list)*100:.2f}%)")
    
    print(f"\n🔗 连续性连通性指标 (键判断使用 margin1={margin1}pm, 连续性评估使用严格标准: 标准键长+15pm):")
    if len(all_continuity_scores) > 0:
        print(f"  平均连续性分数: {all_continuity_scores.mean():.4f} (1.0=完美连通)")
        print(f"  中位数连续性分数: {np.median(all_continuity_scores):.4f}")
        print(f"  连续性分数范围: {all_continuity_scores.min():.4f} - {all_continuity_scores.max():.4f}")
        print(f"  连续性分数标准差: {all_continuity_scores.std():.4f}")
        print(f"  说明: 分数基于所有应该形成键的原子对的整体偏差计算（使用严格标准: 标准键长+15pm）")
        
        print(f"\n  整体偏差统计（所有应该形成键的原子对，无论是否已形成键）:")
        if len(all_overall_mean_deviations) > 0:
            print(f"    平均整体偏差百分比: {all_overall_mean_deviations.mean():.4f}%")
            print(f"    中位数整体偏差百分比: {np.median(all_overall_mean_deviations):.4f}%")
            print(f"    整体偏差范围: {all_overall_mean_deviations.min():.4f}% - {all_overall_mean_deviations.max():.4f}%")
            print(f"    整体偏差标准差: {all_overall_mean_deviations.std():.4f}%")
        
        if len(all_overall_max_deviations) > 0:
            print(f"    最大整体偏差百分比（平均）: {all_overall_max_deviations.mean():.4f}%")
            print(f"    最大整体偏差百分比（中位数）: {np.median(all_overall_max_deviations):.4f}%")
        
        print(f"\n  缺失键统计（应该形成键但未形成键的原子对）:")
        if len(all_missing_bond_ratios) > 0:
            print(f"    平均缺失键比例: {all_missing_bond_ratios.mean():.4f} ({all_missing_bond_ratios.mean()*100:.2f}%)")
            print(f"    中位数缺失键比例: {np.median(all_missing_bond_ratios):.4f} ({np.median(all_missing_bond_ratios)*100:.2f}%)")
            print(f"    缺失键比例范围: {all_missing_bond_ratios.min():.4f} - {all_missing_bond_ratios.max():.4f}")
        
        if len(all_mean_deviations) > 0:
            print(f"    缺失键平均偏差百分比: {all_mean_deviations.mean():.4f}%")
            print(f"    缺失键中位数偏差百分比: {np.median(all_mean_deviations):.4f}%")
            print(f"    缺失键平均偏差范围: {all_mean_deviations.min():.4f}% - {all_mean_deviations.max():.4f}%")
        
        if len(all_max_deviations) > 0:
            print(f"    缺失键最大偏差百分比（平均）: {all_max_deviations.mean():.4f}%")
            print(f"    缺失键最大偏差百分比（中位数）: {np.median(all_max_deviations):.4f}%")
            print(f"    缺失键最大偏差范围: {all_max_deviations.min():.4f}% - {all_max_deviations.max():.4f}%")
        
        if len(all_missing_bond_deviations) > 0:
            print(f"\n  所有缺失键的偏差分布:")
            print(f"    总缺失键数: {len(all_missing_bond_deviations)}")
            print(f"    平均偏差: {all_missing_bond_deviations.mean():.4f}%")
            print(f"    中位数偏差: {np.median(all_missing_bond_deviations):.4f}%")
            print(f"    偏差范围: {all_missing_bond_deviations.min():.4f}% - {all_missing_bond_deviations.max():.4f}%")
            print(f"    偏差标准差: {all_missing_bond_deviations.std():.4f}%")
        else:
            print(f"    注意: 使用严格标准(标准键长+15pm)未检测到缺失键")
    else:
        print("  没有计算连续性指标（可能所有分子都完全连通）")
    
    print(f"\n📐 所有原子对距离统计:")
    if len(all_pair_distances) > 0:
        print(f"  平均距离: {all_pair_distances.mean():.4f} Å")
        print(f"  中位数距离: {np.median(all_pair_distances):.4f} Å")
        print(f"  最小距离: {all_pair_distances.min():.4f} Å")
        print(f"  最大距离: {all_pair_distances.max():.4f} Å")
        
        # 统计可能形成键的距离（< 2.0Å）
        potential_bond_threshold = 2.0
        potential_bonds = (all_pair_distances < potential_bond_threshold).sum()
        print(f"  距离 < {potential_bond_threshold}Å 的原子对数量: {potential_bonds}")
        print(f"  可能形成键的原子对比例: {potential_bonds / len(all_pair_distances) * 100:.2f}%")
        
        # 统计距离过大的原子对
        large_dist_threshold = 3.0
        large_dists = (all_pair_distances > large_dist_threshold).sum()
        print(f"  距离 > {large_dist_threshold}Å 的原子对数量: {large_dists}")
        print(f"  距离过大的原子对比例: {large_dists / len(all_pair_distances) * 100:.2f}%")
    
    print(f"\n📦 分子空间分布统计:")
    if len(coord_ranges) > 0:
        print(f"  平均坐标范围 (X, Y, Z): ({coord_ranges[:, 0].mean():.2f}, {coord_ranges[:, 1].mean():.2f}, {coord_ranges[:, 2].mean():.2f}) Å")
        print(f"  最大坐标范围: {coord_ranges.max():.2f} Å")
        print(f"  平均分子跨度（最大原子对距离）: {coord_spans.mean():.2f} Å")
        print(f"  最大分子跨度: {coord_spans.max():.2f} Å")
        print(f"  中位数分子跨度: {np.median(coord_spans):.2f} Å")
        
        # 统计跨度过大的分子
        large_span_threshold = 10.0  # 超过10Å认为跨度过大
        large_span_count = (coord_spans > large_span_threshold).sum()
        print(f"  跨度 > {large_span_threshold}Å 的分子数: {large_span_count} ({large_span_count/len(coord_spans)*100:.2f}%)")
        
        if len(large_gap_ratios) > 0:
            print(f"  平均大距离原子对比例（>3Å）: {large_gap_ratios.mean():.2f}%")
            print(f"  中位数大距离原子对比例: {np.median(large_gap_ratios):.2f}%")
    
    # 生成可视化
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置字体
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 1. 最近距离分布 - 直方图 + KDE密度曲线
        if len(all_min_distances) > 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # 绘制直方图
            n, bins, patches = ax.hist(all_min_distances, bins=100, edgecolor='black', 
                                       alpha=0.7, density=False, color='steelblue', label='Histogram')
            
            # 添加KDE密度曲线
            kde = stats.gaussian_kde(all_min_distances)
            x_kde = np.linspace(all_min_distances.min(), all_min_distances.max(), 200)
            y_kde = kde(x_kde) * len(all_min_distances) * (bins[1] - bins[0])  # 转换为频率
            ax.plot(x_kde, y_kde, 'r-', linewidth=2, label='KDE Density')
            
            # 添加统计线
            ax.axvline(all_min_distances.mean(), color='red', linestyle='--', linewidth=2,
                      label=f'Mean: {all_min_distances.mean():.3f}Å')
            ax.axvline(np.median(all_min_distances), color='green', linestyle='--', linewidth=2,
                      label=f'Median: {np.median(all_min_distances):.3f}Å')
            
            ax.set_xlabel('Nearest Atom Distance (Å)', fontsize=12)
            ax.set_ylabel('Frequency', fontsize=12)
            ax.set_title('Nearest Atom Distance Distribution', fontsize=13, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            fig_path = output_dir / "nearest_atom_distance_distribution.png"
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            print(f"最近原子距离分布统计图已保存到: {fig_path}")
            plt.close()
        
        # 2. 综合结构分析图（包含最近距离、键长、连通性等）
        if len(all_min_distances) > 0:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # 最近距离直方图
            axes[0, 0].hist(all_min_distances, bins=100, edgecolor='black', alpha=0.7)
            axes[0, 0].axvline(all_min_distances.mean(), color='r', linestyle='--', 
                              label=f'Mean: {all_min_distances.mean():.3f}Å')
            axes[0, 0].axvline(np.median(all_min_distances), color='g', linestyle='--', 
                              label=f'Median: {np.median(all_min_distances):.3f}Å')
            axes[0, 0].set_xlabel('Nearest Atom Distance (Å)')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].set_title('Nearest Atom Distance Distribution')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # 最近距离累积分布
            sorted_dists = np.sort(all_min_distances)
            cumulative = np.arange(1, len(sorted_dists) + 1) / len(sorted_dists)
            axes[0, 1].plot(sorted_dists, cumulative, linewidth=2)
            axes[0, 1].axvline(3.0, color='r', linestyle='--', label='3.0Å threshold')
            axes[0, 1].set_xlabel('Nearest Atom Distance (Å)')
            axes[0, 1].set_ylabel('Cumulative Probability')
            axes[0, 1].set_title('Cumulative Distribution of Nearest Distances')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # 键长分布
            if len(bond_lengths) > 0:
                axes[1, 0].hist(bond_lengths, bins=100, edgecolor='black', alpha=0.7, color='orange')
                axes[1, 0].axvline(bond_lengths.mean(), color='r', linestyle='--', 
                                  label=f'Mean: {bond_lengths.mean():.3f}Å')
                axes[1, 0].axvline(np.median(bond_lengths), color='g', linestyle='--', 
                                  label=f'Median: {np.median(bond_lengths):.3f}Å')
                axes[1, 0].set_xlabel('Bond Length (Å)')
                axes[1, 0].set_ylabel('Frequency')
                axes[1, 0].set_title('Bond Length Distribution')
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)
            else:
                axes[1, 0].text(0.5, 0.5, 'No bonds detected', 
                              ha='center', va='center', transform=axes[1, 0].transAxes)
                axes[1, 0].set_title('Bond Length Distribution (No Data)')
            
            # 连通分量数分布
            unique_components, counts = np.unique(num_components_list, return_counts=True)
            axes[1, 1].bar(unique_components, counts, edgecolor='black', alpha=0.7, color='green')
            axes[1, 1].set_xlabel('Number of Connected Components')
            axes[1, 1].set_ylabel('Number of Molecules')
            axes[1, 1].set_title('Connected Components Distribution')
            axes[1, 1].grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            fig_path = output_dir / "molecule_structure_analysis.png"
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            print(f"综合结构分析图已保存到: {fig_path}")
            plt.close()
        
        # 3. 分子跨度分布
        if len(coord_spans) > 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(coord_spans, bins=50, edgecolor='black', alpha=0.7, color='red')
            ax.axvline(coord_spans.mean(), color='r', linestyle='--', 
                      label=f'Mean: {coord_spans.mean():.2f}Å')
            ax.axvline(np.median(coord_spans), color='g', linestyle='--', 
                      label=f'Median: {np.median(coord_spans):.2f}Å')
            ax.axvline(10.0, color='orange', linestyle='--', label='10.0Å threshold')
            ax.set_xlabel('Molecular Span (Max Atom Pair Distance, Å)')
            ax.set_ylabel('Number of Molecules')
            ax.set_title('Molecular Span Distribution (Reflecting Atom Dispersion)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            fig_path = output_dir / "molecule_span_distribution.png"
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            print(f"分子跨度分布图已保存到: {fig_path}")
            plt.close()
        
        # 2. 所有原子对距离分布（如果数据量不太大）
        if len(all_pair_distances) > 0 and len(all_pair_distances) < 1000000:
            fig, ax = plt.subplots(figsize=(10, 6))
            # 只显示合理范围的距离
            valid_dists = all_pair_distances[all_pair_distances < 10.0]
            ax.hist(valid_dists, bins=200, edgecolor='black', alpha=0.7, color='purple')
            ax.axvline(valid_dists.mean(), color='r', linestyle='--', 
                      label=f'Mean: {valid_dists.mean():.3f}Å')
            ax.axvline(2.0, color='orange', linestyle='--', label='2.0Å (potential bond)')
            ax.set_xlabel('Atom Pair Distance (Å)')
            ax.set_ylabel('Frequency')
            ax.set_title('Atom Pair Distance Distribution (< 10Å)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            fig_path = output_dir / "pairwise_distance_distribution.png"
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            print(f"原子对距离分布图已保存到: {fig_path}")
            plt.close()
        
        # 6. 连续性指标可视化
        if len(all_continuity_scores) > 0:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # 连续性分数分布
            axes[0, 0].hist(all_continuity_scores, bins=50, edgecolor='black', alpha=0.7, color='purple')
            axes[0, 0].axvline(all_continuity_scores.mean(), color='r', linestyle='--', 
                              label=f'Mean: {all_continuity_scores.mean():.3f}')
            axes[0, 0].axvline(np.median(all_continuity_scores), color='g', linestyle='--', 
                              label=f'Median: {np.median(all_continuity_scores):.3f}')
            axes[0, 0].set_xlabel('Connectivity Continuity Score', fontsize=12)
            axes[0, 0].set_ylabel('Number of Molecules', fontsize=12)
            axes[0, 0].set_title('Connectivity Continuity Score Distribution', fontsize=13, fontweight='bold')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # 缺失键偏差分布
            if len(all_missing_bond_deviations) > 0:
                axes[0, 1].hist(all_missing_bond_deviations, bins=50, edgecolor='black', alpha=0.7, color='orange')
                axes[0, 1].axvline(all_missing_bond_deviations.mean(), color='r', linestyle='--', 
                                  label=f'Mean: {all_missing_bond_deviations.mean():.2f}%')
                axes[0, 1].axvline(np.median(all_missing_bond_deviations), color='g', linestyle='--', 
                                  label=f'Median: {np.median(all_missing_bond_deviations):.2f}%')
                axes[0, 1].set_xlabel('Missing Bond Deviation (%)', fontsize=12)
                axes[0, 1].set_ylabel('Frequency', fontsize=12)
                axes[0, 1].set_title('Missing Bond Deviation Distribution', fontsize=13, fontweight='bold')
                axes[0, 1].legend()
                axes[0, 1].grid(True, alpha=0.3)
            else:
                axes[0, 1].text(0.5, 0.5, 'No missing bonds detected', 
                              ha='center', va='center', transform=axes[0, 1].transAxes)
                axes[0, 1].set_title('Missing Bond Deviation Distribution (No Data)')
            
            # 缺失键比例分布
            if len(all_missing_bond_ratios) > 0:
                axes[1, 0].hist(all_missing_bond_ratios * 100, bins=50, edgecolor='black', alpha=0.7, color='cyan')
                axes[1, 0].axvline(all_missing_bond_ratios.mean() * 100, color='r', linestyle='--', 
                                  label=f'Mean: {all_missing_bond_ratios.mean()*100:.2f}%')
                axes[1, 0].axvline(np.median(all_missing_bond_ratios) * 100, color='g', linestyle='--', 
                                  label=f'Median: {np.median(all_missing_bond_ratios)*100:.2f}%')
                axes[1, 0].set_xlabel('Missing Bond Ratio (%)', fontsize=12)
                axes[1, 0].set_ylabel('Number of Molecules', fontsize=12)
                axes[1, 0].set_title('Missing Bond Ratio Distribution', fontsize=13, fontweight='bold')
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)
            else:
                axes[1, 0].text(0.5, 0.5, 'No data', 
                              ha='center', va='center', transform=axes[1, 0].transAxes)
                axes[1, 0].set_title('Missing Bond Ratio Distribution (No Data)')
            
            # 连续性分数 vs 碎片数散点图
            if len(num_components_list) == len(all_continuity_scores):
                scatter = axes[1, 1].scatter(num_components_list, all_continuity_scores, 
                                            alpha=0.6, s=50, c=all_continuity_scores, 
                                            cmap='viridis', edgecolors='black', linewidths=0.5)
                axes[1, 1].set_xlabel('Number of Connected Components', fontsize=12)
                axes[1, 1].set_ylabel('Connectivity Continuity Score', fontsize=12)
                axes[1, 1].set_title('Continuity Score vs Number of Components', fontsize=13, fontweight='bold')
                axes[1, 1].grid(True, alpha=0.3)
                plt.colorbar(scatter, ax=axes[1, 1], label='Continuity Score')
            else:
                axes[1, 1].text(0.5, 0.5, 'Data length mismatch', 
                              ha='center', va='center', transform=axes[1, 1].transAxes)
                axes[1, 1].set_title('Continuity Score vs Number of Components (No Data)')
            
            plt.tight_layout()
            fig_path = output_dir / "connectivity_continuity_analysis.png"
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            print(f"连续性连通性分析图已保存到: {fig_path}")
            plt.close()
        
        # 保存统计结果到文件
        results_file = output_dir / "structure_analysis_results.txt"
        with open(results_file, 'w', encoding='utf-8') as f:
            f.write("分子结构分析结果\n")
            f.write("="*60 + "\n\n")
            f.write(f"总分子数: {len(npz_files)}\n")
            f.write(f"平均原子数: {num_atoms_list.mean():.2f}\n\n")
            
            if len(all_min_distances) > 0:
                f.write("最近原子距离统计:\n")
                f.write(f"  平均: {all_min_distances.mean():.4f} Å\n")
                f.write(f"  中位数: {np.median(all_min_distances):.4f} Å\n")
                f.write(f"  范围: {all_min_distances.min():.4f} - {all_min_distances.max():.4f} Å\n")
                f.write(f"  标准差: {all_min_distances.std():.4f} Å\n\n")
            
            if len(bond_lengths) > 0:
                f.write("键长统计:\n")
                f.write(f"  总键数: {len(bond_lengths)}\n")
                f.write(f"  平均: {bond_lengths.mean():.4f} Å\n")
                f.write(f"  范围: {bond_lengths.min():.4f} - {bond_lengths.max():.4f} Å\n\n")
            
            f.write("连通性统计:\n")
            f.write(f"  连通分子比例: {is_connected_list.sum() / len(is_connected_list) * 100:.2f}%\n")
            f.write(f"  平均连通分量数: {num_components_list.mean():.2f}\n\n")
            
            if len(all_continuity_scores) > 0:
                f.write("连续性连通性指标 (使用 margin1={}pm):\n".format(margin1))
                f.write(f"  平均连续性分数: {all_continuity_scores.mean():.4f}\n")
                f.write(f"  中位数连续性分数: {np.median(all_continuity_scores):.4f}\n")
                f.write(f"  连续性分数范围: {all_continuity_scores.min():.4f} - {all_continuity_scores.max():.4f}\n")
                if len(all_missing_bond_ratios) > 0:
                    f.write(f"  平均缺失键比例: {all_missing_bond_ratios.mean():.4f} ({all_missing_bond_ratios.mean()*100:.2f}%)\n")
                if len(all_mean_deviations) > 0:
                    f.write(f"  平均偏差百分比: {all_mean_deviations.mean():.4f}%\n")
                if len(all_missing_bond_deviations) > 0:
                    f.write(f"  总缺失键数: {len(all_missing_bond_deviations)}\n")
                    f.write(f"  缺失键平均偏差: {all_missing_bond_deviations.mean():.4f}%\n")
        
        print(f"统计结果已保存到: {results_file}")
    
    return {
        'min_distances': all_min_distances,
        'pair_distances': all_pair_distances,
        'bond_lengths': bond_lengths,
        'num_components': num_components_list,
        'is_connected': is_connected_list,
        'num_atoms': num_atoms_list,
        'continuity_scores': all_continuity_scores,
        'missing_bond_ratios': all_missing_bond_ratios,
        'mean_deviations': all_mean_deviations,
        'max_deviations': all_max_deviations,
        'missing_bond_deviations': all_missing_bond_deviations
    }


def main():
    parser = argparse.ArgumentParser(description='分析生成分子的结构问题')
    parser.add_argument(
        '--molecule_dir',
        type=str,
        required=True,
        help='包含 .npz 分子文件的目录路径'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='输出目录（可选）'
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("分子结构分析工具")
    print("="*60)
    print(f"分子目录: {args.molecule_dir}")
    print("="*60)
    
    results = analyze_molecules(args.molecule_dir, args.output_dir)
    
    print("\n" + "="*60)
    print("分析完成！")
    print("="*60)


if __name__ == "__main__":
    main()

