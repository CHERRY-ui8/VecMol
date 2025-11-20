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

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from funcmol.analysis.baselines_evaluation import (
    atom_decoder_dict,
    build_xae_molecule
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
        
        # 1. 最近距离分布
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
            
            plt.tight_layout()
            fig_path = output_dir / "molecule_structure_analysis.png"
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            print(f"\n可视化图表已保存到: {fig_path}")
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
            f.write(f"  平均连通分量数: {num_components_list.mean():.2f}\n")
        
        print(f"统计结果已保存到: {results_file}")
    
    return {
        'min_distances': all_min_distances,
        'pair_distances': all_pair_distances,
        'bond_lengths': bond_lengths,
        'num_components': num_components_list,
        'is_connected': is_connected_list,
        'num_atoms': num_atoms_list
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

