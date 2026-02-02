#!/usr/bin/env python3
"""
分析denoiser生成的分子结果CSV文件
统计各个分子的原子数、各种类原子原子数、比例等
类似analyze_qm9_atoms.py的分析方法
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import os
import argparse
try:
    from scipy import stats as scipy_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Warning: scipy not available, using simplified distribution for QM9 comparison")

# Use default matplotlib fonts for English labels

def analyze_denoiser_results(csv_path):
    """
    分析denoiser生成的分子结果CSV文件
    
    Args:
        csv_path: CSV文件路径
    """
    print("=" * 60)
    print("Denoiser生成分子原子类型统计分析")
    print("=" * 60)
    
    # 读取CSV文件
    if not os.path.exists(csv_path):
        print(f"错误: 未找到文件 {csv_path}")
        return None, None, None
    
    print(f"\n加载CSV文件: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"  共 {len(df)} 个分子")
    
    # 检查必要的列
    required_cols = ['size', 'C_count', 'H_count', 'O_count', 'N_count', 'F_count']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"错误: CSV文件缺少必要的列: {missing_cols}")
        return None, None, None
    
    # 存储所有分子的原子统计
    all_atom_counts = {
        'C': df['C_count'].tolist(),
        'H': df['H_count'].tolist(),
        'O': df['O_count'].tolist(),
        'N': df['N_count'].tolist(),
        'F': df['F_count'].tolist()
    }
    
    total_molecules = len(df)
    total_atoms_per_mol = df['size'].values
    
    # 计算统计信息
    print("\n" + "=" * 60)
    print("统计结果")
    print("=" * 60)
    print(f"\n总分子数: {total_molecules}")
    print(f"\n各原子类型的统计信息:")
    print("-" * 60)
    
    stats = {}
    elements = ['C', 'H', 'O', 'N', 'F']
    
    for element in elements:
        counts = np.array(all_atom_counts[element])
        stats[element] = {
            'mean': np.mean(counts),
            'std': np.std(counts),
            'min': np.min(counts),
            'max': np.max(counts),
            'median': np.median(counts),
            'total': np.sum(counts)
        }
        
        print(f"\n{element} 原子:")
        print(f"  平均每个分子: {stats[element]['mean']:.2f} ± {stats[element]['std']:.2f}")
        print(f"  中位数: {stats[element]['median']:.1f}")
        print(f"  范围: {stats[element]['min']} - {stats[element]['max']}")
        print(f"  总计: {stats[element]['total']:,}")
    
    # 计算每个分子的总原子数
    print("\n" + "-" * 60)
    print("每个分子的总原子数统计:")
    print("-" * 60)
    print(f"  平均每个分子总原子数: {np.mean(total_atoms_per_mol):.2f} ± {np.std(total_atoms_per_mol):.2f}")
    print(f"  中位数: {np.median(total_atoms_per_mol):.1f}")
    print(f"  范围: {np.min(total_atoms_per_mol)} - {np.max(total_atoms_per_mol)}")
    
    # 原子比例（按数量）
    print("\n" + "-" * 60)
    print("原子类型比例（按数量）:")
    print("-" * 60)
    total_all_atoms = sum([stats[e]['total'] for e in elements])
    for element in elements:
        if element in stats:
            percentage = stats[element]['total'] / total_all_atoms * 100
            print(f"  {element}: {percentage:.2f}%")
    
    # 每个分子中各原子的比例
    print("\n" + "-" * 60)
    print("每个分子中各原子类型的平均比例:")
    print("-" * 60)
    element_ratios = {}
    for element in elements:
        ratios = []
        for i in range(total_molecules):
            total = total_atoms_per_mol[i]
            if total > 0:
                ratio = all_atom_counts[element][i] / total * 100
                ratios.append(ratio)
        element_ratios[element] = ratios
        print(f"  {element}: {np.mean(ratios):.2f}% ± {np.std(ratios):.2f}%")
    
    # 分子大小分布
    print("\n" + "-" * 60)
    print("分子大小分布:")
    print("-" * 60)
    size_ranges = [
        (0, 10, "1-10"),
        (10, 15, "11-15"),
        (15, 20, "16-20"),
        (20, 25, "21-25"),
        (25, float('inf'), "25+")
    ]
    for min_size, max_size, label in size_ranges:
        count = np.sum((total_atoms_per_mol >= min_size) & (total_atoms_per_mol < max_size))
        percentage = count / total_molecules * 100
        print(f"  {label} 个原子: {count} 个分子 ({percentage:.2f}%)")
    
    return stats, all_atom_counts, df


def create_visualizations(stats, all_atom_counts, df, output_dir=None):
    """
    创建可视化图表
    
    Args:
        stats: 统计信息字典
        all_atom_counts: 原子计数字典
        df: DataFrame
        output_dir: 输出目录
    """
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))
    
    os.makedirs(output_dir, exist_ok=True)
    
    elements = ['C', 'H', 'O', 'N', 'F']
    colors = {'C': 'black', 'H': 'gray', 'O': 'red', 'N': 'blue', 'F': 'green'}
    
    # 1. 原子数量分布直方图
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, element in enumerate(elements):
        ax = axes[idx]
        counts = np.array(all_atom_counts[element])
        ax.hist(counts, bins=20, color=colors[element], alpha=0.7, edgecolor='black')
        ax.set_xlabel(f'{element} Atom Count', fontsize=12)
        ax.set_ylabel('Number of Molecules', fontsize=12)
        ax.set_title(f'{element} Atom Count Distribution\n(Mean: {stats[element]["mean"]:.2f} ± {stats[element]["std"]:.2f})', 
                    fontsize=12)
        ax.grid(True, alpha=0.3)
    
    # 总原子数分布
    ax = axes[5]
    total_atoms = df['size'].values
    ax.hist(total_atoms, bins=20, color='purple', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Total Atoms', fontsize=12)
    ax.set_ylabel('Number of Molecules', fontsize=12)
    ax.set_title(f'Total Atom Count Distribution\n(Mean: {np.mean(total_atoms):.2f} ± {np.std(total_atoms):.2f})', 
                fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'atom_count_distributions.png'), dpi=300, bbox_inches='tight')
    print(f"\n已保存原子数量分布图: {os.path.join(output_dir, 'atom_count_distributions.png')}")
    plt.close()
    
    # 2. 箱线图：各原子类型数量分布
    fig, ax = plt.subplots(figsize=(12, 6))
    data_for_box = [all_atom_counts[e] for e in elements]
    bp = ax.boxplot(data_for_box, tick_labels=elements, patch_artist=True)
    for patch, element in zip(bp['boxes'], elements):
        patch.set_facecolor(colors[element])
        patch.set_alpha(0.7)
    ax.set_ylabel('Atom Count', fontsize=12)
    ax.set_xlabel('Atom Type', fontsize=12)
    ax.set_title('Atom Count Distribution Boxplot', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'atom_count_boxplot.png'), dpi=300, bbox_inches='tight')
    print(f"已保存原子数量箱线图: {os.path.join(output_dir, 'atom_count_boxplot.png')}")
    plt.close()


def get_qm9_statistics():
    """
    返回QM9数据集的统计信息
    这些数据来自analyze_qm9_atoms.py的实际计算结果
    """
    # 从实际QM9数据集计算得到的准确统计数据
    qm9_stats = {
        'C': {'mean': 6.36, 'std': 1.23, 'min': 0, 'max': 9, 'median': 6.0, 'total': 831925, 'proportion': 35.26},
        'H': {'mean': 9.24, 'std': 2.82, 'min': 0, 'max': 20, 'median': 9.0, 'total': 1208486, 'proportion': 51.22},
        'O': {'mean': 1.40, 'std': 0.88, 'min': 0, 'max': 5, 'median': 1.0, 'total': 183265, 'proportion': 7.77},
        'N': {'mean': 1.01, 'std': 1.07, 'min': 0, 'max': 7, 'median': 1.0, 'total': 132498, 'proportion': 5.62},
        'F': {'mean': 0.02, 'std': 0.22, 'min': 0, 'max': 6, 'median': 0.0, 'total': 3036, 'proportion': 0.13}
    }
    qm9_total_atoms = {'mean': 18.03, 'std': 2.94, 'min': 3, 'max': 29, 'median': 18.0}
    qm9_total_molecules = 130831
    return qm9_stats, qm9_total_atoms, qm9_total_molecules


def compare_with_qm9(stats, total_atoms_per_mol, output_dir=None):
    """
    对比生成结果与QM9数据集
    
    Args:
        stats: 生成分子的统计信息
        total_atoms_per_mol: 每个分子的总原子数
        output_dir: 输出目录
    """
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))
    
    os.makedirs(output_dir, exist_ok=True)
    
    qm9_stats, qm9_total_atoms, qm9_total_molecules = get_qm9_statistics()
    elements = ['C', 'H', 'O', 'N', 'F']
    colors = {'C': 'black', 'H': 'gray', 'O': 'red', 'N': 'blue', 'F': 'green'}
    
    print("\n" + "=" * 60)
    print("Comparison with QM9 Dataset")
    print("=" * 60)
    
    # 打印对比统计
    print("\nAtom Type Mean Count Comparison:")
    print("-" * 60)
    print(f"{'Element':<10} {'Generated':<20} {'QM9':<20} {'Difference':<15}")
    print("-" * 60)
    for element in elements:
        gen_mean = stats[element]['mean']
        qm9_mean = qm9_stats[element]['mean']
        diff = gen_mean - qm9_mean
        diff_pct = (diff / qm9_mean * 100) if qm9_mean > 0 else 0
        print(f"{element:<10} {gen_mean:>6.2f} ± {stats[element]['std']:>5.2f}    "
              f"{qm9_mean:>6.2f} ± {qm9_stats[element]['std']:>5.2f}    "
              f"{diff:>+7.2f} ({diff_pct:>+6.1f}%)")
    
    print("\nTotal Atoms per Molecule Comparison:")
    print("-" * 60)
    gen_total_mean = np.mean(total_atoms_per_mol)
    gen_total_std = np.std(total_atoms_per_mol)
    qm9_total_mean = qm9_total_atoms['mean']
    qm9_total_std = qm9_total_atoms['std']
    diff_total = gen_total_mean - qm9_total_mean
    diff_total_pct = diff_total / qm9_total_mean * 100
    print(f"Generated: {gen_total_mean:.2f} ± {gen_total_std:.2f}")
    print(f"QM9:       {qm9_total_mean:.2f} ± {qm9_total_std:.2f}")
    print(f"Difference: {diff_total:+.2f} ({diff_total_pct:+.1f}%)")
    
    # 1. 对比各原子类型平均数量
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(elements))
    width = 0.35
    
    gen_means = [stats[e]['mean'] for e in elements]
    qm9_means = [qm9_stats[e]['mean'] for e in elements]
    gen_stds = [stats[e]['std'] for e in elements]
    qm9_stds = [qm9_stats[e]['std'] for e in elements]
    
    bars1 = ax.bar(x - width/2, gen_means, width, yerr=gen_stds, label='Generated', 
                   alpha=0.8, capsize=5)
    bars2 = ax.bar(x + width/2, qm9_means, width, yerr=qm9_stds, label='QM9', 
                   alpha=0.8, capsize=5)
    
    # 设置颜色
    for i, element in enumerate(elements):
        bars1[i].set_color(colors[element])
        bars2[i].set_color(colors[element])
        bars2[i].set_alpha(0.5)
    
    ax.set_xlabel('Atom Type', fontsize=12)
    ax.set_ylabel('Mean Count per Molecule', fontsize=12)
    ax.set_title('Mean Atom Count Comparison: Generated vs QM9', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(elements)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'atom_count_comparison.png'), dpi=300, bbox_inches='tight')
    print(f"\n已保存原子数量对比图: {os.path.join(output_dir, 'atom_count_comparison.png')}")
    plt.close()
    
    # 2. 对比总原子数分布
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(total_atoms_per_mol, bins=30, alpha=0.6, label='Generated', 
            color='blue', edgecolor='black', density=True)
    
    # 绘制QM9的理论分布（正态分布近似）
    if HAS_SCIPY:
        qm9_x = np.linspace(qm9_total_atoms['min'], qm9_total_atoms['max'], 100)
        qm9_pdf = scipy_stats.norm.pdf(qm9_x, qm9_total_atoms['mean'], qm9_total_atoms['std'])
        ax.plot(qm9_x, qm9_pdf, 'r-', linewidth=2, label='QM9 (Normal approximation)', alpha=0.8)
    else:
        # 如果没有scipy，只显示一条垂直线表示QM9的平均值
        ax.axvline(qm9_total_atoms['mean'], color='red', linestyle='--', linewidth=2, 
                  label=f'QM9 Mean ({qm9_total_atoms["mean"]:.2f})', alpha=0.8)
    
    ax.set_xlabel('Total Atoms per Molecule', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Total Atom Count Distribution Comparison', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'total_atoms_comparison.png'), dpi=300, bbox_inches='tight')
    print(f"已保存总原子数分布对比图: {os.path.join(output_dir, 'total_atoms_comparison.png')}")
    plt.close()
    
    # 3. 对比原子类型比例
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(elements))
    width = 0.35
    
    gen_proportions = []
    qm9_proportions = []
    total_all_atoms = sum([stats[e]['total'] for e in elements])
    for element in elements:
        gen_prop = stats[element]['total'] / total_all_atoms * 100
        gen_proportions.append(gen_prop)
        qm9_proportions.append(qm9_stats[element]['proportion'])
    
    bars1 = ax.bar(x - width/2, gen_proportions, width, label='Generated', alpha=0.8)
    bars2 = ax.bar(x + width/2, qm9_proportions, width, label='QM9', alpha=0.8)
    
    # 设置颜色
    for i, element in enumerate(elements):
        bars1[i].set_color(colors[element])
        bars2[i].set_color(colors[element])
        bars2[i].set_alpha(0.5)
    
    ax.set_xlabel('Atom Type', fontsize=12)
    ax.set_ylabel('Proportion (%)', fontsize=12)
    ax.set_title('Atom Type Proportion Comparison: Generated vs QM9', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(elements)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'atom_proportion_comparison.png'), dpi=300, bbox_inches='tight')
    print(f"已保存原子比例对比图: {os.path.join(output_dir, 'atom_proportion_comparison.png')}")
    plt.close()
    

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='分析denoiser生成的分子结果CSV文件')
    parser.add_argument(
        '--csv_path',
        type=str,
        default=None,
        help='CSV文件路径（如果未指定，将使用默认路径）'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='输出目录（如果未指定，将保存到CSV文件所在目录的analysis_results文件夹）'
    )
    
    args = parser.parse_args()
    
    # 如果没有指定CSV路径，使用默认路径
    if args.csv_path is None:
        # 默认路径
        csv_path = '/data/huayuchen/Neurl-voxel/exps/funcmol/fm_qm9/20251212/denoiser_evaluation_results.csv'
    else:
        csv_path = args.csv_path
    
    # 分析数据
    result = analyze_denoiser_results(csv_path)
    if result is None:
        return
    
    stats, all_atom_counts, df = result
    
    # 创建可视化 - 保存到实验目录下的analysis_results文件夹
    # 从CSV路径推断实验目录
    if args.output_dir is None:
        exp_dir = os.path.dirname(os.path.abspath(csv_path))
        output_dir = os.path.join(exp_dir, 'analysis_results')
    else:
        output_dir = args.output_dir
    
    print(f"\n" + "=" * 60)
    print("生成可视化图表...")
    print("=" * 60)
    create_visualizations(stats, all_atom_counts, df, output_dir)
    
    # 与QM9数据集对比
    total_atoms_per_mol = df['size'].values
    compare_with_qm9(stats, total_atoms_per_mol, output_dir)
    
    print(f"\n" + "=" * 60)
    print("分析完成！")
    print("=" * 60)
    print(f"结果已保存到: {output_dir}")


if __name__ == "__main__":
    main()

