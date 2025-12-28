"""
原子计数评估模块
分析分子的原子类型统计信息
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def analyze_atom_counts(csv_path):
    """
    分析CSV文件中的原子计数信息
    
    Args:
        csv_path: CSV文件路径
    
    Returns:
        tuple: (stats, all_atom_counts, df)
            - stats: 统计信息字典
            - all_atom_counts: 原子计数字典
            - df: DataFrame
    """
    print("=" * 60)
    print("原子类型统计分析")
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
    
    return stats, all_atom_counts, df


def get_qm9_statistics():
    """
    返回QM9数据集的统计信息
    """
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
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    qm9_stats, qm9_total_atoms, qm9_total_molecules = get_qm9_statistics()
    elements = ['C', 'H', 'O', 'N', 'F']
    
    print("\n" + "=" * 60)
    print("与QM9数据集对比")
    print("=" * 60)
    
    # 对比各原子类型的平均值
    print("\n各原子类型平均值对比:")
    print("-" * 60)
    print(f"{'原子类型':<10} {'生成结果':<20} {'QM9数据集':<20} {'差异':<15}")
    print("-" * 60)
    
    for element in elements:
        if element in stats and element in qm9_stats:
            gen_mean = stats[element]['mean']
            qm9_mean = qm9_stats[element]['mean']
            diff = gen_mean - qm9_mean
            diff_pct = (diff / qm9_mean * 100) if qm9_mean > 0 else 0
            print(f"{element:<10} {gen_mean:>8.2f} ± {stats[element]['std']:>6.2f}    "
                  f"{qm9_mean:>8.2f} ± {qm9_stats[element]['std']:>6.2f}    "
                  f"{diff:>+7.2f} ({diff_pct:>+6.2f}%)")
    
    # 对比总原子数
    gen_total_mean = np.mean(total_atoms_per_mol)
    gen_total_std = np.std(total_atoms_per_mol)
    qm9_total_mean = qm9_total_atoms['mean']
    qm9_total_std = qm9_total_atoms['std']
    diff_total = gen_total_mean - qm9_total_mean
    diff_total_pct = (diff_total / qm9_total_mean * 100) if qm9_total_mean > 0 else 0
    
    print(f"\n{'总原子数':<10} {gen_total_mean:>8.2f} ± {gen_total_std:>6.2f}    "
          f"{qm9_total_mean:>8.2f} ± {qm9_total_std:>6.2f}    "
          f"{diff_total:>+7.2f} ({diff_total_pct:>+6.2f}%)")
    
    # 创建对比可视化
    try:
        from scipy import stats as scipy_stats
        HAS_SCIPY = True
    except ImportError:
        HAS_SCIPY = False
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    colors = {'C': 'black', 'H': 'gray', 'O': 'red', 'N': 'blue', 'F': 'green'}
    
    for idx, element in enumerate(elements):
        ax = axes[idx]
        if element in stats and element in qm9_stats:
            gen_counts = np.array(stats[element]['mean'])
            qm9_counts = np.array(qm9_stats[element]['mean'])
            
            # 绘制对比柱状图
            x = np.arange(1)
            width = 0.35
            ax.bar(x - width/2, gen_counts, width, label='Generated Results', color=colors[element], alpha=0.7)
            ax.bar(x + width/2, qm9_counts, width, label='QM9 Dataset', color=colors[element], alpha=0.5)
            ax.set_ylabel('Average Atom Count', fontsize=12)
            ax.set_title(f'{element} Atom Average Comparison', fontsize=12)
            ax.set_xticks(x)
            ax.set_xticklabels([''])
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    # 总原子数对比
    ax = axes[5]
    x = np.arange(1)
    width = 0.35
    ax.bar(x - width/2, gen_total_mean, width, label='Generated Results', color='purple', alpha=0.7)
    ax.bar(x + width/2, qm9_total_mean, width, label='QM9 Dataset', color='purple', alpha=0.5)
    ax.set_ylabel('Average Total Atom Count', fontsize=12)
    ax.set_title('Total Atom Count Average Comparison', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([''])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    comparison_path = os.path.join(output_dir, 'qm9_comparison.png')
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    print(f"\n已保存QM9对比图: {comparison_path}")
    plt.close()
    
    # 创建所有原子类型的综合对比图
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(elements))
    width = 0.35
    
    gen_means = [stats[element]['mean'] for element in elements]
    qm9_means = [qm9_stats[element]['mean'] for element in elements]
    
    bars1 = ax.bar(x - width/2, gen_means, width, label='Generated Results', color='steelblue', alpha=0.7)
    bars2 = ax.bar(x + width/2, qm9_means, width, label='QM9 Dataset', color='lightblue', alpha=0.7)
    
    ax.set_xlabel('Atom Type', fontsize=12)
    ax.set_ylabel('Average Atom Count', fontsize=12)
    ax.set_title('Atom Type Average Count Comparison: Generated vs QM9', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(elements)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}',
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    comparison_path = os.path.join(output_dir, 'atom_type_comparison.png')
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    print(f"已保存原子类型综合对比图: {comparison_path}")
    plt.close()
