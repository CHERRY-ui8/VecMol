#!/usr/bin/env python3
"""
分析field_evaluation_results.csv中的数据，对比gaussian_mag和tanh两种定义方法
标准：rmsd越小，atom_count_mismatch为True的越多，这个方法越好
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data(file_path):
    """加载和预处理数据"""
    df = pd.read_csv(file_path)
    
    # 处理inf值，将其替换为NaN以便后续分析
    df['rmsd'] = df['rmsd'].replace([np.inf, -np.inf], np.nan)
    
    # 分离两种方法的数据
    gaussian_data = df[df['field_method'] == 'gaussian_mag'].copy()
    tanh_data = df[df['field_method'] == 'tanh'].copy()
    
    return df, gaussian_data, tanh_data

def analyze_rmsd_performance(gaussian_data, tanh_data):
    """分析RMSD性能"""
    print("=" * 60)
    print("RMSD 性能分析")
    print("=" * 60)
    
    # 基本统计
    gaussian_rmsd = gaussian_data['rmsd'].dropna()
    tanh_rmsd = tanh_data['rmsd'].dropna()
    
    print(f"Gaussian Mag 方法:")
    print(f"  有效样本数: {len(gaussian_rmsd)}")
    print(f"  平均RMSD: {gaussian_rmsd.mean():.6f}")
    print(f"  中位数RMSD: {gaussian_rmsd.median():.6f}")
    print(f"  标准差: {gaussian_rmsd.std():.6f}")
    print(f"  最小值: {gaussian_rmsd.min():.6f}")
    print(f"  最大值: {gaussian_rmsd.max():.6f}")
    
    print(f"\nTanh 方法:")
    print(f"  有效样本数: {len(tanh_rmsd)}")
    print(f"  平均RMSD: {tanh_rmsd.mean():.6f}")
    print(f"  中位数RMSD: {tanh_rmsd.median():.6f}")
    print(f"  标准差: {tanh_rmsd.std():.6f}")
    print(f"  最小值: {tanh_rmsd.min():.6f}")
    print(f"  最大值: {tanh_rmsd.max():.6f}")
    
    # 统计检验
    if len(gaussian_rmsd) > 0 and len(tanh_rmsd) > 0:
        # Mann-Whitney U检验（非参数检验）
        statistic, p_value = stats.mannwhitneyu(gaussian_rmsd, tanh_rmsd, alternative='two-sided')
        print(f"\nMann-Whitney U检验 (p-value): {p_value:.6f}")
        
        if p_value < 0.05:
            print("  结论: 两种方法的RMSD分布存在显著差异")
        else:
            print("  结论: 两种方法的RMSD分布无显著差异")
    
    return gaussian_rmsd, tanh_rmsd

def analyze_atom_count_mismatch(gaussian_data, tanh_data):
    """分析atom_count_mismatch性能 - Mismatch越低越好（False表示重建成功）"""
    print("\n" + "=" * 60)
    print("Atom Count Mismatch 分析 (Mismatch越低越好)")
    print("=" * 60)
    
    # 统计True的数量和比例
    gaussian_mismatch_count = gaussian_data['atom_count_mismatch'].sum()
    gaussian_total = len(gaussian_data)
    gaussian_mismatch_rate = gaussian_mismatch_count / gaussian_total
    gaussian_success_rate = 1 - gaussian_mismatch_rate
    
    tanh_mismatch_count = tanh_data['atom_count_mismatch'].sum()
    tanh_total = len(tanh_data)
    tanh_mismatch_rate = tanh_mismatch_count / tanh_total
    tanh_success_rate = 1 - tanh_mismatch_rate
    
    print(f"Gaussian Mag 方法:")
    print(f"  atom_count_mismatch=True (失败) 的数量: {gaussian_mismatch_count}")
    print(f"  atom_count_mismatch=False (成功) 的数量: {gaussian_total - gaussian_mismatch_count}")
    print(f"  总样本数: {gaussian_total}")
    print(f"  失败率: {gaussian_mismatch_rate:.4f} ({gaussian_mismatch_rate*100:.2f}%)")
    print(f"  成功率: {gaussian_success_rate:.4f} ({gaussian_success_rate*100:.2f}%)")
    
    print(f"\nTanh 方法:")
    print(f"  atom_count_mismatch=True (失败) 的数量: {tanh_mismatch_count}")
    print(f"  atom_count_mismatch=False (成功) 的数量: {tanh_total - tanh_mismatch_count}")
    print(f"  总样本数: {tanh_total}")
    print(f"  失败率: {tanh_mismatch_rate:.4f} ({tanh_mismatch_rate*100:.2f}%)")
    print(f"  成功率: {tanh_success_rate:.4f} ({tanh_success_rate*100:.2f}%)")
    
    # 卡方检验
    contingency_table = np.array([
        [gaussian_mismatch_count, gaussian_total - gaussian_mismatch_count],
        [tanh_mismatch_count, tanh_total - tanh_mismatch_count]
    ])
    
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
    print(f"\n卡方检验 (p-value): {p_value:.6f}")
    
    if p_value < 0.05:
        print("  结论: 两种方法的atom_count_mismatch分布存在显著差异")
    else:
        print("  结论: 两种方法的atom_count_mismatch分布无显著差异")
    
    return gaussian_mismatch_rate, tanh_mismatch_rate

def analyze_inf_values(gaussian_data, tanh_data):
    """分析inf值的情况"""
    print("\n" + "=" * 60)
    print("Inf 值分析")
    print("=" * 60)
    
    gaussian_inf_count = gaussian_data['rmsd'].isna().sum()
    gaussian_total = len(gaussian_data)
    gaussian_inf_rate = gaussian_inf_count / gaussian_total
    
    tanh_inf_count = tanh_data['rmsd'].isna().sum()
    tanh_total = len(tanh_data)
    tanh_inf_rate = tanh_inf_count / tanh_total
    
    print(f"Gaussian Mag 方法:")
    print(f"  Inf值数量: {gaussian_inf_count}")
    print(f"  Inf值比例: {gaussian_inf_rate:.4f} ({gaussian_inf_rate*100:.2f}%)")
    
    print(f"\nTanh 方法:")
    print(f"  Inf值数量: {tanh_inf_count}")
    print(f"  Inf值比例: {tanh_inf_rate:.4f} ({tanh_inf_rate*100:.2f}%)")
    
    return gaussian_inf_rate, tanh_inf_rate

def create_visualizations(gaussian_data, tanh_data, gaussian_rmsd, tanh_rmsd):
    """创建可视化图表"""
    # 设置现代化的配色方案和样式
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # 定义配色方案
    colors = {
        'gaussian': '#2E86AB',  # 深蓝色
        'tanh': '#A23B72',      # 深红色
        'background': '#F8F9FA', # 浅灰色背景
        'text': '#2C3E50',      # 深灰色文字
        'grid': '#E5E7EB'       # 浅灰色网格
    }
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor(colors['background'])
    fig.suptitle('Field Methods Comparison: Gaussian Mag vs Tanh', 
                fontsize=18, fontweight='bold', color=colors['text'], y=0.92)
    
    # 1. RMSD分布对比
    ax1 = axes[0]
    ax1.set_facecolor('white')
    
    if len(gaussian_rmsd) > 0:
        ax1.hist(gaussian_rmsd, bins=50, alpha=0.8, label='Gaussian Mag', 
                color=colors['gaussian'], density=True, edgecolor='white', linewidth=0.5)
    if len(tanh_rmsd) > 0:
        ax1.hist(tanh_rmsd, bins=50, alpha=0.8, label='Tanh', 
                color=colors['tanh'], density=True, edgecolor='white', linewidth=0.5)
    
    ax1.set_xlabel('RMSD', fontsize=12, fontweight='bold', color=colors['text'])
    ax1.set_ylabel('Density', fontsize=12, fontweight='bold', color=colors['text'])
    ax1.set_title('RMSD Distribution Comparison', fontsize=14, fontweight='bold', color=colors['text'])
    ax1.legend(fontsize=11, frameon=True, fancybox=True, shadow=True)
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3, color=colors['grid'])
    
    # 美化坐标轴
    ax1.tick_params(colors=colors['text'], labelsize=10)
    for spine in ax1.spines.values():
        spine.set_color(colors['grid'])
    
    # 2. 重建成功率对比
    ax2 = axes[1]
    ax2.set_facecolor('white')
    
    methods = ['Gaussian Mag', 'Tanh']
    mismatch_counts = [gaussian_data['atom_count_mismatch'].sum(), tanh_data['atom_count_mismatch'].sum()]
    total_counts = [len(gaussian_data), len(tanh_data)]
    success_rates = [1 - count/total for count, total in zip(mismatch_counts, total_counts)]
    
    bars = ax2.bar(methods, success_rates, color=[colors['gaussian'], colors['tanh']], 
                   alpha=0.8, edgecolor='white', linewidth=2)
    ax2.set_ylabel('Success Rate', fontsize=12, fontweight='bold', color=colors['text'])
    ax2.set_title('Reconstruction Success Rate', fontsize=14, fontweight='bold', color=colors['text'])
    ax2.set_ylim(0, 1)
    
    # 添加数值标签
    for bar, rate in zip(bars, success_rates):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{rate:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold', color=colors['text'])
    
    # 美化坐标轴
    ax2.tick_params(colors=colors['text'], labelsize=10)
    ax2.grid(True, alpha=0.3, color=colors['grid'], axis='y')
    for spine in ax2.spines.values():
        spine.set_color(colors['grid'])
    
    # 调整布局，给主标题留出更多空间
    plt.tight_layout()
    plt.subplots_adjust(top=0.85, hspace=0.3)
    
    # 保存高质量图片
    plt.savefig('/data/huayuchen/Neurl-voxel/vecmol/field_methods_comparison.png', 
                dpi=300, bbox_inches='tight', facecolor=colors['background'])
    plt.show()

def comprehensive_evaluation(gaussian_data, tanh_data, gaussian_rmsd, tanh_rmsd, 
                           gaussian_mismatch_rate, tanh_mismatch_rate, 
                           gaussian_inf_rate, tanh_inf_rate):
    """综合评估两种方法"""
    print("\n" + "=" * 60)
    print("综合评估结果")
    print("=" * 60)
    
    # 评分系统：RMSD越小越好，mismatch率越低越好，inf率越低越好
    def calculate_score(rmsd_mean, mismatch_rate, inf_rate):
        # 归一化评分 (0-100)
        rmsd_score = max(0, 100 - rmsd_mean * 10000)  # RMSD越小分数越高
        success_rate = 1 - mismatch_rate  # 成功率越高分数越高
        mismatch_penalty = mismatch_rate * 50  # mismatch率越高扣分越多
        inf_penalty = inf_rate * 50  # inf率越高扣分越多
        
        total_score = rmsd_score + success_rate * 100 - mismatch_penalty - inf_penalty
        return max(0, min(100, total_score))
    
    gaussian_rmsd_mean = gaussian_rmsd.mean() if len(gaussian_rmsd) > 0 else float('inf')
    tanh_rmsd_mean = tanh_rmsd.mean() if len(tanh_rmsd) > 0 else float('inf')
    
    gaussian_score = calculate_score(gaussian_rmsd_mean, gaussian_mismatch_rate, gaussian_inf_rate)
    tanh_score = calculate_score(tanh_rmsd_mean, tanh_mismatch_rate, tanh_inf_rate)
    
    print(f"Gaussian Mag 综合评分: {gaussian_score:.2f}/100")
    print(f"  - RMSD表现: {gaussian_rmsd_mean:.6f}")
    print(f"  - 失败率: {gaussian_mismatch_rate:.4f}")
    print(f"  - 成功率: {1-gaussian_mismatch_rate:.4f}")
    print(f"  - Inf率: {gaussian_inf_rate:.4f}")
    
    print(f"\nTanh 综合评分: {tanh_score:.2f}/100")
    print(f"  - RMSD表现: {tanh_rmsd_mean:.6f}")
    print(f"  - 失败率: {tanh_mismatch_rate:.4f}")
    print(f"  - 成功率: {1-tanh_mismatch_rate:.4f}")
    print(f"  - Inf率: {tanh_inf_rate:.4f}")
    
    print(f"\n推荐方法: {'Tanh' if tanh_score > gaussian_score else 'Gaussian Mag'}")
    
    # 详细分析
    print(f"\n详细分析:")
    if tanh_rmsd_mean < gaussian_rmsd_mean:
        print(f"  ✓ Tanh在RMSD方面表现更好 ({tanh_rmsd_mean:.6f} vs {gaussian_rmsd_mean:.6f})")
    else:
        print(f"  ✓ Gaussian Mag在RMSD方面表现更好 ({gaussian_rmsd_mean:.6f} vs {tanh_rmsd_mean:.6f})")
    
    if gaussian_mismatch_rate < tanh_mismatch_rate:
        print(f"  ✓ Gaussian Mag在重建成功率方面表现更好 (失败率: {gaussian_mismatch_rate:.4f} vs {tanh_mismatch_rate:.4f})")
    else:
        print(f"  ✓ Tanh在重建成功率方面表现更好 (失败率: {tanh_mismatch_rate:.4f} vs {gaussian_mismatch_rate:.4f})")
    
    if gaussian_inf_rate < tanh_inf_rate:
        print(f"  ✓ Gaussian Mag在稳定性方面表现更好 (inf率: {gaussian_inf_rate:.4f} vs {tanh_inf_rate:.4f})")
    else:
        print(f"  ✓ Tanh在稳定性方面表现更好 (inf率: {tanh_inf_rate:.4f} vs {gaussian_inf_rate:.4f})")

def main():
    """主函数"""
    file_path = '/data/huayuchen/Neurl-voxel/exps/gt_field/field_evaluation_results.csv'
    
    print("开始分析field_evaluation_results.csv数据...")
    print("分析标准: RMSD越小越好，atom_count_mismatch为False的越多越好（重建成功）")
    
    # 加载数据
    df, gaussian_data, tanh_data = load_and_preprocess_data(file_path)
    
    print(f"\n数据概览:")
    print(f"总样本数: {len(df)}")
    print(f"Gaussian Mag样本数: {len(gaussian_data)}")
    print(f"Tanh样本数: {len(tanh_data)}")
    
    # 分析RMSD性能
    gaussian_rmsd, tanh_rmsd = analyze_rmsd_performance(gaussian_data, tanh_data)
    
    # 分析atom_count_mismatch性能
    gaussian_mismatch_rate, tanh_mismatch_rate = analyze_atom_count_mismatch(gaussian_data, tanh_data)
    
    # 分析inf值
    gaussian_inf_rate, tanh_inf_rate = analyze_inf_values(gaussian_data, tanh_data)
    
    # 创建可视化
    create_visualizations(gaussian_data, tanh_data, gaussian_rmsd, tanh_rmsd)
    
    # 综合评估
    comprehensive_evaluation(gaussian_data, tanh_data, gaussian_rmsd, tanh_rmsd,
                           gaussian_mismatch_rate, tanh_mismatch_rate,
                           gaussian_inf_rate, tanh_inf_rate)
    
    print(f"\n分析完成！可视化图表已保存为: field_methods_comparison.png")

if __name__ == "__main__":
    main()
