"""
可视化评估模块
包含各种评估结果的可视化功能
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def create_bond_visualizations(bond_results, output_dir=None):
    """
    创建键分析的可视化图表
    
    Args:
        bond_results: 键分析结果字典（来自 analyze_bonds）
        output_dir: 输出目录
    """
    if output_dir is None:
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 缺失键偏差分布
    if len(bond_results.get('missing_bond_deviations', [])) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        deviations = bond_results['missing_bond_deviations']
        ax.hist(deviations, bins=50, alpha=0.7, edgecolor='black')
        ax.set_xlabel('偏差百分比 (%)', fontsize=12)
        ax.set_ylabel('缺失键数量', fontsize=12)
        ax.set_title('缺失键偏差分布', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'missing_bond_deviation_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 2. 缺失键比例分布
    if len(bond_results.get('missing_bond_ratios', [])) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        ratios = bond_results['missing_bond_ratios']
        ax.hist(ratios, bins=50, alpha=0.7, edgecolor='black')
        ax.set_xlabel('缺失键比例', fontsize=12)
        ax.set_ylabel('分子数量', fontsize=12)
        ax.set_title('缺失键比例分布', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'missing_bond_ratio_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. 连续性分数分布
    if len(bond_results.get('continuity_scores', [])) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        scores = bond_results['continuity_scores']
        ax.hist(scores, bins=50, alpha=0.7, edgecolor='black')
        ax.set_xlabel('连续性分数', fontsize=12)
        ax.set_ylabel('分子数量', fontsize=12)
        ax.set_title('连续性分数分布', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'continuity_score_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 4. 连续性分数 vs 连通分量数
    if len(bond_results.get('continuity_scores', [])) > 0 and len(bond_results.get('num_components', [])) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        scores = bond_results['continuity_scores']
        components = bond_results['num_components']
        ax.scatter(components, scores, alpha=0.5)
        ax.set_xlabel('连通分量数', fontsize=12)
        ax.set_ylabel('连续性分数', fontsize=12)
        ax.set_title('连续性分数 vs 连通分量数', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'continuity_score_vs_components.png'), dpi=300, bbox_inches='tight')
        plt.close()


def create_structure_visualizations(structure_results, output_dir=None):
    """
    创建结构分析的可视化图表
    
    Args:
        structure_results: 结构分析结果字典
        output_dir: 输出目录
    """
    if output_dir is None:
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 最近距离分布
    if len(structure_results.get('min_distances', [])) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        min_dists = structure_results['min_distances']
        ax.hist(min_dists, bins=50, alpha=0.7, edgecolor='black')
        ax.set_xlabel('最近距离 (Å)', fontsize=12)
        ax.set_ylabel('原子数量', fontsize=12)
        ax.set_title('最近原子距离分布', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'min_distance_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()


def create_quality_visualizations(quality_results, output_dir=None):
    """
    创建质量评估的可视化图表
    
    Args:
        quality_results: 质量评估结果字典
        output_dir: 输出目录
    """
    if output_dir is None:
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 质量指标柱状图
    fig, ax = plt.subplots(figsize=(12, 6))
    metrics = ['有效性', '唯一性', '新颖性', '分子稳定性', '原子稳定性']
    values = [
        quality_results.get('validity', 0) * 100,
        quality_results.get('uniqueness', 0) * 100,
        quality_results.get('novelty', 0) * 100,
        quality_results.get('mol_stable', 0) * 100,
        quality_results.get('atom_stable', 0) * 100
    ]
    bars = ax.bar(metrics, values, alpha=0.7, edgecolor='black')
    ax.set_ylabel('百分比 (%)', fontsize=12)
    ax.set_title('分子质量指标', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.1f}%',
                ha='center', va='bottom')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'quality_metrics.png'), dpi=300, bbox_inches='tight')
    plt.close()


def create_comprehensive_report(all_results, output_dir=None):
    """
    创建综合评估报告
    
    Args:
        all_results: 包含所有评估结果的字典
        output_dir: 输出目录
    """
    if output_dir is None:
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建综合报告文本
    report_path = os.path.join(output_dir, 'comprehensive_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("综合评估报告\n")
        f.write("="*60 + "\n\n")
        
        # 质量评估结果
        if 'quality' in all_results:
            f.write("质量评估结果\n")
            f.write("-"*60 + "\n")
            quality = all_results['quality']
            f.write(f"有效性: {quality.get('validity', 0)*100:.2f}%\n")
            f.write(f"唯一性: {quality.get('uniqueness', 0)*100:.2f}%\n")
            f.write(f"新颖性: {quality.get('novelty', 0)*100:.2f}%\n")
            f.write(f"分子稳定性: {quality.get('mol_stable', 0)*100:.2f}%\n")
            f.write(f"原子稳定性: {quality.get('atom_stable', 0)*100:.2f}%\n\n")
        
        # 键分析结果
        if 'bond' in all_results:
            f.write("键分析结果\n")
            f.write("-"*60 + "\n")
            bond = all_results['bond']
            if len(bond.get('continuity_scores', [])) > 0:
                f.write(f"平均连续性分数: {np.mean(bond['continuity_scores']):.4f}\n")
            if len(bond.get('missing_bond_ratios', [])) > 0:
                f.write(f"平均缺失键比例: {np.mean(bond['missing_bond_ratios'])*100:.2f}%\n")
            f.write(f"平均连通分量数: {np.mean(bond.get('num_components', [0])):.2f}\n\n")
    
    print(f"综合报告已保存到: {report_path}")

