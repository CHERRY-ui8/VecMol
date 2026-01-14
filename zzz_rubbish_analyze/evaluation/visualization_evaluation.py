"""
可视化评估模块
包含各种评估结果的可视化功能
"""

import os
import numpy as np
import matplotlib.pyplot as plt


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
    
    # 缺失键偏差分布（三种标准）
    for standard in ['strict', 'medium', 'relaxed']:
        if standard in bond_results and len(bond_results[standard].get('missing_bond_deviations', [])) > 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            deviations = bond_results[standard]['missing_bond_deviations']
            ax.hist(deviations, bins=50, alpha=0.7, edgecolor='black')
            ax.set_xlabel('Deviation Percentage (%)', fontsize=12)
            ax.set_ylabel('Number of Missing Bonds', fontsize=12)
            ax.set_title(f'Missing Bond Deviation Distribution ({standard.capitalize()} Standard)', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'missing_bond_deviation_distribution_{standard}.png'), dpi=300, bbox_inches='tight')
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
            # 处理 torch.Tensor 类型
            def to_float(val):
                if hasattr(val, 'item'):
                    return float(val.item())
                return float(val) if val is not None else 0.0
            
            f.write(f"有效性: {to_float(quality.get('validity', 0))*100:.2f}%\n")
            f.write(f"唯一性: {to_float(quality.get('uniqueness', 0))*100:.2f}%\n")
            f.write(f"新颖性: {to_float(quality.get('novelty', 0))*100:.2f}%\n")
            # 从quality结果中读取margin值，如果没有则使用默认值
            strict_m1 = quality.get('strict_margin1', 15)
            strict_m2 = quality.get('strict_margin2', 10)
            strict_m3 = quality.get('strict_margin3', 8)
            medium_m1 = quality.get('medium_margin1', 30)
            medium_m2 = quality.get('medium_margin2', 15)
            medium_m3 = quality.get('medium_margin3', 12)
            relaxed_m1 = quality.get('relaxed_margin1', 40)
            relaxed_m2 = quality.get('relaxed_margin2', 20)
            relaxed_m3 = quality.get('relaxed_margin3', 15)
            
            f.write(f"分子稳定性（严格，margin1={strict_m1}pm, margin2={strict_m2}pm, margin3={strict_m3}pm）: {to_float(quality.get('mol_stable', 0))*100:.2f}%\n")
            f.write(f"原子稳定性（严格，margin1={strict_m1}pm, margin2={strict_m2}pm, margin3={strict_m3}pm）: {to_float(quality.get('atom_stable', 0))*100:.2f}%\n")
            if 'mol_stable_medium' in quality:
                f.write(f"分子稳定性（中等，margin1={medium_m1}pm, margin2={medium_m2}pm, margin3={medium_m3}pm）: {to_float(quality.get('mol_stable_medium', 0))*100:.2f}%\n")
                f.write(f"原子稳定性（中等，margin1={medium_m1}pm, margin2={medium_m2}pm, margin3={medium_m3}pm）: {to_float(quality.get('atom_stable_medium', 0))*100:.2f}%\n")
            if 'mol_stable_relaxed' in quality:
                f.write(f"分子稳定性（宽松，margin1={relaxed_m1}pm, margin2={relaxed_m2}pm, margin3={relaxed_m3}pm）: {to_float(quality.get('mol_stable_relaxed', 0))*100:.2f}%\n")
                f.write(f"原子稳定性（宽松，margin1={relaxed_m1}pm, margin2={relaxed_m2}pm, margin3={relaxed_m3}pm）: {to_float(quality.get('atom_stable_relaxed', 0))*100:.2f}%\n")
            f.write("\n")
        
        # 键分析结果
        if 'bond' in all_results:
            f.write("键分析结果\n")
            f.write("-"*60 + "\n")
            bond = all_results['bond']
            
            # 三种标准的结果
            for standard in ['strict', 'medium', 'relaxed']:
                if standard in bond:
                    std_data = bond[standard]
                    f.write(f"\n{standard.capitalize()}标准:\n")
                    f.write(f"  平均连通分量数: {np.mean(std_data.get('num_components', [0])):.2f}\n")
                    f.write(f"  连通分子比例: {np.mean(std_data.get('is_connected', [False])) * 100:.2f}%\n")
                    if len(std_data.get('missing_bond_deviations', [])) > 0:
                        f.write(f"  缺失键总数: {len(std_data['missing_bond_deviations'])}\n")
                        f.write(f"  缺失键平均偏差: {np.mean(std_data['missing_bond_deviations']):.2f}%\n")
                    else:
                        f.write(f"  缺失键总数: 0\n")
            f.write("\n")
    
    print(f"综合报告已保存到: {report_path}")
