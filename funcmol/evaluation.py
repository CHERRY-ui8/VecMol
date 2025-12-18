#!/usr/bin/env python3
"""
综合分子评估脚本
整合多种分析功能：
1. CSV原子统计分析（analyze_denoiser_results）
2. 分子结构分析（analyze_molecule_structure）
3. 分子质量评估（evaluate_molecules_from_codes）
"""

import sys
from pathlib import Path
import os
import hydra
import argparse
try:
    import yaml
except ImportError:
    print("错误: 需要安装 PyYAML 库")
    print("请运行: pip install pyyaml")
    sys.exit(1)
import pandas as pd
import numpy as np

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 导入各个分析模块
from funcmol.analysis.analyze_denoiser_results import (
    create_visualizations,
    compare_with_qm9
)
from funcmol.analysis.analyze_molecule_structure import analyze_molecules
from funcmol.analysis.evaluate_molecules_from_codes import (
    load_molecules_from_npz,
    evaluate_molecules
)

def resolve_path(path, base_dir=None):
    """
    解析路径（支持相对路径和绝对路径）
    
    Args:
        path: 路径字符串
        base_dir: 基础目录（用于解析相对路径）
    
    Returns:
        解析后的绝对路径
    """
    if os.path.isabs(path):
        return path
    elif base_dir:
        return os.path.join(base_dir, path)
    else:
        return path


def merge_csv_files(csv_files, experiment_dir):
    """
    合并多个CSV文件
    
    Args:
        csv_files: CSV文件路径列表
        experiment_dir: 实验目录（用于解析相对路径）
    
    Returns:
        合并后的DataFrame
    """
    all_dfs = []
    for csv_file in csv_files:
        csv_path = resolve_path(csv_file, experiment_dir)
        if not os.path.exists(csv_path):
            print(f"警告: CSV文件不存在，跳过: {csv_path}")
            continue
        
        print(f"加载CSV文件: {csv_path}")
        df = pd.read_csv(csv_path)
        print(f"  共 {len(df)} 行")
        all_dfs.append(df)
    
    if not all_dfs:
        print("错误: 没有找到任何有效的CSV文件！")
        return None
    
    # 合并所有DataFrame
    merged_df = pd.concat(all_dfs, ignore_index=True)
    print(f"\n合并后共 {len(merged_df)} 行")
    
    return merged_df


def run_csv_analysis(config, output_dir):
    """
    运行CSV原子统计分析
    
    Args:
        config: 配置字典
        output_dir: 输出目录
    """
    print("\n" + "=" * 80)
    print("=" * 80)
    print("1. CSV原子统计分析")
    print("=" * 80)
    print("=" * 80)
    
    experiment_dir = config['experiment_dir']
    csv_files = config['csv']['files']
    
    # 合并CSV文件
    merged_df = merge_csv_files(csv_files, experiment_dir)
    if merged_df is None:
        return None
    
    # 检查必要的列
    required_cols = ['size', 'C_count', 'H_count', 'O_count', 'N_count', 'F_count']
    missing_cols = [col for col in required_cols if col not in merged_df.columns]
    if missing_cols:
        print(f"错误: CSV文件缺少必要的列: {missing_cols}")
        return None
    
    # 计算统计信息（复用analyze_denoiser_results的逻辑）
    all_atom_counts = {
        'C': merged_df['C_count'].tolist(),
        'H': merged_df['H_count'].tolist(),
        'O': merged_df['O_count'].tolist(),
        'N': merged_df['N_count'].tolist(),
        'F': merged_df['F_count'].tolist()
    }
    
    total_molecules = len(merged_df)
    total_atoms_per_mol = merged_df['size'].values
    
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
    
    # 创建可视化
    csv_output_dir = os.path.join(output_dir, 'csv_analysis')
    os.makedirs(csv_output_dir, exist_ok=True)
    
    print("\n" + "=" * 60)
    print("生成可视化图表...")
    print("=" * 60)
    create_visualizations(stats, all_atom_counts, merged_df, csv_output_dir)
    
    # 与QM9数据集对比
    if config['analysis'].get('compare_with_qm9', True):
        print("\n" + "=" * 60)
        print("与QM9数据集对比...")
        print("=" * 60)
        compare_with_qm9(stats, total_atoms_per_mol, csv_output_dir)
    
    return {
        'stats': stats,
        'all_atom_counts': all_atom_counts,
        'df': merged_df,
        'total_atoms_per_mol': total_atoms_per_mol
    }


def run_structure_analysis(config, output_dir):
    """
    运行分子结构分析
    
    Args:
        config: 配置字典
        output_dir: 输出目录
    """
    print("\n" + "=" * 80)
    print("=" * 80)
    print("2. 分子结构分析")
    print("=" * 80)
    print("=" * 80)
    
    experiment_dir = config['experiment_dir']
    molecule_dir = resolve_path(config['molecule']['dir'], experiment_dir)
    
    if not os.path.exists(molecule_dir):
        print(f"错误: 分子目录不存在: {molecule_dir}")
        return None
    
    structure_output_dir = os.path.join(output_dir, 'structure_analysis')
    os.makedirs(structure_output_dir, exist_ok=True)
    
    results = analyze_molecules(molecule_dir, structure_output_dir)
    
    return results


def run_quality_evaluation(config, output_dir):
    """
    运行分子质量评估
    
    Args:
        config: 配置字典
        output_dir: 输出目录
    """
    print("\n" + "=" * 80)
    print("=" * 80)
    print("3. 分子质量评估")
    print("=" * 80)
    print("=" * 80)
    
    experiment_dir = config['experiment_dir']
    molecule_dir = resolve_path(config['molecule']['dir'], experiment_dir)
    
    if not os.path.exists(molecule_dir):
        print(f"错误: 分子目录不存在: {molecule_dir}")
        return None
    
    # 加载分子
    print("\n加载分子文件...")
    molecules = load_molecules_from_npz(molecule_dir)
    
    if not molecules:
        print("错误: 没有成功加载任何分子！")
        return None
    
    print(f"成功加载 {len(molecules)} 个分子")
    
    # 评估分子
    quality_output_dir = os.path.join(output_dir, 'quality_evaluation')
    os.makedirs(quality_output_dir, exist_ok=True)
    
    evaluate_molecules(molecules, quality_output_dir)
    
    return {
        'molecules': molecules,
        'num_molecules': len(molecules)
    }


def generate_summary_report(config, results, output_dir):
    """
    生成综合评估报告
    
    Args:
        config: 配置字典
        results: 所有分析结果
        output_dir: 输出目录
    """
    print("\n" + "=" * 80)
    print("=" * 80)
    print("生成综合评估报告")
    print("=" * 80)
    print("=" * 80)
    
    report_path = os.path.join(output_dir, 'comprehensive_report.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("综合分子评估报告\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"实验目录: {config['experiment_dir']}\n")
        f.write(f"生成时间: {pd.Timestamp.now()}\n\n")
        
        # CSV分析结果
        if 'csv_analysis' in results and results['csv_analysis']:
            f.write("-" * 80 + "\n")
            f.write("1. CSV原子统计分析结果\n")
            f.write("-" * 80 + "\n")
            csv_result = results['csv_analysis']
            stats = csv_result['stats']
            f.write(f"总分子数: {len(csv_result['df'])}\n")
            f.write(f"平均总原子数: {np.mean(csv_result['total_atoms_per_mol']):.2f} ± {np.std(csv_result['total_atoms_per_mol']):.2f}\n\n")
            for element in ['C', 'H', 'O', 'N', 'F']:
                if element in stats:
                    f.write(f"{element} 原子: {stats[element]['mean']:.2f} ± {stats[element]['std']:.2f}\n")
            f.write("\n")
        
        # 结构分析结果
        if 'structure_analysis' in results and results['structure_analysis']:
            f.write("-" * 80 + "\n")
            f.write("2. 分子结构分析结果\n")
            f.write("-" * 80 + "\n")
            # 结构分析结果已经在analyze_molecules中保存到文件
            f.write("详细结果请查看 structure_analysis/structure_analysis_results.txt\n\n")
        
        # 质量评估结果
        if 'quality_evaluation' in results and results['quality_evaluation']:
            f.write("-" * 80 + "\n")
            f.write("3. 分子质量评估结果\n")
            f.write("-" * 80 + "\n")
            f.write(f"评估分子数: {results['quality_evaluation']['num_molecules']}\n")
            f.write("详细结果请查看 quality_evaluation/evaluation_results.txt\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("报告生成完成\n")
        f.write("=" * 80 + "\n")
    
    print(f"\n综合评估报告已保存到: {report_path}")

def run_evaluation(config, output_dir=None):
    """
    运行评估的核心逻辑
    
    Args:
        config: 配置字典
        output_dir: 输出目录（可选，如果未指定，将使用配置文件中的设置）
    """
    # 确定输出目录
    experiment_dir = config['experiment_dir']
    if output_dir:
        final_output_dir = output_dir
    elif config['output']['dir']:
        final_output_dir = resolve_path(config['output']['dir'], experiment_dir)
    else:
        final_output_dir = os.path.join(experiment_dir, 'analysis_results')
    
    os.makedirs(final_output_dir, exist_ok=True)
    print(f"输出目录: {final_output_dir}")
    print("=" * 80)
    
    # 存储所有分析结果
    results = {}
    
    # 运行CSV分析
    if config['analysis'].get('csv_analysis', True):
        csv_result = run_csv_analysis(config, final_output_dir)
        results['csv_analysis'] = csv_result
    
    # 运行结构分析
    if config['analysis'].get('structure_analysis', True):
        structure_result = run_structure_analysis(config, final_output_dir)
        results['structure_analysis'] = structure_result
    
    # 运行质量评估
    if config['analysis'].get('quality_evaluation', True):
        quality_result = run_quality_evaluation(config, final_output_dir)
        results['quality_evaluation'] = quality_result
    
    # 生成综合报告
    generate_summary_report(config, results, final_output_dir)
    
    print("\n" + "=" * 80)
    print("=" * 80)
    print("所有分析完成！")
    print("=" * 80)
    print("=" * 80)
    print(f"结果已保存到: {final_output_dir}")
    print("=" * 80)


@hydra.main(config_path="configs", config_name="evaluation", version_base=None)
def main_hydra(config):
    """Entry point for Hydra configuration system"""
    print("=" * 80)
    print("综合分子评估工具 (Hydra模式)")
    print("=" * 80)
    run_evaluation(config)


def main():
    """主函数 - 通过命令行参数运行"""
    parser = argparse.ArgumentParser(
        description='综合分子评估脚本 - 整合多种分析功能',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--config',
        type=str,
        default='funcmol/configs/evaluation.yaml',
        help='配置文件路径（YAML格式，默认为 funcmol/configs/evaluation.yaml）'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='输出目录（如果未指定，将使用配置文件中的设置）'
    )
    
    args = parser.parse_args()
    
    # 加载配置
    # 支持相对路径（相对于项目根目录）或绝对路径
    if os.path.isabs(args.config):
        config_path = args.config
    else:
        config_path = os.path.join(project_root, args.config)
    
    if not os.path.exists(config_path):
        print(f"错误: 配置文件不存在: {config_path}")
        return
    
    print("=" * 80)
    print("综合分子评估工具")
    print("=" * 80)
    print(f"配置文件: {config_path}")
    
    # 加载YAML配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 运行评估
    run_evaluation(config, args.output_dir)


if __name__ == "__main__":
    main()
