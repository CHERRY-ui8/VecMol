"""
综合分子评估主入口
整合所有评估模块：原子计数、结构、键、质量评估
"""

import sys
from pathlib import Path
import os
import argparse
import pandas as pd
import numpy as np

try:
    import yaml
except ImportError:
    yaml = None
    print("警告: PyYAML 未安装，无法从配置文件读取。请运行: pip install pyyaml")

try:
    import hydra
    from omegaconf import DictConfig
    HAS_HYDRA = True
except ImportError:
    HAS_HYDRA = False

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 导入新的评估模块
from funcmol.evaluation.atom_count_evaluation import (
    analyze_atom_counts,
    compare_with_qm9,
    create_visualizations as create_atom_count_visualizations
)
from funcmol.evaluation.bond_evaluation import analyze_bonds
from funcmol.evaluation.quality_evaluation import (
    load_molecules_from_npz,
    evaluate_quality
)
from funcmol.evaluation.visualization_evaluation import (
    create_bond_visualizations,
    create_quality_visualizations,
    create_comprehensive_report
)
from funcmol.evaluation.utils_evaluation import resolve_path


def run_atom_count_evaluation(csv_path, output_dir=None, compare_qm9=True):
    """
    运行原子计数评估
    
    Args:
        csv_path: CSV文件路径
        output_dir: 输出目录
        compare_qm9: 是否与QM9数据集对比
    
    Returns:
        dict: 评估结果
    """
    print("\n" + "=" * 80)
    print("1. 原子计数评估")
    print("=" * 80)
    
    stats, all_atom_counts, df = analyze_atom_counts(csv_path)
    if stats is None:
        return None
    
    if output_dir:
        csv_output_dir = os.path.join(output_dir, 'atom_count_analysis')
        os.makedirs(csv_output_dir, exist_ok=True)
        
        create_atom_count_visualizations(stats, all_atom_counts, df, csv_output_dir)
        
        if compare_qm9:
            total_atoms_per_mol = df['size'].values
            compare_with_qm9(stats, total_atoms_per_mol, csv_output_dir)
    
    return {
        'stats': stats,
        'all_atom_counts': all_atom_counts,
        'df': df
    }


def run_structure_evaluation(molecule_dir, output_dir=None):
    """
    运行结构评估（距离、空间分布等）
    
    Args:
        molecule_dir: 包含 .npz 文件的目录
        output_dir: 输出目录
    
    Returns:
        dict: 评估结果
    """
    print("\n" + "=" * 80)
    print("2. 结构评估")
    print("=" * 80)
    
    # 结构评估功能可以在这里添加
    # 目前主要的结构分析已整合到键评估中
    print("结构评估功能已整合到键评估模块中")
    # 使用参数以避免lint警告
    _ = molecule_dir, output_dir
    
    return {}


def run_bond_evaluation(molecule_dir, output_dir=None, strict_margin1=15, strict_margin2=10, strict_margin3=6):
    """
    运行键评估（键判断、连通性、缺失键检测等）
    
    Args:
        molecule_dir: 包含 .npz 文件的目录
        output_dir: 输出目录
        strict_margin1: 严格margin1值（pm单位）
        strict_margin2: 严格margin2值（pm单位）
        strict_margin3: 严格margin3值（pm单位）
    
    Returns:
        dict: 评估结果
    """
    print("\n" + "=" * 80)
    print("3. 键和连通性评估")
    print("=" * 80)
    
    bond_results = analyze_bonds(
        molecule_dir, 
        output_dir=None,  # 不在analyze_bonds内部保存，统一在外部保存
        strict_margin1=strict_margin1,
        strict_margin2=strict_margin2,
        strict_margin3=strict_margin3
    )
    
    if output_dir and bond_results:
        bond_output_dir = os.path.join(output_dir, 'bond_analysis')
        os.makedirs(bond_output_dir, exist_ok=True)
        create_bond_visualizations(bond_results, bond_output_dir)
    
    return bond_results


def run_quality_evaluation(molecule_dir, output_dir=None):
    """
    运行质量评估（有效性、唯一性、稳定性等）
    
    Args:
        molecule_dir: 包含 .npz 文件的目录
        output_dir: 输出目录
    
    Returns:
        dict: 评估结果
    """
    print("\n" + "=" * 80)
    print("4. 质量评估")
    print("=" * 80)
    
    molecules = load_molecules_from_npz(molecule_dir)
    if not molecules:
        print("没有找到有效的分子文件！")
        return None
    
    quality_results = evaluate_quality(molecules, output_dir=None)  # 不在evaluate_quality内部保存
    
    if output_dir and quality_results:
        quality_output_dir = os.path.join(output_dir, 'quality_analysis')
        os.makedirs(quality_output_dir, exist_ok=True)
        create_quality_visualizations(quality_results, quality_output_dir)
    
    return quality_results


def run_evaluation(molecule_dir=None, csv_path=None, output_dir=None, 
                   compare_qm9=True, strict_margin1=15, strict_margin2=10, strict_margin3=6,
                   config=None):
    """
    运行综合评估
    
    支持两种调用方式：
    1. 直接传入参数（molecule_dir, csv_path等）
    2. 从配置字典读取（config参数）
    
    Args:
        molecule_dir: 包含 .npz 文件的目录（如果config为None则必需）
        csv_path: CSV文件路径（可选）
        output_dir: 输出目录
        compare_qm9: 是否与QM9数据集对比
        strict_margin1: 严格margin1值（pm单位）
        strict_margin2: 严格margin2值（pm单位）
        strict_margin3: 严格margin3值（pm单位）
        config: 配置字典（可选，如果提供则从配置读取参数）
    
    Returns:
        dict: 包含所有评估结果的字典
    """
    # 如果提供了config，从配置读取参数
    if config is not None:
        # 处理 Hydra DictConfig
        if HAS_HYDRA and isinstance(config, DictConfig):
            config = dict(config)
        
        experiment_dir = config.get('experiment_dir', '')
        
        # 从配置读取 molecule_dir
        if molecule_dir is None:
            molecule_dir_rel = config.get('molecule', {}).get('dir', '')
            if molecule_dir_rel:
                molecule_dir = resolve_path(molecule_dir_rel, experiment_dir)
        
        # 从配置读取 csv_path（合并多个CSV文件）
        if csv_path is None and config.get('csv', {}).get('files'):
            csv_files = config['csv']['files']
            experiment_dir = config.get('experiment_dir', '')
            # 合并CSV文件
            all_dfs = []
            for csv_file in csv_files:
                csv_file_path = resolve_path(csv_file, experiment_dir)
                if os.path.exists(csv_file_path):
                    df = pd.read_csv(csv_file_path)
                    all_dfs.append(df)
            
            if all_dfs:
                # 合并所有DataFrame并保存到临时文件
                merged_df = pd.concat(all_dfs, ignore_index=True)
                import tempfile
                temp_csv = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
                merged_df.to_csv(temp_csv.name, index=False)
                csv_path = temp_csv.name
                temp_csv.close()
        
        # 从配置读取 output_dir
        if output_dir is None:
            output_dir_rel = config.get('output', {}).get('dir')
            if output_dir_rel:
                output_dir = resolve_path(output_dir_rel, experiment_dir)
            elif experiment_dir:
                output_dir = os.path.join(experiment_dir, 'analysis_results')
        
        # 从配置读取 compare_qm9
        if 'analysis' in config:
            compare_qm9 = config['analysis'].get('compare_with_qm9', compare_qm9)
        
        # 从配置读取 strict_margin 值（如果配置中有）
        if 'bond_evaluation' in config:
            strict_margin1 = config['bond_evaluation'].get('strict_margin1', strict_margin1)
            strict_margin2 = config['bond_evaluation'].get('strict_margin2', strict_margin2)
            strict_margin3 = config['bond_evaluation'].get('strict_margin3', strict_margin3)
    
    # 验证必需参数
    if molecule_dir is None:
        raise ValueError("molecule_dir 必须提供（直接传入或通过config）")
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    all_results = {}
    
    # 1. 原子计数评估（如果提供了CSV文件）
    if csv_path:
        atom_count_results = run_atom_count_evaluation(csv_path, output_dir, compare_qm9)
        if atom_count_results:
            all_results['atom_count'] = atom_count_results
    
    # 2. 结构评估
    structure_results = run_structure_evaluation(molecule_dir, output_dir)
    if structure_results:
        all_results['structure'] = structure_results
    
    # 3. 键评估
    bond_results = run_bond_evaluation(
        molecule_dir, 
        output_dir, 
        strict_margin1=strict_margin1,
        strict_margin2=strict_margin2,
        strict_margin3=strict_margin3
    )
    if bond_results:
        all_results['bond'] = bond_results
    
    # 4. 质量评估
    quality_results = run_quality_evaluation(molecule_dir, output_dir)
    if quality_results:
        all_results['quality'] = quality_results
    
    # 5. 生成综合报告
    if output_dir:
        create_comprehensive_report(all_results, output_dir)
    
    return all_results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='综合分子评估',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='配置文件路径（YAML格式，如果提供则从配置文件读取参数）'
    )
    parser.add_argument(
        '--molecule_dir',
        type=str,
        default=None,
        help='包含 .npz 分子文件的目录路径（如果未提供config则必需）'
    )
    parser.add_argument(
        '--csv_path',
        type=str,
        default=None,
        help='CSV文件路径（可选，用于原子计数评估）'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='输出目录（可选，如果提供config则从配置读取）'
    )
    parser.add_argument(
        '--compare_qm9',
        action='store_true',
        default=None,
        help='是否与QM9数据集对比（如果提供config则从配置读取）'
    )
    parser.add_argument(
        '--strict_margin1',
        type=int,
        default=None,
        help='严格margin1值（pm单位，如果提供config则从配置读取）'
    )
    parser.add_argument(
        '--strict_margin2',
        type=int,
        default=None,
        help='严格margin2值（pm单位，如果提供config则从配置读取）'
    )
    parser.add_argument(
        '--strict_margin3',
        type=int,
        default=None,
        help='严格margin3值（pm单位，如果提供config则从配置读取）'
    )
    
    args = parser.parse_args()
    
    config = None
    # 如果提供了配置文件，加载配置
    if args.config:
        if yaml is None:
            print("错误: 需要安装 PyYAML 库才能从配置文件读取")
            print("请运行: pip install pyyaml")
            return
        
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
    # 如果提供了命令行参数，它们会覆盖配置文件中的设置
    results = run_evaluation(
        molecule_dir=args.molecule_dir,
        csv_path=args.csv_path,
        output_dir=args.output_dir,
        compare_qm9=args.compare_qm9 if args.compare_qm9 is not None else True,
        strict_margin1=args.strict_margin1 if args.strict_margin1 is not None else 15,
        strict_margin2=args.strict_margin2 if args.strict_margin2 is not None else 10,
        strict_margin3=args.strict_margin3 if args.strict_margin3 is not None else 6,
        config=config
    )
    
    print("\n" + "=" * 80)
    print("评估完成！")
    print("=" * 80)
    
    # 确定输出目录用于显示
    final_output_dir = args.output_dir
    if not final_output_dir and config:
        output_dir_rel = config.get('output', {}).get('dir')
        if output_dir_rel:
            experiment_dir = config.get('experiment_dir', '')
            final_output_dir = resolve_path(output_dir_rel, experiment_dir)
        elif config.get('experiment_dir'):
            final_output_dir = os.path.join(config['experiment_dir'], 'analysis_results')
    
    if final_output_dir:
        print(f"\n所有结果已保存到: {final_output_dir}")


def main_hydra():
    """Hydra入口函数（如果安装了Hydra）"""
    if not HAS_HYDRA:
        print("错误: 需要安装 Hydra 库")
        print("请运行: pip install hydra-core")
        return
    
    @hydra.main(config_path="../configs", config_name="evaluation", version_base=None)
    def hydra_main(config: DictConfig):
        """Entry point for Hydra configuration system"""
        print("=" * 80)
        print("综合分子评估工具 (Hydra模式)")
        print("=" * 80)
        # 从config读取参数，不传递其他参数
        run_evaluation(config=config)
    
    hydra_main()


if __name__ == '__main__':
    main()

