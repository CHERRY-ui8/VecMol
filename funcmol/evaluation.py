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
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 导入新的评估模块
from funcmol.evaluation.atom_count_evaluation import (
    analyze_atom_counts,
    compare_with_qm9
)
from funcmol.evaluation.bond_evaluation import analyze_bonds
from funcmol.evaluation.quality_evaluation import (
    load_molecules_from_npz,
    evaluate_quality
)
from funcmol.evaluation.visualization_evaluation import (
    create_bond_visualizations,
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
        
        if compare_qm9:
            total_atoms_per_mol = df['size'].values
            compare_with_qm9(stats, total_atoms_per_mol, csv_output_dir)
    
    return {
        'stats': stats,
        'all_atom_counts': all_atom_counts,
        'df': df
    }


def run_bond_evaluation(molecule_dir,
                       strict_margin1, strict_margin2, strict_margin3,
                       medium_margin1, medium_margin2, medium_margin3,
                       relaxed_margin1, relaxed_margin2, relaxed_margin3,
                       output_dir=None):
    """
    运行键评估（键判断、连通性、缺失键检测等，使用三种标准）
    
    Args:
        molecule_dir: 包含 .npz 文件的目录
        output_dir: 输出目录
        strict_margin1/2/3: 严格标准的margin值（pm单位）
        medium_margin1/2/3: 中等标准的margin值（pm单位）
        relaxed_margin1/2/3: 宽松标准的margin值（pm单位）
    
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
        strict_margin3=strict_margin3,
        medium_margin1=medium_margin1,
        medium_margin2=medium_margin2,
        medium_margin3=medium_margin3,
        relaxed_margin1=relaxed_margin1,
        relaxed_margin2=relaxed_margin2,
        relaxed_margin3=relaxed_margin3
    )
    
    if output_dir and bond_results:
        bond_output_dir = os.path.join(output_dir, 'bond_analysis')
        os.makedirs(bond_output_dir, exist_ok=True)
        create_bond_visualizations(bond_results, bond_output_dir)
    
    return bond_results


def run_quality_evaluation(molecule_dir,
                          strict_margin1, strict_margin2, strict_margin3,
                          medium_margin1, medium_margin2, medium_margin3,
                          relaxed_margin1, relaxed_margin2, relaxed_margin3,
                          output_dir=None):
    """
    运行质量评估（有效性、唯一性、稳定性等）
    
    Args:
        molecule_dir: 包含 .npz 文件的目录
        output_dir: 输出目录
        strict_margin1/2/3: 严格标准的margin值（pm单位）
        medium_margin1/2/3: 中等标准的margin值（pm单位）
        relaxed_margin1/2/3: 宽松标准的margin值（pm单位），如果为None则使用utils_evaluation中的默认值
    
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
    
    quality_results = evaluate_quality(
        molecules,
        strict_margin1=strict_margin1,
        strict_margin2=strict_margin2,
        strict_margin3=strict_margin3,
        medium_margin1=medium_margin1,
        medium_margin2=medium_margin2,
        medium_margin3=medium_margin3,
        relaxed_margin1=relaxed_margin1,
        relaxed_margin2=relaxed_margin2,
        relaxed_margin3=relaxed_margin3,
        output_dir=None  # 不在evaluate_quality内部保存
    )
        
    return quality_results


def run_evaluation(config):
    """
    运行综合评估（所有参数从配置文件读取）
    
    Args:
        config: 配置字典（必需）
    
    Returns:
        dict: 包含所有评估结果的字典
    """
    # 处理 Hydra DictConfig
    if HAS_HYDRA and isinstance(config, DictConfig):
        config = dict(config)
    
    experiment_dir = config.get('experiment_dir', '')
    
    # 从配置读取 molecule_dir
    molecule_dir_rel = config.get('molecule', {}).get('dir', '')
    if not molecule_dir_rel:
        raise ValueError("配置文件中必须提供 molecule.dir")
    molecule_dir = resolve_path(molecule_dir_rel, experiment_dir)
    
    # 从配置读取 csv_path（合并多个CSV文件）
    csv_path = None
    if config.get('csv', {}).get('files'):
        csv_files = config['csv']['files']
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
    output_dir_rel = config.get('output', {}).get('dir')
    if output_dir_rel:
        output_dir = resolve_path(output_dir_rel, experiment_dir)
    elif experiment_dir:
        output_dir = os.path.join(experiment_dir, 'analysis_results')
    else:
        output_dir = None
    
    # 从配置读取 compare_qm9
    compare_qm9 = config.get('analysis', {}).get('compare_with_qm9', True)
    
    # 从配置读取 margin 值
    bond_eval_config = config.get('bond_evaluation', {})
    strict_margin1 = bond_eval_config.get('strict_margin1', 15)
    strict_margin2 = bond_eval_config.get('strict_margin2', 10)
    strict_margin3 = bond_eval_config.get('strict_margin3', 8)
    medium_margin1 = bond_eval_config.get('medium_margin1', 30)
    medium_margin2 = bond_eval_config.get('medium_margin2', 15)
    medium_margin3 = bond_eval_config.get('medium_margin3', 12)
    relaxed_margin1 = bond_eval_config.get('relaxed_margin1', 40)
    relaxed_margin2 = bond_eval_config.get('relaxed_margin2', 20)
    relaxed_margin3 = bond_eval_config.get('relaxed_margin3', 15)
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    all_results = {}
    
    # 1. 原子计数评估（如果提供了CSV文件）
    if csv_path:
        atom_count_results = run_atom_count_evaluation(csv_path, output_dir, compare_qm9)
        if atom_count_results:
            all_results['atom_count'] = atom_count_results
        
    # 2. 键评估
    bond_results = run_bond_evaluation(
        molecule_dir,
        strict_margin1=strict_margin1,
        strict_margin2=strict_margin2,
        strict_margin3=strict_margin3,
        medium_margin1=medium_margin1,
        medium_margin2=medium_margin2,
        medium_margin3=medium_margin3,
        relaxed_margin1=relaxed_margin1,
        relaxed_margin2=relaxed_margin2,
        relaxed_margin3=relaxed_margin3,
        output_dir=output_dir
    )
    if bond_results:
        all_results['bond'] = bond_results
    
    # 3. 质量评估
    quality_results = run_quality_evaluation(
        molecule_dir,
        strict_margin1,
        strict_margin2,
        strict_margin3,
        medium_margin1,
        medium_margin2,
        medium_margin3,
        relaxed_margin1,
        relaxed_margin2,
        relaxed_margin3,
        output_dir=output_dir
    )
    if quality_results:
        all_results['quality'] = quality_results
    
    # 4. 生成综合报告
    if output_dir:
        create_comprehensive_report(all_results, output_dir)
    
    return all_results


def main():
    """主函数 - 所有配置从 YAML 文件读取"""
    parser = argparse.ArgumentParser(
        description='综合分子评估（所有配置从 YAML 文件读取）',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='配置文件路径（可选，默认使用 funcmol/configs/evaluation.yaml）'
    )
    
    args = parser.parse_args()
    
    # 确定配置文件路径
    if args.config:
        # 如果提供了配置文件路径，使用它
        if os.path.isabs(args.config):
            config_path = args.config
        else:
            config_path = os.path.join(project_root, args.config)
    else:
        # 使用默认配置文件
        config_path = os.path.join(project_root, 'funcmol', 'configs', 'evaluation.yaml')
    
    # 检查配置文件是否存在
    if not os.path.exists(config_path):
        print(f"错误: 配置文件不存在: {config_path}")
        print(f"请创建配置文件或使用 --config 指定配置文件路径")
        return
    
    # 检查 YAML 库
    if yaml is None:
        print("错误: 需要安装 PyYAML 库才能从配置文件读取")
        print("请运行: pip install pyyaml")
        return
    
    # 加载YAML配置
    print("=" * 80)
    print("综合分子评估工具")
    print("=" * 80)
    print(f"配置文件: {config_path}")
    print("=" * 80)
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    if config is None:
        print("错误: 配置文件为空或格式错误")
        return
    
    # 从配置文件运行评估
    results = run_evaluation(config)
    
    print("\n" + "=" * 80)
    print("评估完成！")
    print("=" * 80)
    
    # 确定输出目录用于显示
    if config:
        output_dir_rel = config.get('output', {}).get('dir')
        if output_dir_rel:
            experiment_dir = config.get('experiment_dir', '')
            final_output_dir = resolve_path(output_dir_rel, experiment_dir)
        elif config.get('experiment_dir'):
            final_output_dir = os.path.join(config['experiment_dir'], 'analysis_results')
        else:
            final_output_dir = None
        
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

