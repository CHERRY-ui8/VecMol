"""
评估从 .npz 文件加载的分子质量

使用方法:
    python -m funcmol.analysis.evaluate_molecules_from_codes \
        --molecule_dir /datapool/data2/home/pxg/data/hyc/funcmol-main-neuralfield/exps/funcmol/fm_qm9/20251108/molecule \
        --output_dir /datapool/data2/home/pxg/data/hyc/funcmol-main-neuralfield/funcmol/analysis/analysis_metrics
"""

import sys
from pathlib import Path
import torch
import numpy as np
import argparse
from tqdm import tqdm

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from funcmol.analysis.rdkit_functions import Molecule, check_stability
from funcmol.analysis.baselines_evaluation import (
    SimpleSamplingMetrics,
    atom_decoder_dict,
    build_xae_molecule
)


def load_molecules_from_npz(molecule_dir):
    """
    从 .npz 文件直接加载已生成的分子
    
    Args:
        molecule_dir: 包含 .npz 文件的目录路径
        
    Returns:
        list: Molecule 对象列表
    """
    molecule_dir = Path(molecule_dir)
    npz_files = sorted(molecule_dir.glob("generated_*.npz"))
    
    print(f"找到 {len(npz_files)} 个 .npz 分子文件")
    
    atom_decoder = atom_decoder_dict['qm9_with_h']
    molecules = []
    
    for npz_file in tqdm(npz_files, desc="加载分子文件"):
        try:
            # 加载 npz 文件
            data = np.load(npz_file)
            coords = data['coords']  # (N, 3) 坐标
            types = data['types']    # (N,) 原子类型
            
            # 转换为torch张量
            positions = torch.tensor(coords, dtype=torch.float32)
            atom_types = torch.tensor(types, dtype=torch.long)
            
            # 过滤掉填充的原子（值为 -1）
            valid_mask = atom_types != -1
            if not valid_mask.any():
                continue
            
            positions = positions[valid_mask]
            atom_types = atom_types[valid_mask]
            
            # 构建键类型矩阵
            dataset_info = {'name': 'qm9'}
            _, _, bond_types = build_xae_molecule(
                positions=positions,
                atom_types=atom_types,
                dataset_info=dataset_info,
                atom_decoder=atom_decoder
            )
            
            # 创建零电荷
            charges = torch.zeros_like(atom_types)
            
            # 创建 Molecule 对象
            molecule = Molecule(
                atom_types=atom_types.long(),
                bond_types=bond_types.long(),
                positions=positions.float(),
                charges=charges,
                atom_decoder=atom_decoder
            )
            
            molecules.append(molecule)
            
        except Exception as e:
            print(f"\n加载文件 {npz_file} 时出错: {e}")
            continue
    
    print(f"成功加载 {len(molecules)} 个分子")
    return molecules




def evaluate_molecules(molecules, output_dir=None):
    """
    评估分子质量指标
    
    Args:
        molecules: Molecule 对象列表
        output_dir: 输出目录（可选）
    """
    if not molecules:
        print("没有有效的分子可以评估！")
        return
    
    # 创建简化的数据集信息
    class SimpleDatasetInfo:
        def __init__(self):
            self.atom_decoder = atom_decoder_dict['qm9_with_h']
            self.remove_h = False
            # 添加必要的统计信息（简化版本）
            self.statistics = {
                'test': type('obj', (object,), {
                    'num_nodes': {i: 1 for i in range(1, 30)},
                    'atom_types': torch.ones(5),
                    'bond_types': torch.ones(5),
                    'charge_types': torch.ones(5),
                    'valencies': {atom: {0: 1, 1: 1, 2: 1, 3: 1, 4: 1} for atom in atom_decoder_dict['qm9_with_h']},
                    'bond_lengths': {1: {1.5: 1}, 2: {1.3: 1}, 3: {1.2: 1}, 4: {1.4: 1}},
                    'bond_angles': torch.ones(5, 1801)
                })()
            }
    
    dataset_infos = SimpleDatasetInfo()
    
    # 创建采样指标评估器
    sampling_metrics = SimpleSamplingMetrics(
        train_smiles=[],
        dataset_infos=dataset_infos,
        test=True
    )
    
    # 评估分子
    print("\n" + "="*60)
    print(f"开始评估分子质量（共 {len(molecules)} 个分子）...")
    print("="*60)
    
    print("正在计算有效性、唯一性和新颖性...")
    sampling_metrics(
        molecules=molecules,
        name='generated_molecules',
        current_epoch=0,
        local_rank=0
    )
    print("评估完成！")
    
    # 打印结果
    print("\n" + "="*60)
    print("分子质量评估结果")
    print("="*60)
    
    validity = sampling_metrics.validity_metric
    uniqueness = sampling_metrics.uniqueness
    novelty = sampling_metrics.novelty
    mean_components = sampling_metrics.mean_components
    max_components = sampling_metrics.max_components
    mol_stable = sampling_metrics.mol_stable
    atom_stable = sampling_metrics.atom_stable
    
    print(f"\n📊 总体质量指标:")
    print(f"  有效性 (Validity): {validity*100:.2f}%")
    print(f"  唯一性 (Uniqueness): {uniqueness*100:.2f}%")
    print(f"  新颖性 (Novelty): {novelty*100:.2f}%")
    print(f"  平均连通分量数: {mean_components:.2f}")
    print(f"  最大连通分量数: {max_components:.2f}")
    print(f"  分子稳定性: {float(mol_stable)*100:.2f}%")
    print(f"  原子稳定性: {float(atom_stable)*100:.2f}%")
    
    # 分析单个分子
    print(f"\n📈 单个分子分析（前10个）:")
    valid_molecules = 0
    stable_molecules = 0
    
    for i, mol in enumerate(molecules[:10]):  # 只显示前10个
        if mol.rdkit_mol is not None:
            valid_molecules += 1
            try:
                mol_stable, atom_stable, _ = check_stability(
                    mol, None, atom_decoder=atom_decoder_dict['qm9_with_h']
                )
                if mol_stable.item() > 0.5:
                    stable_molecules += 1
                    print(f"  分子 {i+1}: {mol.num_nodes} 个原子, RDKit有效: 是, 稳定: 是")
                else:
                    print(f"  分子 {i+1}: {mol.num_nodes} 个原子, RDKit有效: 是, 稳定: 否")
            except Exception as e:
                print(f"  分子 {i+1}: {mol.num_nodes} 个原子, RDKit有效: 是, 稳定性检查失败: {e}")
        else:
            print(f"  分子 {i+1}: {mol.num_nodes} 个原子, RDKit有效: 否")
    
    # 统计所有分子
    print(f"\n📊 统计所有分子（共 {len(molecules)} 个）...")
    total_valid = sum(1 for mol in tqdm(molecules, desc="检查有效性", leave=False) if mol.rdkit_mol is not None)
    total_stable = 0
    for mol in tqdm(molecules, desc="检查稳定性"):
        if mol.rdkit_mol is not None:
            try:
                mol_stable, _, _ = check_stability(
                    mol, None, atom_decoder=atom_decoder_dict['qm9_with_h']
                )
                if mol_stable.item() > 0.5:
                    total_stable += 1
            except:
                pass
    
    print(f"\n📊 统计摘要:")
    print(f"  总分子数: {len(molecules)}")
    print(f"  有效分子数: {total_valid}")
    print(f"  稳定分子数: {total_stable}")
    print(f"  有效性: {total_valid/len(molecules)*100:.1f}%")
    print(f"  稳定性: {total_stable/len(molecules)*100:.1f}%")
    
    # 保存结果到文件（如果指定了输出目录）
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results_file = output_dir / "evaluation_results.txt"
        with open(results_file, 'w', encoding='utf-8') as f:
            f.write("分子质量评估结果\n")
            f.write("="*60 + "\n\n")
            f.write(f"总分子数: {len(molecules)}\n")
            f.write(f"有效分子数: {total_valid}\n")
            f.write(f"稳定分子数: {total_stable}\n")
            f.write(f"有效性: {validity*100:.2f}%\n")
            f.write(f"唯一性: {uniqueness*100:.2f}%\n")
            f.write(f"新颖性: {novelty*100:.2f}%\n")
            f.write(f"平均连通分量数: {mean_components:.2f}\n")
            f.write(f"最大连通分量数: {max_components:.2f}\n")
            f.write(f"分子稳定性: {float(mol_stable)*100:.2f}%\n")
            f.write(f"原子稳定性: {float(atom_stable)*100:.2f}%\n")
        
        print(f"\n结果已保存到: {results_file}")


def main():
    parser = argparse.ArgumentParser(description='评估从 .npz 文件加载的分子质量')
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
    print("分子质量评估工具")
    print("="*60)
    print(f"分子目录: {args.molecule_dir}")
    print("="*60)
    
    # 1. 加载分子
    print("\n1. 加载分子文件...")
    molecules = load_molecules_from_npz(args.molecule_dir)
    
    if not molecules:
        print("错误: 没有成功加载任何分子！")
        return
    
    print(f"成功加载 {len(molecules)} 个分子")
    
    # 2. 评估分子
    print("\n2. 评估分子质量...")
    evaluate_molecules(molecules, output_dir=args.output_dir)
    
    print("\n" + "="*60)
    print("评估完成！")
    print("="*60)


if __name__ == "__main__":
    main()

