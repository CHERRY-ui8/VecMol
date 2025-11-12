#!/usr/bin/env python3
"""
统计QM9数据集中分子的原子类型分布
分析每个分子中C、H、O、N、F原子的数量和比例
"""

import torch
import os
import numpy as np
from collections import defaultdict
from tqdm import tqdm

# 原子类型映射（来自 constants.py）
ELEMENTS_HASH_INV = {
    0: 'C',
    1: 'H',
    2: 'O',
    3: 'N',
    4: 'F',
    5: 'S',
    6: 'Cl',
    7: 'Br',
    8: 'P',
    9: 'I',
    10: 'B'
}

def analyze_qm9_atoms(data_dir="/datapool/data2/home/pxg/data/hyc/funcmol-main-neuralfield/funcmol/dataset/data/qm9"):
    """
    分析QM9数据集中所有分子的原子类型分布
    
    Args:
        data_dir: QM9数据集目录路径
    """
    print("=" * 60)
    print("QM9数据集原子类型统计分析")
    print("=" * 60)
    
    # 存储所有分子的原子统计
    all_atom_counts = defaultdict(list)  # {element: [count1, count2, ...]}
    total_molecules = 0
    
    # 遍历train、val、test三个数据集
    for split in ["train", "val", "test"]:
        data_path = os.path.join(data_dir, f"{split}_data.pth")
        
        if not os.path.exists(data_path):
            print(f"警告: 未找到 {data_path}，跳过")
            continue
        
        print(f"\n加载 {split} 数据集...")
        data = torch.load(data_path, weights_only=False)
        print(f"  共 {len(data)} 个分子")
        
        # 统计每个分子的原子类型
        for sample in tqdm(data, desc=f"处理 {split} 数据集"):
            atoms_channel = sample["atoms_channel"]
            
            # atoms_channel 的形状可能是 [1, n_atoms] 或 [n_atoms]
            if len(atoms_channel.shape) > 1:
                atoms_channel = atoms_channel.squeeze(0)
            
            # 统计当前分子中每个原子类型的数量
            atom_counts = defaultdict(int)
            for atom_idx in atoms_channel:
                atom_type = int(atom_idx.item())
                if atom_type in ELEMENTS_HASH_INV:
                    element = ELEMENTS_HASH_INV[atom_type]
                    atom_counts[element] += 1
            
            # 为每个元素类型记录当前分子的原子数
            for element in ['C', 'H', 'O', 'N', 'F']:
                if element not in all_atom_counts:
                    all_atom_counts[element] = []
                # 添加当前分子的原子数
                all_atom_counts[element].append(atom_counts.get(element, 0))
            
            total_molecules += 1
    
    # 计算统计信息
    print("\n" + "=" * 60)
    print("统计结果")
    print("=" * 60)
    print(f"\n总分子数: {total_molecules}")
    print(f"\n各原子类型的统计信息:")
    print("-" * 60)
    
    stats = {}
    for element in ['C', 'H', 'O', 'N', 'F']:
        if element in all_atom_counts and len(all_atom_counts[element]) > 0:
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
            print(f"  占总原子数比例: {stats[element]['total'] / sum([stats[e]['total'] for e in ['C', 'H', 'O', 'N', 'F'] if e in stats]) * 100:.2f}%")
    
    # 计算每个分子的总原子数
    print("\n" + "-" * 60)
    print("每个分子的总原子数统计:")
    print("-" * 60)
    total_atoms_per_mol = []
    for i in range(total_molecules):
        total = sum([all_atom_counts[element][i] for element in ['C', 'H', 'O', 'N', 'F'] if element in all_atom_counts])
        total_atoms_per_mol.append(total)
    
    total_atoms_per_mol = np.array(total_atoms_per_mol)
    print(f"  平均每个分子总原子数: {np.mean(total_atoms_per_mol):.2f} ± {np.std(total_atoms_per_mol):.2f}")
    print(f"  中位数: {np.median(total_atoms_per_mol):.1f}")
    print(f"  范围: {np.min(total_atoms_per_mol)} - {np.max(total_atoms_per_mol)}")
    
    # 原子比例（按数量）
    print("\n" + "-" * 60)
    print("原子类型比例（按数量）:")
    print("-" * 60)
    total_all_atoms = sum([stats[e]['total'] for e in ['C', 'H', 'O', 'N', 'F'] if e in stats])
    for element in ['C', 'H', 'O', 'N', 'F']:
        if element in stats:
            percentage = stats[element]['total'] / total_all_atoms * 100
            print(f"  {element}: {percentage:.2f}%")
    
    return stats, all_atom_counts

if __name__ == "__main__":
    stats, all_atom_counts = analyze_qm9_atoms()
