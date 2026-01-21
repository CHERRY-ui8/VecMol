#!/usr/bin/env python3
"""
分析为什么某些键没有被连接
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from rdkit import Chem
import numpy as np
from funcmol.utils.utils_base import (
    fix_missing_bonds_from_sdf, 
    calculate_atom_valency, 
    get_atom_max_valency,
    BOND_LENGTHS_PM
)

def analyze_molecule():
    # 读取原始文件（OpenBabel处理后）
    test_file = Path("/data/huayuchen/Neurl-voxel/exps/funcmol/fm_drugs/20260116/samples/20260116_version_2_epoch_9_better_converge_withbonds_addH/molecule/genmol_000000_2_847680_obabel.sdf")
    
    with open(test_file, 'r') as f:
        sdf_content = f.read()
    
    mol = Chem.MolFromMolBlock(sdf_content, sanitize=False)
    if mol is None:
        print("无法解析分子")
        return
    
    conf = mol.GetConformer()
    
    print("=" * 80)
    print("分析分子中的原子和键")
    print("=" * 80)
    
    # 1. 找到所有N原子，检查它们的价态
    print("\n1. N原子分析:")
    print("-" * 80)
    n_atoms = []
    for atom_idx in range(mol.GetNumAtoms()):
        atom = mol.GetAtomWithIdx(atom_idx)
        if atom.GetSymbol() == 'N':
            pos = conf.GetAtomPosition(atom_idx)
            current_valency = calculate_atom_valency(mol, atom_idx)
            max_valency = get_atom_max_valency(atom.GetAtomicNum())
            n_atoms.append((atom_idx, pos, current_valency, max_valency))
            print(f"  原子 {atom_idx}: 位置=({pos.x:.3f}, {pos.y:.3f}, {pos.z:.3f}), "
                  f"价态={current_valency}/{max_valency}")
            
            # 找到已连接的原子
            connected = []
            for bond in atom.GetBonds():
                other_idx = bond.GetOtherAtomIdx(atom_idx)
                other_atom = mol.GetAtomWithIdx(other_idx)
                other_pos = conf.GetAtomPosition(other_idx)
                connected.append((other_idx, other_atom.GetSymbol(), other_pos))
            print(f"    已连接: {[(idx, sym) for idx, sym, _ in connected]}")
    
    # 2. 找到所有C原子，检查它们的价态和连接情况
    print("\n2. C原子分析（重点关注价态未饱和的）:")
    print("-" * 80)
    c_atoms = []
    for atom_idx in range(mol.GetNumAtoms()):
        atom = mol.GetAtomWithIdx(atom_idx)
        if atom.GetSymbol() == 'C':
            pos = conf.GetAtomPosition(atom_idx)
            current_valency = calculate_atom_valency(mol, atom_idx)
            max_valency = get_atom_max_valency(atom.GetAtomicNum())
            
            # 找到已连接的原子
            connected = []
            for bond in atom.GetBonds():
                other_idx = bond.GetOtherAtomIdx(atom_idx)
                other_atom = mol.GetAtomWithIdx(other_idx)
                connected.append((other_idx, other_atom.GetSymbol()))
            
            # 统计连接的H数量
            h_count = sum(1 for _, sym in connected if sym == 'H')
            
            if current_valency < max_valency or h_count == 3:
                c_atoms.append((atom_idx, pos, current_valency, max_valency, h_count, connected))
                print(f"  原子 {atom_idx}: 位置=({pos.x:.3f}, {pos.y:.3f}, {pos.z:.3f}), "
                      f"价态={current_valency}/{max_valency}, H数={h_count}")
                print(f"    已连接: {connected}")
    
    # 3. 对于价态未饱和的N原子，找到最近的未连接的C原子
    print("\n3. 分析N原子应该连接的键:")
    print("-" * 80)
    for n_idx, n_pos, n_val, n_max_val in n_atoms:
        if n_val < n_max_val:
            print(f"\n  N原子 {n_idx} (价态 {n_val}/{n_max_val}):")
            n_pos_array = np.array([n_pos.x, n_pos.y, n_pos.z])
            
            # 找到已连接的原子索引
            connected_to_n = set()
            for bond in mol.GetAtomWithIdx(n_idx).GetBonds():
                connected_to_n.add(bond.GetOtherAtomIdx(n_idx))
            
            # 找到所有未连接的C原子
            candidates = []
            for c_idx in range(mol.GetNumAtoms()):
                if c_idx == n_idx or c_idx in connected_to_n:
                    continue
                c_atom = mol.GetAtomWithIdx(c_idx)
                if c_atom.GetSymbol() == 'C':
                    c_pos = conf.GetAtomPosition(c_idx)
                    c_pos_array = np.array([c_pos.x, c_pos.y, c_pos.z])
                    distance = np.linalg.norm(c_pos_array - n_pos_array)
                    
                    # 获取标准键长
                    standard_bond_pm = BOND_LENGTHS_PM.get('N', {}).get('C')
                    if standard_bond_pm:
                        standard_bond_angstrom = standard_bond_pm / 100.0
                        threshold_1_3 = standard_bond_angstrom * 1.3
                        threshold_2_0 = standard_bond_angstrom * 2.0
                        candidates.append((c_idx, distance, standard_bond_angstrom, 
                                         distance <= threshold_1_3, distance <= threshold_2_0))
            
            # 按距离排序
            candidates.sort(key=lambda x: x[1])
            print(f"    最近的C原子候选:")
            for c_idx, dist, std_bond, in_1_3, in_2_0 in candidates[:5]:
                c_atom = mol.GetAtomWithIdx(c_idx)
                c_val = calculate_atom_valency(mol, c_idx)
                c_max_val = get_atom_max_valency(c_atom.GetAtomicNum())
                print(f"      C{c_idx}: 距离={dist:.3f}Å, 标准键长={std_bond:.3f}Å, "
                      f"1.3倍阈值内={in_1_3}, 2.0倍阈值内={in_2_0}, "
                      f"价态={c_val}/{c_max_val}")
    
    # 4. 对于只连接3个H的C原子，找到最近的未连接的C原子
    print("\n4. 分析只连接3个H的C原子应该连接的键:")
    print("-" * 80)
    for c_idx, c_pos, c_val, c_max_val, h_count, connected in c_atoms:
        if h_count == 3:
            print(f"\n  C原子 {c_idx} (连接3个H, 价态 {c_val}/{c_max_val}):")
            c_pos_array = np.array([c_pos.x, c_pos.y, c_pos.z])
            
            # 找到已连接的原子索引
            connected_to_c = set()
            for bond in mol.GetAtomWithIdx(c_idx).GetBonds():
                connected_to_c.add(bond.GetOtherAtomIdx(c_idx))
            
            # 找到所有未连接的C原子
            candidates = []
            for other_idx in range(mol.GetNumAtoms()):
                if other_idx == c_idx or other_idx in connected_to_c:
                    continue
                other_atom = mol.GetAtomWithIdx(other_idx)
                if other_atom.GetSymbol() == 'C':
                    other_pos = conf.GetAtomPosition(other_idx)
                    other_pos_array = np.array([other_pos.x, other_pos.y, other_pos.z])
                    distance = np.linalg.norm(other_pos_array - c_pos_array)
                    
                    # 获取标准键长
                    standard_bond_pm = BOND_LENGTHS_PM.get('C', {}).get('C')
                    if standard_bond_pm:
                        standard_bond_angstrom = standard_bond_pm / 100.0
                        threshold_1_3 = standard_bond_angstrom * 1.3
                        threshold_2_0 = standard_bond_angstrom * 2.0
                        candidates.append((other_idx, distance, standard_bond_angstrom,
                                         distance <= threshold_1_3, distance <= threshold_2_0))
            
            # 按距离排序
            candidates.sort(key=lambda x: x[1])
            print(f"    最近的C原子候选:")
            for other_idx, dist, std_bond, in_1_3, in_2_0 in candidates[:5]:
                other_atom = mol.GetAtomWithIdx(other_idx)
                other_val = calculate_atom_valency(mol, other_idx)
                other_max_val = get_atom_max_valency(other_atom.GetAtomicNum())
                print(f"      C{other_idx}: 距离={dist:.3f}Å, 标准键长={std_bond:.3f}Å, "
                      f"1.3倍阈值内={in_1_3}, 2.0倍阈值内={in_2_0}, "
                      f"价态={other_val}/{other_max_val}")
    
    print("\n" + "=" * 80)

if __name__ == '__main__':
    analyze_molecule()
