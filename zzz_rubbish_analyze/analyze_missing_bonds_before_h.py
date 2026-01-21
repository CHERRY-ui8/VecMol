#!/usr/bin/env python3
"""
分析补H之前的分子状态，找出应该连接但未连接的键
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from rdkit import Chem
import numpy as np
from funcmol.utils.utils_base import (
    add_bonds_with_openbabel,
    xyz_to_sdf,
    calculate_atom_valency, 
    get_atom_max_valency,
    BOND_LENGTHS_PM
)

def analyze_before_h():
    # 读取原始文件（没有键）
    test_file = Path("/data/huayuchen/Neurl-voxel/exps/funcmol/fm_drugs/20260116/samples/20260116_version_2_epoch_9_better_converge/molecule/genmol_000000_2_847680.sdf")
    
    with open(test_file, 'r') as f:
        original_sdf = f.read()
    
    mol_original = Chem.MolFromMolBlock(original_sdf, sanitize=False)
    if mol_original is None:
        print("无法解析原始分子")
        return
    
    # 提取坐标和类型
    elements = ['C', 'H', 'O', 'N', 'F', 'S', 'Cl', 'Br']
    element_to_idx = {elem: idx for idx, elem in enumerate(elements)}
    
    conf = mol_original.GetConformer()
    n_atoms = mol_original.GetNumAtoms()
    coords = np.zeros((n_atoms, 3))
    types = np.zeros(n_atoms, dtype=int)
    
    for i, atom in enumerate(mol_original.GetAtoms()):
        if conf is not None:
            pos = conf.GetAtomPosition(i)
            coords[i] = [pos.x, pos.y, pos.z]
        
        atom_symbol = atom.GetSymbol()
        if atom_symbol in element_to_idx:
            types[i] = element_to_idx[atom_symbol]
        else:
            types[i] = 0  # 默认C
    
    # 使用OpenBabel添加键（不补H）
    sdf_with_bonds = add_bonds_with_openbabel(
        coords,
        types,
        elements,
        fallback_to_xyz_to_sdf=xyz_to_sdf,
        add_hydrogens=False
    )
    
    if not sdf_with_bonds:
        print("OpenBabel处理失败")
        return
    
    mol = Chem.MolFromMolBlock(sdf_with_bonds, sanitize=False)
    if mol is None:
        print("无法解析OpenBabel处理后的分子")
        return
    
    conf = mol.GetConformer()
    
    print("=" * 80)
    print("分析补H之前的分子状态")
    print("=" * 80)
    
    # 1. 找到所有N原子，检查它们的价态
    print("\n1. N原子分析（补H前）:")
    print("-" * 80)
    n_atoms_list = []
    for atom_idx in range(mol.GetNumAtoms()):
        atom = mol.GetAtomWithIdx(atom_idx)
        if atom.GetSymbol() == 'N':
            pos = conf.GetAtomPosition(atom_idx)
            current_valency = calculate_atom_valency(mol, atom_idx)
            max_valency = get_atom_max_valency(atom.GetAtomicNum())
            n_atoms_list.append((atom_idx, pos, current_valency, max_valency))
            print(f"  原子 {atom_idx}: 位置=({pos.x:.3f}, {pos.y:.3f}, {pos.z:.3f}), "
                  f"价态={current_valency}/{max_valency}")
            
            # 找到已连接的原子
            connected = []
            for bond in atom.GetBonds():
                other_idx = bond.GetOtherAtomIdx(atom_idx)
                other_atom = mol.GetAtomWithIdx(other_idx)
                other_pos = conf.GetAtomPosition(other_idx)
                bond_type = bond.GetBondType()
                connected.append((other_idx, other_atom.GetSymbol(), other_pos, bond_type))
            print(f"    已连接: {[(idx, sym, str(bt)) for idx, sym, _, bt in connected]}")
    
    # 2. 找到所有C原子，检查它们的价态和连接情况
    print("\n2. C原子分析（重点关注价态未饱和或只连接H的）:")
    print("-" * 80)
    c_atoms_list = []
    for atom_idx in range(mol.GetNumAtoms()):
        atom = mol.GetAtomWithIdx(atom_idx)
        if atom.GetSymbol() == 'C':
            pos = conf.GetAtomPosition(atom_idx)
            current_valency = calculate_atom_valency(mol, atom_idx)
            max_valency = get_atom_max_valency(atom.GetAtomicNum())
            
            # 找到已连接的原子
            connected = []
            h_count = 0
            for bond in atom.GetBonds():
                other_idx = bond.GetOtherAtomIdx(atom_idx)
                other_atom = mol.GetAtomWithIdx(other_idx)
                if other_atom.GetSymbol() == 'H':
                    h_count += 1
                connected.append((other_idx, other_atom.GetSymbol()))
            
            # 检查是否只连接了H（且H数量<=3）
            only_h = all(sym == 'H' for _, sym in connected) and len(connected) <= 3
            
            if current_valency < max_valency or only_h:
                c_atoms_list.append((atom_idx, pos, current_valency, max_valency, h_count, connected, only_h))
                print(f"  原子 {atom_idx}: 位置=({pos.x:.3f}, {pos.y:.3f}, {pos.z:.3f}), "
                      f"价态={current_valency}/{max_valency}, H数={h_count}, 只连H={only_h}")
                print(f"    已连接: {connected}")
    
    # 3. 对于价态未饱和的N原子，找到最近的未连接的C原子
    print("\n3. 分析N原子应该连接的键:")
    print("-" * 80)
    for n_idx, n_pos, n_val, n_max_val in n_atoms_list:
        if n_val < n_max_val:
            print(f"\n  N原子 {n_idx} (价态 {n_val}/{n_max_val}, 位置=({n_pos.x:.3f}, {n_pos.y:.3f}, {n_pos.z:.3f})):")
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
                        c_val = calculate_atom_valency(mol, c_idx)
                        c_max_val = get_atom_max_valency(c_atom.GetAtomicNum())
                        candidates.append((c_idx, distance, standard_bond_angstrom, 
                                         distance <= threshold_1_3, distance <= threshold_2_0,
                                         c_val, c_max_val))
            
            # 按距离排序
            candidates.sort(key=lambda x: x[1])
            print(f"    最近的C原子候选:")
            for c_idx, dist, std_bond, in_1_3, in_2_0, c_val, c_max_val in candidates[:10]:
                c_pos = conf.GetAtomPosition(c_idx)
                print(f"      C{c_idx}: 位置=({c_pos.x:.3f}, {c_pos.y:.3f}, {c_pos.z:.3f}), "
                      f"距离={dist:.3f}Å, 标准键长={std_bond:.3f}Å, "
                      f"1.3倍阈值内={in_1_3}, 2.0倍阈值内={in_2_0}, "
                      f"价态={c_val}/{c_max_val}")
    
    # 4. 对于只连接H的C原子，找到最近的未连接的C原子
    print("\n4. 分析只连接H的C原子应该连接的键:")
    print("-" * 80)
    for c_idx, c_pos, c_val, c_max_val, h_count, connected, only_h in c_atoms_list:
        if only_h and h_count <= 3:
            print(f"\n  C原子 {c_idx} (只连接{h_count}个H, 价态 {c_val}/{c_max_val}, "
                  f"位置=({c_pos.x:.3f}, {c_pos.y:.3f}, {c_pos.z:.3f})):")
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
                        other_val = calculate_atom_valency(mol, other_idx)
                        other_max_val = get_atom_max_valency(other_atom.GetAtomicNum())
                        candidates.append((other_idx, distance, standard_bond_angstrom,
                                         distance <= threshold_1_3, distance <= threshold_2_0,
                                         other_val, other_max_val))
            
            # 按距离排序
            candidates.sort(key=lambda x: x[1])
            print(f"    最近的C原子候选:")
            for other_idx, dist, std_bond, in_1_3, in_2_0, other_val, other_max_val in candidates[:10]:
                other_pos = conf.GetAtomPosition(other_idx)
                print(f"      C{other_idx}: 位置=({other_pos.x:.3f}, {other_pos.y:.3f}, {other_pos.z:.3f}), "
                      f"距离={dist:.3f}Å, 标准键长={std_bond:.3f}Å, "
                      f"1.3倍阈值内={in_1_3}, 2.0倍阈值内={in_2_0}, "
                      f"价态={other_val}/{other_max_val}")
    
    print("\n" + "=" * 80)

if __name__ == '__main__':
    analyze_before_h()
