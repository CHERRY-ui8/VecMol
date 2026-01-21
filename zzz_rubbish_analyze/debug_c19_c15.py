#!/usr/bin/env python3
"""
调试为什么 C19 和 C15 没有连接
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

def debug_c19_c15():
    # 读取原始文件
    test_file = Path("/data/huayuchen/Neurl-voxel/exps/funcmol/fm_drugs/20260116/samples/20260116_version_2_epoch_9_better_converge/molecule/genmol_000000_2_847680.sdf")
    
    with open(test_file, 'r') as f:
        original_sdf = f.read()
    
    mol_original = Chem.MolFromMolBlock(original_sdf, sanitize=False)
    conf = mol_original.GetConformer()
    
    # 提取坐标和类型
    elements = ['C', 'H', 'O', 'N', 'F', 'S', 'Cl', 'Br']
    element_to_idx = {elem: idx for idx, elem in enumerate(elements)}
    
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
            types[i] = 0
    
    # 使用OpenBabel添加键
    sdf_with_bonds = add_bonds_with_openbabel(
        coords, types, elements,
        fallback_to_xyz_to_sdf=xyz_to_sdf,
        add_hydrogens=False
    )
    
    mol = Chem.MolFromMolBlock(sdf_with_bonds, sanitize=False)
    conf = mol.GetConformer()
    
    print("=" * 80)
    print("调试 C19 和 C15 的连接")
    print("=" * 80)
    
    c19 = mol.GetAtomWithIdx(19)
    c19_pos = conf.GetAtomPosition(19)
    c19_pos_array = np.array([c19_pos.x, c19_pos.y, c19_pos.z])
    c19_val = calculate_atom_valency(mol, 19)
    c19_max = get_atom_max_valency(c19.GetAtomicNum())
    
    print(f"\nC19: 位置=({c19_pos.x:.3f}, {c19_pos.y:.3f}, {c19_pos.z:.3f}), 价态={c19_val}/{c19_max}")
    print(f"已连接的原子: {[mol.GetAtomWithIdx(bond.GetOtherAtomIdx(19)).GetSymbol() for bond in c19.GetBonds()]}")
    
    # 找到已连接的原子索引
    connected_to_c19 = set()
    for bond in c19.GetBonds():
        connected_to_c19.add(bond.GetOtherAtomIdx(19))
    
    # 找到所有未连接的C原子
    print(f"\n未连接的C原子候选:")
    candidates = []
    for other_idx in range(mol.GetNumAtoms()):
        if other_idx == 19 or other_idx in connected_to_c19:
            continue
        other_atom = mol.GetAtomWithIdx(other_idx)
        if other_atom.GetSymbol() == 'C':
            other_pos = conf.GetAtomPosition(other_idx)
            other_pos_array = np.array([other_pos.x, other_pos.y, other_pos.z])
            distance = np.linalg.norm(other_pos_array - c19_pos_array)
            
            # 获取标准键长
            standard_bond_pm = BOND_LENGTHS_PM.get('C', {}).get('C')
            if standard_bond_pm:
                standard_bond_angstrom = standard_bond_pm / 100.0
                threshold_2_0 = standard_bond_angstrom * 2.0
                other_val = calculate_atom_valency(mol, other_idx)
                other_max = get_atom_max_valency(other_atom.GetAtomicNum())
                
                candidates.append((other_idx, distance, threshold_2_0, other_val, other_max, distance <= threshold_2_0))
    
    # 按距离排序
    candidates.sort(key=lambda x: x[1])
    for other_idx, dist, threshold, other_val, other_max, in_threshold in candidates[:10]:
        other_pos = conf.GetAtomPosition(other_idx)
        print(f"  C{other_idx}: 位置=({other_pos.x:.3f}, {other_pos.y:.3f}, {other_pos.z:.3f}), "
              f"距离={dist:.3f}Å, 阈值={threshold:.3f}Å, 在阈值内={in_threshold}, "
              f"价态={other_val}/{other_max}")
        if other_idx == 15:
            print(f"    *** 这是C15 ***")
            print(f"    添加单键后 C19 价态: {c19_val + 1}/{c19_max} (超限: {c19_val + 1 > c19_max})")
            print(f"    添加单键后 C15 价态: {other_val + 1}/{other_max} (超限: {other_val + 1 > other_max})")
    
    print("\n" + "=" * 80)

if __name__ == '__main__':
    debug_c19_c15()
