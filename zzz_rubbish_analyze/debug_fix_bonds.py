#!/usr/bin/env python3
"""
调试 fix_missing_bonds_from_sdf 函数，检查为什么特定键没有连接
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
    BOND_LENGTHS_PM,
    BOND_MARGIN1
)

def debug_specific_bonds():
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
    print("调试特定键的连接")
    print("=" * 80)
    
    # 检查 N49 和 C9
    print("\n1. 检查 N49 和 C9 的连接:")
    print("-" * 80)
    n49 = mol.GetAtomWithIdx(49)
    c9 = mol.GetAtomWithIdx(9)
    
    n49_pos = conf.GetAtomPosition(49)
    c9_pos = conf.GetAtomPosition(9)
    distance = np.linalg.norm(np.array([c9_pos.x, c9_pos.y, c9_pos.z]) - 
                              np.array([n49_pos.x, n49_pos.y, n49_pos.z]))
    
    print(f"  N49 位置: ({n49_pos.x:.3f}, {n49_pos.y:.3f}, {n49_pos.z:.3f})")
    print(f"  C9 位置: ({c9_pos.x:.3f}, {c9_pos.y:.3f}, {c9_pos.z:.3f})")
    print(f"  距离: {distance:.3f} Å")
    
    # 检查是否已连接
    already_connected = False
    for bond in n49.GetBonds():
        if bond.GetOtherAtomIdx(49) == 9:
            already_connected = True
            print(f"  已连接: 是 (键类型: {bond.GetBondType()})")
            break
    if not already_connected:
        print(f"  已连接: 否")
    
    # 检查标准键长和阈值
    standard_bond_pm = BOND_LENGTHS_PM.get('N', {}).get('C')
    if standard_bond_pm:
        standard_bond_angstrom = standard_bond_pm / 100.0
        threshold_1_3 = standard_bond_angstrom * 1.3
        threshold_2_0 = standard_bond_angstrom * 2.0
        threshold_with_margin = (standard_bond_pm + BOND_MARGIN1) / 100.0
        
        print(f"  标准N-C键长: {standard_bond_angstrom:.3f} Å ({standard_bond_pm} pm)")
        print(f"  1.3倍阈值: {threshold_1_3:.3f} Å")
        print(f"  2.0倍阈值: {threshold_2_0:.3f} Å")
        print(f"  标准键长+容差({BOND_MARGIN1}pm): {threshold_with_margin:.3f} Å")
        print(f"  距离是否在1.3倍阈值内: {distance <= threshold_1_3}")
        print(f"  距离是否在2.0倍阈值内: {distance <= threshold_2_0}")
        print(f"  距离是否在(标准+容差)内: {distance <= threshold_with_margin}")
    
    # 检查价态
    n49_val = calculate_atom_valency(mol, 49)
    n49_max = get_atom_max_valency(n49.GetAtomicNum())
    c9_val = calculate_atom_valency(mol, 9)
    c9_max = get_atom_max_valency(c9.GetAtomicNum())
    
    print(f"  N49 价态: {n49_val}/{n49_max}")
    print(f"  C9 价态: {c9_val}/{c9_max}")
    print(f"  添加单键后 N49 价态: {n49_val + 1}/{n49_max} (是否超限: {n49_val + 1 > n49_max})")
    print(f"  添加单键后 C9 价态: {c9_val + 1}/{c9_max} (是否超限: {c9_val + 1 > c9_max})")
    
    # 检查 C19 和 C15
    print("\n2. 检查 C19 和 C15 的连接:")
    print("-" * 80)
    c19 = mol.GetAtomWithIdx(19)
    c15 = mol.GetAtomWithIdx(15)
    
    c19_pos = conf.GetAtomPosition(19)
    c15_pos = conf.GetAtomPosition(15)
    distance = np.linalg.norm(np.array([c15_pos.x, c15_pos.y, c15_pos.z]) - 
                              np.array([c19_pos.x, c19_pos.y, c19_pos.z]))
    
    print(f"  C19 位置: ({c19_pos.x:.3f}, {c19_pos.y:.3f}, {c19_pos.z:.3f})")
    print(f"  C15 位置: ({c15_pos.x:.3f}, {c15_pos.y:.3f}, {c15_pos.z:.3f})")
    print(f"  距离: {distance:.3f} Å")
    
    # 检查是否已连接
    already_connected = False
    for bond in c19.GetBonds():
        if bond.GetOtherAtomIdx(19) == 15:
            already_connected = True
            print(f"  已连接: 是 (键类型: {bond.GetBondType()})")
            break
    if not already_connected:
        print(f"  已连接: 否")
    
    # 检查标准键长和阈值
    standard_bond_pm = BOND_LENGTHS_PM.get('C', {}).get('C')
    if standard_bond_pm:
        standard_bond_angstrom = standard_bond_pm / 100.0
        threshold_1_3 = standard_bond_angstrom * 1.3
        threshold_2_0 = standard_bond_angstrom * 2.0
        threshold_with_margin = (standard_bond_pm + BOND_MARGIN1) / 100.0
        
        print(f"  标准C-C键长: {standard_bond_angstrom:.3f} Å ({standard_bond_pm} pm)")
        print(f"  1.3倍阈值: {threshold_1_3:.3f} Å")
        print(f"  2.0倍阈值: {threshold_2_0:.3f} Å")
        print(f"  标准键长+容差({BOND_MARGIN1}pm): {threshold_with_margin:.3f} Å")
        print(f"  距离是否在1.3倍阈值内: {distance <= threshold_1_3}")
        print(f"  距离是否在2.0倍阈值内: {distance <= threshold_2_0}")
        print(f"  距离是否在(标准+容差)内: {distance <= threshold_with_margin}")
    
    # 检查价态
    c19_val = calculate_atom_valency(mol, 19)
    c19_max = get_atom_max_valency(c19.GetAtomicNum())
    c15_val = calculate_atom_valency(mol, 15)
    c15_max = get_atom_max_valency(c15.GetAtomicNum())
    
    print(f"  C19 价态: {c19_val}/{c19_max}")
    print(f"  C15 价态: {c15_val}/{c15_max}")
    print(f"  添加单键后 C19 价态: {c19_val + 1}/{c19_max} (是否超限: {c19_val + 1 > c19_max})")
    print(f"  添加单键后 C15 价态: {c15_val + 1}/{c15_max} (是否超限: {c15_val + 1 > c15_max})")
    
    # 检查C19连接的原子
    print(f"  C19 已连接的原子: {[mol.GetAtomWithIdx(bond.GetOtherAtomIdx(19)).GetSymbol() for bond in c19.GetBonds()]}")
    
    print("\n" + "=" * 80)

if __name__ == '__main__':
    debug_specific_bonds()
