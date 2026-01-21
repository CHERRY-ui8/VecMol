#!/usr/bin/env python3
"""
验证修复后的结果
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from rdkit import Chem
from funcmol.utils.utils_base import (
    add_bonds_with_openbabel,
    xyz_to_sdf,
    fix_missing_bonds_from_sdf
)

def verify_fix():
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
    
    # 修复缺失的键
    sdf_fixed = fix_missing_bonds_from_sdf(sdf_with_bonds, bond_length_ratio=2.0, max_iterations=10)
    
    mol_before = Chem.MolFromMolBlock(sdf_with_bonds, sanitize=False)
    mol_after = Chem.MolFromMolBlock(sdf_fixed, sanitize=False)
    
    print("=" * 80)
    print("验证修复结果")
    print("=" * 80)
    
    # 检查 N49 和 C9
    print("\n1. N49 和 C9:")
    n49_before = mol_before.GetAtomWithIdx(49)
    n49_after = mol_after.GetAtomWithIdx(49)
    
    connected_before = [mol_before.GetAtomWithIdx(bond.GetOtherAtomIdx(49)).GetSymbol() 
                       for bond in n49_before.GetBonds()]
    connected_after = [mol_after.GetAtomWithIdx(bond.GetOtherAtomIdx(49)).GetSymbol() 
                      for bond in n49_after.GetBonds()]
    
    print(f"  修复前 N49 连接的原子: {connected_before}")
    print(f"  修复后 N49 连接的原子: {connected_after}")
    
    # 检查是否连接到C9
    c9_connected = False
    for bond in n49_after.GetBonds():
        if bond.GetOtherAtomIdx(49) == 9:
            c9_connected = True
            print(f"  ✓ N49 已连接到 C9 (键类型: {bond.GetBondType()})")
            break
    if not c9_connected:
        print(f"  ✗ N49 未连接到 C9")
    
    # 检查 C19 和 C15
    print("\n2. C19 和 C15:")
    c19_before = mol_before.GetAtomWithIdx(19)
    c19_after = mol_after.GetAtomWithIdx(19)
    
    connected_before = [mol_before.GetAtomWithIdx(bond.GetOtherAtomIdx(19)).GetSymbol() 
                       for bond in c19_before.GetBonds()]
    connected_after = [mol_after.GetAtomWithIdx(bond.GetOtherAtomIdx(19)).GetSymbol() 
                      for bond in c19_after.GetBonds()]
    
    print(f"  修复前 C19 连接的原子: {connected_before}")
    print(f"  修复后 C19 连接的原子: {connected_after}")
    
    # 检查是否连接到C15
    c15_connected = False
    for bond in c19_after.GetBonds():
        if bond.GetOtherAtomIdx(19) == 15:
            c15_connected = True
            print(f"  ✓ C19 已连接到 C15 (键类型: {bond.GetBondType()})")
            break
    if not c15_connected:
        print(f"  ✗ C19 未连接到 C15")
    
    print(f"\n总键数: {mol_before.GetNumBonds()} -> {mol_after.GetNumBonds()}")
    print("=" * 80)

if __name__ == '__main__':
    import numpy as np
    verify_fix()
