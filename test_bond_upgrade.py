#!/usr/bin/env python3
"""
测试单键升级为双键的功能
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from rdkit import Chem
from funcmol.utils.utils_base import (
    add_bonds_with_openbabel,
    xyz_to_sdf,
    fix_missing_bonds_from_sdf,
    calculate_atom_valency,
    get_atom_max_valency
)
import numpy as np

def test_bond_upgrade():
    # 读取原始文件
    test_file = Path("/data/huayuchen/Neurl-voxel/exps/funcmol/fm_drugs/20260116/samples/20260116_version_2_epoch_9_better_converge/molecule/genmol_000000_2_847680.sdf")
    
    with open(test_file, 'r') as f:
        original_sdf = f.read()
    
    mol_original = Chem.MolFromMolBlock(original_sdf, sanitize=False)
    conf = mol_original.GetConformer()
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
    
    # OpenBabel处理
    sdf_with_bonds = add_bonds_with_openbabel(coords, types, elements, fallback_to_xyz_to_sdf=xyz_to_sdf, add_hydrogens=False)
    mol_before = Chem.MolFromMolBlock(sdf_with_bonds, sanitize=False)
    
    print("=" * 80)
    print("测试单键升级为双键功能")
    print("=" * 80)
    print(f"OpenBabel处理后: {mol_before.GetNumAtoms()} 原子, {mol_before.GetNumBonds()} 键")
    
    # 统计单键和双键数量
    single_bonds = 0
    double_bonds = 0
    for bond in mol_before.GetBonds():
        if bond.GetBondType() == Chem.BondType.SINGLE:
            single_bonds += 1
        elif bond.GetBondType() == Chem.BondType.DOUBLE:
            double_bonds += 1
    
    print(f"  单键: {single_bonds}, 双键: {double_bonds}")
    
    # 修复和升级
    sdf_fixed = fix_missing_bonds_from_sdf(sdf_with_bonds, bond_length_ratio=1.5, max_iterations=10)
    mol_after = Chem.MolFromMolBlock(sdf_fixed, sanitize=False)
    
    print(f"\n修复和升级后: {mol_after.GetNumAtoms()} 原子, {mol_after.GetNumBonds()} 键")
    
    # 统计单键和双键数量
    single_bonds_after = 0
    double_bonds_after = 0
    for bond in mol_after.GetBonds():
        if bond.GetBondType() == Chem.BondType.SINGLE:
            single_bonds_after += 1
        elif bond.GetBondType() == Chem.BondType.DOUBLE:
            double_bonds_after += 1
    
    print(f"  单键: {single_bonds_after}, 双键: {double_bonds_after}")
    print(f"  单键变化: {single_bonds_after - single_bonds}, 双键变化: {double_bonds_after - double_bonds}")
    
    # 找出被升级的键
    print(f"\n被升级的键:")
    before_bonds = {}
    for bond in mol_before.GetBonds():
        begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        key = (min(begin, end), max(begin, end))
        before_bonds[key] = bond.GetBondType()
    
    upgraded_count = 0
    for bond in mol_after.GetBonds():
        begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        key = (min(begin, end), max(begin, end))
        if key in before_bonds:
            if before_bonds[key] == Chem.BondType.SINGLE and bond.GetBondType() == Chem.BondType.DOUBLE:
                begin_atom = mol_after.GetAtomWithIdx(begin)
                end_atom = mol_after.GetAtomWithIdx(end)
                begin_pos = mol_after.GetConformer().GetAtomPosition(begin)
                end_pos = mol_after.GetConformer().GetAtomPosition(end)
                distance = np.linalg.norm(np.array([end_pos.x, end_pos.y, end_pos.z]) - np.array([begin_pos.x, begin_pos.y, begin_pos.z]))
                print(f"  {begin_atom.GetSymbol()}{begin} - {end_atom.GetSymbol()}{end}: 距离={distance:.3f}Å")
                upgraded_count += 1
    
    if upgraded_count == 0:
        print("  (无)")
    
    print("\n" + "=" * 80)

if __name__ == '__main__':
    test_bond_upgrade()
