#!/usr/bin/env python3
"""
详细测试 fix_missing_bonds_from_sdf 函数
直接使用已经 OpenBabel 处理过的 SDF 文件
"""

import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from funcmol.utils.utils_base import fix_missing_bonds_from_sdf
from rdkit import Chem

def test_fix_bonds_detailed():
    # 使用已经 OpenBabel 处理过的文件
    test_file = Path("/data/huayuchen/Neurl-voxel/exps/funcmol/fm_drugs/20260116/samples/20260116_version_2_epoch_9_better_converge_withbonds_addH/molecule/genmol_000000_2_847680_obabel.sdf")
    
    print("=" * 80)
    print("详细测试 fix_missing_bonds_from_sdf 函数")
    print("=" * 80)
    
    # 1. 读取已经 OpenBabel 处理过的SDF文件
    print(f"\n1. 读取已处理的SDF文件: {test_file}")
    if not test_file.exists():
        print(f"   错误: 文件不存在")
        return
    
    with open(test_file, 'r') as f:
        sdf_with_bonds = f.read()
    
    mol_before = Chem.MolFromMolBlock(sdf_with_bonds, sanitize=False)
    if mol_before is None:
        print("   错误: 无法解析SDF文件")
        return
    
    n_atoms_before = mol_before.GetNumAtoms()
    n_bonds_before = mol_before.GetNumBonds()
    print(f"   处理前: {n_atoms_before} 个原子, {n_bonds_before} 个键")
    
    # 分析价态未饱和的原子
    from funcmol.utils.utils_base import calculate_atom_valency, get_atom_max_valency
    unsaturated_before = []
    for atom_idx in range(mol_before.GetNumAtoms()):
        atom = mol_before.GetAtomWithIdx(atom_idx)
        atomic_num = atom.GetAtomicNum()
        current_valency = calculate_atom_valency(mol_before, atom_idx)
        max_valency = get_atom_max_valency(atomic_num)
        if current_valency < max_valency:
            unsaturated_before.append((atom_idx, atom.GetSymbol(), current_valency, max_valency))
    
    print(f"   价态未饱和的原子数量: {len(unsaturated_before)}")
    
    # 2. 测试不同的 bond_length_ratio
    for ratio in [1.3, 1.5, 2.0]:
        print(f"\n2. 测试 bond_length_ratio = {ratio}")
        print("-" * 80)
        
        sdf_fixed = fix_missing_bonds_from_sdf(sdf_with_bonds, bond_length_ratio=ratio, max_iterations=10)
        
        if sdf_fixed is None:
            print(f"   错误: 返回 None")
            continue
        
        if sdf_fixed == sdf_with_bonds:
            print(f"   ⚠️  返回的SDF与输入完全相同")
            continue
        
        mol_after = Chem.MolFromMolBlock(sdf_fixed, sanitize=False)
        if mol_after is None:
            print(f"   错误: 无法解析修复后的SDF")
            continue
        
        n_bonds_after = mol_after.GetNumBonds()
        print(f"   修复后键数: {n_bonds_after}")
        print(f"   新增键数: {n_bonds_after - n_bonds_before}")
        
        # 分析价态未饱和的原子
        unsaturated_after = []
        for atom_idx in range(mol_after.GetNumAtoms()):
            atom = mol_after.GetAtomWithIdx(atom_idx)
            atomic_num = atom.GetAtomicNum()
            current_valency = calculate_atom_valency(mol_after, atom_idx)
            max_valency = get_atom_max_valency(atomic_num)
            if current_valency < max_valency:
                unsaturated_after.append((atom_idx, atom.GetSymbol(), current_valency, max_valency))
        
        print(f"   修复后价态未饱和的原子数量: {len(unsaturated_after)}")
        
        if n_bonds_after > n_bonds_before:
            print(f"   ✓ 成功添加了 {n_bonds_after - n_bonds_before} 个键")
        else:
            print(f"   ⚠️  键数没有增加")
    
    # 3. 检查实际处理流程中的问题
    print(f"\n3. 检查实际处理流程")
    print("-" * 80)
    
    # 模拟 post_process_bonds.py 中的处理流程
    print("   模拟 post_process_bonds.py 的处理流程...")
    
    # 步骤1: OpenBabel处理（已经完成，使用现有文件）
    sdf_after_obabel = sdf_with_bonds
    
    # 步骤2: 修复缺失的键
    print("   步骤2: 调用 fix_missing_bonds_from_sdf...")
    sdf_after_fix = fix_missing_bonds_from_sdf(sdf_after_obabel, bond_length_ratio=2.0, max_iterations=10)
    
    if sdf_after_fix is None:
        print("   ⚠️  fix_missing_bonds_from_sdf 返回 None")
    elif sdf_after_fix == sdf_after_obabel:
        print("   ⚠️  fix_missing_bonds_from_sdf 返回的SDF与输入相同")
    else:
        mol_after_fix = Chem.MolFromMolBlock(sdf_after_fix, sanitize=False)
        if mol_after_fix:
            print(f"   ✓ 修复后: {mol_after_fix.GetNumBonds()} 个键 (原来 {n_bonds_before} 个)")
    
    # 步骤3: 补H（这里不实际补H，只是检查）
    print("   步骤3: 补H（跳过，仅检查）...")
    
    print("\n" + "=" * 80)

if __name__ == '__main__':
    test_fix_bonds_detailed()
