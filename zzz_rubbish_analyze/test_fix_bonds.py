#!/usr/bin/env python3
"""
测试 fix_missing_bonds_from_sdf 函数的脚本
"""

import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from funcmol.utils.utils_base import fix_missing_bonds_from_sdf, add_bonds_with_openbabel, xyz_to_sdf
from rdkit import Chem
import numpy as np

def test_fix_bonds():
    # 测试文件路径
    test_file = Path("/data/huayuchen/Neurl-voxel/exps/funcmol/fm_drugs/20260116/samples/20260116_version_2_epoch_9_better_converge/molecule/genmol_000000_2_847680.sdf")
    
    print("=" * 80)
    print("测试 fix_missing_bonds_from_sdf 函数")
    print("=" * 80)
    
    # 1. 读取原始SDF文件（没有键）
    print(f"\n1. 读取原始SDF文件: {test_file}")
    with open(test_file, 'r') as f:
        original_sdf = f.read()
    
    # 解析原始SDF获取坐标和类型
    mol_original = Chem.MolFromMolBlock(original_sdf, sanitize=False)
    if mol_original is None:
        print("错误: 无法解析原始SDF文件")
        return
    
    n_atoms_original = mol_original.GetNumAtoms()
    n_bonds_original = mol_original.GetNumBonds()
    print(f"   原始分子: {n_atoms_original} 个原子, {n_bonds_original} 个键")
    
    # 提取坐标和类型
    elements = ['C', 'H', 'O', 'N', 'F', 'S', 'Cl', 'Br']
    element_to_idx = {elem: idx for idx, elem in enumerate(elements)}
    
    coords = np.zeros((n_atoms_original, 3))
    types = np.zeros(n_atoms_original, dtype=int)
    
    conf = mol_original.GetConformer()
    for i, atom in enumerate(mol_original.GetAtoms()):
        if conf is not None:
            pos = conf.GetAtomPosition(i)
            coords[i] = [pos.x, pos.y, pos.z]
        
        atom_symbol = atom.GetSymbol()
        if atom_symbol in element_to_idx:
            types[i] = element_to_idx[atom_symbol]
        else:
            print(f"   警告: 未知元素 {atom_symbol}，使用C代替")
            types[i] = 0
    
    # 2. 使用OpenBabel添加键
    print(f"\n2. 使用OpenBabel添加键...")
    sdf_with_bonds = add_bonds_with_openbabel(
        coords,
        types,
        elements,
        fallback_to_xyz_to_sdf=xyz_to_sdf,
        add_hydrogens=False
    )
    
    if not sdf_with_bonds:
        print("   错误: OpenBabel处理失败")
        return
    
    mol_obabel = Chem.MolFromMolBlock(sdf_with_bonds, sanitize=False)
    if mol_obabel is None:
        print("   错误: 无法解析OpenBabel处理后的SDF")
        return
    
    n_bonds_obabel = mol_obabel.GetNumBonds()
    print(f"   OpenBabel处理后: {mol_obabel.GetNumAtoms()} 个原子, {n_bonds_obabel} 个键")
    
    # 分析价态未饱和的原子
    from funcmol.utils.utils_base import calculate_atom_valency, get_atom_max_valency
    unsaturated_before = []
    for atom_idx in range(mol_obabel.GetNumAtoms()):
        atom = mol_obabel.GetAtomWithIdx(atom_idx)
        atomic_num = atom.GetAtomicNum()
        current_valency = calculate_atom_valency(mol_obabel, atom_idx)
        max_valency = get_atom_max_valency(atomic_num)
        if current_valency < max_valency:
            unsaturated_before.append((atom_idx, atom.GetSymbol(), current_valency, max_valency))
    
    print(f"   价态未饱和的原子数量: {len(unsaturated_before)}")
    if len(unsaturated_before) > 0:
        print("   前10个未饱和原子:")
        for idx, symbol, curr, max_v in unsaturated_before[:10]:
            print(f"     原子 {idx} ({symbol}): 当前价态={curr}, 最大价态={max_v}")
    
    # 3. 调用 fix_missing_bonds_from_sdf
    print(f"\n3. 调用 fix_missing_bonds_from_sdf (bond_length_ratio=2.0)...")
    print(f"   输入SDF长度: {len(sdf_with_bonds)} 字符")
    
    # 添加调试：捕获异常
    import traceback
    try:
        sdf_fixed = fix_missing_bonds_from_sdf(sdf_with_bonds, bond_length_ratio=2.0, max_iterations=10)
    except Exception as e:
        print(f"   异常: {e}")
        traceback.print_exc()
        return
    
    if sdf_fixed is None:
        print("   错误: fix_missing_bonds_from_sdf 返回 None")
        return
    
    print(f"   输出SDF长度: {len(sdf_fixed)} 字符")
    
    if sdf_fixed == sdf_with_bonds:
        print("   ⚠️  警告: 返回的SDF与输入完全相同，可能没有添加任何键")
        print("   可能原因:")
        print("   1. 函数内部出现异常，返回了原始SDF")
        print("   2. 所有原子都已价态饱和")
        print("   3. 没有找到满足距离条件的原子对")
    else:
        print("   ✓ 返回的SDF与输入不同，可能添加了键")
    
    # 4. 分析修复后的结果
    print(f"\n4. 分析修复后的结果...")
    mol_fixed = Chem.MolFromMolBlock(sdf_fixed, sanitize=False)
    if mol_fixed is None:
        print("   错误: 无法解析修复后的SDF")
        return
    
    n_bonds_fixed = mol_fixed.GetNumBonds()
    print(f"   修复后: {mol_fixed.GetNumAtoms()} 个原子, {n_bonds_fixed} 个键")
    print(f"   新增键数: {n_bonds_fixed - n_bonds_obabel}")
    
    # 分析价态未饱和的原子
    unsaturated_after = []
    for atom_idx in range(mol_fixed.GetNumAtoms()):
        atom = mol_fixed.GetAtomWithIdx(atom_idx)
        atomic_num = atom.GetAtomicNum()
        current_valency = calculate_atom_valency(mol_fixed, atom_idx)
        max_valency = get_atom_max_valency(atomic_num)
        if current_valency < max_valency:
            unsaturated_after.append((atom_idx, atom.GetSymbol(), current_valency, max_valency))
    
    print(f"   修复后价态未饱和的原子数量: {len(unsaturated_after)}")
    if len(unsaturated_after) > 0:
        print("   前10个未饱和原子:")
        for idx, symbol, curr, max_v in unsaturated_after[:10]:
            print(f"     原子 {idx} ({symbol}): 当前价态={curr}, 最大价态={max_v}")
    
    # 5. 保存结果用于对比
    output_file = Path("/tmp/test_fix_bonds_result.sdf")
    with open(output_file, 'w') as f:
        f.write(sdf_fixed)
    print(f"\n5. 修复后的SDF已保存到: {output_file}")
    
    # 6. 详细对比
    print(f"\n6. 详细对比:")
    print(f"   OpenBabel键数: {n_bonds_obabel}")
    print(f"   修复后键数: {n_bonds_fixed}")
    print(f"   差异: {n_bonds_fixed - n_bonds_obabel}")
    
    if n_bonds_fixed == n_bonds_obabel:
        print("\n   ⚠️  警告: 键数没有变化，函数可能没有正常工作！")
        print("   可能的原因:")
        print("   1. 所有原子都已价态饱和")
        print("   2. 没有找到满足距离条件的原子对")
        print("   3. 函数内部出现错误但被捕获")
    else:
        print(f"\n   ✓ 成功添加了 {n_bonds_fixed - n_bonds_obabel} 个键")
    
    print("\n" + "=" * 80)

if __name__ == '__main__':
    test_fix_bonds()
