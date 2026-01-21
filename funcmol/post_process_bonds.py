#!/usr/bin/env python3
"""
OpenBabel连键后处理脚本

从包含不带键SDF文件的目录中读取分子，使用OpenBabel添加化学键，
并保存为带键的SDF文件。

功能：
1. 使用OpenBabel从原子坐标推断化学键
2. 可选地自动补齐缺少的H原子（根据价态计算）
3. 去除没有键连接的孤立原子（如Cl、Br等）
4. 补H时使用标准键长（根据原子类型：C-H=1.09Å, O-H=0.96Å等）

用法: python post_process_bonds.py
"""

# 硬编码的输入和输出目录
# exp_date = "20260116"
# exp_version = "version_2"
# ckpt_name = "epoch_9"
# dataset_name = "fm_drugs"  # 数据集名称：fm_qm9 或 fm_drugs
# input_dir = f"/data/huayuchen/Neurl-voxel/exps/funcmol/{dataset_name}/{exp_date}/samples/{exp_date}_{exp_version}_{ckpt_name}/molecule"
# output_dir = f"/data/huayuchen/Neurl-voxel/exps/funcmol/{dataset_name}/{exp_date}/samples/{exp_date}_{exp_version}_{ckpt_name}_withbonds_addH/molecule"
# add_hydrogens = True # 是否自动补齐缺少的H原子（True: 补齐, False: 不补齐）
# keep_largest_component = False # 是否只保留最大连通分支（True: 只保留最大分支, False: 保留所有有键连接的片段）

exp_date = "20260117"
exp_version = "version_0"
ckpt_name = "epoch_269"
dataset_name = "fm_qm9"  # 数据集名称：fm_qm9 或 fm_drugs
# input_dir = '/data/huayuchen/Neurl-voxel/exps/funcmol/fm_drugs/20260116/samples/20260116_version_2_epoch_9_better_converge/molecule'
input_dir = f"/data/huayuchen/Neurl-voxel/exps/funcmol/{dataset_name}/{exp_date}/samples/{exp_date}_{exp_version}_{ckpt_name}/molecule"
# output_dir = '/data/huayuchen/Neurl-voxel/exps/funcmol/fm_drugs/20260116/samples/20260116_version_2_epoch_9_better_converge_withbonds_addH_fix_2.0_from_small_frag/molecule'
output_dir = f"/data/huayuchen/Neurl-voxel/exps/funcmol/{dataset_name}/{exp_date}/samples/{exp_date}_{exp_version}_{ckpt_name}_withbonds_addH_improved/molecule"
add_hydrogens = True # 是否自动补齐缺少的H原子（True: 补齐, False: 不补齐）
keep_largest_component = True # 是否只保留最大连通分支（True: 只保留最大分支, False: 保留所有有键连接的片段）
fix_missing_bonds = True # 是否使用fix_missing_bonds_from_sdf修复缺失的键（True: 使用, False: 直接补H）
bond_length_ratio = 1.5 # 键修复时的标准键长比例阈值（距离≤标准键长×此值时添加键）

import sys
from pathlib import Path
# 添加项目根目录到 Python 路径，以便导入 funcmol 模块
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from tqdm import tqdm
from funcmol.utils.utils_base import add_bonds_with_openbabel, xyz_to_sdf, add_hydrogens_manually_from_sdf, fix_missing_bonds_from_sdf

# 禁用RDKit警告
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


def remove_isolated_atoms_from_sdf(sdf_content: str) -> str:
    """
    从SDF内容中去除没有键连接的孤立原子
    
    
    Args:
        sdf_content: SDF格式字符串
        
    Returns:
        去除孤立原子后的SDF格式字符串
    """
    try:
        from rdkit import Chem
        
        # 解析SDF
        mol = Chem.MolFromMolBlock(sdf_content, sanitize=False)
        if mol is None:
            return sdf_content
        
        # 如果没有键，检查是否所有原子都是孤立的
        if mol.GetNumBonds() == 0:
            # 如果只有一个原子，保留它（可能是单原子分子）
            if mol.GetNumAtoms() == 1:
                return sdf_content
            # 否则返回空（所有原子都是孤立的）
            return ""
        
        # 找到所有有键连接的原子
        atoms_with_bonds = set()
        for bond in mol.GetBonds():
            atoms_with_bonds.add(bond.GetBeginAtomIdx())
            atoms_with_bonds.add(bond.GetEndAtomIdx())
        
        # 如果所有原子都有键，直接返回
        if len(atoms_with_bonds) == mol.GetNumAtoms():
            return sdf_content
        
        # 如果没有有效键的原子，返回空
        if len(atoms_with_bonds) == 0:
            return ""
        
        # 创建新的分子，只包含有键的原子
        new_mol = Chem.RWMol()
        
        # 创建原子索引映射（旧索引 -> 新索引）
        atom_map = {}
        for old_idx in sorted(atoms_with_bonds):
            atom = mol.GetAtomWithIdx(old_idx)
            new_idx = new_mol.AddAtom(atom)
            atom_map[old_idx] = new_idx
        
        # 添加键（只添加两个原子都在新分子中的键）
        for bond in mol.GetBonds():
            begin_old = bond.GetBeginAtomIdx()
            end_old = bond.GetEndAtomIdx()
            if begin_old in atom_map and end_old in atom_map:
                begin_new = atom_map[begin_old]
                end_new = atom_map[end_old]
                new_mol.AddBond(begin_new, end_new, bond.GetBondType())
        
        # 转换为普通分子对象
        new_mol = new_mol.GetMol()
        
        # 复制坐标
        if mol.GetNumConformers() > 0:
            from rdkit.Geometry import Point3D
            conf = mol.GetConformer()
            new_conf = Chem.Conformer(new_mol.GetNumAtoms())
            for old_idx, new_idx in atom_map.items():
                pos = conf.GetAtomPosition(old_idx)
                new_conf.SetAtomPosition(new_idx, pos)
            new_mol.AddConformer(new_conf)
        
        # 转换回SDF格式
        try:
            sdf_block = Chem.MolToMolBlock(new_mol)
            if sdf_block:
                if not sdf_block.strip().endswith("$$$$"):
                    sdf_block = sdf_block.rstrip() + "\n$$$$\n"
                return sdf_block
        except Exception as e:
            print(f"警告: 转换去除孤立原子后的分子到SDF失败: {e}")
            return sdf_content
        
        return sdf_content
    except Exception as e:
        print(f"警告: 去除孤立原子失败: {e}")
        return sdf_content


def process_sdf_file(input_sdf_path: Path, output_sdf_path: Path, elements: list, add_hydrogens: bool = True, keep_largest_component: bool = False, fix_missing_bonds: bool = True, bond_length_ratio: float = 1.5) -> bool:
    """
    处理单个SDF文件，添加OpenBabel键
    
    Args:
        input_sdf_path: 输入SDF文件路径（不带键）
        output_sdf_path: 输出SDF文件路径（带键）
        elements: 元素列表，例如 ["C", "H", "O", "N", "F"]
        add_hydrogens: 是否自动补齐缺少的H原子（True: 补齐, False: 不补齐）
        keep_largest_component: 是否只保留最大连通分支（True: 只保留最大分支, False: 保留所有有键连接的片段）
        fix_missing_bonds: 是否使用fix_missing_bonds_from_sdf修复缺失的键（True: 使用, False: 直接补H）
        bond_length_ratio: 键修复时的标准键长比例阈值（距离≤标准键长×此值时添加键）
        
    Returns:
        bool: 是否成功处理
    """
    try:
        # 从SDF文件读取坐标和类型
        from rdkit import Chem
        import numpy as np
        
        supplier = Chem.SDMolSupplier(str(input_sdf_path), sanitize=False)
        mol = next(supplier, None)
        
        if mol is None:
            print(f"警告: 无法读取SDF文件 {input_sdf_path}")
            return False
        
        # 提取坐标和类型
        n_atoms = mol.GetNumAtoms()
        coords = np.zeros((n_atoms, 3))
        types = np.zeros(n_atoms, dtype=int)
        
        # 获取元素到索引的映射
        element_to_idx = {elem: idx for idx, elem in enumerate(elements)}
        
        conf = mol.GetConformer()
        for i, atom in enumerate(mol.GetAtoms()):
            if conf is not None:
                pos = conf.GetAtomPosition(i)
                coords[i] = [pos.x, pos.y, pos.z]
            
            # 根据原子符号找到对应的元素索引
            atom_symbol = atom.GetSymbol()
            if atom_symbol in element_to_idx:
                types[i] = element_to_idx[atom_symbol]
            else:
                print(f"警告: 未知元素 {atom_symbol}，跳过")
                return False
        
        # 使用OpenBabel添加键（先不补H，以便在补H之前排除单原子碎片）
        sdf_with_bonds = add_bonds_with_openbabel(
            coords,
            types,
            elements,
            fallback_to_xyz_to_sdf=xyz_to_sdf,
            add_hydrogens=False  # 先不补H，add_hydrogens_manually_from_sdf内部会排除单原子碎片后再补H
        )
        
        if not sdf_with_bonds:
            print(f"警告: OpenBabel处理失败 {input_sdf_path}")
            return False
        
        # 去除孤立原子（没有键连接的原子）
        sdf_with_bonds = remove_isolated_atoms_from_sdf(sdf_with_bonds)
        
        # 如果去除孤立原子后SDF为空，跳过
        if not sdf_with_bonds or sdf_with_bonds.strip() == "":
            print(f"警告: 去除孤立原子后分子为空 {input_sdf_path}")
            return False
        
        # 如果启用只保留最大连通分支，提取最大分支（在补H之前完成）
        if keep_largest_component:
            try:
                mol = Chem.MolFromMolBlock(sdf_with_bonds, sanitize=False)
                if mol is not None and mol.GetNumBonds() > 0:
                    # 获取所有连通分量
                    mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
                    if len(mol_frags) > 0:
                        # 选择原子数最多的连通分量
                        largest_mol = max(mol_frags, key=lambda m: m.GetNumAtoms())
                        # 转换回SDF格式
                        sdf_block = Chem.MolToMolBlock(largest_mol)
                        if sdf_block:
                            if not sdf_block.strip().endswith("$$$$"):
                                sdf_block = sdf_block.rstrip() + "\n$$$$\n"
                            sdf_with_bonds = sdf_block
            except Exception as e:
                print(f"警告: 提取最大连通分支失败: {e}")
            
            # 如果提取最大分支后SDF为空，跳过
            if not sdf_with_bonds or sdf_with_bonds.strip() == "":
                print(f"警告: 提取最大连通分支后分子为空 {input_sdf_path}")
                return False
        
        # 修复缺失的键（在补H之前）
        if fix_missing_bonds:
            from rdkit import Chem
            mol_before_fix = Chem.MolFromMolBlock(sdf_with_bonds, sanitize=False)
            n_bonds_before_fix = mol_before_fix.GetNumBonds() if mol_before_fix else 0
            
            sdf_fixed = fix_missing_bonds_from_sdf(sdf_with_bonds, bond_length_ratio=bond_length_ratio)
            if sdf_fixed is not None:
                mol_after_fix = Chem.MolFromMolBlock(sdf_fixed, sanitize=False)
                n_bonds_after_fix = mol_after_fix.GetNumBonds() if mol_after_fix else 0
                if n_bonds_after_fix != n_bonds_before_fix:
                    diff = n_bonds_after_fix - n_bonds_before_fix
                    if diff > 0:
                        print(f"键修复: 添加了 {diff} 个键 (bond_length_ratio={bond_length_ratio})")
                    else:
                        print(f"键修复: 键数变化 {diff} (可能升级了单键为双键) (bond_length_ratio={bond_length_ratio})")
                sdf_with_bonds = sdf_fixed
            
            # 修复键后，再次去除孤立原子（可能修复过程中连接了不应该连接的孤立原子）
            sdf_with_bonds = remove_isolated_atoms_from_sdf(sdf_with_bonds)
            
            # 如果去除孤立原子后SDF为空，跳过
            if not sdf_with_bonds or sdf_with_bonds.strip() == "":
                print(f"警告: 修复键后去除孤立原子，分子为空 {input_sdf_path}")
                return False
        
        # 补H原子（add_hydrogens_manually_from_sdf内部会在补H之前排除单原子碎片）
        if add_hydrogens:
            sdf_with_h = add_hydrogens_manually_from_sdf(sdf_with_bonds)
            if sdf_with_h is not None:
                sdf_with_bonds = sdf_with_h
            else:
                print(f"警告: 补H失败，继续使用未补H的版本 {input_sdf_path}")
        
        # 如果启用只保留最大连通分支，提取最大分支
        if keep_largest_component:
            try:
                from rdkit import Chem
                mol = Chem.MolFromMolBlock(sdf_with_bonds, sanitize=False)
                if mol is not None and mol.GetNumBonds() > 0:
                    # 获取所有连通分量
                    mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
                    if len(mol_frags) > 0:
                        # 选择原子数最多的连通分量
                        largest_mol = max(mol_frags, key=lambda m: m.GetNumAtoms())
                        # 转换回SDF格式
                        sdf_block = Chem.MolToMolBlock(largest_mol)
                        if sdf_block:
                            if not sdf_block.strip().endswith("$$$$"):
                                sdf_block = sdf_block.rstrip() + "\n$$$$\n"
                            sdf_with_bonds = sdf_block
            except Exception as e:
                print(f"警告: 提取最大连通分支失败: {e}")
            
            # 如果提取最大分支后SDF为空，跳过
            if not sdf_with_bonds or sdf_with_bonds.strip() == "":
                print(f"警告: 提取最大连通分支后分子为空 {input_sdf_path}")
                return False
        
        # 保存带键的SDF文件
        output_sdf_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_sdf_path, 'w', encoding='utf-8') as f:
            f.write(sdf_with_bonds)
        
        return True
        
    except Exception as e:
        print(f"错误: 处理 {input_sdf_path} 时出错: {e}")
        return False


def main():
    input_dir_path = Path(input_dir)
    output_dir_path = Path(output_dir)
    
    # 默认元素列表（与 drugs 数据集配置一致：C, H, O, N, F, S, Cl, Br）
    elements = ['C', 'H', 'O', 'N', 'F', 'S', 'Cl', 'Br']
    
    if not input_dir_path.exists():
        print(f"错误: 输入目录不存在: {input_dir_path}")
        return
    
    # 查找所有SDF文件（排除已处理的文件）
    sdf_files = []
    for sdf_file in sorted(input_dir_path.glob("*.sdf")):
        # 排除已经处理过的文件
        if '_obabel.sdf' in sdf_file.name or '_largest_cc.sdf' in sdf_file.name:
            continue
        sdf_files.append(sdf_file)
    
    if len(sdf_files) == 0:
        print(f"警告: 在 {input_dir_path} 中未找到SDF文件")
        return
    
    print(f"找到 {len(sdf_files)} 个SDF文件需要处理")
    print(f"输入目录: {input_dir_path}")
    print(f"输出目录: {output_dir_path}")
    print(f"元素列表: {elements}")
    print(f"自动补齐H原子: {add_hydrogens}")
    print(f"只保留最大连通分支: {keep_largest_component}")
    print(f"修复缺失的键: {fix_missing_bonds}")
    if fix_missing_bonds:
        print(f"键修复比例阈值: {bond_length_ratio}")
    print("="*80)
    
    success_count = 0
    for sdf_file in tqdm(sdf_files, desc="处理SDF文件"):
        # 生成输出文件名：genmol_XXXX.sdf -> genmol_XXXX_obabel.sdf
        output_name = sdf_file.stem + "_obabel.sdf"
        output_path = output_dir_path / output_name
        
        # 如果输出文件已存在，跳过
        if output_path.exists():
            print(f"跳过: {output_path.name} 已存在")
            continue
        
        if process_sdf_file(sdf_file, output_path, elements, add_hydrogens=add_hydrogens, keep_largest_component=keep_largest_component, fix_missing_bonds=fix_missing_bonds, bond_length_ratio=bond_length_ratio):
            success_count += 1
    
    print("="*80)
    print(f"处理完成: {success_count}/{len(sdf_files)} 个文件成功处理")


if __name__ == '__main__':
    main()

