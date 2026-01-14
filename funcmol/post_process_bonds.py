#!/usr/bin/env python3
"""
OpenBabel连键后处理脚本

从包含不带键SDF文件的目录中读取分子，使用OpenBabel添加化学键，
并保存为带键的SDF文件。

用法: python post_process_bonds.py
"""

# 硬编码的输入和输出目录
input_dir = "/datapool/data2/home/pxg/data/hyc/funcmol-main-neuralfield/exps/funcmol/fm_qm9/20260105/samples/20260105_version_1_last_withbonds/molecule"
# output_dir = "/datapool/data2/home/pxg/data/hyc/funcmol-main-neuralfield/exps/funcmol/fm_qm9/20251222/samples/20251222_version_0_last_withbonds/molecule"
output_dir = "/datapool/data2/home/pxg/data/hyc/funcmol-main-neuralfield/exps/funcmol/fm_qm9/20260105/samples/20260105_version_1_last_withbonds_addH/molecule"
add_hydrogens = True # 是否自动补齐缺少的H原子（True: 补齐, False: 不补齐）

from pathlib import Path
from tqdm import tqdm
from funcmol.utils.utils_base import add_bonds_with_openbabel, xyz_to_sdf

# 禁用RDKit警告
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


def process_sdf_file(input_sdf_path: Path, output_sdf_path: Path, elements: list, add_hydrogens: bool = True) -> bool:
    """
    处理单个SDF文件，添加OpenBabel键
    
    Args:
        input_sdf_path: 输入SDF文件路径（不带键）
        output_sdf_path: 输出SDF文件路径（带键）
        elements: 元素列表，例如 ["C", "H", "O", "N", "F"]
        add_hydrogens: 是否自动补齐缺少的H原子（True: 补齐, False: 不补齐）
        
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
        
        # 使用OpenBabel添加键
        sdf_with_bonds = add_bonds_with_openbabel(
            coords,
            types,
            elements,
            fallback_to_xyz_to_sdf=xyz_to_sdf,
            add_hydrogens=add_hydrogens
        )
        
        if not sdf_with_bonds:
            print(f"警告: OpenBabel处理失败 {input_sdf_path}")
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
    
    # 默认元素列表
    elements = ['C', 'H', 'O', 'N', 'F']
    
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
        
        if process_sdf_file(sdf_file, output_path, elements, add_hydrogens=add_hydrogens):
            success_count += 1
    
    print("="*80)
    print(f"处理完成: {success_count}/{len(sdf_files)} 个文件成功处理")


if __name__ == '__main__':
    main()

