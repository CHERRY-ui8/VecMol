#!/usr/bin/env python3
"""
从drugs数据集的SDF文件中找出包含C、H、O、N、F、Cl、Br、S这八种元素的分子
"""
import sys
from rdkit import Chem
from pathlib import Path

# 目标元素集合
target_elements = {'C', 'H', 'O', 'N', 'F', 'Cl', 'Br', 'S'}

# SDF文件路径（可以通过命令行参数指定，默认为val.sdf）
if len(sys.argv) > 1:
    sdf_file = Path(sys.argv[1])
else:
    sdf_file = Path('funcmol/dataset/data/drugs/val.sdf')

if not sdf_file.exists():
    print(f"错误: 文件不存在 {sdf_file}")
    exit(1)

print(f"正在读取 {sdf_file}...")

# 读取SDF文件
supplier = Chem.SDMolSupplier(str(sdf_file), sanitize=False)
results = []

for idx, mol in enumerate(supplier):
    if mol is None:
        continue
    
    # 获取分子中所有元素的集合
    elements = set()
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        elements.add(symbol)
    
    # 检查是否包含所有8种目标元素
    if target_elements.issubset(elements):
        # 统计每种元素的数量
        element_counts = {}
        for atom in mol.GetAtoms():
            symbol = atom.GetSymbol()
            element_counts[symbol] = element_counts.get(symbol, 0) + 1
        
        # 获取分子名称（如果有的话）
        mol_name = mol.GetProp('_Name') if mol.HasProp('_Name') else f"mol_{idx}"
        
        results.append({
            'index': idx,
            'name': mol_name,
            'element_counts': element_counts
        })

# 输出结果
print(f"\n找到 {len(results)} 个分子包含所有8种元素 (C、H、O、N、F、Cl、Br、S)")
print("\n这些分子的信息:")

if results:
    for r in results:
        print(f"\n索引={r['index']}, 名称={r['name']}")
        print(f"  元素计数: ", end="")
        counts_str = ", ".join([f"{elem}={r['element_counts'].get(elem, 0)}" 
                                for elem in sorted(target_elements)])
        print(counts_str)
else:
    print("没有找到符合条件的分子")
