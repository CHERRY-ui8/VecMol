#!/usr/bin/env python3
"""
将现有的.pth数据文件转换为LMDB格式以提高数据加载效率
"""

import os
import pickle
import torch
import lmdb
from tqdm import tqdm
import argparse
from funcmol.utils.constants import ELEMENTS_HASH, PADDING_INDEX


def convert_pth_to_lmdb(pth_path, lmdb_path, keys_path):
    """
    将.pth文件转换为LMDB数据库（简化版本）
    
    Args:
        pth_path (str): .pth文件路径
        lmdb_path (str): 输出LMDB数据库路径
        keys_path (str): 输出keys文件路径
    """
    print(f"Converting {pth_path} to LMDB format...")
    
    # 加载.pth文件
    data = torch.load(pth_path, weights_only=False)
    print(f"Loaded {len(data)} molecules from {pth_path}")
    
    # 检查数据结构
    if len(data) > 0:
        sample = data[0]
        print(f"Sample keys: {list(sample.keys()) if isinstance(sample, dict) else type(sample)}")
    
    # 删除旧的LMDB文件（如果存在）
    if os.path.exists(lmdb_path):
        print(f"Removing existing LMDB file: {lmdb_path}")
        if os.path.isdir(lmdb_path):
            import shutil
            shutil.rmtree(lmdb_path)
        else:
            os.remove(lmdb_path)
    
    # 删除旧的锁文件（如果存在）
    lock_file = lmdb_path + "-lock"
    if os.path.exists(lock_file):
        print(f"Removing existing lock file: {lock_file}")
        os.remove(lock_file)
    
    # 创建LMDB数据库 - 使用简化的参数
    db = lmdb.open(lmdb_path, map_size=20*(1024*1024*1024))
    
    n_valid = 0
    n_invalid = 0
    valid_keys = []  # 存储实际写入的有效键
    try:
        with db.begin(write=True) as txn:
            for i, datum in enumerate(tqdm(data, desc="Converting to LMDB")):
                # 使用索引作为key
                key = str(i).encode()
                
                # 检查数据完整性
                if not isinstance(datum, dict):
                    print(f"Warning: datum {i} is not a dict: {type(datum)}")
                    continue

                # check elements are valid - 只允许0-7的元素（C, H, O, N, F, S, Cl, Br）
                if "atoms_channel" in datum:
                    elements_valid = True
                    atoms = datum["atoms_channel"][datum["atoms_channel"] != PADDING_INDEX]
                    for atom in atoms.unique():
                        atom_id = int(atom.item())
                        # 只允许0-7的元素，过滤掉P(8), I(9), B(10)
                        if atom_id < 0 or atom_id > 7:
                            print(f"Warning: atom {atom} (id: {atom_id}) not in allowed range [0-7], filtering out")
                            elements_valid = False
                            break
                    if elements_valid:
                        n_valid += 1
                        valid_keys.append(str(i))  # 记录有效的键
                    else:
                        n_invalid += 1
                        continue

                # 序列化数据
                try:
                    value = pickle.dumps(datum)
                except Exception as e:
                    print(f"Error serializing datum {i}: {e}")
                    continue
                
                # 写入数据库
                txn.put(key, value)
        
    except Exception as e:
        print(f"Error during conversion: {e}")
        raise
    finally:
        db.close()
    
    # 保存可序列化的keys列表，使用实际的有效键
    torch.save(valid_keys, keys_path)
    
    print(f"Successfully converted to LMDB: {lmdb_path}")
    print(f"Saved keys list: {keys_path}")
    print(f"Database contains {n_valid} molecules")
    print(f"Valid molecules: {n_valid}, Invalid molecules: {n_invalid}")
    
    # 验证生成的LMDB文件
    print("Verifying generated LMDB file...")
    try:
        verify_db = lmdb.open(lmdb_path, readonly=True)
        with verify_db.begin() as txn:
            stat = txn.stat()
            print(f"LMDB verification: {stat['entries']} entries")
        verify_db.close()
        print("✅ LMDB file verification successful")
    except Exception as e:
        print(f"❌ LMDB file verification failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="Convert .pth files to LMDB format")
    parser.add_argument("--data_dir", type=str, required=True, 
                       help="数据目录路径")
    parser.add_argument("--dset_name", type=str, required=True,
                       choices=["qm9", "drugs", "cremp"],
                       help="数据集名称")
    parser.add_argument("--splits", type=str, nargs="+", 
                       default=["train", "val", "test"],
                       help="要转换的数据分割")
    
    args = parser.parse_args()
    
    data_dir = os.path.join(args.data_dir, args.dset_name)
    
    for split in args.splits:
        # 构建文件路径
        if args.dset_name == "cremp":
            fname = f"{split}_50_data"
        else:
            fname = f"{split}_data"
        
        pth_path = os.path.join(data_dir, f"{fname}.pth")
        lmdb_path = os.path.join(data_dir, f"{fname}.lmdb")
        keys_path = os.path.join(data_dir, f"{fname}_keys.pt")
        
        # 检查源文件是否存在
        if not os.path.exists(pth_path):
            print(f"Warning: {pth_path} does not exist, skipping...")
            continue
        
        # 转换文件
        convert_pth_to_lmdb(pth_path, lmdb_path, keys_path)


if __name__ == "__main__":
    main()
