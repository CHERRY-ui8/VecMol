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


def convert_pth_to_lmdb(pth_path, lmdb_path, molid2idx_path):
    """
    将.pth文件转换为LMDB数据库
    
    Args:
        pth_path (str): .pth文件路径
        lmdb_path (str): 输出LMDB数据库路径
        molid2idx_path (str): 输出molid2idx映射文件路径
    """
    print(f"Converting {pth_path} to LMDB format...")
    
    # 加载.pth文件
    data = torch.load(pth_path, weights_only=False)
    print(f"Loaded {len(data)} molecules from {pth_path}")
    
    # 创建LMDB数据库
    db = lmdb.open(
        lmdb_path,
        map_size=10*(1024*1024*1024),   # 10GB
        create=True,
        subdir=False,
        readonly=False,
    )
    
    # 创建molid2idx映射
    molid2idx = {}
    
    with db.begin(write=True, buffers=True) as txn:
        for i, datum in enumerate(tqdm(data, desc="Converting to LMDB")):
            # 使用索引作为key
            key = str(i).encode()
            value = pickle.dumps(datum)
            
            txn.put(key, value)
            
            # 如果数据中有mol_id，使用mol_id作为映射
            if "mol_id" in datum:
                molid2idx[datum["mol_id"]] = i
            else:
                # 否则使用索引作为mol_id
                molid2idx[i] = i
    
    db.close()
    
    # 保存molid2idx映射
    torch.save(molid2idx, molid2idx_path)
    
    print(f"Successfully converted to LMDB: {lmdb_path}")
    print(f"Saved molid2idx mapping: {molid2idx_path}")
    print(f"Database contains {len(molid2idx)} molecules")


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
        molid2idx_path = os.path.join(data_dir, f"{fname}_molid2idx.pt")
        
        # 检查源文件是否存在
        if not os.path.exists(pth_path):
            print(f"Warning: {pth_path} does not exist, skipping...")
            continue
        
        # 检查目标文件是否已存在
        if os.path.exists(lmdb_path) and os.path.exists(molid2idx_path):
            print(f"Warning: {lmdb_path} already exists, skipping...")
            continue
        
        # 转换文件
        convert_pth_to_lmdb(pth_path, lmdb_path, molid2idx_path)


if __name__ == "__main__":
    main()
