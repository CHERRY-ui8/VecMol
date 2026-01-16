#!/usr/bin/env python3
"""
将大型LMDB文件分割成多个小文件，以提高数据加载速度
每个分片可以独立访问，减少多进程竞争
"""

import os
import pickle
import torch
import lmdb
from tqdm import tqdm
import argparse
import shutil
import math


def split_lmdb(input_lmdb_path, input_keys_path, output_dir, num_shards=8):
    """
    将大型LMDB文件分割成多个小文件
    
    Args:
        input_lmdb_path: 输入LMDB文件路径
        input_keys_path: 输入keys文件路径
        output_dir: 输出目录
        num_shards: 分片数量
    """
    print(f"Splitting LMDB: {input_lmdb_path}")
    print(f"Output directory: {output_dir}")
    print(f"Number of shards: {num_shards}")
    
    # 加载keys
    keys = torch.load(input_keys_path, weights_only=False)
    total_samples = len(keys)
    samples_per_shard = math.ceil(total_samples / num_shards)
    
    print(f"Total samples: {total_samples}")
    print(f"Samples per shard: ~{samples_per_shard}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 打开输入LMDB
    input_env = lmdb.open(
        input_lmdb_path,
        readonly=True,
        subdir=True,
        lock=False,
        readahead=True,
        max_readers=256
    )
    
    # 为每个分片创建LMDB
    shard_keys = []
    for shard_id in range(num_shards):
        start_idx = shard_id * samples_per_shard
        end_idx = min((shard_id + 1) * samples_per_shard, total_samples)
        
        if start_idx >= total_samples:
            break
        
        shard_keys_list = keys[start_idx:end_idx]
        shard_keys.append(shard_keys_list)
        
        # 创建分片LMDB路径
        shard_lmdb_path = os.path.join(output_dir, f"codes_aug1_shard{shard_id}.lmdb")
        shard_keys_path = os.path.join(output_dir, f"codes_aug1_shard{shard_id}_keys.pt")
        
        print(f"\nShard {shard_id}: samples {start_idx} to {end_idx-1} ({len(shard_keys_list)} samples)")
        
        # 删除已存在的文件
        if os.path.exists(shard_lmdb_path):
            if os.path.isdir(shard_lmdb_path):
                shutil.rmtree(shard_lmdb_path)
            else:
                os.remove(shard_lmdb_path)
        
        # 创建分片LMDB
        # 估算map_size：每个样本约0.5MB，加上20%余量
        estimated_size = len(shard_keys_list) * 0.5 * 1024 * 1024 * 1.2
        map_size = max(int(estimated_size), 10 * 1024**3)  # 至少10GB
        
        shard_env = lmdb.open(
            shard_lmdb_path,
            map_size=map_size,
            subdir=True,
            create=True,
            readonly=False
        )
        
        # 复制数据
        with shard_env.begin(write=True) as txn:
            for key_str in tqdm(shard_keys_list, desc=f"  Copying shard {shard_id}"):
                key_bytes = key_str.encode('utf-8') if isinstance(key_str, str) else key_str
                
                with input_env.begin() as input_txn:
                    value = input_txn.get(key_bytes)
                    if value is None:
                        print(f"Warning: Key {key_str} not found in input LMDB")
                        continue
                
                txn.put(key_bytes, value)
        
        shard_env.close()
        
        # 保存keys
        torch.save(shard_keys_list, shard_keys_path)
        print(f"  Saved: {shard_lmdb_path} ({len(shard_keys_list)} samples)")
    
    input_env.close()
    
    # 保存全局keys文件（包含所有分片信息）
    global_keys_path = os.path.join(output_dir, "codes_aug1_keys.pt")
    torch.save(keys, global_keys_path)
    
    # 保存分片信息
    shard_info = {
        'num_shards': num_shards,
        'total_samples': total_samples,
        'samples_per_shard': samples_per_shard,
        'shard_keys': shard_keys
    }
    shard_info_path = os.path.join(output_dir, "shard_info.pt")
    torch.save(shard_info, shard_info_path)
    
    print(f"\n✓ Split completed!")
    print(f"  Total shards: {num_shards}")
    print(f"  Shard info saved to: {shard_info_path}")
    print(f"  Global keys saved to: {global_keys_path}")


def main():
    parser = argparse.ArgumentParser(description="Split large LMDB file into multiple shards")
    parser.add_argument("--input_lmdb", type=str, required=True, help="Input LMDB file path")
    parser.add_argument("--input_keys", type=str, required=True, help="Input keys file path")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for shards")
    parser.add_argument("--num_shards", type=int, default=8, help="Number of shards to create")
    
    args = parser.parse_args()
    
    split_lmdb(
        args.input_lmdb,
        args.input_keys,
        args.output_dir,
        args.num_shards
    )


if __name__ == "__main__":
    main()

