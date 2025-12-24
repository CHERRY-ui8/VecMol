#!/usr/bin/env python3
"""
验证 codes.lmdb 和 position_weights.lmdb 之间是否按照顺序一一对应

使用方法:
    python verify_lmdb_correspondence.py --codes_dir <codes目录路径> --split <train/val/test>
    
例如:
    python verify_lmdb_correspondence.py --codes_dir /path/to/codes --split train
"""

import os
import pickle
import torch
import lmdb
import argparse
import random
from pathlib import Path


def verify_lmdb_correspondence(codes_dir, split, num_samples=10, random_sample=True):
    """
    验证 codes.lmdb 和 position_weights.lmdb 之间的对应关系
    
    Args:
        codes_dir: codes文件所在目录
        split: 数据分割名称 (train/val/test)
        num_samples: 要验证的样本数量
        random_sample: 是否随机采样验证
    """
    split_dir = os.path.join(codes_dir, split)
    
    # 检查文件是否存在
    codes_lmdb_path = os.path.join(split_dir, "codes.lmdb")
    codes_keys_path = os.path.join(split_dir, "codes_keys.pt")
    weights_lmdb_path = os.path.join(split_dir, "position_weights.lmdb")
    weights_keys_path = os.path.join(split_dir, "position_weights_keys.pt")
    
    print(f"\n{'='*60}")
    print(f"验证 LMDB 对应关系: {split}")
    print(f"{'='*60}\n")
    
    # 检查文件存在性
    files_exist = {
        'codes.lmdb': os.path.exists(codes_lmdb_path),
        'codes_keys.pt': os.path.exists(codes_keys_path),
        'position_weights.lmdb': os.path.exists(weights_lmdb_path),
        'position_weights_keys.pt': os.path.exists(weights_keys_path),
    }
    
    print("文件检查:")
    for name, exists in files_exist.items():
        status = "✅ 存在" if exists else "❌ 不存在"
        print(f"  {name}: {status}")
    
    if not all(files_exist.values()):
        print("\n❌ 错误: 缺少必要的文件，无法进行验证")
        return False
    
    # 加载 keys
    print(f"\n加载 keys 文件...")
    codes_keys = torch.load(codes_keys_path, weights_only=False)
    weights_keys = torch.load(weights_keys_path, weights_only=False)
    
    print(f"  codes_keys: {len(codes_keys)} 个条目")
    print(f"  position_weights_keys: {len(weights_keys)} 个条目")
    
    # 验证 keys 数量是否一致
    if len(codes_keys) != len(weights_keys):
        print(f"\n❌ 错误: keys 数量不匹配!")
        print(f"  codes: {len(codes_keys)}")
        print(f"  position_weights: {len(weights_keys)}")
        return False
    
    print(f"✅ keys 数量一致: {len(codes_keys)}")
    
    # 验证 keys 内容是否一致
    print(f"\n验证 keys 内容...")
    keys_match = True
    for i, (ck, wk) in enumerate(zip(codes_keys, weights_keys)):
        if str(ck) != str(wk):
            print(f"❌ 错误: 索引 {i} 的 key 不匹配!")
            print(f"  codes key: {ck}")
            print(f"  weights key: {wk}")
            keys_match = False
            break
    
    if keys_match:
        print(f"✅ 所有 keys 内容一致")
    else:
        print(f"❌ keys 内容不匹配，但继续验证数据...")
    
    # 打开 LMDB 数据库
    print(f"\n打开 LMDB 数据库...")
    codes_db = lmdb.open(
        codes_lmdb_path,
        map_size=10*(1024*1024*1024),
        create=False,
        subdir=True,
        readonly=True,
        lock=False,
        readahead=True,
        meminit=False,
        max_readers=1,
    )
    
    weights_db = lmdb.open(
        weights_lmdb_path,
        map_size=10*(1024*1024*1024),
        create=False,
        subdir=True,
        readonly=True,
        lock=False,
        readahead=True,
        meminit=False,
        max_readers=1,
    )
    
    # 验证数据库条目数量
    print(f"\n验证数据库条目数量...")
    with codes_db.begin() as txn:
        codes_stat = txn.stat()
        codes_entries = codes_stat['entries']
    
    with weights_db.begin() as txn:
        weights_stat = txn.stat()
        weights_entries = weights_stat['entries']
    
    print(f"  codes.lmdb: {codes_entries} 个条目")
    print(f"  position_weights.lmdb: {weights_entries} 个条目")
    
    if codes_entries != weights_entries:
        print(f"❌ 错误: 数据库条目数量不匹配!")
        codes_db.close()
        weights_db.close()
        return False
    
    if codes_entries != len(codes_keys):
        print(f"⚠️  警告: 数据库条目数量 ({codes_entries}) 与 keys 数量 ({len(codes_keys)}) 不匹配")
    else:
        print(f"✅ 数据库条目数量一致: {codes_entries}")
    
    # 随机或顺序采样验证
    print(f"\n采样验证 ({num_samples} 个样本)...")
    if random_sample:
        sample_indices = random.sample(range(len(codes_keys)), min(num_samples, len(codes_keys)))
        sample_indices.sort()  # 排序以便顺序读取
        print(f"  随机采样索引: {sample_indices}")
    else:
        sample_indices = list(range(min(num_samples, len(codes_keys))))
        print(f"  顺序采样索引: {sample_indices}")
    
    # 验证每个样本
    all_match = True
    for idx in sample_indices:
        key = codes_keys[idx]
        if isinstance(key, str):
            key_bytes = key.encode('utf-8')
        else:
            key_bytes = key
        
        # 从 codes.lmdb 读取
        try:
            with codes_db.begin(buffers=True) as txn:
                codes_value = txn.get(key_bytes)
                if codes_value is None:
                    print(f"❌ 索引 {idx}: codes.lmdb 中未找到 key '{key}'")
                    all_match = False
                    continue
                code_data = pickle.loads(codes_value)
        except Exception as e:
            print(f"❌ 索引 {idx}: 读取 codes.lmdb 失败: {e}")
            all_match = False
            continue
        
        # 从 position_weights.lmdb 读取
        try:
            with weights_db.begin(buffers=True) as txn:
                weights_value = txn.get(key_bytes)
                if weights_value is None:
                    print(f"❌ 索引 {idx}: position_weights.lmdb 中未找到 key '{key}'")
                    all_match = False
                    continue
                weight_data = pickle.loads(weights_value)
        except Exception as e:
            print(f"❌ 索引 {idx}: 读取 position_weights.lmdb 失败: {e}")
            all_match = False
            continue
        
        # 显示样本信息
        if isinstance(code_data, torch.Tensor):
            code_shape = code_data.shape
            code_dtype = code_data.dtype
            code_stats = f"min={code_data.min().item():.4f}, max={code_data.max().item():.4f}, mean={code_data.mean().item():.4f}"
        else:
            code_shape = "unknown"
            code_dtype = type(code_data)
            code_stats = "N/A"
        
        if isinstance(weight_data, torch.Tensor):
            weight_shape = weight_data.shape
            weight_dtype = weight_data.dtype
            weight_stats = f"min={weight_data.min().item():.4f}, max={weight_data.max().item():.4f}, mean={weight_data.mean().item():.4f}"
        else:
            weight_shape = "unknown"
            weight_dtype = type(weight_data)
            weight_stats = "N/A"
        
        print(f"  索引 {idx} (key='{key}'):")
        print(f"    codes: shape={code_shape}, dtype={code_dtype}, {code_stats}")
        print(f"    weights: shape={weight_shape}, dtype={weight_dtype}, {weight_stats}")
        print(f"    ✅ 两个数据库都能找到对应的数据")
    
    # 关闭数据库
    codes_db.close()
    weights_db.close()
    
    # 总结
    print(f"\n{'='*60}")
    if all_match and keys_match:
        print(f"✅ 验证通过: codes.lmdb 和 position_weights.lmdb 按照顺序一一对应")
        return True
    else:
        print(f"❌ 验证失败: 发现不匹配的情况")
        return False


def verify_all_keys(codes_dir, split):
    """
    验证所有 keys 是否都能在两个数据库中找到
    
    Args:
        codes_dir: codes文件所在目录
        split: 数据分割名称 (train/val/test)
    """
    split_dir = os.path.join(codes_dir, split)
    
    codes_lmdb_path = os.path.join(split_dir, "codes.lmdb")
    codes_keys_path = os.path.join(split_dir, "codes_keys.pt")
    weights_lmdb_path = os.path.join(split_dir, "position_weights.lmdb")
    weights_keys_path = os.path.join(split_dir, "position_weights_keys.pt")
    
    if not all(os.path.exists(p) for p in [codes_lmdb_path, codes_keys_path, weights_lmdb_path, weights_keys_path]):
        print("❌ 错误: 缺少必要的文件")
        return False
    
    codes_keys = torch.load(codes_keys_path, weights_only=False)
    weights_keys = torch.load(weights_keys_path, weights_only=False)
    
    print(f"\n验证所有 {len(codes_keys)} 个 keys...")
    
    codes_db = lmdb.open(
        codes_lmdb_path,
        map_size=10*(1024*1024*1024),
        create=False,
        subdir=True,
        readonly=True,
        lock=False,
        readahead=True,
        meminit=False,
        max_readers=1,
    )
    
    weights_db = lmdb.open(
        weights_lmdb_path,
        map_size=10*(1024*1024*1024),
        create=False,
        subdir=True,
        readonly=True,
        lock=False,
        readahead=True,
        meminit=False,
        max_readers=1,
    )
    
    missing_codes = []
    missing_weights = []
    
    for i, key in enumerate(codes_keys):
        if isinstance(key, str):
            key_bytes = key.encode('utf-8')
        else:
            key_bytes = key
        
        with codes_db.begin() as txn:
            if txn.get(key_bytes) is None:
                missing_codes.append(i)
        
        with weights_db.begin() as txn:
            if txn.get(key_bytes) is None:
                missing_weights.append(i)
        
        if (i + 1) % 1000 == 0:
            print(f"  已检查 {i + 1}/{len(codes_keys)} 个 keys...")
    
    codes_db.close()
    weights_db.close()
    
    if missing_codes:
        print(f"❌ codes.lmdb 中缺失 {len(missing_codes)} 个 keys: {missing_codes[:10]}...")
    else:
        print(f"✅ codes.lmdb 包含所有 keys")
    
    if missing_weights:
        print(f"❌ position_weights.lmdb 中缺失 {len(missing_weights)} 个 keys: {missing_weights[:10]}...")
    else:
        print(f"✅ position_weights.lmdb 包含所有 keys")
    
    return len(missing_codes) == 0 and len(missing_weights) == 0


def main():
    parser = argparse.ArgumentParser(description="验证 codes.lmdb 和 position_weights.lmdb 的对应关系")
    parser.add_argument("--codes_dir", type=str, required=True,
                       help="Codes文件所在目录")
    parser.add_argument("--split", type=str, required=True,
                       choices=["train", "val", "test"],
                       help="数据分割名称")
    parser.add_argument("--num_samples", type=int, default=10,
                       help="要验证的样本数量（默认: 10）")
    parser.add_argument("--random_sample", action="store_true", default=True,
                       help="是否随机采样验证（默认: True）")
    parser.add_argument("--verify_all", action="store_true",
                       help="验证所有 keys（较慢，但更彻底）")
    
    args = parser.parse_args()
    
    # 基本验证
    success = verify_lmdb_correspondence(
        args.codes_dir,
        args.split,
        num_samples=args.num_samples,
        random_sample=args.random_sample
    )
    
    # 如果基本验证通过且用户要求，进行完整验证
    if success and args.verify_all:
        print(f"\n{'='*60}")
        print("进行完整验证（检查所有 keys）...")
        print(f"{'='*60}")
        verify_all_keys(args.codes_dir, args.split)
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
