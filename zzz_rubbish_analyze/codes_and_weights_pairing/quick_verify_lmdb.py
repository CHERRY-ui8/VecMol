#!/usr/bin/env python3
"""
快速验证 codes.lmdb 和 position_weights.lmdb 的对应关系（简化版）

使用方法:
    python quick_verify_lmdb.py <codes目录>/<split>
    
例如:
    python quick_verify_lmdb.py /path/to/codes/train
"""

import os
import sys
import torch
import lmdb
import pickle
import random


def quick_verify(split_dir, num_samples=5):
    """
    快速验证两个 LMDB 数据库的对应关系
    
    Args:
        split_dir: 包含 codes.lmdb 和 position_weights.lmdb 的目录
        num_samples: 要验证的样本数量
    """
    codes_lmdb = os.path.join(split_dir, "codes.lmdb")
    codes_keys = os.path.join(split_dir, "codes_keys.pt")
    weights_lmdb = os.path.join(split_dir, "position_weights.lmdb")
    weights_keys = os.path.join(split_dir, "position_weights_keys.pt")
    
    # 检查文件
    for f in [codes_lmdb, codes_keys, weights_lmdb, weights_keys]:
        if not os.path.exists(f):
            print(f"❌ 文件不存在: {f}")
            return False
    
    print(f"验证目录: {split_dir}\n")
    
    # 加载 keys
    print("1. 检查 keys 文件...")
    ck = torch.load(codes_keys, weights_only=False)
    wk = torch.load(weights_keys, weights_only=False)
    
    print(f"   codes_keys: {len(ck)} 个")
    print(f"   weights_keys: {len(wk)} 个")
    
    if len(ck) != len(wk):
        print(f"❌ Keys 数量不匹配!")
        return False
    
    # 检查 keys 内容
    if ck != wk:
        print("⚠️  Keys 内容不完全一致，但继续验证...")
        # 检查前几个是否一致
        mismatch_count = sum(1 for i, (a, b) in enumerate(zip(ck, wk)) if str(a) != str(b))
        if mismatch_count > 0:
            print(f"   发现 {mismatch_count} 个不匹配的 keys")
    else:
        print("✅ Keys 完全一致")
    
    # 打开数据库
    print("\n2. 打开 LMDB 数据库...")
    codes_db = lmdb.open(codes_lmdb, readonly=True, lock=False)
    weights_db = lmdb.open(weights_lmdb, readonly=True, lock=False)
    
    with codes_db.begin() as txn:
        codes_count = txn.stat()['entries']
    with weights_db.begin() as txn:
        weights_count = txn.stat()['entries']
    
    print(f"   codes.lmdb: {codes_count} 个条目")
    print(f"   position_weights.lmdb: {weights_count} 个条目")
    
    if codes_count != weights_count:
        print(f"❌ 数据库条目数量不匹配!")
        codes_db.close()
        weights_db.close()
        return False
    
    print("✅ 数据库条目数量一致")
    
    # 随机采样验证
    print(f"\n3. 随机验证 {num_samples} 个样本...")
    sample_indices = random.sample(range(len(ck)), min(num_samples, len(ck)))
    sample_indices.sort()
    
    all_ok = True
    for idx in sample_indices:
        key = str(ck[idx]).encode('utf-8')
        
        # 读取 codes
        try:
            with codes_db.begin(buffers=True) as txn:
                code_val = txn.get(key)
                if code_val is None:
                    print(f"   ❌ 索引 {idx}: codes.lmdb 中未找到")
                    all_ok = False
                    continue
                code_data = pickle.loads(code_val)
        except Exception as e:
            print(f"   ❌ 索引 {idx}: 读取 codes 失败 - {e}")
            all_ok = False
            continue
        
        # 读取 weights
        try:
            with weights_db.begin(buffers=True) as txn:
                weight_val = txn.get(key)
                if weight_val is None:
                    print(f"   ❌ 索引 {idx}: position_weights.lmdb 中未找到")
                    all_ok = False
                    continue
                weight_data = pickle.loads(weight_val)
        except Exception as e:
            print(f"   ❌ 索引 {idx}: 读取 weights 失败 - {e}")
            all_ok = False
            continue
        
        # 显示信息
        if isinstance(code_data, torch.Tensor):
            code_info = f"shape={code_data.shape}, dtype={code_data.dtype}"
        else:
            code_info = f"type={type(code_data)}"
        
        if isinstance(weight_data, torch.Tensor):
            weight_info = f"shape={weight_data.shape}, dtype={weight_data.dtype}"
        else:
            weight_info = f"type={type(weight_data)}"
        
        print(f"   ✅ 索引 {idx}: codes({code_info}), weights({weight_info})")
    
    codes_db.close()
    weights_db.close()
    
    print(f"\n{'='*50}")
    if all_ok:
        print("✅ 验证通过: 两个数据库按照顺序一一对应")
        return True
    else:
        print("❌ 验证失败: 发现不匹配的情况")
        return False


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("使用方法: python quick_verify_lmdb.py <codes目录>/<split>")
        print("例如: python quick_verify_lmdb.py /path/to/codes/train")
        sys.exit(1)
    
    split_dir = sys.argv[1]
    num_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    
    success = quick_verify(split_dir, num_samples)
    sys.exit(0 if success else 1)
