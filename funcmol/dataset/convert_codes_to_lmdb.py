#!/usr/bin/env python3
"""
将现有的codes.pt或codes_XXX.pt文件转换为LMDB格式以提高数据加载效率
支持多个codes文件合并转换为单个LMDB数据库
"""

import os
import pickle
import gc
import torch
import lmdb
from tqdm import tqdm
import argparse
import shutil


def convert_codes_to_lmdb(codes_dir, split, lmdb_path, keys_path):
    """
    将codes文件转换为LMDB数据库
    
    Args:
        codes_dir (str): codes文件所在目录
        split (str): 数据分割名称 (train/val/test)
        lmdb_path (str): 输出LMDB数据库路径
        keys_path (str): 输出keys文件路径
    """
    split_dir = os.path.join(codes_dir, split)
    
    if not os.path.exists(split_dir):
        raise FileNotFoundError(f"Codes directory not found: {split_dir}")
    
    # 查找所有codes文件
    list_codes = [
        f for f in os.listdir(split_dir)
        if os.path.isfile(os.path.join(split_dir, f)) and \
        f.startswith("codes") and f.endswith(".pt")
    ]
    
    if not list_codes:
        raise FileNotFoundError(f"No codes files found in {split_dir}")
    
    # 优先使用 codes.pt（向后兼容）
    if "codes.pt" in list_codes:
        list_codes = ["codes.pt"]
    else:
        # 查找所有 codes_XXX.pt 文件（新格式）
        numbered_codes = [f for f in list_codes if f.startswith("codes_") and f.endswith(".pt")]
        if numbered_codes:
            # 按编号排序
            numbered_codes.sort()
            list_codes = numbered_codes
        else:
            # 兼容旧格式：如果有其他codes文件，使用第一个
            list_codes.sort()
            list_codes = [list_codes[0]]
    
    print(f"Found {len(list_codes)} codes files in {split_dir}")
    for code_file in list_codes:
        print(f"  - {code_file}")
    
    # 只加载第一个文件来获取shape信息，假设所有文件shape相同
    print("Loading first file to get shape info...")
    first_code_path = os.path.join(split_dir, list_codes[0])
    first_codes = torch.load(first_code_path, weights_only=False)
    print(f"  Shape: {first_codes.shape}")
    
    samples_per_file = first_codes.shape[0]
    code_shape = first_codes.shape[1:]  # 保存每个样本的shape（去掉batch维度）
    
    # 计算每个样本的实际大小
    if len(first_codes.shape) == 3:
        # 3D tensor: [batch, dim1, dim2]
        actual_size_per_sample = code_shape[0] * code_shape[1] * 4  # float32 = 4 bytes
    elif len(first_codes.shape) == 2:
        # 2D tensor: [batch, dim]
        actual_size_per_sample = code_shape[0] * 4
    else:
        # 1D tensor or other
        actual_size_per_sample = first_codes.numel() // samples_per_file * 4
    
    if len(first_codes.shape) > 1:
        code_dim = first_codes.shape[1]
    else:
        code_dim = first_codes.shape[0]
    
    # 估算总样本数（假设所有文件shape相同）
    total_samples = samples_per_file * len(list_codes)
    del first_codes  # 立即释放内存
    gc.collect()
    
    print(f"Estimated total samples: {total_samples} (assuming {samples_per_file} samples per file)")
    print(f"Code shape per sample: {code_shape}")
    print(f"Code dimension: {code_dim}")
    
    # 删除旧的LMDB文件（如果存在）
    if os.path.exists(lmdb_path):
        print(f"Removing existing LMDB file: {lmdb_path}")
        if os.path.isdir(lmdb_path):
            shutil.rmtree(lmdb_path)
        else:
            os.remove(lmdb_path)
    
    # 删除旧的锁文件（如果存在）
    lock_file = lmdb_path + "-lock"
    if os.path.exists(lock_file):
        print(f"Removing existing lock file: {lock_file}")
        os.remove(lock_file)
    
    # 创建LMDB数据库
    # 加上序列化开销（pickle overhead）和 LMDB overhead
    # pickle序列化会增加约50-100%的开销，LMDB也有少量overhead
    estimated_size_per_sample = actual_size_per_sample * 2.0 + 4096  # 2倍用于序列化和overhead，4096 bytes额外overhead
    map_size = max(100 * (1024 * 1024 * 1024), total_samples * estimated_size_per_sample * 2)  # 至少100GB，或2倍估算大小
    
    print(f"Creating LMDB database with map_size: {map_size / (1024**3):.2f} GB")
    print(f"  Actual size per sample: {actual_size_per_sample / (1024**2):.2f} MB")
    print(f"  Estimated size per sample (with overhead): {estimated_size_per_sample / (1024**2):.2f} MB")
    db = lmdb.open(lmdb_path, map_size=int(map_size))
    
    # 逐个文件处理，边读边写，避免内存爆炸
    # 使用小批次处理，保守地使用内存，避免服务器崩溃
    # 每个样本约0.36MB，500个样本约180MB，加上序列化开销约360MB，在内存可接受范围内
    BATCH_SIZE = 500  # 每批写入500个sample后提交事务，保守设置避免内存峰值
    
    keys = []
    global_index = 0
    try:
        for code_file in list_codes:
            code_path = os.path.join(split_dir, code_file)
            print(f"Processing {code_file}...")
            
            # 加载当前文件（但我们会尽快处理并释放）
            codes = torch.load(code_path, weights_only=False)
            num_samples = codes.shape[0]
            print(f"  Loaded {num_samples} samples, shape: {codes.shape}")
            
            # 分批写入LMDB
            num_batches = (num_samples + BATCH_SIZE - 1) // BATCH_SIZE
            pbar = tqdm(range(0, num_samples, BATCH_SIZE), desc=f"  Writing {code_file}", total=num_batches)
            
            for batch_start in pbar:
                batch_end = min(batch_start + BATCH_SIZE, num_samples)
                
                # 每个批次使用独立的事务
                with db.begin(write=True) as txn:
                    # 逐样本处理，立即序列化和写入，避免在内存中累积
                    for i in range(batch_start, batch_end):
                        key = str(global_index).encode()
                        
                        # 提取单个样本（这会创建一个view，不复制数据）
                        # 但我们需要clone()来确保序列化时不会序列化整个tensor
                        code_sample = codes[i].clone().detach()
                        
                        # 立即序列化
                        try:
                            value = pickle.dumps(code_sample, protocol=pickle.HIGHEST_PROTOCOL)
                            
                            # 检查序列化后的大小（LMDB单个值限制约511MB）
                            value_size_mb = len(value) / (1024 * 1024)
                            if value_size_mb > 500:  # 500MB
                                print(f"\nWarning: code {global_index} is very large: {value_size_mb:.2f} MB")
                            
                            # LMDB单个值大小限制检查（约511MB）
                            if len(value) > 511 * 1024 * 1024:
                                raise ValueError(f"Code {global_index} is too large for LMDB: {value_size_mb:.2f} MB (max 511MB)")
                            
                            # 立即写入，不累积在内存中
                            txn.put(key, value)
                            keys.append(str(global_index))
                            
                        except Exception as e:
                            print(f"\nError serializing code {global_index}: {e}")
                        
                        # 立即释放code_sample和value的引用
                        del code_sample
                        if 'value' in locals():
                            del value
                        
                        global_index += 1
                
                # 每个批次后强制垃圾回收，释放内存
                gc.collect()
                
                # 更新进度条
                pbar.set_postfix({'samples': global_index, 'mem': f'{global_index * 0.36 / 1024:.1f}GB'})
            
            # 立即释放整个文件的内存
            del codes
            gc.collect()
            print(f"  Completed {code_file}, total samples so far: {global_index}")
        
    except Exception as e:
        print(f"Error during conversion: {e}")
        raise
    finally:
        db.close()
    
    # 保存keys列表
    torch.save(keys, keys_path)
    
    print(f"Successfully converted to LMDB: {lmdb_path}")
    print(f"Saved keys list: {keys_path}")
    print(f"Database contains {len(keys)} codes")
    
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
    parser = argparse.ArgumentParser(description="Convert codes files to LMDB format")
    parser.add_argument("--codes_dir", type=str, required=True, 
                       help="Codes文件所在目录")
    parser.add_argument("--splits", type=str, nargs="+", 
                       default=["train", "val", "test"],
                       help="要转换的数据分割")
    
    args = parser.parse_args()
    
    for split in args.splits:
        split_dir = os.path.join(args.codes_dir, split)
        if not os.path.exists(split_dir):
            print(f"Warning: {split_dir} does not exist, skipping...")
            continue
        
        lmdb_path = os.path.join(split_dir, "codes.lmdb")
        keys_path = os.path.join(split_dir, "codes_keys.pt")
        
        print(f"\n{'='*60}")
        print(f"Converting {split} split...")
        print(f"{'='*60}")
        
        try:
            convert_codes_to_lmdb(args.codes_dir, split, lmdb_path, keys_path)
        except Exception as e:
            print(f"Error converting {split}: {e}")
            continue


if __name__ == "__main__":
    main()

