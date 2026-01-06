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


def convert_codes_to_lmdb(codes_dir, split, lmdb_path, keys_path, num_augmentations):
    """
    将codes文件转换为LMDB数据库
    
    Args:
        codes_dir (str): codes文件所在目录
        split (str): 数据分割名称 (train/val/test)
        lmdb_path (str): 输出LMDB数据库路径
        keys_path (str): 输出keys文件路径
        num_augmentations (int): 数据增强数量，用于查找对应格式的文件
    
    Returns:
        num_augmentations (int): 数据增强数量（codes文件的数量）
    """
    split_dir = os.path.join(codes_dir, split)
    
    if not os.path.exists(split_dir):
        raise FileNotFoundError(f"Codes directory not found: {split_dir}")
    
    # 根据数据增强数量查找对应格式的文件：codes_aug{num}_XXX.pt
    list_codes = [
        f for f in os.listdir(split_dir)
        if os.path.isfile(os.path.join(split_dir, f)) and \
        f.startswith(f"codes_aug{num_augmentations}_") and f.endswith(".pt")
    ]
    
    if not list_codes:
        raise FileNotFoundError(
            f"No codes files found in {split_dir} matching pattern 'codes_aug{num_augmentations}_*.pt'.\n"
            f"Please ensure codes files are generated with num_augmentations={num_augmentations}"
        )
    
    # 排序确保顺序正确
    list_codes.sort()
    print(f"Found {len(list_codes)} codes files matching pattern 'codes_aug{num_augmentations}_*.pt'")
    
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
    
    # 返回数据增强数量
    num_augmentations = len(list_codes)
    return num_augmentations


def convert_position_weights_to_lmdb(codes_dir, split, codes_keys, num_augmentations=None):
    """
    将position_weights文件转换为LMDB数据库
    
    Args:
        codes_dir (str): codes文件所在目录
        split (str): 数据分割名称 (train/val/test)
        codes_keys (list): codes的keys列表，用于确保position_weights和codes的索引一致
        num_augmentations (int): 数据增强数量，用于查找对应格式的文件
    """
    split_dir = os.path.join(codes_dir, split)
    
    # 根据数据增强数量查找对应格式的文件：position_weights_aug{num}_XXX.pt
    list_weights = [
        f for f in os.listdir(split_dir)
        if os.path.isfile(os.path.join(split_dir, f)) and \
        f.startswith(f"position_weights_aug{num_augmentations}_") and f.endswith(".pt")
    ]
    
    if not list_weights:
        print(f"No position_weights files found in {split_dir} matching pattern 'position_weights_aug{num_augmentations}_*.pt', skipping...")
        return
    
    # 排序确保顺序正确
    list_weights.sort()
    print(f"\nFound {len(list_weights)} position_weights files matching pattern 'position_weights_aug{num_augmentations}_*.pt'")
    for weight_file in list_weights:
        print(f"  - {weight_file}")
    
    # 加载第一个文件获取shape信息
    first_weight_path = os.path.join(split_dir, list_weights[0])
    first_weights = torch.load(first_weight_path, weights_only=False)
    print(f"  Position weights shape: {first_weights.shape}")
    
    samples_per_file = first_weights.shape[0]
    total_samples = samples_per_file * len(list_weights)
    del first_weights
    gc.collect()
    
    print(f"Estimated total position_weights samples: {total_samples}")
    
    # 验证与codes的数量是否匹配
    if len(codes_keys) != total_samples:
        print(f"Warning: Position weights count ({total_samples}) != codes count ({len(codes_keys)})")
        print("  Will still convert, but indices may not match correctly")
    
    # 创建position_weights LMDB数据库
    # 根据数据增强数量生成文件名，避免覆盖不同版本的文件
    weights_lmdb_path = os.path.join(split_dir, f"position_weights_aug{num_augmentations}.lmdb")
    weights_keys_path = os.path.join(split_dir, f"position_weights_aug{num_augmentations}_keys.pt")
    
    # 删除旧的LMDB文件（如果存在）
    if os.path.exists(weights_lmdb_path):
        print(f"Removing existing position_weights LMDB file: {weights_lmdb_path}")
        if os.path.isdir(weights_lmdb_path):
            shutil.rmtree(weights_lmdb_path)
        else:
            os.remove(weights_lmdb_path)
    
    # 估算大小（position_weights通常比codes小很多）
    estimated_size_per_sample = samples_per_file * 4 * 2.0 + 4096  # float32, 2倍开销
    map_size = max(10 * (1024 * 1024 * 1024), total_samples * estimated_size_per_sample * 2)  # 至少10GB
    
    print(f"Creating position_weights LMDB database with map_size: {map_size / (1024**3):.2f} GB")
    db = lmdb.open(weights_lmdb_path, map_size=int(map_size))
    
    BATCH_SIZE = 500
    keys = []
    global_index = 0
    
    try:
        for weight_file in list_weights:
            weight_path = os.path.join(split_dir, weight_file)
            print(f"Processing {weight_file}...")
            
            weights = torch.load(weight_path, weights_only=False)
            num_samples = weights.shape[0]
            print(f"  Loaded {num_samples} samples, shape: {weights.shape}")
            
            num_batches = (num_samples + BATCH_SIZE - 1) // BATCH_SIZE
            pbar = tqdm(range(0, num_samples, BATCH_SIZE), desc=f"  Writing {weight_file}", total=num_batches)
            
            for batch_start in pbar:
                batch_end = min(batch_start + BATCH_SIZE, num_samples)
                
                with db.begin(write=True) as txn:
                    for i in range(batch_start, batch_end):
                        key = str(global_index).encode()
                        
                        weight_sample = weights[i].clone().detach()
                        
                        try:
                            value = pickle.dumps(weight_sample, protocol=pickle.HIGHEST_PROTOCOL)
                            txn.put(key, value)
                            keys.append(str(global_index))
                        except Exception as e:
                            print(f"\nError serializing position_weight {global_index}: {e}")
                        
                        del weight_sample
                        if 'value' in locals():
                            del value
                        
                        global_index += 1
                
                gc.collect()
                pbar.set_postfix({'samples': global_index})
            
            del weights
            gc.collect()
            print(f"  Completed {weight_file}, total samples so far: {global_index}")
    
    except Exception as e:
        print(f"Error during position_weights conversion: {e}")
        raise
    finally:
        db.close()
    
    # 保存keys列表
    torch.save(keys, weights_keys_path)
    
    print(f"Successfully converted position_weights to LMDB: {weights_lmdb_path}")
    print(f"Saved keys list: {weights_keys_path}")
    print(f"Database contains {len(keys)} position_weights")
    
    # 验证
    print("Verifying position_weights LMDB file...")
    try:
        verify_db = lmdb.open(weights_lmdb_path, readonly=True)
        with verify_db.begin() as txn:
            stat = txn.stat()
            print(f"Position weights LMDB verification: {stat['entries']} entries")
        verify_db.close()
        print("✅ Position weights LMDB file verification successful")
    except Exception as e:
        print(f"❌ Position weights LMDB file verification failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="Convert codes files to LMDB format")
    parser.add_argument("--codes_dir", type=str, required=True, 
                       help="Codes文件所在目录")
    parser.add_argument("--num_augmentations", type=int, required=True,
                       help="数据增强数量（必须与infer_codes时使用的num_augmentations一致）")
    parser.add_argument("--splits", type=str, nargs="+", 
                       default=["train", "val", "test"],
                       help="要转换的数据分割")
    parser.add_argument("--skip_position_weights", action="store_true",
                       help="跳过position_weights文件的转换")
    
    args = parser.parse_args()
    num_augmentations = args.num_augmentations
    
    for split in args.splits:
        split_dir = os.path.join(args.codes_dir, split)
        if not os.path.exists(split_dir):
            print(f"Warning: {split_dir} does not exist, skipping...")
            continue
        
        print(f"\n{'='*60}")
        print(f"Converting {split} split (num_augmentations={num_augmentations})...")
        print(f"{'='*60}")
        
        try:
            # 根据数据增强数量生成文件名
            lmdb_path = os.path.join(split_dir, f"codes_aug{num_augmentations}.lmdb")
            keys_path = os.path.join(split_dir, f"codes_aug{num_augmentations}_keys.pt")
            
            print(f"Looking for codes files matching pattern 'codes_aug{num_augmentations}_*.pt'")
            print(f"Will generate: {os.path.basename(lmdb_path)}")
            
            # 转换codes
            returned_num_aug = convert_codes_to_lmdb(args.codes_dir, split, lmdb_path, keys_path, num_augmentations)
            
            # 确保返回的数量与预期一致
            if returned_num_aug != num_augmentations:
                print(f"Warning: Expected {num_augmentations} augmentations, but got {returned_num_aug}")
            
            # 如果需要转换position_weights，加载keys并转换
            if not args.skip_position_weights:
                if os.path.exists(keys_path):
                    codes_keys = torch.load(keys_path, weights_only=False)
                    convert_position_weights_to_lmdb(args.codes_dir, split, codes_keys, num_augmentations)
        except Exception as e:
            print(f"Error converting {split}: {e}")
            import traceback
            traceback.print_exc()
            continue


if __name__ == "__main__":
    main()

