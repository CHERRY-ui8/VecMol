#!/usr/bin/env python3
"""
直接将temp_batch文件转换为LMDB格式，跳过合并步骤
避免OOM风险，更安全高效

使用方法:
    python convert_temp_codes_to_lmdb.py \
        --temp_dir /data/huayuchen/Neurl-voxel/exps/neural_field/nf_drugs/20260113/lightning_logs/version_0/checkpoints/code_no_aug/train \
        --output_dir /data/huayuchen/Neurl-voxel/exps/neural_field/nf_drugs/20260113/lightning_logs/version_0/checkpoints/code_no_aug/train \
        --num_augmentations 1 \
        --aug_idx 0 \
        --delete_batches  # 可选：转换后删除batch文件
    
    python convert_temp_codes_to_lmdb.py \
    --temp_dir /data/huayuchen/Neurl-voxel/exps/neural_field/nf_drugs/20260113/lightning_logs/version_0/checkpoints/code_no_aug/train \
    --output_dir /data/huayuchen/Neurl-voxel/exps/neural_field/nf_drugs/20260113/lightning_logs/version_0/checkpoints/code_no_aug/train \
    --num_augmentations 1 \
    --aug_idx 0 \
    --only_weights \
    --delete_batches
"""

import os
import pickle
import gc
import torch
import lmdb
from tqdm import tqdm
import argparse
import shutil
from pathlib import Path


def convert_temp_batches_to_lmdb(
    temp_dir,
    output_dir,
    num_augmentations,
    aug_idx=0,
    file_prefix="codes",
    delete_batches=False
):
    """
    直接将temp_batch文件转换为LMDB，跳过合并步骤
    
    Args:
        temp_dir: temp_batches目录路径
        output_dir: 输出目录路径（LMDB文件保存位置）
        num_augmentations: 数据增强数量
        aug_idx: 增强索引（0, 1, 2等）
        file_prefix: 文件前缀 ("codes" 或 "position_weights")
        delete_batches: 是否在转换后删除batch文件
    """
    temp_dir = Path(temp_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("Direct Temp Batch to LMDB Converter")
    print("=" * 80)
    print(f"Temp directory: {temp_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Augmentation index: {aug_idx}")
    print(f"File prefix: {file_prefix}")
    print("=" * 80)
    
    if not temp_dir.exists():
        raise FileNotFoundError(f"Temp directory not found: {temp_dir}")
    
    # 查找所有batch文件
    if file_prefix == "codes":
        pattern = f"codes_{aug_idx:03d}_batch_"
    else:
        pattern = f"position_weights_aug{num_augmentations}_{aug_idx:03d}_batch_"
    
    batch_files = sorted([
        f for f in temp_dir.iterdir()
        if f.is_file() and f.name.startswith(pattern) and f.name.endswith(".pt")
    ])
    
    if not batch_files:
        raise FileNotFoundError(f"No batch files found matching pattern '{pattern}*.pt' in {temp_dir}")
    
    print(f"Found {len(batch_files)} batch files")
    print(f"  First file: {batch_files[0].name}")
    print(f"  Last file: {batch_files[-1].name}")
    
    # 按batch索引排序（从文件名提取）
    def extract_batch_idx(filename):
        """从文件名提取batch索引: codes_000_batch_000000.pt -> 0"""
        parts = filename.stem.split("_")
        if len(parts) >= 4:
            try:
                return int(parts[-1])
            except:
                return -1
        return -1
    
    batch_files.sort(key=lambda f: extract_batch_idx(f))
    
    # 验证文件索引的连续性
    batch_indices = [extract_batch_idx(f) for f in batch_files]
    if -1 in batch_indices:
        print(f"⚠️  WARNING: Some files have invalid batch indices")
    
    expected_range = set(range(min(batch_indices), max(batch_indices) + 1))
    actual_range = set(batch_indices)
    missing = sorted(list(expected_range - actual_range))
    if missing:
        print(f"⚠️  WARNING: Missing batch indices: {missing[:10]}... (showing first 10)")
        print(f"   Total missing: {len(missing)} indices")
    else:
        print(f"✓ All batch indices are continuous")
    
    # 加载第一个batch文件获取shape信息
    print("\nLoading first batch file to get shape info...")
    first_batch = torch.load(batch_files[0], map_location='cpu', weights_only=False)
    samples_per_batch = first_batch.shape[0]
    code_shape = first_batch.shape[1:]  # 每个样本的shape
    
    print(f"  Samples per batch: {samples_per_batch}")
    print(f"  Code shape per sample: {code_shape}")
    
    # 计算每个样本的大小
    if len(first_batch.shape) == 3:
        actual_size_per_sample = code_shape[0] * code_shape[1] * 4  # float32 = 4 bytes
    elif len(first_batch.shape) == 2:
        actual_size_per_sample = code_shape[0] * 4
    else:
        actual_size_per_sample = first_batch.numel() // samples_per_batch * 4
    
    # 估算总样本数
    total_samples = samples_per_batch * len(batch_files)
    print(f"  Estimated total samples: {total_samples}")
    
    del first_batch
    gc.collect()
    
    # 生成输出文件路径
    if file_prefix == "codes":
        lmdb_path = output_dir / f"codes_aug{num_augmentations}_{aug_idx:03d}.lmdb"
        keys_path = output_dir / f"codes_aug{num_augmentations}_{aug_idx:03d}_keys.pt"
    else:
        lmdb_path = output_dir / f"position_weights_aug{num_augmentations}_{aug_idx:03d}.lmdb"
        keys_path = output_dir / f"position_weights_aug{num_augmentations}_{aug_idx:03d}_keys.pt"
    
    # 删除旧的LMDB文件（如果存在）
    if lmdb_path.exists():
        print(f"\nRemoving existing LMDB file: {lmdb_path}")
        if lmdb_path.is_dir():
            shutil.rmtree(lmdb_path)
        else:
            lmdb_path.unlink()
    
    lock_file = lmdb_path.parent / (lmdb_path.name + "-lock")
    if lock_file.exists():
        print(f"Removing existing lock file: {lock_file}")
        lock_file.unlink()
    
    # 创建LMDB数据库
    estimated_size_per_sample = actual_size_per_sample * 2.0 + 4096  # 2倍用于序列化和overhead
    map_size = max(100 * (1024 * 1024 * 1024), total_samples * estimated_size_per_sample * 2)  # 至少100GB
    
    print(f"\nCreating LMDB database with map_size: {map_size / (1024**3):.2f} GB")
    print(f"  Actual size per sample: {actual_size_per_sample / (1024**2):.2f} MB")
    db = lmdb.open(str(lmdb_path), map_size=int(map_size))
    
    # 分批处理，避免内存峰值
    BATCH_SIZE = 500  # 每批写入500个sample后提交事务
    
    keys = []
    global_index = 0
    processed_batches = 0
    
    try:
        for batch_file in tqdm(batch_files, desc="Processing batch files"):
            # 加载当前batch文件
            batch_data = torch.load(batch_file, map_location='cpu', weights_only=False)
            num_samples = batch_data.shape[0]
            
            # 分批写入LMDB
            num_batches = (num_samples + BATCH_SIZE - 1) // BATCH_SIZE
            
            for batch_start in range(0, num_samples, BATCH_SIZE):
                batch_end = min(batch_start + BATCH_SIZE, num_samples)
                
                # 每个批次使用独立的事务
                with db.begin(write=True) as txn:
                    # 逐样本处理，立即序列化和写入
                    for i in range(batch_start, batch_end):
                        key = str(global_index).encode()
                        
                        # 提取单个样本
                        sample = batch_data[i].clone().detach()
                        
                        # 立即序列化
                        try:
                            value = pickle.dumps(sample, protocol=pickle.HIGHEST_PROTOCOL)
                            
                            # 检查序列化后的大小（LMDB单个值限制约511MB）
                            value_size_mb = len(value) / (1024 * 1024)
                            if value_size_mb > 500:  # 500MB
                                print(f"\n⚠️  Warning: code {global_index} is very large: {value_size_mb:.2f} MB")
                            
                            # LMDB单个值大小限制检查
                            if len(value) > 511 * 1024 * 1024:
                                raise ValueError(f"Code {global_index} is too large for LMDB: {value_size_mb:.2f} MB (max 511MB)")
                            
                            # 立即写入
                            txn.put(key, value)
                            keys.append(str(global_index))
                            
                        except Exception as e:
                            print(f"\n❌ Error serializing code {global_index}: {e}")
                            raise
                        
                        # 立即释放sample和value的引用
                        del sample
                        if 'value' in locals():
                            del value
                        
                        global_index += 1
                
                # 每个批次后强制垃圾回收
                gc.collect()
            
            # 立即释放整个batch文件的内存
            del batch_data
            gc.collect()
            
            # 删除batch文件（如果启用）
            if delete_batches:
                try:
                    batch_file.unlink()
                except Exception as e:
                    print(f"⚠️  Warning: Failed to delete {batch_file.name}: {e}")
            
            processed_batches += 1
            
            # 每处理100个batch文件打印一次进度
            if processed_batches % 100 == 0:
                print(f"  Processed {processed_batches}/{len(batch_files)} batch files, total samples: {global_index}")
        
    except Exception as e:
        print(f"\n❌ Error during conversion: {e}")
        raise
    finally:
        db.close()
    
    # 保存keys列表
    torch.save(keys, keys_path)
    
    print(f"\n✅ Successfully converted to LMDB: {lmdb_path}")
    print(f"✅ Saved keys list: {keys_path}")
    print(f"✅ Database contains {len(keys)} codes")
    
    # 验证生成的LMDB文件
    print("\nVerifying generated LMDB file...")
    try:
        verify_db = lmdb.open(str(lmdb_path), readonly=True)
        with verify_db.begin() as txn:
            stat = txn.stat()
            print(f"  LMDB verification: {stat['entries']} entries")
        verify_db.close()
        print("✅ LMDB file verification successful")
    except Exception as e:
        print(f"❌ LMDB file verification failed: {e}")
    
    return len(keys)


def convert_mixed_to_lmdb(
    intermediate_dir,
    temp_batches_dir,
    output_dir,
    num_augmentations,
    aug_idx=0,
    file_prefix="codes",
    delete_after_convert=False
):
    """
    将intermediate文件和temp_batch文件合并转换为LMDB
    
    Args:
        intermediate_dir: intermediate文件目录路径
        temp_batches_dir: temp_batches目录路径
        output_dir: 输出目录路径（LMDB文件保存位置）
        num_augmentations: 数据增强数量
        aug_idx: 增强索引（0, 1, 2等）
        file_prefix: 文件前缀 ("codes" 或 "position_weights")
        delete_after_convert: 是否在转换后删除源文件
    """
    intermediate_dir = Path(intermediate_dir) if intermediate_dir else None
    temp_batches_dir = Path(temp_batches_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("Mixed Intermediate + Batch Files to LMDB Converter")
    print("=" * 80)
    print(f"Intermediate directory: {intermediate_dir}")
    print(f"Temp batches directory: {temp_batches_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Augmentation index: {aug_idx}")
    print(f"File prefix: {file_prefix}")
    print("=" * 80)
    
    # 收集所有要处理的文件
    all_files = []  # [(file_path, file_type), ...] file_type: 'intermediate' or 'batch'
    
    # 1. 收集intermediate文件（按索引排序）
    if intermediate_dir and intermediate_dir.exists():
        intermediate_files = [
            f for f in intermediate_dir.iterdir()
            if f.is_file() and f.name.startswith("intermediate_") and f.name.endswith(".pt")
        ]
        
        # 按索引排序：intermediate_000000.pt -> 0
        def extract_intermediate_idx(filename):
            """从文件名提取intermediate索引: intermediate_000000.pt -> 0"""
            try:
                # intermediate_000000.pt -> 000000 -> 0
                idx_str = filename.stem.replace("intermediate_", "")
                return int(idx_str)
            except:
                return -1
        
        intermediate_files.sort(key=extract_intermediate_idx)
        print(f"\nFound {len(intermediate_files)} intermediate files")
        if intermediate_files:
            print(f"  First: {intermediate_files[0].name}")
            print(f"  Last: {intermediate_files[-1].name}")
        for f in intermediate_files:
            all_files.append((f, 'intermediate'))
    else:
        print(f"\nNo intermediate directory or directory not found, skipping...")
        intermediate_files = []
    
    # 2. 收集batch文件（按数字索引排序）
    if file_prefix == "codes":
        pattern = f"codes_{aug_idx:03d}_batch_"
    else:
        pattern = f"position_weights_aug{num_augmentations}_{aug_idx:03d}_batch_"
    
    batch_files = [
        f for f in temp_batches_dir.iterdir()
        if f.is_file() and f.name.startswith(pattern) and f.name.endswith(".pt")
    ]
    
    # 按batch索引排序：codes_000_batch_003659.pt -> 3659
    def extract_batch_idx(filename):
        """从文件名提取batch索引: codes_000_batch_003659.pt -> 3659"""
        try:
            # codes_000_batch_003659.pt -> 003659 -> 3659
            parts = filename.stem.split("_")
            if len(parts) >= 4:
                return int(parts[-1])
        except:
            pass
        return -1
    
    batch_files.sort(key=extract_batch_idx)
    
    print(f"Found {len(batch_files)} batch files")
    if batch_files:
        first_idx = extract_batch_idx(batch_files[0])
        last_idx = extract_batch_idx(batch_files[-1])
        print(f"  First: {batch_files[0].name} (index: {first_idx})")
        print(f"  Last: {batch_files[-1].name} (index: {last_idx})")
    for f in batch_files:
        all_files.append((f, 'batch'))
    
    if not all_files:
        raise ValueError("No files found to convert!")
    
    print(f"\nTotal files to process: {len(all_files)} ({len(intermediate_files)} intermediate + {len(batch_files)} batch)")
    
    # 加载第一个文件获取shape信息
    print("\nLoading first file to get shape info...")
    first_file, first_type = all_files[0]
    first_data = torch.load(first_file, map_location='cpu', weights_only=False)
    samples_per_file = first_data.shape[0]
    code_shape = first_data.shape[1:]
    
    print(f"  Samples per file: {samples_per_file}")
    print(f"  Code shape per sample: {code_shape}")
    
    # 计算每个样本的大小
    if len(first_data.shape) == 3:
        actual_size_per_sample = code_shape[0] * code_shape[1] * 4
    elif len(first_data.shape) == 2:
        actual_size_per_sample = code_shape[0] * 4
    else:
        actual_size_per_sample = first_data.numel() // samples_per_file * 4
    
    # 估算总样本数
    total_samples = samples_per_file * len(all_files)
    print(f"  Estimated total samples: {total_samples}")
    
    del first_data
    gc.collect()
    
    # 生成输出文件路径
    if file_prefix == "codes":
        lmdb_path = output_dir / f"codes_aug{num_augmentations}_{aug_idx:03d}.lmdb"
        keys_path = output_dir / f"codes_aug{num_augmentations}_{aug_idx:03d}_keys.pt"
    else:
        lmdb_path = output_dir / f"position_weights_aug{num_augmentations}_{aug_idx:03d}.lmdb"
        keys_path = output_dir / f"position_weights_aug{num_augmentations}_{aug_idx:03d}_keys.pt"
    
    # 删除旧的LMDB文件（如果存在）
    if lmdb_path.exists():
        print(f"\nRemoving existing LMDB file: {lmdb_path}")
        if lmdb_path.is_dir():
            shutil.rmtree(lmdb_path)
        else:
            lmdb_path.unlink()
    
    lock_file = lmdb_path.parent / (lmdb_path.name + "-lock")
    if lock_file.exists():
        print(f"Removing existing lock file: {lock_file}")
        lock_file.unlink()
    
    # 创建LMDB数据库
    estimated_size_per_sample = actual_size_per_sample * 2.0 + 4096
    map_size = max(100 * (1024 * 1024 * 1024), total_samples * estimated_size_per_sample * 2)
    
    print(f"\nCreating LMDB database with map_size: {map_size / (1024**3):.2f} GB")
    print(f"  Actual size per sample: {actual_size_per_sample / (1024**2):.2f} MB")
    db = lmdb.open(str(lmdb_path), map_size=int(map_size))
    
    # 分批处理
    BATCH_SIZE = 500
    keys = []
    global_index = 0
    processed_files = 0
    
    try:
        for file_path, file_type in tqdm(all_files, desc="Processing files"):
            # 加载文件
            file_data = torch.load(file_path, map_location='cpu', weights_only=False)
            num_samples = file_data.shape[0]
            
            # 分批写入LMDB
            for batch_start in range(0, num_samples, BATCH_SIZE):
                batch_end = min(batch_start + BATCH_SIZE, num_samples)
                
                with db.begin(write=True) as txn:
                    for i in range(batch_start, batch_end):
                        key = str(global_index).encode()
                        sample = file_data[i].clone().detach()
                        
                        try:
                            value = pickle.dumps(sample, protocol=pickle.HIGHEST_PROTOCOL)
                            
                            value_size_mb = len(value) / (1024 * 1024)
                            if value_size_mb > 500:
                                print(f"\n⚠️  Warning: code {global_index} is very large: {value_size_mb:.2f} MB")
                            
                            if len(value) > 511 * 1024 * 1024:
                                raise ValueError(f"Code {global_index} is too large for LMDB: {value_size_mb:.2f} MB (max 511MB)")
                            
                            txn.put(key, value)
                            keys.append(str(global_index))
                            
                        except Exception as e:
                            print(f"\n❌ Error serializing code {global_index}: {e}")
                            raise
                        
                        del sample
                        if 'value' in locals():
                            del value
                        
                        global_index += 1
                
                gc.collect()
            
            # 释放内存
            del file_data
            gc.collect()
            
            # 删除文件（如果启用）
            if delete_after_convert:
                try:
                    file_path.unlink()
                except Exception as e:
                    print(f"⚠️  Warning: Failed to delete {file_path.name}: {e}")
            
            processed_files += 1
            
            # 每处理100个文件打印一次进度
            if processed_files % 100 == 0:
                print(f"  Processed {processed_files}/{len(all_files)} files, total samples: {global_index}")
        
    except Exception as e:
        print(f"\n❌ Error during conversion: {e}")
        raise
    finally:
        db.close()
    
    # 保存keys列表
    torch.save(keys, keys_path)
    
    print(f"\n✅ Successfully converted to LMDB: {lmdb_path}")
    print(f"✅ Saved keys list: {keys_path}")
    print(f"✅ Database contains {len(keys)} codes")
    
    # 验证
    print("\nVerifying generated LMDB file...")
    try:
        verify_db = lmdb.open(str(lmdb_path), readonly=True)
        with verify_db.begin() as txn:
            stat = txn.stat()
            print(f"  LMDB verification: {stat['entries']} entries")
        verify_db.close()
        print("✅ LMDB file verification successful")
    except Exception as e:
        print(f"❌ LMDB file verification failed: {e}")
    
    return len(keys)


def convert_temp_position_weights_to_lmdb(
    temp_dir,
    output_dir,
    num_augmentations,
    aug_idx=0,
    codes_keys=None,
    delete_batches=False
):
    """
    直接将temp_batch position_weights文件转换为LMDB
    
    Args:
        temp_dir: temp_batches目录路径
        output_dir: 输出目录路径
        num_augmentations: 数据增强数量
        aug_idx: 增强索引
        codes_keys: codes的keys列表，用于验证数量匹配
        delete_batches: 是否在转换后删除batch文件
    """
    return convert_temp_batches_to_lmdb(
        temp_dir=temp_dir,
        output_dir=output_dir,
        num_augmentations=num_augmentations,
        aug_idx=aug_idx,
        file_prefix="position_weights",
        delete_batches=delete_batches
    )


def main():
    parser = argparse.ArgumentParser(description="直接将temp_batch文件转换为LMDB格式")
    parser.add_argument("--temp_dir", type=str, default=None,
                       help="temp_batches目录路径（如果使用--mixed_mode，此参数可选）")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="输出目录路径（LMDB文件保存位置）")
    parser.add_argument("--num_augmentations", type=int, required=True,
                       help="数据增强数量")
    parser.add_argument("--aug_idx", type=int, default=0,
                       help="增强索引（默认0）")
    parser.add_argument("--delete_batches", action="store_true",
                       help="转换后删除batch文件（节省空间）")
    parser.add_argument("--convert_weights", action="store_true",
                       help="同时转换position_weights文件")
    parser.add_argument("--only_weights", action="store_true",
                       help="只转换position_weights文件，跳过codes转换")
    
    # 新增：混合模式参数
    parser.add_argument("--mixed_mode", action="store_true",
                       help="混合模式：同时处理intermediate文件和batch文件")
    parser.add_argument("--intermediate_dir", type=str, default=None,
                       help="intermediate文件目录路径（仅在--mixed_mode时使用）")
    
    args = parser.parse_args()
    
    # 如果指定了--only_weights，直接转换weights并返回
    if args.only_weights:
        if not args.temp_dir:
            raise ValueError("--temp_dir is required when using --only_weights")
        
        print("=" * 80)
        print("Converting Position Weights Only")
        print("=" * 80)
        print(f"Temp directory: {args.temp_dir}")
        print(f"Output directory: {args.output_dir}")
        print(f"Number of augmentations: {args.num_augmentations}")
        print(f"Augmentation index: {args.aug_idx}")
        print(f"Delete batches after conversion: {args.delete_batches}")
        print("=" * 80)
        
        print("\n>> Converting position_weights files...")
        try:
            # 尝试加载codes keys用于验证（可选）
            keys_path = Path(args.output_dir) / f"codes_aug{args.num_augmentations}_{args.aug_idx:03d}_keys.pt"
            codes_keys = None
            if keys_path.exists():
                codes_keys = torch.load(keys_path, weights_only=False)
                print(f"  Loaded codes keys: {len(codes_keys)} samples (for verification)")
            else:
                print(f"  ⚠️  Warning: Codes keys file not found, skipping count verification")
            
            if args.mixed_mode:
                if not args.intermediate_dir:
                    raise ValueError("--intermediate_dir is required when using --mixed_mode with --only_weights")
                # 混合模式：处理intermediate + batch position_weights
                num_weights = convert_mixed_to_lmdb(
                    intermediate_dir=args.intermediate_dir.replace("codes", "position_weights") if args.intermediate_dir else None,
                    temp_batches_dir=args.temp_dir,
                    output_dir=args.output_dir,
                    num_augmentations=args.num_augmentations,
                    aug_idx=args.aug_idx,
                    file_prefix="position_weights",
                    delete_after_convert=args.delete_batches
                )
            else:
                # 普通模式：只处理batch position_weights
                num_weights = convert_temp_position_weights_to_lmdb(
                    temp_dir=args.temp_dir,
                    output_dir=args.output_dir,
                    num_augmentations=args.num_augmentations,
                    aug_idx=args.aug_idx,
                    codes_keys=codes_keys,
                    delete_batches=args.delete_batches
                )
            
            if codes_keys and num_weights != len(codes_keys):
                print(f"⚠️  Warning: Position weights count ({num_weights}) != codes count ({len(codes_keys)})")
            else:
                if codes_keys:
                    print(f"✅ Converted {num_weights} position_weights samples (matches codes count)")
                else:
                    print(f"✅ Converted {num_weights} position_weights samples")
                
        except Exception as e:
            print(f"❌ Error converting position_weights: {e}")
            import traceback
            traceback.print_exc()
            return
        
        print("\n" + "=" * 80)
        print("✅ All done!")
        print("=" * 80)
        return
    
    # 检查模式
    if args.mixed_mode:
        # 混合模式：处理intermediate + batch文件
        if not args.intermediate_dir:
            raise ValueError("--intermediate_dir is required when using --mixed_mode")
        if not args.temp_dir:
            raise ValueError("--temp_dir is required when using --mixed_mode")
        
        print("=" * 80)
        print("Mixed Mode: Intermediate + Batch Files to LMDB Converter")
        print("=" * 80)
        print(f"Intermediate directory: {args.intermediate_dir}")
        print(f"Temp batches directory: {args.temp_dir}")
        print(f"Output directory: {args.output_dir}")
        print(f"Number of augmentations: {args.num_augmentations}")
        print(f"Augmentation index: {args.aug_idx}")
        print(f"Delete files after conversion: {args.delete_batches}")
        print("=" * 80)
        
        # 转换codes文件
        print("\n>> Converting codes files (mixed mode)...")
        try:
            num_samples = convert_mixed_to_lmdb(
                intermediate_dir=args.intermediate_dir,
                temp_batches_dir=args.temp_dir,
                output_dir=args.output_dir,
                num_augmentations=args.num_augmentations,
                aug_idx=args.aug_idx,
                file_prefix="codes",
                delete_after_convert=args.delete_batches
            )
            print(f"✅ Converted {num_samples} codes samples")
        except Exception as e:
            print(f"❌ Error converting codes: {e}")
            import traceback
            traceback.print_exc()
            return
    else:
        # 普通模式：只处理batch文件
        if not args.temp_dir:
            raise ValueError("--temp_dir is required when not using --mixed_mode")
        
        print("=" * 80)
        print("Direct Temp Batch to LMDB Converter")
        print("=" * 80)
        print(f"Temp directory: {args.temp_dir}")
        print(f"Output directory: {args.output_dir}")
        print(f"Number of augmentations: {args.num_augmentations}")
        print(f"Augmentation index: {args.aug_idx}")
        print(f"Delete batches after conversion: {args.delete_batches}")
        print("=" * 80)
        
        # 转换codes文件
        print("\n>> Converting codes files...")
        try:
            num_samples = convert_temp_batches_to_lmdb(
                temp_dir=args.temp_dir,
                output_dir=args.output_dir,
                num_augmentations=args.num_augmentations,
                aug_idx=args.aug_idx,
                file_prefix="codes",
                delete_batches=args.delete_batches
            )
            print(f"✅ Converted {num_samples} codes samples")
        except Exception as e:
            print(f"❌ Error converting codes: {e}")
            import traceback
            traceback.print_exc()
            return
    
    # 转换position_weights文件（如果启用）
    if args.convert_weights:
        print("\n>> Converting position_weights files...")
        try:
            # 加载codes keys用于验证
            keys_path = Path(args.output_dir) / f"codes_aug{args.num_augmentations}_{args.aug_idx:03d}_keys.pt"
            if keys_path.exists():
                codes_keys = torch.load(keys_path, weights_only=False)
                print(f"  Loaded codes keys: {len(codes_keys)} samples")
            else:
                codes_keys = None
                print(f"  ⚠️  Warning: Codes keys file not found, cannot verify position_weights count")
            
            if args.mixed_mode:
                # 混合模式：处理intermediate + batch position_weights
                num_weights = convert_mixed_to_lmdb(
                    intermediate_dir=args.intermediate_dir.replace("codes", "position_weights") if args.intermediate_dir else None,
                    temp_batches_dir=args.temp_dir,
                    output_dir=args.output_dir,
                    num_augmentations=args.num_augmentations,
                    aug_idx=args.aug_idx,
                    file_prefix="position_weights",
                    delete_after_convert=args.delete_batches
                )
            else:
                # 普通模式：只处理batch position_weights
                num_weights = convert_temp_position_weights_to_lmdb(
                    temp_dir=args.temp_dir,
                    output_dir=args.output_dir,
                    num_augmentations=args.num_augmentations,
                    aug_idx=args.aug_idx,
                    codes_keys=codes_keys,
                    delete_batches=args.delete_batches
                )
            
            if codes_keys and num_weights != len(codes_keys):
                print(f"⚠️  Warning: Position weights count ({num_weights}) != codes count ({len(codes_keys)})")
            else:
                print(f"✅ Converted {num_weights} position_weights samples")
                
        except Exception as e:
            print(f"❌ Error converting position_weights: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("✅ All done!")
    print("=" * 80)


if __name__ == "__main__":
    main()

