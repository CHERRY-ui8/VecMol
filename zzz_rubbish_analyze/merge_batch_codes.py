#!/usr/bin/env python3
"""
临时脚本：安全高效地合并batch codes文件
用于在infer_codes.py被killed后，继续完成合并工作

使用方法:
    python merge_batch_codes.py \
        --temp_dir /path/to/temp_batches \
        --output_dir /path/to/output \
        --num_augmentations 3 \
        --merge_batch_size 3 \
        --accumulate_threshold 2
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import argparse
from tqdm import tqdm
import gc


def merge_batch_files_safe(
    temp_dir,
    output_dir,
    num_augmentations,
    merge_batch_size=3,  # 减小批次大小
    accumulate_threshold=2,  # 减小累积阈值
    file_prefix="codes",
    aug_idx=None
):
    """
    安全地合并batch文件，使用更激进的策略避免内存溢出
    
    Args:
        temp_dir: 临时batch文件目录
        output_dir: 输出目录
        num_augmentations: 数据增强数量
        merge_batch_size: 每次合并的batch数量（默认3，比原来的10小）
        accumulate_threshold: 累积阈值（默认2，比原来的5小）
        file_prefix: 文件前缀 ("codes" 或 "position_weights")
        aug_idx: 增强索引，如果为None则处理所有
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 确定要处理的aug_idx列表
    if aug_idx is not None:
        aug_indices = [aug_idx]
    else:
        # 自动检测所有aug_idx
        all_files = os.listdir(temp_dir)
        if file_prefix == "codes":
            pattern_prefix = "codes_"
        else:
            pattern_prefix = f"position_weights_aug{num_augmentations}_"
        
        aug_indices = set()
        for f in all_files:
            if f.startswith(pattern_prefix) and f.endswith(".pt"):
                if file_prefix == "codes":
                    # codes_000_batch_000000.pt
                    parts = f.replace(".pt", "").split("_")
                    if len(parts) >= 2:
                        try:
                            aug_indices.add(int(parts[1]))
                        except:
                            pass
                else:
                    # position_weights_aug3_000_batch_000000.pt
                    parts = f.replace(".pt", "").split("_")
                    if len(parts) >= 4:
                        try:
                            aug_indices.add(int(parts[3]))
                        except:
                            pass
        aug_indices = sorted(list(aug_indices))
    
    print(f">> Found {len(aug_indices)} augmentation indices to process: {aug_indices}")
    
    for aug_idx in aug_indices:
        print(f"\n>> Processing augmentation index {aug_idx}...")
        
        # 获取所有batch文件
        if file_prefix == "codes":
            pattern = f"codes_{aug_idx:03d}_batch_"
        else:
            pattern = f"position_weights_aug{num_augmentations}_{aug_idx:03d}_batch_"
        
        batch_files = sorted([
            os.path.join(temp_dir, f)
            for f in os.listdir(temp_dir)
            if f.startswith(pattern) and f.endswith(".pt")
        ])
        
        if not batch_files:
            print(f"   No batch files found for aug_idx={aug_idx}, skipping...")
            continue
        
        print(f"   Found {len(batch_files)} batch files")
        
        # 输出文件路径
        if file_prefix == "codes":
            output_file = os.path.join(output_dir, f"codes_aug{num_augmentations}_{aug_idx:03d}.pt")
        else:
            output_file = os.path.join(output_dir, f"position_weights_aug{num_augmentations}_{aug_idx:03d}.pt")
        
        # 检查输出文件是否已存在，并判断是否需要追加
        existing_file = None
        existing_samples = 0
        if os.path.exists(output_file):
            print(f"   Output file already exists: {output_file}")
            try:
                existing_file = torch.load(output_file, map_location='cpu', weights_only=False)
                existing_samples = existing_file.shape[0]
                print(f"   Existing file has {existing_samples} samples")
                del existing_file
                gc.collect()
            except Exception as e:
                print(f"   WARNING: Failed to load existing file: {e}")
                response = input(f"   Overwrite existing file? (y/N): ")
                if response.lower() != 'y':
                    print(f"   Skipping aug_idx={aug_idx}...")
                    continue
                existing_samples = 0
        
        # 估算总样本数（通过batch文件索引范围）
        # batch文件名格式: codes_000_batch_000000.pt
        batch_indices = []
        for batch_file in batch_files:
            filename = os.path.basename(batch_file)
            # 提取batch索引: codes_000_batch_000000.pt -> 000000
            parts = filename.replace(".pt", "").split("_")
            if len(parts) >= 4:
                try:
                    batch_idx = int(parts[-1])
                    batch_indices.append(batch_idx)
                except:
                    pass
        
        # 加载第一个batch文件来估算每个batch的样本数
        estimated_samples_per_batch = None
        if batch_files:
            try:
                first_batch = torch.load(batch_files[0], map_location='cpu', weights_only=False)
                estimated_samples_per_batch = first_batch.shape[0]
                del first_batch
                gc.collect()
                print(f"   Estimated samples per batch: {estimated_samples_per_batch}")
            except Exception as e:
                print(f"   WARNING: Failed to load first batch to estimate size: {e}")
        
        # 估算总样本数
        if estimated_samples_per_batch and batch_indices:
            max_batch_idx = max(batch_indices)
            estimated_total_samples = (max_batch_idx + 1) * estimated_samples_per_batch
            print(f"   Estimated total samples (based on batch indices): {estimated_total_samples}")
            
            # 判断是否需要追加
            if existing_samples > 0:
                if existing_samples < estimated_total_samples:
                    print(f"   Existing file is incomplete ({existing_samples} < {estimated_total_samples})")
                    print(f"   Will append remaining {estimated_total_samples - existing_samples} samples")
                    append_mode = True
                else:
                    print(f"   Existing file appears complete ({existing_samples} >= {estimated_total_samples})")
                    response = input(f"   Overwrite? (y/N): ")
                    if response.lower() != 'y':
                        print(f"   Skipping aug_idx={aug_idx}...")
                        continue
                    append_mode = False
            else:
                append_mode = False
        else:
            append_mode = False
            if existing_samples > 0:
                response = input(f"   Overwrite existing file? (y/N): ")
                if response.lower() != 'y':
                    print(f"   Skipping aug_idx={aug_idx}...")
                    continue
        
        # 使用中间文件策略：分批合并并保存到中间文件，最后再合并中间文件
        intermediate_dir = os.path.join(output_dir, f"intermediate_{file_prefix}_aug{aug_idx}")
        os.makedirs(intermediate_dir, exist_ok=True)
        
        # 检查是否已有intermediate文件（恢复模式）
        existing_intermediate_files = []
        if os.path.exists(intermediate_dir):
            existing_intermediate_files = sorted([
                os.path.join(intermediate_dir, f)
                for f in os.listdir(intermediate_dir)
                if f.startswith("intermediate_") and f.endswith(".pt")
            ])
        
        intermediate_files = list(existing_intermediate_files)  # 从已存在的文件开始
        intermediate_idx = len(existing_intermediate_files)  # 从下一个索引开始
        
        if existing_intermediate_files:
            print(f"   Found {len(existing_intermediate_files)} existing intermediate files (resume mode)")
            print(f"   Will continue from intermediate file index {intermediate_idx}")
        
        # 第一阶段：将batch文件合并成中间文件
        if batch_files:
            print(f"   Stage 1: Merging batches into intermediate files...")
        else:
            print(f"   Stage 1: Skipped (no batch files to process)")
        
        merged_codes_list = []
        
        for i in tqdm(range(0, len(batch_files), merge_batch_size), desc=f"   Merging batches"):
            batch_chunk = batch_files[i:i + merge_batch_size]
            
            # 加载当前chunk的所有batch
            chunk_codes = []
            for batch_file in batch_chunk:
                try:
                    codes = torch.load(batch_file, map_location='cpu', weights_only=False)
                    chunk_codes.append(codes)
                    os.remove(batch_file)  # 立即删除临时文件
                except Exception as e:
                    print(f"   ERROR loading {batch_file}: {e}")
                    continue
            
            if not chunk_codes:
                continue
            
            # 合并当前chunk
            merged_chunk = torch.cat(chunk_codes, dim=0)
            merged_codes_list.append(merged_chunk)
            del chunk_codes, merged_chunk
            gc.collect()
            
            # 如果累积的chunks达到阈值，保存为中间文件
            if len(merged_codes_list) >= accumulate_threshold:
                # 合并累积的chunks
                temp_merged = torch.cat(merged_codes_list, dim=0)
                merged_codes_list = []
                
                # 保存到中间文件
                intermediate_file = os.path.join(intermediate_dir, f"intermediate_{intermediate_idx:06d}.pt")
                torch.save(temp_merged, intermediate_file)
                intermediate_files.append(intermediate_file)
                intermediate_idx += 1
                
                del temp_merged
                gc.collect()
                print(f"   Saved intermediate file {intermediate_idx}: shape {intermediate_file}")
        
        # 保存剩余的chunks
        if merged_codes_list:
            temp_merged = torch.cat(merged_codes_list, dim=0)
            merged_codes_list = []
            
            intermediate_file = os.path.join(intermediate_dir, f"intermediate_{intermediate_idx:06d}.pt")
            torch.save(temp_merged, intermediate_file)
            intermediate_files.append(intermediate_file)
            intermediate_idx += 1
            
            del temp_merged
            gc.collect()
        
        total_intermediate = len(intermediate_files)
        if total_intermediate > len(existing_intermediate_files):
            newly_created = total_intermediate - len(existing_intermediate_files)
            print(f"   Created {newly_created} new intermediate files")
            print(f"   Total intermediate files: {total_intermediate} ({len(existing_intermediate_files)} existing + {newly_created} new)")
        else:
            print(f"   Total intermediate files: {total_intermediate} (all from previous run)")
        
        # 第二阶段：合并所有中间文件
        if len(intermediate_files) == 0:
            print(f"   ERROR: No intermediate files found!")
            continue
        
        if len(intermediate_files) == 1:
            # 只有一个中间文件
            new_codes = torch.load(intermediate_files[0], map_location='cpu', weights_only=False)
            os.remove(intermediate_files[0])
        else:
            # 多个中间文件，需要合并
            print(f"   Stage 2: Merging {len(intermediate_files)} intermediate files...")
            
            # 使用更小的批次合并中间文件
            final_list = []
            merge_intermediate_batch = 2  # 每次合并2个中间文件
            
            for i in tqdm(range(0, len(intermediate_files), merge_intermediate_batch), desc=f"   Merging intermediate"):
                intermediate_chunk = intermediate_files[i:i + merge_intermediate_batch]
                
                chunk_data = []
                for inter_file in intermediate_chunk:
                    data = torch.load(inter_file, map_location='cpu', weights_only=False)
                    chunk_data.append(data)
                    os.remove(inter_file)  # 删除已处理的中间文件
                
                merged_chunk = torch.cat(chunk_data, dim=0)
                final_list.append(merged_chunk)
                del chunk_data, merged_chunk
                gc.collect()
                
                # 如果累积太多，先合并一部分
                if len(final_list) >= accumulate_threshold:
                    temp_final = torch.cat(final_list, dim=0)
                    final_list = [temp_final]
                    del temp_final
                    gc.collect()
            
            # 最终合并
            if len(final_list) == 1:
                new_codes = final_list[0]
            else:
                print(f"   Final merge of {len(final_list)} chunks...")
                new_codes = torch.cat(final_list, dim=0)
            
            del final_list
            gc.collect()
        
        # 处理追加或保存
        if append_mode and existing_samples > 0:
            # 追加模式：加载现有文件并追加新数据
            print(f"   Appending to existing file...")
            existing_codes = torch.load(output_file, map_location='cpu', weights_only=False)
            print(f"   Existing: {existing_codes.shape}, New: {new_codes.shape}")
            
            # 合并
            final_codes = torch.cat([existing_codes, new_codes], dim=0)
            del existing_codes, new_codes
            gc.collect()
            
            # 保存
            print(f"   Saving merged file: {output_file}")
            torch.save(final_codes, output_file)
            print(f"   ✓ Saved: {output_file}, shape: {final_codes.shape}")
            del final_codes
            gc.collect()
        else:
            # 直接保存新文件
            print(f"   Saving final file: {output_file}")
            torch.save(new_codes, output_file)
            print(f"   ✓ Saved: {output_file}, shape: {new_codes.shape}")
            del new_codes
            gc.collect()
        
        # 清理中间目录
        try:
            os.rmdir(intermediate_dir)
        except:
            pass  # 目录可能不为空，忽略
        
        print(f"   ✓ Completed aug_idx={aug_idx}")


def main():
    parser = argparse.ArgumentParser(description="安全合并batch codes文件")
    parser.add_argument("--temp_dir", type=str, required=True,
                       help="临时batch文件目录路径")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="输出目录路径")
    parser.add_argument("--num_augmentations", type=int, required=True,
                       help="数据增强数量")
    parser.add_argument("--merge_batch_size", type=int, default=3,
                       help="每次合并的batch数量（默认3，越小越安全）")
    parser.add_argument("--accumulate_threshold", type=int, default=2,
                       help="累积阈值（默认2，越小越安全）")
    parser.add_argument("--aug_idx", type=int, default=None,
                       help="指定处理的aug_idx（默认None，处理所有）")
    parser.add_argument("--merge_weights", action="store_true",
                       help="同时合并position_weights文件")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Batch Codes Merger - Safe and Memory-Efficient")
    print("=" * 80)
    print(f"Temp directory: {args.temp_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Number of augmentations: {args.num_augmentations}")
    print(f"Merge batch size: {args.merge_batch_size}")
    print(f"Accumulate threshold: {args.accumulate_threshold}")
    print("=" * 80)
    
    # 检查temp_dir是否存在
    if not os.path.exists(args.temp_dir):
        raise FileNotFoundError(f"Temp directory not found: {args.temp_dir}")
    
    # 合并codes文件
    print("\n>> Merging codes files...")
    merge_batch_files_safe(
        temp_dir=args.temp_dir,
        output_dir=args.output_dir,
        num_augmentations=args.num_augmentations,
        merge_batch_size=args.merge_batch_size,
        accumulate_threshold=args.accumulate_threshold,
        file_prefix="codes",
        aug_idx=args.aug_idx
    )
    
    # 合并position_weights文件（如果启用）
    if args.merge_weights:
        print("\n>> Merging position_weights files...")
        merge_batch_files_safe(
            temp_dir=args.temp_dir,
            output_dir=args.output_dir,
            num_augmentations=args.num_augmentations,
            merge_batch_size=args.merge_batch_size,
            accumulate_threshold=args.accumulate_threshold,
            file_prefix="position_weights",
            aug_idx=args.aug_idx
        )
    
    print("\n" + "=" * 80)
    print("✓ All done!")
    print("=" * 80)


if __name__ == "__main__":
    main()

