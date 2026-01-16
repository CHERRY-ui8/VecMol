#!/usr/bin/env python3
"""
安全地复制和重命名batch文件
从 codes_001_batch_*.pt 复制到 codes_000_batch_*.pt
每一步都进行验证，确保数据安全
"""

import os
import shutil
from pathlib import Path
from tqdm import tqdm

def safe_copy_and_rename():
    # 源目录和目标目录
    source_dir = Path("/datapool/data2/home/pxg/data/hyc/funcmol-main-neuralfield/exps/neural_field/nf_drugs/20260113/lightning_logs/version_1/checkpoints/codes_no_shuffle/train/temp_batches")
    target_dir = Path("/datapool/data2/home/pxg/data/hyc/funcmol-main-neuralfield/exps/neural_field/nf_drugs/20260113/lightning_logs/version_1/checkpoints/code_no_aug/train/temp_batches")
    
    print("=" * 80)
    print("Safe Batch File Copy and Rename Script")
    print("=" * 80)
    print(f"Source directory: {source_dir}")
    print(f"Target directory: {target_dir}")
    print("=" * 80)
    
    # 步骤1：验证源目录存在
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")
    print(f"✓ Source directory exists")
    
    # 步骤2：列出所有源文件
    source_files = sorted([
        f for f in source_dir.iterdir()
        if f.is_file() and f.name.startswith("codes_001_batch_") and f.name.endswith(".pt")
    ])
    
    if not source_files:
        raise ValueError(f"No codes_001_batch_*.pt files found in {source_dir}")
    
    print(f"✓ Found {len(source_files)} source files (codes_001_batch_*.pt)")
    print(f"  First file: {source_files[0].name}")
    print(f"  Last file: {source_files[-1].name}")
    
    # 验证文件数量
    expected_count = 18227
    if len(source_files) != expected_count:
        print(f"⚠️  WARNING: Expected {expected_count} files, found {len(source_files)}")
        response = input(f"Continue anyway? (y/N): ")
        if response.lower() != 'y':
            print("Aborted by user")
            return
    else:
        print(f"✓ File count matches expected: {len(source_files)}")
    
    # 步骤3：创建目标目录
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"✓ Target directory created/verified: {target_dir}")
    
    # 步骤4：检查目标目录是否已有文件
    existing_target_files = list(target_dir.glob("codes_000_batch_*.pt"))
    if existing_target_files:
        print(f"⚠️  WARNING: Target directory already has {len(existing_target_files)} codes_000_batch_*.pt files")
        print(f"  These files will be preserved (not overwritten)")
        existing_indices = set()
        for f in existing_target_files:
            # 提取batch索引: codes_000_batch_007010.pt -> 7010
            try:
                parts = f.stem.split("_")
                if len(parts) >= 4:
                    batch_idx = int(parts[-1])
                    existing_indices.add(batch_idx)
            except:
                pass
        print(f"  Existing batch indices range: {min(existing_indices) if existing_indices else 'N/A'} to {max(existing_indices) if existing_indices else 'N/A'}")
    
    # 步骤5：准备复制列表（排除已存在的文件）
    files_to_copy = []
    skipped_count = 0
    
    for source_file in source_files:
        # 提取batch索引
        parts = source_file.stem.split("_")
        if len(parts) >= 4:
            batch_idx = parts[-1]  # 保持原始格式，如 "000000"
        else:
            print(f"⚠️  WARNING: Cannot parse batch index from {source_file.name}, skipping")
            skipped_count += 1
            continue
        
        # 生成目标文件名
        target_filename = f"codes_000_batch_{batch_idx}.pt"
        target_file = target_dir / target_filename
        
        # 如果目标文件已存在，跳过
        if target_file.exists():
            skipped_count += 1
            continue
        
        files_to_copy.append((source_file, target_file))
    
    print(f"\n📋 Copy plan:")
    print(f"  Total source files: {len(source_files)}")
    print(f"  Files to copy: {len(files_to_copy)}")
    print(f"  Files skipped (already exist): {skipped_count}")
    
    if not files_to_copy:
        print("⚠️  No files to copy (all already exist or invalid)")
        return
    
    # 步骤6：确认操作
    print(f"\n⚠️  Ready to copy {len(files_to_copy)} files")
    print(f"   This will copy files from {source_dir} to {target_dir}")
    print(f"   Files will be renamed from codes_001_batch_*.pt to codes_000_batch_*.pt")
    response = input(f"\nProceed with copy? (yes/no): ")
    if response.lower() != 'yes':
        print("Aborted by user")
        return
    
    # 步骤7：执行复制（使用硬链接以节省空间和时间，如果失败则使用普通复制）
    print(f"\n🔄 Starting copy operation...")
    copied_count = 0
    failed_count = 0
    
    for source_file, target_file in tqdm(files_to_copy, desc="Copying files"):
        try:
            # 先尝试使用硬链接（节省空间，如果同一文件系统）
            try:
                os.link(source_file, target_file)
            except (OSError, AttributeError):
                # 如果硬链接失败（不同文件系统或权限问题），使用普通复制
                shutil.copy2(source_file, target_file)
            
            # 验证复制成功
            if not target_file.exists():
                raise FileNotFoundError(f"Target file not created: {target_file}")
            
            # 验证文件大小
            source_size = source_file.stat().st_size
            target_size = target_file.stat().st_size
            if source_size != target_size:
                raise ValueError(f"Size mismatch: source={source_size}, target={target_size}")
            
            copied_count += 1
            
        except Exception as e:
            print(f"\n❌ ERROR copying {source_file.name} to {target_file.name}: {e}")
            failed_count += 1
            # 如果目标文件存在但损坏，删除它
            if target_file.exists():
                try:
                    target_file.unlink()
                except:
                    pass
    
    # 步骤8：最终验证
    print(f"\n📊 Copy operation completed:")
    print(f"  ✓ Successfully copied: {copied_count} files")
    if failed_count > 0:
        print(f"  ❌ Failed: {failed_count} files")
    
    # 验证目标目录中的文件
    final_target_files = list(target_dir.glob("codes_000_batch_*.pt"))
    print(f"  ✓ Total codes_000_batch_*.pt files in target: {len(final_target_files)}")
    
    # 检查文件编号的连续性
    if final_target_files:
        batch_indices = []
        for f in final_target_files:
            try:
                parts = f.stem.split("_")
                if len(parts) >= 4:
                    batch_idx = int(parts[-1])
                    batch_indices.append(batch_idx)
            except:
                pass
        
        if batch_indices:
            batch_indices.sort()
            print(f"  ✓ Batch index range: {min(batch_indices)} to {max(batch_indices)}")
            expected_range = set(range(min(batch_indices), max(batch_indices) + 1))
            actual_range = set(batch_indices)
            missing = expected_range - actual_range
            if missing:
                print(f"  ⚠️  WARNING: Missing batch indices: {sorted(list(missing))[:10]}... (showing first 10)")
            else:
                print(f"  ✓ All batch indices are continuous")
    
    print(f"\n✅ Operation completed successfully!")
    print(f"   You can now run merge_batch_codes.py to merge the files")


if __name__ == "__main__":
    try:
        safe_copy_and_rename()
    except KeyboardInterrupt:
        print("\n\n⚠️  Operation interrupted by user")
        print("   Partial files may have been copied. Check target directory.")
    except Exception as e:
        print(f"\n\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()

