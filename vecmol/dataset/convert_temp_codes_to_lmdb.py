#!/usr/bin/env python3
"""
Convert temp_batch files to LMDB format, skip merging step
Avoid OOM risk, more secure and efficient

Usage:
    python convert_temp_codes_to_lmdb.py \
        --temp_dir exps/neural_field/nf_drugs/20260113/lightning_logs/version_0/checkpoints/code_no_aug/train \
        --output_dir exps/neural_field/nf_drugs/20260113/lightning_logs/version_0/checkpoints/code_no_aug/train \
        --num_augmentations 1 \
        --aug_idx 0 \
        --delete_batches  # Optional: delete batch files after conversion
    
    python convert_temp_codes_to_lmdb.py \
    --temp_dir exps/neural_field/nf_drugs/20260113/lightning_logs/version_0/checkpoints/code_no_aug/train \
    --output_dir exps/neural_field/nf_drugs/20260113/lightning_logs/version_0/checkpoints/code_no_aug/train \
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
    Convert temp_batch files to LMDB, skip merging step
    
    Args:
        temp_dir: temp_batches directory path
        output_dir: output directory path (LMDB file save location)
        num_augmentations: data augmentation number
        aug_idx: augmentation index (0, 1, 2, etc.)
        file_prefix: file prefix ("codes" or "position_weights")
        delete_batches: whether to delete batch files after conversion
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
    
    # Find all batch files
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
    
    # Sort by batch index (extract from file name)
    def extract_batch_idx(filename):
        """Extract batch index from file name: codes_000_batch_000000.pt -> 0"""
        parts = filename.stem.split("_")
        if len(parts) >= 4:
            try:
                return int(parts[-1])
            except:
                return -1
        return -1
    
    batch_files.sort(key=lambda f: extract_batch_idx(f))
    
    # Verify continuity of file indices
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
    
    # Load first batch file to get shape information
    print("\nLoading first batch file to get shape info...")
    first_batch = torch.load(batch_files[0], map_location='cpu', weights_only=False)
    samples_per_batch = first_batch.shape[0]
    code_shape = first_batch.shape[1:]  # Shape of each sample
    
    print(f"  Samples per batch: {samples_per_batch}")
    print(f"  Code shape per sample: {code_shape}")
    
    # Calculate the size of each sample
    if len(first_batch.shape) == 3:
        actual_size_per_sample = code_shape[0] * code_shape[1] * 4  # float32 = 4 bytes
    elif len(first_batch.shape) == 2:
        actual_size_per_sample = code_shape[0] * 4
    else:
        actual_size_per_sample = first_batch.numel() // samples_per_batch * 4
    
    # Estimate total number of samples
    total_samples = samples_per_batch * len(batch_files)
    print(f"  Estimated total samples: {total_samples}")
    
    del first_batch
    gc.collect()
    
    # Generate output file path
    if file_prefix == "codes":
        lmdb_path = output_dir / f"codes_aug{num_augmentations}_{aug_idx:03d}.lmdb"
        keys_path = output_dir / f"codes_aug{num_augmentations}_{aug_idx:03d}_keys.pt"
    else:
        lmdb_path = output_dir / f"position_weights_aug{num_augmentations}_{aug_idx:03d}.lmdb"
        keys_path = output_dir / f"position_weights_aug{num_augmentations}_{aug_idx:03d}_keys.pt"
    
    # Delete old LMDB file (if exists)
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
    
    # Create LMDB database
    estimated_size_per_sample = actual_size_per_sample * 2.0 + 4096  # 2x for serialization and overhead
    map_size = max(100 * (1024 * 1024 * 1024), total_samples * estimated_size_per_sample * 2)  # at least 100GB
    
    print(f"\nCreating LMDB database with map_size: {map_size / (1024**3):.2f} GB")
    print(f"  Actual size per sample: {actual_size_per_sample / (1024**2):.2f} MB")
    db = lmdb.open(str(lmdb_path), map_size=int(map_size))
    
    # Batch processing, avoid memory peak
    BATCH_SIZE = 500  # Write 500 samples per batch and commit transaction
    
    keys = []
    global_index = 0
    processed_batches = 0
    
    try:
        for batch_file in tqdm(batch_files, desc="Processing batch files"):
            # Load current batch file
            batch_data = torch.load(batch_file, map_location='cpu', weights_only=False)
            num_samples = batch_data.shape[0]
            
            # Batch write to LMDB
            num_batches = (num_samples + BATCH_SIZE - 1) // BATCH_SIZE
            
            for batch_start in range(0, num_samples, BATCH_SIZE):
                batch_end = min(batch_start + BATCH_SIZE, num_samples)
                
                # Use independent transaction for each batch
                with db.begin(write=True) as txn:
                    # Process each sample immediately, serialize and write
                    for i in range(batch_start, batch_end):
                        key = str(global_index).encode()
                        
                        # Extract single sample
                        sample = batch_data[i].clone().detach()
                        
                        # Immediately serialize
                        try:
                            value = pickle.dumps(sample, protocol=pickle.HIGHEST_PROTOCOL)
                            
                            # Check the size after serialization (LMDB single value limit is about 511MB)
                            value_size_mb = len(value) / (1024 * 1024)
                            if value_size_mb > 500:  # 500MB
                                print(f"\n⚠️  Warning: code {global_index} is very large: {value_size_mb:.2f} MB")
                            
                            # LMDB single value size limit check
                            if len(value) > 511 * 1024 * 1024:
                                raise ValueError(f"Code {global_index} is too large for LMDB: {value_size_mb:.2f} MB (max 511MB)")
                            
                            # Immediately write
                            txn.put(key, value)
                            keys.append(str(global_index))
                            
                        except Exception as e:
                            print(f"\n❌ Error serializing code {global_index}: {e}")
                            raise
                        
                        # Immediately release references to sample and value
                        del sample
                        if 'value' in locals():
                            del value
                        
                        global_index += 1
                
                # Force garbage collection after each batch
                gc.collect()
            
            # Immediately release memory of the entire batch file
            del batch_data
            gc.collect()
            
            # Delete batch file (if enabled)
            if delete_batches:
                try:
                    batch_file.unlink()
                except Exception as e:
                    print(f"⚠️  Warning: Failed to delete {batch_file.name}: {e}")
            
            processed_batches += 1
            
            # Print progress every 100 batch files
            if processed_batches % 100 == 0:
                print(f"  Processed {processed_batches}/{len(batch_files)} batch files, total samples: {global_index}")
        
    except Exception as e:
        print(f"\n❌ Error during conversion: {e}")
        raise
    finally:
        db.close()
    
    # Save keys list
    torch.save(keys, keys_path)
    
    print(f"\n✅ Successfully converted to LMDB: {lmdb_path}")
    print(f"✅ Saved keys list: {keys_path}")
    print(f"✅ Database contains {len(keys)} codes")
    
    # Verify generated LMDB file
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
    Merge intermediate files and temp_batch files and convert to LMDB
    
    Args:
        intermediate_dir: intermediate file directory path
        temp_batches_dir: temp_batches directory path
        output_dir: output directory path (LMDB file save location)
        num_augmentations: data augmentation number
        aug_idx: augmentation index (0, 1, 2, etc.)
        file_prefix: file prefix ("codes" or "position_weights")
        delete_after_convert: whether to delete source files after conversion
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
    
    # Collect all files to process
    all_files = []  # [(file_path, file_type), ...] file_type: 'intermediate' or 'batch'
    
    # 1. Collect intermediate files (sorted by index)
    if intermediate_dir and intermediate_dir.exists():
        intermediate_files = [
            f for f in intermediate_dir.iterdir()
            if f.is_file() and f.name.startswith("intermediate_") and f.name.endswith(".pt")
        ]
        
        # Sort by index: intermediate_000000.pt -> 0
        def extract_intermediate_idx(filename):
            """Extract intermediate index from file name: intermediate_000000.pt -> 0"""
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
    
    # 2. Collect batch files (sorted by index)
    if file_prefix == "codes":
        pattern = f"codes_{aug_idx:03d}_batch_"
    else:
        pattern = f"position_weights_aug{num_augmentations}_{aug_idx:03d}_batch_"
    
    batch_files = [
        f for f in temp_batches_dir.iterdir()
        if f.is_file() and f.name.startswith(pattern) and f.name.endswith(".pt")
    ]
    
    # Sort by batch index: codes_000_batch_003659.pt -> 3659
    def extract_batch_idx(filename):
        """Extract batch index from file name: codes_000_batch_003659.pt -> 3659"""
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
    
    # Load first file to get shape information
    print("\nLoading first file to get shape info...")
    first_file, first_type = all_files[0]
    first_data = torch.load(first_file, map_location='cpu', weights_only=False)
    samples_per_file = first_data.shape[0]
    code_shape = first_data.shape[1:]
    
    print(f"  Samples per file: {samples_per_file}")
    print(f"  Code shape per sample: {code_shape}")
    
    # Calculate the size of each sample
    if len(first_data.shape) == 3:
        actual_size_per_sample = code_shape[0] * code_shape[1] * 4
    elif len(first_data.shape) == 2:
        actual_size_per_sample = code_shape[0] * 4
    else:
        actual_size_per_sample = first_data.numel() // samples_per_file * 4
    
    # Estimate total number of samples
    total_samples = samples_per_file * len(all_files)
    print(f"  Estimated total samples: {total_samples}")
    
    del first_data
    gc.collect()
    
    # Generate output file path
    if file_prefix == "codes":
        lmdb_path = output_dir / f"codes_aug{num_augmentations}_{aug_idx:03d}.lmdb"
        keys_path = output_dir / f"codes_aug{num_augmentations}_{aug_idx:03d}_keys.pt"
    else:
        lmdb_path = output_dir / f"position_weights_aug{num_augmentations}_{aug_idx:03d}.lmdb"
        keys_path = output_dir / f"position_weights_aug{num_augmentations}_{aug_idx:03d}_keys.pt"
    
    # Delete old LMDB file (if exists)
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
    
    # Create LMDB database
    estimated_size_per_sample = actual_size_per_sample * 2.0 + 4096
    map_size = max(100 * (1024 * 1024 * 1024), total_samples * estimated_size_per_sample * 2)
    
    print(f"\nCreating LMDB database with map_size: {map_size / (1024**3):.2f} GB")
    print(f"  Actual size per sample: {actual_size_per_sample / (1024**2):.2f} MB")
    db = lmdb.open(str(lmdb_path), map_size=int(map_size))
    
    # Batch processing
    BATCH_SIZE = 500
    keys = []
    global_index = 0
    processed_files = 0
    
    try:
        for file_path, file_type in tqdm(all_files, desc="Processing files"):
            # Load file
            file_data = torch.load(file_path, map_location='cpu', weights_only=False)
            num_samples = file_data.shape[0]
            
            # Batch write to LMDB
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
            
            # Release memory
            del file_data
            gc.collect()
            
            # Delete file (if enabled)
            if delete_after_convert:
                try:
                    file_path.unlink()
                except Exception as e:
                    print(f"⚠️  Warning: Failed to delete {file_path.name}: {e}")
            
            processed_files += 1
            
            # Print progress every 100 files
            if processed_files % 100 == 0:
                print(f"  Processed {processed_files}/{len(all_files)} files, total samples: {global_index}")
        
    except Exception as e:
        print(f"\n❌ Error during conversion: {e}")
        raise
    finally:
        db.close()
    
    # Save keys list
    torch.save(keys, keys_path)
    
    print(f"\n✅ Successfully converted to LMDB: {lmdb_path}")
    print(f"✅ Saved keys list: {keys_path}")
    print(f"✅ Database contains {len(keys)} codes")
    
    # Verify
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
    Convert temp_batch position_weights files to LMDB
    
    Args:
        temp_dir: temp_batches directory path
        output_dir: output directory path
        num_augmentations: data augmentation number
        aug_idx: augmentation index
        codes_keys: codes keys list, used to verify count matching
        delete_batches: whether to delete batch files after conversion
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
    parser = argparse.ArgumentParser(description="Convert temp_batch files to LMDB format")
    parser.add_argument("--temp_dir", type=str, default=None,
                       help="temp_batches directory path (optional if using --mixed_mode)")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="output directory path (LMDB file save location)")
    parser.add_argument("--num_augmentations", type=int, required=True,
                       help="data augmentation number")
    parser.add_argument("--aug_idx", type=int, default=0,
                       help="augmentation index (default 0)")
    parser.add_argument("--delete_batches", action="store_true",
                       help="delete batch files after conversion (save space)")
    parser.add_argument("--convert_weights", action="store_true",
                       help="convert position_weights files")
    parser.add_argument("--only_weights", action="store_true",
                       help="convert position_weights files, skip codes conversion")
    
    # Mixed mode parameter
    parser.add_argument("--mixed_mode", action="store_true",
                       help="mixed mode: process intermediate and batch files")
    parser.add_argument("--intermediate_dir", type=str, default=None,
                       help="intermediate file directory path (only used when using --mixed_mode)")
    
    args = parser.parse_args()
    
    # If --only_weights is specified, convert weights and return
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
            # Try to load codes keys for verification (optional)
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
                # Mixed mode: process intermediate + batch position_weights
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
                # Normal mode: only process batch position_weights
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
    
    # Check mode
    if args.mixed_mode:
        # Mixed mode: process intermediate + batch files
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
        
        # Convert codes files
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
        # Normal mode: only process batch files
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
        
        # Convert codes files
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
    
    # Convert position_weights files (if enabled)
    if args.convert_weights:
        print("\n>> Converting position_weights files...")
        try:
            # Load codes keys for verification
            keys_path = Path(args.output_dir) / f"codes_aug{args.num_augmentations}_{args.aug_idx:03d}_keys.pt"
            if keys_path.exists():
                codes_keys = torch.load(keys_path, weights_only=False)
                print(f"  Loaded codes keys: {len(codes_keys)} samples")
            else:
                codes_keys = None
                print(f"  ⚠️  Warning: Codes keys file not found, cannot verify position_weights count")
            
            if args.mixed_mode:
                # Mixed mode: process intermediate + batch position_weights
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
                # Normal mode: only process batch position_weights
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

