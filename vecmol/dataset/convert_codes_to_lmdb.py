#!/usr/bin/env python3
"""
convert existing codes.pt or codes_XXX.pt files to LMDB format to improve data loading efficiency
support merging multiple codes files into a single LMDB database
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
    Convert codes files to LMDB database
    
    Args:
        codes_dir (str): codes file directory
        split (str): dataset split name (train/val/test)
        lmdb_path (str): output LMDB database path
        keys_path (str): output keys file path
        num_augmentations (int): num_augmentations, used to find corresponding format files
    
    Returns:
        num_augmentations (int): num_augmentations (number of codes files)
    """
    split_dir = os.path.join(codes_dir, split)
    
    if not os.path.exists(split_dir):
        raise FileNotFoundError(f"Codes directory not found: {split_dir}")
    
    # Find corresponding format files based on num_augmentations: codes_aug{num}_XXX.pt
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
    
    # Sort to ensure correct order
    list_codes.sort()
    print(f"Found {len(list_codes)} codes files matching pattern 'codes_aug{num_augmentations}_*.pt'")
    
    print(f"Found {len(list_codes)} codes files in {split_dir}")
    for code_file in list_codes:
        print(f"  - {code_file}")
    
    # Load only the first file to get shape information, assuming all files have the same shape
    print("Loading first file to get shape info...")
    first_code_path = os.path.join(split_dir, list_codes[0])
    first_codes = torch.load(first_code_path, weights_only=False)
    print(f"  Shape: {first_codes.shape}")
    
    samples_per_file = first_codes.shape[0]
    code_shape = first_codes.shape[1:]  # Save the shape of each sample (remove batch dimension)
    
    # Calculate the actual size of each sample
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
    
    # Estimate total number of samples (assuming all files have the same shape)
    total_samples = samples_per_file * len(list_codes)
    del first_codes  # Immediately release memory
    gc.collect()
    
    print(f"Estimated total samples: {total_samples} (assuming {samples_per_file} samples per file)")
    print(f"Code shape per sample: {code_shape}")
    print(f"Code dimension: {code_dim}")
    
    # Delete old LMDB file (if exists)
    if os.path.exists(lmdb_path):
        print(f"Removing existing LMDB file: {lmdb_path}")
        if os.path.isdir(lmdb_path):
            shutil.rmtree(lmdb_path)
        else:
            os.remove(lmdb_path)
    
    # Delete old lock file (if exists)
    lock_file = lmdb_path + "-lock"
    if os.path.exists(lock_file):
        print(f"Removing existing lock file: {lock_file}")
        os.remove(lock_file)
    
    # Create LMDB database
    # Add serialization overhead (pickle overhead) and LMDB overhead
    # pickle serialization adds ~50-100% overhead; LMDB has some overhead too
    estimated_size_per_sample = actual_size_per_sample * 2.0 + 4096  # 2x for serialization + overhead, 4096 bytes extra
    map_size = max(100 * (1024 * 1024 * 1024), total_samples * estimated_size_per_sample * 2)  # at least 100GB or 2x estimate
    
    print(f"Creating LMDB database with map_size: {map_size / (1024**3):.2f} GB")
    print(f"  Actual size per sample: {actual_size_per_sample / (1024**2):.2f} MB")
    print(f"  Estimated size per sample (with overhead): {estimated_size_per_sample / (1024**2):.2f} MB")
    db = lmdb.open(lmdb_path, map_size=int(map_size))
    
    # deal with each file one by one, read and write, avoid memory explosion
    BATCH_SIZE = 500
    
    keys = []
    global_index = 0
    try:
        for code_file in list_codes:
            code_path = os.path.join(split_dir, code_file)
            print(f"Processing {code_file}...")
            
            # Load current file (but we will process and release it as soon as possible)
            codes = torch.load(code_path, weights_only=False)
            num_samples = codes.shape[0]
            print(f"  Loaded {num_samples} samples, shape: {codes.shape}")
            
            # Write to LMDB in batches
            num_batches = (num_samples + BATCH_SIZE - 1) // BATCH_SIZE
            pbar = tqdm(range(0, num_samples, BATCH_SIZE), desc=f"  Writing {code_file}", total=num_batches)
            
            for batch_start in pbar:
                batch_end = min(batch_start + BATCH_SIZE, num_samples)
                
                # Use independent transactions for each batch
                with db.begin(write=True) as txn:
                    # Process each sample immediately, serialize and write, avoid accumulation in memory
                    for i in range(batch_start, batch_end):
                        key = str(global_index).encode()
                        
                        # Extract single sample (this will create a view, not copy data)
                        # But we need clone() to ensure that the entire tensor is not serialized when serializing
                        code_sample = codes[i].clone().detach()
                        
                        # Immediately serialize
                        try:
                            value = pickle.dumps(code_sample, protocol=pickle.HIGHEST_PROTOCOL)
                            
                            value_size_mb = len(value) / (1024 * 1024)
                            if value_size_mb > 500:  # 500MB
                                print(f"\nWarning: code {global_index} is very large: {value_size_mb:.2f} MB")
                            
                            if len(value) > 511 * 1024 * 1024:
                                raise ValueError(f"Code {global_index} is too large for LMDB: {value_size_mb:.2f} MB (max 511MB)")
                            
                            # Immediately write, avoid accumulation in memory
                            txn.put(key, value)
                            keys.append(str(global_index))
                            
                        except Exception as e:
                            print(f"\nError serializing code {global_index}: {e}")
                        
                        # Immediately release references to code_sample and value
                        del code_sample
                        if 'value' in locals():
                            del value
                        
                        global_index += 1
                
                # Force garbage collection after each batch, release memory
                gc.collect()
                
                # Update progress bar
                pbar.set_postfix({'samples': global_index, 'mem': f'{global_index * 0.36 / 1024:.1f}GB'})
            
            # Immediately release memory of the entire file
            del codes
            gc.collect()
            print(f"  Completed {code_file}, total samples so far: {global_index}")
        
    except Exception as e:
        print(f"Error during conversion: {e}")
        raise
    finally:
        db.close()
    
    # Save keys list
    torch.save(keys, keys_path)
    
    print(f"Successfully converted to LMDB: {lmdb_path}")
    print(f"Saved keys list: {keys_path}")
    print(f"Database contains {len(keys)} codes")
    
    # Verify generated LMDB file
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
    
    # Return data augmentation number
    num_augmentations = len(list_codes)
    return num_augmentations


def convert_position_weights_to_lmdb(codes_dir, split, codes_keys, num_augmentations=None):
    """
    Convert position_weights files to LMDB database
    
    Args:
        codes_dir (str): codes file directory
        split (str): dataset split name (train/val/test)
        codes_keys (list): codes keys list, used to ensure that position_weights and codes have the same indices
        num_augmentations (int): data augmentation number, used to find corresponding format files
    """
    split_dir = os.path.join(codes_dir, split)
    
    # Find corresponding format files based on num_augmentations: position_weights_aug{num}_XXX.pt
    list_weights = [
        f for f in os.listdir(split_dir)
        if os.path.isfile(os.path.join(split_dir, f)) and \
        f.startswith(f"position_weights_aug{num_augmentations}_") and f.endswith(".pt")
    ]
    
    if not list_weights:
        print(f"No position_weights files found in {split_dir} matching pattern 'position_weights_aug{num_augmentations}_*.pt', skipping...")
        return
    
    # Sort to ensure correct order
    list_weights.sort()
    print(f"\nFound {len(list_weights)} position_weights files matching pattern 'position_weights_aug{num_augmentations}_*.pt'")
    for weight_file in list_weights:
        print(f"  - {weight_file}")
    
    # Load first file to get shape information
    first_weight_path = os.path.join(split_dir, list_weights[0])
    first_weights = torch.load(first_weight_path, weights_only=False)
    print(f"  Position weights shape: {first_weights.shape}")
    
    samples_per_file = first_weights.shape[0]
    total_samples = samples_per_file * len(list_weights)
    del first_weights
    gc.collect()
    
    print(f"Estimated total position_weights samples: {total_samples}")
    
    # Verify that the number of position_weights matches the number of codes
    if len(codes_keys) != total_samples:
        print(f"Warning: Position weights count ({total_samples}) != codes count ({len(codes_keys)})")
        print("  Will still convert, but indices may not match correctly")
    
    # Create position_weights LMDB database
    # Generate file name based on num_augmentations, avoid overwriting different versions of files
    weights_lmdb_path = os.path.join(split_dir, f"position_weights_aug{num_augmentations}.lmdb")
    weights_keys_path = os.path.join(split_dir, f"position_weights_aug{num_augmentations}_keys.pt")
    
    # Delete old LMDB file (if exists)
    if os.path.exists(weights_lmdb_path):
        print(f"Removing existing position_weights LMDB file: {weights_lmdb_path}")
        if os.path.isdir(weights_lmdb_path):
            shutil.rmtree(weights_lmdb_path)
        else:
            os.remove(weights_lmdb_path)
    
    # Estimate size (position_weights is usually much smaller than codes)
    estimated_size_per_sample = samples_per_file * 4 * 2.0 + 4096  # float32, 2x overhead
    map_size = max(10 * (1024 * 1024 * 1024), total_samples * estimated_size_per_sample * 2)  # at least 10GB
    
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
    
    # Save keys list
    torch.save(keys, weights_keys_path)
    
    print(f"Successfully converted position_weights to LMDB: {weights_lmdb_path}")
    print(f"Saved keys list: {weights_keys_path}")
    print(f"Database contains {len(keys)} position_weights")
    
    # Verify
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
                       help="Codes file directory")
    parser.add_argument("--num_augmentations", type=int, required=True,
                       help="Data augmentation number (must be consistent with num_augmentations used in infer_codes)")
    parser.add_argument("--splits", type=str, nargs="+", 
                       default=["train", "val", "test"],
                        help="Dataset splits to convert")
    parser.add_argument("--skip_position_weights", action="store_true",
                       help="Skip conversion of position_weights files")
    
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
            # Generate file name based on num_augmentations
            lmdb_path = os.path.join(split_dir, f"codes_aug{num_augmentations}.lmdb")
            keys_path = os.path.join(split_dir, f"codes_aug{num_augmentations}_keys.pt")
            
            print(f"Looking for codes files matching pattern 'codes_aug{num_augmentations}_*.pt'")
            print(f"Will generate: {os.path.basename(lmdb_path)}")
            
            # Convert codes
            returned_num_aug = convert_codes_to_lmdb(args.codes_dir, split, lmdb_path, keys_path, num_augmentations)
            
            # Ensure that the returned number matches the expected number
            if returned_num_aug != num_augmentations:
                print(f"Warning: Expected {num_augmentations} augmentations, but got {returned_num_aug}")
            
            # If position_weights conversion is needed, load keys and convert
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

