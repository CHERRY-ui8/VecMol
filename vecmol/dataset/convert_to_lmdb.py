#!/usr/bin/env python3
"""
Convert existing .pth data files to LMDB format for improved data loading efficiency
"""

import os
import pickle
import torch
import lmdb
from tqdm import tqdm
import argparse
from vecmol.utils.constants import ELEMENTS_HASH, PADDING_INDEX

# Allowed element ID list: C(0), H(1), O(2), N(3), F(4), S(5), Cl(6), Br(7)
# Filter out P(8), I(9), B(10) etc. elements
ALLOWED_ELEMENTS = {0, 1, 2, 3, 4, 5, 6, 7}


def convert_pth_to_lmdb(pth_path, lmdb_path, keys_path):
    """
    Convert .pth file to LMDB database (simplified version)
    
    Args:
        pth_path (str): .pth file path
        lmdb_path (str): output LMDB database path
        keys_path (str): output keys file path
    """
    print(f"Converting {pth_path} to LMDB format...")
    
    # Load .pth file
    data = torch.load(pth_path, weights_only=False)
    print(f"Loaded {len(data)} molecules from {pth_path}")
    
    # Check data structure
    if len(data) > 0:
        sample = data[0]
        print(f"Sample keys: {list(sample.keys()) if isinstance(sample, dict) else type(sample)}")
    
    # Delete old LMDB file (if exists)
    if os.path.exists(lmdb_path):
        print(f"Removing existing LMDB file: {lmdb_path}")
        if os.path.isdir(lmdb_path):
            import shutil
            shutil.rmtree(lmdb_path)
        else:
            os.remove(lmdb_path)
    
    # Delete old lock file (if exists)
    lock_file = lmdb_path + "-lock"
    if os.path.exists(lock_file):
        print(f"Removing existing lock file: {lock_file}")
        os.remove(lock_file)
    
    # Create LMDB database - using simplified parameters
    db = lmdb.open(lmdb_path, map_size=20*(1024*1024*1024))
    
    n_valid = 0
    n_invalid = 0
    valid_keys = []  # Store actual valid keys written
    try:
        with db.begin(write=True) as txn:
            for i, datum in enumerate(tqdm(data, desc="Converting to LMDB")):
                # Use index as key
                key = str(i).encode()
                
                # Check data integrity
                if not isinstance(datum, dict):
                    print(f"Warning: datum {i} is not a dict: {type(datum)}")
                    continue

                # check elements are valid - only allow specified elements (C, H, O, N, F, S, Cl, Br)
                if "atoms_channel" in datum:
                    elements_valid = True
                    atoms = datum["atoms_channel"][datum["atoms_channel"] != PADDING_INDEX]
                    for atom in atoms.unique():
                        atom_id = int(atom.item())
                        # Check if element is in allowed list
                        if atom_id not in ALLOWED_ELEMENTS:
                            print(f"Warning: atom {atom} (id: {atom_id}) not in allowed elements {ALLOWED_ELEMENTS}, filtering out")
                            elements_valid = False
                            break
                    if elements_valid:
                        n_valid += 1
                        valid_keys.append(str(i))  # Record valid keys
                    else:
                        n_invalid += 1
                        continue

                # Serialize data
                try:
                    value = pickle.dumps(datum)
                except Exception as e:
                    print(f"Error serializing datum {i}: {e}")
                    continue
                
                # Write to database
                txn.put(key, value)
        
    except Exception as e:
        print(f"Error during conversion: {e}")
        raise
    finally:
        db.close()
    
    # Save serializable keys list, using actual valid keys
    torch.save(valid_keys, keys_path)
    
    print(f"Successfully converted to LMDB: {lmdb_path}")
    print(f"Saved keys list: {keys_path}")
    print(f"Database contains {n_valid} molecules")
    print(f"Valid molecules: {n_valid}, Invalid molecules: {n_invalid}")
    
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


def main():
    parser = argparse.ArgumentParser(description="Convert .pth files to LMDB format")
    parser.add_argument("--data_dir", type=str, required=True, 
                       help="data directory path")
    parser.add_argument("--dset_name", type=str, required=True,
                       choices=["qm9", "drugs", "drugs_no_h", "cremp"],
                       help="dataset name")
    parser.add_argument("--splits", type=str, nargs="+", 
                       default=["train", "val", "test"],
                       help="data splits to convert")
    
    args = parser.parse_args()
    
    data_dir = os.path.join(args.data_dir, args.dset_name)
    
    for split in args.splits:
        # Build file path
        if args.dset_name == "cremp":
            fname = f"{split}_50_data"
        else:
            fname = f"{split}_data"
        # Note: drugs_no_h uses the same file naming convention as drugs
        
        pth_path = os.path.join(data_dir, f"{fname}.pth")
        lmdb_path = os.path.join(data_dir, f"{fname}.lmdb")
        keys_path = os.path.join(data_dir, f"{fname}_keys.pt")
        
        # Check if source file exists
        if not os.path.exists(pth_path):
            print(f"Warning: {pth_path} does not exist, skipping...")
            continue
        
        # Convert file
        convert_pth_to_lmdb(pth_path, lmdb_path, keys_path)


if __name__ == "__main__":
    main()
