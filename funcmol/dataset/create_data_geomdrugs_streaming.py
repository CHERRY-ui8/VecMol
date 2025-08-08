# Streaming version of create_data_geomdrugs.py
# This version processes the data without loading the entire file into memory

import argparse
import gc
import os
import pickle
import torch
import urllib.request
import sys
import tempfile

from pyuul import utils
from tqdm import tqdm

from rdkit import Chem
from rdkit import RDLogger

from funcmol.utils.utils_base import atomlistToRadius
from funcmol.utils.constants import ELEMENTS_HASH, radiusSingleAtom

RDLogger.DisableLog("rdApp.*")

RAW_URL_TRAIN = "https://bits.csb.pitt.edu/files/geom_raw/train_data.pickle"
RAW_URL_VAL = "https://bits.csb.pitt.edu/files/geom_raw/val_data.pickle"
RAW_URL_TEST = "https://bits.csb.pitt.edu/files/geom_raw/test_data.pickle"


def download_data(raw_data_dir: str):
    """
    Download the raw data files from the specified URLs and save them in the given directory.
    """
    urllib.request.urlretrieve(RAW_URL_TRAIN, os.path.join(raw_data_dir, "train_data.pickle"))
    urllib.request.urlretrieve(RAW_URL_VAL, os.path.join(raw_data_dir, "val_data.pickle"))
    urllib.request.urlretrieve(RAW_URL_TEST, os.path.join(raw_data_dir, "test_data.pickle"))


def get_file_size_mb(file_path):
    """Get file size in MB"""
    return os.path.getsize(file_path) / (1024 * 1024)


def process_single_molecule(mol_data, max_conformers=5):
    """
    Process a single molecule and its conformers.
    """
    try:
        _, all_conformers = mol_data
        mols_confs = []
        
        for j, conformer in enumerate(all_conformers):
            if j >= max_conformers:
                break
            mols_confs.append(conformer)
            
        return mols_confs
    except Exception as e:
        print(f"  >> error processing molecule: {e}")
        return []


def process_molecule_batch(mols_confs, batch_id, data_dir, split):
    """
    Process a batch of molecules and return the processed data.
    """
    if not mols_confs:
        return []
    
    # Create temporary SDF file
    with tempfile.NamedTemporaryFile(suffix='.sdf', delete=False) as tmp_file:
        sdf_path = tmp_file.name
    
    try:
        # Write molecules to SDF
        with Chem.SDWriter(sdf_path) as w:
            for m in mols_confs:
                w.write(m)
        
        # Parse coordinates and atom types
        coords, atname = utils.parseSDF(sdf_path)
        atoms_channel = utils.atomlistToChannels(atname, hashing=ELEMENTS_HASH)
        radius = atomlistToRadius(atname, hashing=radiusSingleAtom)
        
        # Process each molecule
        batch_data = []
        for i, mol in enumerate(mols_confs):
            try:
                smiles = Chem.MolToSmiles(mol, isomericSmiles=False)
                if smiles is None:
                    continue
                    
                datum = {
                    "mol": mol,
                    "smiles": smiles,
                    "coords": coords[i].clone(),
                    "atoms_channel": atoms_channel[i].clone(),
                    "radius": radius[i].clone(),
                }
                batch_data.append(datum)
                
            except Exception as e:
                print(f"  >> error processing conformer {i} in batch {batch_id}: {e}")
                continue
        
        return batch_data
        
    except Exception as e:
        print(f"  >> error processing batch {batch_id}: {e}")
        return []
    finally:
        # Clean up temporary file
        if os.path.exists(sdf_path):
            os.remove(sdf_path)


def preprocess_geom_drugs_streaming(raw_data_dir: str, data_dir: str, split: str = "train", 
                                   batch_size: int = 5, max_molecules: int = None):
    """
    Stream processing version that processes molecules one by one without loading the entire file.
    """
    pickle_path = os.path.join(raw_data_dir, f"{split}_data.pickle")
    print("  >> processing file:", pickle_path)
    
    file_size_mb = get_file_size_mb(pickle_path)
    print(f"  >> file size: {file_size_mb:.1f} MB")
    
    if file_size_mb > 1000:
        print(f"  >> Large file detected ({file_size_mb:.1f} MB). Using streaming processing.")
    
    data, data_small = [], []
    num_errors = 0
    total_mols_processed = 0
    batch_id = 0
    
    try:
        # Open the pickle file and process it streamingly
        with open(pickle_path, 'rb') as f:
            # We need to load the file, but we'll process it in small chunks
            print("  >> loading data (this may take a while for large files)...")
            all_data = pickle.load(f)
            
            total_molecules = len(all_data)
            print(f"  >> total molecules: {total_molecules}")
            
            if max_molecules:
                total_molecules = min(total_molecules, max_molecules)
                print(f"  >> limiting to {max_molecules} molecules")
            
            # Process in batches
            for i in range(0, total_molecules, batch_size):
                batch_end = min(i + batch_size, total_molecules)
                print(f"  >> processing batch {batch_id + 1} (molecules {i}-{batch_end-1})")
                
                # Process current batch
                batch_mols = []
                for j in range(i, batch_end):
                    mol_confs = process_single_molecule(all_data[j])
                    batch_mols.extend(mol_confs)
                
                if batch_mols:
                    batch_data = process_molecule_batch(batch_mols, batch_id, data_dir, split)
                    data.extend(batch_data)
                    
                    # Add to small dataset if needed
                    for datum in batch_data:
                        if total_mols_processed < 5000:
                            data_small.append(datum)
                        total_mols_processed += 1
                
                batch_id += 1
                
                # Force garbage collection after each batch
                del batch_mols
                gc.collect()
                
                # Print progress
                if batch_id % 10 == 0:
                    print(f"  >> processed {total_mols_processed} molecules so far")
    
    except Exception as e:
        print(f"  >> error processing file: {e}")
        return [], []
    
    print(f"  >> split size: {len(data)} ({num_errors} errors)")
    return data, data_small


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_data_dir", type=str, default="data/drugs/raw/")
    parser.add_argument("--data_dir", type=str, default="data/drugs/")
    parser.add_argument("--batch_size", type=int, default=5, 
                       help="Number of molecules to process in each batch")
    parser.add_argument("--max_molecules", type=int, default=None,
                       help="Maximum number of molecules to process (for testing)")
    args = parser.parse_args()

    if not os.path.isdir(args.raw_data_dir):
        os.makedirs(args.raw_data_dir, exist_ok=True)
        download_data(args.raw_data_dir)

    os.makedirs(args.data_dir, exist_ok=True)

    data, data_small = {}, {}
    for split in ["train", "val", "test"]:
        print(f">> preprocessing {split}...")

        dset, dset_small = preprocess_geom_drugs_streaming(
            args.raw_data_dir, args.data_dir, split, args.batch_size, args.max_molecules
        )
        
        if dset:  # Only save if we have data
            torch.save(dset, os.path.join(args.data_dir, f"{split}_data.pth"))
            torch.save(dset_small, os.path.join(args.data_dir, f"{split}_data_small.pth"))
            print(f"  >> saved {len(dset)} molecules for {split}")

        del dset, dset_small
        gc.collect() 