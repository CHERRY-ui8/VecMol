# see:
# https://github.com/Genentech/voxmol/blob/main/voxmol/dataset/create_data_geomdrugs.py

import argparse
import gc
import os
import pickle
import subprocess
import sys
import numpy as np
import torch
import urllib.request
from urllib.request import urlopen

from pyuul import utils
from tqdm import tqdm

from rdkit import Chem
from rdkit import RDLogger

from vecmol.utils.utils_base import atomlistToRadius, xyz_to_sdf
from vecmol.utils.constants import ELEMENTS_HASH, radiusSingleAtom

RDLogger.DisableLog("rdApp.*")

RAW_URL_TRAIN = "https://bits.csb.pitt.edu/files/geom_raw/train_data.pickle"
RAW_URL_VAL = "https://bits.csb.pitt.edu/files/geom_raw/val_data.pickle"
RAW_URL_TEST = "https://bits.csb.pitt.edu/files/geom_raw/test_data.pickle"


def download_with_wget(url: str, filepath: str, filename: str):
    """
    Download using wget (faster, supports resume).
    
    Args:
        url (str): URL to download from
        filepath (str): Local file path to save to
        filename (str): Display name for progress bar
    """
    try:
        subprocess.run(
            ['wget', '--continue', '--progress=bar:force', '-O', filepath, url],
            check=True
        )
        print(f"  >> Successfully downloaded {filename} using wget")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def download_with_aria2(url: str, filepath: str, filename: str):
    """
    Download using aria2c (fastest, supports multi-threaded download and resume).
    
    Args:
        url (str): URL to download from
        filepath (str): Local file path to save to
        filename (str): Display name for progress bar
    """
    try:
        # Use 16 connections for faster download
        subprocess.run(
            ['aria2c', '--continue=true', '--max-connection-per-server=16',
             '--split=16', '--min-split-size=1M', '--max-tries=5',
             '--out', filepath, url],
            check=True
        )
        print(f"  >> Successfully downloaded {filename} using aria2c")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def download_with_progress(url: str, filepath: str, filename: str):
    """
    Download a file with progress bar using urllib (fallback method).
    
    Args:
        url (str): URL to download from
        filepath (str): Local file path to save to
        filename (str): Display name for progress bar
    """
    response = urlopen(url)
    total_size = int(response.headers.get('Content-Length', 0))
    
    with open(filepath, 'wb') as f, tqdm(
        desc=f"  >> Downloading {filename}",
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        while True:
            chunk = response.read(8192)  # 8KB chunks
            if not chunk:
                break
            f.write(chunk)
            pbar.update(len(chunk))


def download_data(raw_data_dir: str, use_fast_download: bool = True):
    """
    Download the raw data files from the specified URLs and save them in the given directory.
    Tries to use faster download tools (aria2c or wget) if available, falls back to urllib.

    Args:
        raw_data_dir (str): The directory where the raw data files will be saved.
        use_fast_download (bool): Whether to try using aria2c or wget first (default: True).

    Returns:
        None
    """
    urls = {
        "train_data.pickle": RAW_URL_TRAIN,
        "val_data.pickle": RAW_URL_VAL,
        "test_data.pickle": RAW_URL_TEST,
    }
    
    for filename, url in urls.items():
        filepath = os.path.join(raw_data_dir, filename)
        success = False
        
        if use_fast_download:
            # Try aria2c first (fastest)
            print(f"  >> Attempting to download {filename} using aria2c...")
            if download_with_aria2(url, filepath, filename):
                success = True
            else:
                # Try wget as fallback
                print(f"  >> aria2c not available, trying wget...")
                if download_with_wget(url, filepath, filename):
                    success = True
        
        # Fallback to urllib if fast tools are not available or failed
        if not success:
            print(f"  >> Using urllib to download {filename}...")
            try:
                download_with_progress(url, filepath, filename)
                success = True
            except Exception as e:
                print(f"  >> Warning: Failed to download {filename}: {e}")
                raise RuntimeError(
                    f"Failed to download required file {filename}. "
                    f"Please check your network connection or proxy settings.\n"
                    f"For faster downloads, install aria2c: sudo apt-get install aria2\n"
                    f"Or install wget: sudo apt-get install wget\n"
                    f"Or download manually from: {url}"
                )
        
        if success and os.path.exists(filepath):
            file_size = os.path.getsize(filepath) / (1024**3)  # Size in GB
            print(f"  >> Successfully downloaded {filename} ({file_size:.2f} GB)")


def remove_hydrogens(mol):
    """
    Remove hydrogen atoms from an RDKit molecule while preserving the conformer coordinates.
    
    Args:
        mol: RDKit molecule object
        
    Returns:
        RDKit molecule object without hydrogen atoms
    """
    # Create a copy of the molecule
    mol_no_h = Chem.RWMol(mol)
    
    # Get indices of atoms to remove (H atoms have atomic number 1)
    atoms_to_remove = []
    for atom in mol_no_h.GetAtoms():
        if atom.GetAtomicNum() == 1:  # Hydrogen
            atoms_to_remove.append(atom.GetIdx())
    
    # Remove atoms in reverse order to maintain correct indices
    for idx in sorted(atoms_to_remove, reverse=True):
        mol_no_h.RemoveAtom(idx)
    
    return mol_no_h.GetMol()


def preprocess_geom_drugs_dataset(raw_data_dir: str, data_dir: str, split: str = "train", exclude_hydrogen: bool = False):
    """
    Preprocesses the geometry drugs dataset.

    Args:
        raw_data_dir (str): The directory path where the raw data is stored.
        data_dir (str): The directory path where the preprocessed data will be saved.
        split (str, optional): The dataset split to preprocess. Defaults to "train".
        exclude_hydrogen (bool, optional): Whether to exclude hydrogen atoms from the dataset. Defaults to False.

    Returns:
        tuple: A tuple containing two lists: the preprocessed data for the specified
            split and a smaller subset of the data.
    """
    print("  >> load data raw from ", os.path.join(raw_data_dir, f"{split}_data.pickle"))
    with open(os.path.join(raw_data_dir, f"{split}_data.pickle"), 'rb') as f:
        all_data = pickle.load(f)

    # get all conformations of all molecules
    print("  >> collecting conformations and removing hydrogen atoms...")
    mols_confs = []
    num_skipped = 0
    for i, data in enumerate(tqdm(all_data, desc="  >> Processing molecules")):
        _, all_conformers = data
        for j, conformer in enumerate(all_conformers):
            if j >= 5:
                break
            # Remove hydrogen atoms if requested
            if exclude_hydrogen:
                try:
                    conformer = remove_hydrogens(conformer)
                    if conformer is None or conformer.GetNumAtoms() == 0:
                        num_skipped += 1
                        continue
                except Exception as e:
                    print(f"  >> Warning: Failed to remove hydrogens from molecule {i}, conformer {j}: {e}")
                    num_skipped += 1
                    continue
            mols_confs.append(conformer)
    if num_skipped > 0:
        print(f"  >> Skipped {num_skipped} conformers due to errors")
    print(f"  >> Collected {len(mols_confs)} conformations")

    # write sdf / load with PyUUL
    print("  >> write .sdf of all conformations and extract coords/types with PyUUL")
    sdf_path = os.path.join(data_dir, f"{split}.sdf")
    # Track which molecules were successfully written to SDF
    successfully_written_mols = []
    num_failures = 0
    
    element_list = sorted(ELEMENTS_HASH.keys(), key=lambda x: ELEMENTS_HASH[x])
    
    # Open SDF file for writing
    sdf_file = open(sdf_path, 'w')
    
    for m in tqdm(mols_confs, desc="  >> Writing SDF"):
        try:
            # This completely bypasses kekulization issues since we only need coords
            conf = m.GetConformer()
            if conf is None:
                num_failures += 1
                continue
            
            n_atoms = m.GetNumAtoms()
            if n_atoms == 0:
                num_failures += 1
                continue
            
            # Extract coordinates
            coords = []
            atom_symbols = []
            for i in range(n_atoms):
                atom = m.GetAtomWithIdx(i)
                atom_symbols.append(atom.GetSymbol())
                pos = conf.GetAtomPosition(i)
                coords.append([pos.x, pos.y, pos.z])
            
            # Convert atom symbols to indices using ELEMENTS_HASH
            coords = np.array(coords)
            atom_types = np.array([ELEMENTS_HASH.get(symbol, -1) for symbol in atom_symbols])
            
            # Check if all atoms are valid
            if np.any(atom_types == -1):
                # Some atoms are not in ELEMENTS_HASH, skip this molecule
                num_failures += 1
                continue
            
            # Generate SDF string using xyz_to_sdf (no bonds, no kekulization needed)
            sdf_block = xyz_to_sdf(coords, atom_types, element_list)
            if sdf_block:
                sdf_file.write(sdf_block)
                successfully_written_mols.append(m)
            else:
                num_failures += 1
        except Exception as e:
            num_failures += 1
            if num_failures <= 10:  # Only print first 10 warnings
                print(f"  >> Warning: Failed to write molecule: {e}")
            elif num_failures == 11:
                print(f"  >> Warning: (Suppressing further write failure messages...)")
    
    sdf_file.close()
    
    if num_failures > 0:
        print(f"  >> Skipped {num_failures} molecules due to write failures")
    print(f"  >> Successfully wrote {len(successfully_written_mols)}/{len(mols_confs)} molecules to SDF")
    
    coords, atname = utils.parseSDF(sdf_path)
    atoms_channel = utils.atomlistToChannels(atname, hashing=ELEMENTS_HASH)
    radius = atomlistToRadius(atname, hashing=radiusSingleAtom)
    
    # Verify that the number of parsed molecules matches the number written
    if len(coords) != len(successfully_written_mols):
        print(f"  >> Warning: Mismatch between written molecules ({len(successfully_written_mols)}) "
              f"and parsed molecules ({len(coords)}). This may cause issues.")

    # create the dataset
    print("  >> create the dataset for this split")
    data, data_small = [], []
    num_errors = 0
    # Only process molecules that were successfully written to SDF
    for i, mol in enumerate(tqdm(successfully_written_mols, desc="  >> Creating dataset")):
        if i >= len(coords):
            print(f"  >> Warning: Index {i} out of bounds for coords (length {len(coords)}), skipping...")
            continue
        smiles = Chem.MolToSmiles(mol, isomericSmiles=False)
        if smiles is None:
            num_errors += 1
        datum = {
            "mol": mol,
            "smiles": smiles,
            "coords": coords[i].clone(),
            "atoms_channel": atoms_channel[i].clone(),
            "radius": radius[i].clone(),
        }

        data.append(datum)
        if i < 5000:
            data_small.append(datum)
    print(f"  >> split size: {len(data)} ({num_errors} errors)")

    return data, data_small


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_data_dir", type=str, default="dataset/data/drugs/raw/")
    parser.add_argument("--data_dir", type=str, default="dataset/data/drugs/")
    parser.add_argument("--exclude_hydrogen", action="store_true", 
                        help="Exclude hydrogen atoms from the dataset during preprocessing")
    parser.add_argument("--use_fast_download", action="store_true", default=True,
                        help="Try to use aria2c or wget for faster downloads (default: True)")
    parser.add_argument("--no_fast_download", dest="use_fast_download", action="store_false",
                        help="Disable fast download tools, use urllib only")
    args = parser.parse_args()

    # If exclude_hydrogen is True, modify the data_dir to use a different folder name
    # to avoid overwriting the original dataset with hydrogen atoms
    if args.exclude_hydrogen:
        # Extract the base directory and dataset name
        base_dir = os.path.dirname(args.data_dir.rstrip('/'))
        dataset_name = os.path.basename(args.data_dir.rstrip('/'))
        # Create a new directory name without hydrogen
        if dataset_name == "drugs":
            new_dataset_name = "drugs_no_h"
        else:
            new_dataset_name = f"{dataset_name}_no_h"
        args.data_dir = os.path.join(base_dir, new_dataset_name)
        print(f">> Excluding hydrogen atoms from the dataset")
        print(f">> Data will be saved to: {args.data_dir} (to avoid overwriting original dataset)")
        # Create the output directory immediately so user can see it exists
        os.makedirs(args.data_dir, exist_ok=True)
        print(f">> Created output directory: {args.data_dir}")
    else:
        print(">> Including all atoms (including hydrogen) in the dataset")

    # Create raw data directory if it doesn't exist
    os.makedirs(args.raw_data_dir, exist_ok=True)
    
    # Check if required pickle files exist, download if missing
    required_files = ["train_data.pickle", "val_data.pickle", "test_data.pickle"]
    missing_files = [f for f in required_files if not os.path.exists(os.path.join(args.raw_data_dir, f))]
    
    if missing_files:
        print(f">> Missing files: {missing_files}")
        print(">> Downloading missing files...")
        print(">> Tip: For faster downloads, install aria2c: sudo apt-get install aria2")
        print(">>      Or use wget: sudo apt-get install wget")
        try:
            download_data(args.raw_data_dir, use_fast_download=args.use_fast_download)
        except Exception as e:
            print(f">> Error: {e}")
            print(">> Please check your network connection, proxy settings, or download files manually.")
            raise

    # Create data directory if it doesn't exist (only needed if exclude_hydrogen is False)
    if not args.exclude_hydrogen:
        os.makedirs(args.data_dir, exist_ok=True)

    data, data_small = {}, {}
    for split in ["train", "val", "test"]:
        print(f">> preprocessing {split}...")
        
        # Check if pickle file exists before preprocessing
        pickle_path = os.path.join(args.raw_data_dir, f"{split}_data.pickle")
        if not os.path.exists(pickle_path):
            raise FileNotFoundError(f"Required file {pickle_path} not found. Please download it first.")

        dset, dset_small = preprocess_geom_drugs_dataset(args.raw_data_dir, args.data_dir, split, 
                                                          exclude_hydrogen=args.exclude_hydrogen)
        
        # Delete SDF file before saving to free up disk space (no longer needed after processing)
        sdf_path = os.path.join(args.data_dir, f"{split}.sdf")
        if os.path.exists(sdf_path):
            sdf_size = os.path.getsize(sdf_path) / (1024**3)  # Size in GB
            os.remove(sdf_path)
            print(f"  >> Deleted {split}.sdf ({sdf_size:.2f}GB) to free up disk space before saving")
        
        torch.save(dset, os.path.join(args.data_dir, f"{split}_data.pth"),)
        torch.save(dset_small, os.path.join(args.data_dir, f"{split}_data_small.pth"),)

        del dset, dset_small
        gc.collect()