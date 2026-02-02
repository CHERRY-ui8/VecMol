# see:
# https://github.com/Genentech/voxmol/blob/main/voxmol/dataset/create_data_qm9.py

import argparse
import gc
import os
import torch
import urllib.request
import pickle
import zipfile

from pyuul import utils
import numpy as np
import pandas as pd
from tqdm import tqdm

from rdkit import Chem
from rdkit import RDLogger

from vecmol.utils.utils_base import atomlistToRadius
from vecmol.utils.constants import ELEMENTS_HASH, radiusSingleAtom


RDLogger.DisableLog("rdApp.*")

RAW_URL = ("https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/"
           "molnet_publish/qm9.zip")
RAW_URL2 = "https://ndownloader.figshare.com/files/3195404"


def save_pickle(array, path):
    with open(path, "wb") as f:
        pickle.dump(array, f)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def split_data(data_dir: str):
    """
    Split the dataset into train, validation, and test sets.

    Args:
        data_dir (str): The directory path where the dataset is located.

    Returns:
        None
    """

    dataset = pd.read_csv(os.path.join(data_dir, "gdb9.sdf.csv"))

    n_samples = len(dataset)
    n_train = 100000
    n_test = int(0.1 * n_samples)
    n_val = n_samples - (n_train + n_test)

    # shuffle dataset with df.sample, then split
    # sample split as previous work
    train, val, test = np.split(dataset.sample(frac=1, random_state=42), [n_train, n_val + n_train])

    train.to_csv(os.path.join(data_dir, "train.csv"))
    val.to_csv(os.path.join(data_dir, "val.csv"))
    test.to_csv(os.path.join(data_dir, "test.csv"))


def download_data(data_dir: str):
    """
    Download the QM9 dataset and save it to the specified directory.

    Args:
        data_dir (str): The directory to save the dataset.

    Returns:
        None
    """
    os.makedirs(data_dir, exist_ok=True)
    path_data_zip = os.path.join(data_dir, "qm9_raw.zip")

    urllib.request.urlretrieve(RAW_URL, path_data_zip)
    with zipfile.ZipFile(path_data_zip, "r") as zip_ref:
        zip_ref.extractall(data_dir)

    os.unlink(path_data_zip)

    path_data_3195404 = os.path.join(data_dir, "uncharacterized.txt")
    urllib.request.urlretrieve(RAW_URL2, path_data_3195404)


def preprocess_QM9_dataset(data_dir: str, split: str = "train"):
    """
    Preprocesses the QM9 dataset.

    Args:
        data_dir (str): The directory path where the dataset is located.
        split (str, optional): The split of the dataset to preprocess. Defaults to "train".

    Returns:
        tuple: A tuple containing two lists: the preprocessed data and a smaller subset of the data.
    """
    target_df = pd.read_csv(os.path.join(data_dir, f"{split}.csv"), index_col=0)

    # Handle uncharacterized.txt file (may be empty if download failed)
    uncharacterized_path = os.path.join(data_dir, "uncharacterized.txt")
    skip = []
    if os.path.exists(uncharacterized_path) and os.path.getsize(uncharacterized_path) > 0:
        try:
            with open(uncharacterized_path, "r") as f:
                content = f.read()
                if content.strip():  # Check if file is not empty
                    skip = [int(x.split()[0]) - 1 for x in content.split("\n")[9:-2]]
        except Exception as e:
            print(f"  >> Warning: Error reading uncharacterized.txt: {e}, continuing without skip list")
            skip = []

    sdf_file_path = os.path.join(data_dir, "gdb9.sdf")
    suppl = Chem.SDMolSupplier(sdf_file_path, removeHs=False, sanitize=False)
    coords, atname = utils.parseSDF(sdf_file_path)
    atoms_channel = utils.atomlistToChannels(atname, hashing=ELEMENTS_HASH)
    radius = atomlistToRadius(atname, hashing=radiusSingleAtom)

    data = []
    data_small = []
    num_errors = 0
    for i, mol in enumerate(tqdm(suppl)):
        if i in skip or i not in target_df.index:
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
            "target": target_df.loc[i].to_dict()
        }
        data.append(datum)
        if i < 5000:
            data_small.append(datum)
    print(f"  >> split size: {len(data)} ({num_errors} errors)")

    return data, data_small


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/qm9/")
    args = parser.parse_args()

    # Check if data directory exists, if not download data
    if not os.path.isdir(args.data_dir):
        os.makedirs(args.data_dir, exist_ok=True)
        download_data(args.data_dir)
        split_data(args.data_dir)
    else:
        # Check if required files exist, download/split if missing
        required_files = ["gdb9.sdf", "gdb9.sdf.csv", "uncharacterized.txt", "train.csv", "val.csv", "test.csv"]
        missing_files = [f for f in required_files if not os.path.exists(os.path.join(args.data_dir, f))]
        
        if missing_files:
            print(f">> Missing files: {missing_files}")
            
            # Split data if CSV files are missing (this doesn't require uncharacterized.txt)
            if any(f in missing_files for f in ["train.csv", "val.csv", "test.csv"]):
                print(">> Splitting dataset...")
                split_data(args.data_dir)
            
            # Download uncharacterized.txt if missing (try with error handling)
            if "uncharacterized.txt" in missing_files:
                print(">> Downloading uncharacterized.txt...")
                path_data_3195404 = os.path.join(args.data_dir, "uncharacterized.txt")
                try:
                    urllib.request.urlretrieve(RAW_URL2, path_data_3195404)
                    print(">> Successfully downloaded uncharacterized.txt")
                except Exception as e:
                    print(f">> Warning: Failed to download uncharacterized.txt: {e}")
                    print(">> Creating empty uncharacterized.txt file (may affect preprocessing)")
                    # Create an empty file as fallback
                    with open(path_data_3195404, "w") as f:
                        f.write("")

    data, data_small = {}, {}
    for split in ["train", "val", "test"]:
        print(f">> preprocessing {split}...")

        dset, dset_small = preprocess_QM9_dataset(args.data_dir, split=split)
        torch.save(dset, os.path.join(args.data_dir, f"{split}_data.pth"),)
        torch.save(dset_small, os.path.join(args.data_dir, f"{split}_data_small.pth"))

        del dset, dset_small
        gc.collect()
