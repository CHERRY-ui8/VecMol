# see:
# https://github.com/Genentech/voxmol/blob/main/voxmol/dataset/create_data_geomdrugs.py

import argparse
import gc
import os
import pickle
import torch
import urllib.request
import sys

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

    Args:
        raw_data_dir (str): The directory where the raw data files will be saved.

    Returns:
        None
    """
    urllib.request.urlretrieve(RAW_URL_TRAIN, os.path.join(raw_data_dir, "train_data.pickle"))
    urllib.request.urlretrieve(RAW_URL_VAL, os.path.join(raw_data_dir, "val_data.pickle"))
    urllib.request.urlretrieve(RAW_URL_TEST, os.path.join(raw_data_dir, "test_data.pickle"))


def get_file_size_mb(file_path):
    """Get file size in MB"""
    return os.path.getsize(file_path) / (1024 * 1024)


def preprocess_geom_drugs_dataset(raw_data_dir: str, data_dir: str, split: str = "train", batch_size: int = 100):
    """
    Preprocesses the geometry drugs dataset with memory-efficient batch processing.

    Args:
        raw_data_dir (str): The directory path where the raw data is stored.
        data_dir (str): The directory path where the preprocessed data will be saved.
        split (str, optional): The dataset split to preprocess. Defaults to "train".
        batch_size (int, optional): Number of molecules to process in each batch. Defaults to 100.

    Returns:
        tuple: A tuple containing two lists: the preprocessed data for the specified
            split and a smaller subset of the data.
    """
    pickle_path = os.path.join(raw_data_dir, f"{split}_data.pickle")
    print("  >> load data raw from ", pickle_path)
    
    # 检查文件大小
    file_size_mb = get_file_size_mb(pickle_path)
    print(f"  >> file size: {file_size_mb:.1f} MB")
    
    if file_size_mb > 1000:  # 如果文件大于1GB
        print(f"  >> Warning: Large file detected ({file_size_mb:.1f} MB). Using memory-efficient processing.")
        return preprocess_large_file(pickle_path, data_dir, split, batch_size)
    else:
        return preprocess_small_file(pickle_path, data_dir, split, batch_size)


def preprocess_large_file(pickle_path: str, data_dir: str, split: str, batch_size: int):
    """
    Memory-efficient processing for large files using streaming approach.
    """
    print("  >> using memory-efficient processing for large file")
    
    data, data_small = [], []
    num_errors = 0
    total_mols_processed = 0
    
    # 使用更小的批次大小来处理大文件
    small_batch_size = min(batch_size, 50)
    
    try:
        # 首先尝试获取文件的基本信息
        print("  >> estimating file structure...")
        with open(pickle_path, 'rb') as f:
            # 只读取一小部分来估计结构
            sample_data = pickle.load(f)
            if isinstance(sample_data, list):
                estimated_total = len(sample_data)
                print(f"  >> estimated total molecules: {estimated_total}")
            else:
                print("  >> unknown file structure, using conservative approach")
                estimated_total = 100000  # 保守估计
    except Exception as e:
        print(f"  >> error reading file: {e}")
        return [], []
    
    # 分批处理
    for batch_start in range(0, estimated_total, small_batch_size):
        try:
            print(f"  >> processing batch starting at molecule {batch_start}")
            
            # 加载当前批次的数据
            with open(pickle_path, 'rb') as f:
                all_data = pickle.load(f)
            
            if batch_start >= len(all_data):
                break
                
            batch_end = min(batch_start + small_batch_size, len(all_data))
            batch_data = all_data[batch_start:batch_end]
            
            # 处理当前批次的构象
            mols_confs = []
            for i, data_item in enumerate(batch_data):
                try:
                    _, all_conformers = data_item
                    for j, conformer in enumerate(all_conformers):
                        if j >= 5:
                            break
                        mols_confs.append(conformer)
                except Exception as e:
                    print(f"  >> error processing molecule {batch_start + i}: {e}")
                    continue
            
            if not mols_confs:
                print(f"  >> no valid conformers in batch {batch_start//small_batch_size + 1}")
                del batch_data, all_data
                gc.collect()
                continue
                
            # 为当前批次写入SDF文件
            batch_sdf_path = os.path.join(data_dir, f"{split}_batch_{batch_start//small_batch_size}.sdf")
            try:
                with Chem.SDWriter(batch_sdf_path) as w:
                    for m in mols_confs:
                        w.write(m)
                
                # 解析当前批次的坐标和原子类型
                coords, atname = utils.parseSDF(batch_sdf_path)
                atoms_channel = utils.atomlistToChannels(atname, hashing=ELEMENTS_HASH)
                radius = atomlistToRadius(atname, hashing=radiusSingleAtom)
                
                # 处理当前批次的分子
                for i, mol in enumerate(tqdm(mols_confs, desc=f"Batch {batch_start//small_batch_size + 1}")):
                    try:
                        smiles = Chem.MolToSmiles(mol, isomericSmiles=False)
                        if smiles is None:
                            num_errors += 1
                            continue
                            
                        datum = {
                            "mol": mol,
                            "smiles": smiles,
                            "coords": coords[i].clone(),
                            "atoms_channel": atoms_channel[i].clone(),
                            "radius": radius[i].clone(),
                        }
                        
                        data.append(datum)
                        if total_mols_processed < 5000:
                            data_small.append(datum)
                        
                        total_mols_processed += 1
                    except Exception as e:
                        print(f"  >> error processing conformer {i}: {e}")
                        num_errors += 1
                        continue
                
                # 清理当前批次的内存
                del batch_data, mols_confs, coords, atname, atoms_channel, radius, all_data
                gc.collect()
                
                # 删除临时SDF文件
                if os.path.exists(batch_sdf_path):
                    os.remove(batch_sdf_path)
                    
            except Exception as e:
                print(f"  >> error processing batch {batch_start//small_batch_size + 1}: {e}")
                if os.path.exists(batch_sdf_path):
                    os.remove(batch_sdf_path)
                continue
                
        except Exception as e:
            print(f"  >> error loading batch starting at {batch_start}: {e}")
            continue
    
    print(f"  >> split size: {len(data)} ({num_errors} errors)")
    return data, data_small


def preprocess_small_file(pickle_path: str, data_dir: str, split: str, batch_size: int):
    """
    Standard processing for smaller files.
    """
    print("  >> using standard processing for small file")
    
    with open(pickle_path, 'rb') as f:
        all_data = pickle.load(f)

    # get all conformations of all molecules
    mols_confs = []
    for i, data in enumerate(all_data):
        _, all_conformers = data
        for j, conformer in enumerate(all_conformers):
            if j >= 5:
                break
            mols_confs.append(conformer)

    # write sdf / load with PyUUL
    print("  >> write .sdf of all conformations and extract coords/types with PyUUL")
    sdf_path = os.path.join(data_dir, f"{split}.sdf")
    with Chem.SDWriter(sdf_path) as w:
        for m in mols_confs:
            w.write(m)
    coords, atname = utils.parseSDF(sdf_path)
    atoms_channel = utils.atomlistToChannels(atname, hashing=ELEMENTS_HASH)
    radius = atomlistToRadius(atname, hashing=radiusSingleAtom)

    # create the dataset
    print("  >> create the dataset for this split")
    data, data_small = [], []
    num_errors = 0
    for i, mol in enumerate(tqdm(mols_confs)):
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
    parser.add_argument("--raw_data_dir", type=str, default="data/drugs/raw/")
    parser.add_argument("--data_dir", type=str, default="data/drugs/")
    parser.add_argument("--batch_size", type=int, default=100, 
                       help="Number of molecules to process in each batch to reduce memory usage")
    args = parser.parse_args()

    if not os.path.isdir(args.raw_data_dir):
        os.makedirs(args.raw_data_dir, exist_ok=True)
        download_data(args.raw_data_dir)

    os.makedirs(args.data_dir, exist_ok=True)

    data, data_small = {}, {}
    for split in ["train", "val", "test"]:
        print(f">> preprocessing {split}...")

        dset, dset_small = preprocess_geom_drugs_dataset(args.raw_data_dir, args.data_dir, split, args.batch_size)
        torch.save(dset, os.path.join(args.data_dir, f"{split}_data.pth"),)
        torch.save(dset_small, os.path.join(args.data_dir, f"{split}_data_small.pth"),)

        del dset, dset_small
        gc.collect()
