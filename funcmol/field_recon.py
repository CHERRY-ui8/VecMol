#!/usr/bin/env python3
"""
Field Reconstruction Evaluation Script

This script can be used for both field type evaluation (stage1) and neural field evaluation (stage2).

0. Define field method (converter), save path (gt_field version, dataset name, )
   - Create GNFConverter with specific gradient_field_method (gaussian_mag, tanh, etc.)
   - Set output directory for results

1. Set data config, load dataset
   - Load FieldDataset with appropriate parameters
   - Support gt_field and nf_field modes

2. Loop for data (Pool.map and pool.imap_unordered):
   2.1 Convert to field using different field generation methods:
       - gt_field mode: Use ground truth coordinates to generate field via mol2gnf
       - nf_field mode: Use trained neural field encoder/decoder to generate field
   2.2 Reconstruct mol
       - Use gnf2mol method with appropriate decoder and codes
       - Support ground truth field and neural field reconstruction
   2.3 Save mol (if use gt_field, save in /exps/gt_field; if use predicted_field, save in the nf path under /exps/neural_field)
       - Currently saves results to CSV format
       - Can be extended to save individual molecular files

3. If mol exist: analyze rmsd. save results (csv: rmsd, data_id, size, atom_count_mismatch, ), summary (mean, std, min, max)
   - Compute RMSD between ground truth and reconstructed coordinates
   - Track atom count mismatches and other metrics
   - Generate summary statistics for each field method
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import torch
import random
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
import hydra
from omegaconf import DictConfig, OmegaConf, ListConfig
from funcmol.dataset.dataset_field import create_gnf_converter
from funcmol.utils.utils_nf import load_neural_field
from funcmol.dataset.dataset_field import FieldDataset

def compute_rmsd(coords1: torch.Tensor, coords2: torch.Tensor) -> float:
    """Compute RMSD using Hungarian algorithm."""
    if coords1.shape[0] != coords2.shape[0]:
        return float('inf')

    if coords1.device != coords2.device:
        coords2 = coords2.to(coords1.device)
    
    dists = torch.cdist(coords1, coords2)
    row_indices, col_indices = linear_sum_assignment(dists.cpu().numpy())
    rmsd = torch.sqrt(torch.mean(dists[row_indices, col_indices] ** 2))
    return float(rmsd.item())


@hydra.main(version_base=None, config_path="configs", config_name="field_recon")
def main(config: DictConfig) -> None:
    # 设置全局随机种子
    seed = config.get('seed', 1234)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 确保CUDA操作的确定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # 设置PyTorch内存优化环境变量
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:128'
    
    # 0. Define field method (converter), save path (gt_field version, dataset name, )
    config_dict = OmegaConf.to_container(config, resolve=True)
    
    # Choose field mode
    field_mode = config.get('field_mode', 'gt_field')
    
    # Set output directory based on field mode
    # 使用与funcmol目录平级的exps目录
    exps_root = Path(__file__).parent.parent / "exps"
    
    if field_mode == 'gt_field':
        output_dir = exps_root / "gt_field"
    elif field_mode == 'nf_field':
        nf_path = config.get("nf_pretrained_path")
        nf_path_parts = Path(nf_path).parts
        try:
            neural_field_idx = nf_path_parts.index('neural_field')
            if neural_field_idx + 2 < len(nf_path_parts):
                exp_name = f"{nf_path_parts[neural_field_idx + 1]}/{nf_path_parts[neural_field_idx + 2]}"
            else:
                exp_name = nf_path_parts[neural_field_idx + 1]
        except ValueError:
            exp_name = Path(nf_path).parent.parent.name
        output_dir = exps_root / "neural_field" / exp_name
    else:
        raise ValueError(f"Unsupported field_mode: {field_mode}. Only 'gt_field' and 'nf_field' are supported.")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Set data config, load dataset
    # Create default converter for dataset, not used in field computation, only used for dataset loading
    default_converter = create_gnf_converter(config_dict)
    
    # Load dataset using OmegaConf - much more elegant!
    dataset = FieldDataset(
        gnf_converter=default_converter,
        dset_name=config.dset.dset_name,
        data_dir=config.dset.data_dir,
        elements=config.dset.elements,
        split=config.get('split', 'val'),  # train/val/test split
        n_points=config.dset.n_points,
        rotate=config.dset.data_aug if config.get('split', 'val') == "train" else False,
        resolution=config.dset.resolution,
        grid_dim=config.dset.grid_dim,
        radius=config.dset.atomic_radius,
        sample_full_grid=config.dset.get('sample_full_grid', False),
        debug_one_mol=config.get('debug_one_mol', False),
        debug_subset=config.get('debug_subset', False),
    )
    field_methods = config.get('field_methods', ['gaussian_mag', 'tanh'])
    
    # For gt_field and nf_field modes, load dataset for reconstruction
    # 优先检查是否手动指定了样本索引
    manual_sample_indices = config.get('sample_indices')
    
    if manual_sample_indices is not None:
        # 如果手动指定了样本索引，使用指定的索引
        # 处理 OmegaConf 的 ListConfig 类型，转换为 Python list
        if isinstance(manual_sample_indices, (list, tuple, ListConfig)):
            # 将 ListConfig 转换为普通 Python list
            if isinstance(manual_sample_indices, ListConfig):
                manual_sample_indices = OmegaConf.to_container(manual_sample_indices, resolve=True)
            sample_indices = [int(idx) for idx in manual_sample_indices]
            # 验证索引是否在有效范围内
            total_samples = len(dataset)
            valid_indices = [idx for idx in sample_indices if 0 <= idx < total_samples]
            if len(valid_indices) != len(sample_indices):
                invalid_indices = [idx for idx in sample_indices if idx < 0 or idx >= total_samples]
                print(f"Warning: Some sample indices are out of range [0, {total_samples-1}]: {invalid_indices}")
                print(f"Using {len(valid_indices)} valid indices out of {len(sample_indices)} specified")
            sample_indices = valid_indices
            sample_indices.sort()  # 保持索引顺序，便于调试
            print(f"Using manually specified sample indices: {sample_indices}")
        else:
            raise ValueError(f"sample_indices must be a list, tuple, or ListConfig, got {type(manual_sample_indices)}")
    else:
        # 如果没有手动指定，使用max_samples进行随机采样
        max_samples = config.get('max_samples')
        if max_samples is None or max_samples <= 0:
            max_samples = len(dataset)
            sample_indices = list(range(len(dataset)))
        else:
            # 从验证集中随机采样max_samples个分子，保留原始索引
            total_samples = len(dataset)
            if max_samples >= total_samples:
                sample_indices = list(range(total_samples))
            else:
                sample_indices = random.sample(range(total_samples), max_samples)
                sample_indices.sort()  # 保持索引顺序，便于调试
    
    print(f"Evaluating {len(sample_indices)} molecules (sampled from {len(dataset)} total) with field_mode: {field_mode}, methods: {field_methods}")
    print(f"Sample indices: {sample_indices[:10]}{'...' if len(sample_indices) > 10 else ''}")
        
    encoder = None
    decoder = None
    
    if field_mode == 'nf_field':
        if config.get("nf_pretrained_path") is None:
            raise ValueError("nf_pretrained_path must be specified for nf_field mode")
        encoder, decoder = load_neural_field(config["nf_pretrained_path"], config)
        print("Loaded neural field encoder/decoder")
        
        # 移动模型到GPU并设置为评估模式
        encoder = encoder.cuda()
        decoder = decoder.cuda()
        encoder.eval()
        decoder.eval()
    
    # 2. Loop for data (Pool.map and pool.imap_unordered)
    # Initialize CSV file with headers
    if field_mode == 'gt_field':
        csv_path = output_dir / "field_evaluation_results.csv"
    elif field_mode == 'nf_field':
        csv_path = output_dir / "nf_evaluation_results_2.csv"
    else:
        raise ValueError(f"Unsupported field_mode: {field_mode}. Only 'gt_field' and 'nf_field' are supported.")
    
    # 根据field_mode决定CSV列
    if field_mode == 'gt_field':
        # gt_field模式保持原有列
        csv_columns = ['sample_idx', 'field_mode', 'field_method', 'rmsd', 'size', 'atom_count_mismatch']
    elif field_mode == 'nf_field':
        # nf_field模式添加原子统计信息
        elements = config.dset.elements  # ["C", "H", "O", "N", "F"]
        csv_columns = ['sample_idx', 'field_mode', 'field_method', 'rmsd', 'size', 'atom_count_mismatch']
        
        # 添加原始分子每种元素的原子数量
        for element in elements:
            csv_columns.append(f'gt_{element}_count')
        
        # 添加重建分子每种元素的原子数量
        for element in elements:
            csv_columns.append(f'recon_{element}_count')
    
    # Only create new CSV file if it doesn't exist
    if not csv_path.exists():
        results_df = pd.DataFrame(columns=csv_columns)
        results_df.to_csv(csv_path, index=False)
    
    # Create molecular data save directory
    mol_save_dir = output_dir / "molecule"
    mol_save_dir.mkdir(parents=True, exist_ok=True)
    
    for sample_idx in tqdm(sample_indices, desc="Processing"):
        # For gt_field and nf_field modes, load sample from dataset
        sample = dataset[sample_idx]
        gt_coords = sample.pos
        gt_types = sample.x
        
        for field_method in field_methods:
            try:
                # 0. Define field method (converter), save path (gt_field version, dataset name, )
                # Create converter with specific field method
                method_config = config_dict.copy()
                method_config['converter']['gradient_field_method'] = field_method
                converter = create_gnf_converter(method_config)
                
                # 初始化decoder和codes变量
                # 注意：decoder在全局作用域中已经定义，这里不需要重新定义
                codes = None
                
                if field_mode == 'gt_field':
                    # 2.1 Convert to field using gt_field function (also can change to mode: using trained nf decoder of encoded gt_mol)
                    # For gt_field mode, we need to create a dummy decoder and codes                    
                    # Create dummy codes (needed for gnf2mol)
                    # 使用正确的维度: [batch, grid_size**3, code_dim]
                    grid_size = config.get('dset', {}).get('grid_size', 9)
                    code_dim = config.get('encoder', {}).get('code_dim', 128)
                    dummy_codes = torch.randn(1, grid_size**3, code_dim, device=gt_coords.device)
                    
                    # Create a dummy decoder that returns the ground truth field
                    class DummyDecoder:
                        def __init__(self, converter, gt_coords, gt_types):
                            self.converter = converter
                            self.gt_coords = gt_coords
                            self.gt_types = gt_types
                        
                        def __call__(self, query_points, codes):
                            return self.converter.mol2gnf(
                                self.gt_coords.unsqueeze(0), 
                                self.gt_types.unsqueeze(0), 
                                query_points
                            )
                    
                    dummy_decoder = DummyDecoder(converter, gt_coords, gt_types)
                    decoder = dummy_decoder  # 设置decoder变量
                    codes = dummy_codes  # 设置codes变量
                    
                elif field_mode == 'nf_field':
                    # 2.1 Convert to field using trained nf decoder of encoded gt_mol
                    # Use neural field encoder to get codes
                    # Create a torch_geometric Batch object for the encoder
                    from torch_geometric.data import Data, Batch
                    
                    # Create a Data object for the single molecule
                    data = Data(
                        pos=gt_coords,  # [n_atoms, 3]
                        x=gt_types,     # [n_atoms]
                        batch=torch.zeros(gt_coords.shape[0], dtype=torch.long, device=gt_coords.device)  # [n_atoms]
                    )
                    
                    # Create a batch with single molecule
                    batch = Batch.from_data_list([data])
                    # Move to GPU
                    batch = batch.cuda()
                    
                    with torch.no_grad():
                        codes = encoder(batch)
                
                # 2.2 Reconstruct mol
                recon_coords, recon_types = converter.gnf2mol(
                    decoder,
                    codes
                )
                
                # 2.3 Save mol (if use gt_field, save in /exps/gt_field; if use predicted_field, save in the nf path under /exps/neural_field)
                # Note: Currently we only save results to CSV, not individual mol files
                
                # 3. If mol exist: analyze rmsd. save results (csv: rmsd, data_id, size, atom_count_mismatch, ), summary (mean, std, min, max)
                # 确保recon_coords在正确的设备上
                recon_coords_device = recon_coords[0].to(gt_coords.device)
                
                # For gt_field and nf_field modes, compare with original samples
                rmsd = compute_rmsd(gt_coords, recon_coords_device)  # recon_coords is batched
                atom_count_mismatch = gt_coords.shape[0] != recon_coords[0].shape[0]
                original_size = gt_coords.shape[0]
                
                # Save result immediately to CSV
                result_row = {
                    'sample_idx': sample_idx,
                    'field_mode': field_mode,
                    'field_method': field_method,
                    'rmsd': rmsd,
                    'size': original_size,
                    'atom_count_mismatch': atom_count_mismatch
                }
                
                # 为nf_field模式添加原子统计信息
                if field_mode == 'nf_field':
                    elements = config.dset.elements  # ["C", "H", "O", "N", "F"]
                    
                    # For nf_field mode, calculate original molecule atom statistics
                    for i, element in enumerate(elements):
                        count = (gt_types == i).sum().item()
                        result_row[f'gt_{element}_count'] = count
                    
                    # 计算重建分子的原子统计
                    # 使用gnf2mol返回的实际原子类型信息
                    recon_types_device = recon_types[0].to(gt_coords.device)
                    # 过滤掉填充的原子（值为-1）
                    valid_recon_types = recon_types_device[recon_types_device != -1]
                    for i, element in enumerate(elements):
                        count = (valid_recon_types == i).sum().item()
                        result_row[f'recon_{element}_count'] = count
                
                # Append to CSV file immediately
                result_df = pd.DataFrame([result_row])
                result_df.to_csv(csv_path, mode='a', header=False, index=False)
                
                # Save molecular coordinates and types(NOTE: sdf/xyz/mol?)
                mol_file = mol_save_dir / f"sample_{sample_idx:04d}_{field_method}.npz"
                np.savez(mol_file, 
                        coords=recon_coords_device.cpu().numpy(),
                        types=gt_types.cpu().numpy(),
                        gt_coords=gt_coords.cpu().numpy(),
                        rmsd=rmsd)
                
            except Exception as e:
                print(f"Error processing sample {sample_idx} with {field_method}: {e}")
                import traceback
                traceback.print_exc()
                # Save error result as well
                error_row = {
                    'sample_idx': sample_idx,
                    'field_mode': field_mode,
                    'field_method': field_method,
                    'rmsd': float('inf'),
                    'size': gt_coords.shape[0],
                    'atom_count_mismatch': True
                }
                
                # 为nf_field模式添加原子统计信息（错误情况下设为0）
                if field_mode == 'nf_field':
                    elements = config.dset.elements  # ["C", "H", "O", "N", "F"]
                    
                    # 计算原始分子的原子统计
                    for i, element in enumerate(elements):
                        count = (gt_types == i).sum().item()
                        error_row[f'gt_{element}_count'] = count
                    
                    # 重建分子统计设为0（因为重建失败）
                    for element in elements:
                        error_row[f'recon_{element}_count'] = 0
                
                error_df = pd.DataFrame([error_row])
                error_df.to_csv(csv_path, mode='a', header=False, index=False)
    
    # Load and analyze results from CSV file
    print(f"Results saved to: {csv_path}")
    
    # Read results from CSV for summary
    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} results from CSV file")
        
        # For gt_field and nf_field modes, show reconstruction statistics
        print("\n=== Summary ===")
        for method in field_methods:
            method_df = df[df['field_method'] == method]
            if len(method_df) > 0:
                rmsd_vals = method_df['rmsd']
                # Filter out infinite values for statistics
                finite_rmsd = rmsd_vals[rmsd_vals != float('inf')]
                if len(finite_rmsd) > 0:
                    print(f"{method}: mean={finite_rmsd.mean():.4f}, std={finite_rmsd.std():.4f}, min={finite_rmsd.min():.4f}, max={finite_rmsd.max():.4f}, count={len(finite_rmsd)}")
                else:
                    print(f"{method}: No valid results (all errors)")
        
        print(f"Atom count mismatches: {df['atom_count_mismatch'].sum()}/{len(df)}")
        print(f"Total samples: {len(df)}")
        print(f"Error samples: {(df['rmsd'] == float('inf')).sum()}")
        
    except Exception as e:
        print(f"Error reading results from CSV: {e}")
        print("Results were saved incrementally during processing.")


if __name__ == "__main__":
    main()