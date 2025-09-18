#!/usr/bin/env python3
"""
Field Reconstruction Evaluation Script

This script can be used for both field type evaluation (stage1) and neural field evaluation (stage2).

0. Define field method (converter), save path (gt_field version, dataset name, )
   - Create GNFConverter with specific gradient_field_method (gaussian_mag, tanh, etc.)
   - Set output directory for results

1. Set data config, load dataset
   - Load FieldDataset with appropriate parameters
   - Support gt_field, nf_field, and denoiser_field modes
   - Support gt_field, nf_field, and denoiser_field modes

2. Loop for data (Pool.map and pool.imap_unordered):
   2.1 Convert to field using different field generation methods:
   2.1 Convert to field using different field generation methods:
       - gt_field mode: Use ground truth coordinates to generate field via mol2gnf
       - nf_field mode: Use trained neural field encoder/decoder to generate field
       - denoiser_field mode: Use FuncMol denoiser to process codes and generate field via decoder
       - denoiser_field mode: Use FuncMol denoiser to process codes and generate field via decoder
   2.2 Reconstruct mol
       - Use gnf2mol method with appropriate decoder and codes
       - Support ground truth field, neural field, and denoiser field reconstruction
       - Support ground truth field, neural field, and denoiser field reconstruction
   2.3 Save mol (if use gt_field, save in /exps/gt_field; if use predicted_field, save in the nf path under /exps/neural_field)
       - Currently saves results to CSV format
       - Can be extended to save individual molecular files

3. If mol exist: analyze rmsd. save results (csv: rmsd, data_id, size, atom_count_mismatch, ), summary (mean, std, min, max)
   - Compute RMSD between ground truth and reconstructed coordinates
   - Track atom count mismatches and other metrics
   - Generate summary statistics for each field method
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
import hydra
from omegaconf import DictConfig, OmegaConf
from lightning import Fabric
from funcmol.utils.gnf_visualizer import create_gnf_converter
from funcmol.utils.utils_nf import load_neural_field
from funcmol.dataset.dataset_field import FieldDataset
from funcmol.models.funcmol import create_funcmol
from funcmol.utils.utils_fm import load_checkpoint_fm

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


def get_next_generated_idx(csv_path):
    """Get the next available generated index by checking existing CSV file."""
    if not csv_path.exists():
        return 0
    
    try:
        df = pd.read_csv(csv_path)
        if 'generated_idx' in df.columns:
            # Extract numeric part from generated_idx and find the maximum
            existing_indices = []
            for idx in df['generated_idx']:
                if isinstance(idx, str) and idx.isdigit():
                    existing_indices.append(int(idx))
                elif isinstance(idx, (int, float)):
                    existing_indices.append(int(idx))
            
            if existing_indices:
                return max(existing_indices) + 1
            else:
                return 0
        else:
            return 0
    except Exception:
        return 0


@hydra.main(version_base=None, config_path="configs", config_name="field_recon")
def main(config: DictConfig) -> None:
    # 设置PyTorch内存优化环境变量
    import os
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
    else:  # denoiser_field
        fm_path = config.get("fm_pretrained_path")
        fm_path_parts = Path(fm_path).parts
        try:
            funcmol_idx = fm_path_parts.index('funcmol')
            if funcmol_idx + 2 < len(fm_path_parts):
                exp_name = f"{fm_path_parts[funcmol_idx + 1]}/{fm_path_parts[funcmol_idx + 2]}"
            else:
                exp_name = fm_path_parts[funcmol_idx + 1]
        except ValueError:
            exp_name = Path(fm_path).parent.parent.name
        output_dir = exps_root / "funcmol" / exp_name
    
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
    
    if field_mode == 'denoiser_field':
        # For denoiser_field mode, max_samples represents the number of molecules to generate
        max_samples = config.get('max_samples')
        if max_samples is None or max_samples <= 0:
            max_samples = 1  # Default to generating 1 molecule
        sample_indices = list(range(max_samples))  # Generate indices from 0 to max_samples-1
        print(f"Generating {max_samples} molecules with field_mode: {field_mode}, methods: {field_methods}")
        print(f"Generation indices: {sample_indices}")
    else:
        # For gt_field and nf_field modes, load dataset for reconstruction
        max_samples = config.get('max_samples')
        if max_samples is None or max_samples <= 0:
            max_samples = len(dataset)
            sample_indices = list(range(len(dataset)))
        else:
            # 从验证集中随机采样max_samples个分子，保留原始索引
            import random
            random.seed(config.get('seed', 1234))  # 使用配置中的种子，确保可重现
            total_samples = len(dataset)
            if max_samples >= total_samples:
                sample_indices = list(range(total_samples))
            else:
                sample_indices = random.sample(range(total_samples), max_samples)
                sample_indices.sort()  # 保持索引顺序，便于调试
        
        print(f"Evaluating {len(sample_indices)} molecules (sampled from {len(dataset)} total) with field_mode: {field_mode}, methods: {field_methods}")
        print(f"Sample indices: {sample_indices[:10]}{'...' if len(sample_indices) > 10 else ''}")
    
    # 配置Fabric，只支持单GPU模式
    fabric = Fabric(
        accelerator="gpu",
        devices=1,
        precision="bf16-mixed"
    )
    print("Using single-GPU mode")
    
    fabric.launch()
    
    encoder = None
    decoder = None
    funcmol = None
    
    if field_mode == 'nf_field':
        if config.get("nf_pretrained_path") is None:
            raise ValueError("nf_pretrained_path must be specified for nf_field mode")
        encoder, decoder = load_neural_field(config["nf_pretrained_path"], fabric, config)
        print("Loaded neural field encoder/decoder")
        
        # 移动模型到GPU并设置为评估模式
        encoder = encoder.cuda()
        decoder = decoder.cuda()
        encoder.eval()
        decoder.eval()
    elif field_mode == 'denoiser_field':
        # Load neural field encoder/decoder for denoiser input/output
        if config.get("nf_pretrained_path") is None:
            raise ValueError("nf_pretrained_path must be specified for denoiser_field mode")
        encoder, decoder = load_neural_field(config["nf_pretrained_path"], fabric, config)
        
        # 移动模型到GPU并设置为评估模式
        encoder = encoder.cuda()
        decoder = decoder.cuda()
        encoder.eval()
        decoder.eval()
        
        # Load FuncMol denoiser
        if config.get("fm_pretrained_path") is None:
            raise ValueError("fm_pretrained_path must be specified for denoiser_field mode")
        funcmol = create_funcmol(config, fabric)

        funcmol = funcmol.cuda()
        funcmol, _ = load_checkpoint_fm(funcmol, config["fm_pretrained_path"], fabric=fabric)
        funcmol.eval()
        print("Loaded neural field encoder/decoder and FuncMol denoiser")
    
    # 2. Loop for data (Pool.map and pool.imap_unordered)
    # Initialize CSV file with headers
    if field_mode == 'gt_field':
        csv_path = output_dir / "field_evaluation_results.csv"
    elif field_mode == 'nf_field':
        csv_path = output_dir / "nf_evaluation_results.csv"
    else: # denoiser_field
        csv_path = output_dir / "denoiser_evaluation_results.csv"
    
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
    else:  # denoiser_field
        # denoiser_field模式只包含生成分子的信息，不包含ground truth
        elements = config.dset.elements  # ["C", "H", "O", "N", "F"]
        csv_columns = ['generated_idx', 'field_mode', 'field_method', 'size']
        
        # 只添加生成分子每种元素的原子数量
        for element in elements:
            csv_columns.append(f'generated_{element}_count')
    
    # Only create new CSV file if it doesn't exist
    if not csv_path.exists():
        results_df = pd.DataFrame(columns=csv_columns)
        results_df.to_csv(csv_path, index=False)
    
    # Create molecular data save directory
    mol_save_dir = output_dir / "molecule"
    mol_save_dir.mkdir(parents=True, exist_ok=True)
    
    for sample_idx in tqdm(sample_indices, desc="Processing"):
        if field_mode == 'denoiser_field':
            # For denoiser_field mode, we don't need ground truth data
            gt_coords = None
            gt_types = None
        else:
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
                
                elif field_mode == 'denoiser_field':
                    # 2.1 Generate new molecules using denoiser
                    # For denoiser_field mode, we generate new molecules from random noise
                    # rather than reconstructing specific samples from the dataset
                    
                    # Generate random noise codes as starting point
                    grid_size = config.get('dset', {}).get('grid_size', 9)
                    code_dim = config.get('encoder', {}).get('code_dim', 128)
                    batch_size = 1
                    
                    # Create random noise codes on the same device as funcmol
                    noise_codes = torch.randn(batch_size, grid_size**3, code_dim, device=next(funcmol.parameters()).device)
                    
                    # Process codes through denoiser to generate new molecular codes
                    with torch.no_grad():
                        denoised_codes = funcmol(noise_codes)
                    
                    codes = denoised_codes  # 设置codes变量
                
                # 2.2 Reconstruct mol
                recon_coords, recon_types = converter.gnf2mol(
                    decoder,
                    codes,
                    fabric=fabric
                )
                
                # 2.3 Save mol (if use gt_field, save in /exps/gt_field; if use predicted_field, save in the nf path under /exps/neural_field)
                # Note: Currently we only save results to CSV, not individual mol files
                
                # 3. If mol exist: analyze rmsd. save results (csv: rmsd, data_id, size, atom_count_mismatch, ), summary (mean, std, min, max)
                # 确保recon_coords在正确的设备上
                if field_mode == 'denoiser_field':
                    # For denoiser_field mode, use CPU for saving
                    recon_coords_device = recon_coords[0].cpu()
                else:
                    # For other modes, use the same device as gt_coords
                    recon_coords_device = recon_coords[0].to(gt_coords.device)
                
                if field_mode == 'denoiser_field':
                    # For denoiser_field mode, we don't compare with original samples
                    # since we're generating new molecules
                    generated_size = recon_coords[0].shape[0]
                    
                    # Get the next available generated index
                    generated_idx = get_next_generated_idx(csv_path)
                    
                    # Save result immediately to CSV
                    result_row = {
                        'generated_idx': generated_idx,
                        'field_mode': field_mode,
                        'field_method': field_method,
                        'size': generated_size
                    }
                else:
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
                
                # 为nf_field和denoiser_field模式添加原子统计信息
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
                elif field_mode == 'denoiser_field':
                    elements = config.dset.elements  # ["C", "H", "O", "N", "F"]
                    
                    # 计算生成分子的原子统计
                    # 使用gnf2mol返回的实际原子类型信息
                    recon_types_device = recon_types[0].cpu()
                    # 过滤掉填充的原子（值为-1）
                    valid_recon_types = recon_types_device[recon_types_device != -1]
                    for i, element in enumerate(elements):
                        count = (valid_recon_types == i).sum().item()
                        result_row[f'generated_{element}_count'] = count
                
                # Append to CSV file immediately
                result_df = pd.DataFrame([result_row])
                result_df.to_csv(csv_path, mode='a', header=False, index=False)
                
                # Save molecular coordinates and types(NOTE: sdf/xyz/mol?)
                if field_mode == 'denoiser_field':
                    mol_file = mol_save_dir / f"generated_{generated_idx:04d}_{field_method}.npz"
                    np.savez(mol_file, 
                            coords=recon_coords_device.cpu().numpy(),
                            types=recon_types[0].cpu().numpy())
                else:
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
                if field_mode == 'denoiser_field':
                    # Get the next available generated index for error case too
                    generated_idx = get_next_generated_idx(csv_path)
                    error_row = {
                        'generated_idx': generated_idx,
                        'field_mode': field_mode,
                        'field_method': field_method,
                        'size': 0
                    }
                    # 生成分子原子统计设为0
                    elements = config.dset.elements  # ["C", "H", "O", "N", "F"]
                    for element in elements:
                        error_row[f'generated_{element}_count'] = 0
                else:
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
        
        if field_mode == 'denoiser_field':
            # For denoiser_field mode, just show basic generation statistics
            print(f"\n=== Summary ===")
            print(f"Total generated molecules: {len(df)}")
            for method in field_methods:
                method_df = df[df['field_method'] == method]
                if len(method_df) > 0:
                    avg_size = method_df['size'].mean()
                    print(f"{method}: average size={avg_size:.1f} atoms, count={len(method_df)}")
        else:
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