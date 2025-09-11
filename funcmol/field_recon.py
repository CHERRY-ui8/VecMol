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
from omegaconf import DictConfig
from lightning import Fabric
from funcmol.utils.gnf_visualizer import create_gnf_converter, load_model
from funcmol.dataset.dataset_field import FieldDataset
from funcmol.models.funcmol import create_funcmol
from funcmol.utils.utils_fm import load_checkpoint_fm
        
from funcmol.models.funcmol import create_funcmol
from funcmol.utils.utils_fm import load_checkpoint_fm
        

def compute_rmsd(coords1: torch.Tensor, coords2: torch.Tensor) -> float:
    """Compute RMSD using Hungarian algorithm."""
    if coords1.shape[0] != coords2.shape[0]:
        return float('inf')
    
    dists = torch.cdist(coords1, coords2)
    row_indices, col_indices = linear_sum_assignment(dists.cpu().numpy())
    rmsd = torch.sqrt(torch.mean(dists[row_indices, col_indices] ** 2))
    return float(rmsd.item())


@hydra.main(version_base=None, config_path="configs", config_name="field_recon")
@hydra.main(version_base=None, config_path="configs", config_name="field_recon")
def main(config: DictConfig) -> None:
    # 0. Define field method (converter), save path (gt_field version, dataset name, )
    config_dict = dict(config)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Choose field mode
    field_mode = config.get('field_mode', 'gt_field')
    
    # 1. Set data config, load dataset
    # Create default converter for dataset, not used in field computation, only used for dataset loading
    default_converter = create_gnf_converter(config_dict)
    
    # Load dataset using OmegaConf - much more elegant!
    dataset = FieldDataset(
        gnf_converter=default_converter,
        dset_name=config.dset.dset_name,
        data_dir=config.dset.data_dir,
        elements=config.dset.elements,
        split=config.get('split', 'test'),  # train/val/test split
        n_points=config.dset.n_points,
        rotate=config.dset.data_aug if config.get('split', 'test') == "train" else False,
        resolution=config.dset.resolution,
        grid_dim=config.dset.grid_dim,
        radius=config.dset.atomic_radius,
        sample_full_grid=config.dset.get('sample_full_grid', False),
        debug_one_mol=config.get('debug_one_mol', False),
        debug_subset=config.get('debug_subset', False),
    )
    max_samples = config.get('max_samples')
    if max_samples is None or max_samples <= 0:
        max_samples = len(dataset)
    field_methods = config.get('field_methods', ['gaussian_mag', 'tanh'])
    
    print(f"Evaluating {max_samples} molecules with field_mode: {field_mode}, methods: {field_methods}")
    
    # Load models based on field mode
    # Load models based on field mode
    if field_mode == 'nf_field':
        fabric = Fabric()
        encoder, decoder = load_model(fabric, config)
        print("Loaded neural field encoder/decoder")
    elif field_mode == 'denoiser_field':
        fabric = Fabric()
        # Load neural field encoder/decoder for denoiser input/output
        encoder, decoder = load_model(fabric, config)
        # Load FuncMol denoiser
        if config.get("fm_pretrained_path") is None:
            raise ValueError("fm_pretrained_path must be specified for denoiser_field mode")
        funcmol = create_funcmol(config, fabric)
        funcmol, code_stats = load_checkpoint_fm(funcmol, config["fm_pretrained_path"], fabric=fabric)
        funcmol = fabric.setup_module(funcmol)
        funcmol.eval()
        print("Loaded neural field encoder/decoder and FuncMol denoiser")
    elif field_mode == 'denoiser_field':
        fabric = Fabric()
        # Load neural field encoder/decoder for denoiser input/output
        encoder, decoder = load_model(fabric, config)
        # Load FuncMol denoiser
        if config.get("fm_pretrained_path") is None:
            raise ValueError("fm_pretrained_path must be specified for denoiser_field mode")
        funcmol = create_funcmol(config, fabric)
        funcmol, code_stats = load_checkpoint_fm(funcmol, config["fm_pretrained_path"], fabric=fabric)
        funcmol = fabric.setup_module(funcmol)
        funcmol.eval()
        print("Loaded neural field encoder/decoder and FuncMol denoiser")
    
    # 2. Loop for data (Pool.map and pool.imap_unordered)
    # Initialize CSV file with headers
    csv_path = output_dir / "field_evaluation_results.csv"
    results_df = pd.DataFrame(columns=['sample_idx', 'field_mode', 'field_method', 'rmsd', 'size', 'atom_count_mismatch'])
    results_df.to_csv(csv_path, index=False)
    
    # Create molecular data save directory
    mol_save_dir = output_dir / field_mode
    mol_save_dir.mkdir(parents=True, exist_ok=True)
    
    for sample_idx in tqdm(range(min(max_samples, len(dataset))), desc="Processing"):
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
                
                if field_mode == 'gt_field':
                    # 2.1 Convert to field using gt_field function (also can change to mode: using trained nf decoder of encoded gt_mol)
                    # For gt_field mode, we need to create a dummy decoder and codes                    
                    # Create dummy codes (needed for gnf2mol)
                    dummy_codes = torch.randn(1, 8**3, 64, device=gt_coords.device)  # dummy codes
                    
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
                    
                    # Generate query points
                    n_query_points = converter.n_query_points
                    coords_min = gt_coords.min(dim=0)[0] - 2.0
                    coords_max = gt_coords.max(dim=0)[0] + 2.0
                    query_points = torch.rand(n_query_points, 3, device=gt_coords.device)
                    query_points = query_points * (coords_max - coords_min) + coords_min
                    query_points = query_points.unsqueeze(0)
                    
                    # Generate field data using dummy decoder
                    field_data = dummy_decoder(query_points, dummy_codes)
                    
                elif field_mode == 'nf_field':
                    # 2.1 Convert to field using trained nf decoder of encoded gt_mol
                    # Use neural field encoder to get codes
                    codes = encoder(gt_coords.unsqueeze(0), gt_types.unsqueeze(0))
                    
                    # Use neural field decoder to get field data
                    n_query_points = converter.n_query_points
                    coords_min = gt_coords.min(dim=0)[0] - 2.0
                    coords_max = gt_coords.max(dim=0)[0] + 2.0
                    query_points = torch.rand(n_query_points, 3, device=gt_coords.device)
                    query_points = query_points * (coords_max - coords_min) + coords_min
                    query_points = query_points.unsqueeze(0)
                    
                    field_data = decoder(query_points, codes)
                
                elif field_mode == 'denoiser_field':
                    # 2.1 Convert to field using denoiser-processed codes
                    # First encode with neural field encoder
                    raw_codes = encoder(gt_coords.unsqueeze(0), gt_types.unsqueeze(0))
                    
                    # Process codes through denoiser
                    with torch.no_grad():
                        denoised_codes = funcmol(raw_codes)
                    
                    # Use neural field decoder to get field data
                    n_query_points = converter.n_query_points
                    coords_min = gt_coords.min(dim=0)[0] - 2.0
                    coords_max = gt_coords.max(dim=0)[0] + 2.0
                    query_points = torch.rand(n_query_points, 3, device=gt_coords.device)
                    query_points = query_points * (coords_max - coords_min) + coords_min
                    query_points = query_points.unsqueeze(0)
                    
                    field_data = decoder(query_points, denoised_codes)
                
                elif field_mode == 'denoiser_field':
                    # 2.1 Convert to field using denoiser-processed codes
                    # First encode with neural field encoder
                    raw_codes = encoder(gt_coords.unsqueeze(0), gt_types.unsqueeze(0))
                    
                    # Process codes through denoiser
                    with torch.no_grad():
                        denoised_codes = funcmol(raw_codes)
                    
                    # Use neural field decoder to get field data
                    n_query_points = converter.n_query_points
                    coords_min = gt_coords.min(dim=0)[0] - 2.0
                    coords_max = gt_coords.max(dim=0)[0] + 2.0
                    query_points = torch.rand(n_query_points, 3, device=gt_coords.device)
                    query_points = query_points * (coords_max - coords_min) + coords_min
                    query_points = query_points.unsqueeze(0)
                    
                    field_data = decoder(query_points, denoised_codes)
                
                # 2.2 Reconstruct mol
                if field_mode == 'gt_field':
                    # Reconstruct mol using gnf2mol with dummy decoder (gt_field mode)
                    recon_coords, _ = converter.gnf2mol(
                        decoder=dummy_decoder,
                        codes=dummy_codes,
                        atom_types=gt_types.unsqueeze(0)
                    )
                elif field_mode == 'nf_field':
                elif field_mode == 'nf_field':
                    # Reconstruct mol using gnf2mol (nf_field mode)
                    recon_coords, _ = converter.gnf2mol(
                        decoder=decoder,
                        codes=codes,
                        atom_types=gt_types.unsqueeze(0)
                    )
                else:  # denoiser_field mode
                    # Reconstruct mol using gnf2mol (denoiser_field mode)
                    recon_coords, _ = converter.gnf2mol(
                        field_data, 
                        decoder=decoder,
                        codes=denoised_codes,
                        atom_types=gt_types.unsqueeze(0)
                    )
                
                # 2.3 Save mol (if use gt_field, save in /exps/gt_field; if use predicted_field, save in the nf path under /exps/neural_field)
                # Note: Currently we only save results to CSV, not individual mol files
                
                # 3. If mol exist: analyze rmsd. save results (csv: rmsd, data_id, size, atom_count_mismatch, ), summary (mean, std, min, max)
                rmsd = compute_rmsd(gt_coords, recon_coords[0])  # recon_coords is batched
                atom_count_mismatch = gt_coords.shape[0] != recon_coords[0].shape[0]
                
                # Save result immediately to CSV
                result_row = {
                    'sample_idx': sample_idx,
                    'field_mode': field_mode,
                    'field_method': field_method,
                    'rmsd': rmsd,
                    'size': gt_coords.shape[0],
                    'atom_count_mismatch': atom_count_mismatch
                }
                
                # Append to CSV file immediately
                result_df = pd.DataFrame([result_row])
                result_df.to_csv(csv_path, mode='a', header=False, index=False)
                
                # Save molecular coordinates and types(NOTE: sdf/xyz/mol?)
                # Save molecular coordinates and types(NOTE: sdf/xyz/mol?)
                mol_file = mol_save_dir / f"sample_{sample_idx:04d}_{field_method}.npz"
                np.savez(mol_file, 
                        coords=recon_coords[0].cpu().numpy(),
                        types=gt_types.cpu().numpy(),
                        gt_coords=gt_coords.cpu().numpy(),
                        rmsd=rmsd)
                
            except Exception as e:
                print(f"Error processing sample {sample_idx} with {field_method}: {e}")
                # Save error result as well
                error_row = {
                    'sample_idx': sample_idx,
                    'field_mode': field_mode,
                    'field_method': field_method,
                    'rmsd': float('inf'),
                    'size': gt_coords.shape[0],
                    'atom_count_mismatch': True
                }
                error_df = pd.DataFrame([error_row])
                error_df.to_csv(csv_path, mode='a', header=False, index=False)
    
    # Load and analyze results from CSV file
    print(f"Results saved to: {csv_path}")
    
    # Read results from CSV for summary
    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} results from CSV file")
        
        # Print summary
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