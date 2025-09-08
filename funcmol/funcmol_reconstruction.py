#!/usr/bin/env python3
"""
FuncMol dataset reconstruction script.
This script generates molecular codes using FuncMol and reconstructs molecules for the entire dataset using EGNN decoder.
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
from lightning import Fabric
from tqdm import tqdm

# 设置 torch.compile 兼容性
try:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True
except ImportError:
    print("Warning: torch._dynamo not available in this PyTorch version")

## set up environment
project_root = Path(os.getcwd()).parent
sys.path.insert(0, str(project_root))

from funcmol.utils.constants import PADDING_INDEX
from funcmol.utils.gnf_visualizer import (
    load_config_from_exp_dir, create_converter, prepare_data, 
    visualize_1d_gradient_field_comparison, GNFVisualizer
)
from funcmol.models.funcmol import create_funcmol
from funcmol.models.egnn import EGNNVectorField


def create_test_config():
    """Create a test configuration for FuncMol."""
    config = {
        "dset": {
            "grid_size": 9,
            "n_channels": 5,
            "anchor_spacing": 1.5,
            "elements": ['C', 'H', 'O', 'N', 'F'],
            "resolution": 0.25
        },
        "decoder": {
            "code_dim": 128,
            "hidden_dim": 128,
            "n_layers": 4,
            "radius": 3.0,
            "cutoff": 3.0,
            "k_neighbors": 32
        },
        "denoiser": {
            "use_gnn": True,
            "n_hidden_units": 1024,
            "num_blocks": 3,
            "dropout": 0.1,
            "k_neighbors": 8,
            "cutoff": 5.0,
            "radius": 3.0,
            "use_radius_graph": True
        },
        "wjs": {
            "n_chains": 10,
            "repeats_wjs": 1,
            "max_steps_wjs": 100,
            "steps_wjs": 100,
            "delta_wjs": 0.5,
            "friction_wjs": 1.0
        },
        "smooth_sigma": 0.5,
        "gnf_converter": {
            "n_iter": 2000,
            "gradient_field_method": "tanh",
            "sigma": 0.5,
            "eps": 1e-6,
            "min_samples": 5,
            "temperature": 1.0,
            "logsumexp_eps": 1e-6,
            "inverse_square_strength": 1.0,
            "gradient_clip_threshold": 1.0,
            "sigma_ratios": {
                "C": 1.0,
                "H": 1.0,
                "O": 1.0,
                "N": 1.0,
                "F": 1.0
            },
            "default_config": {
                "n_query_points": 1000,
                "step_size": 0.01,
                "sig_sf": 0.5,
                "sig_mag": 0.5
            }
        }
    }
    return config


def create_egnn_decoder(config, device):
    """Create EGNN decoder for molecular reconstruction."""
    print(">> Creating EGNN decoder...")
    
    decoder = EGNNVectorField(
        grid_size=config["dset"]["grid_size"],
        hidden_dim=config["decoder"]["hidden_dim"],
        num_layers=config["decoder"]["n_layers"],
        radius=config["decoder"]["radius"],
        n_atom_types=config["dset"]["n_channels"],
        code_dim=config["decoder"]["code_dim"],
        cutoff=config["decoder"]["cutoff"],
        anchor_spacing=config["dset"]["anchor_spacing"],
        device=device,
        k_neighbors=config["decoder"]["k_neighbors"]
    )
    
    return decoder


def create_field_function(decoder, codes, sample_idx):
    """Create field function for a specific sample."""
    def field_func(points):
        # 确保 points 是正确的形状
        if points.dim() == 2:  # [n_points, 3]
            points = points.unsqueeze(0)  # [1, n_points, 3]
        elif points.dim() == 3:  # [batch, n_points, 3]
            pass
        else:
            raise ValueError(f"Unexpected points shape: {points.shape}")
        
        # 使用decoder预测场
        result = decoder(points, codes[sample_idx:sample_idx+1])
        # 确保返回 [n_points, n_atom_types, 3] 形状
        if result.dim() == 4:  # [batch, n_points, n_atom_types, 3]
            return result[0]  # 取第一个batch
        else:
            return result
    
    return field_func


def reconstruct_dataset_with_funcmol(config, fabric, device, output_dir, n_samples=100, batch_size=10):
    """Reconstruct entire dataset using FuncMol generated codes."""
    print("=== FuncMol Dataset Reconstruction ===")
    
    # Create FuncMol model
    print("\n>> Creating FuncMol model...")
    funcmol = create_funcmol(config, fabric)
    funcmol = funcmol.to(device)
    funcmol.eval()
    
    # Create EGNN decoder
    decoder = create_egnn_decoder(config, device)
    decoder = decoder.to(device)
    decoder.eval()
    
    # Create converter for reconstruction
    converter = create_converter(config, device)
    
    # Create visualizer
    visualizer = GNFVisualizer(output_dir)
    
    # Generate codes using FuncMol
    print(f"\n>> Generating {n_samples} codes with FuncMol...")
    with torch.no_grad():
        codes = funcmol.sample(
            config=config,
            fabric=fabric,
            delete_net=False
        )
    
    print(f"Generated codes shape: {codes.shape}")
    print(f"Codes dtype: {codes.dtype}")
    print(f"Codes device: {codes.device}")
    print(f"Codes min: {codes.min().item():.6f}, max: {codes.max().item():.6f}")
    
    # Save generated codes
    codes_path = os.path.join(output_dir, "generated_codes.pt")
    torch.save(codes, codes_path)
    print(f"\n>> Generated codes saved to: {codes_path}")
    
    # Process each generated sample
    print(f"\n>> Processing {n_samples} generated samples...")
    
    all_results = []
    atom_types = [0, 1, 2, 3, 4]  # C, H, O, N, F
    
    for sample_idx in tqdm(range(n_samples), desc="Processing samples"):
        try:
            # Create field function for this sample
            field_func = create_field_function(decoder, codes.to(device), sample_idx)
            
            # Create mock ground truth data for this sample
            # In a real scenario, you would load actual ground truth data
            n_atoms = np.random.randint(5, 20)  # Random number of atoms
            gt_coords = torch.randn(1, n_atoms, 3, device=device)
            gt_types = torch.randint(0, config["dset"]["n_channels"], (1, n_atoms), device=device)
            
            # 1D gradient field visualization
            save_path = os.path.join(output_dir, f"field1d_sample_{sample_idx}")
            
            try:
                gradient_results = visualize_1d_gradient_field_comparison(
                    gt_coords=gt_coords,
                    gt_types=gt_types,
                    converter=converter,
                    field_func=field_func,
                    sample_idx=0,  # Use 0 since we only have one mock sample
                    atom_types=atom_types,
                    x_range=None,
                    y_coord=0.0,
                    z_coord=0.0,
                    save_path=save_path,
                )
            except Exception as e:
                print(f"Error in gradient field visualization for sample {sample_idx}: {e}")
                gradient_results = None
            
            # Reconstruction visualization
            try:
                results = visualizer.create_reconstruction_animation(
                    gt_coords=gt_coords,
                    gt_types=gt_types,
                    converter=converter,
                    field_func=field_func,
                    save_interval=100,
                    animation_name=f"funcmol_gen_sample_{sample_idx}",
                    sample_idx=0  # Use 0 since we only have one mock sample
                )
                
                # Store results
                sample_result = {
                    "sample_idx": sample_idx,
                    "n_atoms": n_atoms,
                    "rmsd": results['final_rmsd'],
                    "reconstruction_loss": results['final_loss'],
                    "kl_1to2": results['final_kl_1to2'],
                    "kl_2to1": results['final_kl_2to1'],
                    "gif_path": results['gif_path'],
                    "comparison_path": results['comparison_path'],
                    "gradient_results": gradient_results
                }
                all_results.append(sample_result)
                
                if sample_idx % 10 == 0:  # Print progress every 10 samples
                    print(f"Sample {sample_idx}: RMSD={results['final_rmsd']:.4f}, Loss={results['final_loss']:.4f}")
                    
            except Exception as e:
                print(f"Error in reconstruction for sample {sample_idx}: {e}")
                continue
                
        except Exception as e:
            print(f"Error processing sample {sample_idx}: {e}")
            continue
    
    # Save all results
    results_path = os.path.join(output_dir, "all_reconstruction_results.pt")
    torch.save(all_results, results_path)
    print(f"\n>> All results saved to: {results_path}")
    
    # Print summary statistics
    if all_results:
        rmsds = [r['rmsd'] for r in all_results]
        losses = [r['reconstruction_loss'] for r in all_results]
        
        print(f"\n=== Reconstruction Summary ===")
        print(f"Successfully processed: {len(all_results)}/{n_samples} samples")
        print(f"Average RMSD: {np.mean(rmsds):.4f} ± {np.std(rmsds):.4f}")
        print(f"Average Loss: {np.mean(losses):.4f} ± {np.std(losses):.4f}")
        print(f"Min RMSD: {np.min(rmsds):.4f}")
        print(f"Max RMSD: {np.max(rmsds):.4f}")
    
    return all_results


def main():
    """Main function for FuncMol dataset reconstruction."""
    print("=== FuncMol Dataset Reconstruction ===")
    
    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_samples = 50  # Number of samples to generate and reconstruct
    batch_size = 10  # Batch size for processing
    
    print(f"Device: {device}")
    print(f"Number of samples: {n_samples}")
    print(f"Batch size: {batch_size}")
    
    # Setup Fabric
    fabric = Fabric(
        accelerator="auto",
        devices=1,
        precision="32-true",
        strategy="auto"
    )
    fabric.launch()
    
    # Create configuration
    config = create_test_config()
    print(f"Configuration created")
    
    # Create output directory
    output_dir = "funcmol_dataset_reconstruction"
    os.makedirs(output_dir, exist_ok=True)
    
    # Run dataset reconstruction
    results = reconstruct_dataset_with_funcmol(
        config=config,
        fabric=fabric,
        device=device,
        output_dir=output_dir,
        n_samples=n_samples,
        batch_size=batch_size
    )
    
    print(f"\n=== Dataset reconstruction completed! ===")
    print(f"Results saved to: {output_dir}")
    print(f"Total samples processed: {len(results) if results else 0}")


if __name__ == "__main__":
    main()
