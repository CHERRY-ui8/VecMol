import os
import sys
import torch
import numpy as np
from pathlib import Path
# from lightning import Fabric  # Not needed for single GPU version
from tqdm import tqdm
import hydra
import yaml
from omegaconf import OmegaConf

# 设置 torch.compile 兼容性
try:
    import torch._dynamo  # pylint: disable=protected-access
    torch._dynamo.config.suppress_errors = True
except ImportError:
    print("Warning: torch._dynamo not available in this PyTorch version")

## set up environment
project_root = Path(os.getcwd()).parent
sys.path.insert(0, str(project_root))

from funcmol.utils.gnf_visualizer import create_converter, GNFVisualizer
from funcmol.models.funcmol import create_funcmol
from funcmol.models.decoder import Decoder


def load_generation_config(config_name: str = "qm9"):
    """Load molecular generation configuration from generation config files.
    
    Args:
        config_name: Name of the generation config (qm9, cremp, drugs)
        
    Returns:
        OmegaConf configuration object
    """
    configs_dir = project_root / "funcmol" / "configs"
    generation_config_path = configs_dir / "generation" / f"{config_name}.yaml"
    
    if not generation_config_path.exists():
        raise FileNotFoundError(f"Generation config file not found: {generation_config_path}")
    
    # 直接加载 generation 配置文件
    with open(generation_config_path, 'r', encoding='utf-8') as f:
        generation_config = yaml.safe_load(f)
    
    # 使用 Hydra 加载基础配置（训练配置）
    config_mapping = {
        "qm9": "train_fm_qm9",
        "cremp": "train_fm_cremp", 
        "drugs": "train_fm_drugs"
    }
    
    training_config_name = config_mapping[config_name]
    
    with hydra.initialize_config_dir(config_dir=str(configs_dir), version_base=None):
        config = hydra.compose(config_name=training_config_name)
    
    # 设置数据目录路径
    if "dset" in config:
        config["dset"]["data_dir"] = str(project_root / "funcmol" / "dataset" / "data")
    
    # 添加生成相关的参数（从 generation 配置文件中读取）
    config["smooth_sigma"] = generation_config.get("smooth_sigma", 0.5)
    # 不在这里添加 n_samples，让 main 函数处理
        
    print(f"Generation configuration loaded: {config_name}")
    print(f"Dataset: {config['dset']['dset_name']}")
    print(f"Grid size: {config['dset']['grid_size']}")
    print(f"Elements: {config['dset']['elements']}")
    print(f"Number of samples: {config.get('n_samples', 'Not specified')}")
    
    return config


def create_test_config():
    """Create a test configuration for FuncMol by loading from generation config files.
    
    This function is kept for backward compatibility but now loads from generation config files
    instead of hardcoding parameters.
    """
    return load_generation_config("qm9")


def create_decoder(config, device):
    """Create decoder for molecular reconstruction."""
    print(">> Creating decoder...")
    
    # 创建decoder配置，从dset配置中获取grid_size和n_channels
    decoder_config = dict(config["decoder"])
    decoder_config["grid_size"] = config["dset"]["grid_size"]
    decoder_config["n_channels"] = config["dset"]["n_channels"]
    decoder_config["device"] = device  # 将device添加到配置中
    
    decoder = Decoder(decoder_config)
    
    return decoder


def create_field_function(decoder, codes, sample_idx):
    """Create field function for a specific sample."""
    def field_func(points):
        # 确保 points 是正确的形状和设备
        if points.dim() == 2:  # [n_points, 3]
            points = points.unsqueeze(0)  # [1, n_points, 3]
        elif points.dim() == 3:  # [batch, n_points, 3]
            pass
        else:
            raise ValueError(f"Unexpected points shape: {points.shape}")
        
        # 确保points在正确的设备上
        points = points.to(codes.device)
        
        # 检查sample_idx是否在有效范围内
        if sample_idx >= codes.size(0):
            raise IndexError(f"sample_idx {sample_idx} out of range for codes with size {codes.size(0)}")
        
        # 使用decoder预测场
        try:
            result = decoder(points, codes[sample_idx:sample_idx+1])
            # 确保返回 [n_points, n_atom_types, 3] 形状
            if result.dim() == 4:  # [batch, n_points, n_atom_types, 3]
                return result[0]  # 取第一个batch
            else:
                return result
        except Exception as e:
            print(f"Error in decoder forward pass for sample {sample_idx}: {e}")
            print(f"  points shape: {points.shape}, device: {points.device}")
            print(f"  codes shape: {codes.shape}, device: {codes.device}")
            print(f"  codes[sample_idx:sample_idx+1] shape: {codes[sample_idx:sample_idx+1].shape}, device: {codes[sample_idx:sample_idx+1].device}")
            raise e
    
    return field_func


def generate_molecules_with_funcmol(config, device, output_dir, n_samples=100):
    """Generate molecules using FuncMol with single GPU."""
    print("=== FuncMol Molecular Generation (Single GPU) ===")
    
    # Create FuncMol model
    print("\n>> Creating FuncMol model...")
    funcmol = create_funcmol(config, None)  # No fabric for single GPU
    funcmol = funcmol.to(device)
    funcmol.eval()
    
    # Load pretrained model and code_stats
    print("\n>> Loading pretrained FuncMol model...")
    try:
        from funcmol.utils.utils_fm import load_checkpoint_fm
        checkpoint_path = os.path.join(config["fm_pretrained_path"], "checkpoint.pth.tar")
        if os.path.exists(checkpoint_path):
            # Create a mock fabric object for loading
            class MockFabric:
                def load(self, path):
                    return torch.load(path, map_location=device)
                def print(self, msg):
                    print(msg)
            
            mock_fabric = MockFabric()
            funcmol, code_stats = load_checkpoint_fm(funcmol, config["fm_pretrained_path"], fabric=mock_fabric)
            print(f">> Loaded code_stats: mean={code_stats['mean']:.6f}, std={code_stats['std']:.6f}")
        else:
            print(f">> Warning: Pretrained model not found at {checkpoint_path}")
            code_stats = None
    except Exception as e:
        print(f">> Warning: Could not load pretrained model: {e}")
        code_stats = None
    
    # Create decoder
    decoder = create_decoder(config, device)
    decoder = decoder.to(device)
    decoder.eval()
    
    # Create converter for reconstruction
    converter = create_converter(config, device)
    
    # Create visualizer
    visualizer = GNFVisualizer(output_dir)
    
    # Generate codes using FuncMol
    print(f"\n>> Generating {n_samples} codes with FuncMol...")
    try:
        with torch.no_grad():
            codes = funcmol.sample(
                config=config,
                fabric=None,  # No fabric for single GPU
                delete_net=False,
                code_stats=code_stats
            )
        
        print(f"Generated codes shape: {codes.shape}")
        print(f"Codes dtype: {codes.dtype}")
        print(f"Codes device: {codes.device}")
        print(f"Codes min: {codes.min().item():.6f}, max: {codes.max().item():.6f}")
        
        # Validate codes shape
        expected_code_dim = config["decoder"]["code_dim"]
        expected_grid_size = config["dset"]["grid_size"]
        expected_n_grid = expected_grid_size ** 3
        
        # FuncMol generates codes with shape [n_samples, n_grid, code_dim]
        if len(codes.shape) != 3:
            raise ValueError(f"Expected 3D codes tensor, got shape {codes.shape}")
        
        if codes.shape[1] != expected_n_grid:
            raise ValueError(f"Grid dimension mismatch: expected {expected_n_grid}, got {codes.shape[1]}")
        
        if codes.shape[2] != expected_code_dim:
            raise ValueError(f"Code dimension mismatch: expected {expected_code_dim}, got {codes.shape[2]}")
        
        if codes.shape[0] != n_samples:
            print(f"Warning: Expected {n_samples} samples, got {codes.shape[0]}")
            # Adjust n_samples to match actual generated codes
            n_samples = codes.shape[0]
        
        # Move codes to the correct device
        codes = codes.to(device)
        print(f"Codes moved to device: {codes.device}")
        
        # Save generated codes
        codes_path = os.path.join(output_dir, "generated_codes.pt")
        torch.save(codes, codes_path)
        print(f"\n>> Generated codes saved to: {codes_path}")
        
    except Exception as e:
        print(f"Error generating codes: {e}")
        raise e
    
    # Process all samples
    print(f"\n>> Processing {n_samples} samples...")
    all_results = []
    
    for sample_idx in tqdm(range(n_samples), desc="Processing samples"):
        try:
            print(f"\nProcessing sample {sample_idx}...")
            
            # Validate sample_idx
            if sample_idx >= codes.size(0):
                print(f"Error: sample_idx {sample_idx} >= codes.size(0) {codes.size(0)}")
                continue
            
            # Create field function for this sample
            field_func = create_field_function(decoder, codes.to(device), sample_idx)
            
            # Test field function with a small batch first
            try:
                test_points = torch.rand(10, 3, device=device) * 2 - 1  # Random points in [-1, 1]
                test_output = field_func(test_points)
                print(f"  Field function test successful: input {test_points.shape} -> output {test_output.shape}")
            except (ValueError, RuntimeError, IndexError) as e:
                print(f"  Field function test failed for sample {sample_idx}: {e}")
                continue
            
            # Generation visualization (no ground truth needed)
            try:
                results = visualizer.create_generation_animation(
                    converter=converter,
                    field_func=field_func,
                    sample_idx=sample_idx,
                    save_interval=100,
                    create_1d_plots=True  # 启用1D场可视化
                )
                
                # Store results
                sample_result = {
                    "sample_idx": sample_idx,
                    "convergence": results['final_convergence'],
                    "gif_path": results['gif_path'],
                    "final_path": results['final_path'],
                    "final_points": results['final_points'],
                    "final_types": results['final_types'],
                    "field_1d_results": results.get('field_1d_results', None)
                }
                all_results.append(sample_result)
                
                if sample_idx % 10 == 0:  # Print progress every 10 samples
                    print(f"Sample {sample_idx}: Convergence={results['final_convergence']:.4f}")
                    
            except (ValueError, RuntimeError, IndexError) as e:
                print(f"Error in reconstruction for sample {sample_idx}: {e}")
                import traceback
                traceback.print_exc()
                continue
                
        except (ValueError, RuntimeError, IndexError) as e:
            print(f"Error processing sample {sample_idx}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save all results
    results_path = os.path.join(output_dir, "all_reconstruction_results.pt")
    torch.save(all_results, results_path)
    print(f"\n>> All results saved to: {results_path}")
    
    # Print summary statistics
    if all_results:
        convergences = [r['convergence'] for r in all_results]
        
        print("\n=== Generation Summary ===")
        print(f"Successfully processed: {len(all_results)}/{n_samples} samples")
        print(f"Average Convergence: {np.mean(convergences):.4f} ± {np.std(convergences):.4f}")
        print(f"Min Convergence: {np.min(convergences):.4f}")
        print(f"Max Convergence: {np.max(convergences):.4f}")
    else:
        print("\n=== Generation Summary ===")
        print("No samples were successfully processed!")
    
    return all_results


def main(config_name: str = "qm9", n_samples: int = None):
    """Main function for FuncMol molecular generation.
    
    Args:
        config_name: Name of the generation config (qm9, cremp, drugs)
        n_samples: Number of samples to generate. If None, will use config file value
    """
    print("=== FuncMol Molecular Generation (Single GPU) ===")
    
    # Load generation configuration
    config = load_generation_config(config_name)
    
    # 读取 generation 配置文件中的 n_samples
    configs_dir = Path("configs")
    generation_config_path = configs_dir / "generation" / f"{config_name}.yaml"
    
    with open(generation_config_path, 'r', encoding='utf-8') as f:
        generation_config = yaml.safe_load(f)
    
    # Use n_samples from command line if provided, otherwise from generation config file
    if n_samples is not None:
        print(f"Using n_samples from command line: {n_samples}")
    else:
        n_samples = generation_config.get("n_samples", 50)
        print(f"Using n_samples from generation config: {n_samples}")
    
    # 加载 converter 配置
    converter_config_name = f"gnf_converter_{config_name}"
    with hydra.initialize_config_dir(config_dir=str(configs_dir.absolute()), version_base=None):
        converter_config = hydra.compose(config_name=f"converter/{converter_config_name}")
    
    # 将 converter 配置和 generation 配置添加到主配置中
    config_dict = dict(config)
    config_dict["gnf_converter"] = converter_config["converter"]
    
    # 添加 generation 配置中的参数
    config_dict["fm_pretrained_path"] = generation_config.get("fm_pretrained_path", None)
    config_dict["nf_pretrained_path"] = generation_config.get("nf_pretrained_path", None)
    config_dict["smooth_sigma"] = generation_config.get("smooth_sigma", 0.5)
    
    # 创建新的配置，允许添加新键
    config = OmegaConf.create(config_dict)
    OmegaConf.set_struct(config, False)  # 允许添加新键
    
    # 现在可以安全地添加n_samples
    config["n_samples"] = n_samples
    
    # Setup device (single GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Number of samples: {n_samples}")
    print(f"Config file: {config_name}")
    
    # Create output directory
    output_dir = f"funcmol_generation_{config_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Run molecular generation
    results = generate_molecules_with_funcmol(
        config=config,
        device=device,
        output_dir=output_dir,
        n_samples=n_samples
    )
    
    print("\n=== Molecular generation completed! ===")
    print(f"Results saved to: {output_dir}")
    print(f"Total samples processed: {len(results) if results else 0}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="FuncMol Molecular Generation")
    parser.add_argument("--config", type=str, default="qm9", 
                       help="Generation config name (qm9, cremp, drugs)")
    parser.add_argument("--n_samples", type=int, default=None,
                       help="Number of samples to generate (overrides config file)")
    
    args = parser.parse_args()
    
    main(config_name=args.config, n_samples=args.n_samples)
