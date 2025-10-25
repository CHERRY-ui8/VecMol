import os
from funcmol.models.funcmol import create_funcmol
import hydra
import torch
import traceback
from funcmol.utils.utils_fm import load_checkpoint_fm
from funcmol.dataset.dataset_field import create_gnf_converter
from funcmol.utils.utils_nf import load_neural_field
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
from lightning import Fabric


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


@hydra.main(config_path="configs", config_name="sample_fm", version_base=None)
def main(config: DictConfig) -> None:
    # 设置PyTorch内存优化环境变量
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:128'
    
    # 转换配置为字典格式
    config_dict = OmegaConf.to_container(config, resolve=True)
    
    # 设置输出目录
    exps_root = Path(__file__).parent.parent / "exps"
    fm_path = config.get("fm_pretrained_path")
    
    # 如果fm_pretrained_path指向.ckpt文件，需要找到对应的目录
    if fm_path.endswith('.ckpt'):
        # 从.ckpt文件路径中提取目录结构
        fm_path_parts = Path(fm_path).parts
        try:
            funcmol_idx = fm_path_parts.index('funcmol')
            if funcmol_idx + 2 < len(fm_path_parts):
                exp_name = f"{fm_path_parts[funcmol_idx + 1]}/{fm_path_parts[funcmol_idx + 2]}"
            else:
                exp_name = fm_path_parts[funcmol_idx + 1]
        except ValueError:
            # 如果找不到funcmol，使用父目录的名称
            exp_name = Path(fm_path).parent.parent.name
    else:
        # 如果fm_pretrained_path指向目录，直接使用
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
    
    # 获取生成参数
    max_samples = config.get('max_samples', 10)  # 默认生成10个分子
    field_methods = config.get('field_methods', ['tanh'])  # 默认使用tanh方法
    
    print(f"Generating {max_samples} molecules with field_methods: {field_methods}")
    print(f"Output directory: {output_dir}")
    
    # 配置Fabric，只支持单GPU模式
    fabric = Fabric(
        accelerator="gpu",
        devices=1,
        precision="bf16-mixed"
    )
    print("Using single-GPU mode")
    
    fabric.launch()
    
    # 加载FuncMol denoiser
    with torch.no_grad():
        funcmol = create_funcmol(config, fabric)
        funcmol, code_stats = load_checkpoint_fm(funcmol, config["fm_pretrained_path"], fabric=fabric)
        funcmol = funcmol.cuda()
        funcmol.eval()
        print("Loaded FuncMol denoiser")
        
        # 加载neural field decoder
        if config.get("nf_pretrained_path") is None:
            raise ValueError("nf_pretrained_path must be specified for denoiser mode")
        encoder, decoder = load_neural_field(config["nf_pretrained_path"], fabric, config)
        decoder = decoder.cuda()
        decoder.eval()
        decoder.set_code_stats(code_stats)
        print("Loaded neural field decoder")
    
    # 初始化CSV文件
    csv_path = output_dir / "denoiser_evaluation_results.csv"
    elements = config.dset.elements  # ["C", "H", "O", "N", "F"]
    csv_columns = ['generated_idx', 'field_method', 'diffusion_method', 'size']
    
    # 添加生成分子每种元素的原子数量
    for element in elements:
        csv_columns.append(f'generated_{element}_count')
    
    # 创建CSV文件（如果不存在）
    if not csv_path.exists():
        results_df = pd.DataFrame(columns=csv_columns)
        results_df.to_csv(csv_path, index=False)
    
    # 创建分子数据保存目录
    mol_save_dir = output_dir / "molecule"
    mol_save_dir.mkdir(parents=True, exist_ok=True)
    
    # 检查使用的扩散方法
    diffusion_method = config.get('diffusion_method', 'old')
    print(f"Using diffusion method: {diffusion_method}")
    
    # 生成分子
    for sample_idx in tqdm(range(max_samples), desc="Generating molecules"):
        for field_method in field_methods:
            try:
                # 创建converter with specific field method
                method_config = config_dict.copy()
                method_config['converter']['gradient_field_method'] = field_method
                converter = create_gnf_converter(method_config)
                
                # 生成codes
                grid_size = config.get('dset', {}).get('grid_size', 9)
                code_dim = config.get('encoder', {}).get('code_dim', 128)
                batch_size = 1
                
                if diffusion_method == "new":
                    # DDPM采样
                    print(f"Using DDPM sampling for sample {sample_idx}")
                    shape = (batch_size, grid_size**3, code_dim)
                    with torch.no_grad():
                        denoised_codes_3d = funcmol.sample_ddpm(shape, progress=False)
                    # 重塑为2D格式以保持兼容性
                    denoised_codes = denoised_codes_3d.view(batch_size, grid_size**3, code_dim)
                else:
                    # 原有方法：随机噪声 + denoiser
                    print(f"Using original sampling for sample {sample_idx}")
                    noise_codes = torch.randn(batch_size, grid_size**3, code_dim, device=next(funcmol.parameters()).device)
                    with torch.no_grad():
                        denoised_codes = funcmol(noise_codes)
                
                # 重建分子
                recon_coords, recon_types = converter.gnf2mol(
                    decoder,
                    denoised_codes,
                    fabric=fabric
                )
                
                # 处理结果
                recon_coords_device = recon_coords[0].cpu()
                generated_size = recon_coords[0].shape[0]
                
                # 获取下一个可用的生成索引
                generated_idx = get_next_generated_idx(csv_path)
                
                # 保存结果到CSV
                result_row = {
                    'generated_idx': generated_idx,
                    'field_method': field_method,
                    'diffusion_method': diffusion_method,
                    'size': generated_size
                }
                
                # 计算生成分子的原子统计
                recon_types_device = recon_types[0].cpu()
                # 过滤掉填充的原子（值为-1）
                valid_recon_types = recon_types_device[recon_types_device != -1]
                for i, element in enumerate(elements):
                    count = (valid_recon_types == i).sum().item()
                    result_row[f'generated_{element}_count'] = count
                
                # 追加到CSV文件
                result_df = pd.DataFrame([result_row])
                result_df.to_csv(csv_path, mode='a', header=False, index=False)
                
                # 保存分子坐标和类型
                mol_file = mol_save_dir / f"generated_{generated_idx:04d}_{field_method}.npz"
                np.savez(mol_file, 
                        coords=recon_coords_device.cpu().numpy(),
                        types=recon_types[0].cpu().numpy())
                
            except Exception as e:
                print(f"Error generating molecule {sample_idx} with {field_method}: {e}")
                traceback.print_exc()
                # 保存错误结果
                generated_idx = get_next_generated_idx(csv_path)
                error_row = {
                    'generated_idx': generated_idx,
                    'field_method': field_method,
                    'diffusion_method': diffusion_method,
                    'size': 0
                }
                # 生成分子原子统计设为0
                for element in elements:
                    error_row[f'generated_{element}_count'] = 0
                
                error_df = pd.DataFrame([error_row])
                error_df.to_csv(csv_path, mode='a', header=False, index=False)
    
    # 分析结果
    print(f"Results saved to: {csv_path}")
    
    try:
        df = pd.read_csv(csv_path)
        print(f"Generated {len(df)} molecules")
        
        print(f"\n=== Summary ===")
        for method in field_methods:
            method_df = df[df['field_method'] == method]
            if len(method_df) > 0:
                avg_size = method_df['size'].mean()
                print(f"{method} ({diffusion_method}): average size={avg_size:.1f} atoms, count={len(method_df)}")
        
    except Exception as e:
        print(f"Error reading results from CSV: {e}")
        print("Results were saved incrementally during processing.")

if __name__ == "__main__":
    main()
