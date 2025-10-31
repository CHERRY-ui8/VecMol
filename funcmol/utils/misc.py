"""
Miscellaneous utility functions for configuration loading and other common tasks.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, Callable
import hydra
from omegaconf import OmegaConf
import torch


# 设置项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(PROJECT_ROOT))


def load_nf_config(config_name: str = "train_nf_qm9") -> OmegaConf:
    """加载Neural Field配置文件。

    Args:
        config_name: 配置文件名，默认为 "train_nf_qm9"

    Returns:
        OmegaConf 配置对象
    """
    config_path = PROJECT_ROOT / "configs" / f"{config_name}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with hydra.initialize_config_dir(config_dir=str(config_path.parent), version_base=None):
        config = hydra.compose(config_name=config_name)
    
    config["dset"]["data_dir"] = str(PROJECT_ROOT / "dataset" / "data")
    print(f"Dataset directory: {config['dset']['data_dir']}")
    
    # 验证配置加载
    if "converter" in config:
        print(f"Config loaded successfully: {config_name}")
        if "n_iter" in config["converter"]:
            print(f"n_iter from converter config: {config['converter']['n_iter']}")
        else:
            print("WARNING: n_iter not found in converter config!")
    else:
        print("WARNING: converter config not found!")
    
    return config


def load_funcmol_config(config_name: str, nf_config: OmegaConf) -> Dict[str, Any]:
    """
    从YAML配置文件加载FuncMol配置
    
    Args:
        config_name: 配置文件名（如 "train_fm_qm9"）
        nf_config: Neural Field配置
    
    Returns:
        dict: FuncMol配置字典
    """
    print(f"Loading configuration from YAML: {config_name}")
    
    # 使用Hydra加载配置
    with hydra.initialize_config_dir(config_dir=str(PROJECT_ROOT / "configs"), version_base=None):
        config = hydra.compose(config_name=config_name)
    
    # 转换为普通字典
    if not isinstance(config, dict):
        config = OmegaConf.to_container(config, resolve=True)
    
    # 确保encoder, decoder, dset配置正确
    config["encoder"] = nf_config.encoder
    config["decoder"] = nf_config.decoder  
    config["dset"] = nf_config.dset
    
    print(f">> Using diffusion_method: {config.get('diffusion_method', 'unknown')}")
    print(f">> DDPM config: {config.get('ddpm', {})}")
    
    return config


def compute_field_at_points(
    points: torch.Tensor,
    mode: str,
    decoder: Optional[torch.nn.Module] = None,
    codes: Optional[torch.Tensor] = None,
    funcmol: Optional[torch.nn.Module] = None,
    converter: Optional[Any] = None,
    gt_coords: Optional[torch.Tensor] = None,
    gt_types: Optional[torch.Tensor] = None,
    config: Optional[Any] = None,
) -> torch.Tensor:
    """
    统一的场计算函数，根据模式计算特定点处的场大小。
    
    Args:
        points: 要计算场的点坐标 [n_points, 3] 或 [batch, n_points, 3]
        mode: 计算模式，支持以下选项：
            - 'denoiser': 使用去噪器生成随机噪声代码
            - 'ddpm': 使用预采样的DDPM代码
            - 'predicted': 使用预计算的预测代码
            - 'gt': 使用真实分子坐标计算场
        decoder: 解码器模型（denoiser, ddpm, predicted模式需要）
        codes: 预计算的代码（ddpm, predicted模式需要）
        funcmol: FuncMol模型（denoiser模式需要）
        converter: GNF转换器（gt模式需要）
        gt_coords: 真实分子坐标（gt模式需要）
        gt_types: 真实原子类型（gt模式需要）
        config: 配置对象（denoiser模式需要）
        device: 计算设备
    
    Returns:
        torch.Tensor: 场值 [n_points, n_atom_types, 3]
    """
    # 确保points是正确的形状
    if points.dim() == 2:  # [n_points, 3]
        points = points.unsqueeze(0)  # [1, n_points, 3]
    elif points.dim() == 3:  # [batch, n_points, 3]
        pass
    else:
        raise ValueError(f"Unexpected points shape: {points.shape}")
    
    if mode == 'denoiser':
        if funcmol is None or decoder is None or config is None:
            raise ValueError("denoiser mode requires funcmol, decoder, and config")
        
        # 生成随机噪声代码
        grid_size = config.dset.grid_size
        code_dim = config.encoder.code_dim
        batch_size = 1
        
        # 创建随机噪声代码
        noise_codes = torch.randn(batch_size, grid_size**3, code_dim, device=points.device)
        
        # 通过 denoiser 生成分子代码
        with torch.no_grad():
            denoised_codes = funcmol(noise_codes)
        
        # 使用 decoder 生成场
        result = decoder(points, denoised_codes[0:1])
        if result.dim() == 4:  # [batch, n_points, n_atom_types, 3]
            return result[0]  # 取第一个batch
        else:
            return result
    
    elif mode == 'ddpm':
        if decoder is None or codes is None:
            raise ValueError("ddpm mode requires decoder and codes")
        
        result = decoder(points, codes[0:1])
        return result[0] if result.dim() == 4 else result
    
    elif mode == 'predicted':
        if decoder is None or codes is None:
            raise ValueError("predicted mode requires decoder and codes")
        
        result = decoder(points, codes[0:1])
        return result[0] if result.dim() == 4 else result
    
    elif mode == 'gt':
        if converter is None or gt_coords is None or gt_types is None:
            raise ValueError("gt mode requires converter, gt_coords, and gt_types")
        
        # 过滤掉padding原子
        from funcmol.utils.constants import PADDING_INDEX
        gt_mask = (gt_types[0] != PADDING_INDEX)
        gt_valid_coords = gt_coords[0][gt_mask]
        gt_valid_types = gt_types[0][gt_mask]
        
        result = converter.mol2gnf(
            gt_valid_coords.unsqueeze(0),
            gt_valid_types.unsqueeze(0),
            points
        )
        return result[0] if result.dim() == 4 else result
    
    else:
        raise ValueError(f"Unsupported mode: {mode}. Supported modes: denoiser, ddpm, predicted, gt")


def create_field_function(
    mode: str,
    decoder: Optional[torch.nn.Module] = None,
    codes: Optional[torch.Tensor] = None,
    funcmol: Optional[torch.nn.Module] = None,
    converter: Optional[Any] = None,
    gt_coords: Optional[torch.Tensor] = None,
    gt_types: Optional[torch.Tensor] = None,
    config: Optional[Any] = None
) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    创建场计算函数的工厂函数。
    不得不创建这样一个函数，因为在做梯度上升的时候，需要实时传入更新后的points对应的field，不能只简单传入固定的field
    
    Args:
        mode: 计算模式
        其他参数: 与compute_field_at_points相同的参数
    
    Returns:
        Callable: 场计算函数
    """
    def field_func(points: torch.Tensor) -> torch.Tensor:
        return compute_field_at_points(
            points=points,
            mode=mode,
            decoder=decoder,
            codes=codes,
            funcmol=funcmol,
            converter=converter,
            gt_coords=gt_coords,
            gt_types=gt_types,
            config=config
        )
    
    return field_func