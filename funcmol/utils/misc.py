"""
Miscellaneous utility functions for configuration loading and other common tasks.
"""

import os
from pathlib import Path
from typing import Dict, Any
import hydra
from omegaconf import OmegaConf


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
