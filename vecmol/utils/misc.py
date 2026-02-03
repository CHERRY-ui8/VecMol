"""
Miscellaneous utility functions for configuration loading and other common tasks.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, Callable, List, Tuple
import hydra
from omegaconf import OmegaConf, ListConfig
import torch
import multiprocessing


# Project root
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(PROJECT_ROOT))


def load_nf_config(config_name: str = "train_nf_qm9") -> OmegaConf:
    """Load Neural Field config file.

    Args:
        config_name: Config file name (default "train_nf_qm9")

    Returns:
        OmegaConf config object
    """
    config_path = PROJECT_ROOT / "configs" / f"{config_name}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with hydra.initialize_config_dir(config_dir=str(config_path.parent), version_base=None):
        config = hydra.compose(config_name=config_name)
    
    config["dset"]["data_dir"] = str(PROJECT_ROOT / "dataset" / "data")
    print(f"Dataset directory: {config['dset']['data_dir']}")
    
    if "converter" in config:
        print(f"Config loaded successfully: {config_name}")
        if "n_iter" in config["converter"]:
            print(f"n_iter from converter config: {config['converter']['n_iter']}")
        else:
            print("WARNING: n_iter not found in converter config!")
    else:
        print("WARNING: converter config not found!")
    
    return config


def load_vecmol_config(config_name: str, nf_config: OmegaConf) -> Dict[str, Any]:
    """Load VecMol config from YAML.
    
    Args:
        config_name: Config file name (e.g. "train_fm_qm9")
        nf_config: Neural Field config
    
    Returns:
        VecMol config dict
    """
    print(f"Loading configuration from YAML: {config_name}")
    with hydra.initialize_config_dir(config_dir=str(PROJECT_ROOT / "configs"), version_base=None):
        config = hydra.compose(config_name=config_name)
    if not isinstance(config, dict):
        config = OmegaConf.to_container(config, resolve=True)
    # encoder, decoder, dset from nf_config
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
    vecmol: Optional[torch.nn.Module] = None,
    converter: Optional[Any] = None,
    gt_coords: Optional[torch.Tensor] = None,
    gt_types: Optional[torch.Tensor] = None,
    config: Optional[Any] = None,
) -> torch.Tensor:
    """
    Unified field computation at points by mode.
    
    Args:
        points: Point coords [n_points, 3] or [batch, n_points, 3]
        mode: 'denoiser' | 'ddpm' | 'predicted' | 'gt'
        decoder: Decoder (denoiser, ddpm, predicted)
        codes: Precomputed codes (ddpm, predicted)
        vecmol: VecMol (denoiser)
        converter: GNF converter (gt)
        gt_coords, gt_types: Ground truth (gt)
        config: Config (denoiser)
    
    Returns:
        Field values [n_points, n_atom_types, 3]
    """
    # Ensure points shape
    if points.dim() == 2:  # [n_points, 3]
        points = points.unsqueeze(0)  # [1, n_points, 3]
    elif points.dim() == 3:  # [batch, n_points, 3]
        pass
    else:
        raise ValueError(f"Unexpected points shape: {points.shape}")
    
    if mode == 'denoiser':
        if vecmol is None or decoder is None or config is None:
            raise ValueError("denoiser mode requires vecmol, decoder, and config")
        
        # Ensure points on correct device
        target_device = next(decoder.parameters()).device
        if points.device != target_device:
            points = points.to(target_device)
        
        # 生成随机噪声代码
        grid_size = config.dset.grid_size
        code_dim = config.encoder.code_dim
        batch_size = 1
        
        # Create random noise codes
        noise_codes = torch.randn(batch_size, grid_size**3, code_dim, device=target_device)
        
        # Generate codes via denoiser
        with torch.no_grad():
            denoised_codes = vecmol(noise_codes)
        
        # Use decoder to compute field
        result = decoder(points, denoised_codes[0:1])
        if result.dim() == 4:  # [batch, n_points, n_atom_types, 3]
            return result[0]  # First batch
        else:
            return result
    
    elif mode == 'ddpm':
        if decoder is None or codes is None:
            raise ValueError("ddpm mode requires decoder and codes")
        
        # Ensure points and codes on same device
        target_device = codes.device
        if points.device != target_device:
            points = points.to(target_device)
        
        result = decoder(points, codes[0:1])
        return result[0] if result.dim() == 4 else result
    
    elif mode == 'predicted':
        if decoder is None or codes is None:
            raise ValueError("predicted mode requires decoder and codes")
        
        # Ensure points and codes on same device
        target_device = codes.device
        if points.device != target_device:
            points = points.to(target_device)
        
        result = decoder(points, codes[0:1])
        return result[0] if result.dim() == 4 else result
    
    elif mode == 'gt':
        if converter is None or gt_coords is None or gt_types is None:
            raise ValueError("gt mode requires converter, gt_coords, and gt_types")
        
        # Filter padding atoms
        from vecmol.utils.constants import PADDING_INDEX
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
    vecmol: Optional[torch.nn.Module] = None,
    converter: Optional[Any] = None,
    gt_coords: Optional[torch.Tensor] = None,
    gt_types: Optional[torch.Tensor] = None,
    config: Optional[Any] = None
) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Factory for field computation function (needed for gradient ascent: field must be computed from updated points).

    Args:
        mode: Computation mode
        Other args: Same as compute_field_at_points

    Returns:
        Callable with device attribute
    """
    # Determine device
    device = None
    if mode in ['ddpm', 'predicted']:
        if codes is not None:
            device = codes.device
        elif decoder is not None:
            device = next(decoder.parameters()).device
    elif mode == 'denoiser':
        if decoder is not None:
            device = next(decoder.parameters()).device
        elif vecmol is not None:
            device = next(vecmol.parameters()).device
    elif mode == 'gt':
        if gt_coords is not None:
            device = gt_coords.device
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def field_func(points: torch.Tensor) -> torch.Tensor:
        return compute_field_at_points(
            points=points,
            mode=mode,
            decoder=decoder,
            codes=codes,
            vecmol=vecmol,
            converter=converter,
            gt_coords=gt_coords,
            gt_types=gt_types,
            config=config
        )
    
    # Attach device to function
    field_func.device = device
    
    return field_func


# ============================================================================
# GPU and multiprocessing utilities
# ============================================================================

def setup_worker_gpu_environment():
    """
    Set GPU env vars when worker subprocess starts. In spawn mode, subprocess re-executes script; use _WORKER_GPU_ID for CUDA_VISIBLE_DEVICES. Call at script start (before importing torch).
    """
    if "_WORKER_GPU_ID" in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["_WORKER_GPU_ID"]
        del os.environ["_WORKER_GPU_ID"]  # Clear temp env var


def parse_gpu_list_from_config(config, default_use_all: bool = True) -> Tuple[List[int], Optional[str]]:
    """
    从配置中解析GPU列表（优先级：配置文件 > 环境变量 > 所有GPU）
    
    Args:
        config: 配置对象（DictConfig或dict）
        default_use_all: 如果没有找到配置，是否使用所有GPU（默认True）
    
    Returns:
        (physical GPU ID list, CUDA_VISIBLE_DEVICES string or None)
    """
    num_gpus = torch.cuda.device_count()
    
    # GPU list: config > env > all
    gpu_list_config = config.get('gpu_list', None)
    main_cuda_visible = None
    
    if gpu_list_config is not None:
        # Read GPU list from config; handle Hydra ListConfig
        if isinstance(gpu_list_config, ListConfig):
            main_gpu_list = [int(gpu) for gpu in OmegaConf.to_container(gpu_list_config, resolve=True)]
        elif isinstance(gpu_list_config, (list, tuple)):
            main_gpu_list = [int(gpu) for gpu in gpu_list_config]
        elif isinstance(gpu_list_config, str):
            main_gpu_list = [int(x.strip()) for x in gpu_list_config.split(",")]
        else:
            main_gpu_list = [int(gpu_list_config)]
        
        # Validate GPU IDs
        for gpu_id in main_gpu_list:
            if gpu_id < 0 or gpu_id >= num_gpus:
                raise ValueError(f"Invalid GPU ID {gpu_id}. Available GPUs: 0-{num_gpus-1}")
        
        # Set env so subprocesses use these GPUs
        main_cuda_visible = ",".join(str(gpu) for gpu in main_gpu_list)
        os.environ["CUDA_VISIBLE_DEVICES"] = main_cuda_visible
        print(f"Using GPUs from config: {main_gpu_list}")
    else:
        # Get main process CUDA_VISIBLE_DEVICES
        main_cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)
        if main_cuda_visible:
            # Parse main process visible GPU list
            main_gpu_list = [int(x.strip()) for x in main_cuda_visible.split(",")]
            print(f"Using GPUs from CUDA_VISIBLE_DEVICES: {main_cuda_visible}, physical GPUs: {main_gpu_list}")
        else:
            # If not set, use default_use_all
            if default_use_all:
                main_gpu_list = list(range(num_gpus))
                print(f"No GPU configuration found, using all {num_gpus} GPUs: {main_gpu_list}")
            else:
                main_gpu_list = [0]  # Default: first GPU only
                main_cuda_visible = "0"
                os.environ["CUDA_VISIBLE_DEVICES"] = main_cuda_visible
                print(f"No GPU configuration found, using default GPU 0")
    
    return main_gpu_list, main_cuda_visible


def setup_multiprocessing_spawn():
    """
    Set multiprocessing start method to 'spawn' (CUDA does not support fork)
    """
    if multiprocessing.get_start_method(allow_none=True) != 'spawn':
        multiprocessing.set_start_method('spawn', force=True)


def create_worker_process_with_gpu(
    worker_func,
    logical_gpu_id: int,
    physical_gpu_id: int,
    args: tuple,
    env_var_name: str = "_WORKER_GPU_ID"
) -> multiprocessing.Process:
    """
    Create worker process and set GPU via env var.

    Args:
        worker_func: Worker function
        logical_gpu_id: Logical GPU ID in worker (usually 0)
        physical_gpu_id: Physical GPU ID in main process
        args: Args for worker_func
        env_var_name: Env var name (default _WORKER_GPU_ID)

    Returns:
        Started Process
    """
    # Pass GPU ID via env so subprocess sets CUDA_VISIBLE_DEVICES first
    original_env = os.environ.get(env_var_name, None)
    os.environ[env_var_name] = str(physical_gpu_id)
    try:
        p = multiprocessing.Process(target=worker_func, args=args)
        p.start()
        return p
    finally:
        # Restore original env var if present
        if original_env is not None:
            os.environ[env_var_name] = original_env
        elif env_var_name in os.environ:
            del os.environ[env_var_name]


def distribute_tasks_to_gpus(
    task_indices: List[int],
    num_gpus: int
) -> Dict[int, List[int]]:
    """
    将任务索引分配给不同的GPU（轮询分配）
    
    Args:
        task_indices: 任务索引列表
        num_gpus: GPU数量
    
    Returns:
        Dict[int, List[int]]: {逻辑GPU ID: [任务索引列表]}
    """
    gpu_assignments = {}
    for idx, task_idx in enumerate(task_indices):
        logical_gpu_id = idx % num_gpus  # Logical GPU ID (0 to num_gpus-1)
        if logical_gpu_id not in gpu_assignments:
            gpu_assignments[logical_gpu_id] = []
        gpu_assignments[logical_gpu_id].append(task_idx)
    return gpu_assignments