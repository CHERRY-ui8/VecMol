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


def load_vecmol_config(config_name: str, nf_config: OmegaConf) -> Dict[str, Any]:
    """
    从YAML配置文件加载VecMol配置
    
    Args:
        config_name: 配置文件名（如 "train_fm_qm9"）
        nf_config: Neural Field配置
    
    Returns:
        dict: VecMol配置字典
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
    vecmol: Optional[torch.nn.Module] = None,
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
        vecmol: VecMol模型（denoiser模式需要）
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
        if vecmol is None or decoder is None or config is None:
            raise ValueError("denoiser mode requires vecmol, decoder, and config")
        
        # 确保 points 在正确的设备上
        target_device = next(decoder.parameters()).device
        if points.device != target_device:
            points = points.to(target_device)
        
        # 生成随机噪声代码
        grid_size = config.dset.grid_size
        code_dim = config.encoder.code_dim
        batch_size = 1
        
        # 创建随机噪声代码
        noise_codes = torch.randn(batch_size, grid_size**3, code_dim, device=target_device)
        
        # 通过 denoiser 生成分子代码
        with torch.no_grad():
            denoised_codes = vecmol(noise_codes)
        
        # 使用 decoder 生成场
        result = decoder(points, denoised_codes[0:1])
        if result.dim() == 4:  # [batch, n_points, n_atom_types, 3]
            return result[0]  # 取第一个batch
        else:
            return result
    
    elif mode == 'ddpm':
        if decoder is None or codes is None:
            raise ValueError("ddpm mode requires decoder and codes")
        
        # 确保 points 和 codes 在同一个设备上
        target_device = codes.device
        if points.device != target_device:
            points = points.to(target_device)
        
        result = decoder(points, codes[0:1])
        return result[0] if result.dim() == 4 else result
    
    elif mode == 'predicted':
        if decoder is None or codes is None:
            raise ValueError("predicted mode requires decoder and codes")
        
        # 确保 points 和 codes 在同一个设备上
        target_device = codes.device
        if points.device != target_device:
            points = points.to(target_device)
        
        result = decoder(points, codes[0:1])
        return result[0] if result.dim() == 4 else result
    
    elif mode == 'gt':
        if converter is None or gt_coords is None or gt_types is None:
            raise ValueError("gt mode requires converter, gt_coords, and gt_types")
        
        # 过滤掉padding原子
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
    创建场计算函数的工厂函数。
    不得不创建这样一个函数，因为在做梯度上升的时候，需要实时传入更新后的points对应的field，不能只简单传入固定的field
    
    Args:
        mode: 计算模式
        其他参数: 与compute_field_at_points相同的参数
    
    Returns:
        Callable: 场计算函数，带有 device 属性
    """
    # 确定 device
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
    
    # 将 device 附加到函数对象上
    field_func.device = device
    
    return field_func


# ============================================================================
# GPU管理和多进程工具函数
# ============================================================================

def setup_worker_gpu_environment():
    """
    在worker子进程启动时设置GPU环境变量。
    在spawn模式下，子进程会重新执行脚本，通过环境变量_WORKER_GPU_ID设置CUDA_VISIBLE_DEVICES。
    
    注意：这个函数应该在脚本的最开始（导入torch之前）调用。
    """
    if "_WORKER_GPU_ID" in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["_WORKER_GPU_ID"]
        del os.environ["_WORKER_GPU_ID"]  # 清理临时环境变量


def parse_gpu_list_from_config(config, default_use_all: bool = True) -> Tuple[List[int], Optional[str]]:
    """
    从配置中解析GPU列表（优先级：配置文件 > 环境变量 > 所有GPU）
    
    Args:
        config: 配置对象（DictConfig或dict）
        default_use_all: 如果没有找到配置，是否使用所有GPU（默认True）
    
    Returns:
        Tuple[List[int], Optional[str]]: (物理GPU ID列表, CUDA_VISIBLE_DEVICES字符串)
            - 物理GPU ID列表：如 [0, 1, 2, 3]
            - CUDA_VISIBLE_DEVICES字符串：如 "0,1,2,3"，如果为None则表示使用所有GPU
    """
    num_gpus = torch.cuda.device_count()
    
    # 获取要使用的GPU列表（优先级：配置文件 > 环境变量 > 所有GPU）
    gpu_list_config = config.get('gpu_list', None)
    main_cuda_visible = None
    
    if gpu_list_config is not None:
        # 从配置文件读取GPU列表
        # 处理 Hydra 的 ListConfig 类型
        if isinstance(gpu_list_config, ListConfig):
            main_gpu_list = [int(gpu) for gpu in OmegaConf.to_container(gpu_list_config, resolve=True)]
        elif isinstance(gpu_list_config, (list, tuple)):
            main_gpu_list = [int(gpu) for gpu in gpu_list_config]
        elif isinstance(gpu_list_config, str):
            main_gpu_list = [int(x.strip()) for x in gpu_list_config.split(",")]
        else:
            main_gpu_list = [int(gpu_list_config)]
        
        # 验证GPU ID是否有效
        for gpu_id in main_gpu_list:
            if gpu_id < 0 or gpu_id >= num_gpus:
                raise ValueError(f"Invalid GPU ID {gpu_id}. Available GPUs: 0-{num_gpus-1}")
        
        # 设置环境变量，确保子进程也使用这些GPU
        main_cuda_visible = ",".join(str(gpu) for gpu in main_gpu_list)
        os.environ["CUDA_VISIBLE_DEVICES"] = main_cuda_visible
        print(f"Using GPUs from config: {main_gpu_list}")
    else:
        # 获取主进程的CUDA_VISIBLE_DEVICES设置
        main_cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)
        if main_cuda_visible:
            # 解析主进程可见的GPU列表
            main_gpu_list = [int(x.strip()) for x in main_cuda_visible.split(",")]
            print(f"Using GPUs from CUDA_VISIBLE_DEVICES: {main_cuda_visible}, physical GPUs: {main_gpu_list}")
        else:
            # 如果没有设置，根据default_use_all决定
            if default_use_all:
                main_gpu_list = list(range(num_gpus))
                print(f"No GPU configuration found, using all {num_gpus} GPUs: {main_gpu_list}")
            else:
                main_gpu_list = [0]  # 默认只使用第一个GPU
                main_cuda_visible = "0"
                os.environ["CUDA_VISIBLE_DEVICES"] = main_cuda_visible
                print(f"No GPU configuration found, using default GPU 0")
    
    return main_gpu_list, main_cuda_visible


def setup_multiprocessing_spawn():
    """
    设置multiprocessing的start method为'spawn'（CUDA不支持fork模式）
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
    创建worker进程，并通过环境变量设置GPU。
    
    Args:
        worker_func: Worker进程函数
        logical_gpu_id: 逻辑GPU ID（在worker进程中的GPU ID，通常是0）
        physical_gpu_id: 物理GPU ID（在主进程中的实际GPU ID）
        args: 传递给worker_func的参数
        env_var_name: 环境变量名称（默认"_WORKER_GPU_ID"）
    
    Returns:
        multiprocessing.Process: 已启动的进程对象
    """
    # 通过环境变量传递GPU ID，这样子进程在重新执行脚本时会先设置CUDA_VISIBLE_DEVICES
    original_env = os.environ.get(env_var_name, None)
    os.environ[env_var_name] = str(physical_gpu_id)
    try:
        p = multiprocessing.Process(target=worker_func, args=args)
        p.start()
        return p
    finally:
        # 恢复原始环境变量（如果存在）
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
        logical_gpu_id = idx % num_gpus  # 逻辑GPU ID（0到num_gpus-1）
        if logical_gpu_id not in gpu_assignments:
            gpu_assignments[logical_gpu_id] = []
        gpu_assignments[logical_gpu_id].append(task_idx)
    return gpu_assignments