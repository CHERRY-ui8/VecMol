import sys
sys.path.append("..")  # 添加上级目录到sys.path以便导入模块

import os
# 注意：在spawn模式下，子进程会重新执行这个脚本
# 如果环境变量_WORKER_GPU_ID已设置，说明这是worker子进程，需要先设置CUDA_VISIBLE_DEVICES
from funcmol.utils.misc import setup_worker_gpu_environment
setup_worker_gpu_environment()

from funcmol.models.funcmol import create_funcmol
import hydra
import torch
import traceback
from funcmol.utils.utils_fm import load_checkpoint_fm, find_checkpoint_path
from funcmol.dataset.dataset_field import create_gnf_converter
from funcmol.utils.utils_nf import load_neural_field, create_neural_field
from funcmol.utils.utils_base import xyz_to_sdf
from funcmol.utils.misc import (
    parse_gpu_list_from_config,
    setup_multiprocessing_spawn,
    create_worker_process_with_gpu
)
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
import multiprocessing
from multiprocessing import Queue
import time
import queue

# ============================================================================
# GPU Worker进程：长期运行，从Queue取任务批量处理
# 模型加载在worker进程内部完成，不再需要全局变量
# ============================================================================

def gpu_worker_process(
    logical_gpu_id: int,
    physical_gpu_id: int,
    config_dict: dict,
    fm_pretrained_path: str,
    nf_pretrained_path: str,
    task_queue: Queue,
    result_queue: Queue
):
    """
    GPU Worker进程：长期运行，从Queue取任务批量处理
    
    Args:
        logical_gpu_id: 逻辑GPU ID（在worker进程中的GPU ID，通常是0）
        physical_gpu_id: 物理GPU ID（在主进程中的实际GPU ID，用于日志）
        config_dict: 配置字典
        fm_pretrained_path: FuncMol checkpoint路径
        nf_pretrained_path: Neural field checkpoint路径
        task_queue: 任务队列（从主进程接收任务）
        result_queue: 结果队列（向主进程发送结果）
    
    注意：CUDA_VISIBLE_DEVICES已经在脚本开头通过环境变量_WORKER_GPU_ID设置了
    这里torch已经被导入并使用了正确的CUDA_VISIBLE_DEVICES设置
    """
    # 验证GPU设置
    if torch.cuda.is_available():
        num_visible_gpus = torch.cuda.device_count()
        if num_visible_gpus != 1:
            print(f"Warning: Expected 1 GPU after setting CUDA_VISIBLE_DEVICES={physical_gpu_id}, but got {num_visible_gpus}")
    
    # 设置PyTorch内存优化环境变量
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:128'
    
    # 设置随机种子（每个worker独立）
    import random
    worker_id = multiprocessing.current_process().pid
    seed = int(time.time() * 1000) % 2**31 + worker_id
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    print(f"[GPU Worker logical={logical_gpu_id}, physical={physical_gpu_id} (PID {worker_id})] Initializing...")
    
    # 加载模型（每个worker独立加载）
    with torch.no_grad():
        funcmol = create_funcmol(config_dict)
        funcmol, code_stats = load_checkpoint_fm(funcmol, fm_pretrained_path)
        funcmol = funcmol.cuda()
        funcmol.eval()
        
        # 根据配置决定是否从 diffusion checkpoint 加载 decoder
        load_decoder_from_diffusion = config_dict.get("load_decoder_from_diffusion_checkpoint", False)
        decoder = None
        
        if load_decoder_from_diffusion:
            # 尝试从 fm_pretrained_path 的 checkpoint 中加载 decoder_state_dict
            # 如果 joint_finetune 启用了，decoder 应该保存在 diffusion checkpoint 中
            try:
                checkpoint_path = find_checkpoint_path(fm_pretrained_path)
                lightning_checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
                
                if "decoder_state_dict" in lightning_checkpoint:
                    print(f"[GPU Worker logical={logical_gpu_id}, physical={physical_gpu_id}] Found decoder_state_dict in diffusion checkpoint, loading from there...")
                    # 从 checkpoint 中获取 config（优先使用 checkpoint 中的 config，如果没有则使用传入的 config）
                    decoder_config = lightning_checkpoint.get("hyper_parameters", config_dict)
                    if decoder_config is None:
                        decoder_config = config_dict
                    
                    # 创建 decoder 模型
                    _, decoder = create_neural_field(decoder_config)
                    
                    # 加载 decoder state dict
                    decoder_state_dict = lightning_checkpoint["decoder_state_dict"]
                    # 处理 _orig_mod. 前缀（如果存在）
                    new_state_dict = {k.replace("_orig_mod.", ""): v for k, v in decoder_state_dict.items()}
                    decoder.load_state_dict(new_state_dict, strict=True)
                    print(f"[GPU Worker logical={logical_gpu_id}, physical={physical_gpu_id}] Successfully loaded decoder from diffusion checkpoint")
                else:
                    print(f"[GPU Worker logical={logical_gpu_id}, physical={physical_gpu_id}] No decoder_state_dict found in diffusion checkpoint, falling back to neural field checkpoint...")
            except Exception as e:
                print(f"[GPU Worker logical={logical_gpu_id}, physical={physical_gpu_id}] Failed to load decoder from diffusion checkpoint: {e}")
                print(f"[GPU Worker logical={logical_gpu_id}, physical={physical_gpu_id}] Falling back to loading decoder from neural field checkpoint...")
                decoder = None
        else:
            print(f"[GPU Worker logical={logical_gpu_id}, physical={physical_gpu_id}] load_decoder_from_diffusion_checkpoint=false, loading decoder from neural field checkpoint...")
        
        # 如果从 diffusion checkpoint 加载失败或配置为不使用，从 neural field checkpoint 加载
        if decoder is None:
            if nf_pretrained_path is None:
                raise ValueError("nf_pretrained_path must be specified when not loading decoder from diffusion checkpoint")
            _encoder, decoder = load_neural_field(nf_pretrained_path, config_dict)
        
        decoder = decoder.cuda()
        decoder.eval()
        decoder.set_code_stats(code_stats)
    
    print(f"[GPU Worker logical={logical_gpu_id}, physical={physical_gpu_id}] Models loaded successfully")
    
    # Worker主循环：持续从Queue取任务并处理
    processed_count = 0
    while True:
        try:
            # 从队列取任务（timeout避免无限阻塞）
            try:
                task = task_queue.get(timeout=1.0)
            except queue.Empty:
                # 检查是否还有任务（通过特殊标记）
                continue
            
            # 检查终止信号
            if task is None:
                print(f"[GPU Worker logical={logical_gpu_id}, physical={physical_gpu_id}] Received termination signal, exiting...")
                break
            
            # 处理任务（GPU计算部分）
            try:
                result = process_gpu_batch(
                    task,
                    funcmol,
                    decoder,
                    code_stats,
                    config_dict,
                    worker_id=logical_gpu_id
                )
                result_queue.put(result)
                processed_count += 1
            except Exception as e:
                print(f"[GPU Worker logical={logical_gpu_id}, physical={physical_gpu_id}] Error processing task {task.get('task_id', 'unknown')}: {e}")
                traceback.print_exc()
                # 发送错误结果
                result_queue.put({
                    'success': False,
                    'task_id': task.get('task_id', -1),
                    'field_method': task.get('field_method', 'unknown'),
                    'error': str(e),
                    'gpu_id': logical_gpu_id
                })
        
        except KeyboardInterrupt:
            print(f"[GPU Worker logical={logical_gpu_id}, physical={physical_gpu_id}] Interrupted, exiting...")
            break
        except Exception as e:
            print(f"[GPU Worker logical={logical_gpu_id}, physical={physical_gpu_id}] Unexpected error: {e}")
            traceback.print_exc()
    
    print(f"[GPU Worker logical={logical_gpu_id}, physical={physical_gpu_id}] Processed {processed_count} tasks, shutting down...")


def process_gpu_batch(
    task: dict,
    funcmol: torch.nn.Module,
    decoder: torch.nn.Module,
    code_stats: dict,
    config_dict: dict,
    worker_id: int = None
) -> dict:
    """
    GPU Worker内部：处理单个任务的GPU计算部分
    
    只做GPU计算，不涉及IO操作
    
    Args:
        task: 任务字典
        funcmol: FuncMol模型
        decoder: Decoder模型
        code_stats: Code统计信息
        config_dict: 配置字典
        worker_id: Worker ID（用于调试）
    
    Returns:
        dict: 包含GPU计算结果的字典（原始数据，未做IO处理）
        {
            'success': bool,
            'task_id': int,
            'field_method': str,
            'diffusion_method': str,
            'denoised_codes': torch.Tensor (CPU),
            'recon_coords': torch.Tensor (CPU),
            'recon_types': torch.Tensor (CPU),
            'error': str (如果失败)
        }
    """
    try:
        field_method = task['field_method']
        diffusion_method = task['diffusion_method']
        grid_size = task['grid_size']
        code_dim = task['code_dim']
        task_id = task['task_id']
        
        # 1. 创建converter with specific field method
        method_config = config_dict.copy()
        method_config['converter']['gradient_field_method'] = field_method
        converter = create_gnf_converter(method_config)
        
        # 2. 生成codes（单个分子，batch_size=1）
        if diffusion_method == "new" or diffusion_method == "new_x0":
            # DDPM采样
            shape = (1, grid_size**3, code_dim)
            with torch.no_grad():
                denoised_codes_3d = funcmol.sample_ddpm(shape, code_stats=code_stats, progress=False)
            denoised_codes = denoised_codes_3d.view(1, grid_size**3, code_dim)
        else:
            # 原有方法：随机噪声 + denoiser
            noise_codes = torch.randn(1, grid_size**3, code_dim, device=next(funcmol.parameters()).device)
            with torch.no_grad():
                denoised_codes = funcmol(noise_codes)
        
        # 3. 重建分子（GPU计算）
        recon_coords, recon_types = converter.gnf2mol(
            decoder,
            denoised_codes,
            enable_timing=False,
            sample_id=task_id
        )
        
        # 4. 将结果移到CPU（准备传输到主进程）
        denoised_codes_cpu = denoised_codes[0].cpu()
        recon_coords_cpu = recon_coords[0].cpu()
        recon_types_cpu = recon_types[0].cpu()
        
        return {
            'success': True,
            'task_id': task_id,
            'field_method': field_method,
            'diffusion_method': diffusion_method,
            'denoised_codes': denoised_codes_cpu,
            'recon_coords': recon_coords_cpu,
            'recon_types': recon_types_cpu,
            'gpu_id': worker_id
        }
    
    except Exception as e:
        return {
            'success': False,
            'task_id': task.get('task_id', -1),
            'field_method': task.get('field_method', 'unknown'),
            'error': str(e),
            'gpu_id': worker_id
        }


def main_process_postprocess(
    gpu_result: dict,
    mol_save_dir: Path,
    elements: list
) -> dict:
    """
    主进程：处理GPU Worker返回的原始结果，进行IO操作和后处理
    
    Args:
        gpu_result: GPU Worker返回的原始结果
        mol_save_dir: 分子保存目录
        elements: 元素列表
    
    Returns:
        dict: 包含完整处理结果的字典（用于CSV写入）
    """
    if not gpu_result['success']:
        return {
            'success': False,
            'task_id': gpu_result['task_id'],
            'field_method': gpu_result.get('field_method', 'unknown'),
            'error': gpu_result.get('error', 'Unknown error')
        }
    
    try:
        task_id = gpu_result['task_id']
        field_method = gpu_result['field_method']
        diffusion_method = gpu_result['diffusion_method']
        
        # 从GPU结果中提取数据
        denoised_codes = gpu_result['denoised_codes']
        recon_coords = gpu_result['recon_coords']
        recon_types = gpu_result['recon_types']
        gpu_id = gpu_result.get('gpu_id', -1)
        
        # 转换为numpy
        recon_coords_np = recon_coords.numpy()
        recon_types_np = recon_types.numpy()
        
        # 过滤掉填充的原子（值为-1）
        valid_mask = recon_types_np != -1
        if valid_mask.any():
            recon_coords_np = recon_coords_np[valid_mask]
            recon_types_np = recon_types_np[valid_mask]
        
        generated_size = len(recon_coords_np)
        
        # 生成唯一ID
        unique_id = f"{task_id:06d}_{gpu_id}_{int(time.time() * 1000000) % 1000000}"
        generated_idx = task_id
        
        # 1. 保存codes
        code_path = mol_save_dir / f"code_{unique_id}_{field_method}.pt"
        torch.save(denoised_codes, code_path)
        
        # 2. 生成SDF
        sdf_string = xyz_to_sdf(recon_coords_np, recon_types_np, elements)
        sdf_path = mol_save_dir / f"genmol_{unique_id}.sdf"
        with open(sdf_path, 'w', encoding='utf-8') as sdf_file:
            sdf_file.write(sdf_string)
        
        # 3. 计算原子统计
        result_row = {
            'generated_idx': generated_idx,
            'field_method': field_method,
            'diffusion_method': diffusion_method,
            'size': generated_size
        }
        
        for i, element in enumerate(elements):
            count = (recon_types_np == i).sum()
            result_row[f'{element}_count'] = count
        
        # 4. 保存分子坐标和类型
        mol_file = mol_save_dir / f"generated_{unique_id}_{field_method}.npz"
        np.savez(mol_file, 
                coords=recon_coords_np,
                types=recon_types_np)
        
        return {
            'success': True,
            'generated_idx': generated_idx,
            'task_id': task_id,
            'field_method': field_method,
            'result_row': result_row,
            'unique_id': unique_id
        }
    
    except Exception as e:
        print(f"Error in postprocessing task {gpu_result.get('task_id', 'unknown')}: {e}")
        traceback.print_exc()
        return {
            'success': False,
            'task_id': gpu_result.get('task_id', -1),
            'field_method': gpu_result.get('field_method', 'unknown'),
            'error': str(e)
        }


def extract_checkpoint_identifier(fm_path: str) -> str:
    """
    Extract a unique identifier from checkpoint path for creating independent output directories.
    
    Args:
        fm_path: Path to the checkpoint file (e.g., .ckpt file)
    
    Returns:
        A unique identifier string, e.g., "20251212_version_2_last" or "version_0_epoch_199"
    """
    path = Path(fm_path)
    parts = path.parts
    
    # Extract date (if present, typically in format YYYYMMDD)
    date = None
    for part in parts:
        if part.isdigit() and len(part) == 8:  # YYYYMMDD format
            try:
                # Validate it's a reasonable date
                year = int(part[:4])
                month = int(part[4:6])
                day = int(part[6:8])
                if 2000 <= year <= 2100 and 1 <= month <= 12 and 1 <= day <= 31:
                    date = part
                    break
            except (ValueError, IndexError):
                continue
    
    # Extract version (e.g., version_0, version_1, version_2)
    version = None
    for part in parts:
        if part.startswith('version_') and part[8:].isdigit():
            version = part
            break
    
    # Extract checkpoint filename and convert to identifier
    checkpoint_name = None
    if path.suffix == '.ckpt':
        filename = path.stem  # filename without extension
        if filename == 'last':
            checkpoint_name = 'last'
        elif filename.startswith('model-epoch='):
            # Extract epoch number: "model-epoch=199" -> "epoch_199"
            epoch_str = filename.replace('model-epoch=', '')
            try:
                epoch_num = int(epoch_str)
                checkpoint_name = f'epoch_{epoch_num}'
            except ValueError:
                checkpoint_name = filename.replace('=', '_').replace('-', '_')
        else:
            # Use filename as-is, but clean it
            checkpoint_name = filename.replace('=', '_').replace('-', '_')
    else:
        # If not a .ckpt file, use the directory name
        checkpoint_name = path.name
    
    # Clean checkpoint_name to ensure it's safe for directory names
    # Remove or replace invalid characters
    invalid_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
    for char in invalid_chars:
        checkpoint_name = checkpoint_name.replace(char, '_')
    
    # Combine parts
    identifier_parts = []
    if date:
        identifier_parts.append(date)
    if version:
        identifier_parts.append(version)
    if checkpoint_name:
        identifier_parts.append(checkpoint_name)
    
    if not identifier_parts:
        # Fallback: use a hash or the full path's last few components
        # Use the last 3 path components as identifier
        if len(parts) >= 3:
            identifier_parts = list(parts[-3:])
        else:
            identifier_parts = [path.name]
    
    identifier = '_'.join(identifier_parts)
    
    # Final cleanup: ensure no consecutive underscores and no leading/trailing underscores
    while '__' in identifier:
        identifier = identifier.replace('__', '_')
    identifier = identifier.strip('_')
    
    return identifier if identifier else 'checkpoint'


@hydra.main(config_path="configs", config_name="sample_fm", version_base=None)
def main(config: DictConfig) -> None:
    # 设置multiprocessing启动方法（必须在主进程设置）
    setup_multiprocessing_spawn()
    
    # 设置PyTorch内存优化环境变量（主进程）
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:128'
    
    # 转换配置为字典格式
    config_dict = OmegaConf.to_container(config, resolve=True)
    
    # 解析GPU配置
    main_gpu_list, _ = parse_gpu_list_from_config(config, default_use_all=True)
    num_gpus = len(main_gpu_list)
    use_multi_gpu = config.get('use_multi_gpu', True) and num_gpus > 1
    
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
    
    # 基础输出目录
    base_output_dir = exps_root / "funcmol" / exp_name
    base_output_dir.mkdir(parents=True, exist_ok=True)
    
    # 提取checkpoint标识并创建独立的子目录
    checkpoint_identifier = extract_checkpoint_identifier(fm_path)
    # checkpoint_identifier = "20260120_version_2_last_decoder_finetuned_0(more epoch)"  # 临时硬编码
    output_dir = base_output_dir / "samples" / checkpoint_identifier
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取生成参数
    max_samples = config.get('max_samples', 10)  # 默认生成10个分子
    field_methods = config.get('field_methods', ['tanh'])  # 默认使用tanh方法
    
    # 根据use_multi_gpu决定worker数量
    if use_multi_gpu:
        num_workers = num_gpus  # 每个GPU一个worker
        print(f"Using multi-GPU mode: {num_gpus} GPUs, {num_workers} workers")
    else:
        num_workers = 1
        print(f"Using single-GPU mode: 1 worker")
    
    print(f"Checkpoint identifier: {checkpoint_identifier}")
    print(f"Generating {max_samples} molecules with field_methods: {field_methods}")
    print(f"Number of workers: {num_workers}")
    print(f"Physical GPUs: {main_gpu_list}")
    print(f"Output directory: {output_dir}")
    print("fm_pretrained_path: ", config["fm_pretrained_path"])
    print("nf_pretrained_path: ", config["nf_pretrained_path"])
    
    # 注意：主进程不再加载模型！模型将在每个worker进程中加载
    
    # 初始化CSV文件（使用新的独立输出目录）
    csv_path = output_dir / "denoiser_evaluation_results.csv"
    elements = config.dset.elements  # ["C", "H", "O", "N", "F"]
    csv_columns = ['generated_idx', 'field_method', 'diffusion_method', 'size']
    
    # 添加生成分子每种元素的原子数量
    for element in elements:
        csv_columns.append(f'{element}_count')
    
    # 创建CSV文件（如果不存在）
    # 注意：如果文件已存在，我们会在增量写入时追加，不会覆盖
    csv_file_exists = csv_path.exists() and csv_path.stat().st_size > 0
    
    # 创建分子数据保存目录（使用新的独立输出目录）
    mol_save_dir = output_dir / "molecule"
    mol_save_dir.mkdir(parents=True, exist_ok=True)
    
    # 检查使用的扩散方法
    diffusion_method = config.get('diffusion_method', 'old')
    print(f"Using diffusion method: {diffusion_method}")
    
    # 获取网格和编码维度
    grid_size = config.get('dset', {}).get('grid_size', 9)
    code_dim = config.get('encoder', {}).get('code_dim', 128)
    
    # ============================================================================
    # 创建任务队列：每个任务 = 一个分子的GPU计算任务
    # ============================================================================
    task_queue = Queue()
    result_queue = Queue()
    
    # 生成所有任务并放入队列
    task_id = 0
    total_tasks = 0
    for mol_idx in range(max_samples):
        for field_method in field_methods:
            task = {
                'task_id': task_id,
                'mol_idx': mol_idx,
                'field_method': field_method,
                'diffusion_method': diffusion_method,
                'grid_size': grid_size,
                'code_dim': code_dim
            }
            task_queue.put(task)
            task_id += 1
            total_tasks += 1
    
    print(f"Created {total_tasks} tasks for {max_samples} molecules × {len(field_methods)} field_methods")
    
    # ============================================================================
    # 启动长期运行的GPU Worker进程
    # ============================================================================
    print(f"\nStarting {num_workers} GPU workers...")
    print(f"Each worker will process molecules independently from the task queue")
    
    # 启动GPU Worker进程
    worker_processes = []
    for i in range(num_workers):
        logical_gpu_id = i  # 逻辑GPU ID（在worker进程中看到的GPU ID，通常是0）
        physical_gpu_id = main_gpu_list[i]  # 物理GPU ID（在主进程中的实际GPU ID）
        
        p = create_worker_process_with_gpu(
            worker_func=gpu_worker_process,
            logical_gpu_id=logical_gpu_id,
            physical_gpu_id=physical_gpu_id,
            args=(
                logical_gpu_id,
                physical_gpu_id,
                config_dict,
                config["fm_pretrained_path"],
                config["nf_pretrained_path"],
                task_queue,
                result_queue
            )
        )
        worker_processes.append(p)
        print(f"Started GPU Worker {i} (logical={logical_gpu_id}, physical={physical_gpu_id}, PID={p.pid})")
    
    # ============================================================================
    # 主进程：收集结果并进行IO后处理
    # ============================================================================
    results = []
    successful_tasks = 0
    failed_tasks = 0
    
    print(f"\nMain process: Collecting results and performing IO postprocessing...")
    
    # 跟踪CSV文件是否已有header（用于增量写入）
    # csv_file_exists 在之前已定义，如果文件已存在且有内容，说明已有header
    csv_has_header = csv_file_exists
    
    # 收集结果（使用tqdm显示进度）
    with tqdm(total=total_tasks, desc="Generating molecules") as pbar:
        while len(results) < total_tasks:
            try:
                # 从结果队列获取GPU计算结果
                gpu_result = result_queue.get(timeout=5.0)
                
                # 在主进程进行IO后处理
                final_result = main_process_postprocess(
                    gpu_result,
                    mol_save_dir,
                    elements
                )
                
                results.append(final_result)
                
                if final_result['success']:
                    successful_tasks += 1
                    # 立即追加到CSV文件（增量写入）
                    try:
                        result_row = final_result['result_row']
                        result_df = pd.DataFrame([result_row])
                        # 第一次写入需要header，后续追加不需要
                        result_df.to_csv(csv_path, mode='a', header=not csv_has_header, index=False)
                        csv_has_header = True  # 第一次写入后，后续都不需要header
                    except Exception as e:
                        print(f"\nWarning: Failed to write result to CSV for task {final_result.get('task_id', 'unknown')}: {e}")
                else:
                    failed_tasks += 1
                
                pbar.update(1)
            
            except queue.Empty:
                # 检查worker进程是否还在运行
                alive_count = sum(1 for p in worker_processes if p.is_alive())
                if alive_count == 0:
                    print("\nWarning: All workers have exited, but not all results received")
                    break
                continue
            except Exception as e:
                print(f"\nError collecting result: {e}")
                traceback.print_exc()
                failed_tasks += 1
                pbar.update(1)
    
    # 发送终止信号给所有worker
    print("\nSending termination signals to workers...")
    for _ in range(num_workers):
        task_queue.put(None)
    
    # 等待所有worker进程结束
    print("Waiting for workers to finish...")
    for p in worker_processes:
        p.join(timeout=10)
        if p.is_alive():
            print(f"Warning: Worker {p.pid} did not terminate, forcing...")
            p.terminate()
            p.join()
    
    print(f"All workers terminated")
    
    # ============================================================================
    # 重新排序CSV文件（可选：使结果按generated_idx排序）
    # ============================================================================
    print(f"\nMerging results: {successful_tasks} successful, {failed_tasks} failed")
    
    # 收集所有成功的结果
    successful_results = [r for r in results if r['success']]
    failed_results = [r for r in results if not r['success']]
    
    # 重新读取并排序CSV文件（使结果按generated_idx排序，便于查看）
    if successful_results and csv_path.exists():
        try:
            # 读取已写入的CSV
            existing_df = pd.read_csv(csv_path)
            if len(existing_df) > 0:
                # 按generated_idx排序并重新写入
                existing_df = existing_df.sort_values('generated_idx')
                existing_df.to_csv(csv_path, mode='w', index=False, header=True)
                print(f"Re-sorted {len(existing_df)} results in CSV by generated_idx")
        except Exception as e:
            print(f"Warning: Failed to re-sort CSV file: {e}")
            print(f"Results were saved incrementally. CSV contains {successful_tasks} successful results.")
    
    # 处理失败的任务（可选：保存错误信息）
    if failed_results:
        print(f"\nFailed tasks ({len(failed_results)}):")
        for r in failed_results[:10]:  # 只显示前10个错误
            print(f"  Task {r['task_id']} ({r.get('field_method', 'unknown')}): {r.get('error', 'Unknown error')}")
        if len(failed_results) > 10:
            print(f"  ... and {len(failed_results) - 10} more failures")
    
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
