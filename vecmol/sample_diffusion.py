import sys
sys.path.append("..")

import os
from vecmol.utils.misc import setup_worker_gpu_environment
setup_worker_gpu_environment()

from vecmol.models.vecmol import create_vecmol
import hydra
import torch
import traceback
from vecmol.utils.utils_diffusion import load_checkpoint_diffusion, find_checkpoint_path
from vecmol.dataset.dataset_field import create_gnf_converter
from vecmol.utils.utils_nf import load_neural_field, create_neural_field
from vecmol.utils.utils_base import xyz_to_sdf
from vecmol.utils.misc import (
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


def gpu_worker_process(
    logical_gpu_id: int,
    physical_gpu_id: int,
    config_dict: dict,
    diffusion_pretrained_path: str,
    nf_pretrained_path: str,
    task_queue: Queue,
    result_queue: Queue
):
    """
    GPU Worker process: long-running, batch processing tasks from Queue
    
    Args:
        logical_gpu_id: Logical GPU ID (GPU ID in worker process, usually 0)
        physical_gpu_id: Physical GPU ID (actual GPU ID in main process, for logging)
        config_dict: Configuration dictionary
        diffusion_pretrained_path: VecMol (diffusion) checkpoint path
        nf_pretrained_path: Neural field checkpoint path
        task_queue: Task queue (from main process to receive tasks)
        result_queue: Result queue (from worker process to send results to main process)
    
    Note: CUDA_VISIBLE_DEVICES is set at the beginning of the script via the _WORKER_GPU_ID environment variable.
    torch is already imported and uses the correct CUDA_VISIBLE_DEVICES setting here.
    """
    # Verify GPU settings
    if torch.cuda.is_available():
        num_visible_gpus = torch.cuda.device_count()
        if num_visible_gpus != 1:
            print(f"Warning: Expected 1 GPU after setting CUDA_VISIBLE_DEVICES={physical_gpu_id}, but got {num_visible_gpus}")
    
    # Set PyTorch memory optimization environment variable
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:128'
    
    # Set random seed (each worker independently)
    import random
    worker_id = multiprocessing.current_process().pid
    seed = int(time.time() * 1000) % 2**31 + worker_id
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    print(f"[GPU Worker logical={logical_gpu_id}, physical={physical_gpu_id} (PID {worker_id})] Initializing...")
    
    # Load models (each worker independently)
    with torch.no_grad():
        vecmol = create_vecmol(config_dict)
        vecmol, code_stats = load_checkpoint_diffusion(vecmol, diffusion_pretrained_path)
        vecmol = vecmol.cuda()
        vecmol.eval()
        
        # Decide whether to load decoder from diffusion checkpoint based on config
        load_decoder_from_diffusion = config_dict.get("load_decoder_from_diffusion_checkpoint", False)
        decoder = None
        
        if load_decoder_from_diffusion:
            # Try to load decoder_state_dict from checkpoint in diffusion_pretrained_path
            # If joint_finetune is enabled, decoder should be saved in diffusion checkpoint
            try:
                checkpoint_path = find_checkpoint_path(diffusion_pretrained_path)
                lightning_checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
                
                if "decoder_state_dict" in lightning_checkpoint:
                    print(f"[GPU Worker logical={logical_gpu_id}, physical={physical_gpu_id}] Found decoder_state_dict in diffusion checkpoint, loading from there...")
                    # Get config from checkpoint (use checkpoint config if available, otherwise use input config)
                    decoder_config = lightning_checkpoint.get("hyper_parameters", config_dict)
                    if decoder_config is None:
                        decoder_config = config_dict
                    
                    # Create decoder model
                    _, decoder = create_neural_field(decoder_config)
                    
                    # Load decoder state dict
                    decoder_state_dict = lightning_checkpoint["decoder_state_dict"]
                    # Handle _orig_mod. prefix (if present)
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
        
        # If loading from diffusion checkpoint fails or config is set to not use, load from neural field checkpoint
        if decoder is None:
            if nf_pretrained_path is None:
                raise ValueError("nf_pretrained_path must be specified when not loading decoder from diffusion checkpoint")
            _encoder, decoder = load_neural_field(nf_pretrained_path, config_dict)
        
        decoder = decoder.cuda()
        decoder.eval()
        decoder.set_code_stats(code_stats)
    
    print(f"[GPU Worker logical={logical_gpu_id}, physical={physical_gpu_id}] Models loaded successfully")
    
    # Worker main loop: continuously fetch tasks from Queue and process
    processed_count = 0
    try:
        while True:
            try:
                # Fetch task from queue (timeout to avoid infinite blocking)
                try:
                    task = task_queue.get(timeout=1.0)
                except queue.Empty:
                    # Check if there are any tasks (via special marker)
                    continue
                
                # Check termination signal
                if task is None:
                    print(f"[GPU Worker logical={logical_gpu_id}, physical={physical_gpu_id}] Received termination signal, exiting...")
                    break
                
                # Process task (GPU computation part)
                try:
                    result = process_gpu_batch(
                        task,
                        vecmol,
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
                    # Send error result, ensure each task has a result returned
                    result_queue.put({
                        'success': False,
                        'task_id': task.get('task_id', -1),
                        'field_method': task.get('field_method', 'unknown'),
                        'diffusion_method': task.get('diffusion_method', 'unknown'),
                        'error': str(e),
                        'gpu_id': logical_gpu_id
                    })
                    processed_count += 1
            
            except KeyboardInterrupt:
                print(f"[GPU Worker logical={logical_gpu_id}, physical={physical_gpu_id}] Interrupted, exiting...")
                break
            except Exception as e:
                print(f"[GPU Worker logical={logical_gpu_id}, physical={physical_gpu_id}] Unexpected error in worker loop: {e}")
                traceback.print_exc()
                # Continue running, do not exit worker process
                continue
    finally:
        print(f"[GPU Worker logical={logical_gpu_id}, physical={physical_gpu_id}] Processed {processed_count} tasks, shutting down...")


def process_gpu_batch(
    task: dict,
    vecmol: torch.nn.Module,
    decoder: torch.nn.Module,
    code_stats: dict,
    config_dict: dict,
    worker_id: int = None
) -> dict:
    """
    GPU Worker internal: process GPU computation part of single task
    
    Only do GPU computation, no IO operations
    
    Args:
        task: Task dictionary
        vecmol: VecMol model
        decoder: Decoder model
        code_stats: Code statistics
        config_dict: Configuration dictionary
        worker_id: Worker ID (for debugging)
    
    Returns:
        dict: Dictionary containing GPU computation results (raw data, no IO processing)
        {
            'success': bool,
            'task_id': int,
            'field_method': str,
            'diffusion_method': str,
            'denoised_codes': torch.Tensor (CPU),
            'recon_coords': torch.Tensor (CPU),
            'recon_types': torch.Tensor (CPU),
            'error': str (if failed)
        }
    """
    try:
        field_method = task['field_method']
        diffusion_method = task['diffusion_method']
        grid_size = task['grid_size']
        code_dim = task['code_dim']
        task_id = task['task_id']
        
        # 1. Create converter with specific field method
        method_config = config_dict.copy()
        method_config['converter']['gradient_field_method'] = field_method
        converter = create_gnf_converter(method_config)
        
        # 2. Generate codes (single molecule, batch_size=1)
        # DDPM sampling
        shape = (1, grid_size**3, code_dim)
        with torch.no_grad():
            denoised_codes_3d = vecmol.sample_ddpm(shape, code_stats=code_stats, progress=False)
        denoised_codes = denoised_codes_3d.view(1, grid_size**3, code_dim)
        
        # 3. Reconstruct molecule (GPU computation)
        recon_coords, recon_types = converter.gnf2mol(
            decoder,
            denoised_codes,
            enable_timing=False,
            sample_id=task_id
        )
        
        # Check if there are valid atoms (-1 is fill value)
        recon_types_device = recon_types[0]
        valid_mask = recon_types_device != -1
        if not valid_mask.any():
            # If no valid atoms, this is a failed reconstruction
            raise ValueError(f"Reconstruction failed: no valid atoms found for task {task_id} with method {field_method}")
        
        # 4. Move results to CPU (prepare for transmission to main process)
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
    Main process: process raw results from GPU Worker, perform IO operations and postprocessing
    
    Args:
        gpu_result: Raw results from GPU Worker
        mol_save_dir: Molecule save directory
        elements: Element list
    
    Returns:
        dict: Dictionary containing complete processing results (for CSV writing)
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
        
        # Extract data from GPU results
        denoised_codes = gpu_result['denoised_codes']
        recon_coords = gpu_result['recon_coords']
        recon_types = gpu_result['recon_types']
        gpu_id = gpu_result.get('gpu_id', -1)
        
        # Convert to numpy
        recon_coords_np = recon_coords.numpy()
        recon_types_np = recon_types.numpy()
        
        # Filter out fill atoms (value -1)
        valid_mask = recon_types_np != -1
        if valid_mask.any():
            recon_coords_np = recon_coords_np[valid_mask]
            recon_types_np = recon_types_np[valid_mask]
        
        generated_size = len(recon_coords_np)
        
        # Generate unique ID
        unique_id = f"{task_id:06d}_{gpu_id}_{int(time.time() * 1000000) % 1000000}"
        generated_idx = task_id
        
        # 1. Save codes
        code_path = mol_save_dir / f"code_{unique_id}_{field_method}.pt"
        torch.save(denoised_codes, code_path)
        
        # 2. Generate SDF
        sdf_string = xyz_to_sdf(recon_coords_np, recon_types_np, elements)
        sdf_path = mol_save_dir / f"genmol_{unique_id}.sdf"
        with open(sdf_path, 'w', encoding='utf-8') as sdf_file:
            sdf_file.write(sdf_string)
        
        # 3. Calculate atom statistics
        result_row = {
            'generated_idx': generated_idx,
            'field_method': field_method,
            'diffusion_method': diffusion_method,
            'size': generated_size
        }
        
        for i, element in enumerate(elements):
            count = (recon_types_np == i).sum()
            result_row[f'{element}_count'] = count
        
        # 4. Save molecule coordinates and types
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


def extract_checkpoint_identifier(diffusion_path: str) -> str:
    """
    Extract a unique identifier from checkpoint path for creating independent output directories.
    
    Args:
        diffusion_path: Path to the checkpoint file (e.g., .ckpt file)
    
    Returns:
        A unique identifier string, e.g., "20251212_version_2_last" or "version_0_epoch_199"
    """
    path = Path(diffusion_path)
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


@hydra.main(config_path="configs", config_name="sample_diffusion", version_base=None)
def main(config: DictConfig) -> None:
    # Set multiprocessing startup method (must be set in main process)
    setup_multiprocessing_spawn()
    
    # Set PyTorch memory optimization environment variable (main process)
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:128'
    
    # Convert config to dictionary format
    config_dict = OmegaConf.to_container(config, resolve=True)
    
    # Parse GPU configuration
    main_gpu_list, _ = parse_gpu_list_from_config(config, default_use_all=True)
    num_gpus = len(main_gpu_list)
    use_multi_gpu = config.get('use_multi_gpu', True) and num_gpus > 1
    
    # Set output directory
    exps_root = Path(__file__).parent.parent / "exps"
    diffusion_path = config.get("diffusion_pretrained_path")
    
    # If diffusion_pretrained_path points to a .ckpt file, need to find the corresponding directory
    if diffusion_path.endswith('.ckpt'):
        # Extract directory structure from .ckpt file path
        diffusion_path_parts = Path(diffusion_path).parts
        try:
            vecmol_idx = diffusion_path_parts.index('vecmol')
            if vecmol_idx + 2 < len(diffusion_path_parts):
                exp_name = f"{diffusion_path_parts[vecmol_idx + 1]}/{diffusion_path_parts[vecmol_idx + 2]}"
            else:
                exp_name = diffusion_path_parts[vecmol_idx + 1]
        except ValueError:
            # If vecmol is not found, use the name of the parent directory
            exp_name = Path(diffusion_path).parent.parent.name
    else:
        # If diffusion_pretrained_path points to a directory, use it directly
        diffusion_path_parts = Path(diffusion_path).parts
        try:
            vecmol_idx = diffusion_path_parts.index('vecmol')
            if vecmol_idx + 2 < len(diffusion_path_parts):
                exp_name = f"{diffusion_path_parts[vecmol_idx + 1]}/{diffusion_path_parts[vecmol_idx + 2]}"
            else:
                exp_name = diffusion_path_parts[vecmol_idx + 1]
        except ValueError:
            exp_name = Path(diffusion_path).parent.parent.name
    
    # Base output directory
    base_output_dir = exps_root / "vecmol" / exp_name
    base_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract checkpoint identifier and create independent subdirectories
    checkpoint_identifier = extract_checkpoint_identifier(diffusion_path)
    # checkpoint_identifier = "20260120_version_2_last_decoder_finetuned_0(more epoch)"  # Temporary hardcoding
    output_dir = base_output_dir / "samples" / checkpoint_identifier
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get generation parameters
    max_samples = config.get('max_samples', 10)  # Default generate 10 molecules
    field_methods = config.get('field_methods', ['tanh'])  # Default use tanh method
    
    # Determine worker number based on use_multi_gpu
    if use_multi_gpu:
        num_workers = num_gpus  # One worker per GPU
        print(f"Using multi-GPU mode: {num_gpus} GPUs, {num_workers} workers")
    else:
        num_workers = 1
        print(f"Using single-GPU mode: 1 worker")
    
    print(f"Checkpoint identifier: {checkpoint_identifier}")
    print(f"Generating {max_samples} molecules with field_methods: {field_methods}")
    print(f"Number of workers: {num_workers}")
    print(f"Physical GPUs: {main_gpu_list}")
    print(f"Output directory: {output_dir}")
    print("diffusion_pretrained_path: ", config["diffusion_pretrained_path"])
    print("nf_pretrained_path: ", config["nf_pretrained_path"])
    
    
    # Initialize CSV file (using new independent output directory)
    csv_path = output_dir / "denoiser_evaluation_results.csv"
    elements = config.dset.elements  # ["C", "H", "O", "N", "F"]
    csv_columns = ['generated_idx', 'field_method', 'diffusion_method', 'size']
    
    # Add atom count for each element in generated molecules
    for element in elements:
        csv_columns.append(f'{element}_count')
    
    # Create CSV file (if not exists)
    # Note: If file exists, we will append during incremental writing, not overwrite
    csv_file_exists = csv_path.exists() and csv_path.stat().st_size > 0
    
    # Create molecule data save directory (using new independent output directory)
    mol_save_dir = output_dir / "molecule"
    mol_save_dir.mkdir(parents=True, exist_ok=True)
    
    # Check used diffusion method
    diffusion_method = config.get('diffusion_method', 'ddpm_x0')
    print(f"Using diffusion method: {diffusion_method}")
    
    # Get grid and code dimension
    grid_size = config.get('dset', {}).get('grid_size', 9)
    code_dim = config.get('encoder', {}).get('code_dim', 128)
    
    task_queue = Queue()
    result_queue = Queue()
    
    # Generate all tasks and put them into queue
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
    print(f"\nStarting {num_workers} GPU workers...")
    print(f"Each worker will process molecules independently from the task queue")
    
    # Start GPU Worker processes
    worker_processes = []
    for i in range(num_workers):
        logical_gpu_id = i  # Logical GPU ID (GPU ID seen in worker process, usually 0)
        physical_gpu_id = main_gpu_list[i]  # Physical GPU ID (actual GPU ID in main process)
        
        p = create_worker_process_with_gpu(
            worker_func=gpu_worker_process,
            logical_gpu_id=logical_gpu_id,
            physical_gpu_id=physical_gpu_id,
            args=(
                logical_gpu_id,
                physical_gpu_id,
                config_dict,
                config["diffusion_pretrained_path"],
                config["nf_pretrained_path"],
                task_queue,
                result_queue
            )
        )
        worker_processes.append(p)
        print(f"Started GPU Worker {i} (logical={logical_gpu_id}, physical={physical_gpu_id}, PID={p.pid})")
    
    results = []
    successful_tasks = 0
    failed_tasks = 0
    
    print(f"\nMain process: Collecting results and performing IO postprocessing...")
    
    # Track if CSV file has header (for incremental writing)
    csv_has_header = csv_file_exists
    
    # Collect results (using tqdm to show progress)
    consecutive_empty_count = 0
    max_consecutive_empty = 10  # Allow 10 consecutive empty queues, then check process status
    
    with tqdm(total=total_tasks, desc="Generating molecules") as pbar:
        while len(results) < total_tasks:
            try:
                # Get GPU computation results from result queue
                gpu_result = result_queue.get(timeout=5.0)
                
                # Perform IO postprocessing in main process
                final_result = main_process_postprocess(
                    gpu_result,
                    mol_save_dir,
                    elements
                )
                
                results.append(final_result)
                consecutive_empty_count = 0  # Reset empty queue count
                
                if final_result['success']:
                    successful_tasks += 1
                    # Immediately append to CSV file (incremental writing)
                    try:
                        result_row = final_result['result_row']
                        result_df = pd.DataFrame([result_row])
                        # First write needs header, subsequent appends do not need header
                        result_df.to_csv(csv_path, mode='a', header=not csv_has_header, index=False)
                        csv_has_header = True  # After first write, subsequent writes do not need header
                    except Exception as e:
                        print(f"\nWarning: Failed to write result to CSV for task {final_result.get('task_id', 'unknown')}: {e}")
                else:
                    failed_tasks += 1
                
                pbar.update(1)
            
            except queue.Empty:
                consecutive_empty_count += 1
                # Check if worker processes are still running
                alive_count = sum(1 for p in worker_processes if p.is_alive())
                if alive_count == 0:
                    # All workers have exited, check if there are any incomplete tasks
                    if len(results) < total_tasks:
                        print(f"\nWarning: All workers have exited, but only {len(results)}/{total_tasks} results received")
                        print(f"Missing {total_tasks - len(results)} results. This may indicate worker crashes or incomplete processing.")
                    break
                elif consecutive_empty_count >= max_consecutive_empty:
                    # Check worker status after multiple consecutive empty queues
                    print(f"\nWarning: Queue empty for {consecutive_empty_count} consecutive checks. Checking worker status...")
                    for i, p in enumerate(worker_processes):
                        if not p.is_alive():
                            print(f"  Worker {i} (PID {p.pid}) is not alive (exitcode: {p.exitcode})")
                    consecutive_empty_count = 0  # Reset count, continue waiting
                continue
            except Exception as e:
                print(f"\nError collecting result: {e}")
                traceback.print_exc()
                failed_tasks += 1
                pbar.update(1)
    
    # Send termination signals to all workers
    print("\nSending termination signals to workers...")
    for _ in range(num_workers):
        task_queue.put(None)
    
    # Wait for all worker processes to finish
    print("Waiting for workers to finish...")
    for p in worker_processes:
        p.join(timeout=10)
        if p.is_alive():
            print(f"Warning: Worker {p.pid} did not terminate, forcing...")
            p.terminate()
            p.join()
    
    print(f"All workers terminated")
    print(f"\nMerging results: {successful_tasks} successful, {failed_tasks} failed")
    
    # Collect all successful results
    successful_results = [r for r in results if r['success']]
    failed_results = [r for r in results if not r['success']]
    
    # Re-read and sort CSV file (sort results by generated_idx for easier viewing)
    if successful_results and csv_path.exists():
        try:
            # Read existing CSV
            existing_df = pd.read_csv(csv_path)
            if len(existing_df) > 0:
                # Sort by generated_idx and re-write
                existing_df = existing_df.sort_values('generated_idx')
                existing_df.to_csv(csv_path, mode='w', index=False, header=True)
                print(f"Re-sorted {len(existing_df)} results in CSV by generated_idx")
        except Exception as e:
            print(f"Warning: Failed to re-sort CSV file: {e}")
            print(f"Results were saved incrementally. CSV contains {successful_tasks} successful results.")
    
    # Process failed tasks (optional: save error information)
    if failed_results:
        print(f"\nFailed tasks ({len(failed_results)}):")
        for r in failed_results[:10]:  # Only show first 10 errors
            print(f"  Task {r['task_id']} ({r.get('field_method', 'unknown')}): {r.get('error', 'Unknown error')}")
        if len(failed_results) > 10:
            print(f"  ... and {len(failed_results) - 10} more failures")
    
    # Analyze results
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
