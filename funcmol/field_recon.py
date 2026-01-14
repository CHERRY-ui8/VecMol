#!/usr/bin/env python3
"""
Field Reconstruction Evaluation Script

This script can be used for both field type evaluation (stage1) and neural field evaluation (stage2).

0. Define field method (converter), save path (gt_field version, dataset name, )
   - Create GNFConverter with specific gradient_field_method (gaussian_mag, tanh, etc.)
   - Set output directory for results

1. Set data config, load dataset
   - Load FieldDataset with appropriate parameters
   - Support gt_field and nf_field modes

2. Loop for data (Pool.map and pool.imap_unordered):
   2.1 Convert to field using different field generation methods:
       - gt_field mode: Use ground truth coordinates to generate field via mol2gnf
       - nf_field mode: Use trained neural field encoder/decoder to generate field
   2.2 Reconstruct mol
       - Use gnf2mol method with appropriate decoder and codes
       - Support ground truth field and neural field reconstruction
   2.3 Save mol (if use gt_field, save in /exps/gt_field; if use predicted_field, save in the nf path under /exps/neural_field)
       - Currently saves results to CSV format
       - Can be extended to save individual molecular files

3. If mol exist: analyze rmsd. save results (csv: rmsd, data_id, size, atom_count_mismatch, ), summary (mean, std, min, max)
   - Compute RMSD between ground truth and reconstructed coordinates
   - Track atom count mismatches and other metrics
   - Generate summary statistics for each field method
"""

import os
# 注意：在spawn模式下，子进程会重新执行这个脚本
# 如果环境变量_WORKER_GPU_ID已设置，说明这是worker子进程，需要先设置CUDA_VISIBLE_DEVICES
if "_WORKER_GPU_ID" in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["_WORKER_GPU_ID"]
    del os.environ["_WORKER_GPU_ID"]  # 清理临时环境变量

import torch
import random
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
import hydra
from omegaconf import DictConfig, OmegaConf, ListConfig
from funcmol.dataset.dataset_field import create_gnf_converter
from funcmol.utils.utils_nf import load_neural_field
from funcmol.dataset.dataset_field import FieldDataset
import multiprocessing
import threading

def compute_rmsd(coords1: torch.Tensor, coords2: torch.Tensor) -> float:
    """Compute RMSD using Hungarian algorithm."""
    if coords1.shape[0] != coords2.shape[0]:
        return float('inf')

    if coords1.device != coords2.device:
        coords2 = coords2.to(coords1.device)
    
    # 确保类型一致（Float32）
    if coords1.dtype != coords2.dtype:
        coords2 = coords2.to(coords1.dtype)
    
    dists = torch.cdist(coords1, coords2)
    row_indices, col_indices = linear_sum_assignment(dists.cpu().numpy())
    rmsd = torch.sqrt(torch.mean(dists[row_indices, col_indices] ** 2))
    return float(rmsd.item())


# 全局锁，用于线程安全的CSV写入
csv_write_lock = threading.Lock()


def worker_process(gpu_id, physical_gpu_id, tasks, result_queue):
    """
    Worker进程函数，处理分配给特定GPU的任务
    必须在模块级别定义，以便在spawn模式下可以被pickle
    
    注意：在spawn模式下，子进程会重新执行整个脚本
    我们通过环境变量_WORKER_GPU_ID在脚本执行前设置CUDA_VISIBLE_DEVICES
    """
    # 注意：CUDA_VISIBLE_DEVICES已经在脚本开头通过环境变量_WORKER_GPU_ID设置了
    # 这里torch已经被导入并使用了正确的CUDA_VISIBLE_DEVICES设置
    
    # 验证GPU设置
    if torch.cuda.is_available():
        num_visible_gpus = torch.cuda.device_count()
        if num_visible_gpus != 1:
            print(f"Warning: Expected 1 GPU after setting CUDA_VISIBLE_DEVICES={physical_gpu_id}, but got {num_visible_gpus}")
    
    # 处理所有分配给这个GPU的任务
    for task in tasks:
        try:
            result = process_single_molecule(task)
            result_queue.put(result)
        except Exception as e:
            result_queue.put({'success': False, 'error': str(e), 'task': task[0]})

def process_single_molecule(args):
    """
    处理单个分子的函数，用于多进程并行处理
    
    Args:
        args: 包含所有必要参数的元组
            (sample_idx, gpu_id, method_config, field_mode, field_method, 
             output_dir, csv_path, mol_save_dir, data_dir, dset_name, elements,
             encoder_path, decoder_path, full_config_dict)
    
    Returns:
        dict: 处理结果
    """
    (sample_idx, logical_gpu_id, physical_gpu_id, method_config, field_mode, field_method, 
     output_dir, csv_path, mol_save_dir, data_dir, dset_name, elements,
     encoder_path, decoder_path, full_config_dict) = args
    
    # 将字符串路径转换为Path对象
    mol_save_dir = Path(mol_save_dir)
    csv_path = Path(csv_path)
    
    try:
        # 注意：CUDA_VISIBLE_DEVICES已经在worker_process中设置
        # 这里不需要再次设置，因为每个worker进程已经在启动时设置了
        import torch
        
        # 加载配置
        from omegaconf import DictConfig, OmegaConf
        config = OmegaConf.create(full_config_dict)
        method_config_obj = OmegaConf.create(method_config)
        
        # 在子进程中重新创建数据集（因为dataset对象不能序列化）
        default_converter = create_gnf_converter(full_config_dict)
        dataset = FieldDataset(
            gnf_converter=default_converter,
            dset_name=dset_name,
            data_dir=data_dir,
            elements=elements,
            split=config.get('split', 'val'),
            n_points=config.dset.n_points,
            rotate=config.dset.data_aug if config.get('split', 'val') == "train" else False,
            resolution=config.dset.resolution,
            grid_dim=config.dset.grid_dim,
            sample_full_grid=config.dset.get('sample_full_grid', False),
            debug_one_mol=config.get('debug_one_mol', False),
            debug_subset=config.dset.get('debug_subset', False),
        )
        
        # 确保LMDB连接在子进程中正确建立（spawn模式下需要）
        # 在spawn模式下，每个子进程都需要重新建立LMDB连接
        if hasattr(dataset, 'use_lmdb') and dataset.use_lmdb:
            # 强制重新连接，确保每个子进程都有独立的连接
            if dataset.db is not None:
                try:
                    # 检查连接是否有效
                    with dataset.db.begin() as txn:
                        txn.stat()
                except Exception:
                    # 连接无效，关闭并重新连接
                    dataset._close_db()
            if dataset.db is None:
                dataset._connect_db()
        
        # 加载数据集样本（带重试机制）
        max_retries = 3
        for retry in range(max_retries):
            try:
                sample = dataset[sample_idx]
                break
            except Exception as e:
                if retry < max_retries - 1:
                    # 如果是LMDB错误，尝试重新连接
                    if "lmdb" in str(e).lower() or "closed" in str(e).lower() or "deleted" in str(e).lower():
                        if hasattr(dataset, 'use_lmdb') and dataset.use_lmdb:
                            dataset._close_db()
                            dataset._connect_db()
                    continue
                else:
                    raise
        gt_coords = sample.pos
        gt_types = sample.x
        
        # 创建converter
        converter = create_gnf_converter(method_config)
        
        # 初始化decoder和codes
        codes = None
        decoder = None
        
        if field_mode == 'gt_field':
            # Create dummy codes and decoder
            grid_size = config.get('dset', {}).get('grid_size', 9)
            code_dim = config.get('encoder', {}).get('code_dim', 128)
            dummy_codes = torch.randn(1, grid_size**3, code_dim, device=gt_coords.device)
            
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
            decoder = dummy_decoder
            codes = dummy_codes
            
        elif field_mode == 'nf_field':
            # 加载encoder和decoder（每个进程独立加载）
            encoder, decoder = load_neural_field(encoder_path, config)
            encoder = encoder.cuda()
            decoder = decoder.cuda()
            encoder.eval()
            decoder.eval()
            
            # 编码当前分子
            from torch_geometric.data import Data, Batch
            data = Data(
                pos=sample.pos,
                x=sample.x,
                batch=torch.zeros(sample.pos.shape[0], dtype=torch.long, device=sample.pos.device)
            )
            batch_pyg = Batch.from_data_list([data])
            batch_pyg = batch_pyg.cuda()
            
            with torch.no_grad():
                batch_codes = encoder(batch_pyg)  # [1, grid_size**3, code_dim]
                codes = batch_codes[0:1]
        
        # 从配置中读取保存参数
        autoregressive_config = method_config.get("converter", {}).get("autoregressive_clustering", {})
        save_clustering_history = autoregressive_config.get("enable_clustering_history", False)
        save_gradient_ascent_sdf = autoregressive_config.get("save_gradient_ascent_sdf", False)
        enable_timing = autoregressive_config.get("enable_timing", False)
        
        # 构建gnf2mol调用参数
        gnf2mol_kwargs = {
            "decoder": decoder,
            "codes": codes,
            "sample_id": sample_idx
        }
        
        if save_clustering_history:
            gnf2mol_kwargs["save_clustering_history"] = True
            gnf2mol_kwargs["clustering_history_dir"] = str(output_dir / "clustering_history")
        
        if save_gradient_ascent_sdf:
            gnf2mol_kwargs["save_gradient_ascent_sdf"] = True
            gnf2mol_kwargs["gradient_ascent_sdf_dir"] = str(output_dir / "gradient_ascent_sdf")
            gnf2mol_kwargs["gradient_ascent_sdf_interval"] = autoregressive_config.get("gradient_ascent_sdf_interval", 100)
        
        if enable_timing:
            gnf2mol_kwargs["enable_timing"] = True
        
        recon_coords, recon_types = converter.gnf2mol(**gnf2mol_kwargs)
        
        # 处理结果
        recon_coords_device = recon_coords[0].to(gt_coords.device)
        rmsd = compute_rmsd(gt_coords, recon_coords_device)
        atom_count_mismatch = gt_coords.shape[0] != recon_coords[0].shape[0]
        original_size = gt_coords.shape[0]
        
        # 构建结果行
        result_row = {
            'sample_idx': sample_idx,
            'field_mode': field_mode,
            'field_method': field_method,
            'rmsd': rmsd,
            'size': original_size,
            'atom_count_mismatch': atom_count_mismatch
        }
        
        # 为nf_field模式添加原子统计信息
        if field_mode == 'nf_field':
            elements = elements  # 使用传入的elements参数
            for j, element in enumerate(elements):
                count = (gt_types == j).sum().item()
                result_row[f'gt_{element}_count'] = count
            
            recon_types_device = recon_types[0].to(gt_coords.device)
            valid_recon_types = recon_types_device[recon_types_device != -1]
            for j, element in enumerate(elements):
                count = (valid_recon_types == j).sum().item()
                result_row[f'recon_{element}_count'] = count
        
        # 先保存分子坐标和类型（如果失败，不写入CSV）
        mol_file = mol_save_dir / f"sample_{sample_idx:04d}_{field_method}.npz"
        try:
            np.savez(mol_file, 
                    coords=recon_coords_device.cpu().numpy(),
                    types=gt_types.cpu().numpy(),
                    gt_coords=gt_coords.cpu().numpy(),
                    rmsd=rmsd)
        except Exception as save_error:
            # 如果保存文件失败，抛出异常，让外层except处理
            raise Exception(f"Failed to save molecule file: {save_error}") from save_error
        
        # 返回结果，不在子进程中写入CSV（避免多进程写入冲突）
        # CSV写入将在主进程统一进行
        return {
            'success': True, 
            'sample_idx': sample_idx, 
            'rmsd': rmsd,
            'result_row': result_row  # 包含完整的结果行数据
        }
        
    except Exception as e:
        print(f"Error processing sample {sample_idx} with {field_method} on GPU {physical_gpu_id} (logical {logical_gpu_id}): {e}")
        import traceback
        traceback.print_exc()
        
        # 线程安全地写入错误结果
        error_row = {
            'sample_idx': sample_idx,
            'field_mode': field_mode,
            'field_method': field_method,
            'rmsd': float('inf'),
            'size': gt_coords.shape[0] if 'gt_coords' in locals() else 0,
            'atom_count_mismatch': True
        }
        
        if field_mode == 'nf_field':
            elements = elements if 'elements' in locals() else []
            if 'gt_types' in locals():
                for j, element in enumerate(elements):
                    count = (gt_types == j).sum().item()
                    error_row[f'gt_{element}_count'] = count
            for element in elements:
                error_row[f'recon_{element}_count'] = 0
        
        # 返回错误结果，不在子进程中写入CSV（避免多进程写入冲突）
        # CSV写入将在主进程统一进行
        return {
            'success': False, 
            'sample_idx': sample_idx, 
            'error': str(e),
            'error_row': error_row  # 包含完整的错误行数据
        }


# @hydra.main(version_base=None, config_path="configs", config_name="field_recon_qm9")
@hydra.main(version_base=None, config_path="configs", config_name="field_recon_drugs")
def main(config: DictConfig) -> None:
    # 设置全局随机种子
    seed = config.get('seed', 1234)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 确保CUDA操作的确定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # 设置PyTorch的确定性模式（PyTorch 1.7+）
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except AttributeError:
        # 旧版本PyTorch不支持此选项
        pass
    # 设置环境变量以确保完全确定性
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # 确保CUDA操作的确定性
    
    # 设置PyTorch内存优化环境变量
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:128'
    
    # 0. Define field method (converter), save path (gt_field version, dataset name, )
    config_dict = OmegaConf.to_container(config, resolve=True)
    
    # Choose field mode
    field_mode = config.get('field_mode', 'gt_field')
    
    # Set output directory based on field mode
    # 使用与funcmol目录平级的exps目录
    exps_root = Path(__file__).parent.parent / "exps"
    
    if field_mode == 'gt_field':
        output_dir = exps_root / "gt_field"
    elif field_mode == 'nf_field':
        nf_path = config.get("nf_pretrained_path")
        nf_path_parts = Path(nf_path).parts
        try:
            neural_field_idx = nf_path_parts.index('neural_field')
            if neural_field_idx + 2 < len(nf_path_parts):
                exp_name = f"{nf_path_parts[neural_field_idx + 1]}/{nf_path_parts[neural_field_idx + 2]}"
            else:
                exp_name = nf_path_parts[neural_field_idx + 1]
        except ValueError:
            exp_name = Path(nf_path).parent.parent.name
        output_dir = exps_root / "neural_field" / exp_name
    else:
        raise ValueError(f"Unsupported field_mode: {field_mode}. Only 'gt_field' and 'nf_field' are supported.")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Set data config, load dataset
    # Create default converter for dataset, not used in field computation, only used for dataset loading
    default_converter = create_gnf_converter(config_dict)
    
    # Load dataset using OmegaConf - much more elegant!
    dataset = FieldDataset(
        gnf_converter=default_converter,
        dset_name=config.dset.dset_name,
        data_dir=config.dset.data_dir,
        elements=config.dset.elements,
        split=config.get('split', 'val'),  # train/val/test split
        n_points=config.dset.n_points,
        rotate=config.dset.data_aug if config.get('split', 'val') == "train" else False,
        resolution=config.dset.resolution,
        grid_dim=config.dset.grid_dim,
        sample_full_grid=config.dset.get('sample_full_grid', False),
        debug_one_mol=config.get('debug_one_mol', False),
        debug_subset=config.get('debug_subset', False),
    )
    field_methods = config.get('field_methods', ['gaussian_mag', 'tanh'])
    
    # For gt_field and nf_field modes, load dataset for reconstruction
    # 优先检查是否手动指定了样本索引
    manual_sample_indices = config.get('sample_indices')
    
    if manual_sample_indices is not None:
        # 如果手动指定了样本索引，使用指定的索引
        # 处理 OmegaConf 的 ListConfig 类型，转换为 Python list
        if isinstance(manual_sample_indices, (list, tuple, ListConfig)):
            # 将 ListConfig 转换为普通 Python list
            if isinstance(manual_sample_indices, ListConfig):
                manual_sample_indices = OmegaConf.to_container(manual_sample_indices, resolve=True)
            sample_indices = [int(idx) for idx in manual_sample_indices]
            # 验证索引是否在有效范围内
            total_samples = len(dataset)
            valid_indices = [idx for idx in sample_indices if 0 <= idx < total_samples]
            if len(valid_indices) != len(sample_indices):
                invalid_indices = [idx for idx in sample_indices if idx < 0 or idx >= total_samples]
                print(f"Warning: Some sample indices are out of range [0, {total_samples-1}]: {invalid_indices}")
                print(f"Using {len(valid_indices)} valid indices out of {len(sample_indices)} specified")
            sample_indices = valid_indices
            sample_indices.sort()  # 保持索引顺序，便于调试
            print(f"Using manually specified sample indices: {sample_indices}")
        else:
            raise ValueError(f"sample_indices must be a list, tuple, or ListConfig, got {type(manual_sample_indices)}")
    else:
        # 如果没有手动指定，使用max_samples进行随机采样
        max_samples = config.get('max_samples')
        if max_samples is None or max_samples <= 0:
            max_samples = len(dataset)
            sample_indices = list(range(len(dataset)))
        else:
            # 从验证集中随机采样max_samples个分子，保留原始索引
            total_samples = len(dataset)
            if max_samples >= total_samples:
                sample_indices = list(range(total_samples))
            else:
                sample_indices = random.sample(range(total_samples), max_samples)
                sample_indices.sort()  # 保持索引顺序，便于调试
    
    print(f"Evaluating {len(sample_indices)} molecules (sampled from {len(dataset)} total) with field_mode: {field_mode}, methods: {field_methods}")
    print(f"Sample indices: {sample_indices[:10]}{'...' if len(sample_indices) > 10 else ''}")
        
    encoder = None
    decoder = None
    
    if field_mode == 'nf_field':
        if config.get("nf_pretrained_path") is None:
            raise ValueError("nf_pretrained_path must be specified for nf_field mode")
        encoder, decoder = load_neural_field(config["nf_pretrained_path"], config)
        print("Loaded neural field encoder/decoder")
        
        # 移动模型到GPU并设置为评估模式
        encoder = encoder.cuda()
        decoder = decoder.cuda()
        encoder.eval()
        decoder.eval()
    
    # 2. Loop for data (Pool.map and pool.imap_unordered)
    # Initialize CSV file with headers
    if field_mode == 'gt_field':
        csv_path = output_dir / "field_evaluation_results.csv"
    elif field_mode == 'nf_field':
        csv_path = output_dir / "nf_evaluation_results.csv"
    else:
        raise ValueError(f"Unsupported field_mode: {field_mode}. Only 'gt_field' and 'nf_field' are supported.")
    
    # 根据field_mode决定CSV列
    if field_mode == 'gt_field':
        # gt_field模式保持原有列
        csv_columns = ['sample_idx', 'field_mode', 'field_method', 'rmsd', 'size', 'atom_count_mismatch']
    elif field_mode == 'nf_field':
        # nf_field模式添加原子统计信息
        elements = config.dset.elements  # ["C", "H", "O", "N", "F"]
        csv_columns = ['sample_idx', 'field_mode', 'field_method', 'rmsd', 'size', 'atom_count_mismatch']
        
        # 添加原始分子每种元素的原子数量
        for element in elements:
            csv_columns.append(f'gt_{element}_count')
        
        # 添加重建分子每种元素的原子数量
        for element in elements:
            csv_columns.append(f'recon_{element}_count')
    
    # Only create new CSV file if it doesn't exist
    if not csv_path.exists():
        results_df = pd.DataFrame(columns=csv_columns)
        results_df.to_csv(csv_path, index=False)
    
    # Create molecular data save directory
    mol_save_dir = output_dir / "molecule"
    mol_save_dir.mkdir(parents=True, exist_ok=True)
    
    # 检测可用的GPU数量
    num_gpus = torch.cuda.device_count()
    
    # 获取要使用的GPU列表（优先级：配置文件 > 环境变量 > 所有GPU）
    gpu_list_config = config.get('gpu_list', None)
    main_cuda_visible = None  # 初始化变量
    
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
            # 如果没有设置，使用所有GPU
            main_gpu_list = list(range(num_gpus))
            print(f"No GPU configuration found, using all {num_gpus} GPUs: {main_gpu_list}")
    
    # 更新num_gpus为实际使用的GPU数量
    num_gpus = len(main_gpu_list)
    use_multi_gpu = config.get('use_multi_gpu', True) and num_gpus > 1
    
    if use_multi_gpu:
        print(f"Detected {num_gpus} GPUs, using multi-GPU parallel processing")
        # 将分子索引分配给不同的GPU（使用逻辑GPU ID 0到num_gpus-1）
        gpu_assignments = {}
        for idx, sample_idx in enumerate(sample_indices):
            logical_gpu_id = idx % num_gpus  # 逻辑GPU ID（0到num_gpus-1）
            if logical_gpu_id not in gpu_assignments:
                gpu_assignments[logical_gpu_id] = []
            gpu_assignments[logical_gpu_id].append(sample_idx)
        
        print(f"GPU assignments (logical GPU ID -> sample count): {[(gpu_id, len(indices)) for gpu_id, indices in gpu_assignments.items()]}")
        
        # 准备多进程参数
        nf_pretrained_path = config.get("nf_pretrained_path") if field_mode == 'nf_field' else None
        
        # 为每个field_method和每个GPU创建任务
        all_tasks = []
        for field_method in field_methods:
            method_config = config_dict.copy()
            method_config['converter']['gradient_field_method'] = field_method
            
            for logical_gpu_id, gpu_sample_indices in gpu_assignments.items():
                # 获取对应的物理GPU ID
                if main_cuda_visible:
                    physical_gpu_id = main_gpu_list[logical_gpu_id]
                else:
                    physical_gpu_id = logical_gpu_id
                
                for sample_idx in gpu_sample_indices:
                    all_tasks.append((
                        sample_idx, logical_gpu_id, physical_gpu_id, method_config, field_mode, field_method,
                        str(output_dir), str(csv_path), str(mol_save_dir), 
                        config.dset.data_dir, config.dset.dset_name, config.dset.elements,
                        nf_pretrained_path, nf_pretrained_path, config_dict
                    ))
        
        # 使用多进程并行处理
        # 设置multiprocessing的start method为'spawn'，因为CUDA不支持fork模式
        if multiprocessing.get_start_method(allow_none=True) != 'spawn':
            multiprocessing.set_start_method('spawn', force=True)
        
        num_workers_config = config.get('num_workers', num_gpus)
        if num_workers_config is None:
            num_workers_config = num_gpus
        num_workers = min(num_gpus, len(all_tasks), num_workers_config)
        print(f"Using {num_workers} worker processes for parallel processing")
        print(f"Total tasks: {len(all_tasks)}")
        
        # 统计每个GPU的任务数量
        gpu_task_counts = {}
        for task in all_tasks[:10]:  # 只检查前10个任务
            logical_gpu_id = task[1]
            physical_gpu_id = task[2]
            if logical_gpu_id not in gpu_task_counts:
                gpu_task_counts[logical_gpu_id] = {'physical': physical_gpu_id, 'count': 0}
            gpu_task_counts[logical_gpu_id]['count'] += 1
        print(f"Task distribution (first 10 tasks): {gpu_task_counts}")
        
        # 统计所有任务的GPU分布
        all_gpu_counts = {}
        for task in all_tasks:
            logical_gpu_id = task[1]
            all_gpu_counts[logical_gpu_id] = all_gpu_counts.get(logical_gpu_id, 0) + 1
        print(f"Task distribution (all tasks): {all_gpu_counts}")
        
        # 为每个worker创建独立的进程组，每个组处理一个GPU的任务
        # 我们需要按GPU分组任务，然后为每个GPU创建一个worker进程
        from collections import defaultdict
        tasks_by_gpu = defaultdict(list)
        for task in all_tasks:
            logical_gpu_id = task[1]
            tasks_by_gpu[logical_gpu_id].append(task)
        
        print(f"Tasks grouped by GPU: {[(gpu_id, len(tasks)) for gpu_id, tasks in tasks_by_gpu.items()]}")
        
        # 使用Process而不是Pool，这样可以在每个进程启动时设置CUDA_VISIBLE_DEVICES
        from multiprocessing import Process, Queue
        import queue as queue_module
        
        result_queue = Queue()
        processes = []
        
        # 为每个GPU创建worker进程（使用模块级别的worker_process函数）
        for logical_gpu_id, tasks in tasks_by_gpu.items():
            physical_gpu_id = main_gpu_list[logical_gpu_id] if main_cuda_visible else logical_gpu_id
            # 通过环境变量传递GPU ID，这样子进程在重新执行脚本时会先设置CUDA_VISIBLE_DEVICES
            # 注意：在Python 3.9之前，Process不支持env参数，所以我们需要在worker_process中设置
            # 但我们可以通过修改os.environ来实现（虽然这不是最佳实践，但在spawn模式下可以工作）
            original_env = os.environ.get("_WORKER_GPU_ID", None)
            os.environ["_WORKER_GPU_ID"] = str(physical_gpu_id)
            try:
                p = Process(target=worker_process, args=(logical_gpu_id, physical_gpu_id, tasks, result_queue))
                p.start()
                processes.append(p)
            finally:
                # 恢复原始环境变量（如果存在）
                if original_env is not None:
                    os.environ["_WORKER_GPU_ID"] = original_env
                elif "_WORKER_GPU_ID" in os.environ:
                    del os.environ["_WORKER_GPU_ID"]
        
        # 收集结果并实时写入CSV（在主进程中，避免多进程写入冲突）
        results = []
        completed = 0
        total_tasks = len(all_tasks)
        with tqdm(total=total_tasks, desc="Processing molecules") as pbar:
            while completed < total_tasks:
                try:
                    result = result_queue.get(timeout=1)
                    results.append(result)
                    completed += 1
                    
                    # 实时写入CSV（在主进程中，避免多进程写入冲突）
                    if result.get('success', False) and 'result_row' in result:
                        result_df = pd.DataFrame([result['result_row']])
                        result_df.to_csv(csv_path, mode='a', header=False, index=False)
                    elif 'error_row' in result:
                        error_df = pd.DataFrame([result['error_row']])
                        error_df.to_csv(csv_path, mode='a', header=False, index=False)
                    
                    pbar.update(1)
                except queue_module.Empty:
                    # 检查进程是否还在运行
                    if all(not p.is_alive() for p in processes):
                        break
        
        # 等待所有进程完成
        for p in processes:
            p.join()
        
        # 统计结果
        success_count = sum(1 for r in results if r.get('success', False))
        print(f"Successfully processed {success_count}/{len(results)} molecules")
        
    else:
        # 单GPU或单进程处理（保持原有逻辑）
        print(f"Using single GPU/process mode")
        batch_size = config.get('batch_size', 1)
        print(f"Using batch_size={batch_size} for molecular-level parallelization")
        
        # 批次处理分子
        total_batches = (len(sample_indices) + batch_size - 1) // batch_size
        for batch_idx in tqdm(range(total_batches), desc="Processing batches"):
            # 计算当前批次的实际大小和索引
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(sample_indices))
            current_batch_indices = sample_indices[start_idx:end_idx]
            current_batch_size = len(current_batch_indices)
            
            # 批次加载数据
            batch_samples = []
            batch_gt_coords = []
            batch_gt_types = []
            batch_codes_list = []
            
            for i, sample_idx in enumerate(current_batch_indices):
                sample = dataset[sample_idx]
                batch_samples.append(sample)
                batch_gt_coords.append(sample.pos)
                batch_gt_types.append(sample.x)
            
            # 对于nf_field模式，批次编码
            if field_mode == 'nf_field':
                from torch_geometric.data import Data, Batch
                batch_data_list = []
                for sample in batch_samples:
                    data = Data(
                        pos=sample.pos,
                        x=sample.x,
                        batch=torch.zeros(sample.pos.shape[0], dtype=torch.long, device=sample.pos.device)
                    )
                    batch_data_list.append(data)
                
                batch_pyg = Batch.from_data_list(batch_data_list)
                batch_pyg = batch_pyg.cuda()
                
                with torch.no_grad():
                    batch_codes = encoder(batch_pyg)  # [batch_size, grid_size**3, code_dim]
                    batch_codes_list = [batch_codes[i:i+1] for i in range(current_batch_size)]
            
            # 对每个field_method处理
            for field_method in field_methods:
                try:
                    # 创建converter with specific field method
                    method_config = config_dict.copy()
                    method_config['converter']['gradient_field_method'] = field_method
                    converter = create_gnf_converter(method_config)
                    
                    # 批次处理当前batch的所有分子
                    for i, sample_idx in enumerate(current_batch_indices):
                        try:
                            gt_coords = batch_gt_coords[i]
                            gt_types = batch_gt_types[i]
                            
                            # 初始化decoder和codes变量
                            codes = None
                        
                            if field_mode == 'gt_field':
                                # Create dummy codes and decoder
                                grid_size = config.get('dset', {}).get('grid_size', 9)
                                code_dim = config.get('encoder', {}).get('code_dim', 128)
                                dummy_codes = torch.randn(1, grid_size**3, code_dim, device=gt_coords.device)
                                
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
                                decoder = dummy_decoder
                                codes = dummy_codes
                                
                            elif field_mode == 'nf_field':
                                decoder = decoder  # 使用全局decoder
                                codes = batch_codes_list[i]  # 使用批次编码的对应分子
                            
                            # 从配置中读取保存参数
                            autoregressive_config = config_dict.get("converter", {}).get("autoregressive_clustering", {})
                            save_clustering_history = autoregressive_config.get("enable_clustering_history", False)
                            save_gradient_ascent_sdf = autoregressive_config.get("save_gradient_ascent_sdf", False)
                            enable_timing = autoregressive_config.get("enable_timing", False)
                            
                            # 构建gnf2mol调用参数
                            gnf2mol_kwargs = {
                                "decoder": decoder,
                                "codes": codes,
                                "sample_id": sample_idx
                            }
                            
                            if save_clustering_history:
                                gnf2mol_kwargs["save_clustering_history"] = True
                                gnf2mol_kwargs["clustering_history_dir"] = str(output_dir / "clustering_history")
                            
                            if save_gradient_ascent_sdf:
                                gnf2mol_kwargs["save_gradient_ascent_sdf"] = True
                                gnf2mol_kwargs["gradient_ascent_sdf_dir"] = str(output_dir / "gradient_ascent_sdf")
                                gnf2mol_kwargs["gradient_ascent_sdf_interval"] = autoregressive_config.get("gradient_ascent_sdf_interval", 100)
                            
                            if enable_timing:
                                gnf2mol_kwargs["enable_timing"] = True
                            
                            recon_coords, recon_types = converter.gnf2mol(**gnf2mol_kwargs)
                            
                            # 处理结果
                            recon_coords_device = recon_coords[0].to(gt_coords.device)
                            rmsd = compute_rmsd(gt_coords, recon_coords_device)
                            atom_count_mismatch = gt_coords.shape[0] != recon_coords[0].shape[0]
                            original_size = gt_coords.shape[0]
                            
                            # Save result immediately to CSV
                            result_row = {
                                'sample_idx': sample_idx,
                                'field_mode': field_mode,
                                'field_method': field_method,
                                'rmsd': rmsd,
                                'size': original_size,
                                'atom_count_mismatch': atom_count_mismatch
                            }
                            
                            # 为nf_field模式添加原子统计信息
                            if field_mode == 'nf_field':
                                elements = config.dset.elements
                                for j, element in enumerate(elements):
                                    count = (gt_types == j).sum().item()
                                    result_row[f'gt_{element}_count'] = count
                                
                                recon_types_device = recon_types[0].to(gt_coords.device)
                                valid_recon_types = recon_types_device[recon_types_device != -1]
                                for j, element in enumerate(elements):
                                    count = (valid_recon_types == j).sum().item()
                                    result_row[f'recon_{element}_count'] = count
                            
                            # Append to CSV file immediately
                            result_df = pd.DataFrame([result_row])
                            result_df.to_csv(csv_path, mode='a', header=False, index=False)
                            
                            # Save molecular coordinates and types
                            mol_file = mol_save_dir / f"sample_{sample_idx:04d}_{field_method}.npz"
                            np.savez(mol_file, 
                                    coords=recon_coords_device.cpu().numpy(),
                                    types=gt_types.cpu().numpy(),
                                    gt_coords=gt_coords.cpu().numpy(),
                                    rmsd=rmsd)
                        
                        except Exception as e:
                            print(f"Error processing sample {sample_idx} with {field_method}: {e}")
                            import traceback
                            traceback.print_exc()
                            # Save error result
                            error_row = {
                                'sample_idx': sample_idx,
                                'field_mode': field_mode,
                                'field_method': field_method,
                                'rmsd': float('inf'),
                                'size': gt_coords.shape[0],
                                'atom_count_mismatch': True
                            }
                            
                            if field_mode == 'nf_field':
                                elements = config.dset.elements
                                for j, element in enumerate(elements):
                                    count = (gt_types == j).sum().item()
                                    error_row[f'gt_{element}_count'] = count
                                for element in elements:
                                    error_row[f'recon_{element}_count'] = 0
                            
                            error_df = pd.DataFrame([error_row])
                            error_df.to_csv(csv_path, mode='a', header=False, index=False)
                
                except Exception as e:
                    print(f"Error processing batch {batch_idx} with {field_method}: {e}")
                import traceback
                traceback.print_exc()
                # Save error results for all molecules in the batch
                for i, sample_idx in enumerate(current_batch_indices):
                    gt_coords = batch_gt_coords[i]
                    gt_types = batch_gt_types[i]
                    error_row = {
                        'sample_idx': sample_idx,
                        'field_mode': field_mode,
                        'field_method': field_method,
                        'rmsd': float('inf'),
                        'size': gt_coords.shape[0],
                        'atom_count_mismatch': True
                    }
                    
                    if field_mode == 'nf_field':
                        elements = config.dset.elements
                        for j, element in enumerate(elements):
                            count = (gt_types == j).sum().item()
                            error_row[f'gt_{element}_count'] = count
                        for element in elements:
                            error_row[f'recon_{element}_count'] = 0
                    
                    error_df = pd.DataFrame([error_row])
                    error_df.to_csv(csv_path, mode='a', header=False, index=False)
    
    # Load and analyze results from CSV file
    print(f"Results saved to: {csv_path}")
    
    # Read results from CSV for summary
    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} results from CSV file")
        
        # For gt_field and nf_field modes, show reconstruction statistics
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