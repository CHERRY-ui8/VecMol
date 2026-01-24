#!/usr/bin/env python3
"""
Neural Field场质量评估脚本

评估neural field重建的场与原始场之间的质量差异。
计算MSE和PSNR指标，不需要进行分子重建。
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import random
import numpy as np
import torch
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, Dict, List
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端

# 导入项目模块
from funcmol.dataset.dataset_field import FieldDataset, create_gnf_converter
from funcmol.utils.utils_nf import load_neural_field
from funcmol.utils.constants import PADDING_INDEX
from funcmol.models.funcmol import create_funcmol
from funcmol.utils.utils_fm import load_checkpoint_fm

# 设置项目根目录（funcmol目录）
PROJECT_ROOT = Path(__file__).parent.parent


def compute_mse(pred_field: torch.Tensor, gt_field: torch.Tensor) -> float:
    """
    计算均方误差 (MSE)
    
    Args:
        pred_field: 预测场 [n_points, n_atom_types, 3]
        gt_field: 真实场 [n_points, n_atom_types, 3]
        
    Returns:
        MSE值
    """
    mse = torch.mean((pred_field - gt_field) ** 2).item()
    return mse


def compute_psnr(mse: float, max_val: Optional[float] = None) -> float:
    """
    计算峰值信噪比 (PSNR)
    
    Args:
        mse: 均方误差
        max_val: 信号的最大值。如果为None，则使用gt_field的最大值
        
    Returns:
        PSNR值（单位：dB）
    """
    if mse == 0:
        return float('inf')
    
    if max_val is None:
        # 使用默认的最大值，或者可以从gt_field计算
        max_val = 1.0  # 假设场的最大值约为1.0，可以根据实际情况调整
    
    psnr = 20 * torch.log10(torch.tensor(max_val)) - 10 * torch.log10(torch.tensor(mse))
    return psnr.item()


def compute_field_magnitude_mse(pred_field: torch.Tensor, gt_field: torch.Tensor) -> float:
    """
    计算向量场模长的MSE
    
    Args:
        pred_field: 预测场 [n_points, n_atom_types, 3]
        gt_field: 真实场 [n_points, n_atom_types, 3]
        
    Returns:
        模长MSE值
    """
    pred_mag = torch.norm(pred_field, dim=-1)  # [n_points, n_atom_types]
    gt_mag = torch.norm(gt_field, dim=-1)  # [n_points, n_atom_types]
    mse = torch.mean((pred_mag - gt_mag) ** 2).item()
    return mse


def compute_rmsd(pred_field: torch.Tensor, gt_field: torch.Tensor) -> float:
    """
    计算梯度场的RMSD (Root Mean Square Deviation)
    
    Args:
        pred_field: 预测场 [n_points, n_atom_types, 3]
        gt_field: 真实场 [n_points, n_atom_types, 3]
        
    Returns:
        RMSD值
    """
    # 计算每个点的平方误差，然后取平均，再开方
    squared_diff = (pred_field - gt_field) ** 2  # [n_points, n_atom_types, 3]
    mse = torch.mean(squared_diff).item()  # 对所有维度求平均
    rmsd = np.sqrt(mse)
    return rmsd


def compute_distance_to_nearest_atom(
    query_points: torch.Tensor, 
    atom_positions: torch.Tensor
) -> torch.Tensor:
    """
    计算每个查询点到最近原子的距离
    
    Args:
        query_points: 查询点坐标 [n_points, 3]
        atom_positions: 原子坐标 [n_atoms, 3]
        
    Returns:
        每个查询点到最近原子的距离 [n_points]
    """
    if len(atom_positions) == 0:
        # 如果没有原子，返回一个很大的距离值
        return torch.full((len(query_points),), float('inf'), device=query_points.device)
    
    # 计算所有查询点到所有原子的距离
    distances = torch.cdist(query_points, atom_positions)  # [n_points, n_atoms]
    # 找到每个查询点到最近原子的距离
    min_distances = distances.min(dim=-1)[0]  # [n_points]
    return min_distances


def compute_rmsd_by_distance(
    pred_field: torch.Tensor,
    gt_field: torch.Tensor,
    query_points: torch.Tensor,
    atom_positions: torch.Tensor,
    distance_bins: Optional[List[float]] = None
) -> Tuple[Dict[str, float], Dict[str, int], torch.Tensor]:
    """
    按距离最近原子的距离分组计算RMSD loss
    
    Args:
        pred_field: 预测场 [n_points, n_atom_types, 3]
        gt_field: 真实场 [n_points, n_atom_types, 3]
        query_points: 查询点坐标 [n_points, 3]
        atom_positions: 原子坐标 [n_atoms, 3]
        distance_bins: 距离区间的边界，例如 [0, 0.5, 1.0, 1.5, 2.0, float('inf')]
                      如果为None，使用默认区间
        
    Returns:
        (rmsd_by_bin, counts_by_bin, distances)
        - rmsd_by_bin: 每个距离区间的RMSD值字典
        - counts_by_bin: 每个距离区间的点数字典
        - distances: 每个查询点到最近原子的距离 [n_points]
    """
    # 计算每个查询点到最近原子的距离
    distances = compute_distance_to_nearest_atom(query_points, atom_positions)
    
    # 默认距离区间（更细的划分）
    if distance_bins is None:
        distance_bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0, float('inf')]
    
    # 计算每个点的平方误差
    squared_diff = (pred_field - gt_field) ** 2  # [n_points, n_atom_types, 3]
    # 对原子类型和空间维度求平均，得到每个点的MSE
    pointwise_mse = torch.mean(squared_diff, dim=(-2, -1))  # [n_points]
    
    rmsd_by_bin = {}
    counts_by_bin = {}
    
    # 对每个距离区间计算RMSD
    for i in range(len(distance_bins) - 1):
        bin_min = distance_bins[i]
        bin_max = distance_bins[i + 1]
        
        # 找到属于这个区间的点
        if bin_max == float('inf'):
            mask = (distances >= bin_min)
        else:
            mask = (distances >= bin_min) & (distances < bin_max)
        
        count = mask.sum().item()
        counts_by_bin[f"{bin_min:.2f}-{bin_max:.2f}" if bin_max != float('inf') else f"{bin_min:.2f}+"] = count
        
        if count > 0:
            # 计算这个区间的MSE
            bin_mse = pointwise_mse[mask].mean().item()
            bin_rmsd = np.sqrt(bin_mse)
            rmsd_by_bin[f"{bin_min:.2f}-{bin_max:.2f}" if bin_max != float('inf') else f"{bin_min:.2f}+"] = bin_rmsd
        else:
            rmsd_by_bin[f"{bin_min:.2f}-{bin_max:.2f}" if bin_max != float('inf') else f"{bin_min:.2f}+"] = 0.0
    
    return rmsd_by_bin, counts_by_bin, distances


def plot_rmsd_by_distance(
    all_rmsd_by_distance: List[Dict[str, float]],
    all_counts_by_distance: List[Dict[str, int]],
    output_path: Path,
    title: str = "RMSD Loss by Distance to Nearest Atom"
) -> None:
    """
    绘制不同距离区间的RMSD loss分布图
    
    Args:
        all_rmsd_by_distance: 所有样本的距离分组RMSD列表
        all_counts_by_distance: 所有样本的距离分组点数列表
        output_path: 输出图片路径
        title: 图表标题
    """
    # 收集所有距离区间的名称（应该是一致的）
    if len(all_rmsd_by_distance) == 0:
        print("警告: 没有数据可以绘制")
        return
    
    bin_names = list(all_rmsd_by_distance[0].keys())
    
    # 计算每个区间的平均RMSD和总点数
    avg_rmsd_by_bin = {}
    total_counts_by_bin = {}
    
    for bin_name in bin_names:
        rmsd_values = []
        counts = []
        for rmsd_dict, count_dict in zip(all_rmsd_by_distance, all_counts_by_distance):
            if bin_name in rmsd_dict and count_dict.get(bin_name, 0) > 0:
                rmsd_values.append(rmsd_dict[bin_name])
                counts.append(count_dict[bin_name])
        
        if len(rmsd_values) > 0:
            # 使用加权平均（按点数加权）
            total_count = sum(counts)
            if total_count > 0:
                weighted_rmsd = sum(r * c for r, c in zip(rmsd_values, counts)) / total_count
                avg_rmsd_by_bin[bin_name] = weighted_rmsd
                total_counts_by_bin[bin_name] = total_count
            else:
                avg_rmsd_by_bin[bin_name] = 0.0
                total_counts_by_bin[bin_name] = 0
        else:
            avg_rmsd_by_bin[bin_name] = 0.0
            total_counts_by_bin[bin_name] = 0
    
    # 创建图表（只绘制RMSD图，不绘制点数分布图）
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    # RMSD vs 距离区间
    bin_names_sorted = sorted(bin_names, key=lambda x: float(x.split('-')[0]) if '-' in x else float(x.replace('+', '')))
    rmsd_values = [avg_rmsd_by_bin[bn] for bn in bin_names_sorted]
    counts = [total_counts_by_bin[bn] for bn in bin_names_sorted]
    
    # 绘制柱状图
    bars = ax.bar(range(len(bin_names_sorted)), rmsd_values, alpha=0.7, color='steelblue')
    ax.set_xlabel('Distance to Nearest Atom (Å)', fontsize=12)
    ax.set_ylabel('RMSD Loss', fontsize=12)
    ax.set_title('Average RMSD Loss by Distance', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(bin_names_sorted)))
    ax.set_xticklabels(bin_names_sorted, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    
    # 不添加数值标签，保持图表简洁
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"图表已保存到: {output_path}")


def compute_relative_error_by_distance(
    pred_field: torch.Tensor,
    gt_field: torch.Tensor,
    query_points: torch.Tensor,
    atom_positions: torch.Tensor,
    distance_bins: Optional[List[float]] = None
) -> Tuple[Dict[str, float], Dict[str, int], torch.Tensor]:
    """
    按距离最近原子的距离分组计算相对误差 (RMSD / GT field magnitude)
    
    Args:
        pred_field: 预测场 [n_points, n_atom_types, 3]
        gt_field: 真实场 [n_points, n_atom_types, 3]
        query_points: 查询点坐标 [n_points, 3]
        atom_positions: 原子坐标 [n_atoms, 3]
        distance_bins: 距离区间的边界，例如 [0, 0.5, 1.0, 1.5, 2.0, float('inf')]
                      如果为None，使用默认区间
        
    Returns:
        (relative_error_by_bin, counts_by_bin, distances)
        - relative_error_by_bin: 每个距离区间的相对误差值字典
        - counts_by_bin: 每个距离区间的点数字典
        - distances: 每个查询点到最近原子的距离 [n_points]
    """
    # 计算每个查询点到最近原子的距离
    distances = compute_distance_to_nearest_atom(query_points, atom_positions)
    
    # 默认距离区间（更细的划分）
    if distance_bins is None:
        distance_bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0, float('inf')]
    
    # 计算每个点的平方误差
    squared_diff = (pred_field - gt_field) ** 2  # [n_points, n_atom_types, 3]
    # 对原子类型和空间维度求平均，得到每个点的MSE
    pointwise_mse = torch.mean(squared_diff, dim=(-2, -1))  # [n_points]
    pointwise_rmsd = torch.sqrt(pointwise_mse)  # [n_points]
    
    # 计算每个点的GT field模长（对所有原子类型和空间维度求平均后的模长）
    gt_field_magnitude = torch.norm(gt_field, dim=-1)  # [n_points, n_atom_types]
    gt_field_magnitude_mean = torch.mean(gt_field_magnitude, dim=-1)  # [n_points] - 对原子类型求平均
    
    # 计算相对误差：RMSD / GT field magnitude
    # 避免除零，添加小的epsilon
    epsilon = 1e-8
    relative_error = pointwise_rmsd / (gt_field_magnitude_mean + epsilon)  # [n_points]
    
    relative_error_by_bin = {}
    counts_by_bin = {}
    
    # 对每个距离区间计算相对误差
    for i in range(len(distance_bins) - 1):
        bin_min = distance_bins[i]
        bin_max = distance_bins[i + 1]
        
        # 找到属于这个区间的点
        if bin_max == float('inf'):
            mask = (distances >= bin_min)
        else:
            mask = (distances >= bin_min) & (distances < bin_max)
        
        count = mask.sum().item()
        counts_by_bin[f"{bin_min:.2f}-{bin_max:.2f}" if bin_max != float('inf') else f"{bin_min:.2f}+"] = count
        
        if count > 0:
            # 计算这个区间的平均相对误差
            bin_relative_error = relative_error[mask].mean().item()
            relative_error_by_bin[f"{bin_min:.2f}-{bin_max:.2f}" if bin_max != float('inf') else f"{bin_min:.2f}+"] = bin_relative_error
        else:
            relative_error_by_bin[f"{bin_min:.2f}-{bin_max:.2f}" if bin_max != float('inf') else f"{bin_min:.2f}+"] = 0.0
    
    return relative_error_by_bin, counts_by_bin, distances


def plot_relative_error_by_distance(
    all_relative_error_by_distance: List[Dict[str, float]],
    all_counts_by_distance: List[Dict[str, int]],
    output_path: Path,
    title: str = "Relative Error (RMSD / GT Field Magnitude) by Distance"
) -> None:
    """
    绘制不同距离区间的相对误差分布图
    
    Args:
        all_relative_error_by_distance: 所有样本的距离分组相对误差列表
        all_counts_by_distance: 所有样本的距离分组点数列表
        output_path: 输出图片路径
        title: 图表标题
    """
    # 收集所有距离区间的名称（应该是一致的）
    if len(all_relative_error_by_distance) == 0:
        print("警告: 没有数据可以绘制")
        return
    
    bin_names = list(all_relative_error_by_distance[0].keys())
    
    # 计算每个区间的平均相对误差和总点数
    avg_relative_error_by_bin = {}
    total_counts_by_bin = {}
    
    for bin_name in bin_names:
        relative_error_values = []
        counts = []
        for rel_err_dict, count_dict in zip(all_relative_error_by_distance, all_counts_by_distance):
            if bin_name in rel_err_dict and count_dict.get(bin_name, 0) > 0:
                relative_error_values.append(rel_err_dict[bin_name])
                counts.append(count_dict[bin_name])
        
        if len(relative_error_values) > 0:
            # 使用加权平均（按点数加权）
            total_count = sum(counts)
            if total_count > 0:
                weighted_relative_error = sum(r * c for r, c in zip(relative_error_values, counts)) / total_count
                avg_relative_error_by_bin[bin_name] = weighted_relative_error
                total_counts_by_bin[bin_name] = total_count
            else:
                avg_relative_error_by_bin[bin_name] = 0.0
                total_counts_by_bin[bin_name] = 0
        else:
            avg_relative_error_by_bin[bin_name] = 0.0
            total_counts_by_bin[bin_name] = 0
    
    # 创建图表
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    # 相对误差 vs 距离区间
    bin_names_sorted = sorted(bin_names, key=lambda x: float(x.split('-')[0]) if '-' in x else float(x.replace('+', '')))
    relative_error_values = [avg_relative_error_by_bin[bn] for bn in bin_names_sorted]
    
    # 绘制柱状图
    ax.bar(range(len(bin_names_sorted)), relative_error_values, alpha=0.7, color='coral')
    ax.set_xlabel('Distance to Nearest Atom (Å)', fontsize=12)
    ax.set_ylabel('Relative Error (RMSD / GT Magnitude)', fontsize=12)
    ax.set_title('Relative Error by Distance', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(bin_names_sorted)))
    ax.set_xticklabels(bin_names_sorted, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    
    # 不添加数值标签，保持图表简洁
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"图表已保存到: {output_path}")


@hydra.main(config_path="../configs", config_name="field_quality_qm9", version_base=None)
def main(config: DictConfig) -> None:
    """
    主函数：评估neural field场的质量
    
    Args:
        config: Hydra配置对象
    """
    # 设置随机种子
    seed = config.get('seed', 1234)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # 转换配置为字典
    config_dict = OmegaConf.to_container(config, resolve=True)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 1. 加载neural field模型
    print("\n" + "="*80)
    print("加载Neural Field模型")
    print("="*80)
    nf_checkpoint_path = OmegaConf.select(config, "nf_pretrained_path")
    if not nf_checkpoint_path:
        raise ValueError("配置文件中必须提供 nf_pretrained_path")
    
    nf_checkpoint_path = str(nf_checkpoint_path)
    
    print(f"加载checkpoint: {nf_checkpoint_path}")
    if not os.path.exists(nf_checkpoint_path):
        raise FileNotFoundError(f"Checkpoint文件不存在: {nf_checkpoint_path}\n请检查配置文件中的路径是否正确。")
    
    checkpoint = torch.load(nf_checkpoint_path, map_location='cpu', weights_only=False)
    nf_config = checkpoint.get("hyper_parameters", {})
    
    # 创建checkpoint字典
    nf_checkpoint = {
        "config": nf_config,
        "enc_state_dict": checkpoint["enc_state_dict"],
        "dec_state_dict": checkpoint["dec_state_dict"]
    }
    
    # 加载encoder和decoder
    encoder, decoder = load_neural_field(nf_checkpoint, None)
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    encoder.eval()
    decoder.eval()
    print("模型加载完成")
    
    # 可选：加载FuncMol denoiser用于评估denoised_field
    funcmol = None
    code_stats = None
    evaluate_denoised_field = config.get("evaluate_denoised_field", False)
    
    if evaluate_denoised_field:
        print("\n" + "="*80)
        print("加载FuncMol Denoiser模型")
        print("="*80)
        fm_checkpoint_path = OmegaConf.select(config, "fm_pretrained_path")
        if not fm_checkpoint_path:
            raise ValueError("评估denoised_field需要提供 fm_pretrained_path")
        
        fm_checkpoint_path = str(fm_checkpoint_path)
        
        if not os.path.exists(fm_checkpoint_path):
            raise FileNotFoundError(f"FuncMol checkpoint文件不存在: {fm_checkpoint_path}\n请检查配置文件中的路径是否正确。")
        
        # 从当前配置中构建FuncMol配置
        # 使用当前config中的相关配置，而不是从其他配置文件加载
        funcmol_config = {
            "smooth_sigma": config_dict.get("smooth_sigma", 0.0),
            "diffusion_method": config_dict.get("diffusion_method", "new_x0"),
            "denoiser": config_dict.get("denoiser", {}),
            "ddpm": config_dict.get("ddpm", {"num_timesteps": 1000, "use_time_weight": True}),
            "decoder": config_dict.get("decoder", {}),
            "dset": config_dict.get("dset", {}),
            "encoder": config_dict.get("encoder", {}),
        }
        
        # 创建FuncMol模型
        funcmol = create_funcmol(funcmol_config)
        funcmol = funcmol.to(device)
        
        # 加载checkpoint
        funcmol, code_stats = load_checkpoint_fm(funcmol, fm_checkpoint_path)
        funcmol.eval()
        
        # 设置decoder的code_stats（如果需要）
        if hasattr(decoder, 'set_code_stats') and code_stats is not None:
            decoder.set_code_stats(code_stats)
        
        print("FuncMol Denoiser加载完成，将使用完全去噪（从随机噪声开始）生成denoised_field")
    
    # 2. 创建GNFConverter
    print("\n" + "="*80)
    print("创建GNFConverter")
    print("="*80)
    converter = create_gnf_converter(config_dict)
    print("GNFConverter创建完成")
    
    # 3. 加载验证集数据
    print("\n" + "="*80)
    print("加载验证集数据")
    print("="*80)
    
    # 创建FieldDataset
    # 将相对路径转换为绝对路径
    data_dir = config_dict.get("dset", {}).get("data_dir", "dataset/data")
    if not os.path.isabs(data_dir):
        # PROJECT_ROOT 指向 funcmol 目录，所以直接拼接 dataset/data
        data_dir = str(PROJECT_ROOT / data_dir)
    
    # 从dset配置中读取atom_distance_threshold，如果没有则使用默认值0.5
    atom_distance_threshold = config_dict.get("dset", {}).get("atom_distance_threshold", 0.5)
    # 从dset配置中读取targeted_sampling_ratio，如果没有则使用默认值0.5（与train_nf保持一致）
    targeted_sampling_ratio = config_dict.get("dset", {}).get("targeted_sampling_ratio", 0.5)
    
    dataset = FieldDataset(
        gnf_converter=converter,
        dset_name=config_dict.get("dset", {}).get("dset_name", "qm9"),
        data_dir=data_dir,
        elements=config_dict.get("dset", {}).get("elements", None),
        split=config.get("split", "val"),
        n_points=config_dict.get("dset", {}).get("n_points", 4000),
        rotate=False,  # 评估时不使用旋转
        resolution=config_dict.get("dset", {}).get("resolution", 0.25),
        grid_dim=config_dict.get("dset", {}).get("grid_dim", 32),
        sample_full_grid=config.get("sample_full_grid", False),
        targeted_sampling_ratio=targeted_sampling_ratio,  # 使用与train_nf相同的采样比例
        atom_distance_threshold=atom_distance_threshold,
    )
    
    # 限制评估的样本数量
    num_samples = config.get("num_samples", None)
    total_samples = len(dataset)  # 保存原始数据集大小
    if num_samples is not None and num_samples < total_samples:
        # 随机选择样本
        indices = list(range(total_samples))
        random.Random(seed).shuffle(indices)
        selected_indices = indices[:num_samples]
        # 创建子集（通过修改field_idxs）
        dataset.field_idxs = torch.tensor(selected_indices)
        print(f"从 {total_samples} 个样本中选择 {num_samples} 个进行评估")
    else:
        num_samples = total_samples
        print(f"使用全部 {num_samples} 个样本进行评估")
    
    # 创建DataLoader
    from torch_geometric.loader import DataLoader
    batch_size = config.get("batch_size", 1)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # 评估时使用单进程
        pin_memory=True,
    )
    
    # 4. 评估每个样本
    print("\n" + "="*80)
    print("开始场质量评估")
    print("="*80)
    
    all_results = []
    all_rmsd_by_distance = []  # 存储所有样本的predicted field距离分组RMSD
    all_counts_by_distance = []  # 存储所有样本的距离分组点数
    all_relative_error_by_distance = []  # 存储所有样本的predicted field距离分组相对误差
    
    # 如果评估denoised_field，也收集其数据
    all_denoised_rmsd_by_distance = []  # 存储所有样本的denoised field距离分组RMSD
    all_denoised_relative_error_by_distance = []  # 存储所有样本的denoised field距离分组相对误差
    all_denoised_results = []  # 存储denoised field的评估结果
    
    # 从配置中读取距离区间，如果没有则使用默认值
    distance_bins = config.get("distance_bins", None)
    if distance_bins is not None:
        distance_bins = [float(x) for x in distance_bins]
    
    for batch_idx, batch in enumerate(tqdm(loader, desc="评估进度")):
        batch = batch.to(device)
        
        # 获取查询点
        query_points = batch.xs  # [N_total_points, 3] 或 [B, n_points, 3]
        n_points = config_dict.get("dset", {}).get("n_points", 4000)
        B = batch.num_graphs
        
        # 处理batch维度
        if query_points.dim() == 2:
            # [N_total_points, 3] -> [B, n_points, 3]
            query_points = query_points.view(B, n_points, 3)
        elif query_points.dim() == 3:
            pass  # 已经是 [B, n_points, 3]
        else:
            raise ValueError(f"Unexpected query_points shape: {query_points.shape}")
        
        # 获取ground truth场（从数据集中）
        gt_field = batch.target_field  # [N_total_points, n_atom_types, 3] 或 [B, n_points, n_atom_types, 3]
        if gt_field.dim() == 3:
            # [N_total_points, n_atom_types, 3] -> [B, n_points, n_atom_types, 3]
            gt_field = gt_field.view(B, n_points, -1, 3)
        elif gt_field.dim() == 4:
            pass  # 已经是 [B, n_points, n_atom_types, 3]
        else:
            raise ValueError(f"Unexpected gt_field shape: {gt_field.shape}")
        
        try:
            # 使用encoder生成codes（直接处理整个batch）
            with torch.no_grad():
                codes = encoder(batch)  # [B, grid_size³, code_dim]
            
            # 使用decoder预测场（predicted field）
            with torch.no_grad():
                pred_field = decoder(query_points, codes)  # [B, n_points, n_atom_types, 3]
            
            # 可选：计算denoised_field（从随机噪声完全去噪）
            denoised_field = None
            if evaluate_denoised_field and funcmol is not None:
                with torch.no_grad():
                    # 获取codes的形状信息
                    B_codes = codes.shape[0]
                    n_grid = codes.shape[1]  # grid_size³
                    code_dim = codes.shape[2]
                    
                    # 从随机噪声开始，使用完整的DDPM采样过程生成完全去噪的codes
                    # shape: (batch_size, grid_size³, code_dim)
                    denoised_codes = funcmol.sample_ddpm(
                        shape=(B_codes, n_grid, code_dim),
                        code_stats=code_stats,
                        progress=False,  # 评估时不显示进度条
                        clip_denoised=False
                    )  # [B, grid_size³, code_dim]
                    
                    # 使用denoised_codes生成field
                    denoised_field = decoder(query_points, denoised_codes)  # [B, n_points, n_atom_types, 3]
            
            # 对每个batch中的样本进行处理
            for b in range(B):
                sample_gt_field = gt_field[b]  # [n_points, n_atom_types, 3]
                sample_pred_field = pred_field[b]  # [n_points, n_atom_types, 3]
                sample_query_points = query_points[b]  # [n_points, 3]
                
                # 获取该样本的原子位置
                atom_mask = (batch.x[batch.batch == b] != PADDING_INDEX)
                sample_atom_positions = batch.pos[batch.batch == b][atom_mask]  # [n_atoms, 3]
                
                # 计算predicted field的RMSD
                rmsd = compute_rmsd(sample_pred_field, sample_gt_field)
                
                # 按距离分组计算predicted field的RMSD
                rmsd_by_bin, counts_by_bin, distances = compute_rmsd_by_distance(
                    sample_pred_field,
                    sample_gt_field,
                    sample_query_points,
                    sample_atom_positions,
                    distance_bins=distance_bins
                )
                
                # 按距离分组计算相对误差 (RMSD / GT field magnitude)
                relative_error_by_bin, _, _ = compute_relative_error_by_distance(
                    sample_pred_field,
                    sample_gt_field,
                    sample_query_points,
                    sample_atom_positions,
                    distance_bins=distance_bins
                )
                
                # 保存距离分组数据用于后续绘图
                all_rmsd_by_distance.append(rmsd_by_bin)
                all_counts_by_distance.append(counts_by_bin)
                all_relative_error_by_distance.append(relative_error_by_bin)
                
                # 如果评估denoised_field，也计算其RMSD和相对误差
                denoised_rmsd = None
                denoised_rmsd_by_bin = None
                denoised_relative_error_by_bin = None
                if evaluate_denoised_field and denoised_field is not None:
                    sample_denoised_field = denoised_field[b]  # [n_points, n_atom_types, 3]
                    denoised_rmsd = compute_rmsd(sample_denoised_field, sample_gt_field)
                    
                    # 按距离分组计算denoised field的RMSD
                    denoised_rmsd_by_bin, _, _ = compute_rmsd_by_distance(
                        sample_denoised_field,
                        sample_gt_field,
                        sample_query_points,
                        sample_atom_positions,
                        distance_bins=distance_bins
                    )
                    
                    # 按距离分组计算denoised field的相对误差
                    denoised_relative_error_by_bin, _, _ = compute_relative_error_by_distance(
                        sample_denoised_field,
                        sample_gt_field,
                        sample_query_points,
                        sample_atom_positions,
                        distance_bins=distance_bins
                    )
                    
                    # 保存denoised field的距离分组数据用于后续绘图
                    all_denoised_rmsd_by_distance.append(denoised_rmsd_by_bin)
                    all_denoised_relative_error_by_distance.append(denoised_relative_error_by_bin)
                
                # 获取原子数量
                n_atoms = atom_mask.sum().item()
                
                # 计算最小距离
                distances_np = distances.cpu().numpy()
                min_distance = float(np.min(distances_np))
                
                # 保存predicted field结果
                result = {
                    'sample_idx': batch_idx * batch_size + b,
                    'rmsd': rmsd,  # predicted field的RMSD
                    'min_distance_to_atom': min_distance,
                    'n_points': n_points,
                    'n_atoms': n_atoms,
                }
                # 添加每个距离区间的predicted field RMSD
                for bin_name, bin_rmsd in rmsd_by_bin.items():
                    result[f'rmsd_{bin_name}'] = bin_rmsd
                # 添加每个距离区间的点数
                for bin_name, bin_count in counts_by_bin.items():
                    result[f'count_{bin_name}'] = bin_count
                
                all_results.append(result)
                
                # 如果评估了denoised_field，单独保存denoised field结果
                if evaluate_denoised_field and denoised_rmsd is not None:
                    denoised_result = {
                        'sample_idx': batch_idx * batch_size + b,
                        'rmsd': denoised_rmsd,  # denoised field的RMSD
                        'min_distance_to_atom': min_distance,
                        'n_points': n_points,
                        'n_atoms': n_atoms,
                    }
                    # 添加每个距离区间的denoised field RMSD
                    for bin_name, bin_rmsd in denoised_rmsd_by_bin.items():
                        denoised_result[f'rmsd_{bin_name}'] = bin_rmsd
                    # 添加每个距离区间的点数（与predicted field相同）
                    for bin_name, bin_count in counts_by_bin.items():
                        denoised_result[f'count_{bin_name}'] = bin_count
                    
                    all_denoised_results.append(denoised_result)
                
        except (RuntimeError, ValueError, IndexError) as e:
            print(f"警告: 处理batch {batch_idx} 时出错: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if len(all_results) == 0:
        print("错误: 没有成功评估任何样本")
        return
    
    # 5. 计算统计信息
    print("\n" + "="*80)
    print("评估结果统计")
    print("="*80)
    
    df = pd.DataFrame(all_results)
    
    # 计算RMSD统计
    mean_rmsd = df['rmsd'].mean()
    std_rmsd = df['rmsd'].std()
    
    print(f"总样本数: {len(all_results)}")
    print("\n预测场(Predicted Field) RMSD:")
    print(f"  平均值: {mean_rmsd:.6f} ± {std_rmsd:.6f}")
    print(f"  最小值: {df['rmsd'].min():.6f}")
    print(f"  最大值: {df['rmsd'].max():.6f}")
    
    
    print("\n距离最近原子的距离统计:")
    print(f"  最小距离: {df['min_distance_to_atom'].min():.4f} Å")
    
    # 打印各距离区间的predicted field RMSD统计
    print("\n各距离区间的预测场(Predicted Field) RMSD统计:")
    rmsd_bin_cols = [col for col in df.columns if col.startswith('rmsd_') and not col.startswith('denoised_rmsd_')]
    for col in sorted(rmsd_bin_cols, key=lambda x: float(x.split('_')[1].split('-')[0]) if '-' in x else float(x.split('_')[1].replace('+', ''))):
        bin_name = col.replace('rmsd_', '')
        mean_val = df[col].mean()
        std_val = df[col].std()
        count_col = f'count_{bin_name}'
        if count_col in df.columns:
            total_points = df[count_col].sum()
            print(f"  {bin_name} Å: {mean_val:.6f} ± {std_val:.6f} (总点数: {total_points})")
        else:
            print(f"  {bin_name} Å: {mean_val:.6f} ± {std_val:.6f}")
    
    # 如果评估了denoised_field，打印denoised field的统计信息
    if evaluate_denoised_field and len(all_denoised_results) > 0:
        df_denoised = pd.DataFrame(all_denoised_results)
        mean_denoised_rmsd = df_denoised['rmsd'].mean()
        std_denoised_rmsd = df_denoised['rmsd'].std()
        print("\n去噪场(Denoised Field) RMSD:")
        print(f"  平均值: {mean_denoised_rmsd:.6f} ± {std_denoised_rmsd:.6f}")
        print(f"  最小值: {df_denoised['rmsd'].min():.6f}")
        print(f"  最大值: {df_denoised['rmsd'].max():.6f}")
        print(f"\nRMSD差异 (Denoised - Predicted):")
        diff = df_denoised['rmsd'].values - df['rmsd'].values
        print(f"  平均值: {diff.mean():.6f} ± {diff.std():.6f}")
        print(f"  最小值: {diff.min():.6f}")
        print(f"  最大值: {diff.max():.6f}")
        
        print("\n各距离区间的去噪场(Denoised Field) RMSD统计:")
        denoised_rmsd_bin_cols = [col for col in df_denoised.columns if col.startswith('rmsd_')]
        for col in sorted(denoised_rmsd_bin_cols, key=lambda x: float(x.split('_')[1].split('-')[0]) if '-' in x else float(x.split('_')[1].replace('+', ''))):
            bin_name = col.replace('rmsd_', '')
            mean_val = df_denoised[col].mean()
            std_val = df_denoised[col].std()
            count_col = f'count_{bin_name}'
            if count_col in df_denoised.columns:
                total_points = df_denoised[count_col].sum()
                print(f"  {bin_name} Å: {mean_val:.6f} ± {std_val:.6f} (总点数: {total_points})")
            else:
                print(f"  {bin_name} Å: {mean_val:.6f} ± {std_val:.6f}")
    
    # 6. 保存结果
    print("\n" + "="*80)
    print("保存结果")
    print("="*80)
    
    # 获取输出目录，如果是相对路径则相对于项目根目录（funcmol的上一级）
    output_dir_str = config.get("output_dir", "exps/analysis/field_quality")
    if os.path.isabs(output_dir_str):
        output_dir = Path(output_dir_str)
    else:
        # 相对于项目根目录（funcmol的上一级）
        output_dir = PROJECT_ROOT.parent / output_dir_str
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存CSV - predicted field
    dataset_type = config.get("dataset_type", "qm9")
    csv_path = output_dir / f"field_quality_predicted_{dataset_type}.csv"
    df.to_csv(csv_path, index=False)
    print(f"Predicted Field详细结果已保存到: {csv_path}")
    
    # 保存汇总统计 - predicted field
    summary = {
        'dataset_type': dataset_type,
        'field_type': 'predicted',
        'num_samples': len(all_results),
        'mean_rmsd': mean_rmsd,
        'std_rmsd': std_rmsd,
        'min_distance_to_atom': df['min_distance_to_atom'].min(),
    }
    
    # 添加各距离区间的平均RMSD
    for col in sorted(rmsd_bin_cols, key=lambda x: float(x.split('_')[1].split('-')[0]) if '-' in x else float(x.split('_')[1].replace('+', ''))):
        bin_name = col.replace('rmsd_', '')
        summary[f'mean_rmsd_{bin_name}'] = df[col].mean()
        summary[f'std_rmsd_{bin_name}'] = df[col].std()
    
    summary_df = pd.DataFrame([summary])
    summary_path = output_dir / f"field_quality_summary_predicted_{dataset_type}.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Predicted Field汇总统计已保存到: {summary_path}")
    
    # 如果评估了denoised_field，单独保存denoised field的CSV和统计
    if evaluate_denoised_field and len(all_denoised_results) > 0:
        df_denoised = pd.DataFrame(all_denoised_results)
        denoised_csv_path = output_dir / f"field_quality_denoised_{dataset_type}.csv"
        df_denoised.to_csv(denoised_csv_path, index=False)
        print(f"Denoised Field详细结果已保存到: {denoised_csv_path}")
        
        # 保存denoised field的汇总统计
        mean_denoised_rmsd = df_denoised['rmsd'].mean()
        std_denoised_rmsd = df_denoised['rmsd'].std()
        denoised_summary = {
            'dataset_type': dataset_type,
            'field_type': 'denoised',
            'num_samples': len(all_denoised_results),
            'mean_rmsd': mean_denoised_rmsd,
            'std_rmsd': std_denoised_rmsd,
            'min_distance_to_atom': df_denoised['min_distance_to_atom'].min(),
        }
        
        # 添加各距离区间的denoised field RMSD
        denoised_rmsd_bin_cols = [col for col in df_denoised.columns if col.startswith('rmsd_')]
        for col in sorted(denoised_rmsd_bin_cols, key=lambda x: float(x.split('_')[1].split('-')[0]) if '-' in x else float(x.split('_')[1].replace('+', ''))):
            bin_name = col.replace('rmsd_', '')
            denoised_summary[f'mean_rmsd_{bin_name}'] = df_denoised[col].mean()
            denoised_summary[f'std_rmsd_{bin_name}'] = df_denoised[col].std()
        
        denoised_summary_df = pd.DataFrame([denoised_summary])
        denoised_summary_path = output_dir / f"field_quality_summary_denoised_{dataset_type}.csv"
        denoised_summary_df.to_csv(denoised_summary_path, index=False)
        print(f"Denoised Field汇总统计已保存到: {denoised_summary_path}")
    
    # 7. 绘制Predicted Field的距离分组RMSD图表
    print("\n" + "="*80)
    print("生成Predicted Field距离分组RMSD图表")
    print("="*80)
    
    if len(all_rmsd_by_distance) > 0:
        plot_path = output_dir / f"rmsd_by_distance_predicted_{dataset_type}.png"
        plot_rmsd_by_distance(
            all_rmsd_by_distance,
            all_counts_by_distance,
            plot_path,
            title=f"Predicted Field RMSD Loss by Distance ({dataset_type})"
        )
    else:
        print("警告: 没有数据可以绘制Predicted Field RMSD图")
    
    # 8. 绘制Predicted Field的距离分组相对误差图表
    print("\n" + "="*80)
    print("生成Predicted Field距离分组相对误差图表")
    print("="*80)
    
    if len(all_relative_error_by_distance) > 0:
        relative_error_plot_path = output_dir / f"relative_error_by_distance_predicted_{dataset_type}.png"
        plot_relative_error_by_distance(
            all_relative_error_by_distance,
            all_counts_by_distance,
            relative_error_plot_path,
            title=f"Predicted Field Relative Error by Distance ({dataset_type})"
        )
    else:
        print("警告: 没有数据可以绘制Predicted Field相对误差图")
    
    # 9. 如果评估了denoised_field，绘制Denoised Field的图表
    if evaluate_denoised_field and len(all_denoised_rmsd_by_distance) > 0:
        print("\n" + "="*80)
        print("生成Denoised Field距离分组RMSD图表")
        print("="*80)
        
        denoised_plot_path = output_dir / f"rmsd_by_distance_denoised_{dataset_type}.png"
        plot_rmsd_by_distance(
            all_denoised_rmsd_by_distance,
            all_counts_by_distance,
            denoised_plot_path,
            title=f"Denoised Field RMSD Loss by Distance ({dataset_type})"
        )
        
        print("\n" + "="*80)
        print("生成Denoised Field距离分组相对误差图表")
        print("="*80)
        
        if len(all_denoised_relative_error_by_distance) > 0:
            denoised_relative_error_plot_path = output_dir / f"relative_error_by_distance_denoised_{dataset_type}.png"
            plot_relative_error_by_distance(
                all_denoised_relative_error_by_distance,
                all_counts_by_distance,
                denoised_relative_error_plot_path,
                title=f"Denoised Field Relative Error by Distance ({dataset_type})"
            )
        else:
            print("警告: 没有数据可以绘制Denoised Field相对误差图")
    
    print("\n" + "="*80)
    print("评估完成！")
    print("="*80)


if __name__ == "__main__":
    main()

