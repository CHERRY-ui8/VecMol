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
    
    # 默认距离区间
    if distance_bins is None:
        distance_bins = [0, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, float('inf')]
    
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
    
    # 创建图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 左图：RMSD vs 距离区间
    bin_names_sorted = sorted(bin_names, key=lambda x: float(x.split('-')[0]) if '-' in x else float(x.replace('+', '')))
    rmsd_values = [avg_rmsd_by_bin[bn] for bn in bin_names_sorted]
    counts = [total_counts_by_bin[bn] for bn in bin_names_sorted]
    
    # 绘制柱状图
    bars = ax1.bar(range(len(bin_names_sorted)), rmsd_values, alpha=0.7, color='steelblue')
    ax1.set_xlabel('Distance to Nearest Atom (Å)', fontsize=12)
    ax1.set_ylabel('RMSD Loss', fontsize=12)
    ax1.set_title('Average RMSD Loss by Distance', fontsize=14, fontweight='bold')
    ax1.set_xticks(range(len(bin_names_sorted)))
    ax1.set_xticklabels(bin_names_sorted, rotation=45, ha='right')
    ax1.grid(axis='y', alpha=0.3)
    
    # 在柱状图上添加数值标签
    for i, (bar, rmsd, count) in enumerate(zip(bars, rmsd_values, counts)):
        if count > 0:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{rmsd:.4f}\n(n={count})',
                    ha='center', va='bottom', fontsize=9)
    
    # 右图：点数分布
    ax2.bar(range(len(bin_names_sorted)), counts, alpha=0.7, color='coral')
    ax2.set_xlabel('Distance to Nearest Atom (Å)', fontsize=12)
    ax2.set_ylabel('Number of Points', fontsize=12)
    ax2.set_title('Point Distribution by Distance', fontsize=14, fontweight='bold')
    ax2.set_xticks(range(len(bin_names_sorted)))
    ax2.set_xticklabels(bin_names_sorted, rotation=45, ha='right')
    ax2.grid(axis='y', alpha=0.3)
    
    # 在柱状图上添加数值标签
    for i, (count, bar) in enumerate(zip(counts, ax2.patches)):
        if count > 0:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{count}',
                    ha='center', va='bottom', fontsize=9)
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"图表已保存到: {output_path}")


@hydra.main(config_path="../configs", config_name="field_quality_drugs", version_base=None)
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
    nf_checkpoint_path = config.get("nf_pretrained_path")
    if not nf_checkpoint_path:
        raise ValueError("配置文件中必须提供 nf_pretrained_path")
    
    print(f"加载checkpoint: {nf_checkpoint_path}")
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
    # 评估时只从格点采样，不采样邻近点（targeted_sampling_ratio=0）
    # 将相对路径转换为绝对路径
    data_dir = config_dict.get("dset", {}).get("data_dir", "dataset/data")
    if not os.path.isabs(data_dir):
        # PROJECT_ROOT 指向 funcmol 目录，所以直接拼接 dataset/data
        data_dir = str(PROJECT_ROOT / data_dir)
    
    # 从dset配置中读取atom_distance_threshold，如果没有则使用默认值0.5
    atom_distance_threshold = config_dict.get("dset", {}).get("atom_distance_threshold", 0.5)
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
        targeted_sampling_ratio=0,  # 评估时只从格点采样，不采样邻近点
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
    all_rmsd_by_distance = []  # 存储所有样本的距离分组RMSD
    all_counts_by_distance = []  # 存储所有样本的距离分组点数
    
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
            
            # 使用decoder预测场
            with torch.no_grad():
                pred_field = decoder(query_points, codes)  # [B, n_points, n_atom_types, 3]
            
            # 对每个batch中的样本进行处理
            for b in range(B):
                sample_gt_field = gt_field[b]  # [n_points, n_atom_types, 3]
                sample_pred_field = pred_field[b]  # [n_points, n_atom_types, 3]
                sample_query_points = query_points[b]  # [n_points, 3]
                
                # 获取该样本的原子位置
                atom_mask = (batch.x[batch.batch == b] != PADDING_INDEX)
                sample_atom_positions = batch.pos[batch.batch == b][atom_mask]  # [n_atoms, 3]
                
                # 计算RMSD（梯度场的RMSD）
                rmsd = compute_rmsd(sample_pred_field, sample_gt_field)
                
                # 按距离分组计算RMSD
                rmsd_by_bin, counts_by_bin, distances = compute_rmsd_by_distance(
                    sample_pred_field,
                    sample_gt_field,
                    sample_query_points,
                    sample_atom_positions,
                    distance_bins=distance_bins
                )
                
                # 保存距离分组数据用于后续绘图
                all_rmsd_by_distance.append(rmsd_by_bin)
                all_counts_by_distance.append(counts_by_bin)
                
                # 获取原子数量
                n_atoms = atom_mask.sum().item()
                
                # 计算距离统计
                distances_np = distances.cpu().numpy()
                mean_distance = float(np.mean(distances_np))
                min_distance = float(np.min(distances_np))
                max_distance = float(np.max(distances_np))
                
                # 保存结果
                result = {
                    'sample_idx': batch_idx * batch_size + b,
                    'rmsd': rmsd,
                    'mean_distance_to_atom': mean_distance,
                    'min_distance_to_atom': min_distance,
                    'max_distance_to_atom': max_distance,
                    'n_points': n_points,
                    'n_atoms': n_atoms,
                }
                # 添加每个距离区间的RMSD
                for bin_name, bin_rmsd in rmsd_by_bin.items():
                    result[f'rmsd_{bin_name}'] = bin_rmsd
                # 添加每个距离区间的点数
                for bin_name, bin_count in counts_by_bin.items():
                    result[f'count_{bin_name}'] = bin_count
                
                all_results.append(result)
                
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
    print("\n梯度场RMSD:")
    print(f"  平均值: {mean_rmsd:.6f} ± {std_rmsd:.6f}")
    print(f"  最小值: {df['rmsd'].min():.6f}")
    print(f"  最大值: {df['rmsd'].max():.6f}")
    
    print("\n距离最近原子的距离统计:")
    print(f"  平均距离: {df['mean_distance_to_atom'].mean():.4f} ± {df['mean_distance_to_atom'].std():.4f} Å")
    print(f"  最小距离: {df['min_distance_to_atom'].min():.4f} Å")
    print(f"  最大距离: {df['max_distance_to_atom'].max():.4f} Å")
    
    # 打印各距离区间的RMSD统计
    print("\n各距离区间的RMSD统计:")
    rmsd_bin_cols = [col for col in df.columns if col.startswith('rmsd_')]
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
    
    # 6. 保存结果
    print("\n" + "="*80)
    print("保存结果")
    print("="*80)
    
    output_dir = Path(config.get("output_dir", "exps/analysis/field_quality"))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存CSV
    dataset_type = config.get("dataset_type", "qm9")
    csv_path = output_dir / f"field_quality_{dataset_type}.csv"
    df.to_csv(csv_path, index=False)
    print(f"详细结果已保存到: {csv_path}")
    
    # 保存汇总统计
    summary = {
        'dataset_type': dataset_type,
        'num_samples': len(all_results),
        'mean_rmsd': mean_rmsd,
        'std_rmsd': std_rmsd,
        'mean_distance_to_atom': df['mean_distance_to_atom'].mean(),
        'std_distance_to_atom': df['mean_distance_to_atom'].std(),
    }
    
    # 添加各距离区间的平均RMSD
    for col in sorted(rmsd_bin_cols, key=lambda x: float(x.split('_')[1].split('-')[0]) if '-' in x else float(x.split('_')[1].replace('+', ''))):
        bin_name = col.replace('rmsd_', '')
        summary[f'mean_rmsd_{bin_name}'] = df[col].mean()
        summary[f'std_rmsd_{bin_name}'] = df[col].std()
    
    summary_df = pd.DataFrame([summary])
    summary_path = output_dir / f"field_quality_summary_{dataset_type}.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"汇总统计已保存到: {summary_path}")
    
    # 7. 绘制距离分组RMSD图表
    print("\n" + "="*80)
    print("生成距离分组RMSD图表")
    print("="*80)
    
    if len(all_rmsd_by_distance) > 0:
        plot_path = output_dir / f"rmsd_by_distance_{dataset_type}.png"
        plot_rmsd_by_distance(
            all_rmsd_by_distance,
            all_counts_by_distance,
            plot_path,
            title=f"RMSD Loss by Distance to Nearest Atom ({dataset_type})"
        )
    else:
        print("警告: 没有数据可以绘制")
    
    print("\n" + "="*80)
    print("评估完成！")
    print("="*80)


if __name__ == "__main__":
    main()

