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
from typing import Optional
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf

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
                
                # 计算MSE（向量场的MSE，对所有分量）
                mse = compute_mse(sample_pred_field, sample_gt_field)
                
                # 计算模长MSE
                magnitude_mse = compute_field_magnitude_mse(sample_pred_field, sample_gt_field)
                
                # 计算PSNR
                # 使用gt_field的最大值作为max_val
                max_val = torch.max(torch.abs(sample_gt_field)).item()
                if max_val == 0:
                    max_val = 1.0  # 避免除零
                psnr = compute_psnr(mse, max_val=max_val)
                
                # 计算模长PSNR
                max_mag = torch.max(torch.norm(sample_gt_field, dim=-1)).item()
                if max_mag == 0:
                    max_mag = 1.0
                magnitude_psnr = compute_psnr(magnitude_mse, max_val=max_mag)
                
                # 获取原子数量
                atom_mask = (batch.x[batch.batch == b] != PADDING_INDEX)
                n_atoms = atom_mask.sum().item()
                
                # 保存结果
                result = {
                    'sample_idx': batch_idx * batch_size + b,
                    'mse': mse,
                    'magnitude_mse': magnitude_mse,
                    'psnr': psnr,
                    'magnitude_psnr': magnitude_psnr,
                    'max_val': max_val,
                    'max_magnitude': max_mag,
                    'n_points': n_points,
                    'n_atoms': n_atoms,
                }
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
    
    # 计算平均值和标准差
    mean_mse = df['mse'].mean()
    std_mse = df['mse'].std()
    mean_magnitude_mse = df['magnitude_mse'].mean()
    std_magnitude_mse = df['magnitude_mse'].std()
    mean_psnr = df['psnr'].mean()
    std_psnr = df['psnr'].std()
    mean_magnitude_psnr = df['magnitude_psnr'].mean()
    std_magnitude_psnr = df['magnitude_psnr'].std()
    
    print(f"总样本数: {len(all_results)}")
    print("\n向量场MSE:")
    print(f"  平均值: {mean_mse:.6f} ± {std_mse:.6f}")
    print(f"  最小值: {df['mse'].min():.6f}")
    print(f"  最大值: {df['mse'].max():.6f}")
    
    print("\n向量场模长MSE:")
    print(f"  平均值: {mean_magnitude_mse:.6f} ± {std_magnitude_mse:.6f}")
    print(f"  最小值: {df['magnitude_mse'].min():.6f}")
    print(f"  最大值: {df['magnitude_mse'].max():.6f}")
    
    print("\n向量场PSNR (dB):")
    print(f"  平均值: {mean_psnr:.2f} ± {std_psnr:.2f}")
    print(f"  最小值: {df['psnr'].min():.2f}")
    print(f"  最大值: {df['psnr'].max():.2f}")
    
    print("\n向量场模长PSNR (dB):")
    print(f"  平均值: {mean_magnitude_psnr:.2f} ± {std_magnitude_psnr:.2f}")
    print(f"  最小值: {df['magnitude_psnr'].min():.2f}")
    print(f"  最大值: {df['magnitude_psnr'].max():.2f}")
    
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
        'mean_mse': mean_mse,
        'std_mse': std_mse,
        'mean_magnitude_mse': mean_magnitude_mse,
        'std_magnitude_mse': std_magnitude_mse,
        'mean_psnr': mean_psnr,
        'std_psnr': std_psnr,
        'mean_magnitude_psnr': mean_magnitude_psnr,
        'std_magnitude_psnr': std_magnitude_psnr,
    }
    
    summary_df = pd.DataFrame([summary])
    summary_path = output_dir / f"field_quality_summary_{dataset_type}.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"汇总统计已保存到: {summary_path}")
    
    print("\n" + "="*80)
    print("评估完成！")
    print("="*80)


if __name__ == "__main__":
    main()

