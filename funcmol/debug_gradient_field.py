import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils.constants import PADDING_INDEX
from utils.utils_nf import create_neural_field, load_neural_field
from utils.gnf_converter import GNFConverter
from dataset.dataset_field import create_field_loaders, create_gnf_converter
from lightning import Fabric
from torch_geometric.utils import to_dense_batch
import hydra
from omegaconf import OmegaConf

def debug_gradient_field_comparison():
    """调试梯度场预测差异"""
    
    # 设置环境
    fabric = Fabric(
        accelerator="cpu",
        devices=1,
        precision="32-true",
        strategy="auto"
    )
    fabric.launch()
    device = torch.device("cpu")
    
    # 加载配置
    config_dir = str(Path.cwd() / "configs")
    with hydra.initialize_config_dir(config_dir=config_dir, version_base=None):
        config = hydra.compose(config_name="train_nf_qm9")
    
    config["dset"]["data_dir"] = str(Path.cwd() / "dataset" / "data")
    
    # 加载模型
    enc, dec = create_neural_field(config, fabric)
    enc = enc.to(device)
    dec = dec.to(device)
    
    # 创建转换器
    converter = create_gnf_converter(config, device="cpu")
    
    # 加载数据
    loader_val = create_field_loaders(config, converter, split="val", fabric=fabric)
    batch = next(iter(loader_val)).to(device)
    
    coords, _ = to_dense_batch(batch.pos, batch.batch, fill_value=0)
    atoms_channel, _ = to_dense_batch(batch.x, batch.batch, fill_value=PADDING_INDEX)
    gt_coords = coords
    gt_types = atoms_channel
    
    with torch.no_grad():
        codes = enc(batch)
    
    sample_idx = 0
    atom_type = 0  # C原子
    
    # 获取当前样本的有效原子
    gt_mask = (gt_types[sample_idx] != PADDING_INDEX)
    gt_valid_coords = gt_coords[sample_idx][gt_mask]
    gt_valid_types = gt_types[sample_idx][gt_mask]
    
    print(f"样本 {sample_idx} 的原子坐标:")
    print(f"坐标: {gt_valid_coords}")
    print(f"类型: {gt_valid_types}")
    
    # 测试1: 使用训练时的随机采样策略
    print("\n=== 测试1: 随机采样策略 ===")
    n_random_points = 1000
    random_query_points = torch.rand(n_random_points, 3, device=device) * 2 - 1
    
    # 计算ground truth
    gt_field_random = converter.mol2gnf(
        gt_valid_coords.unsqueeze(0),
        gt_valid_types.unsqueeze(0),
        random_query_points.unsqueeze(0)
    )
    gt_gradients_random = torch.norm(gt_field_random[0, :, atom_type, :], dim=1)
    
    # 计算预测
    pred_field_random = dec(random_query_points.unsqueeze(0), codes[sample_idx:sample_idx+1])
    pred_gradients_random = torch.norm(pred_field_random[0, :, atom_type, :], dim=1)
    
    # 计算误差
    mse_random = torch.mean((gt_gradients_random - pred_gradients_random) ** 2).item()
    mae_random = torch.mean(torch.abs(gt_gradients_random - pred_gradients_random)).item()
    
    print(f"随机采样 - MSE: {mse_random:.6f}, MAE: {mae_random:.6f}")
    
    # 测试2: 使用1D直线采样策略
    print("\n=== 测试2: 1D直线采样策略 ===")
    n_1d_points = 1000
    x = torch.linspace(-1, 1, n_1d_points, device=device)
    molecule_center = gt_valid_coords.mean(dim=0)
    y_coord = molecule_center[1].item()
    z_coord = molecule_center[2].item()
    
    query_points_1d = torch.zeros(n_1d_points, 3, device=device)
    query_points_1d[:, 0] = x
    query_points_1d[:, 1] = y_coord
    query_points_1d[:, 2] = z_coord
    
    print(f"1D采样线: y={y_coord:.4f}, z={z_coord:.4f}")
    
    # 计算ground truth
    gt_field_1d = converter.mol2gnf(
        gt_valid_coords.unsqueeze(0),
        gt_valid_types.unsqueeze(0),
        query_points_1d.unsqueeze(0)
    )
    gt_gradients_1d = torch.norm(gt_field_1d[0, :, atom_type, :], dim=1)
    
    # 计算预测
    pred_field_1d = dec(query_points_1d.unsqueeze(0), codes[sample_idx:sample_idx+1])
    pred_gradients_1d = torch.norm(pred_field_1d[0, :, atom_type, :], dim=1)
    
    # 计算误差
    mse_1d = torch.mean((gt_gradients_1d - pred_gradients_1d) ** 2).item()
    mae_1d = torch.mean(torch.abs(gt_gradients_1d - pred_gradients_1d)).item()
    
    print(f"1D采样 - MSE: {mse_1d:.6f}, MAE: {mae_1d:.6f}")
    
    # 测试3: 在原子附近采样
    print("\n=== 测试3: 原子附近采样 ===")
    n_near_points = 1000
    # 在C原子附近采样
    c_atom_pos = gt_valid_coords[gt_valid_types == atom_type][0]
    near_query_points = c_atom_pos.unsqueeze(0) + torch.randn(n_near_points, 3, device=device) * 0.1
    
    # 计算ground truth
    gt_field_near = converter.mol2gnf(
        gt_valid_coords.unsqueeze(0),
        gt_valid_types.unsqueeze(0),
        near_query_points.unsqueeze(0)
    )
    gt_gradients_near = torch.norm(gt_field_near[0, :, atom_type, :], dim=1)
    
    # 计算预测
    pred_field_near = dec(near_query_points.unsqueeze(0), codes[sample_idx:sample_idx+1])
    pred_gradients_near = torch.norm(pred_field_near[0, :, atom_type, :], dim=1)
    
    # 计算误差
    mse_near = torch.mean((gt_gradients_near - pred_gradients_near) ** 2).item()
    mae_near = torch.mean(torch.abs(gt_gradients_near - pred_gradients_near)).item()
    
    print(f"原子附近采样 - MSE: {mse_near:.6f}, MAE: {mae_near:.6f}")
    
    # 可视化对比
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 随机采样对比
    axes[0, 0].scatter(random_query_points[:, 0].cpu(), random_query_points[:, 1].cpu(), 
                      c=gt_gradients_random.cpu(), s=10, alpha=0.6, label='Ground Truth')
    axes[0, 0].scatter(random_query_points[:, 0].cpu(), random_query_points[:, 1].cpu(), 
                      c=pred_gradients_random.cpu(), s=10, alpha=0.6, marker='x', label='Predicted')
    axes[0, 0].set_title(f'Random Sampling (MSE: {mse_random:.4f})')
    axes[0, 0].legend()
    
    # 1D采样对比
    axes[0, 1].plot(x.cpu(), gt_gradients_1d.cpu(), label='Ground Truth', linewidth=2)
    axes[0, 1].plot(x.cpu(), pred_gradients_1d.cpu(), label='Predicted', linewidth=2, linestyle='--')
    axes[0, 1].plot(x.cpu(), pred_gradients_1d.cpu(), label='Predicted', linewidth=2, linestyle='--')
    axes[0, 1].set_title(f'1D Sampling (MSE: {mse_1d:.4f})')
    axes[0, 1].legend()
    
    # 原子附近采样对比
    axes[1, 0].scatter(near_query_points[:, 0].cpu(), near_query_points[:, 1].cpu(), 
                      c=gt_gradients_near.cpu(), s=10, alpha=0.6, label='Ground Truth')
    axes[1, 0].scatter(near_query_points[:, 0].cpu(), near_query_points[:, 1].cpu(), 
                      c=pred_gradients_near.cpu(), s=10, alpha=0.6, marker='x', label='Predicted')
    axes[1, 0].set_title(f'Near Atom Sampling (MSE: {mse_near:.4f})')
    axes[1, 0].legend()
    
    # 误差分布对比
    error_random = gt_gradients_random - pred_gradients_random
    error_1d = gt_gradients_1d - pred_gradients_1d
    error_near = gt_gradients_near - pred_gradients_near
    
    axes[1, 1].hist(error_random.cpu(), bins=50, alpha=0.7, label=f'Random (std: {error_random.std():.4f})')
    axes[1, 1].hist(error_1d.cpu(), bins=50, alpha=0.7, label=f'1D (std: {error_1d.std():.4f})')
    axes[1, 1].hist(error_near.cpu(), bins=50, alpha=0.7, label=f'Near (std: {error_near.std():.4f})')
    axes[1, 1].set_title('Error Distribution Comparison')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('gradient_field_debug_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n=== 总结 ===")
    print(f"随机采样误差: MSE={mse_random:.6f}, MAE={mae_random:.6f}")
    print(f"1D采样误差: MSE={mse_1d:.6f}, MAE={mae_1d:.6f}")
    print(f"原子附近采样误差: MSE={mse_near:.6f}, MAE={mae_near:.6f}")
    
    if mse_1d > mse_random * 10:
        print("\n⚠️  1D采样误差明显大于随机采样，可能原因:")
        print("1. 模型在特定直线上泛化能力不足")
        print("2. 选择的y,z坐标不是模型训练时常见的区域")
        print("3. 1D采样暴露了模型的局部缺陷")

if __name__ == "__main__":
    debug_gradient_field_comparison() 