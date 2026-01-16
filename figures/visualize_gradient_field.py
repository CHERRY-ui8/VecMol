#!/usr/bin/env python3
"""
可视化QM9样本的梯度场

从QM9数据集加载一个样本，使用encoder编码成codes，然后用decoder解码并用pyvista可视化梯度场。
每种原子类型分别可视化，使用不同颜色标注。
"""

import os
import sys
import argparse
import numpy as np
import torch
import pyvista as pv
from pathlib import Path
from omegaconf import OmegaConf

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from torch_geometric.loader import DataLoader
from funcmol.utils.utils_nf import load_neural_field
from funcmol.dataset.dataset_field import create_field_loaders, create_gnf_converter
from funcmol.utils.constants import ELEMENTS_HASH_INV

# 原子颜色映射（来自 gnf_visualizer.py）
ATOM_COLORS = {
    0: 'green',      # C
    1: 'gray',       # H
    2: 'red',        # O
    3: 'blue',       # N
    4: 'deeppink',   # F
    5: 'yellow',     # S
    6: 'yellowgreen', # Cl
    7: 'brown'       # Br
}


def load_config_and_models(config_path, nf_pretrained_path=None):
    """加载配置和模型"""
    # 加载配置文件
    config = OmegaConf.load(config_path)
    config = OmegaConf.to_container(config, resolve=True)
    
    # 如果提供了checkpoint路径，覆盖配置中的路径
    if nf_pretrained_path:
        config["nf_pretrained_path"] = nf_pretrained_path
    
    # 验证checkpoint路径
    nf_pretrained_path = config.get("nf_pretrained_path")
    if not nf_pretrained_path:
        raise ValueError("必须指定 nf_pretrained_path 参数来指定Lightning checkpoint路径")
    
    if not nf_pretrained_path.endswith('.ckpt') or not os.path.exists(nf_pretrained_path):
        raise ValueError(f"指定的checkpoint文件不存在或格式不正确: {nf_pretrained_path}")
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 设置随机种子
    seed = config.get("seed", 1234)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # 加载Lightning checkpoint
    checkpoint = torch.load(nf_pretrained_path, map_location='cpu', weights_only=False)
    
    # 从checkpoint中提取配置
    if 'hyper_parameters' in checkpoint:
        config_model = checkpoint['hyper_parameters']
    else:
        config_model = checkpoint.get('config', {})
    
    # 转换为普通字典
    try:
        import omegaconf.dictconfig
        if isinstance(config_model, omegaconf.dictconfig.DictConfig):
            config_model = OmegaConf.to_container(config_model, resolve=True)
        if isinstance(config, omegaconf.dictconfig.DictConfig):
            config = OmegaConf.to_container(config, resolve=True)
    except (ImportError, AttributeError):
        # 如果OmegaConf版本不支持DictConfig，尝试其他方法
        if hasattr(OmegaConf, 'is_dict') and OmegaConf.is_dict(config_model):
            config_model = OmegaConf.to_container(config_model, resolve=True)
        if hasattr(OmegaConf, 'is_dict') and OmegaConf.is_dict(config):
            config = OmegaConf.to_container(config, resolve=True)
    
    # 合并配置（yaml配置优先）
    for key in config.keys():
        if key in config_model and isinstance(config_model[key], dict) and isinstance(config[key], dict):
            config_model[key].update(config[key])
        else:
            config_model[key] = config[key]
    config = config_model
    
    # 加载模型
    print(f"Loading model from: {nf_pretrained_path}")
    encoder, decoder = load_neural_field(nf_pretrained_path, config)
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    encoder.eval()
    decoder.eval()
    
    # 创建GNFConverter实例
    gnf_converter = create_gnf_converter(config)
    
    return config, encoder, decoder, gnf_converter, device


def sample_query_points(dataset, n_points=200, device='cpu'):
    """从数据集中采样query points"""
    full_grid_high_res = dataset.full_grid_high_res
    grid_size = len(full_grid_high_res)
    
    # 核心修改：先确保索引在 full_grid_high_res 相同的设备上（通常是 CPU）
    # 或者直接不指定 device，默认就在 CPU
    grid_indices = torch.randperm(grid_size)[:min(n_points * 2, grid_size)]
    
    # 先索引获取子集，然后再移动到目标 device
    grid_points = full_grid_high_res[grid_indices].to(device)
    
    unique_points = torch.unique(grid_points, dim=0)
    
    # 如果unique points不够，补充
    if unique_points.size(0) < n_points:
        remaining = n_points - unique_points.size(0)
        # 同样，先在 CPU 上生成随机索引，再取值并移动
        additional_indices = torch.randint(grid_size, (remaining,))
        additional_points = full_grid_high_res[additional_indices].to(device)
        unique_points = torch.cat([unique_points, additional_points], dim=0)
    
    # 这里的 unique_points 已经在 device 上了，所以这里的 indices 可以在 device 上
    indices = torch.randperm(unique_points.size(0), device=device)[:n_points]
    query_points = unique_points[indices]
    
    return query_points


def visualize_gradient_field(
    config_path,
    nf_pretrained_path=None,
    sample_idx=0,
    n_points=200,
    split="train",
    output_path=None,
    elev=30.0,
    azim=60.0,
):
    """
    可视化梯度场
    
    Args:
        config_path: 配置文件路径
        nf_pretrained_path: checkpoint路径（可选，如果提供会覆盖配置文件中的路径）
        sample_idx: 要可视化的样本索引（默认0）
        n_points: 采样点数量（默认200）
        split: 数据集分割（默认"train"）
        output_path: 输出文件路径（可选，如果为None则显示交互式窗口）
        elev: 仰角（默认30.0）
        azim: 方位角（默认60.0）
    """
    # 加载配置和模型
    config, encoder, decoder, gnf_converter, device = load_config_and_models(
        config_path, nf_pretrained_path
    )
    
    # 创建数据加载器
    config["split"] = split
    loader = create_field_loaders(config, gnf_converter, split=split)
    
    # 禁用shuffle以保持顺序
    loader = DataLoader(
        loader.dataset,
        batch_size=1,
        num_workers=0,  # 设为0以避免多进程问题
        shuffle=False,
        pin_memory=False,
    )
    
    # 获取指定样本
    dataset = loader.dataset
    if sample_idx >= len(dataset):
        raise ValueError(f"样本索引 {sample_idx} 超出范围（数据集大小: {len(dataset)}）")
    
    # 获取样本
    sample_data = dataset[sample_idx]
    # 转换为Batch对象
    from torch_geometric.data import Batch
    # 确保batch属性存在
    if not hasattr(sample_data, 'batch') or sample_data.batch is None:
        sample_data.batch = torch.zeros(sample_data.pos.shape[0], dtype=torch.long)
    batch = Batch.from_data_list([sample_data]).to(device)
    
    print(f"Processing sample {sample_idx}...")
    
    # 编码
    with torch.no_grad():
        codes = encoder(batch)  # [1, n_grid, code_dim]
    
    # 采样query points
    query_points = sample_query_points(dataset, n_points=n_points, device=device)
    query_points = query_points.unsqueeze(0)  # [1, n_points, 3]
    
    print(f"Sampled {n_points} query points")
    
    # 解码
    with torch.no_grad():
        vector_field = decoder(query_points, codes)  # [1, n_points, n_atom_types, 3]
    
    # 转换为numpy
    origins = query_points[0].cpu().numpy()  # [n_points, 3]
    vector_field_np = vector_field[0].cpu().numpy()  # [n_points, n_atom_types, 3]
    
    # 获取原子类型信息
    n_atom_types = vector_field_np.shape[1]
    
    # 创建plotter（如果保存图片，使用off_screen模式）
    if output_path:
        plotter = pv.Plotter(off_screen=True)
    else:
        plotter = pv.Plotter()
    
    # 为每种原子类型分别可视化
    for atom_type_idx in range(n_atom_types):
        # 提取该原子类型的向量场
        vectors_atom_type = vector_field_np[:, atom_type_idx, :]  # [n_points, 3]
        
        # 计算向量模长，用于过滤零向量或过小的向量
        magnitudes = np.linalg.norm(vectors_atom_type, axis=1)
        threshold = magnitudes.max() * 0.01  # 过滤小于最大值1%的向量
        
        # 过滤有效向量
        valid_mask = magnitudes > threshold
        if valid_mask.sum() == 0:
            print(f"Warning: No valid vectors for atom type {atom_type_idx}")
            continue
        
        origins_filtered = origins[valid_mask]
        vectors_filtered = vectors_atom_type[valid_mask]
        
        # 创建向量场
        pdata = pv.vector_poly_data(origins_filtered, vectors_filtered)
        
        # 生成箭头
        glyph = pdata.glyph(orient='vectors', scale='mag')
        
        # 获取颜色和名称
        color = ATOM_COLORS.get(atom_type_idx, 'gray')
        atom_name = ELEMENTS_HASH_INV.get(atom_type_idx, f"Type{atom_type_idx}")
        
        # 添加到plotter
        plotter.add_mesh(glyph, color=color, label=f"{atom_name}", opacity=0.8)
        
        print(f"Added {valid_mask.sum()} vectors for {atom_name} (color: {color})")
    
    # 设置坐标轴样式（参考 empty_3d_axis.py）
    # 计算坐标范围
    all_coords = np.vstack([origins])
    margin = 0.5
    x_min, x_max = all_coords[:, 0].min() - margin, all_coords[:, 0].max() + margin
    y_min, y_max = all_coords[:, 1].min() - margin, all_coords[:, 1].max() + margin
    z_min, z_max = all_coords[:, 2].min() - margin, all_coords[:, 2].max() + margin
    
    # PyVista使用show_grid来设置坐标轴范围和标签
    plotter.show_grid(
        bounds=[x_min, x_max, y_min, y_max, z_min, z_max],
        xtitle="X (Å)",
        ytitle="Y (Å)",
        ztitle="Z (Å)"
    )
    
    # 设置视角（参考 empty_3d_axis.py: elev=30, azim=60）
    # PyVista的视角设置：先设置为等轴测视图，然后调整
    plotter.view_isometric()
    # 注意：PyVista的camera.elevation和azimuth与matplotlib不完全相同
    # 这里使用view_isometric()作为基础，用户可以在交互窗口中手动调整
    
    # 添加图例
    legend_labels = [f"{ELEMENTS_HASH_INV.get(i, f'Type{i}')}" 
                     for i in range(n_atom_types) 
                     if i in ATOM_COLORS]
    if legend_labels:
        plotter.add_legend(labels=legend_labels)
    
    # 显示或保存
    if output_path:
        # 保存图片（off_screen模式下直接截图）
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        plotter.screenshot(output_path)
        print(f"Saved visualization to: {output_path}")
        plotter.close()
    else:
        # 显示交互式窗口
        plotter.show()


def parse_args():
    parser = argparse.ArgumentParser(
        description="可视化QM9样本的梯度场",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--config",
        type=str,
        default="funcmol/configs/infer_codes.yaml",
        help="配置文件路径",
    )
    parser.add_argument(
        "--nf_pretrained_path",
        type=str,
        default=None,
        help="Neural field checkpoint路径（可选，会覆盖配置文件中的路径）",
    )
    parser.add_argument(
        "--sample_idx",
        type=int,
        default=0,
        help="要可视化的样本索引",
    )
    parser.add_argument(
        "--n_points",
        type=int,
        default=200,
        help="采样点数量（默认200，比训练时使用的500更少以便可视化）",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val", "test"],
        help="数据集分割",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出文件路径（可选，如果为None则显示交互式窗口）",
    )
    parser.add_argument(
        "--elev",
        type=float,
        default=30.0,
        help="仰角（elevation angle）",
    )
    parser.add_argument(
        "--azim",
        type=float,
        default=60.0,
        help="方位角（azimuth angle）",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 确保配置文件路径是绝对路径
    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(project_root, config_path)
    
    visualize_gradient_field(
        config_path=config_path,
        nf_pretrained_path=args.nf_pretrained_path,
        sample_idx=args.sample_idx,
        n_points=args.n_points,
        split=args.split,
        output_path=args.output,
        elev=args.elev,
        azim=args.azim,
    )


if __name__ == "__main__":
    main()

