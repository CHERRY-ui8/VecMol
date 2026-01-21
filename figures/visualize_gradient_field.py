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
import matplotlib.colors as mcolors

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


def load_config_and_gnf_converter(config_path):
    """
    只加载配置和GNFConverter，不使用neural field checkpoint
    直接使用ground truth field（通过gnf_converter.mol2gnf计算）
    """
    from omegaconf import OmegaConf
    
    # 确保配置文件路径是绝对路径
    if not os.path.isabs(config_path):
        config_path = os.path.join(project_root, config_path)
    
    # 使用 Hydra 加载配置（因为配置文件使用了 defaults）
    try:
        import hydra
        from hydra import initialize_config_dir, compose
        
        # 获取配置文件所在目录和文件名
        config_dir = os.path.dirname(config_path)
        config_name = os.path.basename(config_path).replace('.yaml', '')
        
        # 使用 Hydra 初始化并加载配置
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            cfg = compose(config_name=config_name)
            config = OmegaConf.to_container(cfg, resolve=True)
    except Exception as e:
        # 如果 Hydra 加载失败，尝试直接使用 OmegaConf
        print(f"Warning: Failed to load config with Hydra: {e}, trying direct OmegaConf.load()")
        config = OmegaConf.load(config_path)
        config = OmegaConf.to_container(config, resolve=True)
    
    # 修复数据路径：将相对路径转换为绝对路径
    if "dset" in config and "data_dir" in config["dset"]:
        data_dir = config["dset"]["data_dir"]
        if not os.path.isabs(data_dir):
            if data_dir.startswith("dataset/data"):
                config["dset"]["data_dir"] = os.path.join(project_root, "funcmol", data_dir)
            else:
                config["dset"]["data_dir"] = os.path.join(project_root, data_dir)
            print(f"Converted data_dir to absolute path: {config['dset']['data_dir']}")
    
    # 创建GNFConverter实例
    from funcmol.dataset.dataset_field import create_gnf_converter
    gnf_converter = create_gnf_converter(config)
    
    return config, gnf_converter


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
    # 注意：某些配置可能包含无法解析的插值（如 oc.env），使用 resolve=False 避免错误
    try:
        import omegaconf.dictconfig
        if isinstance(config_model, omegaconf.dictconfig.DictConfig):
            try:
                config_model = OmegaConf.to_container(config_model, resolve=True)
            except Exception as e:
                # 如果解析失败（可能因为不支持的插值），尝试不解析插值
                print(f"Warning: Failed to resolve config_model with resolve=True: {e}, trying resolve=False")
                config_model = OmegaConf.to_container(config_model, resolve=False)
        if isinstance(config, omegaconf.dictconfig.DictConfig):
            try:
                config = OmegaConf.to_container(config, resolve=True)
            except Exception as e:
                print(f"Warning: Failed to resolve config with resolve=True: {e}, trying resolve=False")
                config = OmegaConf.to_container(config, resolve=False)
    except (ImportError, AttributeError):
        # 如果OmegaConf版本不支持DictConfig，尝试其他方法
        if hasattr(OmegaConf, 'is_dict') and OmegaConf.is_dict(config_model):
            try:
                config_model = OmegaConf.to_container(config_model, resolve=True)
            except Exception as e:
                print(f"Warning: Failed to resolve config_model: {e}, trying resolve=False")
                config_model = OmegaConf.to_container(config_model, resolve=False)
        if hasattr(OmegaConf, 'is_dict') and OmegaConf.is_dict(config):
            try:
                config = OmegaConf.to_container(config, resolve=True)
            except Exception as e:
                print(f"Warning: Failed to resolve config: {e}, trying resolve=False")
                config = OmegaConf.to_container(config, resolve=False)
    
    # 合并配置（yaml配置优先）
    for key in config.keys():
        if key in config_model and isinstance(config_model[key], dict) and isinstance(config[key], dict):
            config_model[key].update(config[key])
        else:
            config_model[key] = config[key]
    config = config_model
    
    # 修复数据路径：将相对路径转换为绝对路径
    if "dset" in config and "data_dir" in config["dset"]:
        data_dir = config["dset"]["data_dir"]
        if not os.path.isabs(data_dir):
            # 如果路径以 "dataset/data" 开头，说明是相对于 funcmol/ 目录的
            # 需要转换为 funcmol/dataset/data 的绝对路径
            if data_dir.startswith("dataset/data"):
                config["dset"]["data_dir"] = os.path.join(project_root, "funcmol", data_dir)
            else:
                # 其他情况，尝试相对于项目根目录
                config["dset"]["data_dir"] = os.path.join(project_root, data_dir)
            print(f"Converted data_dir to absolute path: {config['dset']['data_dir']}")
    
    # 加载模型
    print(f"Loading model from: {nf_pretrained_path}")
    encoder, decoder = load_neural_field(nf_pretrained_path, config)
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    encoder.eval()
    decoder.eval()
    
    # 确保 converter 配置中有所有必需的键（为缺失的键添加默认值）
    if "converter" in config:
        gnf_config = config["converter"]
    elif "gnf_converter" in config:
        gnf_config = config["gnf_converter"]
    else:
        gnf_config = {}
        config["converter"] = gnf_config
    
    # 为缺失的键添加默认值
    default_converter_values = {
        "field_variance_k_neighbors": 10,
        "field_variance_weight": 0.01,
        "gradient_sampling_candidate_multiplier": 3,
        "enable_early_stopping": False,
        "convergence_threshold": 1e-6,
        "min_iterations": 50,
    }
    for key, default_value in default_converter_values.items():
        if key not in gnf_config:
            print(f"Warning: Missing key '{key}' in converter config, using default value: {default_value}")
            gnf_config[key] = default_value
    
    # 创建GNFConverter实例
    gnf_converter = create_gnf_converter(config)
    
    return config, encoder, decoder, gnf_converter, device


def sample_query_points_from_grid(encoder, n_points=200, device='cpu'):
    """从encoder的grid坐标中采样query points"""
    # 获取encoder的grid坐标
    grid_coords = encoder.grid_coords  # [n_grid, 3]
    grid_size = grid_coords.shape[0]
    
    # 如果需要的点数超过grid大小，就使用所有grid点
    if n_points >= grid_size:
        query_points = grid_coords.to(device)
    else:
        # 随机采样n_points个grid点
        indices = torch.randperm(grid_size, device=device)[:n_points]
        query_points = grid_coords[indices].to(device)
    
    return query_points


def visualize_gradient_field(
    config_path,
    nf_pretrained_path=None,
    sample_idx=0,
    n_points_per_atom_type=None,
    split="val",
    output_path=None,
    elev=30.0,
    azim=60.0,
    axis_range=3.0,
    use_real_arrow_length=True,
    normalized_arrow_length=0.1,
    real_arrow_length_scale=1.0,
    arrow_scale=1.0,
    sample_from_grid=False,
    grid_dim=None,
    resolution=None,
    large_gradient_light_color=False,
    show_atoms=False,
    atom_style='transparent',
    atom_radius=0.3,
    show_atom_types=None,
):
    """
    可视化梯度场（使用ground truth field，不依赖neural field checkpoint）
    
    Args:
        config_path: 配置文件路径
        nf_pretrained_path: 已废弃，不再使用（保留以保持兼容性）
        sample_idx: 要可视化的样本索引（默认0）
        n_points_per_atom_type: 每种原子类型的采样点数量，可以是整数（所有类型相同）或字典（如{0: 500, 1: 300, 2: 400, 3: 200, 4: 100}）
        split: 数据集分割（默认"train"）
        output_path: 输出文件路径（可选，如果为None则显示交互式窗口）
        elev: 仰角（默认30.0）
        azim: 方位角（默认60.0）
        axis_range: 坐标轴范围，例如3.0表示从-3到3（默认3.0）
        use_real_arrow_length: 是否使用真实梯度大小作为箭头长度（True）还是所有箭头使用相同长度（False，默认True）
        normalized_arrow_length: 当use_real_arrow_length=False时，归一化后箭头放大到的长度（默认0.1）
        real_arrow_length_scale: 当use_real_arrow_length=True时，真实箭头长度的缩放因子（默认1.0，值越大箭头越长）
        arrow_scale: 箭头粗细缩放因子，值越大箭头越粗（默认1.0）
        sample_from_grid: 是否只从grid点采样（True）还是从显示范围内任意采样（False，默认False）
        grid_dim: grid的维度（如64表示64x64x64），如果为None则从dataset配置读取（默认None）
        resolution: grid点之间的间距（单位：埃），如果为None则从dataset配置读取（默认None）
        large_gradient_light_color: 当use_real_arrow_length=True时，如果为True，则梯度大的箭头颜色更浅（默认False，梯度小颜色深）
        show_atoms: 是否在真实原子位置显示小球（默认False）
        atom_style: 原子小球的样式，'transparent'（透明）、'wireframe'（3D线框）或'circle_2d'（2D虚线圆圈，默认'transparent'）
        atom_radius: 原子小球的半径（单位：埃，默认0.3）
        show_atom_types: 要显示的元素类型列表，例如[0, 1]表示只显示C和H。如果为None，则显示所有元素（默认None）
    """
    # 加载配置和GNFConverter（不使用neural field checkpoint）
    config, gnf_converter = load_config_and_gnf_converter(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
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
    batch = Batch.from_data_list([sample_data])
    
    print(f"Processing sample {sample_idx}...")
    print("Using ground truth field (computed via gnf_converter.mol2gnf)")
    
    # 获取分子的实际坐标和原子类型（用于计算ground truth field）
    atom_coords = batch.pos.cpu().numpy()  # [n_atoms, 3]
    atom_types = batch.x.cpu().numpy()  # [n_atoms]
    mol_x_center = (atom_coords[:, 0].min() + atom_coords[:, 0].max()) / 2
    mol_y_center = (atom_coords[:, 1].min() + atom_coords[:, 1].max()) / 2
    mol_z_center = (atom_coords[:, 2].min() + atom_coords[:, 2].max()) / 2
    
    # 硬编码显示范围：以分子中心为中心，范围从-axis_range到axis_range
    x_min, x_max = mol_x_center - axis_range, mol_x_center + axis_range
    y_min, y_max = mol_y_center - axis_range, mol_y_center + axis_range
    z_min, z_max = mol_z_center - axis_range, mol_z_center + axis_range
    
    print(f"Molecule coordinate range: X[{atom_coords[:, 0].min():.2f}, {atom_coords[:, 0].max():.2f}], "
          f"Y[{atom_coords[:, 1].min():.2f}, {atom_coords[:, 1].max():.2f}], "
          f"Z[{atom_coords[:, 2].min():.2f}, {atom_coords[:, 2].max():.2f}]")
    print(f"Molecule center: ({mol_x_center:.2f}, {mol_y_center:.2f}, {mol_z_center:.2f})")
    print(f"Hard-coded display bounds: X[{x_min:.2f}, {x_max:.2f}], Y[{y_min:.2f}, {y_max:.2f}], Z[{z_min:.2f}, {z_max:.2f}]")
    print(f"Display range: [-{axis_range}, {axis_range}] centered at molecule")
    
    # 只处理CHONF这五种元素（索引0,1,2,3,4）
    atom_types_to_visualize = [0, 1, 2, 3, 4]  # C, H, O, N, F
    atom_type_names = ['C', 'H', 'O', 'N', 'F']
    
    # 处理n_points_per_atom_type参数
    # 如果是整数，转换为字典（所有类型使用相同值）
    # 如果是字典，直接使用
    if n_points_per_atom_type is None:
        n_points_per_atom_type = 500  # 默认值
    if isinstance(n_points_per_atom_type, int):
        n_points_dict = {idx: n_points_per_atom_type for idx in atom_types_to_visualize}
    elif isinstance(n_points_per_atom_type, dict):
        n_points_dict = n_points_per_atom_type
        # 确保所有要可视化的类型都有值
        for idx in atom_types_to_visualize:
            if idx not in n_points_dict:
                n_points_dict[idx] = 500  # 默认值
    else:
        raise ValueError("n_points_per_atom_type must be an int or a dict")
    
    # 根据采样方式选择采样源
    if sample_from_grid:
        # 确定grid_dim和resolution（优先使用手动设置的参数）
        if grid_dim is None:
            grid_dim = dataset.grid_dim  # 从dataset配置读取
        if resolution is None:
            resolution = dataset.resolution  # 从dataset配置读取
        
        # 使用指定的grid_dim和resolution创建grid
        from funcmol.models.decoder import get_grid
        _, grid_coords = get_grid(grid_dim, resolution)
        grid_coords = grid_coords.cpu()  # 确保在CPU上
        n_grid_points = len(grid_coords)
        
        # 计算grid的总范围和点间隔
        # grid是从-half_span到half_span，共grid_dim个点
        total_span = (grid_dim - 1) * resolution
        half_span = total_span / 2
        
        print(f"Grid configuration (manually controlled):")
        print(f"  - Grid dimension: {grid_dim}x{grid_dim}x{grid_dim} = {n_grid_points} points")
        print(f"  - Resolution: {resolution:.2f} Å (distance between adjacent grid points)")
        print(f"  - Total span: {total_span:.2f} Å (from -{half_span:.2f} to {half_span:.2f})")
        print(f"  - Grid points range: X, Y, Z ∈ [-{half_span:.2f}, {half_span:.2f}] Å")
    else:
        # 从显示范围内任意采样（使用全局空间）
        grid_coords = None
        n_grid_points = None
        print(f"Sampling from global space within display bounds")
    
    # 为每种原子类型分别采样独立的点
    # 存储每种原子类型的采样点和向量场
    origins_by_atom_type = {}
    vector_fields_by_atom_type = {}
    
    for atom_type_idx in atom_types_to_visualize:
        # 获取该原子类型的采样点数量
        n_points = n_points_dict[atom_type_idx]
        
        # 如果采样点数量为0，跳过这个元素类型
        if n_points <= 0:
            print(f"Skipping {atom_type_names[atom_type_idx]} (atom type {atom_type_idx}) because n_points={n_points}")
            continue
        
        # 根据采样方式选择采样点
        if sample_from_grid:
            # 从grid点中采样（只使用grid点，不补充随机点）
            grid_coords_np = grid_coords.numpy()  # [n_grid_points, 3]
            
            # 先过滤掉超出显示范围的grid点
            in_range_mask = (
                (grid_coords_np[:, 0] >= x_min) & (grid_coords_np[:, 0] <= x_max) &
                (grid_coords_np[:, 1] >= y_min) & (grid_coords_np[:, 1] <= y_max) &
                (grid_coords_np[:, 2] >= z_min) & (grid_coords_np[:, 2] <= z_max)
            )
            grid_points_in_range = grid_coords_np[in_range_mask]
            n_grid_in_range = len(grid_points_in_range)
            
            print(f"  Grid points in display range: {n_grid_in_range} / {n_grid_points}")
            
            if n_grid_in_range == 0:
                print(f"Warning: No grid points in display range for {atom_type_names[atom_type_idx]}, skipping")
                continue
            
            # 从在显示范围内的grid点中采样
            if n_points >= n_grid_in_range:
                # 如果需要的点数超过范围内的grid点数，使用所有范围内的grid点
                query_points = grid_points_in_range
                print(f"  Using all {n_grid_in_range} grid points in range (requested {n_points})")
            else:
                # 随机采样n_points个范围内的grid点
                indices = np.random.choice(n_grid_in_range, n_points, replace=False)
                query_points = grid_points_in_range[indices]
                print(f"  Randomly sampled {n_points} grid points from {n_grid_in_range} in range")
        else:
            # 从显示范围内随机采样n_points个点
            # 使用均匀分布随机采样
            query_points = np.random.uniform(
                low=[x_min, y_min, z_min],
                high=[x_max, y_max, z_max],
                size=(n_points, 3)
            )
        
        # 将query_points转换为tensor（用于gnf_converter计算）
        query_points_tensor = torch.tensor(query_points, dtype=torch.float32).unsqueeze(0)  # [1, n_points, 3]
        
        # 准备原子坐标和类型（用于计算ground truth field）
        atom_coords_tensor = torch.tensor(atom_coords, dtype=torch.float32).unsqueeze(0)  # [1, n_atoms, 3]
        atom_types_tensor = torch.tensor(atom_types, dtype=torch.long).unsqueeze(0)  # [1, n_atoms]
        
        # 使用gnf_converter计算ground truth field
        with torch.no_grad():
            vector_field = gnf_converter.mol2gnf(
                atom_coords_tensor,  # [1, n_atoms, 3]
                atom_types_tensor,   # [1, n_atoms]
                query_points_tensor  # [1, n_points, 3]
            )  # [1, n_points, n_atom_types, 3]
        
        # 提取该原子类型的向量场
        vector_field_atom_type = vector_field[0, :, atom_type_idx, :].cpu().numpy()  # [n_points, 3]
        
        # 存储
        origins_by_atom_type[atom_type_idx] = query_points
        vector_fields_by_atom_type[atom_type_idx] = vector_field_atom_type
        
        sample_source = "grid" if sample_from_grid else "global space"
        print(f"Sampled {len(query_points)} independent points for {atom_type_names[atom_type_idx]} (atom type {atom_type_idx}) from {sample_source}")
    
    # 检查是否有显示环境
    has_display = os.environ.get('DISPLAY') is not None
    
    # 如果没有指定输出路径且没有显示环境，自动生成输出路径
    if not output_path and not has_display:
        # 自动生成输出文件名
        output_dir = os.path.join(project_root, "figures")
        os.makedirs(output_dir, exist_ok=True)
        output_filename = f"gradient_field_sample_{sample_idx}.png"
        output_path = os.path.join(output_dir, output_filename)
        print(f"No DISPLAY available, will save to: {output_path}")
    
    # 创建plotter（如果保存图片或没有显示环境，使用off_screen模式）
    # 设置高分辨率窗口大小以提高图片清晰度
    window_size = [1920, 1920]  # 高分辨率：1920x1920
    if output_path or not has_display:
        plotter = pv.Plotter(off_screen=True, window_size=window_size)
    else:
        plotter = pv.Plotter(window_size=window_size)
    
    # 为每种原子类型分别可视化（只处理CHONF）
    for atom_type_idx in atom_types_to_visualize:
        # 如果该原子类型没有采样点（n_points=0），跳过
        if atom_type_idx not in origins_by_atom_type:
            continue
        
        # 获取该原子类型的采样点和向量场
        origins_filtered_all = origins_by_atom_type[atom_type_idx]
        vectors_atom_type = vector_fields_by_atom_type[atom_type_idx]
        
        # 计算向量模长
        magnitudes = np.linalg.norm(vectors_atom_type, axis=1)
        threshold = magnitudes.max() * 0.01  # 过滤小于最大值1%的向量
        
        # 过滤有效向量
        valid_mask = magnitudes > threshold
        if valid_mask.sum() == 0:
            print(f"Warning: No valid vectors for atom type {atom_type_idx}")
            continue
        
        # 使用已经过滤后的origins
        origins_filtered = origins_filtered_all[valid_mask]
        vectors_filtered = vectors_atom_type[valid_mask]
        magnitudes_filtered = magnitudes[valid_mask]
        
        # 根据模长设置颜色
        # 默认：模长越小，颜色越深（重要性越大）
        # 如果use_real_arrow_length=True且large_gradient_light_color=True：模长越大，颜色越浅
        max_mag = magnitudes_filtered.max()
        min_mag = magnitudes_filtered.min()
        if max_mag > min_mag:
            # 归一化到[0, 1]
            normalized_mags = (magnitudes_filtered - min_mag) / (max_mag - min_mag)
            # 根据选项决定颜色映射方向
            if use_real_arrow_length and large_gradient_light_color:
                # 梯度大 -> 颜色浅（不反转，直接使用normalized_mags）
                color_values = normalized_mags  # 大模长 -> 大值 -> 浅色
            else:
                # 默认：梯度小 -> 颜色深（反转）
                color_values = 1.0 - normalized_mags  # 小模长 -> 大值 -> 深色
        else:
            color_values = np.ones(len(magnitudes_filtered))  # 所有点颜色相同
        
        # 获取基础颜色
        base_color = ATOM_COLORS.get(atom_type_idx, 'gray')
        # 将颜色名称转换为RGB
        if isinstance(base_color, str):
            # 使用matplotlib的颜色名称转换为RGB
            rgb_base = np.array(mcolors.to_rgb(base_color))
        else:
            rgb_base = np.array(base_color)
        
        # 根据color_values调整颜色深度和亮度
        # color_values越大（模长越小），颜色越深
        # 使用更鲜艳的颜色：提高最小亮度，让颜色更鲜艳
        # 使用HSV色彩空间来调整，保持色调，提高饱和度和亮度
        
        # 将基础RGB转换为HSV（rgb_base是单个颜色）
        hsv_base = np.array(mcolors.rgb_to_hsv(rgb_base))  # [H, S, V]
        
        # 使用透明度（opacity）来区分梯度大小，而不是混合白色
        # 梯度大 -> 透明度高（opacity低，看起来更透明/不明显）
        # 梯度小 -> 透明度低（opacity高，看起来更明显/突出）
        if use_real_arrow_length and large_gradient_light_color:
            # 梯度大 -> color_values大 -> opacity应该低（透明）
            # 使用非线性映射增强对比度
            enhanced_color_values = np.power(color_values, 0.7)
            opacity_min = 0.3  # 梯度大时的最小opacity（更透明）
            opacity_max = 1.0  # 梯度小时的最大opacity（完全不透明）
            opacity_values = opacity_min + (opacity_max - opacity_min) * (1.0 - enhanced_color_values)
        else:
            # 默认：梯度小 -> color_values大 -> opacity应该高（明显）
            # 使用非线性映射增强对比度
            enhanced_color_values = np.power(color_values, 0.7)
            opacity_min = 0.3  # 梯度大时的最小opacity（更透明）
            opacity_max = 1.0  # 梯度小时的最大opacity（完全不透明）
            opacity_values = opacity_min + (opacity_max - opacity_min) * enhanced_color_values
        
        # 保持RGB颜色不变（不混合白色，保持颜色可辨识）
        # 使用HSV调整，保持颜色鲜艳
        hsv_base = np.array(mcolors.rgb_to_hsv(rgb_base))
        # 提高饱和度，让颜色更鲜艳
        s_boost = 0.3
        if hsv_base[1] < 0.1:  # 如果是低饱和度颜色（如灰色），不增加饱和度
            s_values = hsv_base[1]
        else:
            s_values = np.clip(hsv_base[1] + s_boost, 0, 1)
        
        # 构建HSV颜色数组（每个点一个颜色，但色调和饱和度相同，只调整亮度以保持鲜艳）
        hsv_colors = np.zeros((len(color_values), 3))
        hsv_colors[:, 0] = hsv_base[0]  # 色调保持不变
        hsv_colors[:, 1] = s_values      # 提高饱和度
        hsv_colors[:, 2] = 1.0           # 保持最大亮度，让颜色鲜艳
        
        # 转换回RGB
        colors_rgb = mcolors.hsv_to_rgb(hsv_colors)
        colors_rgb = np.clip(colors_rgb, 0, 1)
        
        # 根据选项决定是否使用真实箭头长度
        if not use_real_arrow_length:
            # 将所有向量归一化为单位向量（保持方向，但长度统一）
            # 计算每个向量的模长
            vector_norms = np.linalg.norm(vectors_filtered, axis=1, keepdims=True)
            # 避免除以零
            vector_norms = np.where(vector_norms > 1e-10, vector_norms, 1.0)
            # 归一化为单位向量
            vectors_filtered_normalized = vectors_filtered / vector_norms
            # 放大到指定的长度
            vectors_for_glyph = vectors_filtered_normalized * normalized_arrow_length
        else:
            # 使用原始向量（真实梯度大小），但可以按比例缩放
            vectors_for_glyph = vectors_filtered * real_arrow_length_scale
        
        # 根据opacity调整颜色（模拟透明度效果）
        # 透明度高（opacity低）-> 颜色向背景色混合（变淡）
        # 透明度低（opacity高）-> 颜色保持鲜艳
        background_color = np.array([1.0, 1.0, 1.0])  # 白色背景
        colors_rgb_with_opacity = np.zeros_like(colors_rgb)
        
        for i in range(len(colors_rgb)):
            # 根据opacity调整颜色：opacity低时，向背景色混合
            # 混合比例 = 1 - opacity，这样opacity低时混合更多背景色（变淡/透明）
            mix_ratio = 1.0 - opacity_values[i]
            colors_rgb_with_opacity[i] = colors_rgb[i] * (1.0 - mix_ratio) + background_color * mix_ratio
        
        # 确保颜色在有效范围内
        colors_rgb_with_opacity = np.clip(colors_rgb_with_opacity, 0, 1)
        
        # 创建向量场
        pdata = pv.vector_poly_data(origins_filtered, vectors_for_glyph)
        
        # 将调整后的颜色信息添加到数据中（使用RGB格式，保持颜色可辨识）
        # PyVista使用uint8格式的RGB颜色（0-255范围）
        colors_rgb_uint8 = (colors_rgb_with_opacity * 255).astype(np.uint8)
        pdata['colors'] = colors_rgb_uint8
        
        # 生成箭头
        # 创建自定义箭头几何体，通过arrow_scale控制粗细
        # 箭头粗细通过arrow的shaft_radius和tip_radius控制
        arrow_geom = pv.Arrow(
            tip_radius=0.1 * arrow_scale,
            shaft_radius=0.05 * arrow_scale,
            tip_length=0.25,
            shaft_resolution=20,
            tip_resolution=20
        )
        
        if use_real_arrow_length:
            # 使用真实模长作为箭头长度
            glyph = pdata.glyph(orient='vectors', scale='mag', geom=arrow_geom)
        else:
            # 所有箭头使用相同的长度（已经归一化并放大到normalized_arrow_length）
            glyph = pdata.glyph(orient='vectors', scale=True, factor=1.0, geom=arrow_geom)
        
        # 获取名称
        atom_name = ELEMENTS_HASH_INV.get(atom_type_idx, f"Type{atom_type_idx}")
        
        # 使用RGB颜色添加到plotter（颜色已经根据opacity调整过，模拟透明度效果）
        plotter.add_mesh(glyph, scalars='colors', rgb=True, label=f"{atom_name}", opacity=1.0)
        
        print(f"Added {valid_mask.sum()} vectors for {atom_name} (base color: {base_color}, magnitude range: [{min_mag:.4f}, {max_mag:.4f}])")
    
    # 在真实原子位置绘制小球（如果启用）
    if show_atoms:
        # 获取原子类型（注意：PyG Batch中原子类型存储在batch.x中）
        atom_types = batch.x.cpu().numpy()  # [n_atoms]
        
        # 确定要显示的元素类型
        if show_atom_types is None:
            # 如果未指定，显示所有CHONF元素
            atom_types_to_show = atom_types_to_visualize
        else:
            # 如果指定了，只显示指定的元素类型
            if isinstance(show_atom_types, (list, tuple)):
                atom_types_to_show = [int(idx) for idx in show_atom_types]
            else:
                atom_types_to_show = [int(show_atom_types)]
            print(f"Only showing atoms of types: {atom_types_to_show}")
        
        # 为每种原子类型绘制小球
        for atom_type_idx in atom_types_to_show:
            # 找到该类型的原子
            atom_mask = (atom_types == atom_type_idx)
            if not atom_mask.any():
                continue
            
            atom_positions = atom_coords[atom_mask]  # [n_atoms_of_this_type, 3]
            
            # 获取该原子类型的基础颜色
            base_color = ATOM_COLORS.get(atom_type_idx, 'gray')
            if isinstance(base_color, str):
                rgb_base = np.array(mcolors.to_rgb(base_color))
            else:
                rgb_base = np.array(base_color)
            
            # 转换为uint8格式
            color_uint8 = (rgb_base * 255).astype(np.uint8)
            
            # 为每个原子创建小球
            for pos in atom_positions:
                # 创建球体
                sphere = pv.Sphere(radius=atom_radius, center=pos, theta_resolution=20, phi_resolution=20)
                
                # 根据样式选择绘制方式
                if atom_style == 'transparent':
                    # 透明球体
                    plotter.add_mesh(sphere, color=color_uint8, opacity=0.3, show_edges=False)
                elif atom_style == 'wireframe':
                    # 3D线框球体（只显示边缘，使用style='wireframe'）
                    # 减少线条密度：降低theta_resolution和phi_resolution
                    # 使用更细的线条和更低的opacity，让它更不醒目
                    sphere_wireframe = pv.Sphere(
                        radius=atom_radius, 
                        center=pos, 
                        theta_resolution=8,   # 减少经线数量（默认20）
                        phi_resolution=8      # 减少纬线数量（默认20）
                    )
                    plotter.add_mesh(
                        sphere_wireframe, 
                        color=color_uint8, 
                        style='wireframe', 
                        line_width=1,         # 更细的线条
                        opacity=0.5           # 降低不透明度，让它更不醒目
                    )
                elif atom_style == 'circle_2d':
                    # 2D虚线圆圈（在XY平面上）
                    # 使用较少的点来创建虚线效果（点之间有空隙）
                    n_segments = 16  # 虚线段的数量（更少，看起来更松散）
                    angles = np.linspace(0, 2 * np.pi, n_segments, endpoint=False)
                    
                    # 创建多个小线段来模拟虚线圆圈
                    all_points = []
                    all_lines = []
                    point_idx = 0
                    
                    for i in range(n_segments):
                        # 每个虚线段由2个点组成（起点和终点）
                        angle_start = angles[i]
                        angle_end = angles[(i + 1) % n_segments]
                        
                        # 起点
                        point_start = np.array([
                            pos[0] + atom_radius * np.cos(angle_start),
                            pos[1] + atom_radius * np.sin(angle_start),
                            pos[2]
                        ])
                        # 终点（稍微缩短，创建虚线间隙效果）
                        gap_factor = 0.85  # 缩短15%，创建间隙
                        point_end = np.array([
                            pos[0] + atom_radius * gap_factor * np.cos(angle_end),
                            pos[1] + atom_radius * gap_factor * np.sin(angle_end),
                            pos[2]
                        ])
                        
                        all_points.append(point_start)
                        all_points.append(point_end)
                        
                        # 创建线段：2个点，从point_idx到point_idx+1
                        all_lines.extend([2, point_idx, point_idx + 1])
                        point_idx += 2
                    
                    # 创建PolyData
                    circle_poly = pv.PolyData(np.array(all_points))
                    circle_poly.lines = np.array(all_lines)
                    
                    # 添加虚线圆圈（使用细线条和较低opacity，让它更不醒目）
                    plotter.add_mesh(
                        circle_poly,
                        color=color_uint8,
                        line_width=1,
                        opacity=0.5,  # 降低不透明度，让它更不醒目
                        render_lines_as_tubes=False
                    )
                else:
                    # 默认使用透明样式
                    plotter.add_mesh(sphere, color=color_uint8, opacity=0.3, show_edges=False)
            
            atom_name = ELEMENTS_HASH_INV.get(atom_type_idx, f"Type{atom_type_idx}")
            print(f"Added {atom_mask.sum()} {atom_name} atoms (style: {atom_style}, radius: {atom_radius})")
    
    # 计算显示中心（用于设置相机焦点）
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    z_center = (z_min + z_max) / 2
    
    # 计算显示范围大小，并添加边距以避免箭头被截断
    range_size = max(x_max - x_min, y_max - y_min, z_max - z_min)
    padding = range_size * 0.2  # 增加边距，确保有足够空间
    
    # 扩展边界以包含边距
    bounds_with_padding = [
        x_min - padding, x_max + padding,
        y_min - padding, y_max + padding,
        z_min - padding, z_max + padding
    ]
    
    # 设置相机裁剪范围
    plotter.camera.clipping_range = (0.1, 1000.0)
    
    # 设置视图中心
    plotter.camera.focal_point = (x_center, y_center, z_center)
    
    # 重置相机以适应新的边界（包含边距，确保视图范围足够大）
    plotter.reset_camera(bounds=bounds_with_padding)
    
    # 根据elev和azim参数设置相机视角
    # 计算相机距离（基于显示范围的大小，增加距离以确保视野足够大）
    camera_distance = range_size * 4.0  # 增加相机距离到4.0，确保视野足够大
    
    # 将角度转换为弧度
    elev_rad = np.radians(elev)
    azim_rad = np.radians(azim)
    
    # 根据仰角和方位角计算相机位置
    # 在球坐标系中：x = r * sin(elev) * cos(azim), y = r * sin(elev) * sin(azim), z = r * cos(elev)
    # 但PyVista的坐标系可能不同，需要调整
    # 通常：azim是水平旋转（xy平面），elev是垂直角度（从水平面向上）
    camera_x = x_center + camera_distance * np.cos(elev_rad) * np.cos(azim_rad)
    camera_y = y_center + camera_distance * np.cos(elev_rad) * np.sin(azim_rad)
    camera_z = z_center + camera_distance * np.sin(elev_rad)
    
    # 设置相机位置
    plotter.camera.position = (camera_x, camera_y, camera_z)
    plotter.camera.focal_point = (x_center, y_center, z_center)
    plotter.camera.up = (0, 0, 1)  # 设置上方向为z轴正方向
    
    # 再次确保相机焦点和边界正确（使用带边距的边界）
    plotter.reset_camera(bounds=bounds_with_padding)
    
    # 设置相机的视野角度（FOV），确保能看到更多内容
    # 增大视野角度可以让相机看到更大的范围
    plotter.camera.view_angle = 40.0  # 默认是30度，增大到60度可以看到更多内容
    
    # 不添加图例（用户要求去掉右上角的图注）
    # legend_labels = [f"{atom_type_names[i]}" for i in atom_types_to_visualize]
    # if legend_labels:
    #     plotter.add_legend(labels=legend_labels)
    
    # 显示或保存
    if output_path:
        # 保存图片前，再次确保相机设置正确
        plotter.reset_camera(bounds=bounds_with_padding)
        plotter.camera.focal_point = (x_center, y_center, z_center)
        # 重新应用视角设置
        camera_distance = range_size * 3.0  # 使用相同的相机距离
        elev_rad = np.radians(elev)
        azim_rad = np.radians(azim)
        camera_x = x_center + camera_distance * np.cos(elev_rad) * np.cos(azim_rad)
        camera_y = y_center + camera_distance * np.cos(elev_rad) * np.sin(azim_rad)
        camera_z = z_center + camera_distance * np.sin(elev_rad)
        plotter.camera.position = (camera_x, camera_y, camera_z)
        plotter.camera.up = (0, 0, 1)
        plotter.camera.view_angle = 40.0  # 增大视野角度
        
        # 保存图片（off_screen模式下直接截图）
        # 确保输出目录存在
        output_dir = os.path.dirname(output_path) if os.path.dirname(output_path) else '.'
        os.makedirs(output_dir, exist_ok=True)
        # 使用高分辨率截图（scale参数可以进一步提高分辨率）
        # scale=2表示2倍分辨率，最终分辨率 = window_size * scale = 1920*2 = 3840x3840
        plotter.screenshot(output_path, scale=2)
        print(f"Saved high-resolution visualization to: {output_path} (resolution: {window_size[0]*2}x{window_size[1]*2})")
        plotter.close()
    elif not has_display:
        # 没有显示环境，即使没有指定输出路径也应该保存
        # 这种情况应该已经在上面处理了，这里作为保险
        output_dir = os.path.join(project_root, "figures")
        os.makedirs(output_dir, exist_ok=True)
        output_filename = f"gradient_field_sample_{sample_idx}.png"
        output_path = os.path.join(output_dir, output_filename)
        # 使用高分辨率截图
        # scale=2表示2倍分辨率，最终分辨率 = window_size * scale = 1920*2 = 3840x3840
        plotter.screenshot(output_path, scale=2)
        print(f"Saved high-resolution visualization to: {output_path} (resolution: {window_size[0]*2}x{window_size[1]*2})")
        plotter.close()
    else:
        # 显示交互式窗口
        plotter.show()


def find_simple_molecules(config_path, split="val", n_samples=30, nf_pretrained_path=None):
    """
    扫描数据集中的前N个分子，找出原子数量最少的简单分子
    
    Args:
        config_path: 配置文件路径
        split: 数据集分割（默认"val"）
        n_samples: 扫描的分子数量（默认30）
        nf_pretrained_path: checkpoint路径（可选，扫描时不需要）
    
    Returns:
        list: 包含 (sample_idx, n_atoms, atom_types_dict) 的列表，按原子数量排序
    """
    # 确保配置文件路径是绝对路径
    if not os.path.isabs(config_path):
        config_path = os.path.join(project_root, config_path)
    
    # 加载配置：优先从 checkpoint 加载（如果提供），否则使用 Hydra 加载配置文件
    from omegaconf import OmegaConf
    
    if nf_pretrained_path and os.path.exists(nf_pretrained_path):
        # 如果提供了 checkpoint，从 checkpoint 加载配置（与 load_config_and_models 逻辑一致）
        print(f"Loading config from checkpoint: {nf_pretrained_path}")
        checkpoint = torch.load(nf_pretrained_path, map_location='cpu', weights_only=False)
        if 'hyper_parameters' in checkpoint:
            config = checkpoint['hyper_parameters']
        else:
            config = checkpoint.get('config', {})
        
        # 转换为普通字典
        try:
            import omegaconf.dictconfig
            if isinstance(config, omegaconf.dictconfig.DictConfig):
                config = OmegaConf.to_container(config, resolve=True)
        except (ImportError, AttributeError):
            if hasattr(OmegaConf, 'is_dict') and OmegaConf.is_dict(config):
                config = OmegaConf.to_container(config, resolve=True)
        
        # 加载 yaml 配置并合并
        yaml_config = OmegaConf.load(config_path)
        yaml_config = OmegaConf.to_container(yaml_config, resolve=True)
        for key in yaml_config.keys():
            if key in config and isinstance(config[key], dict) and isinstance(yaml_config[key], dict):
                config[key].update(yaml_config[key])
            else:
                config[key] = yaml_config[key]
    else:
        # 如果没有 checkpoint，使用 Hydra 加载配置
        try:
            import hydra
            from hydra import initialize_config_dir, compose
            
            # 获取配置文件所在目录和文件名
            config_dir = os.path.dirname(config_path)
            config_name = os.path.basename(config_path).replace('.yaml', '')
            
            # 使用 Hydra 初始化并加载配置
            with initialize_config_dir(config_dir=config_dir, version_base=None):
                cfg = compose(config_name=config_name)
                config = OmegaConf.to_container(cfg, resolve=True)
        except Exception as e:
            # 如果 Hydra 加载失败，尝试直接使用 OmegaConf
            print(f"Warning: Failed to load config with Hydra: {e}, trying direct OmegaConf.load()")
            config = OmegaConf.load(config_path)
            config = OmegaConf.to_container(config, resolve=True)
    
    # 处理数据路径
    if "dset" in config and "data_dir" in config["dset"]:
        data_dir = config["dset"]["data_dir"]
        if not os.path.isabs(data_dir):
            if data_dir.startswith("dataset/data"):
                config["dset"]["data_dir"] = os.path.join(project_root, "funcmol", data_dir)
            else:
                config["dset"]["data_dir"] = os.path.join(project_root, data_dir)
    
    # 创建GNFConverter（只需要用于数据集加载）
    from funcmol.dataset.dataset_field import create_gnf_converter
    gnf_converter = create_gnf_converter(config)
    
    # 创建数据加载器
    config["split"] = split
    loader = create_field_loaders(config, gnf_converter, split=split)
    
    # 禁用shuffle以保持顺序
    loader = DataLoader(
        loader.dataset,
        batch_size=1,
        num_workers=0,
        shuffle=False,
        pin_memory=False,
    )
    
    dataset = loader.dataset
    n_samples = min(n_samples, len(dataset))
    
    print(f"\n扫描 {split} 数据集中的前 {n_samples} 个分子...")
    print("=" * 80)
    
    molecule_info = []
    
    for sample_idx in range(n_samples):
        try:
            sample_data = dataset[sample_idx]
            from torch_geometric.data import Batch
            if not hasattr(sample_data, 'batch') or sample_data.batch is None:
                sample_data.batch = torch.zeros(sample_data.pos.shape[0], dtype=torch.long)
            batch = Batch.from_data_list([sample_data])
            
            # 获取原子信息
            atom_coords = batch.pos.cpu().numpy()  # [n_atoms, 3]
            atom_types = batch.x.cpu().numpy()  # [n_atoms]
            n_atoms = len(atom_coords)
            
            # 统计每种原子类型的数量
            atom_types_dict = {}
            for atom_type_idx in range(5):  # C, H, O, N, F
                count = (atom_types == atom_type_idx).sum()
                if count > 0:
                    atom_name = ELEMENTS_HASH_INV.get(atom_type_idx, f"Type{atom_type_idx}")
                    atom_types_dict[atom_name] = int(count)
            
            molecule_info.append((sample_idx, n_atoms, atom_types_dict))
            
        except Exception as e:
            print(f"Warning: Failed to process sample {sample_idx}: {e}")
            continue
    
    # 按原子数量排序
    molecule_info.sort(key=lambda x: x[1])
    
    # 打印结果
    print(f"\n前 {n_samples} 个分子按原子数量排序（从少到多）：")
    print("-" * 80)
    print(f"{'索引':<8} {'原子数':<8} {'原子组成':<40}")
    print("-" * 80)
    
    for sample_idx, n_atoms, atom_types_dict in molecule_info:
        atom_composition = ", ".join([f"{name}:{count}" for name, count in sorted(atom_types_dict.items())])
        print(f"{sample_idx:<8} {n_atoms:<8} {atom_composition:<40}")
    
    print("-" * 80)
    print(f"\n最简单的分子：索引 {molecule_info[0][0]}，原子数 {molecule_info[0][1]}")
    print(f"原子组成：{', '.join([f'{name}:{count}' for name, count in sorted(molecule_info[0][2].items())])}")
    
    return molecule_info


def main():
    # 在这里直接修改参数
    config_path = "funcmol/configs/infer_codes.yaml"  # 配置文件路径
    nf_pretrained_path = None  # 已废弃，不再使用（保留以保持兼容性）
    sample_idx = 5  # 要可视化的样本索引
    # 每种原子类型的采样点数量（可以分别为每种元素设置）
    # 格式：字典，键是原子类型索引（0=C, 1=H, 2=O, 3=N, 4=F）
    # 也可以使用整数，表示所有类型使用相同的数量
    n_points_per_atom_type = {
        0: 500,  # C
        1: 0,  # H
        2: 0,  # O
        3: 0,  # N
        4: 0,  # F
    }
    split = "val"  # 数据集分割：train, val, test
    
    # 可选：扫描前30个分子，找出原子数量最少的简单分子
    # 取消下面的注释来运行扫描（扫描完成后会自动退出，不进行可视化）
    SCAN_MOLECULES = False  # 设置为 True 来运行扫描
    if SCAN_MOLECULES:
        find_simple_molecules(config_path, split=split, n_samples=3000, nf_pretrained_path=None)
        exit(0)  # 扫描完成后退出，不进行可视化
    output_path = None  # 输出文件路径（None则自动保存到figures目录）
    elev = 30.0  # 仰角
    azim = 60.0  # 方位角
    axis_range = 2  # 坐标轴范围，例如3.0表示从-3到3
    use_real_arrow_length = True  # 是否使用真实梯度大小作为箭头长度（True）还是所有箭头使用相同长度（False）
    normalized_arrow_length = 0.3  # 当use_real_arrow_length=False时，归一化后箭头放大到的长度
    real_arrow_length_scale = 0.6  # 当use_real_arrow_length=True时，真实箭头长度的缩放因子（值越大箭头越长）
    arrow_scale = 1.5  # 箭头粗细缩放因子，值越大箭头越粗
    sample_from_grid = True  # 是否只从grid点采样（True）还是从显示范围内任意采样（False）
    grid_dim = None  # grid的维度（如64表示64x64x64），None则从dataset配置读取
    resolution = 0.9 # grid点之间的间距（单位：埃），None则从dataset配置读取，例如0.25
    large_gradient_light_color = True  # 当use_real_arrow_length=True时，如果为True，则梯度大的箭头颜色更浅
    show_atoms = True  # 是否在真实原子位置显示小球
    atom_style = 'wireframe'  # 原子小球的样式：'transparent'（透明）、'wireframe'（3D线框）或'circle_2d'（2D虚线圆圈）
    atom_radius = 0.1  # 原子小球的半径（单位：埃）
    show_atom_types = [0]  # 要显示的元素类型列表，例如[0, 1]表示只显示C和H。如果为None，则显示所有元素
    
    # 确保配置文件路径是绝对路径
    if not os.path.isabs(config_path):
        config_path = os.path.join(project_root, config_path)
    
    visualize_gradient_field(
        config_path=config_path,
        nf_pretrained_path=nf_pretrained_path,
        sample_idx=sample_idx,
        n_points_per_atom_type=n_points_per_atom_type,
        split=split,
        output_path=output_path,
        elev=elev,
        azim=azim,
        axis_range=axis_range,
        use_real_arrow_length=use_real_arrow_length,
        normalized_arrow_length=normalized_arrow_length,
        real_arrow_length_scale=real_arrow_length_scale,
        arrow_scale=arrow_scale,
        sample_from_grid=sample_from_grid,
        grid_dim=grid_dim,
        resolution=resolution,
        large_gradient_light_color=large_gradient_light_color,
        show_atoms=show_atoms,
        atom_style=atom_style,
        atom_radius=atom_radius,
        show_atom_types=show_atom_types,
    )


if __name__ == "__main__":
    main()

