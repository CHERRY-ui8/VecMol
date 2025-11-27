"""
3D矢量场可视化示例脚本
快速生成用于Figure 1的场可视化
"""

import torch
import os
import sys
from omegaconf import OmegaConf

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from visualize_field_3d import FieldVisualizer3D
from funcmol.utils.gnf_converter import GNFConverter

def create_converter_from_config(config_path: str) -> GNFConverter:
    """从配置文件创建GNFConverter的辅助函数"""
    config = OmegaConf.load(config_path)
    
    # 获取方法特定配置
    gradient_field_method = config.get('gradient_field_method', 'tanh')
    method_configs = config.get('method_configs', {})
    default_config = config.get('default_config', {})
    
    # 根据方法选择参数
    if gradient_field_method in method_configs:
        method_config = method_configs[gradient_field_method]
        n_query_points = method_config.get('n_query_points', 1000)
        step_size = method_config.get('step_size', 0.05)
        sig_sf = method_config.get('sig_sf', 0.1)
        sig_mag = method_config.get('sig_mag', 0.45)
        eps = method_config.get('eps', default_config.get('eps', 0.15))
        min_samples = method_config.get('min_samples', default_config.get('min_samples', 20))
    else:
        n_query_points = default_config.get('n_query_points', 600)
        step_size = default_config.get('step_size', 0.02)
        sig_sf = default_config.get('sig_sf', 0.1)
        sig_mag = default_config.get('sig_mag', 0.45)
        eps = default_config.get('eps', 0.15)
        min_samples = default_config.get('min_samples', 20)
    
    # 创建GNFConverter
    converter = GNFConverter(
        sigma=config.get('sigma', 1.0),
        n_query_points=n_query_points,
        n_iter=config.get('n_iter', 500),
        step_size=step_size,
        eps=eps,
        min_samples=min_samples,
        sigma_ratios=config.get('sigma_ratios', {}),
        gradient_field_method=gradient_field_method,
        temperature=config.get('temperature', 1.0),
        logsumexp_eps=config.get('logsumexp_eps', 1e-8),
        inverse_square_strength=config.get('inverse_square_strength', 1.0),
        gradient_clip_threshold=config.get('gradient_clip_threshold', 0.3),
        sig_sf=sig_sf,
        sig_mag=sig_mag,
        gradient_sampling_candidate_multiplier=config.get('gradient_sampling_candidate_multiplier', 3),
        n_atom_types=5,  # QM9有5种原子类型：C, H, O, N, F
        enable_early_stopping=config.get('enable_early_stopping', False),
        convergence_threshold=config.get('convergence_threshold', 1e-6),
        min_iterations=config.get('min_iterations', 50)
    )
    
    return converter

def example_visualize_single_molecule():
    """示例：可视化单个分子的矢量场"""
    
    # 加载配置并创建converter
    config_path = 'funcmol/configs/converter/gnf_converter_qm9.yaml'
    converter = create_converter_from_config(config_path)
    
    # 创建可视化器
    visualizer = FieldVisualizer3D(
        converter=converter,
        grid_resolution=15,  # 可以调整，值越大越精细但越慢
        arrow_scale=0.5,     # 箭头大小
        arrow_density=0.3    # 箭头密度
    )
    
    # 创建一个简单的测试分子（例如：甲烷）
    # 你可以替换为实际的分子数据
    coords = torch.tensor([
        [0.0, 0.0, 0.0],      # C (中心)
        [1.09, 0.0, 0.0],     # H
        [-1.09, 0.0, 0.0],    # H
        [0.0, 1.09, 0.0],     # H
        [0.0, -1.09, 0.0],    # H
    ]).float()
    
    atom_types = torch.tensor([0, 1, 1, 1, 1]).long()  # C, H, H, H, H
    
    # 创建输出目录
    output_dir = 'field_visualizations'
    os.makedirs(output_dir, exist_ok=True)
    
    print("生成3D矢量场可视化...")
    
    # 1. 箭头图可视化（最直观，推荐用于Figure 1）
    print("  - 生成箭头图（C类型场）...")
    fig1 = visualizer.visualize_quiver(
        coords=coords,
        atom_types=atom_types,
        atom_type_idx=0,  # C类型
        save_path=os.path.join(output_dir, 'field_quiver_C.png'),
        show_molecule=True,
        show_arrows=True,
        max_arrows=500,
        arrow_length_scale=1.0,
        figsize=(14, 12)
    )
    
    print("  - 生成箭头图（H类型场）...")
    fig2 = visualizer.visualize_quiver(
        coords=coords,
        atom_types=atom_types,
        atom_type_idx=1,  # H类型
        save_path=os.path.join(output_dir, 'field_quiver_H.png'),
        show_molecule=True,
        show_arrows=True,
        max_arrows=500,
        arrow_length_scale=1.0,
        figsize=(14, 12)
    )
    
    # 2. 多类型场对比（展示所有原子类型的场）
    print("  - 生成多类型场对比图...")
    fig3 = visualizer.visualize_multi_type_field(
        coords=coords,
        atom_types=atom_types,
        save_path=os.path.join(output_dir, 'field_multi_type.png'),
        atom_types_to_show=[0, 1],  # C和H
        figsize=(16, 8)
    )
    
    # 3. 场强度可视化（使用点云颜色）
    print("  - 生成场强度图...")
    fig4 = visualizer.visualize_field_intensity(
        coords=coords,
        atom_types=atom_types,
        atom_type_idx=0,  # C类型
        save_path=os.path.join(output_dir, 'field_intensity_C.png'),
        figsize=(14, 12)
    )
    
    print(f"\n所有可视化已保存到: {output_dir}/")
    print("\n推荐使用箭头图（quiver）用于Figure 1，因为它最直观地展示了矢量场的方向和强度")


def example_visualize_from_dataset():
    """示例：可视化多个原子类型的场"""
    
    # 加载配置并创建converter
    config_path = 'funcmol/configs/converter/gnf_converter_qm9.yaml'
    converter = create_converter_from_config(config_path)
    
    # 使用一个更复杂的测试分子（包含多种原子类型）
    # 例如：甲醇 CH3OH
    coords = torch.tensor([
        [0.0, 0.0, 0.0],      # C
        [1.09, 0.0, 0.0],     # H
        [-0.36, 1.02, 0.0],   # H
        [-0.36, -0.51, 0.88], # H
        [0.0, 0.0, 1.43],     # O
        [0.0, 0.0, 2.48],     # H (OH)
    ]).float()
    
    atom_types = torch.tensor([0, 1, 1, 1, 2, 1]).long()  # C, H, H, H, O, H
    
    # 创建可视化器
    visualizer = FieldVisualizer3D(
        converter=converter,
        grid_resolution=15,
        arrow_scale=0.5,
        arrow_density=0.3
    )
    
    # 生成可视化
    output_dir = 'field_visualizations'
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"可视化分子（{len(coords)}个原子）...")
    
    # 为每个原子类型生成可视化
    for atom_type_idx in range(5):  # C, H, O, N, F
        atom_name = ['C', 'H', 'O', 'N', 'F'][atom_type_idx]
        # 检查该分子是否包含这种原子类型
        if (atom_types == atom_type_idx).any():
            print(f"  - 生成{atom_name}类型场...")
            
            fig = visualizer.visualize_quiver(
                coords=coords,
                atom_types=atom_types,
                atom_type_idx=atom_type_idx,
                save_path=os.path.join(output_dir, f'field_quiver_{atom_name}.png'),
                show_molecule=True,
                show_arrows=True,
                max_arrows=800,  # 可以增加箭头数量
                arrow_length_scale=1.0,
                figsize=(14, 12)
            )
        else:
            print(f"  - 跳过{atom_name}类型（分子中不包含）")
    
    print(f"\n所有可视化已保存到: {output_dir}/")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='3D Field Visualization Example')
    parser.add_argument('--mode', type=str, default='single', 
                       choices=['single', 'dataset'],
                       help='Visualization mode: single molecule or from dataset')
    
    args = parser.parse_args()
    
    if args.mode == 'single':
        example_visualize_single_molecule()
    elif args.mode == 'dataset':
        example_visualize_from_dataset()

