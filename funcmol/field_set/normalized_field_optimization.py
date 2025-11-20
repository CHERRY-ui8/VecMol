import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path
from omegaconf import OmegaConf

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from funcmol.utils.gnf_converter import GNFConverter

# 加载配置文件
def load_config():
    """从配置文件加载参数"""
    config_path = Path(__file__).parent.parent / "configs" / "converter" / "gnf_converter_qm9.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    config = OmegaConf.load(config_path)
    return config

# 全局配置
CONFIG = load_config()

def analyze_real_space_sig_sf():
    """
    分析真实空间中的sig_sf参数
    """
    print("=== 真实空间中的sig_sf参数分析 ===")
    
    # 根据配置文件分析
    grid_dim = 48
    resolution = 0.25
    max_diameter = grid_dim * resolution  # 12
    
    # 典型的原子间距（真实空间）
    # 真实C-C键长约为1.5埃
    real_atom_distance = 1.5
    target_decay_distance = real_atom_distance * 0.5  # 0.75
    
    print(f"最大直径: {max_diameter} 埃")
    print(f"真实原子间距: {real_atom_distance}")
    print(f"目标衰减距离: {target_decay_distance}")
    
    # 测试更合理的sig_sf范围（针对真实空间）
    sig_sf_values = np.linspace(0.1, 1.0, 50)
    threshold = 0.05  # field衰减阈值
    decay_distances = []
    
    for sig_sf in sig_sf_values:
        # 模拟两个原子，距离为real_atom_distance
        for dist_a in np.arange(0, real_atom_distance, 0.01):
            dist_b = real_atom_distance - dist_a
            val = np.exp(-dist_a / sig_sf) / (np.exp(-dist_a / sig_sf) + np.exp(-dist_b / sig_sf))
            if val < threshold:
                decay_distances.append(dist_a)
                break
        else:
            decay_distances.append(real_atom_distance)
    
    # 找到最接近目标衰减距离的sig_sf值
    target_idx = np.argmin(np.abs(np.array(decay_distances) - target_decay_distance))
    best_sig_sf = sig_sf_values[target_idx]
    actual_decay = decay_distances[target_idx]
    
    print(f"推荐的sig_sf值: {best_sig_sf:.4f}")
    print(f"实际衰减距离: {actual_decay:.4f}")
    
    # 绘制结果
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    plt.plot(sig_sf_values, decay_distances, 'b-', linewidth=2)
    plt.axhline(y=target_decay_distance, color='r', linestyle='--', alpha=0.7, label=f'Target distance: {target_decay_distance:.4f}')
    plt.axvline(x=best_sig_sf, color='g', linestyle='--', alpha=0.7, label=f'Recommended sig_sf: {best_sig_sf:.4f}')
    plt.xlabel('sig_sf value')
    plt.ylabel('Decay distance (Angstroms)')
    plt.title('sig_sf vs Decay Distance in Real Space')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    return best_sig_sf, real_atom_distance

def test_real_space_field_behavior():
    """
    测试真实空间中的field行为
    """
    print("\n=== 测试真实空间中的field行为 ===")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, real_atom_distance = analyze_real_space_sig_sf()
    
    # 创建测试场景：两个相邻原子（真实坐标）
    coords = torch.tensor([[[-real_atom_distance/2, 0.0, 0.0], [real_atom_distance/2, 0.0, 0.0]]], device=device)
    atom_types = torch.tensor([[0, 0]], device=device)
    
    # 沿着连接线的查询点（真实空间）
    query_points = torch.linspace(-3.0, 3.0, 200, device=device).reshape(-1, 1)
    query_points = torch.cat([query_points, torch.zeros_like(query_points), torch.zeros_like(query_points)], dim=1)
    query_points = query_points.unsqueeze(0)
    
    sigma_ratios = {'C': 1.0, 'H': 1.0, 'O': 1.0, 'N': 1.0, 'F': 1.0}
    
    # 测试三种方法（包括tanh）
    methods = ['tanh', 'gaussian_mag', 'distance']
    
    plt.subplot(2, 3, 2)
    
    for method in methods:
        # 从配置文件读取参数
        if hasattr(CONFIG.method_configs, method):
            method_config = getattr(CONFIG.method_configs, method)
            sig_sf = method_config.sig_sf
            sig_mag = method_config.sig_mag
            step_size = method_config.step_size
            n_query_points = method_config.n_query_points
            eps = getattr(method_config, 'eps', CONFIG.default_config.eps)
            min_samples = getattr(method_config, 'min_samples', CONFIG.default_config.min_samples)
        else:
            # distance方法使用默认配置
            method_config = CONFIG.default_config
            sig_sf = method_config.sig_sf
            sig_mag = method_config.sig_mag
            step_size = method_config.step_size
            n_query_points = method_config.n_query_points
            eps = method_config.eps
            min_samples = method_config.min_samples
        
        converter = GNFConverter(
            sigma=CONFIG.sigma,
            n_query_points=n_query_points,
            n_iter=CONFIG.n_iter,
            step_size=step_size,
            eps=eps,
            min_samples=min_samples,
            sigma_ratios=sigma_ratios,
            gradient_field_method=method,
            sig_sf=sig_sf,
            sig_mag=sig_mag,
            temperature=CONFIG.temperature,
            logsumexp_eps=CONFIG.logsumexp_eps,
            inverse_square_strength=CONFIG.inverse_square_strength,
            gradient_clip_threshold=CONFIG.gradient_clip_threshold,
            gradient_sampling_candidate_multiplier=CONFIG.gradient_sampling_candidate_multiplier,
            gradient_sampling_temperature=CONFIG.gradient_sampling_temperature,
            n_atom_types=5
        )
        
        vector_field = converter.mol2gnf(coords, atom_types, query_points)
        field_values = vector_field[0, :, 0, 0].cpu().numpy()
        positions = query_points[0, :, 0].cpu().numpy()
        
        plt.plot(positions, field_values, label=f'{method}', linewidth=2)
    
    plt.axvline(x=-real_atom_distance/2, color='r', linestyle='--', alpha=0.7, label='Atom 1')
    plt.axvline(x=real_atom_distance/2, color='g', linestyle='--', alpha=0.7, label='Atom 2')
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.xlabel('Position (Angstroms)')
    plt.ylabel('Field value')
    plt.title('Field Distribution in Real Space')
    plt.legend()
    plt.grid(True, alpha=0.3)

def analyze_sig_mag_for_real_space():
    """
    分析真实空间中的sig_mag参数
    """
    print("\n=== 分析真实空间中的sig_mag参数 ===")
    
    # 测试不同的sig_mag值（针对真实空间）
    sig_mag_values = np.linspace(0.3, 2.0, 50)
    
    plt.subplot(2, 3, 3)
    
    # 测试三种magnitude方法（包括tanh）
    methods = ['tanh', 'gaussian_mag', 'distance']
    colors = ['b', 'g', 'r']
    
    for i, method in enumerate(methods):
        magnitude_values = []
        for sig_mag in sig_mag_values:
            # 计算在距离0.75处的magnitude值（真实空间中的典型距离）
            dist = 0.75
            if method == 'tanh':
                mag = np.tanh(dist / sig_mag)
            elif method == 'gaussian_mag':
                mag = np.exp(-dist**2 / (2 * sig_mag**2)) * dist
            elif method == 'distance':
                mag = np.clip(dist, 0, 1)
            else:
                mag = 0.0  # 默认值
            
            magnitude_values.append(mag)
        
        plt.plot(sig_mag_values, magnitude_values, color=colors[i], label=method, linewidth=2)
    
    plt.xlabel('sig_mag value')
    plt.ylabel('Magnitude at distance 0.75')
    plt.title('Magnitude Response in Real Space')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 从配置文件读取实际使用的sig_mag值
    tanh_sig_mag = CONFIG.method_configs.tanh.sig_mag
    gaussian_sig_mag = CONFIG.method_configs.gaussian_mag.sig_mag
    
    print("配置文件中的sig_mag值:")
    print(f"  tanh方法: {tanh_sig_mag}")
    print(f"  gaussian_mag方法: {gaussian_sig_mag}")
    print("  distance方法: 无需调整（固定为距离值）")

def compare_real_vs_normalized():
    """
    比较真实空间和归一化空间的参数差异
    """
    print("\n=== 真实空间 vs 归一化空间参数比较 ===")
    
    # 归一化空间参数（基于0.25原子间距）
    normalized_atom_distance = 0.25
    normalized_best_sig_sf = 0.01
    
    # 真实空间参数
    scale_factor = 6  # 1/(1/6) = 6
    real_atom_distance = normalized_atom_distance * scale_factor
    real_best_sig_sf, _ = analyze_real_space_sig_sf()
    
    print("归一化空间:")
    print(f"  原子间距: {normalized_atom_distance} 埃")
    print(f"  推荐sig_sf: {normalized_best_sig_sf}")
    
    print("真实空间:")
    print(f"  原子间距: {real_atom_distance:.4f}")
    print(f"  推荐sig_sf: {real_best_sig_sf:.4f}")
    
    print(f"缩放比例: {scale_factor}")
    print(f"sig_sf缩放比例: {real_best_sig_sf / normalized_best_sig_sf:.4f}")

def generate_real_space_recommendations():
    """
    生成真实空间的参数推荐
    """
    print("\n=== 真实空间参数推荐 ===")
    
    print("\n🎯 配置文件中的参数:")
    print(f"1. sig_sf值:")
    print(f"   - tanh方法: {CONFIG.method_configs.tanh.sig_sf}")
    print(f"   - gaussian_mag方法: {CONFIG.method_configs.gaussian_mag.sig_sf}")
    
    print(f"\n2. sig_mag值:")
    print(f"   - tanh方法: {CONFIG.method_configs.tanh.sig_mag}")
    print(f"   - gaussian_mag方法: {CONFIG.method_configs.gaussian_mag.sig_mag}")
    
    print(f"\n3. 其他参数:")
    print(f"   - step_size (tanh): {CONFIG.method_configs.tanh.step_size}")
    print(f"   - step_size (gaussian_mag): {CONFIG.method_configs.gaussian_mag.step_size}")
    print(f"   - n_query_points (tanh): {CONFIG.method_configs.tanh.n_query_points}")
    print(f"   - n_query_points (gaussian_mag): {CONFIG.method_configs.gaussian_mag.n_query_points}")

def create_real_space_usage_example():
    """
    创建真实空间的使用示例
    """
    print("\n=== 真实空间使用示例 ===")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, real_atom_distance = analyze_real_space_sig_sf()
    
    # 创建示例分子（真实坐标）
    coords = torch.tensor([
        [[-real_atom_distance/2, 0.0, 0.0], [real_atom_distance/2, 0.0, 0.0]]  # 两个原子，间距1.5埃
    ], device=device)
    
    atom_types = torch.tensor([[0, 0]], device=device)
    
    # 创建查询点网格（真实空间）
    x = torch.linspace(-3.0, 3.0, 50, device=device)
    y = torch.linspace(-1.5, 1.5, 25, device=device)
    z = torch.linspace(-0.5, 0.5, 10, device=device)
    X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
    query_points = torch.stack([X.flatten(), Y.flatten(), Z.flatten()], dim=1)
    query_points = query_points.unsqueeze(0)
    
    sigma_ratios = {'C': 1.0, 'H': 1.0, 'O': 1.0, 'N': 1.0, 'F': 1.0}
    
    # 使用配置文件中的gaussian_mag参数
    method_config = CONFIG.method_configs.gaussian_mag
    converter = GNFConverter(
        sigma=CONFIG.sigma,
        n_query_points=method_config.n_query_points,
        n_iter=CONFIG.n_iter,
        step_size=method_config.step_size,
        eps=method_config.eps,
        min_samples=method_config.min_samples,
        sigma_ratios=sigma_ratios,
        gradient_field_method='gaussian_mag',
        sig_sf=method_config.sig_sf,
        sig_mag=method_config.sig_mag,
        temperature=CONFIG.temperature,
        logsumexp_eps=CONFIG.logsumexp_eps,
        inverse_square_strength=CONFIG.inverse_square_strength,
        gradient_clip_threshold=CONFIG.gradient_clip_threshold,
        gradient_sampling_candidate_multiplier=CONFIG.gradient_sampling_candidate_multiplier,
        gradient_sampling_temperature=CONFIG.gradient_sampling_temperature,
        n_atom_types=5
    )
    
    vector_field = converter.mol2gnf(coords, atom_types, query_points)
    
    print(f"Field形状: {vector_field.shape}")
    print(f"Field值范围: [{vector_field.min().item():.4f}, {vector_field.max().item():.4f}]")
    print(f"Field均值: {vector_field.mean().item():.4f}")
    print(f"Field标准差: {vector_field.std().item():.4f}")
    
    # 检查在原子位置附近的field值
    atom_positions = coords[0]
    query_positions = query_points[0]
    distances_to_atoms = torch.cdist(query_positions, atom_positions)
    min_distances, _ = torch.min(distances_to_atoms, dim=1)
    
    # 选择距离原子较近的点（距离 < 0.3埃）
    near_atom_mask = min_distances < 0.3
    if near_atom_mask.any():
        near_atom_field = vector_field[0, near_atom_mask, 0, :]
        print(f"近原子field均值: {near_atom_field.mean().item():.4f}")
        print(f"近原子field标准差: {near_atom_field.std().item():.4f}")
    
    # 检查在原子中间位置的field值
    mid_point_mask = (min_distances > 0.6) & (min_distances < 0.9)
    if mid_point_mask.any():
        mid_field = vector_field[0, mid_point_mask, 0, :]
        print(f"中间位置field均值: {mid_field.mean().item():.4f}")
        print(f"中间位置field标准差: {mid_field.std().item():.4f}")

def analyze_tanh_method_specifically():
    """
    专门分析tanh方法的最优参数
    """
    print("\n=== 专门分析tanh方法参数 ===")
    
    # 测试不同的sig_mag值对tanh方法的影响
    distances = np.linspace(0, 2.0, 100)
    
    plt.subplot(2, 3, 4)
    
    # 使用配置文件中的tanh参数
    tanh_config = CONFIG.method_configs.tanh
    config_sig_mag = tanh_config.sig_mag
    
    # 分析tanh方法在不同sig_mag下的行为（包括配置文件中的值）
    test_sig_mags = [0.5, 1.0, 1.5, config_sig_mag]
    # 确保配置文件中的值在列表中且唯一
    if config_sig_mag not in test_sig_mags:
        test_sig_mags.append(config_sig_mag)
    test_sig_mags = sorted(set(test_sig_mags))
    
    for sig_mag in test_sig_mags:
        tanh_values = np.tanh(distances / sig_mag)
        label = f'sig_mag={sig_mag}'
        if sig_mag == config_sig_mag:
            label += ' (config)'
        plt.plot(distances, tanh_values, label=label, linewidth=2 if sig_mag == config_sig_mag else 1.5)
    
    plt.xlabel('Distance (Angstroms)')
    plt.ylabel('Tanh magnitude')
    plt.title('Tanh Method: Distance vs Magnitude')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 显示配置文件中的参数
    target_distance = 0.75  # 原子间距的一半
    config_magnitude = np.tanh(target_distance / config_sig_mag)
    
    print("\n🎯 Tanh方法配置参数（来自配置文件）:")
    print(f"  sig_sf = {tanh_config.sig_sf}")
    print(f"  sig_mag = {config_sig_mag}")
    print(f"  在距离{target_distance}埃处的magnitude: {config_magnitude:.4f}")
    
    return tanh_config.sig_sf, config_sig_mag

def compare_all_three_methods():
    """
    比较三种方法的特性（包括tanh）
    """
    print("\n=== 三种方法特性比较 ===")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, real_atom_distance = analyze_real_space_sig_sf()
    
    # 创建测试场景
    coords = torch.tensor([[[-real_atom_distance/2, 0.0, 0.0], [real_atom_distance/2, 0.0, 0.0]]], device=device)
    atom_types = torch.tensor([[0, 0]], device=device)
    
    # 沿着连接线的查询点
    query_points = torch.linspace(-3.0, 3.0, 200, device=device).reshape(-1, 1)
    query_points = torch.cat([query_points, torch.zeros_like(query_points), torch.zeros_like(query_points)], dim=1)
    query_points = query_points.unsqueeze(0)
    
    sigma_ratios = {'C': 1.0, 'H': 1.0, 'O': 1.0, 'N': 1.0, 'F': 1.0}
    
    # 测试三种方法（包括tanh）
    methods = ['tanh', 'gaussian_mag', 'distance']
    colors = ['blue', 'red', 'green']
    
    plt.subplot(2, 3, 5)
    
    for i, method in enumerate(methods):
        # 从配置文件读取参数
        if hasattr(CONFIG.method_configs, method):
            method_config = getattr(CONFIG.method_configs, method)
            sig_sf = method_config.sig_sf
            sig_mag = method_config.sig_mag
            step_size = method_config.step_size
            n_query_points = method_config.n_query_points
            eps = getattr(method_config, 'eps', CONFIG.default_config.eps)
            min_samples = getattr(method_config, 'min_samples', CONFIG.default_config.min_samples)
        else:
            # distance方法使用默认配置
            method_config = CONFIG.default_config
            sig_sf = method_config.sig_sf
            sig_mag = method_config.sig_mag
            step_size = method_config.step_size
            n_query_points = method_config.n_query_points
            eps = method_config.eps
            min_samples = method_config.min_samples
        
        converter = GNFConverter(
            sigma=CONFIG.sigma,
            n_query_points=n_query_points,
            n_iter=CONFIG.n_iter,
            step_size=step_size,
            eps=eps,
            min_samples=min_samples,
            sigma_ratios=sigma_ratios,
            gradient_field_method=method,
            sig_sf=sig_sf,
            sig_mag=sig_mag,
            temperature=CONFIG.temperature,
            logsumexp_eps=CONFIG.logsumexp_eps,
            inverse_square_strength=CONFIG.inverse_square_strength,
            gradient_clip_threshold=CONFIG.gradient_clip_threshold,
            gradient_sampling_candidate_multiplier=CONFIG.gradient_sampling_candidate_multiplier,
            gradient_sampling_temperature=CONFIG.gradient_sampling_temperature,
            n_atom_types=5
        )
        
        vector_field = converter.mol2gnf(coords, atom_types, query_points)
        field_values = vector_field[0, :, 0, 0].cpu().numpy()
        positions = query_points[0, :, 0].cpu().numpy()
        
        plt.plot(positions, field_values, color=colors[i], label=f'{method}', linewidth=2)
    
    plt.axvline(x=-real_atom_distance/2, color='r', linestyle='--', alpha=0.7, label='Atom 1')
    plt.axvline(x=real_atom_distance/2, color='g', linestyle='--', alpha=0.7, label='Atom 2')
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.xlabel('Position (Angstroms)')
    plt.ylabel('Field value')
    plt.title('Comparison of All Three Methods')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 分析每种方法的特性
    print("\n📊 三种方法特性分析:")
    print("1. Tanh方法（推荐）:")
    print("   - 优点: 平滑连续，有界输出[0,1]，数值稳定，实验表现最优")
    print("   - 缺点: 远距离处饱和为1，可能不如gaussian_mag衰减快")
    print("   - 适用: 需要稳定、有界field的场景（最终选择）")
    
    print("\n2. Gaussian_mag方法:")
    print("   - 优点: 在原子位置处为0，衰减快，峰值在sig_mag处")
    print("   - 缺点: 无上界，远距离处可能数值不稳定")
    print("   - 适用: 需要避免相邻原子干扰的场景")
    
    print("\n3. Distance方法:")
    print("   - 优点: 线性特性，简单直观，有界[0,1]")
    print("   - 缺点: 在距离=1处不连续（虽然实际中距离非负）")
    print("   - 适用: 需要线性field的场景")

if __name__ == "__main__":
    # 设置中文字体
    import matplotlib
    matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
    matplotlib.rcParams['axes.unicode_minus'] = False
    matplotlib.rcParams['font.size'] = 10
    
    # 创建图形
    plt.figure(figsize=(18, 12))
    
    # 运行分析
    analyze_real_space_sig_sf()
    test_real_space_field_behavior()
    analyze_sig_mag_for_real_space()
    compare_real_vs_normalized()
    generate_real_space_recommendations()
    create_real_space_usage_example()
    analyze_tanh_method_specifically()
    compare_all_three_methods()
    
    # 保存结果
    plt.tight_layout()
    plt.savefig('real_space_field_optimization.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n✅ 真实空间分析完成！")
    print("📊 结果已保存为 'real_space_field_optimization.png'") 