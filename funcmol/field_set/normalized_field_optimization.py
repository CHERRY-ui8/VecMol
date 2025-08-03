import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from funcmol.utils.gnf_converter import GNFConverter

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
    best_sig_sf, real_atom_distance = analyze_real_space_sig_sf()
    
    # 创建测试场景：两个相邻原子（真实坐标）
    coords = torch.tensor([[[-real_atom_distance/2, 0.0, 0.0], [real_atom_distance/2, 0.0, 0.0]]], device=device)
    atom_types = torch.tensor([[0, 0]], device=device)
    
    # 沿着连接线的查询点（真实空间）
    query_points = torch.linspace(-3.0, 3.0, 200, device=device).reshape(-1, 1)
    query_points = torch.cat([query_points, torch.zeros_like(query_points), torch.zeros_like(query_points)], dim=1)
    query_points = query_points.unsqueeze(0)
    
    sigma_ratios = {'C': 1.0, 'H': 1.0, 'O': 1.0, 'N': 1.0, 'F': 1.0}
    
    # 测试三种方法
    methods = ['sigmoid', 'gaussian_mag', 'distance']
    sig_mag_values = [0.5, 0.8, 1.0]  # 针对真实空间调整
    
    plt.subplot(2, 3, 2)
    
    for i, method in enumerate(methods):
        for j, sig_mag in enumerate(sig_mag_values):
            converter = GNFConverter(
                sigma=0.5,
                n_query_points=100,
                n_iter=10,
                step_size=0.01,
                eps=0.1,
                min_samples=2,
                sigma_ratios=sigma_ratios,
                gradient_field_method=method,
                sig_sf=best_sig_sf,
                sig_mag=sig_mag,
                device=device
            )
            
            vector_field = converter.mol2gnf(coords, atom_types, query_points)
            field_values = vector_field[0, :, 0, 0].cpu().numpy()
            positions = query_points[0, :, 0].cpu().numpy()
            
            # 只绘制第一个sig_mag值的结果
            if j == 0:
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
    distances = np.linspace(0, 2.0, 100)  # 真实距离范围
    
    plt.subplot(2, 3, 3)
    
    # 测试三种magnitude方法
    methods = ['sigmoid', 'gaussian_mag', 'distance']
    colors = ['b', 'g', 'r']
    
    for i, method in enumerate(methods):
        magnitude_values = []
        for sig_mag in sig_mag_values:
            # 计算在距离0.75处的magnitude值（真实空间中的典型距离）
            dist = 0.75
            if method == 'sigmoid':
                mag = np.tanh(dist / sig_mag)
            elif method == 'gaussian_mag':
                mag = np.exp(-dist**2 / (2 * sig_mag**2)) * dist
            elif method == 'distance':
                mag = np.clip(dist, 0, 1)
            
            magnitude_values.append(mag)
        
        plt.plot(sig_mag_values, magnitude_values, color=colors[i], label=method, linewidth=2)
    
    plt.xlabel('sig_mag value')
    plt.ylabel('Magnitude at distance 0.75')
    plt.title('Magnitude Response in Real Space')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 推荐sig_mag值（使得在距离0.75处magnitude约为0.5）
    target_magnitude = 0.5
    best_sig_mag_sigmoid = 0.75 / np.arctanh(target_magnitude)  # 对于sigmoid
    best_sig_mag_gaussian = np.sqrt(-0.5 * np.log(target_magnitude / 0.75))  # 对于gaussian
    
    print(f"推荐的sig_mag值（真实空间）:")
    print(f"  sigmoid方法: {best_sig_mag_sigmoid:.3f}")
    print(f"  gaussian_mag方法: {best_sig_mag_gaussian:.3f}")
    print(f"  distance方法: 无需调整（固定为距离值）")

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
    
    print(f"归一化空间:")
    print(f"  原子间距: {normalized_atom_distance} 埃")
    print(f"  推荐sig_sf: {normalized_best_sig_sf}")
    
    print(f"真实空间:")
    print(f"  原子间距: {real_atom_distance:.4f}")
    print(f"  推荐sig_sf: {real_best_sig_sf:.4f}")
    
    print(f"缩放比例: {scale_factor}")
    print(f"sig_sf缩放比例: {real_best_sig_sf / normalized_best_sig_sf:.4f}")

def generate_real_space_recommendations():
    """
    生成真实空间的参数推荐
    """
    print("\n=== 真实空间参数推荐 ===")
    
    best_sig_sf, real_atom_distance = analyze_real_space_sig_sf()
    
    print(f"\n🎯 真实空间推荐配置:")
    print(f"1. sig_sf = {best_sig_sf:.4f}")
    print(f"   - 针对真实坐标优化")
    print(f"   - 确保field在相邻原子之间衰减到接近0")
    
    print(f"\n2. sig_mag 推荐值（真实空间）:")
    print(f"   - sigmoid方法: 0.5-0.8")
    print(f"   - gaussian_mag方法: 0.3-0.6")
    print(f"   - distance方法: 无需调整")
    
    print(f"\n3. 完整配置示例:")
    print(f"   # 真实空间推荐配置")
    print(f"   converter = GNFConverter(")
    print(f"       gradient_field_method='gaussian_mag',")
    print(f"       sig_sf={best_sig_sf:.4f},")
    print(f"       sig_mag=0.5,")
    print(f"       temperature=1.0,")
    print(f"       device='cuda'")
    print(f"   )")
    
    print(f"\n4. 使用建议:")
    print(f"   - 如果神经网络学习困难，尝试增大sig_sf到{best_sig_sf*1.5:.4f}")
    print(f"   - 如果field过于平滑，尝试减小sig_sf到{best_sig_sf*0.7:.4f}")
    print(f"   - 如果相邻原子干扰严重，使用gaussian_mag方法")
    print(f"   - 如果需要线性特性，使用distance方法")

def create_real_space_usage_example():
    """
    创建真实空间的使用示例
    """
    print("\n=== 真实空间使用示例 ===")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    best_sig_sf, real_atom_distance = analyze_real_space_sig_sf()
    
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
    
    # 测试推荐配置
    converter = GNFConverter(
        sigma=0.5,
        n_query_points=100,
        n_iter=10,
        step_size=0.01,
        eps=0.1,
        min_samples=2,
        sigma_ratios=sigma_ratios,
        gradient_field_method='gaussian_mag',
        sig_sf=best_sig_sf,
        sig_mag=0.5,
        device=device
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

def analyze_sigmoid_method_specifically():
    """
    专门分析sigmoid方法的最优参数
    """
    print("\n=== 专门分析sigmoid方法参数 ===")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    best_sig_sf, real_atom_distance = analyze_real_space_sig_sf()
    
    # 测试不同的sig_mag值对sigmoid方法的影响
    sig_mag_values = np.linspace(0.3, 2.0, 50)
    distances = np.linspace(0, 2.0, 100)
    
    plt.subplot(2, 3, 4)
    
    # 分析sigmoid方法在不同sig_mag下的行为
    for sig_mag in [0.5, 0.8, 1.0, 1.5]:
        sigmoid_values = np.tanh(distances / sig_mag)
        plt.plot(distances, sigmoid_values, label=f'sig_mag={sig_mag}', linewidth=2)
    
    plt.xlabel('Distance (Angstroms)')
    plt.ylabel('Sigmoid magnitude')
    plt.title('Sigmoid Method: Distance vs Magnitude')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 分析在典型原子间距处的magnitude值
    target_distance = 0.75  # 原子间距的一半
    magnitude_at_target = []
    
    for sig_mag in sig_mag_values:
        mag = np.tanh(target_distance / sig_mag)
        magnitude_at_target.append(mag)
    
    # 找到使得在目标距离处magnitude约为0.5的sig_mag值
    target_magnitude = 0.5
    best_sig_mag_sigmoid = target_distance / np.arctanh(target_magnitude)
    
    print(f"\n🎯 Sigmoid方法推荐参数:")
    print(f"  sig_sf = {best_sig_sf:.4f}")
    print(f"  sig_mag = {best_sig_mag_sigmoid:.4f}")
    print(f"  在距离{target_distance}埃处的magnitude: {np.tanh(target_distance/best_sig_mag_sigmoid):.4f}")
    
    return best_sig_sf, best_sig_mag_sigmoid

def compare_all_three_methods():
    """
    比较三种方法的特性
    """
    print("\n=== 三种方法特性比较 ===")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    best_sig_sf, real_atom_distance = analyze_real_space_sig_sf()
    
    # 创建测试场景
    coords = torch.tensor([[[-real_atom_distance/2, 0.0, 0.0], [real_atom_distance/2, 0.0, 0.0]]], device=device)
    atom_types = torch.tensor([[0, 0]], device=device)
    
    # 沿着连接线的查询点
    query_points = torch.linspace(-3.0, 3.0, 200, device=device).reshape(-1, 1)
    query_points = torch.cat([query_points, torch.zeros_like(query_points), torch.zeros_like(query_points)], dim=1)
    query_points = query_points.unsqueeze(0)
    
    sigma_ratios = {'C': 1.0, 'H': 1.0, 'O': 1.0, 'N': 1.0, 'F': 1.0}
    
    # 测试三种方法
    methods = ['sigmoid', 'gaussian_mag', 'distance']
    colors = ['blue', 'red', 'green']
    
    plt.subplot(2, 3, 5)
    
    for i, method in enumerate(methods):
        # 为每种方法选择最优的sig_mag
        if method == 'sigmoid':
            sig_mag = 0.75  # 基于sigmoid分析的结果
        elif method == 'gaussian_mag':
            sig_mag = 0.45  # 基于gaussian分析的结果
        else:  # distance
            sig_mag = 0.5   # 对distance方法不重要
        
        converter = GNFConverter(
            sigma=0.5,
            n_query_points=100,
            n_iter=10,
            step_size=0.01,
            eps=0.1,
            min_samples=2,
            sigma_ratios=sigma_ratios,
            gradient_field_method=method,
            sig_sf=best_sig_sf,
            sig_mag=sig_mag,
            device=device
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
    print(f"\n📊 三种方法特性分析:")
    print(f"1. Sigmoid方法:")
    print(f"   - 优点: 平滑连续，在远距离处有渐近线")
    print(f"   - 缺点: 在近距离处可能不够尖锐")
    print(f"   - 适用: 需要平滑field的场景")
    
    print(f"\n2. Gaussian_mag方法:")
    print(f"   - 优点: 在原子位置处为0，衰减快")
    print(f"   - 缺点: 在远距离处衰减可能过快")
    print(f"   - 适用: 需要避免相邻原子干扰的场景")
    
    print(f"\n3. Distance方法:")
    print(f"   - 优点: 线性特性，简单直观")
    print(f"   - 缺点: 在原子位置处不连续")
    print(f"   - 适用: 需要线性field的场景")

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
    analyze_sigmoid_method_specifically()
    compare_all_three_methods()
    
    # 保存结果
    plt.tight_layout()
    plt.savefig('real_space_field_optimization.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n✅ 真实空间分析完成！")
    print("📊 结果已保存为 'real_space_field_optimization.png'") 