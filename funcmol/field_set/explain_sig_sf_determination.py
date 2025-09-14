import numpy as np
import matplotlib.pyplot as plt

def explain_sig_sf_determination():
    """
    详细解释sig_sf参数的确定过程
    """
    print("=== sig_sf参数确定过程详解 ===")
    
    # 1. 基本设定
    real_atom_distance = 1.5  # 真实C-C键长（埃）
    target_decay_distance = 0.75  # 目标衰减距离（原子间距的一半）
    threshold = 0.05  # field衰减阈值
    
    print(f"1. 基本参数设定:")
    print(f"   - 真实原子间距: {real_atom_distance} 埃")
    print(f"   - 目标衰减距离: {target_decay_distance} 埃")
    print(f"   - 衰减阈值: {threshold}")
    
    # 2. 理解softmax权重函数
    print(f"\n2. Softmax权重函数分析:")
    print(f"   w_softmax = exp(-dist / sig_sf) / sum(exp(-dist / sig_sf))")
    print(f"   这个函数控制field的影响范围")
    
    # 3. 测试不同的sig_sf值
    sig_sf_values = np.linspace(0.1, 1.0, 50)
    decay_distances = []
    
    print(f"\n3. 测试不同sig_sf值的衰减距离:")
    print(f"   sig_sf值 | 衰减距离 | 是否接近目标")
    print(f"   --------|----------|------------")
    
    for sig_sf in sig_sf_values:
        # 模拟两个原子的情况
        for dist_a in np.arange(0, real_atom_distance, 0.01):
            dist_b = real_atom_distance - dist_a
            # 计算softmax权重
            val = np.exp(-dist_a / sig_sf) / (np.exp(-dist_a / sig_sf) + np.exp(-dist_b / sig_sf))
            if val < threshold:
                decay_distances.append(dist_a)
                is_close = "✓" if abs(dist_a - target_decay_distance) < 0.1 else "✗"
                print(f"   {sig_sf:.3f}    | {dist_a:.3f}     | {is_close}")
                break
        else:
            decay_distances.append(real_atom_distance)
            print(f"   {sig_sf:.3f}    | {real_atom_distance:.3f}     | ✗")
    
    # 4. 找到最优sig_sf值
    target_idx = np.argmin(np.abs(np.array(decay_distances) - target_decay_distance))
    best_sig_sf = sig_sf_values[target_idx]
    actual_decay = decay_distances[target_idx]
    
    print(f"\n4. 最优sig_sf值确定:")
    print(f"   - 最优sig_sf: {best_sig_sf:.4f}")
    print(f"   - 实际衰减距离: {actual_decay:.4f}")
    print(f"   - 与目标距离的误差: {abs(actual_decay - target_decay_distance):.4f}")
    
    # 5. 可视化分析
    plt.figure(figsize=(15, 5))
    
    # 子图1: sig_sf vs 衰减距离
    plt.subplot(1, 3, 1)
    plt.plot(sig_sf_values, decay_distances, 'b-', linewidth=2, marker='o')
    plt.axhline(y=target_decay_distance, color='r', linestyle='--', alpha=0.7, 
                label=f'Target: {target_decay_distance:.2f}')
    plt.axvline(x=best_sig_sf, color='g', linestyle='--', alpha=0.7, 
                label=f'Best sig_sf: {best_sig_sf:.3f}')
    plt.xlabel('sig_sf value')
    plt.ylabel('Decay distance (Angstroms)')
    plt.title('sig_sf vs Decay Distance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 子图2: 不同sig_sf下的softmax权重
    plt.subplot(1, 3, 2)
    distances = np.linspace(0, real_atom_distance, 100)
    for sig_sf in [0.05, 0.1, 0.2, 0.3]:
        weights = np.exp(-distances / sig_sf) / (np.exp(-distances / sig_sf) + np.exp(-(real_atom_distance - distances) / sig_sf))
        plt.plot(distances, weights, label=f'sig_sf={sig_sf}', linewidth=2)
    
    plt.axhline(y=threshold, color='r', linestyle='--', alpha=0.7, label=f'Threshold: {threshold}')
    plt.axvline(x=target_decay_distance, color='orange', linestyle='--', alpha=0.7, 
                label=f'Target distance: {target_decay_distance}')
    plt.xlabel('Distance from atom 1 (Angstroms)')
    plt.ylabel('Softmax weight')
    plt.title('Softmax Weights vs Distance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 子图3: 误差分析
    plt.subplot(1, 3, 3)
    errors = np.abs(np.array(decay_distances) - target_decay_distance)
    plt.plot(sig_sf_values, errors, 'r-', linewidth=2, marker='o')
    plt.axvline(x=best_sig_sf, color='g', linestyle='--', alpha=0.7, 
                label=f'Best sig_sf: {best_sig_sf:.3f}')
    plt.xlabel('sig_sf value')
    plt.ylabel('Error from target distance')
    plt.title('Error Analysis')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('sig_sf_explanation.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 6. 数学原理解释
    print(f"\n5. 数学原理:")
    print(f"   - sig_sf控制softmax函数的'温度'")
    print(f"   - 较小的sig_sf使field更尖锐，衰减更快")
    print(f"   - 较大的sig_sf使field更平滑，衰减更慢")
    print(f"   - 目标是在原子间距的一半处衰减到阈值以下")
    
    # 7. 实际应用建议
    print(f"\n6. 实际应用建议:")
    print(f"   - 如果field过于尖锐: 增大sig_sf")
    print(f"   - 如果field过于平滑: 减小sig_sf")
    print(f"   - 如果相邻原子干扰: 减小sig_sf")
    print(f"   - 如果神经网络学习困难: 增大sig_sf")
    
    return best_sig_sf

if __name__ == "__main__":
    best_sig_sf = explain_sig_sf_determination()
    print(f"\n✅ sig_sf确定过程完成！")
    print(f"📊 推荐sig_sf值: {best_sig_sf:.4f}")
    print(f"📈 详细分析图已保存为 'sig_sf_explanation.png'") 