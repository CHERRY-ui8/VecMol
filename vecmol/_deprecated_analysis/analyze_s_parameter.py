"""
分析cosine调度中s参数对噪声比例的影响
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
from vecmol.models.ddpm import get_beta_schedule, prepare_diffusion_constants

def analyze_s_parameter_effect():
    """分析不同s值对噪声比例的影响"""
    num_timesteps = 1000
    beta_start = 0.0001  # 这些参数在cosine调度中实际上不会被使用
    beta_end = 0.02
    
    # 测试不同的s值
    s_values = [0.004, 0.008, 0.012, 0.016, 0.020]
    timesteps = np.arange(1, num_timesteps + 1)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Alpha_cumprod对比
    ax1 = axes[0, 0]
    for s in s_values:
        betas = get_beta_schedule(beta_start, beta_end, num_timesteps, schedule="cosine", s=s)
        diffusion_consts = prepare_diffusion_constants(betas)
        alphas_cumprod = diffusion_consts["alphas_cumprod"]
        ax1.plot(timesteps, alphas_cumprod.numpy(), label=f's={s}', linewidth=2)
    ax1.set_xlabel('Timestep t', fontsize=12)
    ax1.set_ylabel(r'$\bar{\alpha}_t$ (Cumulative Alpha)', fontsize=12)
    ax1.set_title('Effect of s on Cumulative Alpha', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 噪声比例对比 (1 - alpha_cumprod)
    ax2 = axes[0, 1]
    for s in s_values:
        betas = get_beta_schedule(beta_start, beta_end, num_timesteps, schedule="cosine", s=s)
        diffusion_consts = prepare_diffusion_constants(betas)
        alphas_cumprod = diffusion_consts["alphas_cumprod"]
        noise_level = 1 - alphas_cumprod
        ax2.plot(timesteps, noise_level.numpy(), label=f's={s}', linewidth=2)
    ax2.set_xlabel('Timestep t', fontsize=12)
    ax2.set_ylabel(r'Noise Level (1 - $\bar{\alpha}_t$)', fontsize=12)
    ax2.set_title('Effect of s on Noise Level', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    # 3. 关键时间步的噪声比例对比
    ax3 = axes[1, 0]
    key_timesteps = [100, 250, 500, 750, 1000]
    for s in s_values:
        betas = get_beta_schedule(beta_start, beta_end, num_timesteps, schedule="cosine", s=s)
        diffusion_consts = prepare_diffusion_constants(betas)
        alphas_cumprod = diffusion_consts["alphas_cumprod"]
        noise_levels = []
        for t in key_timesteps:
            noise_level = (1 - alphas_cumprod[t-1]).item()
            noise_levels.append(noise_level)
        ax3.plot(key_timesteps, noise_levels, marker='o', label=f's={s}', linewidth=2, markersize=8)
    ax3.set_xlabel('Timestep t', fontsize=12)
    ax3.set_ylabel(r'Noise Level (1 - $\bar{\alpha}_t$)', fontsize=12)
    ax3.set_title('Noise Level at Key Timesteps', fontsize=14)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')
    
    # 4. Beta序列对比
    ax4 = axes[1, 1]
    for s in s_values:
        betas = get_beta_schedule(beta_start, beta_end, num_timesteps, schedule="cosine", s=s)
        ax4.plot(timesteps, betas.numpy(), label=f's={s}', linewidth=2, alpha=0.7)
    ax4.set_xlabel('Timestep t', fontsize=12)
    ax4.set_ylabel(r'$\beta_t$', fontsize=12)
    ax4.set_title('Effect of s on Beta Schedule', fontsize=14)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = "s_parameter_analysis.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 分析图已保存到: {save_path}")
    
    # 打印数值对比
    print("\n" + "="*60)
    print("不同s值在关键时间步的噪声比例对比")
    print("="*60)
    print(f"{'Timestep':<12}", end="")
    for s in s_values:
        print(f"s={s:<8}", end="")
    print()
    print("-"*60)
    
    for t in key_timesteps:
        print(f"{t:<12}", end="")
        for s in s_values:
            betas = get_beta_schedule(beta_start, beta_end, num_timesteps, schedule="cosine", s=s)
            diffusion_consts = prepare_diffusion_constants(betas)
            alphas_cumprod = diffusion_consts["alphas_cumprod"]
            noise_level = (1 - alphas_cumprod[t-1]).item()
            print(f"{noise_level:<12.6f}", end="")
        print()
    
    # 总结
    print("\n" + "="*60)
    print("总结：如何减小噪声比例")
    print("="*60)
    print("""
    从分析结果可以看出：
    1. s参数对噪声比例的影响：
       - s越大，在相同时间步的噪声比例越大（alpha_cumprod越小）
       - s越小，在相同时间步的噪声比例越小（alpha_cumprod越大）
    
    2. 要减小噪声比例，应该：
       - 减小s参数（例如：从0.008减小到0.004或0.006）
       - 这会让alpha_cumprod在整个扩散过程中保持更大的值
       - 从而减小噪声比例 sqrt(1 - alpha_cumprod)
    
    3. 建议调整范围：
       - 当前默认值：s = 0.008
       - 减小噪声：s = 0.004 ~ 0.006
       - 注意：s过小可能导致扩散过程过快，影响训练稳定性
       - 建议逐步减小，观察训练效果
    """)

if __name__ == "__main__":
    analyze_s_parameter_effect()
