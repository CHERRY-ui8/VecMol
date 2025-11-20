"""
分析diffusion超参数设置，并给出建议
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from funcmol.models.ddpm import get_beta_schedule, prepare_diffusion_constants

def analyze_beta_schedule(beta_start, beta_end, num_timesteps, schedule="linear"):
    """分析beta调度"""
    betas = get_beta_schedule(beta_start, beta_end, num_timesteps, schedule=schedule)
    diffusion_consts = prepare_diffusion_constants(betas)
    
    alphas_cumprod = diffusion_consts["alphas_cumprod"]
    alphas = diffusion_consts["alphas"]
    
    # 计算关键指标
    alpha_bar_final = alphas_cumprod[-1].item()  # 最终alpha_bar，应该接近0
    alpha_bar_initial = alphas_cumprod[0].item()  # 初始alpha_bar，应该接近1
    
    # 计算SNR (Signal-to-Noise Ratio)
    snr = alphas_cumprod / (1 - alphas_cumprod + 1e-8)
    
    print(f"\n{'='*60}")
    print(f"Beta Schedule Analysis: {schedule}")
    print(f"{'='*60}")
    print(f"Beta range: [{beta_start:.6f}, {beta_end:.6f}]")
    print(f"Alpha range: [{alphas.min().item():.6f}, {alphas.max().item():.6f}]")
    print(f"Alpha_bar (cumulative): [{alpha_bar_initial:.6f}, {alpha_bar_final:.6f}]")
    print(f"Final SNR: {snr[-1].item():.6f}")
    print(f"Initial SNR: {snr[0].item():.6f}")
    print(f"SNR ratio (initial/final): {snr[0].item() / (snr[-1].item() + 1e-8):.2f}")
    
    return betas, diffusion_consts, snr

def plot_beta_analysis(betas, diffusion_consts, snr, schedule_name, save_path=None):
    """绘制beta分析图"""
    timesteps = torch.arange(1, len(betas) + 1)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Beta值
    axes[0, 0].plot(timesteps, betas.numpy(), 'b-', linewidth=2)
    axes[0, 0].set_xlabel('Timestep t', fontsize=12)
    axes[0, 0].set_ylabel(r'$\beta_t$', fontsize=12)
    axes[0, 0].set_title(f'Beta Schedule: {schedule_name}', fontsize=14)
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Alpha_bar (cumulative)
    axes[0, 1].plot(timesteps, diffusion_consts["alphas_cumprod"].numpy(), 'g-', linewidth=2)
    axes[0, 1].set_xlabel('Timestep t', fontsize=12)
    axes[0, 1].set_ylabel(r'$\bar{\alpha}_t$', fontsize=12)
    axes[0, 1].set_title('Cumulative Alpha', fontsize=14)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_yscale('log')
    
    # 3. SNR
    axes[1, 0].plot(timesteps, snr.numpy(), 'r-', linewidth=2)
    axes[1, 0].set_xlabel('Timestep t', fontsize=12)
    axes[1, 0].set_ylabel('SNR', fontsize=12)
    axes[1, 0].set_title('Signal-to-Noise Ratio', fontsize=14)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_yscale('log')
    
    # 4. 噪声水平 (1 - alpha_bar)
    noise_level = 1 - diffusion_consts["alphas_cumprod"]
    axes[1, 1].plot(timesteps, noise_level.numpy(), 'm-', linewidth=2)
    axes[1, 1].set_xlabel('Timestep t', fontsize=12)
    axes[1, 1].set_ylabel('Noise Level', fontsize=12)
    axes[1, 1].set_title('Noise Level (1 - ᾱ_t)', fontsize=14)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✅ 图像已保存到: {save_path}")
    else:
        plt.show()

if __name__ == "__main__":
    # 当前配置
    print("="*60)
    print("当前配置分析")
    print("="*60)
    current_betas, current_consts, current_snr = analyze_beta_schedule(
        beta_start=0.0005,
        beta_end=0.05,
        num_timesteps=100,
        schedule="linear"
    )
    plot_beta_analysis(current_betas, current_consts, current_snr, "Current (Linear)", 
                      "current_beta_schedule.png")
    
    # 推荐配置1: 降低beta_end
    print("\n" + "="*60)
    print("推荐配置1: 降低beta_end (更保守的噪声添加)")
    print("="*60)
    rec1_betas, rec1_consts, rec1_snr = analyze_beta_schedule(
        beta_start=0.0001,
        beta_end=0.02,
        num_timesteps=100,
        schedule="linear"
    )
    plot_beta_analysis(rec1_betas, rec1_consts, rec1_snr, "Recommended 1 (Linear, β_end=0.02)", 
                      "recommended1_beta_schedule.png")
    
    # 推荐配置2: 使用cosine schedule
    print("\n" + "="*60)
    print("推荐配置2: 使用Cosine Schedule (更平滑的噪声添加)")
    print("="*60)
    rec2_betas, rec2_consts, rec2_snr = analyze_beta_schedule(
        beta_start=0.0001,
        beta_end=0.02,
        num_timesteps=100,
        schedule="cosine"
    )
    plot_beta_analysis(rec2_betas, rec2_consts, rec2_snr, "Recommended 2 (Cosine)", 
                      "recommended2_beta_schedule.png")
    
    # 推荐配置3: 增加时间步数
    print("\n" + "="*60)
    print("推荐配置3: 增加时间步数到1000 (更细粒度的扩散)")
    print("="*60)
    rec3_betas, rec3_consts, rec3_snr = analyze_beta_schedule(
        beta_start=0.0001,
        beta_end=0.02,
        num_timesteps=1000,
        schedule="linear"
    )
    plot_beta_analysis(rec3_betas, rec3_consts, rec3_snr, "Recommended 3 (Linear, T=1000)", 
                      "recommended3_beta_schedule.png")
    
    # 总结和建议
    print("\n" + "="*60)
    print("总结和建议")
    print("="*60)
    print("""
1. 损失差异分析：
   - new版本（预测噪声）的loss ~0.04 是正常的
   - new_x0版本（预测x0）的loss ~0.006 也是正常的
   - 两者不能直接比较，因为：
     * 预测噪声：目标是标准正态分布N(0,1)，MSE loss通常在0.01-0.1范围
     * 预测x0：目标是归一化后的数据，数值范围更小，loss可以更低
   - 关键要看生成质量，而不是loss的绝对值

2. 当前超参数问题：
   - beta_end=0.05 可能偏高，导致每一步添加的噪声过大
   - 对于100个时间步，建议beta_end=0.02左右
   - 当前配置可能导致训练不稳定或生成质量下降

3. 推荐配置：
   - 保守方案：beta_start=0.0001, beta_end=0.02, num_timesteps=100, schedule="linear"
   - 平滑方案：beta_start=0.0001, beta_end=0.02, num_timesteps=100, schedule="cosine"
   - 精细方案：beta_start=0.0001, beta_end=0.02, num_timesteps=1000, schedule="linear"
   
4. 选择建议：
   - 如果训练时间有限：使用保守方案（100步，linear）
   - 如果追求更好的生成质量：使用平滑方案（100步，cosine）
   - 如果计算资源充足：使用精细方案（1000步，linear）
    """)

