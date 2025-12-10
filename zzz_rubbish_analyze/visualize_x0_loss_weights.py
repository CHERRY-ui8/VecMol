"""
可视化x0预测loss的理论权重曲线
根据理论，x0预测loss的权重项 W(t) 应该是：
W(t) ∝ (√(ᾱ_{t-1}β_t) / (1 - ᾱ_t))²
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# 添加路径以便导入ddpm模块
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from funcmol.models.ddpm import get_beta_schedule, prepare_diffusion_constants

def compute_theoretical_weight(diffusion_consts, smooth_method="none", smooth_scale=1.0):
    """
    计算理论权重：W(t) = (√(ᾱ_{t-1}β_t) / (1 - ᾱ_t))²
    
    Args:
        diffusion_consts: 扩散常数字典
        smooth_method: 平滑方法
            - "none": 原始理论权重（不处理）
            - "log": log(1 + W(t)) - 对数变换，适合处理大值
            - "strong_log": log(1 + W(t) / scale) - 可调节的对数变换，scale越大压缩越强
            - "log_sqrt": log(1 + sqrt(W(t))) - 先开方再对数，双重压缩
        smooth_scale: 平滑尺度参数（仅用于strong_log，建议值：100-100000）
    
    Returns:
        weights: 权重张量 [num_timesteps]
    """
    alphas_cumprod_prev = diffusion_consts["alphas_cumprod_prev"]  # ᾱ_{t-1}
    betas = diffusion_consts["betas"]  # β_t
    alphas_cumprod = diffusion_consts["alphas_cumprod"]  # ᾱ_t
    
    # 数值稳定性
    eps = 1e-20
    one_minus_alphas_cumprod = (1.0 - alphas_cumprod).clamp(min=eps)
    
    # 计算权重：W(t) = (√(ᾱ_{t-1}β_t) / (1 - ᾱ_t))²
    sqrt_alpha_bar_prev_beta = torch.sqrt(alphas_cumprod_prev * betas)
    weights = (sqrt_alpha_bar_prev_beta / one_minus_alphas_cumprod) ** 2
    
    # 应用平滑变换
    if smooth_method == "none":
        pass  # 不处理
    elif smooth_method == "log":
        # log(1 + W(t)) - 对数变换，压缩大值
        weights = torch.log(1.0 + weights)
    elif smooth_method == "strong_log":
        # log(1 + W(t) / scale) - 可调节的对数变换，scale越大压缩越强
        # 添加小的epsilon防止权重变成0
        weights = torch.log(1.0 + weights / smooth_scale).clamp(min=1e-8)
    elif smooth_method == "log_sqrt":
        # log(1 + sqrt(W(t))) - 先开方再对数，双重压缩
        weights = torch.log(1.0 + torch.sqrt(weights.clamp(min=1e-20)))
    else:
        raise ValueError(f"Unknown smooth_method: {smooth_method}. Use 'none', 'log', or 'strong_log'.")
    
    return weights

def compute_version3_weight(diffusion_consts):
    """
    计算版本3中使用的权重：(1 - ᾱ_t)
    
    Args:
        diffusion_consts: 扩散常数字典
    
    Returns:
        weights: 权重张量 [num_timesteps]
    """
    alphas_cumprod = diffusion_consts["alphas_cumprod"]
    weights = (1.0 - alphas_cumprod).clamp(min=1e-8)
    return weights

def compute_uniform_weight(num_timesteps):
    """
    计算均匀权重（无加权）
    
    Args:
        num_timesteps: 时间步数
    
    Returns:
        weights: 全1的权重张量
    """
    return torch.ones(num_timesteps)

def normalize_weights(weights):
    """
    归一化权重，使其平均值为1（保持loss scale稳定）
    
    Args:
        weights: 权重张量
    
    Returns:
        normalized_weights: 归一化后的权重
    """
    return weights / weights.mean()

def plot_theoretical_weights_comparison(config, save_path="x0_loss_weights_comparison.png"):
    """
    Plot theoretical weights comparison for linear and cosine schedules
    Shows original weights vs log-smoothed weights
    
    Args:
        config: DDPM配置字典
        save_path: 保存路径
    """
    # 获取DDPM配置
    ddpm_config = config.get("ddpm", {})
    num_timesteps = ddpm_config.get("num_timesteps", 1000)
    beta_start = ddpm_config.get("beta_start", 1e-4)
    beta_end = ddpm_config.get("beta_end", 0.02)
    
    # 计算linear和cosine的理论权重
    timesteps = np.arange(1, num_timesteps + 1)
    
    # Linear schedule - 计算不同平滑方法
    betas_linear = get_beta_schedule(beta_start, beta_end, num_timesteps, schedule="linear")
    diffusion_consts_linear = prepare_diffusion_constants(betas_linear)
    weights_linear_original = compute_theoretical_weight(diffusion_consts_linear, smooth_method="none")
    weights_linear_log = compute_theoretical_weight(diffusion_consts_linear, smooth_method="log")
    # 尝试不同的strong_log scale值（更大的scale值来获得更强的平滑）
    weights_linear_strong_log_100 = compute_theoretical_weight(diffusion_consts_linear, smooth_method="strong_log", smooth_scale=100.0)
    weights_linear_strong_log_1000 = compute_theoretical_weight(diffusion_consts_linear, smooth_method="strong_log", smooth_scale=1000.0)
    weights_linear_strong_log_100000 = compute_theoretical_weight(diffusion_consts_linear, smooth_method="strong_log", smooth_scale=100000.0)
    
    weights_linear_original_norm = normalize_weights(weights_linear_original)
    weights_linear_log_norm = normalize_weights(weights_linear_log)
    weights_linear_strong_log_100_norm = normalize_weights(weights_linear_strong_log_100)
    weights_linear_strong_log_1000_norm = normalize_weights(weights_linear_strong_log_1000)
    weights_linear_strong_log_100000_norm = normalize_weights(weights_linear_strong_log_100000)
    
    # Cosine schedule - 计算不同平滑方法
    betas_cosine = get_beta_schedule(beta_start, beta_end, num_timesteps, schedule="cosine")
    diffusion_consts_cosine = prepare_diffusion_constants(betas_cosine)
    weights_cosine_original = compute_theoretical_weight(diffusion_consts_cosine, smooth_method="none")
    weights_cosine_log = compute_theoretical_weight(diffusion_consts_cosine, smooth_method="log")
    weights_cosine_strong_log_100 = compute_theoretical_weight(diffusion_consts_cosine, smooth_method="strong_log", smooth_scale=100.0)
    weights_cosine_strong_log_1000 = compute_theoretical_weight(diffusion_consts_cosine, smooth_method="strong_log", smooth_scale=1000.0)
    weights_cosine_strong_log_100000 = compute_theoretical_weight(diffusion_consts_cosine, smooth_method="strong_log", smooth_scale=100000.0)
    
    weights_cosine_original_norm = normalize_weights(weights_cosine_original)
    weights_cosine_log_norm = normalize_weights(weights_cosine_log)
    weights_cosine_strong_log_100_norm = normalize_weights(weights_cosine_strong_log_100)
    weights_cosine_strong_log_1000_norm = normalize_weights(weights_cosine_strong_log_1000)
    weights_cosine_strong_log_100000_norm = normalize_weights(weights_cosine_strong_log_100000)
    
    # 转换为numpy
    weights_linear_orig_np = weights_linear_original_norm.numpy()
    weights_linear_log_np = weights_linear_log_norm.numpy()
    weights_linear_strong_log_100_np = weights_linear_strong_log_100_norm.numpy()
    weights_linear_strong_log_1000_np = weights_linear_strong_log_1000_norm.numpy()
    weights_linear_strong_log_100000_np = weights_linear_strong_log_100000_norm.numpy()
    
    weights_cosine_orig_np = weights_cosine_original_norm.numpy()
    weights_cosine_log_np = weights_cosine_log_norm.numpy()
    weights_cosine_strong_log_100_np = weights_cosine_strong_log_100_norm.numpy()
    weights_cosine_strong_log_1000_np = weights_cosine_strong_log_1000_norm.numpy()
    weights_cosine_strong_log_100000_np = weights_cosine_strong_log_100000_norm.numpy()
    
    # 创建图形：1行2列
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. Original weights (normalized)
    ax1 = axes[0]
    ax1.plot(timesteps, weights_linear_orig_np, label='Linear Schedule', linewidth=2, color='#1f77b4')
    ax1.plot(timesteps, weights_cosine_orig_np, label='Cosine Schedule', linewidth=2, color='#ff7f0e')
    ax1.set_xlabel('Timestep t', fontsize=12)
    ax1.set_ylabel('Normalized Weight W(t)', fontsize=12)
    ax1.set_title('Original Theoretical Weights\n(Normalized, Mean = 1)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11)
    ax1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    
    # 2. Strong log-smoothed weights (normalized) - 展示不同scale的效果
    ax2 = axes[1]
    ax2.plot(timesteps, weights_linear_log_np, label='Linear: log(1+W)', linewidth=2, color='#1f77b4', linestyle='-')
    ax2.plot(timesteps, weights_linear_strong_log_100_np, label='Linear: log(1+W/100)', linewidth=2, color='#1f77b4', linestyle='--', alpha=0.8)
    ax2.plot(timesteps, weights_linear_strong_log_1000_np, label='Linear: log(1+W/1000)', linewidth=2, color='#1f77b4', linestyle=':', alpha=0.7)
    ax2.plot(timesteps, weights_linear_strong_log_100000_np, label='Linear: log(1+W/100000)', linewidth=2, color='#1f77b4', linestyle='-.', alpha=0.6)
    
    ax2.plot(timesteps, weights_cosine_log_np, label='Cosine: log(1+W)', linewidth=2, color='#ff7f0e', linestyle='-')
    ax2.plot(timesteps, weights_cosine_strong_log_100_np, label='Cosine: log(1+W/100)', linewidth=2, color='#ff7f0e', linestyle='--', alpha=0.8)
    ax2.plot(timesteps, weights_cosine_strong_log_1000_np, label='Cosine: log(1+W/1000)', linewidth=2, color='#ff7f0e', linestyle=':', alpha=0.7)
    ax2.plot(timesteps, weights_cosine_strong_log_100000_np, label='Cosine: log(1+W/100000)', linewidth=2, color='#ff7f0e', linestyle='-.', alpha=0.6)
    
    ax2.set_xlabel('Timestep t', fontsize=12)
    ax2.set_ylabel('Normalized Weight W(t)', fontsize=12)
    ax2.set_title('Strong Log-Smoothed Weights\n(log(1+W/scale), Normalized, Mean = 1)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9, ncol=2)
    ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Theoretical weights comparison plot saved to: {save_path}")
    
    # Print key information
    print("\n" + "="*60)
    print("Theoretical Weight Analysis Summary")
    print("="*60)
    
    # Original weights
    print(f"\n--- Original Weights (Normalized) ---")
    print(f"Linear Schedule:")
    print(f"  Weight range: [{weights_linear_orig_np.min():.6f}, {weights_linear_orig_np.max():.6f}]")
    max_ratio_linear_orig = weights_linear_orig_np.max() / weights_linear_orig_np.min()
    print(f"  Weight ratio (max/min): {max_ratio_linear_orig:.2f}x")
    
    print(f"\nCosine Schedule:")
    print(f"  Weight range: [{weights_cosine_orig_np.min():.6f}, {weights_cosine_orig_np.max():.6f}]")
    max_ratio_cosine_orig = weights_cosine_orig_np.max() / weights_cosine_orig_np.min()
    print(f"  Weight ratio (max/min): {max_ratio_cosine_orig:.2f}x")
    
    # Log-smoothed weights
    print(f"\n--- Log-Smoothed Weights (Normalized) ---")
    print(f"Linear Schedule:")
    eps_ratio = 1e-10
    ratio_linear_log = weights_linear_log_np.max() / max(weights_linear_log_np.min(), eps_ratio)
    ratio_linear_100 = weights_linear_strong_log_100_np.max() / max(weights_linear_strong_log_100_np.min(), eps_ratio)
    ratio_linear_1000 = weights_linear_strong_log_1000_np.max() / max(weights_linear_strong_log_1000_np.min(), eps_ratio)
    ratio_linear_100000 = weights_linear_strong_log_100000_np.max() / max(weights_linear_strong_log_100000_np.min(), eps_ratio)
    
    print(f"  log(1+W): range=[{weights_linear_log_np.min():.6f}, {weights_linear_log_np.max():.6f}], ratio={ratio_linear_log:.2f}x")
    print(f"  log(1+W/100): range=[{weights_linear_strong_log_100_np.min():.6f}, {weights_linear_strong_log_100_np.max():.6f}], ratio={ratio_linear_100:.2f}x")
    print(f"  log(1+W/1000): range=[{weights_linear_strong_log_1000_np.min():.6f}, {weights_linear_strong_log_1000_np.max():.6f}], ratio={ratio_linear_1000:.2f}x")
    print(f"  log(1+W/100000): range=[{weights_linear_strong_log_100000_np.min():.6f}, {weights_linear_strong_log_100000_np.max():.6f}], ratio={ratio_linear_100000:.2f}x")
    
    print(f"\nCosine Schedule:")
    ratio_cosine_log = weights_cosine_log_np.max() / max(weights_cosine_log_np.min(), eps_ratio)
    ratio_cosine_100 = weights_cosine_strong_log_100_np.max() / max(weights_cosine_strong_log_100_np.min(), eps_ratio)
    ratio_cosine_1000 = weights_cosine_strong_log_1000_np.max() / max(weights_cosine_strong_log_1000_np.min(), eps_ratio)
    ratio_cosine_100000 = weights_cosine_strong_log_100000_np.max() / max(weights_cosine_strong_log_100000_np.min(), eps_ratio)
    
    print(f"  log(1+W): range=[{weights_cosine_log_np.min():.6f}, {weights_cosine_log_np.max():.6f}], ratio={ratio_cosine_log:.2f}x")
    print(f"  log(1+W/100): range=[{weights_cosine_strong_log_100_np.min():.6f}, {weights_cosine_strong_log_100_np.max():.6f}], ratio={ratio_cosine_100:.2f}x")
    print(f"  log(1+W/1000): range=[{weights_cosine_strong_log_1000_np.min():.6f}, {weights_cosine_strong_log_1000_np.max():.6f}], ratio={ratio_cosine_1000:.2f}x")
    ratio_cosine_100000 = weights_cosine_strong_log_100000_np.max() / max(weights_cosine_strong_log_100000_np.min(), eps_ratio)
    print(f"  log(1+W/100000): range=[{weights_cosine_strong_log_100000_np.min():.6f}, {weights_cosine_strong_log_100000_np.max():.6f}], ratio={ratio_cosine_100000:.2f}x")
    
    # Check which smoothing works best
    max_ratio_linear_log = ratio_linear_log
    max_ratio_linear_100 = ratio_linear_100
    max_ratio_linear_1000 = ratio_linear_1000
    max_ratio_linear_100000 = ratio_linear_100000
    
    max_ratio_cosine_log = ratio_cosine_log
    max_ratio_cosine_100 = ratio_cosine_100
    max_ratio_cosine_1000 = ratio_cosine_1000
    max_ratio_cosine_100000 = ratio_cosine_100000
    
    print(f"\n--- Best Smoothing Results ---")
    best_linear_ratio = min(max_ratio_linear_log, max_ratio_linear_100, max_ratio_linear_1000, max_ratio_linear_100000)
    best_cosine_ratio = min(max_ratio_cosine_log, max_ratio_cosine_100, max_ratio_cosine_1000, max_ratio_cosine_100000)
    
    if best_linear_ratio < 100 and best_cosine_ratio < 100:
        print(f"✅ Strong log smoothing successfully reduces weight ratios to reasonable range!")
        if best_linear_ratio == max_ratio_linear_100000:
            print(f"   Linear: {best_linear_ratio:.2f}x (using log(1+W/100000))")
        elif best_linear_ratio == max_ratio_linear_1000:
            print(f"   Linear: {best_linear_ratio:.2f}x (using log(1+W/1000))")
        elif best_linear_ratio == max_ratio_linear_100:
            print(f"   Linear: {best_linear_ratio:.2f}x (using log(1+W/100))")
        else:
            print(f"   Linear: {best_linear_ratio:.2f}x (using log(1+W))")
            
        if best_cosine_ratio == max_ratio_cosine_100000:
            print(f"   Cosine: {best_cosine_ratio:.2f}x (using log(1+W/100000))")
        elif best_cosine_ratio == max_ratio_cosine_1000:
            print(f"   Cosine: {best_cosine_ratio:.2f}x (using log(1+W/1000))")
        elif best_cosine_ratio == max_ratio_cosine_100:
            print(f"   Cosine: {best_cosine_ratio:.2f}x (using log(1+W/100))")
        else:
            print(f"   Cosine: {best_cosine_ratio:.2f}x (using log(1+W))")
    else:
        print(f"⚠️  Strong log smoothing helps but may need even stronger smoothing or clipping.")
        print(f"   Best Linear ratio: {best_linear_ratio:.2f}x, Best Cosine ratio: {best_cosine_ratio:.2f}x")
    
    return weights_linear_strong_log_100000_norm, weights_cosine_strong_log_100000_norm

if __name__ == "__main__":
    # 使用默认配置（与train_fm_qm9.yaml一致）
    config = {
        "ddpm": {
            "num_timesteps": 1000,
            "beta_start": 0.0001,
            "beta_end": 0.02,
            "schedule": "linear"
        }
    }
    
    # 绘制理论权重对比图（linear vs cosine）
    plot_theoretical_weights_comparison(
        config, 
        save_path="x0_loss_weights_comparison.png"
    )

