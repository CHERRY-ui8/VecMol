#!/usr/bin/env python3
"""
分析真实的neural field encoder输出的codes数值范围
"""

import torch
import matplotlib.pyplot as plt
import sys
import os

# 添加项目路径
sys.path.append('/datapool/data3/storage/pengxingang/pxg/hyc/funcmol-main-neuralfield')

import hydra
from funcmol.utils.utils_fm import add_noise_to_code
from funcmol.utils.utils_nf import load_neural_field, normalize_code

def analyze_real_codes():
    """
    分析真实的neural field encoder输出的codes
    """
    
    print("=== 分析真实Neural Field Encoder输出的Codes ===")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 1. 加载配置
    with hydra.initialize(config_path="funcmol/configs", version_base=None):
        cfg = hydra.compose(config_name="train_fm_qm9")
    
    print(f"\n=== 配置信息 ===")
    print(f"smooth_sigma: {cfg.smooth_sigma}")
    print(f"normalize_codes: {cfg.normalize_codes}")
    print(f"code_dim: {cfg.decoder.code_dim}")
    print(f"grid_size: {cfg.dset.grid_size}")
    
    # 2. 加载neural field模型
    print(f"\n=== 加载Neural Field模型 ===")
    # TODO：可修改
    nf_checkpoint_path = "/datapool/data3/storage/pengxingang/pxg/hyc/funcmol-main-neuralfield/exps/neural_field/nf_qm9/20250911/lightning_logs/version_1/checkpoints/model-epoch=39.ckpt"
    
    try:
        # 加载checkpoint (设置weights_only=False)
        checkpoint = torch.load(nf_checkpoint_path, map_location=device, weights_only=False)
        print(f"成功加载checkpoint: {nf_checkpoint_path}")
        
        enc, dec = load_neural_field(checkpoint, None, None)
        enc.eval()
        dec.eval()
        print("成功加载encoder和decoder")
        
    except Exception as e:
        print(f"加载模型失败: {e}")
        return None, None, None
    
    # 3. 直接加载预计算的codes
    print(f"\n=== 加载预计算的Codes ===")
    try:
        codes_path = "/datapool/data3/storage/pengxingang/pxg/hyc/funcmol-main-neuralfield/exps/neural_field/nf_qm9/20250911/lightning_logs/version_1/checkpoints/codes/train/codes.pt"
        
        if os.path.exists(codes_path):
            # 加载预计算的codes
            raw_codes = torch.load(codes_path, map_location=device)
            print(f"成功加载预计算的codes: {raw_codes.shape}")
            
            # 只取前1000个样本进行分析
            if raw_codes.shape[0] > 10000:
                raw_codes = raw_codes[:10000]
                print(f"使用前1000个样本进行分析")
        else:
            print(f"预计算的codes文件不存在: {codes_path}")
            return None, None, None
            
    except Exception as e:
        print(f"加载codes失败: {e}")
        return None, None, None
    
    # 4. 分析真实codes
    print(f"\n=== 分析真实Encoder输出的Codes ===")
    
    # 应用normalization
    if cfg.normalize_codes:
        # 计算code_stats
        code_stats = {
            'mean': raw_codes.mean(dim=(0, 1), keepdim=True),
            'std': raw_codes.std(dim=(0, 1), keepdim=True)
        }
        normalized_codes = normalize_code(raw_codes, code_stats)
    else:
        normalized_codes = raw_codes
        
    # 应用denoiser加噪
    smooth_codes = add_noise_to_code(normalized_codes, smooth_sigma=cfg.smooth_sigma)
    
    print(f"\n=== 真实Codes统计信息 ===")
    print(f"Raw codes shape: {raw_codes.shape}")
    print(f"Normalized codes shape: {normalized_codes.shape}")
    print(f"Smooth codes shape: {smooth_codes.shape}")
    
    # 5. 分析raw codes
    print(f"\n=== Raw Codes (真实Encoder输出) 分析 ===")
    analyze_codes_stats(raw_codes, "Raw Codes")
    
    # 6. 分析normalized codes
    print(f"\n=== Normalized Codes 分析 ===")
    analyze_codes_stats(normalized_codes, "Normalized Codes")
    
    # 7. 分析smooth codes
    print(f"\n=== Smooth Codes (Denoiser加噪后) 分析 ===")
    analyze_codes_stats(smooth_codes, "Smooth Codes")
    
    # 8. 创建可视化
    create_visualization(raw_codes, normalized_codes, smooth_codes, cfg.smooth_sigma)
    
    return raw_codes, normalized_codes, smooth_codes

def analyze_codes_stats(codes, name):
    """
    分析codes的统计信息
    """
    codes_np = codes.detach().cpu().numpy().flatten()
    
    print(f"\n{name} 基本统计:")
    print(f"  形状: {codes.shape}")
    print(f"  总元素数: {codes.numel():,}")
    print(f"  最小值: {codes.min().item():.6f}")
    print(f"  最大值: {codes.max().item():.6f}")
    print(f"  均值: {codes.mean().item():.6f}")
    print(f"  标准差: {codes.std().item():.6f}")
    print(f"  中位数: {codes.median().item():.6f}")
    
    # 检查异常值
    nan_count = torch.isnan(codes).sum().item()
    inf_count = torch.isinf(codes).sum().item()
    print(f"  NaN数量: {nan_count}")
    print(f"  Inf数量: {inf_count}")
    
    # 分析3σ异常值比例
    mean_val = codes.mean().item()
    std_val = codes.std().item()
    lower_3sigma = mean_val - 3 * std_val
    upper_3sigma = mean_val + 3 * std_val
    
    beyond_3sigma = ((codes < lower_3sigma) | (codes > upper_3sigma)).float()
    beyond_3sigma_ratio = beyond_3sigma.mean().item()
    beyond_3sigma_count = beyond_3sigma.sum().item()
    
    print(f"\n{name} 3σ异常值分析:")
    print(f"  3σ范围: [{lower_3sigma:.6f}, {upper_3sigma:.6f}]")
    print(f"  超出3σ的值数量: {beyond_3sigma_count:,} / {codes.numel():,}")
    print(f"  超出3σ的值比例: {beyond_3sigma_ratio:.6f} ({beyond_3sigma_ratio*100:.4f}%)")
    print(f"  理论期望比例: 0.002700 (0.2700%)")
    print(f"  实际/理论比例: {beyond_3sigma_ratio/0.0027:.2f}x")
    
    # 判断状态
    if beyond_3sigma_ratio > 0.01:
        status = "⚠️ 异常"
    elif beyond_3sigma_ratio > 0.005:
        status = "⚠️ 注意"
    else:
        status = "✅ 正常"
    print(f"  状态: {status}")
    
    # 分析不同σ范围的异常值比例
    print(f"\n{name} 不同σ范围异常值比例:")
    for sigma in [1, 2, 3, 4, 5]:
        lower_bound = mean_val - sigma * std_val
        upper_bound = mean_val + sigma * std_val
        beyond_sigma = ((codes < lower_bound) | (codes > upper_bound)).float()
        beyond_ratio = beyond_sigma.mean().item()
        
        theoretical_ratios = [0.3173, 0.0455, 0.0027, 0.0001, 0.0000]
        theoretical_ratio = theoretical_ratios[sigma-1]
        
        print(f"  ±{sigma}σ: {beyond_ratio:.4f} ({beyond_ratio*100:5.2f}%) [理论: {theoretical_ratio:.4f} ({theoretical_ratio*100:5.2f}%)]")

def create_visualization(raw_codes, normalized_codes, smooth_codes, smooth_sigma):
    """创建可视化图表"""
    
    # 转换为numpy
    raw_np = raw_codes.detach().cpu().numpy().flatten()
    norm_np = normalized_codes.detach().cpu().numpy().flatten()
    smooth_np = smooth_codes.detach().cpu().numpy().flatten()
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Raw codes分布
    axes[0, 0].hist(raw_np, bins=100, alpha=0.7, density=True, color='blue')
    axes[0, 0].axvline(raw_np.mean() - 3*raw_np.std(), color='red', linestyle='--', label='-3σ')
    axes[0, 0].axvline(raw_np.mean() + 3*raw_np.std(), color='red', linestyle='--', label='+3σ')
    axes[0, 0].set_title('Raw Codes (真实Encoder输出)')
    axes[0, 0].set_xlabel('Value')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Normalized codes分布
    axes[0, 1].hist(norm_np, bins=100, alpha=0.7, density=True, color='green')
    axes[0, 1].axvline(norm_np.mean() - 3*norm_np.std(), color='red', linestyle='--', label='-3σ')
    axes[0, 1].axvline(norm_np.mean() + 3*norm_np.std(), color='red', linestyle='--', label='+3σ')
    axes[0, 1].set_title('Normalized Codes')
    axes[0, 1].set_xlabel('Value')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Smooth codes分布
    axes[1, 0].hist(smooth_np, bins=100, alpha=0.7, density=True, color='orange')
    axes[1, 0].axvline(smooth_np.mean() - 3*smooth_np.std(), color='red', linestyle='--', label='-3σ')
    axes[1, 0].axvline(smooth_np.mean() + 3*smooth_np.std(), color='red', linestyle='--', label='+3σ')
    axes[1, 0].set_title(f'Smooth Codes (σ={smooth_sigma})')
    axes[1, 0].set_xlabel('Value')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 对比分布
    axes[1, 1].hist(raw_np, bins=50, alpha=0.5, density=True, label='Raw Codes', color='blue')
    axes[1, 1].hist(norm_np, bins=50, alpha=0.5, density=True, label='Normalized Codes', color='green')
    axes[1, 1].hist(smooth_np, bins=50, alpha=0.5, density=True, label='Smooth Codes', color='orange')
    axes[1, 1].set_title('Codes Distribution Comparison')
    axes[1, 1].set_xlabel('Value')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/datapool/data3/storage/pengxingang/pxg/hyc/funcmol-main-neuralfield/codes_analysis.png', dpi=300, bbox_inches='tight')
    print("\n可视化图表已保存到: codes_analysis.png")
    plt.close()

if __name__ == "__main__":
    print("开始分析真实的neural field encoder输出的codes...")
    try:
        raw_codes, normalized_codes, smooth_codes = analyze_real_codes()
        print("\n分析完成！")
    except Exception as e:
        print(f"分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
