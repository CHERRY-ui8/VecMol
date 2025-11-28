#!/usr/bin/env python3
"""
分析对于当前数值范围的codes，smooth_sigma是否合理
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# 添加项目路径
sys.path.append('/home/huayuchen/Neurl-voxel')

from funcmol.utils.utils_fm import add_noise_to_code
from funcmol.utils.utils_nf import normalize_code

def analyze_smooth_sigma_appropriateness():
    """
    分析smooth_sigma对于当前codes数值范围的合理性
    """
    
    print("=== 分析smooth_sigma的合理性 ===")
    
    device = torch.device("cpu")
    print(f"使用设备: {device}")
    
    # 1. 加载真实的codes
    codes_path = "/home/huayuchen/Neurl-voxel/exps/neural_field/nf_qm9/20250911/lightning_logs/version_1/checkpoints/codes/train/codes.pt"
    
    print(f"\n=== 加载真实Codes ===")
    try:
        raw_codes = torch.load(codes_path, map_location=device)
        # 只取前1000个样本进行分析
        raw_codes = raw_codes[:1000]
        print(f"成功加载codes: {raw_codes.shape}")
    except Exception as e:
        print(f"加载失败: {e}")
        return
    
    # 2. 应用normalization
    print(f"\n=== 应用Normalization ===")
    code_stats = {
        'mean': raw_codes.mean(dim=(0, 1), keepdim=True),
        'std': raw_codes.std(dim=(0, 1), keepdim=True)
    }
    normalized_codes = normalize_code(raw_codes, code_stats)
    
    print(f"Raw codes - 范围: [{raw_codes.min().item():.3f}, {raw_codes.max().item():.3f}], 标准差: {raw_codes.std().item():.3f}")
    print(f"Normalized codes - 范围: [{normalized_codes.min().item():.3f}, {normalized_codes.max().item():.3f}], 标准差: {normalized_codes.std().item():.3f}")
    
    # 3. 分析不同smooth_sigma值的影响
    print(f"\n=== 分析不同smooth_sigma值的影响 ===")
    
    sigma_values = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
    
    print(f"{'σ值':<8} {'噪声比例':<12} {'数值范围':<20} {'3σ异常值比例':<15} {'状态'}")
    print("-" * 80)
    
    results = []
    
    for sigma in sigma_values:
        # 应用加噪
        smooth_codes = add_noise_to_code(normalized_codes, smooth_sigma=sigma)
        
        # 计算噪声比例
        noise = smooth_codes - normalized_codes
        noise_ratio = noise.std().item() / normalized_codes.std().item()
        
        # 计算数值范围
        value_range = f"[{smooth_codes.min().item():.2f}, {smooth_codes.max().item():.2f}]"
        
        # 计算3σ异常值比例
        mean_val = smooth_codes.mean().item()
        std_val = smooth_codes.std().item()
        beyond_3sigma = ((smooth_codes < mean_val - 3 * std_val) | (smooth_codes > mean_val + 3 * std_val)).float()
        beyond_3sigma_ratio = beyond_3sigma.mean().item()
        
        # 判断状态
        if beyond_3sigma_ratio > 0.01:
            status = "⚠️ 严重异常"
        elif beyond_3sigma_ratio > 0.005:
            status = "⚠️ 异常"
        elif beyond_3sigma_ratio > 0.003:
            status = "⚠️ 注意"
        else:
            status = "✅ 正常"
        
        print(f"{sigma:<8} {noise_ratio:.3f} ({noise_ratio*100:5.1f}%) {value_range:<20} {beyond_3sigma_ratio:.4f} ({beyond_3sigma_ratio*100:5.2f}%) {status}")
        
        results.append({
            'sigma': sigma,
            'noise_ratio': noise_ratio,
            'value_range': value_range,
            'beyond_3sigma_ratio': beyond_3sigma_ratio,
            'status': status
        })
    
    # 4. 分析当前smooth_sigma=0.5的合理性
    print(f"\n=== 当前smooth_sigma=0.5的合理性分析 ===")
    
    current_sigma = 0.5
    smooth_codes_current = add_noise_to_code(normalized_codes, smooth_sigma=current_sigma)
    noise_current = smooth_codes_current - normalized_codes
    noise_ratio_current = noise_current.std().item() / normalized_codes.std().item()
    
    mean_val = smooth_codes_current.mean().item()
    std_val = smooth_codes_current.std().item()
    beyond_3sigma_current = ((smooth_codes_current < mean_val - 3 * std_val) | (smooth_codes_current > mean_val + 3 * std_val)).float()
    beyond_3sigma_ratio_current = beyond_3sigma_current.mean().item()
    
    print(f"当前smooth_sigma: {current_sigma}")
    print(f"噪声比例: {noise_ratio_current:.3f} ({noise_ratio_current*100:.1f}%)")
    print(f"数值范围: [{smooth_codes_current.min().item():.3f}, {smooth_codes_current.max().item():.3f}]")
    print(f"3σ异常值比例: {beyond_3sigma_ratio_current:.4f} ({beyond_3sigma_ratio_current*100:.2f}%)")
    
    # 5. 推荐合适的smooth_sigma范围
    print(f"\n=== 推荐合适的smooth_sigma范围 ===")
    
    # 找到噪声比例在合理范围内的sigma值
    reasonable_sigmas = []
    for result in results:
        noise_ratio = result['noise_ratio']
        beyond_3sigma_ratio = result['beyond_3sigma_ratio']
        
        # 合理的噪声比例应该在0.1-0.5之间，3σ异常值比例应该接近理论值
        if 0.1 <= noise_ratio <= 0.5 and beyond_3sigma_ratio <= 0.005:
            reasonable_sigmas.append(result)
    
    if reasonable_sigmas:
        print("推荐的smooth_sigma值:")
        for result in reasonable_sigmas:
            print(f"  σ={result['sigma']}: 噪声比例={result['noise_ratio']:.3f}, 3σ异常值比例={result['beyond_3sigma_ratio']:.4f}")
    else:
        print("⚠️ 没有找到完全合理的smooth_sigma值，建议:")
        # 找到最接近合理的值
        best_result = min(results, key=lambda x: abs(x['beyond_3sigma_ratio'] - 0.0027))
        print(f"  最接近理论期望的σ值: {best_result['sigma']} (3σ异常值比例: {best_result['beyond_3sigma_ratio']:.4f})")
    
    # 6. 分析噪声强度对denoiser学习的影响
    print(f"\n=== 噪声强度对Denoiser学习的影响分析 ===")
    
    print("噪声比例分析:")
    print("  < 0.1: 噪声太小，denoiser可能学习不到有效的去噪能力")
    print("  0.1-0.3: 适中的噪声，有利于denoiser学习")
    print("  0.3-0.5: 较强的噪声，需要denoiser有更强的学习能力")
    print("  > 0.5: 噪声过强，可能导致denoiser学习困难")
    
    print(f"\n当前噪声比例: {noise_ratio_current:.3f}")
    if noise_ratio_current < 0.1:
        print("  → 噪声太小，建议增加smooth_sigma")
    elif noise_ratio_current > 0.5:
        print("  → 噪声过强，建议减少smooth_sigma")
    else:
        print("  → 噪声强度适中")
    
    # 7. 创建可视化
    create_sigma_analysis_plot(results, current_sigma, noise_ratio_current, beyond_3sigma_ratio_current)
    
    return results

def create_sigma_analysis_plot(results, current_sigma, current_noise_ratio, current_beyond_3sigma_ratio):
    """创建smooth_sigma分析的可视化图表"""
    
    sigmas = [r['sigma'] for r in results]
    noise_ratios = [r['noise_ratio'] for r in results]
    beyond_3sigma_ratios = [r['beyond_3sigma_ratio'] for r in results]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 噪声比例 vs smooth_sigma
    axes[0, 0].plot(sigmas, noise_ratios, 'bo-', linewidth=2, markersize=6)
    axes[0, 0].axvline(current_sigma, color='red', linestyle='--', alpha=0.7, label=f'Current σ={current_sigma}')
    axes[0, 0].axhline(0.1, color='green', linestyle=':', alpha=0.7, label='Min recommended')
    axes[0, 0].axhline(0.5, color='orange', linestyle=':', alpha=0.7, label='Max recommended')
    axes[0, 0].set_xlabel('smooth_sigma')
    axes[0, 0].set_ylabel('Noise Ratio')
    axes[0, 0].set_title('Noise Ratio vs smooth_sigma')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 3σ异常值比例 vs smooth_sigma
    axes[0, 1].plot(sigmas, beyond_3sigma_ratios, 'ro-', linewidth=2, markersize=6)
    axes[0, 1].axvline(current_sigma, color='red', linestyle='--', alpha=0.7, label=f'Current σ={current_sigma}')
    axes[0, 1].axhline(0.0027, color='green', linestyle=':', alpha=0.7, label='Theoretical (0.27%)')
    axes[0, 1].axhline(0.005, color='orange', linestyle=':', alpha=0.7, label='Warning threshold (0.5%)')
    axes[0, 1].set_xlabel('smooth_sigma')
    axes[0, 1].set_ylabel('3σ Outlier Ratio')
    axes[0, 1].set_title('3σ Outlier Ratio vs smooth_sigma')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_yscale('log')
    
    # 3. 噪声比例分布
    axes[1, 0].bar(range(len(sigmas)), noise_ratios, alpha=0.7, color='blue')
    axes[1, 0].axhline(0.1, color='green', linestyle='--', alpha=0.7, label='Min recommended')
    axes[1, 0].axhline(0.5, color='orange', linestyle='--', alpha=0.7, label='Max recommended')
    axes[1, 0].set_xlabel('smooth_sigma index')
    axes[1, 0].set_ylabel('Noise Ratio')
    axes[1, 0].set_title('Noise Ratio Distribution')
    axes[1, 0].set_xticks(range(len(sigmas)))
    axes[1, 0].set_xticklabels([f'{s:.2f}' for s in sigmas], rotation=45)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 综合评估
    # 计算综合评分 (噪声比例越接近0.3越好，3σ异常值比例越接近0.0027越好)
    scores = []
    for r in results:
        noise_score = 1 - abs(r['noise_ratio'] - 0.3) / 0.3  # 0.3是最佳噪声比例
        outlier_score = 1 - abs(r['beyond_3sigma_ratio'] - 0.0027) / 0.0027  # 0.0027是理论期望
        combined_score = (noise_score + outlier_score) / 2
        scores.append(combined_score)
    
    axes[1, 1].plot(sigmas, scores, 'go-', linewidth=2, markersize=6)
    axes[1, 1].axvline(current_sigma, color='red', linestyle='--', alpha=0.7, label=f'Current σ={current_sigma}')
    axes[1, 1].set_xlabel('smooth_sigma')
    axes[1, 1].set_ylabel('Combined Score')
    axes[1, 1].set_title('Combined Quality Score')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/home/huayuchen/Neurl-voxel/smooth_sigma_analysis.png', dpi=300, bbox_inches='tight')
    print("\n可视化图表已保存到: smooth_sigma_analysis.png")
    plt.close()

if __name__ == "__main__":
    print("开始分析smooth_sigma的合理性...")
    results = analyze_smooth_sigma_appropriateness()
    print("\n分析完成！")
