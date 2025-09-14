#!/usr/bin/env python3
"""
查看codes文件的详细信息
"""

import torch
import sys
import os

# 添加项目路径
sys.path.append('/datapool/data3/storage/pengxingang/pxg/hyc/funcmol-main-neuralfield')

def check_codes_info():
    """查看codes文件的详细信息"""
    
    device = torch.device("cpu")  # 使用CPU避免内存问题
    print(f"使用设备: {device}")
    
    codes_path = "/datapool/data3/storage/pengxingang/pxg/hyc/funcmol-main-neuralfield/exps/neural_field/nf_qm9/20250911/lightning_logs/version_1/checkpoints/codes/train/codes.pt"
    
    print(f"\n=== Codes文件信息 ===")
    print(f"文件路径: {codes_path}")
    
    # 检查文件是否存在
    if not os.path.exists(codes_path):
        print("❌ 文件不存在!")
        return
    
    # 获取文件大小
    file_size = os.path.getsize(codes_path)
    print(f"文件大小: {file_size:,} bytes ({file_size/1024/1024/1024:.2f} GB)")
    
    # 加载codes
    print(f"\n=== 加载Codes ===")
    try:
        codes = torch.load(codes_path, map_location=device)
        print(f"✅ 成功加载codes")
        
        print(f"\n=== Codes详细信息 ===")
        print(f"数据类型: {type(codes)}")
        
        if isinstance(codes, torch.Tensor):
            print(f"Tensor形状: {codes.shape}")
            print(f"数据类型: {codes.dtype}")
            print(f"设备: {codes.device}")
            print(f"总元素数: {codes.numel():,}")
            
            # 计算内存占用
            element_size = codes.element_size()
            total_memory = codes.numel() * element_size
            print(f"元素大小: {element_size} bytes")
            print(f"总内存占用: {total_memory:,} bytes ({total_memory/1024/1024/1024:.2f} GB)")
            
            # 基本统计信息
            print(f"\n=== 基本统计信息 ===")
            print(f"最小值: {codes.min().item():.6f}")
            print(f"最大值: {codes.max().item():.6f}")
            print(f"均值: {codes.mean().item():.6f}")
            print(f"标准差: {codes.std().item():.6f}")
            print(f"中位数: {codes.median().item():.6f}")
            
            # 检查异常值
            nan_count = torch.isnan(codes).sum().item()
            inf_count = torch.isinf(codes).sum().item()
            print(f"NaN数量: {nan_count}")
            print(f"Inf数量: {inf_count}")
            
            # 3σ异常值分析
            mean_val = codes.mean().item()
            std_val = codes.std().item()
            lower_3sigma = mean_val - 3 * std_val
            upper_3sigma = mean_val + 3 * std_val
            
            beyond_3sigma = ((codes < lower_3sigma) | (codes > upper_3sigma)).float()
            beyond_3sigma_ratio = beyond_3sigma.mean().item()
            beyond_3sigma_count = beyond_3sigma.sum().item()
            
            print(f"\n=== 3σ异常值分析 ===")
            print(f"3σ范围: [{lower_3sigma:.6f}, {upper_3sigma:.6f}]")
            print(f"超出3σ的值数量: {beyond_3sigma_count:,} / {codes.numel():,}")
            print(f"超出3σ的值比例: {beyond_3sigma_ratio:.6f} ({beyond_3sigma_ratio*100:.4f}%)")
            print(f"理论期望比例: 0.002700 (0.2700%)")
            print(f"实际/理论比例: {beyond_3sigma_ratio/0.0027:.2f}x")
            
            # 判断状态
            if beyond_3sigma_ratio > 0.01:
                status = "⚠️ 严重异常"
            elif beyond_3sigma_ratio > 0.005:
                status = "⚠️ 异常"
            elif beyond_3sigma_ratio > 0.003:
                status = "⚠️ 注意"
            else:
                status = "✅ 正常"
            print(f"状态: {status}")
            
        elif isinstance(codes, (list, tuple)):
            print(f"列表/元组长度: {len(codes)}")
            if len(codes) > 0:
                print(f"第一个元素类型: {type(codes[0])}")
                if isinstance(codes[0], torch.Tensor):
                    print(f"第一个元素形状: {codes[0].shape}")
        else:
            print(f"其他类型: {type(codes)}")
            
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_codes_info()
