#!/usr/bin/env python3
import os
import torch

def test_gpu_setup():
    print("=== GPU设置测试 ===")
    
    # 设置CUDA_VISIBLE_DEVICES
    gpu_devices = [6, 7]
    gpu_str = ",".join(map(str, gpu_devices))
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_str
    
    print(f"设置 CUDA_VISIBLE_DEVICES = {gpu_str}")
    
    # 检查CUDA是否可用
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"可用GPU数量: {num_gpus}")
        
        for i in range(num_gpus):
            gpu_name = torch.cuda.get_device_name(i)
            print(f"GPU {i}: {gpu_name}")
            
            # 尝试分配一些内存
            try:
                device = torch.device(f"cuda:{i}")
                x = torch.randn(1000, 1000, device=device)
                print(f"  - 成功在GPU {i}上分配内存")
                del x
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"  - 在GPU {i}上分配内存失败: {e}")
    else:
        print("CUDA不可用")

if __name__ == "__main__":
    test_gpu_setup()
