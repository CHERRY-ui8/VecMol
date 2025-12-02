"""
梯度上升模块：梯度上升优化过程
"""
import torch
import torch.nn as nn
import time
import gc
from typing import Optional, Dict


class GradientAscentOptimizer:
    """梯度上升优化器"""
    
    def __init__(
        self,
        n_iter: int,
        step_size: float,
        sigma_params: Dict[int, float],
        sigma: float,
        enable_early_stopping: bool = True,
        convergence_threshold: float = 1e-6,
        min_iterations: int = 50,
    ):
        """
        初始化梯度上升优化器
        
        Args:
            n_iter: 最大迭代次数
            step_size: 步长
            sigma_params: 每个原子类型的sigma参数字典
            sigma: 默认sigma值
            enable_early_stopping: 是否启用早停机制
            convergence_threshold: 收敛阈值
            min_iterations: 最少迭代次数
        """
        self.n_iter = n_iter
        self.step_size = step_size
        self.sigma_params = sigma_params
        self.sigma = sigma
        self.enable_early_stopping = enable_early_stopping
        self.convergence_threshold = convergence_threshold
        self.min_iterations = min_iterations
    
    def batch_gradient_ascent(
        self,
        points: torch.Tensor,
        atom_types: torch.Tensor,
        current_codes: torch.Tensor,
        device: torch.device,
        decoder: nn.Module,
        iteration_callback: Optional[callable] = None,
        enable_timing: bool = False
    ) -> torch.Tensor:
        """
        批量梯度上升，对所有原子类型的点同时进行梯度上升。
        支持自适应停止：当梯度变化很小时提前停止。
        
        Args:
            points: 采样点 [n_total_points, 3]
            atom_types: 原子类型 [n_total_points]
            current_codes: 当前编码 [1, code_dim]
            device: 设备
            decoder: 解码器模型
            iteration_callback: 可选的回调函数，在每次迭代时调用，参数为 (iteration_idx, current_points, atom_types)
            enable_timing: 是否启用时间统计
            
        Returns:
            final_points: 最终点位置 [n_total_points, 3]
        """
        z = points.clone()
        prev_grad_norm = None
        
        t_iter_start = time.perf_counter() if enable_timing else None
        iteration_times = [] if enable_timing else None
        
        for iter_idx in range(self.n_iter):
            t_single_iter_start = time.perf_counter() if enable_timing else None
            z_batch = z.unsqueeze(0)  # [1, n_total_points, 3]
            
            try:                
                # 使用torch.no_grad()包装，这是最关键的优化
                with torch.no_grad():
                    current_field = decoder(z_batch, current_codes)  # [1, n_total_points, n_atom_types, 3]
                
                # 为每个点选择对应原子类型的梯度
                # 使用高级索引选择梯度
                point_indices = torch.arange(z.size(0), device=device)  # [n_total_points]
                type_indices = atom_types  # [n_total_points]
                grad = current_field[0, point_indices, type_indices, :]  # [n_total_points, 3]
                
                # 检查梯度是否包含NaN/Inf
                if torch.isnan(grad).any() or torch.isinf(grad).any():
                    break
                
                # 使用原子类型特定的sigma调整步长
                sigma_ratios = torch.tensor([self.sigma_params.get(t.item(), self.sigma) / self.sigma 
                                           for t in atom_types], device=device)  # [n_total_points]
                adjusted_step_sizes = self.step_size * sigma_ratios.unsqueeze(-1)  # [n_total_points, 1]
                
                # 计算当前梯度的模长
                current_grad_norm = torch.norm(grad, dim=-1).mean().item()
                
                # 检查收敛条件（仅在启用早停时）
                if self.enable_early_stopping:
                    if iter_idx >= self.min_iterations and prev_grad_norm is not None:
                        grad_change = abs(current_grad_norm - prev_grad_norm)
                        if grad_change < self.convergence_threshold:
                            # 在停止前调用一次回调
                            if iteration_callback is not None:
                                iteration_callback(iter_idx, z.clone(), atom_types)
                            break
                
                prev_grad_norm = current_grad_norm
                
                # 更新采样点位置
                z = z + adjusted_step_sizes * grad
                
                # 调用迭代回调（如果提供）
                if iteration_callback is not None:
                    iteration_callback(iter_idx, z.clone(), atom_types)
                
            except (RuntimeError, ValueError, IndexError):
                # 发生错误时也清理内存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()
                break
            
            # 记录单次迭代时间
            if enable_timing and t_single_iter_start is not None:
                iteration_times.append(time.perf_counter() - t_single_iter_start)
        
        # 打印梯度上升的详细时间信息
        if enable_timing and iteration_times:
            total_iter_time = sum(iteration_times)
            avg_iter_time = total_iter_time / len(iteration_times) if iteration_times else 0.0
            print(f"    [梯度上升] 总迭代数={len(iteration_times)}, "
                  f"总时间={total_iter_time:.3f}s, "
                  f"平均每次迭代={avg_iter_time*1000:.2f}ms")
        
        return z

