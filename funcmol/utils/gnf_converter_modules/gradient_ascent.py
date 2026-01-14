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
        gradient_batch_size: Optional[int] = None,
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
            gradient_batch_size: 梯度计算时的批次大小。如果为None，则一次性处理所有点（batch_size=1）。
                                如果指定，则将点分成多个子批次处理，可能提高GPU利用率。
        """
        self.n_iter = n_iter
        self.step_size = step_size
        self.sigma_params = sigma_params
        self.sigma = sigma
        self.enable_early_stopping = enable_early_stopping
        self.convergence_threshold = convergence_threshold
        self.min_iterations = min_iterations
        self.gradient_batch_size = gradient_batch_size
    
    def batch_gradient_ascent(
        self,
        points: torch.Tensor,
        atom_types: torch.Tensor,
        current_codes: torch.Tensor,
        device: torch.device,
        decoder: nn.Module,
        iteration_callback: Optional[callable] = None,
        enable_timing: bool = False,
        clone_for_callback: bool = False
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
            clone_for_callback: 是否在回调中拷贝张量（默认False以减少内存开销）
            
        Returns:
            final_points: 最终点位置 [n_total_points, 3]
        """
        z = points.clone()
        prev_grad_norm = None  # 存储为标量（用于比较）
        prev_grad_norm_tensor = None  # 存储在 GPU 上的张量（用于避免同步）
        
        t_iter_start = time.perf_counter() if enable_timing else None
        iteration_times = [] if enable_timing else None
        
        n_total_points = z.size(0)
        
        # 确定是否使用分批处理
        use_batching = (self.gradient_batch_size is not None and 
                       self.gradient_batch_size > 0 and 
                       n_total_points > self.gradient_batch_size)
        
        # 打印调试信息（仅在启用时间统计时）
        if enable_timing and use_batching:
            n_batches = (n_total_points + self.gradient_batch_size - 1) // self.gradient_batch_size
            print(f"    [梯度上升] 使用分批处理: 总点数={n_total_points}, "
                  f"批次大小={self.gradient_batch_size}, 批次数={n_batches}")
        elif enable_timing:
            print(f"    [梯度上升] 未使用分批处理: 总点数={n_total_points}, "
                  f"配置的批次大小={self.gradient_batch_size}")
        
        # 预计算sigma_ratios和adjusted_step_sizes（避免每次迭代重复计算）
        # 优化：使用张量索引替代 .item() 循环，避免 GPU-CPU 同步
        # 创建 sigma 值查找表
        max_atom_type = int(atom_types.max().item()) if len(atom_types) > 0 else 0
        sigma_lookup = torch.full((max_atom_type + 1,), self.sigma, device=device, dtype=torch.float32)
        for atom_type, sigma_val in self.sigma_params.items():
            if atom_type <= max_atom_type:
                sigma_lookup[atom_type] = sigma_val
        
        # 使用索引查找，完全在 GPU 上操作
        sigma_ratios = (sigma_lookup[atom_types] / self.sigma)  # [n_total_points]
        adjusted_step_sizes = self.step_size * sigma_ratios.unsqueeze(-1)  # [n_total_points, 1]
        
        for iter_idx in range(self.n_iter):
            t_single_iter_start = time.perf_counter() if enable_timing else None
            
            try:
                # 使用torch.no_grad()包装，这是最关键的优化
                with torch.no_grad():
                    if use_batching:
                        # 分批处理：将点分成多个子批次
                        grad_list = []
                        batch_size = self.gradient_batch_size
                        
                        for batch_start in range(0, n_total_points, batch_size):
                            batch_end = min(batch_start + batch_size, n_total_points)
                            z_batch_subset = z[batch_start:batch_end].unsqueeze(0)  # [1, batch_size, 3]
                            
                            # 计算当前子批次的梯度场
                            current_field_subset = decoder(z_batch_subset, current_codes)  # [1, batch_size, n_atom_types, 3]
                            
                            # 为每个点选择对应原子类型的梯度
                            batch_indices = torch.arange(batch_end - batch_start, device=device)
                            batch_types = atom_types[batch_start:batch_end]
                            grad_subset = current_field_subset[0, batch_indices, batch_types, :]  # [batch_size, 3]
                            
                            grad_list.append(grad_subset)
                        
                        # 合并所有批次的梯度
                        grad = torch.cat(grad_list, dim=0)  # [n_total_points, 3]
                    else:
                        # 原始方式：一次性处理所有点
                        z_batch = z.unsqueeze(0)  # [1, n_total_points, 3]
                        current_field = decoder(z_batch, current_codes)  # [1, n_total_points, n_atom_types, 3]
                        
                        # 为每个点选择对应原子类型的梯度
                        point_indices = torch.arange(n_total_points, device=device)  # [n_total_points]
                        type_indices = atom_types  # [n_total_points]
                        grad = current_field[0, point_indices, type_indices, :]  # [n_total_points, 3]
                
                # 检查梯度是否包含NaN/Inf（在 GPU 上检查，避免同步）
                if torch.isnan(grad).any() or torch.isinf(grad).any():
                    break
                
                # 优化：在 GPU 上计算梯度模长，避免每次迭代都同步到 CPU
                current_grad_norm_tensor = torch.norm(grad, dim=-1).mean()  # 保持在 GPU 上
                
                # 检查收敛条件（仅在启用早停时）
                if self.enable_early_stopping:
                    if iter_idx >= self.min_iterations and prev_grad_norm_tensor is not None:
                        # 在 GPU 上计算梯度变化，避免同步
                        grad_change_tensor = torch.abs(current_grad_norm_tensor - prev_grad_norm_tensor)
                        # 只在满足条件时才同步到 CPU 进行检查
                        if grad_change_tensor.item() < self.convergence_threshold:
                            # 在停止前调用一次回调
                            if iteration_callback is not None:
                                callback_points = z.clone() if clone_for_callback else z
                                iteration_callback(iter_idx, callback_points, atom_types)
                            break
                
                # 更新梯度模长（保持在 GPU 上）
                prev_grad_norm_tensor = current_grad_norm_tensor
                # 为了向后兼容，也更新标量版本（但只在需要时）
                if enable_timing or (self.enable_early_stopping and iter_idx < self.min_iterations):
                    prev_grad_norm = current_grad_norm_tensor.item()
                
                # 更新采样点位置
                z = z + adjusted_step_sizes * grad
                
                # 调用迭代回调（如果提供）
                # 优化：默认不拷贝，减少内存开销
                if iteration_callback is not None:
                    callback_points = z.clone() if clone_for_callback else z
                    iteration_callback(iter_idx, callback_points, atom_types)
                
            except (RuntimeError, ValueError, IndexError):
                # 发生错误时直接中断，不清理缓存（避免性能开销）
                # 缓存清理应在外层统一处理
                break
            
            # 记录单次迭代时间
            if enable_timing and t_single_iter_start is not None:
                iteration_times.append(time.perf_counter() - t_single_iter_start)
        
        # 打印梯度上升的详细时间信息
        if enable_timing and iteration_times:
            total_iter_time = sum(iteration_times)
            avg_iter_time = total_iter_time / len(iteration_times) if iteration_times else 0.0
            if use_batching:
                n_batches = (n_total_points + self.gradient_batch_size - 1) // self.gradient_batch_size
                print(f"    [梯度上升] 总迭代数={len(iteration_times)}, "
                      f"总时间={total_iter_time:.3f}s, "
                      f"平均每次迭代={avg_iter_time*1000:.2f}ms, "
                      f"分批处理: 总点数={n_total_points}, 批次大小={self.gradient_batch_size}, 每迭代批次数={n_batches}")
            else:
                print(f"    [梯度上升] 总迭代数={len(iteration_times)}, "
                      f"总时间={total_iter_time:.3f}s, "
                      f"平均每次迭代={avg_iter_time*1000:.2f}ms, "
                      f"未分批: 总点数={n_total_points}")
        
        return z

