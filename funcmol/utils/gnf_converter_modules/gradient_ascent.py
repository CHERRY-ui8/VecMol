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
        enable_timing: bool = False
    ) -> torch.Tensor:
        """
        批量梯度上升，对所有原子类型的点同时进行梯度上升。
        支持自适应停止：当梯度变化很小时提前停止。
        支持批量处理：如果points是[B, n_points, 3]，则批量处理B个样本。
        
        Args:
            points: 采样点 [n_total_points, 3] 或 [B, n_total_points, 3]（批量模式）
            atom_types: 原子类型 [n_total_points] 或 [B, n_total_points]（批量模式）
            current_codes: 当前编码 [1, code_dim] 或 [B, code_dim]（批量模式）
            device: 设备
            decoder: 解码器模型
            iteration_callback: 可选的回调函数，在每次迭代时调用，参数为 (iteration_idx, current_points, atom_types, batch_idx=None)
                                如果batch_idx=None，则current_points是[B, n_points, 3]，需要分别处理每个batch
                                如果batch_idx有值，则current_points是[n_points, 3]，处理单个batch
            enable_timing: 是否启用时间统计
            
        Returns:
            final_points: 最终点位置 [n_total_points, 3] 或 [B, n_total_points, 3]（批量模式）
        """
        # 判断是否是批量模式
        is_batch_mode = points.dim() == 3  # [B, n_points, 3]
        
        if is_batch_mode:
            # 批量模式
            B, n_total_points, _ = points.shape
            z = points.clone()  # [B, n_total_points, 3]
            atom_types_batch = atom_types  # [B, n_total_points]
            codes_batch = current_codes  # [B, code_dim]
        else:
            # 单样本模式（向后兼容）
            B = 1
            n_total_points = points.size(0)
            z = points.clone().unsqueeze(0)  # [1, n_total_points, 3]
            atom_types_batch = atom_types.unsqueeze(0)  # [1, n_total_points]
            codes_batch = current_codes  # [1, code_dim]
        
        # 为每个batch维护独立的prev_grad_norm（用于早停）
        prev_grad_norms = [None] * B
        active_batches = list(range(B))  # 跟踪仍在运行的batch
        
        t_iter_start = time.perf_counter() if enable_timing else None
        iteration_times = [] if enable_timing else None
        
        # 确定是否使用点级分批处理（将点分成多个子批次）
        use_point_batching = (self.gradient_batch_size is not None and 
                              self.gradient_batch_size > 0 and 
                              n_total_points > self.gradient_batch_size)
        
        # 打印调试信息（仅在启用时间统计时）
        if enable_timing:
            if is_batch_mode:
                print(f"    [梯度上升] 批量模式: 样本数={B}, 每样本点数={n_total_points}")
            if use_point_batching:
                n_batches = (n_total_points + self.gradient_batch_size - 1) // self.gradient_batch_size
                print(f"    [梯度上升] 点级分批: 总点数={n_total_points}, "
                      f"批次大小={self.gradient_batch_size}, 每迭代批次数={n_batches}")
        
        # 预计算sigma_ratios和adjusted_step_sizes（避免每次迭代重复计算）
        # 对于批量模式，为每个batch分别计算
        if is_batch_mode:
            adjusted_step_sizes_list = []
            for b in range(B):
                sigma_ratios = torch.tensor([self.sigma_params.get(t.item(), self.sigma) / self.sigma 
                                           for t in atom_types_batch[b]], device=device)  # [n_total_points]
                adjusted_step_sizes = self.step_size * sigma_ratios.unsqueeze(-1)  # [n_total_points, 1]
                adjusted_step_sizes_list.append(adjusted_step_sizes)
            adjusted_step_sizes_batch = torch.stack(adjusted_step_sizes_list, dim=0)  # [B, n_total_points, 1]
        else:
            sigma_ratios = torch.tensor([self.sigma_params.get(t.item(), self.sigma) / self.sigma 
                                       for t in atom_types], device=device)  # [n_total_points]
            adjusted_step_sizes = self.step_size * sigma_ratios.unsqueeze(-1)  # [n_total_points, 1]
        
        for iter_idx in range(self.n_iter):
            if len(active_batches) == 0:
                break  # 所有batch都已收敛
                
            t_single_iter_start = time.perf_counter() if enable_timing else None
            
            try:                
                # 使用torch.no_grad()包装，这是最关键的优化
                with torch.no_grad():
                    if is_batch_mode:
                        # 批量模式：一次性处理所有batch
                        if use_point_batching:
                            # 点级分批：将每个batch的点分成多个子批次
                            grad_batch_list = []
                            point_batch_size = self.gradient_batch_size
                            
                            for point_batch_start in range(0, n_total_points, point_batch_size):
                                point_batch_end = min(point_batch_start + point_batch_size, n_total_points)
                                z_subset = z[:, point_batch_start:point_batch_end, :]  # [B, point_batch_size, 3]
                                
                                # 批量调用decoder
                                current_field_subset = decoder(z_subset, codes_batch)  # [B, point_batch_size, n_atom_types, 3]
                                
                                # 为每个batch和每个点选择对应原子类型的梯度
                                grad_subset_list = []
                                for b in range(B):
                                    point_indices = torch.arange(point_batch_end - point_batch_start, device=device)
                                    type_indices = atom_types_batch[b, point_batch_start:point_batch_end]
                                    grad_subset = current_field_subset[b, point_indices, type_indices, :]  # [point_batch_size, 3]
                                    grad_subset_list.append(grad_subset)
                                
                                grad_batch_list.append(torch.stack(grad_subset_list, dim=0))  # [B, point_batch_size, 3]
                            
                            # 合并所有点级批次的梯度
                            grad_batch = torch.cat(grad_batch_list, dim=1)  # [B, n_total_points, 3]
                        else:
                            # 一次性处理所有点
                            current_field_batch = decoder(z, codes_batch)  # [B, n_total_points, n_atom_types, 3]
                            
                            # 为每个batch和每个点选择对应原子类型的梯度
                            grad_batch_list = []
                            for b in range(B):
                                point_indices = torch.arange(n_total_points, device=device)
                                type_indices = atom_types_batch[b]  # [n_total_points]
                                grad_b = current_field_batch[b, point_indices, type_indices, :]  # [n_total_points, 3]
                                grad_batch_list.append(grad_b)
                            grad_batch = torch.stack(grad_batch_list, dim=0)  # [B, n_total_points, 3]
                        
                        # 检查梯度是否包含NaN/Inf
                        if torch.isnan(grad_batch).any() or torch.isinf(grad_batch).any():
                            break
                        
                        # 计算每个batch的梯度模长
                        current_grad_norms = torch.norm(grad_batch, dim=-1).mean(dim=-1)  # [B]
                        
                        # 更新采样点位置
                        z = z + adjusted_step_sizes_batch * grad_batch  # [B, n_total_points, 3]
                        
                        # 检查收敛条件（仅在启用早停时）
                        if self.enable_early_stopping:
                            still_active = []
                            for b in active_batches:
                                current_grad_norm = current_grad_norms[b].item()
                                if iter_idx >= self.min_iterations and prev_grad_norms[b] is not None:
                                    grad_change = abs(current_grad_norm - prev_grad_norms[b])
                                    if grad_change < self.convergence_threshold:
                                        # 在停止前调用一次回调
                                        if iteration_callback is not None:
                                            iteration_callback(iter_idx, z[b].clone(), atom_types_batch[b], batch_idx=b)
                                        continue  # 这个batch已收敛，跳过
                                prev_grad_norms[b] = current_grad_norm
                                still_active.append(b)
                            active_batches = still_active
                        else:
                            prev_grad_norms = [current_grad_norms[b].item() for b in range(B)]
                        
                        # 调用迭代回调（如果提供）
                        if iteration_callback is not None:
                            # 批量模式：传递所有batch的数据，batch_idx=None
                            iteration_callback(iter_idx, z.clone(), atom_types_batch, batch_idx=None)
                    else:
                        # 单样本模式（向后兼容）
                        if use_point_batching:
                            # 点级分批处理
                            grad_list = []
                            point_batch_size = self.gradient_batch_size
                            
                            for point_batch_start in range(0, n_total_points, point_batch_size):
                                point_batch_end = min(point_batch_start + point_batch_size, n_total_points)
                                z_subset = z[0, point_batch_start:point_batch_end, :].unsqueeze(0)  # [1, point_batch_size, 3]
                                
                                current_field_subset = decoder(z_subset, codes_batch)  # [1, point_batch_size, n_atom_types, 3]
                                
                                point_indices = torch.arange(point_batch_end - point_batch_start, device=device)
                                type_indices = atom_types_batch[0, point_batch_start:point_batch_end]
                                grad_subset = current_field_subset[0, point_indices, type_indices, :]  # [point_batch_size, 3]
                                
                                grad_list.append(grad_subset)
                            
                            grad = torch.cat(grad_list, dim=0)  # [n_total_points, 3]
                        else:
                            # 一次性处理所有点
                            current_field = decoder(z, codes_batch)  # [1, n_total_points, n_atom_types, 3]
                            
                            point_indices = torch.arange(n_total_points, device=device)
                            type_indices = atom_types_batch[0]
                            grad = current_field[0, point_indices, type_indices, :]  # [n_total_points, 3]
                        
                        # 检查梯度是否包含NaN/Inf
                        if torch.isnan(grad).any() or torch.isinf(grad).any():
                            break
                        
                        # 计算当前梯度的模长
                        current_grad_norm = torch.norm(grad, dim=-1).mean().item()
                        
                        # 检查收敛条件（仅在启用早停时）
                        if self.enable_early_stopping:
                            if iter_idx >= self.min_iterations and prev_grad_norms[0] is not None:
                                grad_change = abs(current_grad_norm - prev_grad_norms[0])
                                if grad_change < self.convergence_threshold:
                                    # 在停止前调用一次回调
                                    if iteration_callback is not None:
                                        iteration_callback(iter_idx, z[0].clone(), atom_types_batch[0], batch_idx=0)
                                    break
                        
                        prev_grad_norms[0] = current_grad_norm
                        
                        # 更新采样点位置
                        z[0] = z[0] + adjusted_step_sizes * grad
                        
                        # 调用迭代回调（如果提供）
                        if iteration_callback is not None:
                            iteration_callback(iter_idx, z[0].clone(), atom_types_batch[0], batch_idx=0)
                
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
            if is_batch_mode:
                batch_info = f"批量模式: {B}个样本"
            else:
                batch_info = "单样本模式"
            if use_point_batching:
                n_batches = (n_total_points + self.gradient_batch_size - 1) // self.gradient_batch_size
                print(f"    [梯度上升] 总迭代数={len(iteration_times)}, "
                      f"总时间={total_iter_time:.3f}s, "
                      f"平均每次迭代={avg_iter_time*1000:.2f}ms, "
                      f"{batch_info}, 点级分批: 总点数={n_total_points}, 批次大小={self.gradient_batch_size}, 每迭代批次数={n_batches}")
            else:
                print(f"    [梯度上升] 总迭代数={len(iteration_times)}, "
                      f"总时间={total_iter_time:.3f}s, "
                      f"平均每次迭代={avg_iter_time*1000:.2f}ms, "
                      f"{batch_info}, 未点级分批: 总点数={n_total_points}")
        
        # 返回结果
        if is_batch_mode:
            return z  # [B, n_total_points, 3]
        else:
            return z[0]  # [n_total_points, 3]

