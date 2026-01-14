import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, Optional, Tuple


# ===========================
# 1. β 调度与常数
# ===========================
def get_beta_schedule(beta_start: float, beta_end: float, num_timesteps: int, 
                    schedule: str = "linear", w: float = 6.0, s: float = 0.008) -> torch.Tensor:
    """
    获取β调度序列
    
    Args:
        beta_start: 起始β值
        beta_end: 结束β值
        num_timesteps: 时间步数
        schedule: 调度类型 ("linear", "cosine", 或 "sigmoid")
        w: sigmoid调度的宽度参数（仅用于sigmoid调度）
        s: cosine调度的偏移参数（仅用于cosine调度），控制余弦曲线的偏移，默认0.008
    
    Returns:
        betas: β序列张量
    """
    if schedule == "linear":
        betas = np.linspace(beta_start, beta_end, num_timesteps, dtype=np.float64)
    elif schedule == "cosine":
        # 修复：使用正确的边界，避免alphas_cumprod[-1]=0的问题
        # s参数控制余弦曲线的偏移，影响噪声添加的曲线形状
        timesteps = num_timesteps
        x = np.linspace(0, timesteps, timesteps + 1)
        alphas_cumprod = np.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    elif schedule == "sigmoid":
        # Use w parameter for sigmoid steepness
        xs = np.linspace(-w, w, num_timesteps)
        betas = sigmoid(xs) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(f"Schedule {schedule} not implemented")
    
    # 数值稳定性：clamp betas到合理范围
    betas = np.clip(betas, 1e-8, 0.999)
    return torch.tensor(betas, dtype=torch.float32)


def prepare_diffusion_constants(betas: torch.Tensor, device: Optional[torch.device] = None) -> Dict[str, torch.Tensor]:
    """
    准备扩散过程的常数
    
    Args:
        betas: β序列
        device: 目标设备（如果提供，所有常数将放在该设备上）
    
    Returns:
        包含所有扩散常数的字典
    """
    if device is not None:
        betas = betas.to(device)
    
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = torch.cat([
        torch.tensor([1.0], device=betas.device, dtype=betas.dtype), 
        alphas_cumprod[:-1]
    ], dim=0)

    # 数值稳定性：使用eps防止除0和sqrt(0)
    eps = 1e-20
    
    # 计算posterior_variance时防止除0
    one_minus_alphas_cumprod = (1 - alphas_cumprod).clamp(min=eps)
    posterior_variance = (betas * (1 - alphas_cumprod_prev) / one_minus_alphas_cumprod).clamp(min=1e-20)

    return {
        "betas": betas,
        "alphas": alphas,
        "alphas_cumprod": alphas_cumprod,
        "alphas_cumprod_prev": alphas_cumprod_prev,
        "sqrt_alphas_cumprod": torch.sqrt(alphas_cumprod.clamp(min=eps)),
        "sqrt_one_minus_alphas_cumprod": torch.sqrt(one_minus_alphas_cumprod),
        "sqrt_alphas_cumprod_prev": torch.sqrt(alphas_cumprod_prev.clamp(min=eps)),
        "sqrt_one_minus_alphas_cumprod_prev": torch.sqrt((1 - alphas_cumprod_prev).clamp(min=eps)),
        "sqrt_recip_alphas": torch.sqrt((1.0 / alphas).clamp(min=eps)),
        "sqrt_alphas": torch.sqrt(alphas.clamp(min=eps)),  # 单步量 sqrt(α_t)
        "sqrt_beta": torch.sqrt((1.0 - alphas).clamp(min=eps)),  # 单步量 sqrt(β_t) = sqrt(1 - α_t)
        "posterior_variance": posterior_variance,
    }


# ===========================
# 2. 前向扩散 q(x_t | x_0)
# ===========================
def q_sample(x_start: torch.Tensor, t: torch.Tensor, diffusion_consts: Dict[str, torch.Tensor], 
             noise: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    前向扩散过程：从x_0采样x_t
    
    Args:
        x_start: 原始数据 [B, N*N*N, code_dim] (3维输入)
        t: 时间步 [B]
        diffusion_consts: 扩散常数字典
        noise: 可选的外部噪声
    
    Returns:
        x_t: 加噪后的数据 [B, N*N*N, code_dim]
    """
    if noise is None:
        noise = torch.randn_like(x_start)
    else:
        # 确保noise在正确的设备上
        noise = noise.to(x_start.device)
    
    sqrt_alphas_cumprod_t = extract(diffusion_consts["sqrt_alphas_cumprod"], t, x_start)
    sqrt_one_minus_alphas_cumprod_t = extract(diffusion_consts["sqrt_one_minus_alphas_cumprod"], t, x_start)
    
    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise


def extract(a: torch.Tensor, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    从时间序列a中取出对应t的值，并扩展维度匹配x
    
    Args:
        a: 时间序列张量 [T]
        t: 时间步索引 [B]
        x: 目标张量，用于确定输出形状 [B, ...]
    
    Returns:
        扩展后的时间步值 [B, 1, 1, ...] (匹配x的形状)
    
    Note:
        优化版本：假设所有张量已在正确设备上，移除设备检查以减少开销
    """
    # 直接索引（假设t和a在同一设备）
    out = a[t]
    
    # 扩展维度以匹配x的形状: [B] -> [B, 1, 1, ...]
    # 使用view更高效，避免不必要的reshape操作
    out = out.view((t.shape[0],) + (1,) * (x.dim() - 1))
    return out


def extract_batch(a: torch.Tensor, t_indices: torch.Tensor, batch_size: int, x_shape: Tuple[int, ...]) -> torch.Tensor:
    """
    批量从时间序列a中取出对应timestep的值，并扩展维度匹配x
    
    优化版本：一次性提取所有timestep的常数，避免循环中的重复调用
    
    Args:
        a: 时间序列张量 [T]
        t_indices: 时间步索引序列 [num_timesteps]，按采样顺序排列（从T-1到0）
        batch_size: batch大小
        x_shape: 目标张量形状，用于确定输出形状 (B, ...)
    
    Returns:
        扩展后的时间步值 [num_timesteps, B, 1, 1, ...]
    """
    # 批量索引：一次性获取所有timestep的值
    out = a[t_indices]  # [num_timesteps]
    
    # 扩展维度以匹配x的形状: [num_timesteps] -> [num_timesteps, B, 1, 1, ...]
    # 首先添加batch维度，然后添加空间维度
    out = out.unsqueeze(1)  # [num_timesteps, 1]
    out = out.expand(-1, batch_size)  # [num_timesteps, B]
    
    # 添加剩余的空间维度
    if len(x_shape) > 1:
        out = out.view((len(t_indices), batch_size) + (1,) * (len(x_shape) - 1))
    
    return out


# ===========================
# 3. 时间嵌入工具函数
# ===========================
def get_time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """
    获取时间步嵌入
    
    Args:
        t: 时间步 [B]
        dim: 嵌入维度
    
    Returns:
        时间嵌入 [B, dim]
    """
    device = t.device
    half = dim // 2
    
    # 数值稳定性：防止half_dim-1为0
    if half > 1:
        exponents = torch.arange(half, device=device, dtype=torch.float32) / (half - 1)
    else:
        exponents = torch.zeros(half, device=device, dtype=torch.float32)
    
    freqs = torch.exp(-math.log(10000.0) * exponents)
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    
    # 处理奇数维度：在末尾补0
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros(emb.shape[0], 1, device=device)], dim=-1)
    
    return emb


# ===========================
# 5. 反向采样
# ===========================
@torch.no_grad()
def p_sample(model: nn.Module, x_t: torch.Tensor, t: torch.Tensor, 
             diffusion_consts: Dict[str, torch.Tensor], clip_denoised: bool = False,
             precomputed_consts: Optional[Dict[str, torch.Tensor]] = None, timestep_idx: Optional[int] = None) -> torch.Tensor:
    """
    单步反向采样
    
    Args:
        model: 去噪模型
        x_t: 当前状态 [B, N*N*N, code_dim]
        t: 时间步 [B]
        diffusion_consts: 扩散常数
        clip_denoised: 是否将去噪后的结果裁剪到合理范围（用于数值稳定性）
        precomputed_consts: 预计算的常数字典（优化路径，可选）
        timestep_idx: 当前timestep在序列中的索引（用于从预计算常数中提取，可选）
    
    Returns:
        去噪后的状态 [B, N*N*N, code_dim]
    """
    # 优化路径：使用预计算的常数
    if precomputed_consts is not None and timestep_idx is not None:
        betas_t = precomputed_consts["betas"][timestep_idx]  # [B, 1, 1, ...]
        sqrt_recip_alphas_t = precomputed_consts["sqrt_recip_alphas"][timestep_idx]  # [B, 1, 1, ...]
        sqrt_one_minus_alphas_cumprod_t = precomputed_consts["sqrt_one_minus_alphas_cumprod"][timestep_idx]  # [B, 1, 1, ...]
        posterior_variance_t = precomputed_consts["posterior_variance"][timestep_idx]  # [B, 1, 1, ...]
    else:
        # 回退到原始路径（保持向后兼容性）
        betas_t = extract(diffusion_consts["betas"], t, x_t)
        sqrt_recip_alphas_t = extract(diffusion_consts["sqrt_recip_alphas"], t, x_t)
        sqrt_one_minus_alphas_cumprod_t = extract(diffusion_consts["sqrt_one_minus_alphas_cumprod"], t, x_t)
        posterior_variance_t = extract(diffusion_consts["posterior_variance"], t, x_t)

    # 预测噪声
    predicted_noise = model(x_t, t)
    
    # 数值稳定性：防止除零和数值溢出
    # 添加小的epsilon防止sqrt_one_minus_alphas_cumprod_t接近0时的不稳定
    eps = 1e-8
    sqrt_one_minus_alphas_cumprod_t_safe = torch.clamp(sqrt_one_minus_alphas_cumprod_t, min=eps)
    
    # 计算模型均值
    model_mean = sqrt_recip_alphas_t * (x_t - betas_t / sqrt_one_minus_alphas_cumprod_t_safe * predicted_noise)
    
    # 数值稳定性：裁剪model_mean到合理范围
    if clip_denoised:
        # 根据归一化后的codes范围裁剪（通常归一化后范围在[-3, 3]左右）
        model_mean = torch.clamp(model_mean, -3.0, 3.0)
    
    # 优化：使用预计算的is_last_timestep标志，避免条件判断
    # 注意：在采样循环中，timestep_idx=num_timesteps-1 对应最后一个timestep（t=0）
    is_last = (t[0] == 0) if timestep_idx is None else (t[0] == 0)
    
    if is_last:
        return model_mean
    else:
        noise = torch.randn_like(x_t)
        x_prev = model_mean + torch.sqrt(posterior_variance_t) * noise
        
        # 数值稳定性：最终结果也进行裁剪
        if clip_denoised:
            x_prev = torch.clamp(x_prev, -3.0, 3.0)
        
        return x_prev


@torch.no_grad()
def p_sample_loop(model: nn.Module, shape: Tuple[int, ...], diffusion_consts: Dict[str, torch.Tensor], 
                  device: torch.device, progress: bool = True, clip_denoised: bool = False) -> torch.Tensor:
    """
    完整的反向采样循环（预测噪声epsilon版本）
    
    优化版本：预计算所有timestep的常数，移除循环内的extract()调用和tqdm同步
    
    Args:
        model: 去噪模型，预测噪声
        shape: 输出形状 [B, N*N*N, code_dim]
        diffusion_consts: 扩散常数
        device: 设备
        progress: 是否显示进度条（已优化：不再使用tqdm以避免GPU-CPU同步，保留参数以保持向后兼容性）
        clip_denoised: 是否将去噪后的结果裁剪到合理范围（用于数值稳定性）
    
    Returns:
        生成的样本 [B, N*N*N, code_dim]
    """
    # progress 参数保留以保持向后兼容性，但不再使用以避免GPU-CPU同步
    _ = progress
    batch_size = shape[0]
    x_t = torch.randn(shape, device=device)
    num_timesteps = diffusion_consts["betas"].shape[0]
    
    # 预计算所有timestep的常数（从T-1到0）
    timestep_indices = torch.arange(num_timesteps - 1, -1, -1, device=device, dtype=torch.long)
    
    # 预计算所有需要的常数，形状: [num_timesteps, B, 1, 1, ...]
    precomputed_consts = {
        "betas": extract_batch(diffusion_consts["betas"], timestep_indices, batch_size, shape),
        "sqrt_recip_alphas": extract_batch(diffusion_consts["sqrt_recip_alphas"], timestep_indices, batch_size, shape),
        "sqrt_one_minus_alphas_cumprod": extract_batch(diffusion_consts["sqrt_one_minus_alphas_cumprod"], timestep_indices, batch_size, shape),
        "posterior_variance": extract_batch(diffusion_consts["posterior_variance"], timestep_indices, batch_size, shape),
    }
    
    # 优化循环：移除tqdm以避免GPU-CPU同步，直接使用预计算的常数
    for timestep_idx in range(num_timesteps):
        t = torch.full((batch_size,), timestep_indices[timestep_idx].item(), device=device, dtype=torch.long)
        x_t = p_sample(model, x_t, t, diffusion_consts, clip_denoised=clip_denoised,
                      precomputed_consts=precomputed_consts, timestep_idx=timestep_idx)
    
    return x_t


@torch.no_grad()
def p_sample_x0(model: nn.Module, x_t: torch.Tensor, t: torch.Tensor, 
                diffusion_consts: Dict[str, torch.Tensor], clip_denoised: bool = False,
                precomputed_consts: Optional[Dict[str, torch.Tensor]] = None, timestep_idx: Optional[int] = None) -> torch.Tensor:
    """
    单步反向采样（预测x0版本）
    
    Args:
        model: 去噪模型，预测x0
        x_t: 当前状态 [B, N*N*N, code_dim]
        t: 时间步 [B]
        diffusion_consts: 扩散常数
        clip_denoised: 是否将predicted_x0裁剪到合理范围（可选，用于数值稳定性）
        precomputed_consts: 预计算的常数字典（优化路径，可选）
        timestep_idx: 当前timestep在序列中的索引（用于从预计算常数中提取，可选）
    
    Returns:
        去噪后的状态 [B, N*N*N, code_dim]
    """
    # 预测x0
    predicted_x0 = model(x_t, t)
    
    # 可选的裁剪操作（用于数值稳定性）
    if clip_denoised:
        # 根据数据范围裁剪，这里假设数据已归一化到[-3, 3]或类似范围
        # 可以根据实际数据分布调整（归一化后的codes通常在[-3, 3]范围内）
        predicted_x0 = torch.clamp(predicted_x0, -3.0, 3.0)
    
    # 优化路径：使用预计算的常数
    if precomputed_consts is not None and timestep_idx is not None:
        betas_t = precomputed_consts["betas"][timestep_idx]  # [B, 1, 1, ...]
        alphas_cumprod_t = precomputed_consts["alphas_cumprod"][timestep_idx]  # [B, 1, 1, ...]
        alphas_cumprod_prev_t = precomputed_consts["alphas_cumprod_prev"][timestep_idx]  # [B, 1, 1, ...]
        sqrt_alphas_t = precomputed_consts["sqrt_alphas"][timestep_idx]  # [B, 1, 1, ...]
        sqrt_alphas_cumprod_prev_t = precomputed_consts["sqrt_alphas_cumprod_prev"][timestep_idx]  # [B, 1, 1, ...]
        posterior_variance_t = precomputed_consts["posterior_variance"][timestep_idx]  # [B, 1, 1, ...]
    else:
        # 回退到原始路径（保持向后兼容性）
        betas_t = extract(diffusion_consts["betas"], t, x_t)
        alphas_cumprod_t = extract(diffusion_consts["alphas_cumprod"], t, x_t)  # 累计量 α̅_t
        alphas_cumprod_prev_t = extract(diffusion_consts["alphas_cumprod_prev"], t, x_t)  # 累计量 α̅_{t-1}
        sqrt_alphas_t = extract(diffusion_consts["sqrt_alphas"], t, x_t)  # 单步量 sqrt(α_t)
        sqrt_alphas_cumprod_prev_t = extract(diffusion_consts["sqrt_alphas_cumprod_prev"], t, x_t)  # 累计量 sqrt(α̅_{t-1})
        posterior_variance_t = extract(diffusion_consts["posterior_variance"], t, x_t)
    
    # 优化：使用预计算的is_last_timestep标志，避免条件判断
    # 注意：在采样循环中，timestep_idx=num_timesteps-1 对应最后一个timestep（t=0）
    is_last = (t[0] == 0) if timestep_idx is None else (t[0] == 0)
    
    if is_last:
        # 最后一步，直接返回预测的x0
        return predicted_x0
    else:
        # 使用DDPM标准的采样公式计算x_{t-1}的均值
        # μ_t = (√(α_t) * (1 - α̅_{t-1}) / (1 - α̅_t)) * x_t + (√(α̅_{t-1}) * β_t / (1 - α̅_t)) * x̂_0
        eps = 1e-8
        one_minus_alphas_cumprod_t = (1.0 - alphas_cumprod_t).clamp(min=eps)
        one_minus_alphas_cumprod_prev_t = (1.0 - alphas_cumprod_prev_t).clamp(min=eps)
        
        # 计算model_mean：直接使用predicted_x0，不需要先计算predicted_noise
        model_mean = (
            sqrt_alphas_t * one_minus_alphas_cumprod_prev_t / one_minus_alphas_cumprod_t * x_t
            + sqrt_alphas_cumprod_prev_t * betas_t / one_minus_alphas_cumprod_t * predicted_x0
        )
        
        # 添加噪声（使用后验方差）
        noise = torch.randn_like(x_t)
        x_prev = model_mean + torch.sqrt(posterior_variance_t) * noise
        
        # 数值稳定性：最终结果也进行裁剪
        if clip_denoised:
            x_prev = torch.clamp(x_prev, -3.0, 3.0)
        
        return x_prev


@torch.no_grad()
def p_sample_loop_x0(model: nn.Module, shape: Tuple[int, ...], diffusion_consts: Dict[str, torch.Tensor], 
                     device: torch.device, progress: bool = True, clip_denoised: bool = False) -> torch.Tensor:
    """
    完整的反向采样循环（预测x0版本）
    
    优化版本：预计算所有timestep的常数，移除循环内的extract()调用和tqdm同步
    
    Args:
        model: 去噪模型，预测x0
        shape: 输出形状 [B, N*N*N, code_dim]
        diffusion_consts: 扩散常数
        device: 设备
        progress: 是否显示进度条（已优化：不再使用tqdm以避免GPU-CPU同步，保留参数以保持向后兼容性）
        clip_denoised: 是否将predicted_x0裁剪到合理范围（用于数值稳定性，默认False）
    
    Returns:
        生成的样本 [B, N*N*N, code_dim]
    """
    # progress 参数保留以保持向后兼容性，但不再使用以避免GPU-CPU同步
    _ = progress
    batch_size = shape[0]
    x_t = torch.randn(shape, device=device)
    num_timesteps = diffusion_consts["betas"].shape[0]
    
    # 预计算所有timestep的常数（从T-1到0）
    timestep_indices = torch.arange(num_timesteps - 1, -1, -1, device=device, dtype=torch.long)
    
    # 预计算所有需要的常数，形状: [num_timesteps, B, 1, 1, ...]
    precomputed_consts = {
        "betas": extract_batch(diffusion_consts["betas"], timestep_indices, batch_size, shape),
        "alphas_cumprod": extract_batch(diffusion_consts["alphas_cumprod"], timestep_indices, batch_size, shape),
        "alphas_cumprod_prev": extract_batch(diffusion_consts["alphas_cumprod_prev"], timestep_indices, batch_size, shape),
        "sqrt_alphas": extract_batch(diffusion_consts["sqrt_alphas"], timestep_indices, batch_size, shape),
        "sqrt_alphas_cumprod_prev": extract_batch(diffusion_consts["sqrt_alphas_cumprod_prev"], timestep_indices, batch_size, shape),
        "posterior_variance": extract_batch(diffusion_consts["posterior_variance"], timestep_indices, batch_size, shape),
    }
    
    # 优化循环：移除tqdm以避免GPU-CPU同步，直接使用预计算的常数
    for timestep_idx in range(num_timesteps):
        t = torch.full((batch_size,), timestep_indices[timestep_idx].item(), device=device, dtype=torch.long)
        x_t = p_sample_x0(model, x_t, t, diffusion_consts, clip_denoised=clip_denoised,
                         precomputed_consts=precomputed_consts, timestep_idx=timestep_idx)
    
    return x_t


# ===========================
# 6. DDPM 训练损失计算
# ===========================
def compute_ddpm_loss(model: nn.Module, x_0: torch.Tensor, diffusion_consts: Dict[str, torch.Tensor], 
                     device: torch.device, position_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    计算DDPM训练损失（预测噪声epsilon版本）
    
    Args:
        model: 去噪模型
        x_0: 原始数据 [B, N*N*N, code_dim]
        diffusion_consts: 扩散常数
        device: 设备
        position_weights: 位置权重 [B, N*N*N]，可选
    
    Returns:
        训练损失
    """
    batch_size = x_0.shape[0]
    
    # 随机采样时间步
    t = torch.randint(0, diffusion_consts["betas"].shape[0], (batch_size,), device=device).long()
    
    # 生成噪声
    noise = torch.randn_like(x_0)
    
    # 前向扩散
    x_t = q_sample(x_0, t, diffusion_consts, noise)
    
    # 预测噪声
    predicted_noise = model(x_t, t)
    
    # 计算损失
    if position_weights is not None:
        # 应用位置权重：对每个位置计算MSE，然后加权平均
        # predicted_noise: [B, N*N*N, code_dim], noise: [B, N*N*N, code_dim]
        # position_weights: [B, N*N*N]
        squared_diff = (predicted_noise - noise) ** 2  # [B, N*N*N, code_dim]
        squared_diff_per_pos = squared_diff.mean(dim=-1)  # [B, N*N*N]
        # 应用权重并取平均
        loss = (position_weights * squared_diff_per_pos).mean()
    else:
        # 不使用位置权重，直接计算MSE
        loss = F.mse_loss(predicted_noise, noise)
    
    return loss


def compute_ddpm_loss_x0(model: nn.Module, x_0: torch.Tensor, diffusion_consts: Dict[str, torch.Tensor], 
                         device: torch.device, use_time_weight: bool = True, 
                         position_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    计算DDPM训练损失（预测x0版本）
    
    Args:
        model: 去噪模型，预测x0
        x_0: 原始数据 [B, N*N*N, code_dim]
        diffusion_consts: 扩散常数
        device: 设备
        use_time_weight: 是否使用时间步权重 1 + num_timesteps/(t+1)
        position_weights: 位置权重 [B, N*N*N]，可选
    
    Returns:
        训练损失
    """
    batch_size = x_0.shape[0]
    num_timesteps = diffusion_consts["betas"].shape[0]
    
    # 随机采样时间步
    t = torch.randint(0, num_timesteps, (batch_size,), device=device).long()
    
    # 生成噪声
    noise = torch.randn_like(x_0)
    
    # 前向扩散
    x_t = q_sample(x_0, t, diffusion_consts, noise)
    
    # 预测x0
    predicted_x0 = model(x_t, t)
    
    # 计算每个位置的损失
    if position_weights is not None:
        # 应用位置权重：对每个位置计算MSE
        # predicted_x0: [B, N*N*N, code_dim], x_0: [B, N*N*N, code_dim]
        # position_weights: [B, N*N*N]
        squared_diff = (predicted_x0 - x_0) ** 2  # [B, N*N*N, code_dim]
        squared_diff_per_pos = squared_diff.mean(dim=-1)  # [B, N*N*N]
        # 应用位置权重
        weighted_loss_per_pos = position_weights * squared_diff_per_pos  # [B, N*N*N]
        # 计算每个样本的损失 [B]
        loss_per_sample = weighted_loss_per_pos.mean(dim=1)  # [B]
    else:
        # 不使用位置权重，直接计算每个样本的损失 [B]
        loss_per_sample = F.mse_loss(predicted_x0, x_0, reduction='none').mean(dim=tuple(range(1, predicted_x0.ndim)))
    
    if use_time_weight:
        # 计算时间步权重：1 + num_timesteps / (t + 1)
        # t越小，权重越大（t=0时权重最大，t=num_timesteps-1时权重最小）
        weights = 1.0 + num_timesteps / (t.float() + 1.0)  # [B]
        # 应用权重并取平均
        loss = (loss_per_sample * weights).mean()
    else:
        # 不使用权重，直接取平均
        loss = loss_per_sample.mean()
    
    # 数值检查
    if torch.isnan(loss) or torch.isinf(loss):
        print("[WARNING] Loss is NaN or Inf in compute_ddpm_loss_x0")
        print(f"  predicted_x0: min={predicted_x0.min().item():.6f}, max={predicted_x0.max().item():.6f}, mean={predicted_x0.mean().item():.6f}")
        print(f"  x_0: min={x_0.min().item():.6f}, max={x_0.max().item():.6f}, mean={x_0.mean().item():.6f}")
        if use_time_weight:
            weights = 1.0 + num_timesteps / (t.float() + 1.0)
            print(f"  weights: min={weights.min().item():.6f}, max={weights.max().item():.6f}, mean={weights.mean().item():.6f}")
        if position_weights is not None:
            print(f"  position_weights: min={position_weights.min().item():.6f}, max={position_weights.max().item():.6f}, mean={position_weights.mean().item():.6f}")
        # 返回一个小的非零值以避免训练崩溃
        loss = torch.tensor(1e-6, device=device, requires_grad=True)
    
    return loss


# ===========================
# 7. 工具函数
# ===========================
def sigmoid(x):
    return 1 / (np.exp(-x) + 1)

def create_diffusion_constants(config: dict, device: Optional[torch.device] = None) -> Dict[str, torch.Tensor]:
    """
    创建扩散常数
    
    Args:
        config: 配置字典
        device: 目标设备（如果提供，所有常数将放在该设备上）
    
    Returns:
        扩散常数字典
    """
    ddpm_config = config.get("ddpm", {})
    betas = get_beta_schedule(
        beta_start=ddpm_config.get("beta_start", 1e-4),
        beta_end=ddpm_config.get("beta_end", 0.02),
        num_timesteps=ddpm_config.get("num_timesteps", 1000),
        schedule=ddpm_config.get("schedule", "linear"),
        w=ddpm_config.get("w", 6.0),
        s=ddpm_config.get("s", 0.008),  # cosine调度的偏移参数
    )
    return prepare_diffusion_constants(betas, device=device)
