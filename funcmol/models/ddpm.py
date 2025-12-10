import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from tqdm import tqdm
from typing import Dict, Optional, Tuple


# ===========================
# 1. β 调度与常数
# ===========================
def get_beta_schedule(beta_start: float, beta_end: float, num_timesteps: int, 
                    schedule: str = "linear", w: float = 6.0) -> torch.Tensor:
    """
    获取β调度序列
    
    Args:
        beta_start: 起始β值
        beta_end: 结束β值
        num_timesteps: 时间步数
        schedule: 调度类型 ("linear", "cosine", 或 "sigmoid")
        w: sigmoid调度的宽度参数（仅用于sigmoid调度）
    
    Returns:
        betas: β序列张量
    """
    if schedule == "linear":
        betas = np.linspace(beta_start, beta_end, num_timesteps, dtype=np.float64)
    elif schedule == "cosine":
        # 修复：使用正确的边界，避免alphas_cumprod[-1]=0的问题
        s = 0.008
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
    """
    # 确保t在a的设备上，然后索引
    if t.device != a.device:
        t = t.to(a.device)
    out = a[t]
    
    # 移动到x的设备
    if out.device != x.device:
        out = out.to(x.device)
    
    # 扩展维度以匹配x的形状: [B] -> [B, 1, 1, ...]
    # 使用view更高效
    out = out.view((t.shape[0],) + (1,) * (x.dim() - 1))
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
             diffusion_consts: Dict[str, torch.Tensor], clip_denoised: bool = False) -> torch.Tensor:
    """
    单步反向采样
    
    Args:
        model: 去噪模型
        x_t: 当前状态 [B, N*N*N, code_dim]
        t: 时间步 [B]
        diffusion_consts: 扩散常数
        clip_denoised: 是否将去噪后的结果裁剪到合理范围（用于数值稳定性）
    
    Returns:
        去噪后的状态 [B, N*N*N, code_dim]
    """
    betas_t = extract(diffusion_consts["betas"], t, x_t)
    sqrt_recip_alphas_t = extract(diffusion_consts["sqrt_recip_alphas"], t, x_t)
    sqrt_one_minus_alphas_cumprod_t = extract(diffusion_consts["sqrt_one_minus_alphas_cumprod"], t, x_t)

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
    
    posterior_variance_t = extract(diffusion_consts["posterior_variance"], t, x_t)

    if t[0] == 0:
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
    
    Args:
        model: 去噪模型，预测噪声
        shape: 输出形状 [B, N*N*N, code_dim]
        diffusion_consts: 扩散常数
        device: 设备
        progress: 是否显示进度条
        clip_denoised: 是否将去噪后的结果裁剪到合理范围（用于数值稳定性）
    
    Returns:
        生成的样本 [B, N*N*N, code_dim]
    """
    x_t = torch.randn(shape, device=device)
    num_timesteps = diffusion_consts["betas"].shape[0]
    
    iterator = reversed(range(num_timesteps))
    if progress:
        iterator = tqdm(iterator, desc="DDPM Sampling")
    
    for i in iterator:
        t = torch.full((shape[0],), i, device=device, dtype=torch.long)
        x_t = p_sample(model, x_t, t, diffusion_consts, clip_denoised=clip_denoised)
    
    return x_t


@torch.no_grad()
def p_sample_x0(model: nn.Module, x_t: torch.Tensor, t: torch.Tensor, 
                diffusion_consts: Dict[str, torch.Tensor], clip_denoised: bool = False) -> torch.Tensor:
    """
    单步反向采样（预测x0版本）
    
    Args:
        model: 去噪模型，预测x0
        x_t: 当前状态 [B, N*N*N, code_dim]
        t: 时间步 [B]
        diffusion_consts: 扩散常数
        clip_denoised: 是否将predicted_x0裁剪到合理范围（可选，用于数值稳定性）
    
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
    
    # 获取需要的常数
    betas_t = extract(diffusion_consts["betas"], t, x_t)
    alphas_cumprod_t = extract(diffusion_consts["alphas_cumprod"], t, x_t)  # 累计量 α̅_t
    alphas_cumprod_prev_t = extract(diffusion_consts["alphas_cumprod_prev"], t, x_t)  # 累计量 α̅_{t-1}
    sqrt_alphas_t = extract(diffusion_consts["sqrt_alphas"], t, x_t)  # 单步量 sqrt(α_t)
    sqrt_alphas_cumprod_prev_t = extract(diffusion_consts["sqrt_alphas_cumprod_prev"], t, x_t)  # 累计量 sqrt(α̅_{t-1})
    posterior_variance_t = extract(diffusion_consts["posterior_variance"], t, x_t)
    
    if t[0] == 0:
        # 最后一步，直接返回预测的x0
        return predicted_x0
    else:
        # 使用DDPM标准的采样公式计算x_{t-1}的均值
        # μ_t = (√(α̅_{t-1}) * β_t / (1 - α̅_t)) * x_t + (√(α_t) * (1 - α̅_{t-1}) / (1 - α̅_t)) * x̂_0
        eps = 1e-8
        one_minus_alphas_cumprod_t = (1.0 - alphas_cumprod_t).clamp(min=eps)
        one_minus_alphas_cumprod_prev_t = (1.0 - alphas_cumprod_prev_t).clamp(min=eps)
        
        # 计算model_mean：直接使用predicted_x0，不需要先计算predicted_noise
        model_mean = (
            sqrt_alphas_cumprod_prev_t * betas_t / one_minus_alphas_cumprod_t * x_t
            + sqrt_alphas_t * one_minus_alphas_cumprod_prev_t / one_minus_alphas_cumprod_t * predicted_x0
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
    
    Args:
        model: 去噪模型，预测x0
        shape: 输出形状 [B, N*N*N, code_dim]
        diffusion_consts: 扩散常数
        device: 设备
        progress: 是否显示进度条
        clip_denoised: 是否将predicted_x0裁剪到合理范围（用于数值稳定性，默认False）
    
    Returns:
        生成的样本 [B, N*N*N, code_dim]
    """
    x_t = torch.randn(shape, device=device)
    num_timesteps = diffusion_consts["betas"].shape[0]
    
    iterator = reversed(range(num_timesteps))
    if progress:
        iterator = tqdm(iterator, desc="DDPM Sampling (x0)")
    
    for i in iterator:
        t = torch.full((shape[0],), i, device=device, dtype=torch.long)
        x_t = p_sample_x0(model, x_t, t, diffusion_consts, clip_denoised=clip_denoised)
    
    return x_t


# ===========================
# 6. DDPM 训练损失计算
# ===========================
def compute_ddpm_loss(model: nn.Module, x_0: torch.Tensor, diffusion_consts: Dict[str, torch.Tensor], 
                     device: torch.device) -> torch.Tensor:
    """
    计算DDPM训练损失（预测噪声epsilon版本）
    
    Args:
        model: 去噪模型
        x_0: 原始数据 [B, N*N*N, code_dim]
        diffusion_consts: 扩散常数
        device: 设备
    
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
    loss = F.mse_loss(predicted_noise, noise)
    
    return loss


def compute_ddpm_loss_x0(model: nn.Module, x_0: torch.Tensor, diffusion_consts: Dict[str, torch.Tensor], 
                         device: torch.device) -> torch.Tensor:
    """
    计算DDPM训练损失（预测x0版本）
    
    Args:
        model: 去噪模型，预测x0
        x_0: 原始数据 [B, N*N*N, code_dim]
        diffusion_consts: 扩散常数
        device: 设备
    
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
    
    # 预测x0
    predicted_x0 = model(x_t, t)
    
    # 计算损失：直接预测x0
    # 注意：对于预测x0的版本，通常不需要时间步权重，因为预测x0比预测噪声更稳定
    loss = F.mse_loss(predicted_x0, x_0, reduction='mean')
    
    # 数值稳定性检查：如果loss过大，可能是数值问题
    if torch.isnan(loss) or torch.isinf(loss):
        print("[WARNING] Loss is NaN or Inf in compute_ddpm_loss_x0")
        print(f"  predicted_x0: min={predicted_x0.min().item():.6f}, max={predicted_x0.max().item():.6f}, mean={predicted_x0.mean().item():.6f}")
        print(f"  x_0: min={x_0.min().item():.6f}, max={x_0.max().item():.6f}, mean={x_0.mean().item():.6f}")
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
    )
    return prepare_diffusion_constants(betas, device=device)
