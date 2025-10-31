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
        s = 0.008
        steps = num_timesteps + 1
        x = np.linspace(0, steps, steps)
        alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        # betas = np.clip(betas, 0, 0.999)
    elif schedule == "sigmoid":
        # Use w parameter for sigmoid steepness
        betas = np.linspace(-w, w, num_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
        # betas = np.clip(betas, 0, 0.999)
    else:
        raise NotImplementedError(f"Schedule {schedule} not implemented")
    return torch.tensor(betas, dtype=torch.float32)


def prepare_diffusion_constants(betas: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    准备扩散过程的常数
    
    Args:
        betas: β序列
    
    Returns:
        包含所有扩散常数的字典
    """
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]], dim=0)

    return {
        "betas": betas,
        "alphas": alphas,
        "alphas_cumprod": alphas_cumprod,
        "alphas_cumprod_prev": alphas_cumprod_prev,
        "sqrt_alphas_cumprod": torch.sqrt(alphas_cumprod),
        "sqrt_one_minus_alphas_cumprod": torch.sqrt(1 - alphas_cumprod),
        "sqrt_alphas_cumprod_prev": torch.sqrt(alphas_cumprod_prev),
        "sqrt_one_minus_alphas_cumprod_prev": torch.sqrt(1 - alphas_cumprod_prev),
        "sqrt_recip_alphas": torch.sqrt(1.0 / alphas),
        "posterior_variance": betas * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod),
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
    
    sqrt_alphas_cumprod_t = extract(diffusion_consts["sqrt_alphas_cumprod"], t, x_start)
    sqrt_one_minus_alphas_cumprod_t = extract(diffusion_consts["sqrt_one_minus_alphas_cumprod"], t, x_start)
    
    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise


def extract(a: torch.Tensor, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    从时间序列a中取出对应t的值，并扩展维度匹配x
    
    Args:
        a: 时间序列张量
        t: 时间步索引
        x: 目标张量，用于确定输出形状
    
    Returns:
        扩展后的时间步值
    """
    t = t.to(a.device)
    out = a[t].to(x.device)
    while len(out.shape) < len(x.shape):
        out = out.unsqueeze(-1)
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
    half_dim = dim // 2
    embeddings = math.log(10000) / (half_dim - 1)
    embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
    embeddings = t[:, None] * embeddings[None, :]
    embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
    return embeddings


# ===========================
# 5. 反向采样
# ===========================
@torch.no_grad()
def p_sample(model: nn.Module, x_t: torch.Tensor, t: torch.Tensor, 
             diffusion_consts: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    单步反向采样
    
    Args:
        model: 去噪模型
        x_t: 当前状态 [B, N*N*N, code_dim]
        t: 时间步 [B]
        diffusion_consts: 扩散常数
    
    Returns:
        去噪后的状态 [B, N*N*N, code_dim]
    """
    betas_t = extract(diffusion_consts["betas"], t, x_t)
    sqrt_recip_alphas_t = extract(diffusion_consts["sqrt_recip_alphas"], t, x_t)
    sqrt_one_minus_alphas_cumprod_t = extract(diffusion_consts["sqrt_one_minus_alphas_cumprod"], t, x_t)

    # 预测噪声
    predicted_noise = model(x_t, t)
    
    # 计算模型均值
    model_mean = sqrt_recip_alphas_t * (x_t - betas_t / sqrt_one_minus_alphas_cumprod_t * predicted_noise)
    posterior_variance_t = extract(diffusion_consts["posterior_variance"], t, x_t)

    if t[0] == 0:
        return model_mean
    else:
        noise = torch.randn_like(x_t)
        return model_mean + torch.sqrt(posterior_variance_t) * noise


@torch.no_grad()
def p_sample_loop(model: nn.Module, shape: Tuple[int, ...], diffusion_consts: Dict[str, torch.Tensor], 
                  device: torch.device, progress: bool = True) -> torch.Tensor:
    """
    完整的反向采样循环（预测噪声epsilon版本）
    
    Args:
        model: 去噪模型，预测噪声
        shape: 输出形状 [B, N*N*N, code_dim]
        diffusion_consts: 扩散常数
        device: 设备
        progress: 是否显示进度条
    
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
        x_t = p_sample(model, x_t, t, diffusion_consts)
    
    return x_t


@torch.no_grad()
def p_sample_x0(model: nn.Module, x_t: torch.Tensor, t: torch.Tensor, 
                diffusion_consts: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    单步反向采样（预测x0版本）
    
    Args:
        model: 去噪模型，预测x0
        x_t: 当前状态 [B, N*N*N, code_dim]
        t: 时间步 [B]
        diffusion_consts: 扩散常数
    
    Returns:
        去噪后的状态 [B, N*N*N, code_dim]
    """
    # 预测x0
    predicted_x0 = model(x_t, t)
    
    # 获取需要的常数
    sqrt_alphas_cumprod_prev_t = extract(diffusion_consts["sqrt_alphas_cumprod_prev"], t, x_t)
    sqrt_one_minus_alphas_cumprod_prev_t = extract(diffusion_consts["sqrt_one_minus_alphas_cumprod_prev"], t, x_t)
    
    if t[0] == 0:
        # 最后一步，直接返回预测的x0
        return predicted_x0
    else:
        # 从预测的x0计算x_{t-1}
        noise = torch.randn_like(x_t)
        x_prev = sqrt_alphas_cumprod_prev_t * predicted_x0 + sqrt_one_minus_alphas_cumprod_prev_t * noise
        return x_prev


@torch.no_grad()
def p_sample_loop_x0(model: nn.Module, shape: Tuple[int, ...], diffusion_consts: Dict[str, torch.Tensor], 
                     device: torch.device, progress: bool = True) -> torch.Tensor:
    """
    完整的反向采样循环（预测x0版本）
    
    Args:
        model: 去噪模型，预测x0
        shape: 输出形状 [B, N*N*N, code_dim]
        diffusion_consts: 扩散常数
        device: 设备
        progress: 是否显示进度条
    
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
        x_t = p_sample_x0(model, x_t, t, diffusion_consts)
    
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
    loss = F.mse_loss(predicted_x0, x_0)
    
    return loss


# ===========================
# 7. 工具函数
# ===========================
def sigmoid(x):
    return 1 / (np.exp(-x) + 1)

def create_diffusion_constants(config: dict) -> Dict[str, torch.Tensor]:
    """
    创建扩散常数
    
    Args:
        config: 配置字典
    
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
    return prepare_diffusion_constants(betas)
