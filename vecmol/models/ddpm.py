import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, Optional, Tuple


# ===========================
# 1. Beta schedule and constants
# ===========================
def get_beta_schedule(beta_start: float, beta_end: float, num_timesteps: int,
                    schedule: str = "linear", s: float = 0.008) -> torch.Tensor:
    """
    Get beta schedule sequence.

    Args:
        beta_start: Start beta value
        beta_end: End beta value
        num_timesteps: Number of timesteps
        schedule: Schedule type ("linear" or "cosine")
        s: Cosine schedule offset (cosine only), default 0.008

    Returns:
        betas: Beta sequence tensor
    """
    if schedule == "linear":
        betas = np.linspace(beta_start, beta_end, num_timesteps, dtype=np.float64)
    elif schedule == "cosine":
        timesteps = num_timesteps
        x = np.linspace(0, timesteps, timesteps + 1)
        alphas_cumprod = np.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    else:
        raise NotImplementedError(f"Schedule {schedule} not implemented. Use 'linear' or 'cosine'.")
    
    # Numerical stability: clamp betas
    betas = np.clip(betas, 1e-8, 0.999)
    return torch.tensor(betas, dtype=torch.float32)


def prepare_diffusion_constants(betas: torch.Tensor, device: Optional[torch.device] = None) -> Dict[str, torch.Tensor]:
    """
    Prepare diffusion constants.
    
    Args:
        betas: Beta sequence
        device: Target device (if provided, all constants will be on this device)
    
    Returns:
        Dict of diffusion constants
    """
    if device is not None:
        betas = betas.to(device)
    
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = torch.cat([
        torch.tensor([1.0], device=betas.device, dtype=betas.dtype), 
        alphas_cumprod[:-1]
    ], dim=0)

    # Numerical stability: eps to avoid div by 0 and sqrt(0)
    eps = 1e-20
    
    # Avoid div by 0 when computing posterior_variance
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
        "sqrt_alphas": torch.sqrt(alphas.clamp(min=eps)),
        "sqrt_beta": torch.sqrt((1.0 - alphas).clamp(min=eps)),
        "posterior_variance": posterior_variance,
    }


# ===========================
# 2. Forward diffusion q(x_t | x_0)
# ===========================
def q_sample(x_start: torch.Tensor, t: torch.Tensor, diffusion_consts: Dict[str, torch.Tensor], 
             noise: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Forward diffusion: sample x_t from x_0.
    
    Args:
        x_start: Raw data [B, N*N*N, code_dim]
        t: Timestep [B]
        diffusion_consts: Diffusion constants dict
        noise: Optional external noise
    
    Returns:
        x_t: Noisy data [B, N*N*N, code_dim]
    """
    if noise is None:
        noise = torch.randn_like(x_start)
    else:
        noise = noise.to(x_start.device)
    
    sqrt_alphas_cumprod_t = extract(diffusion_consts["sqrt_alphas_cumprod"], t, x_start)
    sqrt_one_minus_alphas_cumprod_t = extract(diffusion_consts["sqrt_one_minus_alphas_cumprod"], t, x_start)
    
    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise


def extract(a: torch.Tensor, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Extract value at t from time series a and expand dims to match x.
    
    Args:
        a: Time series tensor [T]
        t: Timestep indices [B]
        x: Target tensor for output shape [B, ...]
    
    Returns:
        Expanded timestep values [B, 1, 1, ...]
    """
    out = a[t]
    
    # Expand dims to match x: [B] -> [B, 1, 1, ...]
    out = out.view((t.shape[0],) + (1,) * (x.dim() - 1))
    return out


def extract_batch(a: torch.Tensor, t_indices: torch.Tensor, batch_size: int, x_shape: Tuple[int, ...]) -> torch.Tensor:
    """
    Batch-extract timestep values from time series a and expand to match x.
    
    Args:
        a: Time series tensor [T]
        t_indices: Timestep indices [num_timesteps], sampling order (T-1 to 0)
        batch_size: Batch size
        x_shape: Target shape (B, ...)
    
    Returns:
        Expanded values [num_timesteps, B, 1, 1, ...]
    """
    out = a[t_indices]
    out = out.unsqueeze(1)
    out = out.expand(-1, batch_size)
    
    # Add remaining spatial dims
    if len(x_shape) > 1:
        out = out.view((len(t_indices), batch_size) + (1,) * (len(x_shape) - 1))
    
    return out


# ===========================
# 3. Time embedding
# ===========================
def get_time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Get timestep embedding.
    
    Args:
        t: Timestep [B]
        dim: Embedding dimension
    
    Returns:
        Time embedding [B, dim]
    """
    device = t.device
    half = dim // 2
    
    # Numerical stability: avoid half-1 == 0
    if half > 1:
        exponents = torch.arange(half, device=device, dtype=torch.float32) / (half - 1)
    else:
        exponents = torch.zeros(half, device=device, dtype=torch.float32)
    
    freqs = torch.exp(-math.log(10000.0) * exponents)
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    
    # Odd dim: pad with 0
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros(emb.shape[0], 1, device=device)], dim=-1)
    
    return emb


# ===========================
# 5. Reverse sampling
# ===========================
@torch.no_grad()
def p_sample(model: nn.Module, x_t: torch.Tensor, t: torch.Tensor, 
             diffusion_consts: Dict[str, torch.Tensor], clip_denoised: bool = False,
             precomputed_consts: Optional[Dict[str, torch.Tensor]] = None, timestep_idx: Optional[int] = None) -> torch.Tensor:
    """
    Single reverse sampling step.
    
    Args:
        model: Denoising model
        x_t: Current state [B, N*N*N, code_dim]
        t: Timestep [B]
        diffusion_consts: Diffusion constants
        clip_denoised: Whether to clip denoised output for numerical stability
        precomputed_consts: Precomputed constants (optional)
        timestep_idx: Index of current timestep in sequence (optional)
    
    Returns:
        Denoised state [B, N*N*N, code_dim]
    """
    if precomputed_consts is not None and timestep_idx is not None:
        betas_t = precomputed_consts["betas"][timestep_idx]  # [B, 1, 1, ...]
        sqrt_recip_alphas_t = precomputed_consts["sqrt_recip_alphas"][timestep_idx]  # [B, 1, 1, ...]
        sqrt_one_minus_alphas_cumprod_t = precomputed_consts["sqrt_one_minus_alphas_cumprod"][timestep_idx]  # [B, 1, 1, ...]
        posterior_variance_t = precomputed_consts["posterior_variance"][timestep_idx]  # [B, 1, 1, ...]
    else:
        # Fallback path (backward compatibility)
        betas_t = extract(diffusion_consts["betas"], t, x_t)
        sqrt_recip_alphas_t = extract(diffusion_consts["sqrt_recip_alphas"], t, x_t)
        sqrt_one_minus_alphas_cumprod_t = extract(diffusion_consts["sqrt_one_minus_alphas_cumprod"], t, x_t)
        posterior_variance_t = extract(diffusion_consts["posterior_variance"], t, x_t)

    predicted_noise = model(x_t, t)
    
    # Numerical stability: avoid div by zero
    eps = 1e-8
    sqrt_one_minus_alphas_cumprod_t_safe = torch.clamp(sqrt_one_minus_alphas_cumprod_t, min=eps)
    
    model_mean = sqrt_recip_alphas_t * (x_t - betas_t / sqrt_one_minus_alphas_cumprod_t_safe * predicted_noise)
    
    if clip_denoised:
        model_mean = torch.clamp(model_mean, -3.0, 3.0)
    
    # is_last: timestep_idx=num_timesteps-1 corresponds to t=0
    is_last = (t[0] == 0) if timestep_idx is None else (t[0] == 0)
    
    if is_last:
        return model_mean
    else:
        noise = torch.randn_like(x_t)
        x_prev = model_mean + torch.sqrt(posterior_variance_t) * noise
        
        if clip_denoised:
            x_prev = torch.clamp(x_prev, -3.0, 3.0)
        
        return x_prev


@torch.no_grad()
def p_sample_loop(model: nn.Module, shape: Tuple[int, ...], diffusion_consts: Dict[str, torch.Tensor], 
                  device: torch.device, progress: bool = True, clip_denoised: bool = False) -> torch.Tensor:
    """
    Full reverse sampling loop (predict noise epsilon).
    
    Args:
        model: Denoising model (predicts noise)
        shape: Output shape [B, N*N*N, code_dim]
        diffusion_consts: Diffusion constants
        device: Device
        progress: Kept for backward compatibility (not used)
        clip_denoised: Whether to clip denoised output
    
    Returns:
        Generated samples [B, N*N*N, code_dim]
    """
    _ = progress
    batch_size = shape[0]
    x_t = torch.randn(shape, device=device)
    num_timesteps = diffusion_consts["betas"].shape[0]
    
    timestep_indices = torch.arange(num_timesteps - 1, -1, -1, device=device, dtype=torch.long)
    
    precomputed_consts = {
        "betas": extract_batch(diffusion_consts["betas"], timestep_indices, batch_size, shape),
        "sqrt_recip_alphas": extract_batch(diffusion_consts["sqrt_recip_alphas"], timestep_indices, batch_size, shape),
        "sqrt_one_minus_alphas_cumprod": extract_batch(diffusion_consts["sqrt_one_minus_alphas_cumprod"], timestep_indices, batch_size, shape),
        "posterior_variance": extract_batch(diffusion_consts["posterior_variance"], timestep_indices, batch_size, shape),
    }
    
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
    Single reverse sampling step (predict x0).
    
    Args:
        model: Denoising model (predicts x0)
        x_t: Current state [B, N*N*N, code_dim]
        t: Timestep [B]
        diffusion_consts: Diffusion constants
        clip_denoised: Whether to clip predicted_x0
        precomputed_consts: Precomputed constants (optional)
        timestep_idx: Current timestep index (optional)
    
    Returns:
        Denoised state [B, N*N*N, code_dim]
    """
    predicted_x0 = model(x_t, t)
    
    if clip_denoised:
        predicted_x0 = torch.clamp(predicted_x0, -3.0, 3.0)
    
    if precomputed_consts is not None and timestep_idx is not None:
    if precomputed_consts is not None and timestep_idx is not None:
        betas_t = precomputed_consts["betas"][timestep_idx]  # [B, 1, 1, ...]
        alphas_cumprod_t = precomputed_consts["alphas_cumprod"][timestep_idx]  # [B, 1, 1, ...]
        alphas_cumprod_prev_t = precomputed_consts["alphas_cumprod_prev"][timestep_idx]  # [B, 1, 1, ...]
        sqrt_alphas_t = precomputed_consts["sqrt_alphas"][timestep_idx]  # [B, 1, 1, ...]
        sqrt_alphas_cumprod_prev_t = precomputed_consts["sqrt_alphas_cumprod_prev"][timestep_idx]  # [B, 1, 1, ...]
        posterior_variance_t = precomputed_consts["posterior_variance"][timestep_idx]  # [B, 1, 1, ...]
    else:
        betas_t = extract(diffusion_consts["betas"], t, x_t)
        alphas_cumprod_t = extract(diffusion_consts["alphas_cumprod"], t, x_t)
        alphas_cumprod_prev_t = extract(diffusion_consts["alphas_cumprod_prev"], t, x_t)
        sqrt_alphas_t = extract(diffusion_consts["sqrt_alphas"], t, x_t)
        sqrt_alphas_cumprod_prev_t = extract(diffusion_consts["sqrt_alphas_cumprod_prev"], t, x_t)
        posterior_variance_t = extract(diffusion_consts["posterior_variance"], t, x_t)
    
    is_last = (t[0] == 0) if timestep_idx is None else (t[0] == 0)
    
    if is_last:
        return predicted_x0
    else:
        # DDPM sampling formula for x_{t-1} mean
        eps = 1e-8
        one_minus_alphas_cumprod_t = (1.0 - alphas_cumprod_t).clamp(min=eps)
        one_minus_alphas_cumprod_prev_t = (1.0 - alphas_cumprod_prev_t).clamp(min=eps)
        
        model_mean = (
            sqrt_alphas_t * one_minus_alphas_cumprod_prev_t / one_minus_alphas_cumprod_t * x_t
            + sqrt_alphas_cumprod_prev_t * betas_t / one_minus_alphas_cumprod_t * predicted_x0
        )
        
        noise = torch.randn_like(x_t)
        x_prev = model_mean + torch.sqrt(posterior_variance_t) * noise
        
        if clip_denoised:
            x_prev = torch.clamp(x_prev, -3.0, 3.0)
        
        return x_prev


@torch.no_grad()
def p_sample_loop_x0(model: nn.Module, shape: Tuple[int, ...], diffusion_consts: Dict[str, torch.Tensor], 
                     device: torch.device, progress: bool = True, clip_denoised: bool = False) -> torch.Tensor:
    """
    Full reverse sampling loop (predict x0).
    
    Args:
        model: Denoising model (predicts x0)
        shape: Output shape [B, N*N*N, code_dim]
        diffusion_consts: Diffusion constants
        device: Device
        progress: Kept for backward compatibility (not used)
        clip_denoised: Whether to clip predicted_x0 (default False)
    
    Returns:
        Generated samples [B, N*N*N, code_dim]
    """
    _ = progress
    batch_size = shape[0]
    x_t = torch.randn(shape, device=device)
    num_timesteps = diffusion_consts["betas"].shape[0]
    
    timestep_indices = torch.arange(num_timesteps - 1, -1, -1, device=device, dtype=torch.long)
    
    precomputed_consts = {
        "betas": extract_batch(diffusion_consts["betas"], timestep_indices, batch_size, shape),
        "alphas_cumprod": extract_batch(diffusion_consts["alphas_cumprod"], timestep_indices, batch_size, shape),
        "alphas_cumprod_prev": extract_batch(diffusion_consts["alphas_cumprod_prev"], timestep_indices, batch_size, shape),
        "sqrt_alphas": extract_batch(diffusion_consts["sqrt_alphas"], timestep_indices, batch_size, shape),
        "sqrt_alphas_cumprod_prev": extract_batch(diffusion_consts["sqrt_alphas_cumprod_prev"], timestep_indices, batch_size, shape),
        "posterior_variance": extract_batch(diffusion_consts["posterior_variance"], timestep_indices, batch_size, shape),
    }
    
    for timestep_idx in range(num_timesteps):
        t = torch.full((batch_size,), timestep_indices[timestep_idx].item(), device=device, dtype=torch.long)
        x_t = p_sample_x0(model, x_t, t, diffusion_consts, clip_denoised=clip_denoised,
                         precomputed_consts=precomputed_consts, timestep_idx=timestep_idx)
    
    return x_t


# ===========================
# 6. DDPM training loss
# ===========================
def compute_ddpm_loss(model: nn.Module, x_0: torch.Tensor, diffusion_consts: Dict[str, torch.Tensor], 
                     device: torch.device, position_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Compute DDPM training loss (predict noise epsilon).
    
    Args:
        model: Denoising model
        x_0: Raw data [B, N*N*N, code_dim]
        diffusion_consts: Diffusion constants
        device: Device
        position_weights: Position weights [B, N*N*N], optional
    
    Returns:
        Training loss
    """
    # Ensure all tensors involved (x_0, diffusion constants, timesteps) live on the same device
    device = x_0.device

    # Move diffusion constants to the current data device if needed
    for k, v in diffusion_consts.items():
        if v.device != device:
            diffusion_consts[k] = v.to(device)

    batch_size = x_0.shape[0]
    
    t = torch.randint(0, diffusion_consts["betas"].shape[0], (batch_size,), device=device).long()
    noise = torch.randn_like(x_0)
    x_t = q_sample(x_0, t, diffusion_consts, noise)
    predicted_noise = model(x_t, t)
    
    if position_weights is not None:
        squared_diff = (predicted_noise - noise) ** 2
        squared_diff_per_pos = squared_diff.mean(dim=-1)
        loss = (position_weights * squared_diff_per_pos).mean()
    else:
        loss = F.mse_loss(predicted_noise, noise)
    
    return loss


def compute_ddpm_loss_x0(model: nn.Module, x_0: torch.Tensor, diffusion_consts: Dict[str, torch.Tensor], 
                         device: torch.device, use_time_weight: bool = True, 
                         position_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Compute DDPM training loss (predict x0).
    
    Args:
        model: Denoising model (predicts x0)
        x_0: Raw data [B, N*N*N, code_dim]
        diffusion_consts: Diffusion constants
        device: Device
        use_time_weight: Whether to use timestep weight 1 + num_timesteps/(t+1)
        position_weights: Position weights [B, N*N*N], optional
    
    Returns:
        Training loss
    """
    # Ensure all tensors involved (x_0, diffusion constants, timesteps) on the same device
    device = x_0.device

    # Move diffusion constants to the current data device if needed
    for k, v in diffusion_consts.items():
        if v.device != device:
            diffusion_consts[k] = v.to(device)

    batch_size = x_0.shape[0]
    num_timesteps = diffusion_consts["betas"].shape[0]
    
    t = torch.randint(0, num_timesteps, (batch_size,), device=device).long()
    noise = torch.randn_like(x_0)
    x_t = q_sample(x_0, t, diffusion_consts, noise)
    predicted_x0 = model(x_t, t)
    
    if position_weights is not None:
        squared_diff = (predicted_x0 - x_0) ** 2
        squared_diff_per_pos = squared_diff.mean(dim=-1)
        weighted_loss_per_pos = position_weights * squared_diff_per_pos
        loss_per_sample = weighted_loss_per_pos.mean(dim=1)
    else:
        loss_per_sample = F.mse_loss(predicted_x0, x_0, reduction='none').mean(dim=tuple(range(1, predicted_x0.ndim)))
    
    if use_time_weight:
        weights = 1.0 + num_timesteps / (t.float() + 1.0)
        loss = (loss_per_sample * weights).mean()
    else:
        loss = loss_per_sample.mean()
    
    if torch.isnan(loss) or torch.isinf(loss):
        print("[WARNING] Loss is NaN or Inf in compute_ddpm_loss_x0")
        print(f"  predicted_x0: min={predicted_x0.min().item():.6f}, max={predicted_x0.max().item():.6f}, mean={predicted_x0.mean().item():.6f}")
        print(f"  x_0: min={x_0.min().item():.6f}, max={x_0.max().item():.6f}, mean={x_0.mean().item():.6f}")
        if use_time_weight:
            weights = 1.0 + num_timesteps / (t.float() + 1.0)
            print(f"  weights: min={weights.min().item():.6f}, max={weights.max().item():.6f}, mean={weights.mean().item():.6f}")
        if position_weights is not None:
            print(f"  position_weights: min={position_weights.min().item():.6f}, max={position_weights.max().item():.6f}, mean={position_weights.mean().item():.6f}")
        loss = torch.tensor(1e-6, device=device, requires_grad=True)
    
    return loss


# ===========================
# 7. Create diffusion constants (from config)
# ===========================
def create_diffusion_constants(config: dict, device: Optional[torch.device] = None) -> Dict[str, torch.Tensor]:
    """
    Create diffusion constants.
    
    Args:
        config: Config dict
        device: Target device (if provided, all constants on this device)
    
    Returns:
        Diffusion constants dict
    """
    ddpm_config = config.get("ddpm", {})
    betas = get_beta_schedule(
        beta_start=ddpm_config.get("beta_start", 1e-4),
        beta_end=ddpm_config.get("beta_end", 0.02),
        num_timesteps=ddpm_config.get("num_timesteps", 1000),
        schedule=ddpm_config.get("schedule", "linear"),
        s=ddpm_config.get("s", 0.008),
    )
    return prepare_diffusion_constants(betas, device=device)
