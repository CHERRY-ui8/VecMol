"""
数据增强模块：用于denoiser训练时的数据增强
支持旋转和平移增强
"""
import torch
import torch.nn.functional as F
import numpy as np


def apply_rotation_3d(codes: torch.Tensor, grid_size: int) -> torch.Tensor:
    """
    对3D网格codes应用随机旋转增强
    
    Args:
        codes: 输入codes [B, N*N*N, code_dim]
        grid_size: 网格大小 N
        
    Returns:
        旋转后的codes [B, N*N*N, code_dim]
    """
    batch_size, n_points, code_dim = codes.shape
    device = codes.device
    
    # Reshape到3D网格: [B, N, N, N, code_dim]
    codes_3d = codes.view(batch_size, grid_size, grid_size, grid_size, code_dim)
    
    # 生成随机旋转角度（绕x, y, z轴）
    angles = torch.rand(3, device=device) * 2 * np.pi  # [0, 2π]
    
    # 创建旋转矩阵
    cos_x, sin_x = torch.cos(angles[0]), torch.sin(angles[0])
    cos_y, sin_y = torch.cos(angles[1]), torch.sin(angles[1])
    cos_z, sin_z = torch.cos(angles[2]), torch.sin(angles[2])
    
    # 使用eye和stack创建旋转矩阵
    Rx = torch.eye(3, device=device, dtype=codes.dtype)
    Rx[1, 1] = cos_x
    Rx[1, 2] = -sin_x
    Rx[2, 1] = sin_x
    Rx[2, 2] = cos_x
    
    Ry = torch.eye(3, device=device, dtype=codes.dtype)
    Ry[0, 0] = cos_y
    Ry[0, 2] = sin_y
    Ry[2, 0] = -sin_y
    Ry[2, 2] = cos_y
    
    Rz = torch.eye(3, device=device, dtype=codes.dtype)
    Rz[0, 0] = cos_z
    Rz[0, 1] = -sin_z
    Rz[1, 0] = sin_z
    Rz[1, 1] = cos_z
    
    # 组合旋转矩阵
    R = Rz @ Ry @ Rx  # [3, 3]
    
    # 创建归一化的网格坐标 [-1, 1]
    grid_1d = torch.linspace(-1, 1, grid_size, device=device)
    mesh = torch.meshgrid(grid_1d, grid_1d, grid_1d, indexing='ij')
    coords = torch.stack(mesh, dim=-1)  # [N, N, N, 3]
    
    # 应用旋转
    coords_flat = coords.reshape(-1, 3)  # [N*N*N, 3]
    coords_rotated = coords_flat @ R.T  # [N*N*N, 3]
    coords_rotated = coords_rotated.reshape(grid_size, grid_size, grid_size, 3)
    
    # 将codes转换为适合grid_sample的格式: [B, code_dim, N, N, N]
    codes_permuted = codes_3d.permute(0, 4, 1, 2, 3)  # [B, code_dim, N, N, N]
    
    # 创建采样网格: [B, N, N, N, 3]
    grid_for_sample = coords_rotated.unsqueeze(0).expand(batch_size, -1, -1, -1, -1)
    
    # 使用grid_sample进行旋转采样（一次性处理所有通道）
    # grid_sample期望输入为 [B, C, D, H, W] 和 [B, D, H, W, 3]
    rotated_codes_3d = F.grid_sample(
        codes_permuted,  # [B, code_dim, N, N, N]
        grid_for_sample,  # [B, N, N, N, 3]
        mode='bilinear',
        padding_mode='border',
        align_corners=True
    )  # [B, code_dim, N, N, N]
    
    # Permute回 [B, N, N, N, code_dim] 并reshape
    rotated_codes_3d = rotated_codes_3d.permute(0, 2, 3, 4, 1)  # [B, N, N, N, code_dim]
    rotated_codes = rotated_codes_3d.view(batch_size, n_points, code_dim)
    
    return rotated_codes


def apply_translation_3d(codes: torch.Tensor, grid_size: int, anchor_spacing: float) -> torch.Tensor:
    """
    对3D网格codes应用随机平移增强
    
    Args:
        codes: 输入codes [B, N*N*N, code_dim]
        grid_size: 网格大小 N
        anchor_spacing: 锚点间距（单位：埃）
        
    Returns:
        平移后的codes [B, N*N*N, code_dim]
    """
    batch_size, n_points, code_dim = codes.shape
    device = codes.device
    
    # Reshape到3D网格: [B, N, N, N, code_dim]
    codes_3d = codes.view(batch_size, grid_size, grid_size, grid_size, code_dim)
    
    # 计算平移距离：1/2个anchor_spacing
    translation_distance = anchor_spacing / 2.0
    
    # 生成随机平移向量（在[-translation_distance, translation_distance]范围内）
    translation = (torch.rand(3, device=device) * 2 - 1) * translation_distance  # [3]
    
    # 创建归一化的网格坐标 [-1, 1]
    grid_1d = torch.linspace(-1, 1, grid_size, device=device)
    mesh = torch.meshgrid(grid_1d, grid_1d, grid_1d, indexing='ij')
    coords = torch.stack(mesh, dim=-1)  # [N, N, N, 3]
    
    # 计算归一化的平移量
    # 网格的总跨度是 (grid_size - 1) * anchor_spacing
    # 归一化后的平移量 = translation / (total_span / 2)
    total_span = (grid_size - 1) * anchor_spacing
    half_span = total_span / 2
    translation_normalized = translation / half_span  # 归一化到[-1, 1]范围
    
    # 应用平移
    coords_translated = coords + translation_normalized.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # [N, N, N, 3]
    
    # 将codes转换为适合grid_sample的格式: [B, code_dim, N, N, N]
    codes_permuted = codes_3d.permute(0, 4, 1, 2, 3)  # [B, code_dim, N, N, N]
    
    # 创建采样网格: [B, N, N, N, 3]
    grid_for_sample = coords_translated.unsqueeze(0).expand(batch_size, -1, -1, -1, -1)
    
    # 使用grid_sample进行平移采样（一次性处理所有通道）
    translated_codes_3d = F.grid_sample(
        codes_permuted,  # [B, code_dim, N, N, N]
        grid_for_sample,  # [B, N, N, N, 3]
        mode='bilinear',
        padding_mode='border',
        align_corners=True
    )  # [B, code_dim, N, N, N]
    
    # Permute回 [B, N, N, N, code_dim] 并reshape
    translated_codes_3d = translated_codes_3d.permute(0, 2, 3, 4, 1)  # [B, N, N, N, code_dim]
    translated_codes = translated_codes_3d.view(batch_size, n_points, code_dim)
    
    return translated_codes


def augment_codes(codes: torch.Tensor, grid_size: int, anchor_spacing: float, 
                  apply_rotation: bool = True, apply_translation: bool = True) -> torch.Tensor:
    """
    对codes应用数据增强（旋转和平移）
    
    Args:
        codes: 输入codes [B, N*N*N, code_dim]
        grid_size: 网格大小 N
        anchor_spacing: 锚点间距（单位：埃）
        apply_rotation: 是否应用旋转增强
        apply_translation: 是否应用平移增强
        
    Returns:
        增强后的codes [B, N*N*N, code_dim]
    """
    augmented_codes = codes
    
    # 随机决定是否应用每种增强（50%概率）
    if apply_rotation and torch.rand(1).item() > 0.5:
        augmented_codes = apply_rotation_3d(augmented_codes, grid_size)
    
    if apply_translation and torch.rand(1).item() > 0.5:
        augmented_codes = apply_translation_3d(augmented_codes, grid_size, anchor_spacing)
    
    return augmented_codes

