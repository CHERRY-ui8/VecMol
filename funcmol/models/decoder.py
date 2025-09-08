import torch
from torch import nn
import numpy as np
from typing import Optional, Dict, Any
from funcmol.models.egnn import EGNNVectorField

class Decoder(nn.Module):
    def __init__(self, config, device):
        """
        Initializes the Decoder class.

        Args:
            config (dict): Configuration dictionary containing parameters for the decoder.
            device (torch.device): The device to run the model on.
        """
        super().__init__()
        self.device = device
        self.grid_dim = config["grid_size"]  # Add grid_dim attribute
        self.n_channels = config["n_channels"]  # Add n_channels attribute
        self.code_stats: Optional[Dict[str, Any]] = None  # Initialize code_stats attribute
        
        # Initialize coords using get_grid function
        _, self.coords = get_grid(self.grid_dim, resolution=0.25)
        self.coords = self.coords.to(device)
        
        self.net = EGNNVectorField(
            grid_size=config["grid_size"],
            hidden_dim=config["hidden_dim"],
            num_layers=config["n_layers"],
            radius=config["radius"],
            n_atom_types=config["n_channels"],
            code_dim=config["code_dim"],
            cutoff=config.get("cutoff", None),  # Add cutoff parameter with default None
            anchor_spacing=config.get("anchor_spacing", 2.0),  # Add anchor_spacing parameter with default 2.0
            device=device,
            k_neighbors=config.get("k_neighbors", 32)  # Add k_neighbors parameter with default 32
        )

    def forward(self, x, codes):
        """
        x: [B, n_points, 3]
        codes: [B, n_grid, code_dim]
        """
        # Pass the 3D tensors directly to the EGNN, which handles batching internally.
        vector_field = self.net(x, codes)
        
        # The output from EGNN is already in the correct shape: [B, n_points, n_atom_types, 3]
        return vector_field

    def render_code(
        self,
        codes: torch.Tensor,
        batch_size_render: int,
        fabric: object = None,
    ) -> torch.Tensor:
        """
        Renders molecules from given codes.

        Args:
            codes (torch.Tensor): Tensor containing the codes to be rendered.
                                  The codes need to be unnormalized before rendering.
            batch_size_render (int): The size of the batches for rendering.
            fabric (object, optional): An object that provides a print method for logging.
                                       Defaults to None.

        Returns:
            torch.Tensor: A tensor representing the rendered molecules in a grid format.
        """
        # 添加调试信息
        fabric.print(f">> render_code - input codes shape: {codes.shape}")
        fabric.print(f">> render_code - batch_size_render: {batch_size_render}")
        
        # 检查输入
        if torch.isnan(codes).any():
            fabric.print(">> ERROR: Input codes contains NaN values!")
            raise ValueError("Input codes contains NaN values")
        
        if torch.isinf(codes).any():
            fabric.print(">> ERROR: Input codes contains Inf values!")
            raise ValueError("Input codes contains Inf values")
        
        # PS: code need to be unnormealized before rendering
        with torch.no_grad():
            if fabric:
                fabric.print(f">> Rendering molecules - batches of {batch_size_render}")
            if codes.device != self.device:
                codes = codes.to(self.device)
            
            # 添加调试信息：检查coords的维度
            if fabric:
                fabric.print(f">> self.coords shape: {self.coords.shape}")
                fabric.print(f">> self.grid_dim: {self.grid_dim}")
                fabric.print(f">> Expected coords size: {self.grid_dim**3}")
            
            # 检查coords维度是否正确
            expected_coords_size = self.grid_dim ** 3
            if self.coords.numel() != expected_coords_size * 3:
                fabric.print(f">> ERROR: coords size mismatch! Expected {expected_coords_size * 3}, got {self.coords.numel()}")
                raise ValueError(f"coords size mismatch: expected {expected_coords_size * 3}, got {self.coords.numel()}")
            
            # 检查并调整batch_size_render，确保不会超过xs的大小
            if batch_size_render > expected_coords_size:
                if fabric:
                    fabric.print(f">> WARNING: batch_size_render ({batch_size_render}) > expected_coords_size ({expected_coords_size})")
                    fabric.print(f">> Adjusting batch_size_render to {expected_coords_size}")
                batch_size_render = expected_coords_size
            
            xs = self.coords.reshape(1, -1, 3)
            if fabric:
                fabric.print(f">> xs shape after reshape: {xs.shape}")
                fabric.print(f">> Final batch_size_render: {batch_size_render}")
            
            # 检查codes的batch大小，如果超过batch_size_render则分批处理
            codes_batch_size = codes.size(0)
            if codes_batch_size > batch_size_render:
                # 分批处理codes
                codes_batches = torch.split(codes, batch_size_render, dim=0)
                if fabric:
                    fabric.print(f">> Splitting codes into {len(codes_batches)} batches for rendering")
                
                pred_list = []
                for i, codes_batch in enumerate(codes_batches):
                    if fabric:
                        fabric.print(f">> Processing codes batch {i+1}/{len(codes_batches)}, codes shape: {codes_batch.shape}")
                    
                    # 为每个codes batch创建对应的xs
                    xs_batch = xs.expand(codes_batch.size(0), -1, -1)  # [batch_size, 729, 3]
                    if fabric:
                        fabric.print(f">> xs_batch shape: {xs_batch.shape}")
                    
                    pred_batch = self.forward_batched(xs_batch, codes_batch, batch_size_render=batch_size_render, threshold=0.2, fabric=fabric)
                    pred_list.append(pred_batch)
                
                pred = torch.cat(pred_list, dim=0)  # 在batch维度上拼接
            else:
                # 直接处理，确保xs和codes的batch维度匹配
                xs_expanded = xs.expand(codes.size(0), -1, -1)  # [batch_size, 729, 3]
                if fabric:
                    fabric.print(f">> xs_expanded shape: {xs_expanded.shape}")
                pred = self.forward_batched(xs_expanded, codes, batch_size_render=batch_size_render, threshold=0.2, fabric=fabric)
            
            # pred的形状是[B, N, T, 3]，需要转换为[B, T, N, 3]然后reshape
            if pred.dim() == 4:  # [B, N, T, 3]
                pred = pred.permute(0, 2, 1, 3)  # [B, T, N, 3]
                grid = pred.reshape(-1, self.n_channels, self.grid_dim, self.grid_dim, self.grid_dim)
            else:  # [B, N, T, 3] -> [B, T, N, 3]
                grid = pred.permute(0, 2, 1).reshape(-1, self.n_channels, self.grid_dim, self.grid_dim, self.grid_dim)
        return grid

    def forward_batched(self, xs, codes, batch_size_render=100_000, threshold=None, to_cpu=True, fabric=None):
        """
        When memory is limited, render the grid in batches.
        """
        # 添加调试信息
        if fabric:
            fabric.print(f">> forward_batched - xs shape: {xs.shape}")
            fabric.print(f">> forward_batched - codes shape: {codes.shape}")
            fabric.print(f">> forward_batched - batch_size_render: {batch_size_render}")
        
        # 现在xs和codes的batch维度应该匹配，直接调用decoder
        try:
            pred = self(xs, codes)
            if to_cpu:
                pred = pred.cpu()
            if threshold is not None:
                pred[pred < threshold] = 0
        except Exception as e:
            if fabric:
                fabric.print(f">> ERROR in forward_batched: {e}")
                fabric.print(f">>   xs shape: {xs.shape}")
                fabric.print(f">>   codes shape: {codes.shape}")
            raise e
        
        return pred





    def set_code_stats(self, code_stats: dict) -> None:
        """
        Set the code statistics.

        Args:
            code_stats: Code statistics.
        """
        self.code_stats = code_stats

    def unnormalize_code(self, codes: torch.Tensor):
        """
        Unnormalizes the given codes based on the provided normalization parameters.
        NOTE: This function is deprecated and will be removed in future versions.
        For now, it returns the original codes without unnormalization.

        Args:
            codes (torch.Tensor): The codes to be unnormalized.
            code_stats (dict): The statistics for the codes.

        Returns:
            torch.Tensor: The original codes (no unnormalization applied).
        """
        # Return original codes without unnormalization
        return codes


########################################################################################
## auxiliary functions


def get_grid(grid_dim, resolution=0.25):
    """
    Create a grid based on real-world distances.
    
    Args:
        grid_dim (int): Number of grid points per dimension
        resolution (float): Distance between grid points in Angstroms (default: 0.25)
    
    Returns:
        tuple: (discrete_grid, full_grid) where discrete_grid is 1D array and full_grid is 3D coordinates
    """
    # Calculate the total span in Angstroms
    total_span = (grid_dim - 1) * resolution
    half_span = total_span / 2
    
    # Create grid points in real space (Angstroms)
    discrete_grid = np.linspace(-half_span, half_span, grid_dim)
    
    # Create full 3D grid
    full_grid = torch.Tensor(
        [[a, b, c] for a in discrete_grid for b in discrete_grid for c in discrete_grid]
    )
    return discrete_grid, full_grid