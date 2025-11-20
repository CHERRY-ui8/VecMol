import torch
from torch import nn
import numpy as np
from funcmol.models.egnn import EGNNVectorField

class Decoder(nn.Module):
    def __init__(self, config):
        """
        Initializes the Decoder class.

        Args:
            config (dict): Configuration dictionary containing parameters for the decoder.
            device (torch.device): The device to run the model on.
        """
        super().__init__()
        # self.device = device
        # self.grid_dim = config["grid_size"]  # Add grid_dim attribute
        # self.n_channels = config["n_channels"]  # Add n_channels attribute
        # self.code_stats: Optional[Dict[str, Any]] = None  # Initialize code_stats attribute
                
        self.net = EGNNVectorField(
            grid_size=config["grid_size"],
            hidden_dim=config["hidden_dim"],
            num_layers=config["n_layers"],
            radius=config["radius"],
            n_atom_types=config["n_channels"],
            code_dim=config["code_dim"],
            cutoff=config.get("cutoff", None),  # Add cutoff parameter with default None
            anchor_spacing=config.get("anchor_spacing", 2.0),  # Add anchor_spacing parameter with default 2.0
            # device=device,
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

    def set_code_stats(self, code_stats: dict) -> None:
        """
        Set the code statistics.

        Args:
            code_stats: Code statistics.
        """
        self.code_stats = code_stats


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