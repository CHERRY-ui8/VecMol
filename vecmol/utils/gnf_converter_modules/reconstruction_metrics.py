"""
Reconstruction metrics: quality evaluation for reconstructed molecules.
"""
import torch
import numpy as np
from typing import Dict
from vecmol.utils.constants import PADDING_INDEX


class ReconstructionMetrics:
    """Reconstruction quality evaluator."""

    @staticmethod
    def compute_reconstruction_metrics(
        recon_coords: torch.Tensor,
        recon_types: torch.Tensor,
        gt_coords: torch.Tensor,
        gt_types: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute metrics between reconstructed and ground-truth molecules.

        Args:
            recon_coords: Reconstructed coords [batch_size, n_atoms, 3]
            recon_types: Reconstructed atom types [batch_size, n_atoms]
            gt_coords: Ground-truth coords [batch_size, n_atoms, 3]
            gt_types: Ground-truth atom types [batch_size, n_atoms]

        Returns:
            Dict of metric names to values.
        """
        from vecmol.utils.utils_nf import compute_rmsd

        batch_size = recon_coords.size(0)
        metrics = {
            'avg_rmsd': 0.0,
            'min_rmsd': float('inf'),
            'max_rmsd': 0.0,
            'successful_reconstructions': 0
        }

        rmsd_values = []

        for b in range(batch_size):
            # Filter out padding atoms
            gt_mask = gt_types[b] != PADDING_INDEX
            recon_mask = recon_types[b] != -1  # -1 is padding

            if gt_mask.sum() > 0 and recon_mask.sum() > 0:
                gt_valid_coords = gt_coords[b, gt_mask]
                recon_valid_coords = recon_coords[b, recon_mask]

                rmsd = compute_rmsd(gt_valid_coords, recon_valid_coords)
                rmsd_value = rmsd.item() if hasattr(rmsd, 'item') else float(rmsd)
                rmsd_values.append(rmsd_value)
                
                metrics['avg_rmsd'] += rmsd_value
                metrics['min_rmsd'] = min(metrics['min_rmsd'], rmsd_value)
                metrics['max_rmsd'] = max(metrics['max_rmsd'], rmsd_value)
                metrics['successful_reconstructions'] += 1
        
        if metrics['successful_reconstructions'] > 0:
            metrics['avg_rmsd'] /= metrics['successful_reconstructions']
            metrics['rmsd_std'] = np.std(rmsd_values) if rmsd_values else 0.0
        else:
            metrics['avg_rmsd'] = float('inf')
            metrics['rmsd_std'] = 0.0
        
        return metrics

