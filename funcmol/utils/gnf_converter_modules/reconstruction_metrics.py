"""
评估指标模块：重建质量评估
"""
import torch
import numpy as np
from typing import Dict
from funcmol.utils.constants import PADDING_INDEX


class ReconstructionMetrics:
    """重建质量评估器"""
    
    @staticmethod
    def compute_reconstruction_metrics(
        recon_coords: torch.Tensor,
        recon_types: torch.Tensor,
        gt_coords: torch.Tensor,
        gt_types: torch.Tensor
    ) -> Dict[str, float]:
        """
        计算重建分子与真实分子之间的评估指标。
        
        Args:
            recon_coords: 重建的分子坐标 [batch_size, n_atoms, 3]
            recon_types: 重建的原子类型 [batch_size, n_atoms]
            gt_coords: 真实的分子坐标 [batch_size, n_atoms, 3]
            gt_types: 真实的原子类型 [batch_size, n_atoms]
            
        Returns:
            Dict[str, float]: 包含各种评估指标的字典
        """
        from funcmol.utils.utils_nf import compute_rmsd
        
        batch_size = recon_coords.size(0)
        metrics = {
            'avg_rmsd': 0.0,
            'min_rmsd': float('inf'),
            'max_rmsd': 0.0,
            'successful_reconstructions': 0
        }
        
        rmsd_values = []
        
        for b in range(batch_size):
            # 过滤掉填充的原子
            gt_mask = gt_types[b] != PADDING_INDEX
            recon_mask = recon_types[b] != -1  # 假设-1是填充值
            
            if gt_mask.sum() > 0 and recon_mask.sum() > 0:
                gt_valid_coords = gt_coords[b, gt_mask]
                recon_valid_coords = recon_coords[b, recon_mask]
                
                # 计算RMSD
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

