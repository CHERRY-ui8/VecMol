import torch
import torch.nn as nn
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from typing import Callable, Optional, Tuple, List, Dict, Any
from scipy.spatial import cKDTree
import torch.nn.functional as F
from funcmol.utils.constants import PADDING_INDEX

class MolecularStructure:
    """Data class to represent a molecular structure."""

    def __init__(self, coords: torch.Tensor, atom_types: Optional[torch.Tensor] = None):
        """
        Args:
            coords: Atomic coordinates (N, 3)
            atom_types: Integer tensor of atom types (N,), each unique value represents a different element
        """
        self.coords = coords  # shape: (N, 3)
        self.atom_types = atom_types  # shape: (N,)
        if atom_types is not None:
            self.unique_types = torch.unique(atom_types)
        else:
            self.unique_types = None
    
    def __repr__(self):
        return f"MolecularStructure(coords={self.coords}, atom_types={self.atom_types})"
    
    def __eq__(self, other):
        if not isinstance(other, MolecularStructure):
            return NotImplemented
        return torch.allclose(self.coords, other.coords) and (
            self.atom_types is None and other.atom_types is None or
            torch.equal(self.atom_types, other.atom_types)
        )

    def split_by_type(self) -> List[torch.Tensor]:
        """Split coordinates by atom type."""
        if self.atom_types is None:
            return [self.coords]
        
        return [self.coords[self.atom_types == t] for t in self.unique_types]

class GNFConverter(nn.Module):
    """
    Converter between molecular structures and Gradient Neural Fields (GNF).
    
    The GNF is defined as r = f(X,z) where:
    - X is the set of atomic coordinates
    - z is a query point in 3D space
    - r is the gradient at point z
    """
    
    def __init__(self,
                sigma: float = 1.0,
                n_query_points: int = 10000,
                n_iter: int = 10000,
                step_size: float = 0.001,
                merge_threshold: float = 0.005, # Distance threshold for merging points
                device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__()
        self.sigma = sigma
        self.n_query_points = n_query_points
        self.n_iter = n_iter
        self.step_size = step_size
        self.merge_threshold = merge_threshold
        self.device = device
        self.trajectories = []  # 存储采样点轨迹
        self.nearest_atoms = []  # 存储每个采样点最近的原子
    
    def forward(self, mol: MolecularStructure, query_points: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: compute gradient field at query points.
        
        Args:
            mol: Molecular structure containing atomic coordinates
            query_points: Points to evaluate gradient field at (shape: (M, 3))
            
        Returns:
            Gradients at query points (shape: (M, 3))
        """
        return self.mol2gnf(mol.coords, query_points)

    def compute_transformed_gradients(self,
            individual_gradients: torch.Tensor,
            log_method: str = 'log',  # 'log1p' or 'log'
            normalize_directions: bool = True,
            magnitude_clip: float = None,  # Clip gradient magnitudes before log (e.g., 1.0)
            step_clip: float = None,       # Clip final summed gradient (e.g., 10.0)
            lr: float = 0.3,               # Step size scale factor
            eps: float = 1e-8              # To avoid log(0)
        ) -> torch.Tensor:
        """
        Transforms individual gradients via direction normalization and log-magnitude scaling,
        then sums over source atoms to produce final gradients.

        Args:
            individual_gradients: (N, M, 3) tensor, where N is num of query points, M is num of atoms.
            log_method: 'log1p' (recommended) or 'log'.
            normalize_directions: Whether to normalize gradient vectors to unit directions.
            magnitude_clip: Optional float, clip gradient magnitudes before log.
            step_clip: Optional float, clip final gradient vectors after sum.
            lr: Step size factor.
            eps: Small value to avoid division by zero or log(0).

        Returns:
            final_gradients: (M, 3) tensor, summed transformed gradients per atom.
        """

        # Compute gradient magnitudes (N, M, 1)
        gradient_magnitudes = torch.norm(individual_gradients, dim=2, keepdim=True)

        # Optional: clip magnitudes to prevent log explosion
        if magnitude_clip is not None:
            gradient_magnitudes = torch.clamp(gradient_magnitudes, max=magnitude_clip)

        # Compute gradient directions (unit vectors)
        if normalize_directions:
            gradient_directions = individual_gradients / (gradient_magnitudes + eps)
        else:
            gradient_directions = individual_gradients.clone()

        # Compute log-scaled magnitudes
        if log_method == 'log1p':
            log_magnitudes = torch.log1p(gradient_magnitudes / eps)
        elif log_method == 'log':
            log_magnitudes = torch.log(gradient_magnitudes + eps)
        else:
            raise ValueError(f"Unsupported log_method: {log_method}")

        # Multiply direction by scaled magnitude (N, M, 3)
        transformed_gradients = gradient_directions * log_magnitudes

        # Sum across all query points (N → 1), get (M, 3)
        final_gradients = torch.sum(transformed_gradients, dim=0)

        # Optional: clip the final gradients
        if step_clip is not None:
            final_gradients = torch.clamp(final_gradients, min=-step_clip, max=step_clip)

        # Apply learning rate scaling
        return lr * final_gradients

    
    def mol2gnf(self, coords: torch.Tensor, query_points: torch.Tensor, 
                atom_types: Optional[torch.Tensor] = None, version: int = 1) -> torch.Tensor:
        """
        Convert molecular coordinates to GNF values at query points.
        
        Args:
            coords: Atom coordinates (shape: (N, 3))
            query_points: Query points (shape: (M, 3))
            atom_types: Optional atom types (shape: (N,))
            version: Version of GNF calculation to use
            
        Returns:
            GNF values at query points (shape: (M, 3))
        """
        if atom_types is None:
            return self._compute_gnf(coords, query_points, version)
        
        # 如果提供了原子类型，分别计算每种类型的梯度场
        unique_types = torch.unique(atom_types)
        total_gradients = torch.zeros_like(query_points)
        
        for atom_type in unique_types:
            mask = atom_types == atom_type
            type_coords = coords[mask]
            type_gradients = self._compute_gnf(type_coords, query_points, version)
            total_gradients += type_gradients
        
        return total_gradients

    def _compute_gnf(self, coords: torch.Tensor, query_points: torch.Tensor, version: int) -> torch.Tensor:
        """Internal function to compute GNF for a single atom type"""
        coords = coords.unsqueeze(1)      # (N, 1, 3)
        query_points = query_points.unsqueeze(0)  # (1, M, 3)
        
        diff = coords - query_points      # (N, M, 3)
        dist_sq = torch.sum(diff ** 2, dim=-1, keepdim=True)  # (N, M, 1)
        
        individual_gradients = diff * torch.exp(-dist_sq / (2 * self.sigma ** 2)) / (self.sigma ** 2)

        if version == 1:
            return torch.sum(individual_gradients, dim=0)  # (M, 3)

        if version == 2:  # 使用LogSumExp函数处理梯度场，可以有效拉近远近点的影响差异
            eps = 1e-8

            # 对每个原子产生的梯度分别进行幅度的对数变换
            gradient_magnitudes = torch.norm(individual_gradients, dim=2, keepdim=True)  # (N, M, 1)，计算每个梯度向量的模长
            gradient_directions = individual_gradients / (gradient_magnitudes + eps)  # (N, M, 3)，归一化

            # 计算LogSumExp
            log_sum_exp = torch.logsumexp(gradient_magnitudes, dim=0, keepdim=True)  # (1, M, 1)

            # 检查是否有负值
            if torch.any(log_sum_exp < 0):
                # 如果出现负值，我们可以选择将其截断为非负值
                log_sum_exp = torch.clamp(log_sum_exp, min=0)

            # 计算最终梯度：方向之和 × LogSumExp(log(||∇fᵢ(z)||))
            final_gradients = torch.sum(gradient_directions, dim=0) * log_sum_exp.squeeze(0)  # (M, 3)

            return final_gradients

        if version == 3:
            final_gradients = self.compute_transformed_gradients(
                individual_gradients,
                log_method='log',
                normalize_directions=True,
                magnitude_clip=0.5,
                step_clip=1.0,
                lr=0.1
            )
            return final_gradients

        if version == 4:
            temperature = 0.5
            distances = torch.sqrt(dist_sq.squeeze(-1))  # (N, M)
            weights = torch.softmax(-distances / temperature, dim=0)  # (N, M)
            weights = weights.unsqueeze(-1)  # (N, M, 1)
            weighted_gradients = diff * weights  # (N, M, 3)
            return torch.sum(weighted_gradients, dim=0)  # (M, 3)
        
        return torch.sum(individual_gradients, dim=0)  # (M, 3)

    def gnf2mol(self, grad_field: torch.Tensor, 
                decoder: nn.Module,
                codes: torch.Tensor,
                atom_types: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        直接用梯度场重建分子坐标。
        Args:
            grad_field: [batch, n_points, n_atom_types, 3]  # 神经场输出
            decoder: 解码器模型，用于动态计算向量场
            codes: [batch, grid_size**3, code_dim]  # 编码器的输出
            atom_types: 可选的原子类型
        Returns:
            (final_coords, final_types)
        """
        device = grad_field.device
        batch_size, n_points, n_atom_types, _ = grad_field.shape
        n_query_points = min(self.n_query_points, n_points)

        all_coords = []
        all_types = []
        # 对每个batch分别处理
        for b in range(batch_size):
            coords_list = []
            types_list = []
            # 对每个原子类型分别做流动和聚类
            for t in range(n_atom_types):
                # 1. 初始化采样点（可用均匀网格或随机点）
                z = torch.rand(n_query_points, 3, device=device) * 2 - 1  # [-1,1]区间
                
                # 2. 梯度上升（每次迭代都重新计算梯度！）
                for i in range(self.n_iter):
                    # 在新位置计算梯度场
                    z_batch = z.unsqueeze(0)  # [1, n_query_points, 3]
                    current_field = decoder(z_batch, codes[b:b+1])  # [1, n_query_points, n_atom_types, 3]
                    grad = current_field[0, :, t, :]  # [n_query_points, 3]
                    
                    # 更新采样点位置
                    z = z + self.step_size * grad
                                    
                # 3. 聚类/合并
                z_np = z.detach().cpu().numpy()
                merged_points = self._merge_points(z_np)
                if len(merged_points) > 0:
                    coords_list.append(torch.from_numpy(merged_points).to(device))
                    types_list.append(torch.full((len(merged_points),), t, dtype=torch.long, device=device))
            
            # 合并所有类型
            if coords_list:
                all_coords.append(torch.cat(coords_list, dim=0))
                all_types.append(torch.cat(types_list, dim=0))
            else:
                all_coords.append(torch.empty(0, 3, device=device))
                all_types.append(torch.empty(0, dtype=torch.long, device=device))
        
        # pad到batch最大长度
        max_atoms = max([c.size(0) for c in all_coords]) if all_coords else 0
        final_coords = torch.stack([F.pad(c, (0,0,0,max_atoms-c.size(0))) if c.size(0)<max_atoms else c for c in all_coords], dim=0)
        final_types = torch.stack([F.pad(t, (0,max_atoms-t.size(0)), value=-1) if t.size(0)<max_atoms else t for t in all_types], dim=0)
        return final_coords, final_types

    def _merge_points(self, points: np.ndarray) -> np.ndarray:
        """Merge close points and automatically determine the number of atoms."""
        # 确保输入数据形状正确
        if len(points.shape) == 3:  # [batch_size, n_points, 3]
            points = points.reshape(-1, 3)  # 展平为 [n_points, 3]
        elif len(points.shape) != 2 or points.shape[1] != 3:
            raise ValueError(f"Expected points to be of shape (n, 3) or (batch_size, n, 3), got {points.shape}")
        
        # 计算点密度
        tree = cKDTree(points)
        # 查询每个点的k个最近邻
        k = min(5, len(points))  # 使用较小的k值
        distances, _ = tree.query(points, k=k)
        # 计算局部密度（使用最近邻距离的倒数）
        densities = 1.0 / (np.mean(distances[:, 1:], axis=1) + 1e-8)  # 排除自身
        
        # 使用基于密度分布的自适应阈值
        density_mean = np.mean(densities)
        density_std = np.std(densities)
        
        # 使用 mean + α * std 作为阈值，α可以根据需要调整
        alpha = 0.5  # 可以根据实际需求调整这个参数
        density_threshold = density_mean + alpha * density_std
        
        # 如果阈值太高（筛选掉太多点），使用更保守的阈值
        significant_points = densities > density_threshold
        if np.sum(significant_points) < len(points) * 0.4:  # 如果保留的点太少
            # 使用分位数作为备选方案，但使用更保守的阈值
            density_threshold = np.percentile(densities, 60)  # 保留密度较高的40%的点
            significant_points = densities > density_threshold
        
        if np.sum(significant_points) > 0:
            return points[significant_points]
        
        # 如果没有显著点，使用更保守的合并策略
        merged = []
        used = set()
        
        # 使用更小的合并阈值
        conservative_threshold = self.merge_threshold * 0.5
        
        for i in range(len(points)):
            if i in used:
                continue
            
            # 找到当前点的所有近邻
            neighbors = [i]
            for j in range(len(points)):
                if j in used or j == i:
                    continue
                if np.linalg.norm(points[i] - points[j]) < conservative_threshold:
                    neighbors.append(j)
                    used.add(j)
            
            # 计算簇的中心
            cluster = points[neighbors]
            center = np.mean(cluster, axis=0)
            merged.append(center)
            used.add(i)
        
        merged = np.array(merged)
        return merged

    def compute_reconstruction_metrics(
        self,
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
    
    def visualize_conversion_results(
        self,
        recon_coords: torch.Tensor,
        recon_types: torch.Tensor,
        gt_coords: Optional[torch.Tensor] = None,
        gt_types: Optional[torch.Tensor] = None,
        save_dir: Optional[str] = None,
        sample_indices: Optional[list] = None
    ) -> None:
        """
        可视化转换结果，比较重建分子与真实分子。
        
        Args:
            recon_coords: 重建的分子坐标
            recon_types: 重建的原子类型
            gt_coords: 可选的真实分子坐标
            gt_types: 可选的真实原子类型
            save_dir: 保存可视化结果的目录
            sample_indices: 要可视化的样本索引列表
        """
        if save_dir is None:
            return
        
        import os
        from funcmol.utils.visualize_molecules import visualize_molecule_comparison, visualize_single_molecule
        
        os.makedirs(save_dir, exist_ok=True)
        
        batch_size = recon_coords.size(0)
        if sample_indices is None:
            sample_indices = list(range(min(batch_size, 5)))  # 默认可视化前5个样本
        
        for i in sample_indices:
            if i >= batch_size:
                continue
                
            # 过滤填充的原子
            recon_mask = recon_types[i] != -1
            recon_valid_coords = recon_coords[i, recon_mask]
            recon_valid_types = recon_types[i, recon_mask]
            
            if gt_coords is not None and gt_types is not None:
                gt_mask = gt_types[i] != PADDING_INDEX
                gt_valid_coords = gt_coords[i, gt_mask]
                gt_valid_types = gt_types[i, gt_mask]
                
                # 比较可视化
                save_path = os.path.join(save_dir, f"sample{i:02d}_comparison.png")
                visualize_molecule_comparison(
                    gt_valid_coords,
                    gt_valid_types,
                    recon_valid_coords,
                    recon_valid_types,
                    save_path=save_path
                )
            else:
                # 只可视化重建分子
                save_path = os.path.join(save_dir, f"sample{i:02d}_reconstructed.png")
                visualize_single_molecule(
                    recon_valid_coords,
                    recon_valid_types,
                    save_path=save_path
                )