import torch
import torch.nn as nn
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from typing import Callable, Optional, Tuple, List
from scipy.spatial import cKDTree

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
        """计算分子梯度场"""
        # 确保输入维度正确
        if len(coords.shape) == 2:  # [N, 3]
            coords = coords.unsqueeze(0)  # [1, N, 3]
        if len(query_points.shape) == 2:  # [M, 3]
            query_points = query_points.unsqueeze(0)  # [1, M, 3]
            
        N = coords.shape[1]  # 原子数量
        M = query_points.shape[1]  # 查询点数量
        
        # 计算原子到查询点的距离向量
        diff = coords.unsqueeze(2) - query_points.unsqueeze(1)  # [B, N, M, 3]
        dist = torch.norm(diff, dim=-1)  # [B, N, M]
        
        # 计算高斯权重
        weights = torch.exp(-0.5 * (dist / self.sigma) ** 2)  # [B, N, M]
        
        # 归一化权重
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)  # [B, N, M]
        
        # 计算梯度场
        if version == 1:
            # 使用距离向量的归一化版本
            diff_norm = diff / (dist.unsqueeze(-1) + 1e-8)  # [B, N, M, 3]
            gradients = torch.sum(weights.unsqueeze(-1) * diff_norm, dim=1)  # [B, M, 3]
        else:
            # 使用原始距离向量
            gradients = torch.sum(weights.unsqueeze(-1) * diff, dim=1)  # [B, M, 3]
            
        return gradients

    def gnf2mol(self, coords: torch.Tensor, 
                atom_types: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Convert gradient field to molecule coordinates.
        
        Args:
            coords: Initial coordinates [batch_size, n_points, n_atom_types, 3]
            atom_types: Optional atom types [batch_size, n_atoms]
            
        Returns:
            Tuple of (final_coords, atom_types)
        """
        device = coords.device
        batch_size = coords.shape[0]
        n_points = coords.shape[1]
        n_atom_types = coords.shape[2]
        
        # 减少查询点数量
        n_query_points = min(self.n_query_points, 200)  # 限制最大查询点数量
        
        # 初始化采样点
        z = torch.rand(batch_size, n_query_points, 3, device=device) * 30 - 15
        
        # 创建GNF函数
        def gnf_func(points):
            # 使用更小的批次大小
            chunk_size = 50  # 减小块大小
            n_chunks = (n_query_points + chunk_size - 1) // chunk_size
            gradients = []
            
            for i in range(n_chunks):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, n_query_points)
                chunk_points = points[:, start_idx:end_idx]
                
                # 重塑输入以匹配维度
                chunk_points = chunk_points.view(batch_size, -1, 1, 3)  # [B, M', 1, 3]
                
                # 分批处理原子类型
                type_chunk_size = 2  # 每次处理2种原子类型
                n_type_chunks = (n_atom_types + type_chunk_size - 1) // type_chunk_size
                type_gradients = []
                
                for j in range(n_type_chunks):
                    start_type = j * type_chunk_size
                    end_type = min((j + 1) * type_chunk_size, n_atom_types)
                    
                    # 只处理当前类型的原子
                    coords_chunk = coords[:, :, start_type:end_type, :]  # [B, N, T', 3]
                    coords_reshaped = coords_chunk.unsqueeze(2)  # [B, N, 1, T', 3]
                    
                    # 计算距离
                    diff = coords_reshaped - chunk_points.unsqueeze(1)  # [B, N, M', T', 3]
                    dist = torch.norm(diff, dim=-1)  # [B, N, M', T']
                    
                    # 计算高斯权重
                    weights = torch.exp(-0.5 * (dist / self.sigma) ** 2)  # [B, N, M', T']
                    weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)  # [B, N, M', T']
                    
                    # 计算梯度场
                    diff_norm = diff / (dist.unsqueeze(-1) + 1e-8)  # [B, N, M', T', 3]
                    chunk_gradients = torch.sum(weights.unsqueeze(-1) * diff_norm, dim=1)  # [B, M', T', 3]
                    type_gradients.append(chunk_gradients)
                    
                    # 清理不需要的张量
                    del diff, dist, weights, diff_norm, chunk_gradients
                    torch.cuda.empty_cache()
                
                # 合并所有类型的结果
                type_gradients = torch.cat(type_gradients, dim=2)  # [B, M', T, 3]
                gradients.append(type_gradients)
                
                # 清理不需要的张量
                del type_gradients
                torch.cuda.empty_cache()
            
            # 合并所有块的结果
            gradients = torch.cat(gradients, dim=1)  # [B, M, T, 3]
            return gradients.view(batch_size, n_query_points, -1)  # [B, M, T*3]
        
        # 梯度上升
        for i in range(self.n_iter):
            gradients = gnf_func(z)
            z = z + self.step_size * gradients[..., :3]  # 只使用前三个维度
            
            # 清理不需要的张量
            del gradients
            torch.cuda.empty_cache()
        
        # 转换为numpy进行合并
        z_np = z.detach().cpu().numpy()
        
        # 合并点
        merged_points = self._merge_points(z_np)
        
        # 转换回tensor
        final_coords = torch.from_numpy(merged_points).to(device)
        
        return final_coords, atom_types

    def _find_nearest_atoms(self, points: torch.Tensor, atoms: torch.Tensor) -> torch.Tensor:
        """找到每个采样点最近的原子"""
        dist = torch.cdist(points, atoms) # x1：形状为(N,D)；x2：形状为(M,D)；返回值：形状为 (N,M) 的张量，每个元素 (i,j) 表示 x1[i] 和 x2[j] 之间的距离
        return torch.argmin(dist, dim=1)

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