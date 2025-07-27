import torch
import torch.nn as nn
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from typing import Callable, Optional, Tuple, List, Dict, Any
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN
import torch.nn.functional as F
from funcmol.utils.constants import PADDING_INDEX

class GNFConverter(nn.Module):
    """
    Converter between molecular structures and Gradient Neural Fields (GNF).
    
    The GNF is defined as r = f(X,z) where:
    - X is the set of atomic coordinates
    - z is a query point in 3D space
    - r is the gradient at point z
    """
    
    def __init__(self,
                sigma: float,
                n_query_points: int,
                n_iter: int,
                step_size: float,
                eps: float,  # DBSCAN的邻域半径参数
                min_samples: int,  # DBSCAN的最小样本数参数
                sigma_ratios: Dict[str, float],
                gradient_field_method: str = "softmax",  # 梯度场计算方法: "gaussian" 或 "softmax"
                temperature: float = 1.0,  # softmax温度参数，控制分布尖锐程度
                device: str = "cuda" if torch.cuda.is_available() else "cpu"):  # 原子类型比例系数
        super().__init__()
        self.sigma = sigma
        self.n_query_points = n_query_points
        self.n_iter = n_iter
        self.step_size = step_size
        self.eps = eps
        self.min_samples = min_samples
        self.device = device
        self.sigma_ratios = sigma_ratios
        self.gradient_field_method = gradient_field_method  # 保存梯度场计算方法
        self.temperature = temperature  # 保存temperature参数
        
        # 为不同类型的原子设置不同的 sigma 参数
        # 原子类型索引映射：0=C, 1=H, 2=O, 3=N, 4=F
        atom_type_mapping = {0: 'C', 1: 'H', 2: 'O', 3: 'N', 4: 'F'}
        self.sigma_params = {}
        for atom_idx, atom_symbol in atom_type_mapping.items():
            ratio = self.sigma_ratios.get(atom_symbol, 1.0)  # 默认比例为1.0
            self.sigma_params[atom_idx] = sigma * ratio
    
    def mol2gnf(self, coords: torch.Tensor, atom_types: torch.Tensor, 
                query_points: torch.Tensor) -> torch.Tensor:
        """
        将分子坐标和原子类型转换为GNF（梯度神经场）在查询点的向量场。
        梯度场定义为指向原子位置的向量场，使得在梯度上升时点会向原子位置移动。
        
        Args:
            coords: 原子坐标，shape为 [batch, n_atoms, 3] 或 [n_atoms, 3]
            atom_types: 原子类型，shape为 [batch, n_atoms] 或 [n_atoms]
            query_points: 查询点，shape为 [batch, n_points, 3] 或 [n_points, 3]
        
        Returns:
            vector_field: shape为 [batch, n_points, n_atom_types, 3] 的张量，
                其中 n_atom_types 通常为5（C、H、O、N、F），每个原子类型对应一个通道。
                始终保留batch维度。
        """
        # 兼容batch维度
        if coords.dim() == 2:
            coords = coords.unsqueeze(0)  # [1, N, 3]
        if atom_types.dim() == 1:
            atom_types = atom_types.unsqueeze(0)  # [1, N]
        if query_points.dim() == 2:
            query_points = query_points.unsqueeze(0)  # [1, M, 3]
            
        n_atom_types = 5  # 默认支持5种原子类型
        batch_size, n_points, _ = query_points.shape
        device = self.device
        vector_field = torch.zeros(batch_size, n_points, n_atom_types, 3, device=device)
        
        for b in range(batch_size):
            # 创建一维掩码
            mask = (atom_types[b] != PADDING_INDEX)  # [n_atoms]
            valid_coords = coords[b][mask]  # [n_valid_atoms, 3]
            valid_types = atom_types[b][mask].long()  # [n_valid_atoms]
            
            if valid_coords.size(0) == 0:
                continue
                
            for t in range(n_atom_types):
                type_mask = (valid_types == t)
                if type_mask.sum() > 0:
                    type_coords = valid_coords[type_mask]  # [n_type_atoms, 3]
                    # 使用当前原子类型特定的 sigma 参数
                    sigma = self.sigma_params.get(t, self.sigma)
                    # 计算当前类型原子的梯度场
                    coords_exp = type_coords.unsqueeze(1)      # (n_type_atoms, 1, 3)
                    q_exp = query_points[b].unsqueeze(0)       # (1, n_points, 3)
                    diff = coords_exp - q_exp                  # (n_type_atoms, n_points, 3)
                    dist_sq = torch.sum(diff ** 2, dim=-1, keepdim=True)  # (n_type_atoms, n_points, 1)
                    
                    if self.gradient_field_method == "gaussian":  # 高斯定义：基于距离的高斯权重
                        # 注意：这里不需要改变 diff 的符号，因为我们希望梯度指向原子位置
                        individual_gradients = diff * torch.exp(-dist_sq / (2 * sigma ** 2)) / (sigma ** 2)
                        type_gradients = torch.sum(individual_gradients, dim=0)  # (n_points, 3)
                    elif self.gradient_field_method == "softmax":  # 基于softmax的定义：使用温度控制的权重
                        distances = torch.sqrt(dist_sq.squeeze(-1))  # (n_type_atoms, n_points)
                        weights = torch.softmax(-distances / self.temperature, dim=0)  # (n_type_atoms, n_points)
                        weights = weights.unsqueeze(-1)  # (n_type_atoms, n_points, 1)
                        weighted_gradients = diff * weights  # (n_type_atoms, n_points, 3)
                        type_gradients = torch.sum(weighted_gradients, dim=0)  # (n_points, 3)
                    
                    vector_field[b, :, t, :] = type_gradients
        return vector_field

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
                for _ in range(self.n_iter):
                    # 在新位置计算梯度场
                    z_batch = z.unsqueeze(0)  # [1, n_query_points, 3]
                    current_field = decoder(z_batch, codes[b:b+1])  # [1, n_query_points, n_atom_types, 3]
                    grad = current_field[0, :, t, :]  # [n_query_points, 3]
                    
                    # 使用原子类型特定的sigma调整步长，保持与训练时的一致性
                    sigma_ratio = self.sigma_params.get(t, self.sigma) / self.sigma
                    adjusted_step_size = self.step_size * sigma_ratio
                    
                    # 更新采样点位置
                    z = z + adjusted_step_size * grad
                                    
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
        return final_coords, final_types # [batch, n_atoms, 3]



    def _merge_points(self, points: np.ndarray) -> np.ndarray:
        """Use DBSCAN to merge close points and determine atom centers."""
        if len(points.shape) == 3:
            points = points.reshape(-1, 3)
        elif len(points.shape) != 2 or points.shape[1] != 3:
            raise ValueError(f"Expected points shape (n, 3), got {points.shape}")

        if len(points) == 0:
            return points

        # 使用类的DBSCAN参数
        clustering = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(points)
        labels = clustering.labels_  # -1表示噪声点

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        print(f"[DBSCAN] Total points: {len(points)}, Clusters found: {n_clusters}, Noise points: {(labels == -1).sum()}")

        # 计算每个簇的中心
        merged = []
        for label in set(labels):
            if label == -1:
                continue  # 跳过噪声点
            cluster_points = points[labels == label]
            center = np.mean(cluster_points, axis=0)
            merged.append(center)

        return np.array(merged)

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