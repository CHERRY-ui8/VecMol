import os
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any, Union
import torch
import hydra
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import imageio.v2 as imageio
from omegaconf import OmegaConf
from lightning import Fabric
from torch_geometric.utils import to_dense_batch
from funcmol.utils.constants import PADDING_INDEX
from funcmol.utils.utils_nf import create_neural_field, load_neural_field, get_latest_model_path
from funcmol.utils.gnf_converter import GNFConverter
from funcmol.dataset.dataset_field import create_field_loaders, create_gnf_converter

# 设置项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(PROJECT_ROOT))

# 禁用 Dynamo 错误提示
torch._dynamo.config.suppress_errors = True

class MoleculeMetrics:
    """分子重建指标计算类，提供 RMSD、重建损失和 KL 散度计算方法。"""
    
    @staticmethod
    def compute_rmsd(coords1: torch.Tensor, coords2: torch.Tensor) -> torch.Tensor:
        """计算两个坐标集之间的对称 RMSD，保留梯度。

        Args:
            coords1: 第一个坐标集，形状 [N, 3]
            coords2: 第二个坐标集，形状 [M, 3]

        Returns:
            对称 RMSD 值，标量张量
        """
        coords1 = coords1.detach().requires_grad_(True)
        coords2 = coords2.detach().requires_grad_(True)
        
        dist1 = torch.sqrt(torch.sum((coords1.unsqueeze(1) - coords2.unsqueeze(0))**2, dim=2) + 1e-8)
        dist2 = torch.sqrt(torch.sum((coords2.unsqueeze(1) - coords1.unsqueeze(0))**2, dim=2) + 1e-8)
        
        min_dist1 = torch.min(dist1, dim=1)[0]
        min_dist2 = torch.min(dist2, dim=1)[0]
        
        return (torch.mean(min_dist1) + torch.mean(min_dist2)) / 2
    
    @staticmethod
    def compute_rmsd_scalar(coords1: torch.Tensor, coords2: torch.Tensor) -> float:
        """计算两个坐标集之间的对称 RMSD，返回标量值。

        Args:
            coords1: 第一个坐标集，形状 [N, 3]
            coords2: 第二个坐标集，形状 [M, 3]

        Returns:
            对称 RMSD 标量值
        """
        return MoleculeMetrics.compute_rmsd(coords1, coords2).item()
    
    @staticmethod
    def compute_reconstruction_loss(coords: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        """计算重建损失，保留梯度。

        Args:
            coords: 真实坐标，形状 [N, 3]
            points: 重建点，形状 [M, 3]

        Returns:
            重建损失值，标量张量
        """
        dist1 = torch.sum((coords.unsqueeze(1) - points.unsqueeze(0))**2, dim=2)
        eps = 1e-8
        min_dist_to_samples = torch.min(dist1 + eps, dim=1)[0]
        min_dist_to_atoms = torch.min(dist1 + eps, dim=0)[0]
        
        coverage_loss = torch.mean(min_dist_to_samples)
        clustering_loss = torch.mean(min_dist_to_atoms)
        
        return torch.sqrt(coverage_loss + 0.1 * clustering_loss)
    
    @staticmethod
    def compute_reconstruction_loss_scalar(coords: torch.Tensor, points: torch.Tensor) -> float:
        """计算重建损失，返回标量值。

        Args:
            coords: 真实坐标，形状 [N, 3]
            points: 重建点，形状 [M, 3]

        Returns:
            重建损失标量值
        """
        return MoleculeMetrics.compute_reconstruction_loss(coords, points).item()
    
    @staticmethod
    def compute_kl_divergences(coords1: torch.Tensor, coords2: torch.Tensor, temperature: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算两个坐标集之间的双向 KL 散度，保留梯度。

        Args:
            coords1: 第一个坐标集，形状 [N, 3]
            coords2: 第二个坐标集，形状 [M, 3]
            temperature: 温度参数，控制 softmax 锐度

        Returns:
            Tuple[KL_1to2, KL_2to1]，两个方向的 KL 散度
        """
        dist_matrix = torch.sum((coords1.unsqueeze(1) - coords2.unsqueeze(0))**2, dim=2)
        p1_given_2 = torch.softmax(-dist_matrix / temperature, dim=0)
        p2_given_1 = torch.softmax(-dist_matrix / temperature, dim=1)
        eps = 1e-8
        
        kl_1to2 = torch.mean(torch.sum(p1_given_2 * torch.log(p1_given_2 / (1.0/coords2.shape[0] + eps) + eps), dim=0))
        kl_2to1 = torch.mean(torch.sum(p2_given_1 * torch.log(p2_given_1 / (1.0/coords1.shape[0] + eps) + eps), dim=1))
        
        return kl_1to2, kl_2to1
    
    @staticmethod
    def compute_kl_divergences_scalar(coords1: torch.Tensor, coords2: torch.Tensor, temperature: float = 0.1) -> Tuple[float, float]:
        """计算两个坐标集之间的双向 KL 散度，返回标量值。

        Args:
            coords1: 第一个坐标集，形状 [N, 3]
            coords2: 第二个坐标集，形状 [M, 3]
            temperature: 温度参数，控制 softmax 锐度

        Returns:
            Tuple[KL_1to2, KL_2to1]，两个方向的 KL 散度标量值
        """
        kl_1to2, kl_2to1 = MoleculeMetrics.compute_kl_divergences(coords1, coords2, temperature)
        return kl_1to2.item(), kl_2to1.item()

class MoleculeVisualizer:
    """分子结构可视化基类，定义基本绘图属性和方法。"""
    
    def __init__(self):
        self.atom_colors = {
            0: 'black',  # C
            1: 'gray',   # H
            2: 'red',    # O
            3: 'blue',   # N
            4: 'green'   # F
        }
        plt.style.use('default')
        plt.rcParams['figure.dpi'] = 150
        plt.rcParams['savefig.dpi'] = 300
    
    def _get_atom_colors(self, atom_types: torch.Tensor) -> List[str]:
        """根据原子类型获取颜色列表。

        Args:
            atom_types: 原子类型张量，形状 [N]

        Returns:
            颜色列表，与原子类型对应
        """
        return [self.atom_colors.get(int(atom_type), 'gray') for atom_type in atom_types]
    
    def _setup_3d_axis(self, ax: plt.Axes, coords_list: List[np.ndarray], margin: float = 0.5):
        """设置 3D 坐标轴的通用属性 - 优化版本。

        Args:
            ax: 3D 绘图轴
            coords_list: 坐标数组列表，用于确定轴范围
            margin: 坐标轴边距
        """
        ax.set_xlabel('X (Å)')
        ax.set_ylabel('Y (Å)')
        ax.set_zlabel('Z (Å)')
        ax.grid(True, alpha=0.3)
        
        all_coords = np.vstack(coords_list)
        x_min, x_max = all_coords[:, 0].min() - margin, all_coords[:, 0].max() + margin
        y_min, y_max = all_coords[:, 1].min() - margin, all_coords[:, 1].max() + margin
        z_min, z_max = all_coords[:, 2].min() - margin, all_coords[:, 2].max() + margin
        
        x_range = x_max - x_min
        y_range = y_max - y_min
        z_range = z_max - z_min
        
        print(f"Axis ranges: X[{x_min:.3f}, {x_max:.3f}], Y[{y_min:.3f}, {y_max:.3f}], Z[{z_min:.3f}, {z_max:.3f}]")
        
        if z_range < max(x_range, y_range) * 0.3:
            z_center = (z_min + z_max) / 2
            new_z_range = max(x_range, y_range) * 0.5
            z_min = z_center - new_z_range / 2
            z_max = z_center + new_z_range / 2
            print(f"Extended Z range to: [{z_min:.3f}, {z_max:.3f}]")
        
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)
        ax.set_box_aspect([1, 1, 1])

class GNFVisualizer(MoleculeVisualizer):
    """GNF 重建过程可视化类，生成分子对比图和重建动画。"""
    
    def __init__(self, output_dir: str = "gnf_reconstruction_results"):
        """初始化可视化器。

        Args:
            output_dir: 输出文件保存目录
        """
        super().__init__()
        self.output_dir = output_dir
        self.metrics = MoleculeMetrics()
        os.makedirs(output_dir, exist_ok=True)
    
    def visualize_molecule_comparison(self, 
                                     orig_coords: torch.Tensor, 
                                     recon_coords: torch.Tensor,
                                     orig_types: Optional[torch.Tensor] = None,
                                     recon_types: Optional[torch.Tensor] = None,
                                     save_path: Optional[str] = None,
                                     title: str = "Molecule Comparison"):
        """可视化原始分子和重建分子的对比。

        Args:
            orig_coords: 原始分子坐标，形状 [N, 3]
            recon_coords: 重建分子坐标，形状 [M, 3]
            orig_types: 原始原子类型，形状 [N]
            recon_types: 重建原子类型，形状 [M]
            save_path: 保存路径，若为 None 则显示图像
            title: 图像标题

        Returns:
            matplotlib Figure 对象
        """
        fig = plt.figure(figsize=(15, 7))
        
        ax1 = fig.add_subplot(131, projection='3d')
        orig_coords_np = orig_coords.detach().cpu().numpy()
        
        if orig_types is not None:
            orig_types_np = orig_types.detach().cpu().numpy()
            for atom_type in range(5):
                mask = (orig_types_np == atom_type)
                if mask.sum() > 0:
                    ax1.scatter(orig_coords_np[mask, 0], orig_coords_np[mask, 1], orig_coords_np[mask, 2], 
                               c=self.atom_colors[atom_type], marker='o', s=100, 
                               label=f'Original {["C", "H", "O", "N", "F"][atom_type]}')
        else:
            ax1.scatter(orig_coords_np[:, 0], orig_coords_np[:, 1], orig_coords_np[:, 2], 
                       c='blue', marker='o', s=100, label='Original', alpha=0.8)
        ax1.set_title("Original Molecule")
        ax1.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0))
        
        ax2 = fig.add_subplot(132, projection='3d')
        recon_coords_np = recon_coords.detach().cpu().numpy()
        
        if recon_types is not None:
            recon_types_np = recon_types.detach().cpu().numpy()
            for atom_type in range(5):
                mask = (recon_types_np == atom_type)
                if mask.sum() > 0:
                    ax2.scatter(recon_coords_np[mask, 0], recon_coords_np[mask, 1], recon_coords_np[mask, 2], 
                               c=self.atom_colors[atom_type], marker='o', s=100, 
                               label=f'Reconstructed {["C", "H", "O", "N", "F"][atom_type]}')
        else:
            ax2.scatter(recon_coords_np[:, 0], recon_coords_np[:, 1], recon_coords_np[:, 2], 
                       c='red', marker='o', s=100, label='Reconstructed', alpha=0.8)
        ax2.set_title("Reconstructed Molecule")
        ax2.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0))
        
        ax3 = fig.add_subplot(133, projection='3d')
        
        if orig_types is not None:
            for atom_type in range(5):
                mask = (orig_types_np == atom_type)
                if mask.sum() > 0:
                    ax3.scatter(orig_coords_np[mask, 0], orig_coords_np[mask, 1], orig_coords_np[mask, 2], 
                               c=self.atom_colors[atom_type], marker='o', s=100, alpha=0.5, 
                               label=f'Original {["C", "H", "O", "N", "F"][atom_type]}')
        
        if recon_types is not None:
            for atom_type in range(5):
                mask = (recon_types_np == atom_type)
                if mask.sum() > 0:
                    ax3.scatter(recon_coords_np[mask, 0], recon_coords_np[mask, 1], recon_coords_np[mask, 2], 
                               c=self.atom_colors[atom_type], marker='s', s=100, alpha=0.5, 
                               label=f'Reconstructed {["C", "H", "O", "N", "F"][atom_type]}')
        else:
            ax3.scatter(recon_coords_np[:, 0], recon_coords_np[:, 1], recon_coords_np[:, 2], 
                       c='red', marker='s', s=100, alpha=0.5, label='Reconstructed')
        ax3.set_title("Comparison")
        ax3.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0))
        
        coords_list = [orig_coords_np, recon_coords_np]
        for ax in [ax1, ax2, ax3]:
            self._setup_3d_axis(ax, coords_list, margin=1.0)
            ax.view_init(elev=30, azim=60)
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        
        return fig
    
    def visualize_reconstruction_step(self, 
                                     coords: torch.Tensor,
                                     current_points: torch.Tensor,
                                     iteration: int,
                                     save_path: str,
                                     coords_types: Optional[torch.Tensor] = None,
                                     points_types: Optional[torch.Tensor] = None):
        """可视化重建过程的单个步骤 - 优化版本"""
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        coords_np = coords.detach().cpu().numpy()
        if coords_types is not None:
            coords_types_np = coords_types.detach().cpu().numpy()
            for atom_type in range(5):
                mask = (coords_types_np == atom_type)
                if mask.sum() > 0:
                    ax.scatter(coords_np[mask, 0], coords_np[mask, 1], coords_np[mask, 2], 
                              c=self.atom_colors[atom_type], marker='o', s=100, 
                              label=f'Original {["C", "H", "O", "N", "F"][atom_type]}', alpha=0.7)
        else:
            ax.scatter(coords_np[:, 0], coords_np[:, 1], coords_np[:, 2], 
                      c='blue', marker='o', s=100, label='Original', alpha=0.8)
        
        points_np = current_points.detach().cpu().numpy()
        if points_types is not None:
            points_types_np = points_types.detach().cpu().numpy()
            for atom_type in range(5):
                mask = (points_types_np == atom_type)
                if mask.sum() > 0:
                    ax.scatter(points_np[mask, 0], points_np[mask, 1], points_np[mask, 2],
                              c=self.atom_colors[atom_type], marker='.', s=20, 
                              label=f'Current {["C", "H", "O", "N", "F"][atom_type]}', alpha=0.5)
        else:
            ax.scatter(points_np[:, 0], points_np[:, 1], points_np[:, 2], 
                      c='red', marker='.', s=20, label='Current Points', alpha=0.5)
        
        coords_list = [coords_np, points_np]
        self._setup_3d_axis(ax, coords_list, margin=1.0)
        
        ax.view_init(elev=30, azim=60)
        ax.set_box_aspect([1, 1, 1])
        
        ax.set_title(f"Iteration {iteration}")
        ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0))
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_reconstruction_animation(self,
                                       gt_coords: torch.Tensor,
                                       gt_types: torch.Tensor,
                                       converter: GNFConverter,
                                       decoder: torch.nn.Module,
                                       codes: torch.Tensor,
                                       save_interval: int = 50,
                                       animation_name: str = "reconstruction",
                                       field_option: str = "predicted_field",
                                       sample_idx: int = 0) -> Dict[str, Any]:
        """创建分子重建过程的动画

        Args:
            gt_coords: 真实分子坐标，形状 [batch, n_atoms, 3]
            gt_types: 真实原子类型，形状 [batch, n_atoms]
            converter: GNF 转换器
            decoder: 解码器模型
            codes: 编码器输出的潜在表示，形状 [batch, code_dim]
            save_interval: 保存帧的间隔
            animation_name: 动画文件名前缀
            field_option: 梯度场类型（"gt_field" 或 "predicted_field"）
            sample_idx: 样本索引

        Returns:
            包含动画路径、对比图路径、指标历史和最终结果的字典
        """
        device = gt_coords.device
        
        gt_mask = (gt_types[sample_idx] != PADDING_INDEX)
        gt_valid_coords = gt_coords[sample_idx][gt_mask]
        gt_valid_types = gt_types[sample_idx][gt_mask]
        
        print(f"\nStarting reconstruction for molecule {sample_idx}")
        print(f"Ground truth atoms: {len(gt_valid_coords)}")
        print(f"Atom types: {gt_valid_types.tolist()}")
        print(f"Ground truth coords range: "
              f"X[{gt_valid_coords[:, 0].min():.3f}, {gt_valid_coords[:, 0].max():.3f}], "
              f"Y[{gt_valid_coords[:, 1].min():.3f}, {gt_valid_coords[:, 1].max():.3f}], "
              f"Z[{gt_valid_coords[:, 2].min():.3f}, {gt_valid_coords[:, 2].max():.3f}]")
        
        all_atom_types = list(range(5))
        
        coords_min = gt_valid_coords.min(dim=0)[0]
        coords_max = gt_valid_coords.max(dim=0)[0]
        coords_range = coords_max - coords_min
        margin = coords_range * 0.5
        init_min = coords_min - margin
        init_max = coords_max + margin
        
        if (init_max[2] - init_min[2]) < 1.0:
            z_center = (init_max[2] + init_min[2]) / 2
            init_min[2] = z_center - 0.5
            init_max[2] = z_center + 0.5
        
        print(f"Initial point range: "
              f"X[{init_min[0]:.3f}, {init_max[0]:.3f}], "
              f"Y[{init_min[1]:.3f}, {init_max[1]:.3f}], "
              f"Z[{init_min[2]:.3f}, {init_max[2]:.3f}]")
        
        z_dict = {}
        for atom_type in all_atom_types:
            z_dict[atom_type] = torch.rand(converter.n_query_points, 3, device=device) * (init_max - init_min) + init_min
            print(f"Initial points for atom type {atom_type}: "
                  f"Z[{z_dict[atom_type][:, 2].min():.3f}, {z_dict[atom_type][:, 2].max():.3f}]")
        
        def gnf_func(points):
            if field_option == "gt_field":
                vector_field = converter.mol2gnf(
                    gt_valid_coords.unsqueeze(0),
                    gt_valid_types.unsqueeze(0),
                    points.unsqueeze(0)
                )[0]
            else:
                vector_field = decoder(points.unsqueeze(0), codes[sample_idx:sample_idx+1])[0]
            
            for atom_type in all_atom_types:
                z_grad_range = vector_field[:, atom_type, 2].abs().mean().item()
                print(f"Atom type {atom_type}: Mean Z gradient magnitude = {z_grad_range:.6f}")
            return vector_field
        
        frame_paths = []
        metrics_history = {
            'iterations': [],
            'loss': [],
            'rmsd': [],
            'kl_1to2': [],
            'kl_2to1': []
        }
        
        for i in range(converter.n_iter):
            with torch.no_grad():
                for atom_type in all_atom_types:
                    z = z_dict[atom_type]
                    gradients = gnf_func(z)
                    type_gradients = gradients[:, atom_type, :]
                    
                    if i % 1000 == 0:
                        grad_norm = torch.norm(type_gradients, dim=1).mean().item()
                        z_range = z.max(dim=0)[0] - z.min(dim=0)[0]
                        print(f"Iteration {i}, Atom type {atom_type}: grad norm = {grad_norm:.6f}, "
                              f"Z range = {z_range[2]:.6f}")
                    
                    z_dict[atom_type] = z + torch.tensor(converter.step_size, device=z.device) * type_gradients
            
            if i % save_interval == 0 or i == converter.n_iter - 1:
                frame_path = os.path.join(self.output_dir, f"frame_sample_{sample_idx}_{i:04d}.png")
                
                all_points = []
                all_types = []
                for atom_type in all_atom_types:
                    points = z_dict[atom_type]
                    if len(points) > 0:
                        all_points.append(points)
                        all_types.extend([atom_type] * len(points))
                
                if all_points:
                    current_points = torch.cat(all_points, dim=0)
                    current_types = torch.tensor(all_types, device=device)
                else:
                    current_points = torch.empty((0, 3), device=device)
                    current_types = torch.empty((0,), device=device)
                
                print(f"Frame {i}: Z range = {current_points[:, 2].max() - current_points[:, 2].min():.6f}")
                
                self.visualize_reconstruction_step(
                    gt_valid_coords, current_points, i, frame_path, 
                    gt_valid_types, current_types
                )
                frame_paths.append(frame_path)
                
                if len(current_points) > 0:
                    metrics_history['iterations'].append(i)
                    metrics_history['loss'].append(
                        self.metrics.compute_reconstruction_loss_scalar(gt_valid_coords, current_points)
                    )
                    metrics_history['rmsd'].append(
                        self.metrics.compute_rmsd_scalar(gt_valid_coords, current_points)
                    )
                    kl_1to2, kl_2to1 = self.metrics.compute_kl_divergences_scalar(
                        gt_valid_coords, current_points
                    )
                    metrics_history['kl_1to2'].append(kl_1to2)
                    metrics_history['kl_2to1'].append(kl_2to1)
        
        gradient_field_method = getattr(converter, 'gradient_field_method', 'unknown')
        
        gif_path = os.path.join(self.output_dir, f"{animation_name}_{field_option}_{gradient_field_method}.gif")
        with imageio.get_writer(gif_path, mode='I', duration=0.1, fps=15, loop=0) as writer:
            for frame_path in frame_paths:
                frame = imageio.imread(frame_path)
                writer.append_data(frame)
                os.remove(frame_path)
        
        final_points = []
        final_types = []
        for atom_type in all_atom_types:
            points = z_dict[atom_type].detach().cpu().numpy()
            if len(points) > 0:
                merged_points = converter._merge_points(points)
                if len(merged_points) > 0:
                    final_points.append(torch.from_numpy(merged_points).to(device))
                    final_types.extend([atom_type] * len(merged_points))
        
        if final_points:
            final_points = torch.cat(final_points, dim=0)
            final_types = torch.tensor(final_types, device=device)
        else:
            final_points = torch.empty((0, 3), device=device)
            final_types = torch.empty((0,), device=device)
        
        comparison_path = os.path.join(self.output_dir, f"{animation_name}_{field_option}_{gradient_field_method}_final.png")
        self.visualize_molecule_comparison(
            gt_valid_coords,
            final_points,
            gt_valid_types,
            final_types,
            save_path=comparison_path
        )
        
        return {
            'gif_path': gif_path,
            'comparison_path': comparison_path,
            'metrics_history': metrics_history,
            'final_points': final_points,
            'final_types': final_types,
            'final_rmsd': metrics_history['rmsd'][-1] if metrics_history['rmsd'] else float('inf'),
            'final_loss': metrics_history['loss'][-1] if metrics_history['loss'] else float('inf'),
            'final_kl_1to2': metrics_history['kl_1to2'][-1] if metrics_history['kl_1to2'] else float('inf'),
            'final_kl_2to1': metrics_history['kl_2to1'][-1] if metrics_history['kl_2to1'] else float('inf')
        }

def visualize_reconstruction(gt_coords: torch.Tensor,
                            gt_types: torch.Tensor,
                            converter: GNFConverter,
                            decoder: torch.nn.Module,
                            codes: torch.Tensor,
                            output_dir: str = "gnf_reconstruction_results",
                            sample_idx: int = 0,
                            field_option: str = "predicted_field",
                            gradient_field_method: Optional[str] = None) -> Dict[str, Any]:
    """便捷函数：可视化分子重建过程。

    Args:
        gt_coords: 真实分子坐标，形状 [batch, n_atoms, 3]
        gt_types: 真实原子类型，形状 [batch, n_atoms]
        converter: GNF 转换器
        decoder: 解码器模型
        codes: 编码器输出的潜在表示，形状 [batch, code_dim]
        output_dir: 输出文件保存目录
        sample_idx: 样本索引
        field_option: 梯度场类型（"gt_field" 或 "predicted_field"）
        gradient_field_method: 梯度场计算方法

    Returns:
        包含动画路径、对比图路径、指标历史和最终结果的字典
    """
    visualizer = GNFVisualizer(output_dir)
    if gradient_field_method is not None:
        converter.gradient_field_method = gradient_field_method
    return visualizer.create_reconstruction_animation(
        gt_coords, gt_types, converter, decoder, codes,
        save_interval=100,
        animation_name=f"recon_sample_{sample_idx}",
        field_option=field_option,
        sample_idx=sample_idx
    )

def visualize_1d_gradient_field_comparison(
    gt_coords: torch.Tensor,
    gt_types: torch.Tensor,
    converter: GNFConverter,
    decoder: torch.nn.Module,
    codes: torch.Tensor,
    sample_idx: int = 0,
    atom_type: int = 0,
    x_range: Optional[tuple] = None,
    n_points: int = 3000,
    y_coord: float = 0.0,
    z_coord: float = 0.0,
    save_path: Optional[str] = None,
    gradient_field_method: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """可视化一维方向上的梯度场对比（真实 vs 预测）。

    Args:
        gt_coords: 真实分子坐标，形状 [batch, n_atoms, 3]
        gt_types: 真实原子类型，形状 [batch, n_atoms]
        converter: GNF 转换器
        decoder: 解码器模型
        codes: 编码器输出的潜在表示，形状 [batch, code_dim]
        sample_idx: 样本索引
        atom_type: 原子类型（0=C, 1=H, 2=O, 3=N, 4=F）
        x_range: x 轴范围，None 时自动计算
        n_points: 采样点数
        y_coord: y 坐标固定值
        z_coord: z 坐标固定值
        save_path: 保存路径，None 时显示图像
        gradient_field_method: 梯度场计算方法

    Returns:
        包含梯度场统计信息和数据的字典，若无目标原子类型则返回 None
    """
    device = gt_coords.device
    if gradient_field_method is not None:
        converter.gradient_field_method = gradient_field_method
    
    gt_mask = (gt_types[sample_idx] != PADDING_INDEX)
    gt_valid_coords = gt_coords[sample_idx][gt_mask]
    gt_valid_types = gt_types[sample_idx][gt_mask]
    
    target_atoms = gt_valid_types == atom_type
    if target_atoms.sum() == 0:
        print(f"警告：样本 {sample_idx} 中没有类型为 {['C', 'H', 'O', 'N', 'F'][atom_type]} 的原子")
        return None
    
    if x_range is None:
        x_min = gt_valid_coords[:, 0].min().item()
        x_max = gt_valid_coords[:, 0].max().item()
        margin = (x_max - x_min) * 0.2
        x_range = (x_min - margin, x_max + margin)
        print(f"自动计算 x 轴范围: {x_range}")
    
    x = torch.linspace(x_range[0], x_range[1], n_points, device=device)
    query_points = torch.zeros(n_points, 3, device=device)
    query_points[:, 0], query_points[:, 1], query_points[:, 2] = x, y_coord, z_coord
    
    with torch.no_grad():
        gt_field = converter.mol2gnf(
            gt_valid_coords.unsqueeze(0),
            gt_valid_types.unsqueeze(0),
            query_points.unsqueeze(0)
        )
        gt_gradients_3d = gt_field[0, :, atom_type, :]
        gt_gradients = torch.norm(gt_gradients_3d, dim=1)
        gt_gradients_x, gt_gradients_y, gt_gradients_z = gt_gradients_3d[:, 0], gt_gradients_3d[:, 1], gt_gradients_3d[:, 2]
        
        pred_field = decoder(query_points.unsqueeze(0), codes[sample_idx:sample_idx+1])
        pred_gradients_3d = pred_field[0, :, atom_type, :]
        pred_gradients = torch.norm(pred_gradients_3d, dim=1)
        pred_gradients_x, pred_gradients_y, pred_gradients_z = pred_gradients_3d[:, 0], pred_gradients_3d[:, 1], pred_gradients_3d[:, 2]
    
    fig, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8)) = plt.subplots(2, 4, figsize=(24, 12))
    
    ax1.plot(x.cpu().numpy(), gt_gradients.cpu().numpy(), 
             label=f'Ground Truth ({["C", "H", "O", "N", "F"][atom_type]})', 
             linewidth=2, color='blue', alpha=0.8)
    ax1.plot(x.cpu().numpy(), pred_gradients.cpu().numpy(), 
             label=f'Predicted ({["C", "H", "O", "N", "F"][atom_type]})', 
             linewidth=2, color='red', alpha=0.8, linestyle='--')
    
    atom_positions = gt_valid_coords[target_atoms, 0].cpu().numpy()
    if len(atom_positions) > 0:
        for i, pos in enumerate(atom_positions):
            ax1.axvline(x=pos, color='green', linestyle=':', alpha=0.7, linewidth=1.5)
            ax1.text(pos, ax1.get_ylim()[1] * 0.9, f'{["C", "H", "O", "N", "F"][atom_type]}{i+1}', 
                     rotation=90, verticalalignment='top', fontsize=8)
    
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=0.5)
    ax1.set_xlabel('X Position (Å)', fontsize=12)
    ax1.set_ylabel('Gradient Field Magnitude', fontsize=12)
    ax1.set_title(f'Gradient Field Magnitude - {["C", "H", "O", "N", "F"][atom_type]} Atoms\n'
                  f'Sample {sample_idx}, Line: y={y_coord:.2f}, z={z_coord:.2f}', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11)
    
    mse = torch.mean((gt_gradients - pred_gradients) ** 2).item()
    mae = torch.mean(torch.abs(gt_gradients - pred_gradients)).item()
    ax1.text(0.02, 0.98, f'MSE: {mse:.6f}\nMAE: {mae:.6f}', 
             transform=ax1.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), fontsize=10)
    
    ax2.plot(x.cpu().numpy(), gt_gradients_x.cpu().numpy(), 
             label=f'Ground Truth X ({["C", "H", "O", "N", "F"][atom_type]})', 
             linewidth=2, color='blue', alpha=0.8)
    ax2.plot(x.cpu().numpy(), pred_gradients_x.cpu().numpy(), 
             label=f'Predicted X ({["C", "H", "O", "N", "F"][atom_type]})', 
             linewidth=2, color='red', alpha=0.8, linestyle='--')
    
    if len(atom_positions) > 0:
        for i, pos in enumerate(atom_positions):
            ax2.axvline(x=pos, color='green', linestyle=':', alpha=0.7, linewidth=1.5)
    
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=0.5)
    ax2.set_xlabel('X Position (Å)', fontsize=12)
    ax2.set_ylabel('X Component of Gradient Field', fontsize=12)
    ax2.set_title(f'X Direction Component - {["C", "H", "O", "N", "F"][atom_type]} Atoms\n'
                  f'(Positive = Right, Negative = Left)', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=11)
    
    mse_x = torch.mean((gt_gradients_x - pred_gradients_x) ** 2).item()
    mae_x = torch.mean(torch.abs(gt_gradients_x - pred_gradients_x)).item()
    ax2.text(0.02, 0.98, f'MSE: {mse_x:.6f}\nMAE: {mae_x:.6f}', 
             transform=ax2.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8), fontsize=10)
    
    ax3.plot(x.cpu().numpy(), gt_gradients_y.cpu().numpy(), 
             label=f'Ground Truth Y ({["C", "H", "O", "N", "F"][atom_type]})', 
             linewidth=2, color='blue', alpha=0.8)
    ax3.plot(x.cpu().numpy(), pred_gradients_y.cpu().numpy(), 
             label=f'Predicted Y ({["C", "H", "O", "N", "F"][atom_type]})', 
             linewidth=2, color='red', alpha=0.8, linestyle='--')
    
    if len(atom_positions) > 0:
        for i, pos in enumerate(atom_positions):
            ax3.axvline(x=pos, color='green', linestyle=':', alpha=0.7, linewidth=1.5)
    
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=0.5)
    ax3.set_xlabel('X Position (Å)', fontsize=12)
    ax3.set_ylabel('Y Component of Gradient Field', fontsize=12)
    ax3.set_title(f'Y Direction Component - {["C", "H", "O", "N", "F"][atom_type]} Atoms\n'
                  f'(Positive = Up, Negative = Down)', fontsize=14)
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=11)
    
    mse_y = torch.mean((gt_gradients_y - pred_gradients_y) ** 2).item()
    mae_y = torch.mean(torch.abs(gt_gradients_y - pred_gradients_y)).item()
    ax3.text(0.02, 0.98, f'MSE: {mse_y:.6f}\nMAE: {mae_y:.6f}', 
             transform=ax3.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8), fontsize=10)
    
    ax4.plot(x.cpu().numpy(), gt_gradients_z.cpu().numpy(), 
             label=f'Ground Truth Z ({["C", "H", "O", "N", "F"][atom_type]})', 
             linewidth=2, color='blue', alpha=0.8)
    ax4.plot(x.cpu().numpy(), pred_gradients_z.cpu().numpy(), 
             label=f'Predicted Z ({["C", "H", "O", "N", "F"][atom_type]})', 
             linewidth=2, color='red', alpha=0.8, linestyle='--')
    
    if len(atom_positions) > 0:
        for i, pos in enumerate(atom_positions):
            ax4.axvline(x=pos, color='green', linestyle=':', alpha=0.7, linewidth=1.5)
    
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=0.5)
    ax4.set_xlabel('X Position (Å)', fontsize=12)
    ax4.set_ylabel('Z Component of Gradient Field', fontsize=12)
    ax4.set_title(f'Z Direction Component - {["C", "H", "O", "N", "F"][atom_type]} Atoms\n'
                  f'(Positive = Forward, Negative = Backward)', fontsize=14)
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=11)
    
    mse_z = torch.mean((gt_gradients_z - pred_gradients_z) ** 2).item()
    mae_z = torch.mean(torch.abs(gt_gradients_z - pred_gradients_z)).item()
    ax4.text(0.02, 0.98, f'MSE: {mse_z:.6f}\nMAE: {mae_z:.6f}', 
             transform=ax4.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8), fontsize=10)
    
    error = gt_gradients - pred_gradients
    ax5.plot(x.cpu().numpy(), error.cpu().numpy(), 
             label='Magnitude Error (GT - Pred)', 
             linewidth=2, color='purple', alpha=0.8)
    
    if len(atom_positions) > 0:
        for i, pos in enumerate(atom_positions):
            ax5.axvline(x=pos, color='green', linestyle=':', alpha=0.7, linewidth=1.5)
    
    ax5.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=0.5)
    ax5.set_xlabel('X Position (Å)', fontsize=12)
    ax5.set_ylabel('Magnitude Error', fontsize=12)
    ax5.set_title(f'Magnitude Error Analysis - {["C", "H", "O", "N", "F"][atom_type]} Atoms', fontsize=14)
    ax5.grid(True, alpha=0.3)
    ax5.legend(fontsize=11)
    
    error_mean = torch.mean(error).item()
    error_std = torch.std(error).item()
    ax5.text(0.02, 0.98, f'Error Mean: {error_mean:.6f}\nError Std: {error_std:.6f}', 
             transform=ax5.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8), fontsize=10)
    
    error_x = gt_gradients_x - pred_gradients_x
    ax6.plot(x.cpu().numpy(), error_x.cpu().numpy(), 
             label='X Component Error (GT - Pred)', 
             linewidth=2, color='orange', alpha=0.8)
    
    if len(atom_positions) > 0:
        for i, pos in enumerate(atom_positions):
            ax6.axvline(x=pos, color='green', linestyle=':', alpha=0.7, linewidth=1.5)
    
    ax6.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=0.5)
    ax6.set_xlabel('X Position (Å)', fontsize=12)
    ax6.set_ylabel('X Component Error', fontsize=12)
    ax6.set_title(f'X Direction Error Analysis - {["C", "H", "O", "N", "F"][atom_type]} Atoms', fontsize=14)
    ax6.grid(True, alpha=0.3)
    ax6.legend(fontsize=11)
    
    error_x_mean = torch.mean(error_x).item()
    error_x_std = torch.std(error_x).item()
    ax6.text(0.02, 0.98, f'Error Mean: {error_x_mean:.6f}\nError Std: {error_x_std:.6f}', 
             transform=ax6.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8), fontsize=10)
    
    error_y = gt_gradients_y - pred_gradients_y
    ax7.plot(x.cpu().numpy(), error_y.cpu().numpy(), 
             label='Y Component Error (GT - Pred)', 
             linewidth=2, color='purple', alpha=0.8)
    
    if len(atom_positions) > 0:
        for i, pos in enumerate(atom_positions):
            ax7.axvline(x=pos, color='green', linestyle=':', alpha=0.7, linewidth=1.5)
    
    ax7.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=0.5)
    ax7.set_xlabel('X Position (Å)', fontsize=12)
    ax7.set_ylabel('Y Component Error', fontsize=12)
    ax7.set_title(f'Y Direction Error Analysis - {["C", "H", "O", "N", "F"][atom_type]} Atoms', fontsize=14)
    ax7.grid(True, alpha=0.3)
    ax7.legend(fontsize=11)
    
    error_y_mean = torch.mean(error_y).item()
    error_y_std = torch.std(error_y).item()
    ax7.text(0.02, 0.98, f'Error Mean: {error_y_mean:.6f}\nError Std: {error_y_std:.6f}', 
             transform=ax7.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8), fontsize=10)
    
    error_z = gt_gradients_z - pred_gradients_z
    ax8.plot(x.cpu().numpy(), error_z.cpu().numpy(), 
             label='Z Component Error (GT - Pred)', 
             linewidth=2, color='brown', alpha=0.8)
    
    if len(atom_positions) > 0:
        for i, pos in enumerate(atom_positions):
            ax8.axvline(x=pos, color='green', linestyle=':', alpha=0.7, linewidth=1.5)
    
    ax8.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=0.5)
    ax8.set_xlabel('X Position (Å)', fontsize=12)
    ax8.set_ylabel('Z Component Error', fontsize=12)
    ax8.set_title(f'Z Direction Error Analysis - {["C", "H", "O", "N", "F"][atom_type]} Atoms', fontsize=14)
    ax8.grid(True, alpha=0.3)
    ax8.legend(fontsize=11)
    
    error_z_mean = torch.mean(error_z).item()
    error_z_std = torch.std(error_z).item()
    ax8.text(0.02, 0.98, f'Error Mean: {error_z_mean:.6f}\nError Std: {error_z_std:.6f}', 
             transform=ax8.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8), fontsize=10)
    
    plt.tight_layout()
    
    gradient_field_method = getattr(converter, 'gradient_field_method', 'unknown')
    if save_path:
        atom_name = ["C", "H", "O", "N", "F"][atom_type]
        base_name = f"gradient_field_comparison_sample_{sample_idx}_atom_{atom_name}_{gradient_field_method}"
        save_path = os.path.join(os.path.dirname(save_path) or '.', f"{base_name}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Gradient field comparison (atom_type={atom_name}, {gradient_field_method}) saved to: {save_path}")
    else:
        plt.show()
    
    return {
        'save_path': save_path if save_path else None,
        'mse': mse,
        'mae': mae,
        'mse_x': mse_x,
        'mae_x': mae_x,
        'mse_y': mse_y,
        'mae_y': mae_y,
        'mse_z': mse_z,
        'mae_z': mae_z,
        'error_mean': error_mean,
        'error_std': error_std,
        'error_x_mean': error_x_mean,
        'error_x_std': error_x_std,
        'error_y_mean': error_y_mean,
        'error_y_std': error_y_std,
        'error_z_mean': error_z_mean,
        'error_z_std': error_z_std,
        'gt_gradients': gt_gradients.cpu().numpy(),
        'pred_gradients': pred_gradients.cpu().numpy(),
        'gt_gradients_x': gt_gradients_x.cpu().numpy(),
        'pred_gradients_x': pred_gradients_x.cpu().numpy(),
        'gt_gradients_y': gt_gradients_y.cpu().numpy(),
        'pred_gradients_y': pred_gradients_y.cpu().numpy(),
        'gt_gradients_z': gt_gradients_z.cpu().numpy(),
        'pred_gradients_z': pred_gradients_z.cpu().numpy(),
        'error': error.cpu().numpy(),
        'error_x': error_x.cpu().numpy(),
        'error_y': error_y.cpu().numpy(),
        'error_z': error_z.cpu().numpy(),
        'x_positions': x.cpu().numpy(),
        'atom_positions': atom_positions
    }

def setup_environment(devices: str = "1", accelerator: str = "cpu", precision: str = "32-true") -> Tuple[Fabric, torch.device]:
    """初始化运行环境。

    Args:
        devices: CUDA 设备编号，默认为 "1"
        accelerator: 加速器类型，默认为 "cpu"
        precision: 计算精度，默认为 "32-true"

    Returns:
        Tuple[Fabric, torch.device]，Fabric 实例和设备
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = devices
    fabric = Fabric(accelerator=accelerator, devices=1, precision=precision, strategy="auto")
    fabric.launch()
    device = torch.device(accelerator)
    print(f"Using device: {device}")
    return fabric, device

def load_config(config_name: str = "train_nf_qm9") -> OmegaConf:
    """加载配置文件。

    Args:
        config_name: 配置文件名，默认为 "train_nf_qm9"

    Returns:
        OmegaConf 配置对象
    """
    config_path = PROJECT_ROOT / "configs" / f"{config_name}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with hydra.initialize_config_dir(config_dir=str(config_path.parent), version_base=None):
        config = hydra.compose(config_name=config_name)
    
    config["dset"]["data_dir"] = str(PROJECT_ROOT / "funcmol" / "dataset" / "data")
    print(f"Dataset directory: {config['dset']['data_dir']}")
    return config

def load_model(fabric: Fabric, config: OmegaConf, model_dir: Optional[str] = None) -> Tuple[torch.nn.Module, torch.nn.Module]:
    """加载预训练模型。

    Args:
        fabric: Fabric 实例
        config: 配置对象
        model_dir: 模型检查点所在的文件夹路径，若为 None 则自动选择最新的模型

    Returns:
        Tuple[encoder, decoder]，编码器和解码器模型
    """
    if model_dir is None:
        dset_name = config["dset"]["dset_name"]
        model_prefix = f"nf_{dset_name}_" if dset_name in ["drugs", "qm9"] else "nf_"
        latest_exp_dir = get_latest_model_path("exps/neural_field", model_prefix)
        model_path = Path(latest_exp_dir) / "model.pt"
    else:
        model_path = Path(model_dir) / "model.pt"
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
    
    print(f"Loading model from: {model_path}")
    enc, dec = create_neural_field(config, fabric)
    
    if hasattr(enc, '_orig_mod'):
        enc = enc._orig_mod
    if hasattr(dec, '_orig_mod'):
        dec = dec._orig_mod
    
    checkpoint = fabric.load(str(model_path))
    enc = load_neural_field(checkpoint, fabric, config)[0]
    dec = load_neural_field(checkpoint, fabric, config)[1]
    print("Model loaded successfully!")
    
    return enc, dec

def create_converter(config: OmegaConf, device: torch.device) -> GNFConverter:
    """创建 GNF 转换器。

    Args:
        config: 配置对象
        device: 计算设备

    Returns:
        GNFConverter 实例
    """
    gnf_config = config.get("converter", config.get("gnf_converter", {}))
    sigma_ratios = gnf_config.get("sigma_ratios", None)
    if sigma_ratios is not None and not isinstance(sigma_ratios, dict):
        sigma_ratios = OmegaConf.to_container(sigma_ratios, resolve=True)
    
    gradient_field_method = gnf_config.get("gradient_field_method", "softmax")
    method_configs = gnf_config.get("method_configs", {})
    default_config = gnf_config.get("default_config", {})
    
    if gradient_field_method in method_configs:
        method_config = method_configs[gradient_field_method]
        n_query_points = method_config.get("n_query_points", gnf_config.get("n_query_points", 1000))
        step_size = method_config.get("step_size", gnf_config.get("step_size", 0.01))
        sig_sf = method_config.get("sig_sf", gnf_config.get("sig_sf", 0.1))
        sig_mag = method_config.get("sig_mag", gnf_config.get("sig_mag", 0.45))
    else:
        n_query_points = default_config.get("n_query_points", gnf_config.get("n_query_points", 1000))
        step_size = default_config.get("step_size", gnf_config.get("step_size", 0.01))
        sig_sf = default_config.get("sig_sf", gnf_config.get("sig_sf", 0.1))
        sig_mag = default_config.get("sig_mag", gnf_config.get("sig_mag", 0.45))
    
    return GNFConverter(
        sigma=gnf_config.get("sigma", 0.5),
        n_query_points=n_query_points,
        n_iter=gnf_config.get("n_iter", 2000),
        step_size=step_size,
        eps=gnf_config.get("eps", 0.1),
        min_samples=gnf_config.get("min_samples", 5),
        device=device,
        sigma_ratios=sigma_ratios,
        gradient_field_method=gradient_field_method,
        temperature=gnf_config.get("temperature", 0.004),
        logsumexp_eps=gnf_config.get("logsumexp_eps", 1e-8),
        inverse_square_strength=gnf_config.get("inverse_square_strength", 1.0),
        gradient_clip_threshold=gnf_config.get("gradient_clip_threshold", 0.3),
        sig_sf=sig_sf,
        sig_mag=sig_mag
    )

def prepare_data(fabric: Fabric, config: OmegaConf, device: torch.device) -> Tuple[Any, torch.Tensor, torch.Tensor]:
    """准备数据。

    Args:
        fabric: Fabric 实例
        config: 配置对象
        device: 计算设备

    Returns:
        Tuple[batch, coords, atoms_channel]，数据批次、坐标和原子类型
    """
    gnf_converter = create_gnf_converter(config, device="cpu")
    loader_val = create_field_loaders(config, gnf_converter, split="val", fabric=fabric)
    batch = next(iter(loader_val)).to(device)
    coords, _ = to_dense_batch(batch.pos, batch.batch, fill_value=0)
    atoms_channel, _ = to_dense_batch(batch.x, batch.batch, fill_value=PADDING_INDEX)
    
    print(f"Input coords range: "
          f"X[{coords[:, :, 0].min():.3f}, {coords[:, :, 0].max():.3f}], "
          f"Y[{coords[:, :, 1].min():.3f}, {coords[:, :, 1].max():.3f}], "
          f"Z[{coords[:, :, 2].min():.3f}, {coords[:, :, 2].max():.3f}]")
    
    return batch, coords, atoms_channel

def main(config_name: str = "train_nf_qm9", 
         devices: str = "1", 
         accelerator: str = "cpu", 
         precision: str = "32-true", 
         sample_idx: int = 0, 
         output_dir: str = "gnf_reconstruction_results",
         field_option: str = "predicted_field") -> Dict[str, Any]:
    """主函数：加载模型和数据，执行 GNF 可视化。

    Args:
        config_name: 配置文件名
        devices: CUDA 设备编号
        accelerator: 加速器类型
        precision: 计算精度
        sample_idx: 样本索引
        output_dir: 输出文件保存目录
        field_option: 梯度场类型（"gt_field" 或 "predicted_field"）

    Returns:
        可视化结果字典
    """
    fabric, device = setup_environment(devices, accelerator, precision)
    config = load_config(config_name)
    encoder, decoder = load_model(fabric, config)
    converter = create_converter(config, device)
    batch, gt_coords, gt_types = prepare_data(fabric, config, device)
    
    print("Generating codes...")
    with torch.no_grad():
        codes = encoder(batch)
    
    print(f"Visualizing molecule sample {sample_idx}")
    visualizer = GNFVisualizer(output_dir)
    
    try:
        results = visualizer.create_reconstruction_animation(
            gt_coords=gt_coords,
            gt_types=gt_types,
            converter=converter,
            decoder=decoder,
            codes=codes,
            save_interval=100,
            animation_name=f"recon_sample_{sample_idx}",
            field_option=field_option,
            sample_idx=sample_idx
        )
        
        print("\n=== 重建结果 ===")
        print(f"RMSD: {results['final_rmsd']:.4f}")
        print(f"Reconstruction Loss: {results['final_loss']:.4f}")
        print(f"KL Divergence (orig->recon): {results['final_kl_1to2']:.4f}")
        print(f"KL Divergence (recon->orig): {results['final_kl_2to1']:.4f}")
        print("\n输出文件:")
        print(f"  GIF动画: {results['gif_path']}")
        print(f"  对比图: {results['comparison_path']}")
        
        print("GNF visualization completed successfully!")
        return results
    
    except Exception as e:
        print(f"Error during visualization: {e}")
        import traceback
        traceback.print_exc()
        return {}

if __name__ == "__main__":
    main(
        config_name=os.environ.get("CONFIG_NAME", "train_nf_qm9"),
        devices=os.environ.get("CUDA_VISIBLE_DEVICES", "1"),
        accelerator=os.environ.get("ACCELERATOR", "cpu"),
        precision=os.environ.get("PRECISION", "32-true"),
        sample_idx=int(os.environ.get("SAMPLE_IDX", "0")),
        output_dir=os.environ.get("OUTPUT_DIR", "gnf_reconstruction_results"),
        field_option=os.environ.get("FIELD_OPTION", "predicted_field")
    )