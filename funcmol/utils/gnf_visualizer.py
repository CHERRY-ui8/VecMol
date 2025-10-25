import os
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any, Union
import torch
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from omegaconf import OmegaConf
from lightning import Fabric
from torch_geometric.utils import to_dense_batch
from funcmol.utils.constants import PADDING_INDEX
from funcmol.utils.gnf_converter import GNFConverter

# 设置项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(PROJECT_ROOT))

# 禁用 Dynamo 错误提示
try:
    torch._dynamo.config.suppress_errors = True
except AttributeError:
    # PyTorch 版本 < 2.0 不支持 torch._dynamo
    pass

class MoleculeMetrics:
    """分子重建指标计算类，提供 RMSD、重建损失和 KL 散度计算方法。"""
    
    @staticmethod
    def compute_rmsd_scalar(coords1: torch.Tensor, coords2: torch.Tensor) -> float:
        """计算两个坐标集之间的对称 RMSD，返回标量值。

        Args:
            coords1: 第一个坐标集，形状 [N, 3]
            coords2: 第二个坐标集，形状 [M, 3]

        Returns:
            对称 RMSD 标量值
        """
        coords1 = coords1.detach()
        coords2 = coords2.detach()
        
        dist1 = torch.sqrt(torch.sum((coords1.unsqueeze(1) - coords2.unsqueeze(0))**2, dim=2) + 1e-8)
        dist2 = torch.sqrt(torch.sum((coords2.unsqueeze(1) - coords1.unsqueeze(0))**2, dim=2) + 1e-8)
        
        min_dist1 = torch.min(dist1, dim=1)[0]
        min_dist2 = torch.min(dist2, dim=1)[0]
        
        return ((torch.mean(min_dist1) + torch.mean(min_dist2)) / 2).item()
    
    @staticmethod
    def compute_reconstruction_loss_scalar(coords: torch.Tensor, points: torch.Tensor) -> float:
        """计算重建损失，返回标量值。

        Args:
            coords: 真实坐标，形状 [N, 3]
            points: 重建点，形状 [M, 3]

        Returns:
            重建损失标量值
        """
        dist1 = torch.sum((coords.unsqueeze(1) - points.unsqueeze(0))**2, dim=2)
        eps = 1e-8
        min_dist_to_samples = torch.min(dist1 + eps, dim=1)[0]
        min_dist_to_atoms = torch.min(dist1 + eps, dim=0)[0]
        
        coverage_loss = torch.mean(min_dist_to_samples)
        clustering_loss = torch.mean(min_dist_to_atoms)
        
        return torch.sqrt(coverage_loss + 0.1 * clustering_loss).item()
    
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
        dist_matrix = torch.sum((coords1.unsqueeze(1) - coords2.unsqueeze(0))**2, dim=2)
        p1_given_2 = torch.softmax(-dist_matrix / temperature, dim=0)
        p2_given_1 = torch.softmax(-dist_matrix / temperature, dim=1)
        eps = 1e-8
        
        kl_1to2 = torch.mean(torch.sum(p1_given_2 * torch.log(p1_given_2 / (1.0/coords2.shape[0] + eps) + eps), dim=0))
        kl_2to1 = torch.mean(torch.sum(p2_given_1 * torch.log(p2_given_1 / (1.0/coords1.shape[0] + eps) + eps), dim=1))
        
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
        """设置 3D 坐标轴的通用属性。

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
        
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)
        ax.set_box_aspect([1, 1, 1])

class GNFVisualizer(MoleculeVisualizer):
    """GNF 重建过程可视化类，生成分子对比图和重建动画。"""
    
    def __init__(self, output_dir: str = None):
        """初始化可视化器。

        Args:
            output_dir: 输出文件保存目录，若为None则使用默认路径
        """
        super().__init__()
        if output_dir is None:
            # 默认输出到../exps/neural_field/下，与训练脚本保持一致
            self.output_dir = str(PROJECT_ROOT.parent / "exps" / "neural_field" / "gnf_reconstruction_results")
        else:
            self.output_dir = output_dir
        
        # 创建recon子目录
        self.recon_dir = os.path.join(self.output_dir, "recon")
        os.makedirs(self.recon_dir, exist_ok=True)
        
        self.metrics = MoleculeMetrics()
        os.makedirs(self.output_dir, exist_ok=True)
    
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
            # 确保保存目录存在
            save_dir = os.path.dirname(save_path)
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
                print(f"Created directory: {save_dir}")
            
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
        """可视化重建过程的单个步骤"""
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
        
        # 确保保存目录存在
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
            print(f"Created directory: {save_dir}")
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_reconstruction_animation(self,
                                       gt_coords: torch.Tensor,
                                       gt_types: torch.Tensor,
                                       converter: GNFConverter,
                                       field_func,
                                       save_interval: int = 50,
                                       animation_name: str = "reconstruction",
                                       sample_idx: int = 0) -> Dict[str, Any]:
        """创建分子重建过程的动画

        Args:
            gt_coords: 真实分子坐标，形状 [batch, n_atoms, 3]
            gt_types: 真实原子类型，形状 [batch, n_atoms]
            converter: GNF 转换器
            field_func: 梯度场函数，接受points参数并返回梯度场
            save_interval: 保存帧的间隔
            animation_name: 动画文件名前缀
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
        
        all_atom_types = list(range(5))
        
        coords_min = gt_valid_coords.min(dim=0)[0]
        coords_max = gt_valid_coords.max(dim=0)[0]
        coords_range = coords_max - coords_min
        margin = coords_range * 0.5
        init_min = coords_min - margin
        init_max = coords_max + margin
        
        z_dict = {}
        for atom_type in all_atom_types:
            z_dict[atom_type] = torch.rand(converter.n_query_points, 3, device=device) * (init_max - init_min) + init_min
        
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
                    gradients = field_func(z)
                    # 确保梯度场形状正确
                    if gradients.dim() == 4:  # [batch, n_points, n_atom_types, 3]
                        gradients = gradients[0]  # 取第一个batch
                    
                    type_gradients = gradients[:, atom_type, :]
                    
                    z_dict[atom_type] = z + torch.tensor(converter.step_size, device=z.device) * type_gradients
            
            if i % save_interval == 0 or i == converter.n_iter - 1:
                frame_path = os.path.join(self.recon_dir, f"frame_sample_{sample_idx}_{i:04d}.png")
                
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
        
        gif_path = os.path.join(self.recon_dir, f"{animation_name}.gif")
        with imageio.get_writer(gif_path, mode='I', duration=0.1, fps=15, loop=1) as writer:
            for frame_path in frame_paths:
                try:
                    # 检查文件是否存在且大小大于0
                    if not os.path.exists(frame_path):
                        print(f"Warning: Frame file {frame_path} does not exist, skipping...")
                        continue
                    
                    # 等待文件完全写入
                    import time
                    time.sleep(0.01)  # 短暂等待确保文件写入完成
                    
                    # 检查文件大小
                    if os.path.getsize(frame_path) == 0:
                        print(f"Warning: Frame file {frame_path} is empty, skipping...")
                        continue
                    
                    frame = imageio.imread(frame_path)
                    writer.append_data(frame)
                    
                except Exception as e:
                    print(f"Warning: Failed to read frame {frame_path}: {e}")
                    continue
                finally:
                    # 确保清理文件
                    try:
                        if os.path.exists(frame_path):
                            os.remove(frame_path)
                    except:
                        pass
        
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
        
        comparison_path = os.path.join(self.recon_dir, f"{animation_name}_final.png")
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

    def create_generation_animation(self,
                                   converter: GNFConverter,
                                   field_func,
                                   sample_idx: int = 0,
                                   save_interval: int = 50,
                                   create_1d_plots: bool = True,
                                   use_recon_dir: bool = True,
                                   fixed_axis: bool = True,
                                   use_intelligent_sampling: bool = False,
                                   decoder=None,
                                   codes=None,
                                   fabric=None) -> Dict[str, Any]:
        """创建分子生成过程的动画
        
        从 codes 生成的场函数开始，通过梯度下降找到原子位置，生成完整的分子结构。
        不需要任何 ground truth 参考。
        
        Args:
            converter: GNF 转换器，用于将场函数转换为原子位置
            field_func: 场函数，输入查询点 [B, n_points, 3]，输出场信息 [B, n_points, n_atom_types, 3]
            sample_idx: 样本索引
            save_interval: 保存帧的间隔
            create_1d_plots: 是否创建1D场可视化
            use_recon_dir: 是否使用recon子目录，False时直接在主目录下生成文件
            fixed_axis: 是否使用固定坐标轴，True时坐标轴范围在第一帧确定后不再变化
            use_intelligent_sampling: 是否使用智能采样（与field_recon.py相同的方法）
            decoder: 解码器（智能采样时需要）
            codes: 分子代码（智能采样时需要）
            fabric: 日志对象（智能采样时需要）
            
        Returns:
            包含动画路径、最终结果和指标历史的字典
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 选择输出目录
        if use_recon_dir:
            output_dir = self.recon_dir
        else:
            output_dir = self.output_dir
        
        print(f"\nStarting generation for molecule {sample_idx}")
        
        all_atom_types = list(range(5))
        
        # 初始化随机点
        if use_intelligent_sampling and decoder is not None and codes is not None:
            # 使用与 field_recon.py 相同的智能采样方法
            print("Using intelligent sampling method...")
            init_min, init_max = -7.0, 7.0
            n_candidates = converter.n_query_points * converter.gradient_sampling_candidate_multiplier
            candidate_points = torch.rand(n_candidates, 3, device=device) * (init_max - init_min) + init_min
            
            # 计算候选点的梯度场强度
            candidate_batch = candidate_points.unsqueeze(0)  # [1, n_candidates, 3]
            
            try:
                with torch.no_grad():
                    candidate_field = decoder(candidate_batch, codes)
            except Exception as e:
                if fabric:
                    fabric.print(f">> ERROR in decoder call: {e}")
                raise e
            
            # 为每个原子类型进行智能采样
            z_dict = {}
            for t in all_atom_types:
                candidate_grad = candidate_field[0, :, t, :]  # [n_candidates, 3]
                grad_magnitudes = torch.norm(candidate_grad, dim=1)  # [n_candidates]
                probabilities = torch.softmax(grad_magnitudes / converter.gradient_sampling_temperature, dim=0)
                sampled_indices = torch.multinomial(probabilities, converter.n_query_points, replacement=False)
                z_dict[t] = candidate_points[sampled_indices]  # [n_query_points, 3]
        else:
            # 使用简单的均匀随机采样
            init_min = torch.tensor([-3.0, -3.0, -3.0], device=device)
            init_max = torch.tensor([3.0, 3.0, 3.0], device=device)
            
            z_dict = {}
            for atom_type in all_atom_types:
                z_dict[atom_type] = torch.rand(converter.n_query_points, 3, device=device) * (init_max - init_min) + init_min
        
        frame_paths = []
        metrics_history = {
            'iterations': [],
            'convergence': []  # 记录收敛情况
        }
        
        # 如果使用固定坐标轴，在第一帧确定坐标轴范围
        fixed_axis_limits = None
        
        for i in range(converter.n_iter):
            with torch.no_grad():
                for atom_type in all_atom_types:
                    z = z_dict[atom_type]
                    gradients = field_func(z)
                    # 确保梯度场形状正确
                    if gradients.dim() == 4:  # [batch, n_points, n_atom_types, 3]
                        gradients = gradients[0]  # 取第一个batch
                    
                    type_gradients = gradients[:, atom_type, :]
                    
                    z_dict[atom_type] = z + torch.tensor(converter.step_size, device=z.device) * type_gradients
            
            if i % save_interval == 0 or i == converter.n_iter - 1:
                frame_path = os.path.join(output_dir, f"frame_gen_sample_{sample_idx}_{i:04d}.png")
                
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
                
                # 只显示生成的分子，不显示"原始分子"
                # 如果是第一帧且使用固定坐标轴，确定坐标轴范围
                if i == 0 and fixed_axis and len(current_points) > 0:
                    points_np = current_points.detach().cpu().numpy()
                    margin = 1.0
                    fixed_axis_limits = {
                        'x_min': points_np[:, 0].min() - margin,
                        'x_max': points_np[:, 0].max() + margin,
                        'y_min': points_np[:, 1].min() - margin,
                        'y_max': points_np[:, 1].max() + margin,
                        'z_min': points_np[:, 2].min() - margin,
                        'z_max': points_np[:, 2].max() + margin
                    }
                
                self.visualize_generation_step(
                    current_points, i, frame_path, current_types, fixed_axis_limits
                )
                frame_paths.append(frame_path)
                
                # 计算收敛指标（点之间的平均距离变化）
                if len(current_points) > 1:
                    distances = torch.pdist(current_points)
                    convergence = torch.std(distances).item()
                    metrics_history['convergence'].append(convergence)
        
        gif_path = os.path.join(output_dir, f"funcmol_gen_sample_{sample_idx}.gif")
        with imageio.get_writer(gif_path, mode='I', duration=0.1, fps=15, loop=1) as writer:
            for frame_path in frame_paths:
                try:
                    # 检查文件是否存在且大小大于0
                    if not os.path.exists(frame_path):
                        print(f"Warning: Frame file {frame_path} does not exist, skipping...")
                        continue
                    
                    # 等待文件完全写入
                    import time
                    time.sleep(0.01)  # 短暂等待确保文件写入完成
                    
                    # 检查文件大小
                    if os.path.getsize(frame_path) == 0:
                        print(f"Warning: Frame file {frame_path} is empty, skipping...")
                        continue
                    
                    frame = imageio.imread(frame_path)
                    writer.append_data(frame)
                    
                except Exception as e:
                    print(f"Warning: Failed to read frame {frame_path}: {e}")
                    continue
                finally:
                    # 确保清理文件
                    try:
                        if os.path.exists(frame_path):
                            os.remove(frame_path)
                    except:
                        pass
        
        # 生成最终分子结构
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
        
        # 保存最终生成的分子
        final_path = os.path.join(output_dir, f"funcmol_gen_sample_{sample_idx}_final.png")
        self.visualize_generated_molecule(
            final_points, final_types, save_path=final_path
        )
        
        # 创建1D场可视化（如果启用）
        field_1d_results = None
        if create_1d_plots:
            try:
                # 创建1D场可视化目录
                field_1d_dir = os.path.join(output_dir, "field_1d")
                os.makedirs(field_1d_dir, exist_ok=True)
                
                # 生成1D场可视化
                field_1d_save_path = os.path.join(field_1d_dir, f"field_1d_sample_{sample_idx}.png")
                field_1d_results = visualize_1d_gradient_field_generation(
                    field_func=field_func,
                    sample_idx=sample_idx,
                    atom_types=[0, 1, 2, 3, 4],  # C, H, O, N, F
                    x_range=None,  # 使用默认范围
                    n_points=3000,
                    y_coord=0.0,
                    z_coord=0.0,
                    save_path=field_1d_save_path
                )
                print(f"1D field visualization created for sample {sample_idx}")
            except Exception as e:
                print(f"Warning: Failed to create 1D field visualization for sample {sample_idx}: {e}")
                field_1d_results = None

        return {
            'gif_path': gif_path,
            'final_path': final_path,
            'metrics_history': metrics_history,
            'final_points': final_points,
            'final_types': final_types,
            'final_convergence': metrics_history['convergence'][-1] if metrics_history['convergence'] else float('inf'),
            'field_1d_results': field_1d_results
        }
    
    def visualize_generation_step(self, 
                                 current_points: torch.Tensor,
                                 iteration: int,
                                 save_path: str,
                                 current_types: torch.Tensor,
                                 fixed_axis_limits: Optional[Dict] = None):
        """可视化分子生成过程的单个步骤"""
        fig = plt.figure(figsize=(8, 8))  # 改为与reconstruction_step一致
        ax = fig.add_subplot(111, projection='3d')
        
        points_np = current_points.detach().cpu().numpy()
        if current_types is not None:
            types_np = current_types.detach().cpu().numpy()
            for atom_type in range(5):
                mask = (types_np == atom_type)
                if mask.sum() > 0:
                    ax.scatter(points_np[mask, 0], points_np[mask, 1], points_np[mask, 2], 
                              c=self.atom_colors[atom_type], marker='.', s=20, 
                              label=f'{["C", "H", "O", "N", "F"][atom_type]}', alpha=0.5)
        else:
            ax.scatter(points_np[:, 0], points_np[:, 1], points_np[:, 2], 
                      c='blue', marker='.', s=20, label='Generated Points', alpha=0.5)
        
        # 设置坐标轴
        if fixed_axis_limits is not None:
            # 使用固定的坐标轴范围
            ax.set_xlim(fixed_axis_limits['x_min'], fixed_axis_limits['x_max'])
            ax.set_ylim(fixed_axis_limits['y_min'], fixed_axis_limits['y_max'])
            ax.set_zlim(fixed_axis_limits['z_min'], fixed_axis_limits['z_max'])
            ax.set_xlabel('X (Å)')
            ax.set_ylabel('Y (Å)')
            ax.set_zlabel('Z (Å)')
            ax.grid(True, alpha=0.3)
        elif len(points_np) > 0:
            # 动态调整坐标轴范围
            self._setup_3d_axis(ax, [points_np], margin=1.0)
        
        ax.view_init(elev=30, azim=60)
        ax.set_box_aspect([1, 1, 1])
        
        ax.set_title(f"Generated Molecule - Iteration {iteration}")
        ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0))
        
        # 确保保存目录存在
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def visualize_generated_molecule(self,
                                   final_points: torch.Tensor,
                                   final_types: torch.Tensor,
                                   save_path: str):
        """可视化最终生成的分子"""
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        points_np = final_points.detach().cpu().numpy()
        if final_types is not None and len(final_types) > 0:
            types_np = final_types.detach().cpu().numpy()
            for atom_type in range(5):
                mask = (types_np == atom_type)
                if mask.sum() > 0:
                    ax.scatter(points_np[mask, 0], points_np[mask, 1], points_np[mask, 2], 
                              c=self.atom_colors[atom_type], marker='o', s=150, 
                              label=f'{["C", "H", "O", "N", "F"][atom_type]}', alpha=0.9)
        else:
            ax.scatter(points_np[:, 0], points_np[:, 1], points_np[:, 2], 
                      c='blue', marker='o', s=150, label='Generated Molecule', alpha=0.9)
        
        # 设置坐标轴
        if len(points_np) > 0:
            self._setup_3d_axis(ax, [points_np], margin=1.0)
        
        ax.view_init(elev=30, azim=60)
        ax.set_box_aspect([1, 1, 1])
        
        ax.set_title("Generated Molecule (Final Result)")
        ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0))
        
        # 确保保存目录存在
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

def visualize_1d_gradient_field_comparison(
    gt_coords: torch.Tensor,
    gt_types: torch.Tensor,
    converter: GNFConverter,
    field_func,
    sample_idx: int = 0,
    atom_types: Union[int, List[int]] = 0,
    x_range: Optional[tuple] = None,
    n_points: int = 3000,
    y_coord: float = 0.0,
    z_coord: float = 0.0,
    save_path: Optional[str] = None,
    display_sample_idx: Optional[int] = None
) -> Optional[Dict[str, Any]]:
    """可视化一维方向上的梯度场对比（真实 vs 预测）。

    Args:
        gt_coords: 真实分子坐标，形状 [batch, n_atoms, 3]
        gt_types: 真实原子类型，形状 [batch, n_atoms]
        converter: GNF 转换器
        field_func: 梯度场函数
        sample_idx: 数据访问的样本索引（用于访问gt_coords和gt_types）
        atom_types: 原子类型列表或单个原子类型（0=C, 1=H, 2=O, 3=N, 4=F）
        x_range: x 轴范围，None 时自动计算
        n_points: 采样点数
        y_coord: y 坐标固定值
        z_coord: z 坐标固定值
        save_path: 保存路径，None 时显示图像
        display_sample_idx: 用于显示和文件名的样本索引，如果为None则使用sample_idx

    Returns:
        包含梯度场统计信息和数据的字典，若无目标原子类型则返回 None
    """
    device = gt_coords.device
    
    # 设置显示用的样本索引
    if display_sample_idx is None:
        display_sample_idx = sample_idx
    
    # 确保 atom_types 是列表
    if isinstance(atom_types, int):
        atom_types = [atom_types]
    
    gt_mask = (gt_types[sample_idx] != PADDING_INDEX)
    gt_valid_coords = gt_coords[sample_idx][gt_mask]
    gt_valid_types = gt_types[sample_idx][gt_mask]
    
    # 检查所有原子类型是否存在
    available_atom_types = []
    for atom_type in atom_types:
        target_atoms = gt_valid_types == atom_type
        if target_atoms.sum() > 0:
            available_atom_types.append(atom_type)
        else:
            # 扩展原子类型名称列表以支持更多元素
            atom_names = ['C', 'H', 'O', 'N', 'F', 'S', 'Cl', 'Br']
            if atom_type < len(atom_names):
                atom_name = atom_names[atom_type]
            else:
                atom_name = f"Type{atom_type}"
            print(f"警告：样本 {display_sample_idx} 中没有类型为 {atom_name} 的原子")
    
    if not available_atom_types:
        print("没有找到任何指定的原子类型")
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
        pred_field = field_func(query_points.unsqueeze(0))
        # 确保预测场形状正确
        if pred_field.dim() == 4:  # [batch, n_points, n_atom_types, 3]
            pred_field = pred_field[0]  # 取第一个batch
    
    # 为每个原子类型创建单独的2×4图
    all_results = {}
    
    for atom_type in available_atom_types:
        # 扩展原子类型名称列表以支持更多元素
        atom_names = ["C", "H", "O", "N", "F", "S", "Cl", "Br"]
        if atom_type < len(atom_names):
            atom_name = atom_names[atom_type]
        else:
            atom_name = f"Type{atom_type}"
        
        # 获取梯度场数据
        gt_gradients_3d = gt_field[0, :, atom_type, :]
        gt_gradients = torch.norm(gt_gradients_3d, dim=1)
        gt_gradients_x, gt_gradients_y, gt_gradients_z = gt_gradients_3d[:, 0], gt_gradients_3d[:, 1], gt_gradients_3d[:, 2]
        
        pred_gradients_3d = pred_field[:, atom_type, :]
        pred_gradients = torch.norm(pred_gradients_3d, dim=1)
        pred_gradients_x, pred_gradients_y, pred_gradients_z = pred_gradients_3d[:, 0], pred_gradients_3d[:, 1], pred_gradients_3d[:, 2]
        
        # 获取原子位置
        target_atoms = gt_valid_types == atom_type
        atom_positions = gt_valid_coords[target_atoms, 0].cpu().numpy()
        
        # 创建2×4的子图
        fig, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8)) = plt.subplots(2, 4, figsize=(24, 12))
        
        # 子图1: 梯度场幅度对比
        ax1.plot(x.cpu().numpy(), gt_gradients.cpu().numpy(), 
                 label=f'Ground Truth ({atom_name})', 
                 linewidth=2, color='blue', alpha=0.8)
        ax1.plot(x.cpu().numpy(), pred_gradients.cpu().numpy(), 
                 label=f'Predicted ({atom_name})', 
                 linewidth=2, color='red', alpha=0.8, linestyle='--')
        
        if len(atom_positions) > 0:
            for i, pos in enumerate(atom_positions):
                ax1.axvline(x=pos, color='green', linestyle=':', alpha=0.7, linewidth=1.5)
                ax1.text(pos, ax1.get_ylim()[1] * 0.9, f'{atom_name}{i+1}', 
                         rotation=90, verticalalignment='top', fontsize=8)
        
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=0.5)
        ax1.set_xlabel('X Position (Å)', fontsize=12)
        ax1.set_ylabel('Gradient Field Magnitude', fontsize=12)
        ax1.set_title(f'Gradient Field Magnitude - {atom_name} Atoms\n'
                      f'Sample {display_sample_idx}, Line: y={y_coord:.2f}, z={z_coord:.2f}', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=11)
        
        mse = torch.mean((gt_gradients - pred_gradients) ** 2).item()
        mae = torch.mean(torch.abs(gt_gradients - pred_gradients)).item()
        ax1.text(0.02, 0.98, f'MSE: {mse:.6f}\nMAE: {mae:.6f}', 
                 transform=ax1.transAxes, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), fontsize=10)
        
        # 子图2: X方向分量对比
        ax2.plot(x.cpu().numpy(), gt_gradients_x.cpu().numpy(), 
                 label=f'Ground Truth X ({atom_name})', 
                 linewidth=2, color='blue', alpha=0.8)
        ax2.plot(x.cpu().numpy(), pred_gradients_x.cpu().numpy(), 
                 label=f'Predicted X ({atom_name})', 
                 linewidth=2, color='red', alpha=0.8, linestyle='--')
        
        if len(atom_positions) > 0:
            for i, pos in enumerate(atom_positions):
                ax2.axvline(x=pos, color='green', linestyle=':', alpha=0.7, linewidth=1.5)
        
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=0.5)
        ax2.set_xlabel('X Position (Å)', fontsize=12)
        ax2.set_ylabel('X Component of Gradient Field', fontsize=12)
        ax2.set_title(f'X Direction Component - {atom_name} Atoms\n'
                      f'(Positive = Right, Negative = Left)', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=11)
        
        mse_x = torch.mean((gt_gradients_x - pred_gradients_x) ** 2).item()
        mae_x = torch.mean(torch.abs(gt_gradients_x - pred_gradients_x)).item()
        ax2.text(0.02, 0.98, f'MSE: {mse_x:.6f}\nMAE: {mae_x:.6f}', 
                 transform=ax2.transAxes, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8), fontsize=10)
        
        # 子图3: Y方向分量对比
        ax3.plot(x.cpu().numpy(), gt_gradients_y.cpu().numpy(), 
                 label=f'Ground Truth Y ({atom_name})', 
                 linewidth=2, color='blue', alpha=0.8)
        ax3.plot(x.cpu().numpy(), pred_gradients_y.cpu().numpy(), 
                 label=f'Predicted Y ({atom_name})', 
                 linewidth=2, color='red', alpha=0.8, linestyle='--')
        
        if len(atom_positions) > 0:
            for i, pos in enumerate(atom_positions):
                ax3.axvline(x=pos, color='green', linestyle=':', alpha=0.7, linewidth=1.5)
        
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=0.5)
        ax3.set_xlabel('X Position (Å)', fontsize=12)
        ax3.set_ylabel('Y Component of Gradient Field', fontsize=12)
        ax3.set_title(f'Y Direction Component - {atom_name} Atoms\n'
                      f'(Positive = Up, Negative = Down)', fontsize=14)
        ax3.grid(True, alpha=0.3)
        ax3.legend(fontsize=11)
        
        mse_y = torch.mean((gt_gradients_y - pred_gradients_y) ** 2).item()
        mae_y = torch.mean(torch.abs(gt_gradients_y - pred_gradients_y)).item()
        ax3.text(0.02, 0.98, f'MSE: {mse_y:.6f}\nMAE: {mae_y:.6f}', 
                 transform=ax3.transAxes, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8), fontsize=10)
        
        # 子图4: Z方向分量对比
        ax4.plot(x.cpu().numpy(), gt_gradients_z.cpu().numpy(), 
                 label=f'Ground Truth Z ({atom_name})', 
                 linewidth=2, color='blue', alpha=0.8)
        ax4.plot(x.cpu().numpy(), pred_gradients_z.cpu().numpy(), 
                 label=f'Predicted Z ({atom_name})', 
                 linewidth=2, color='red', alpha=0.8, linestyle='--')
        
        if len(atom_positions) > 0:
            for i, pos in enumerate(atom_positions):
                ax4.axvline(x=pos, color='green', linestyle=':', alpha=0.7, linewidth=1.5)
        
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=0.5)
        ax4.set_xlabel('X Position (Å)', fontsize=12)
        ax4.set_ylabel('Z Component of Gradient Field', fontsize=12)
        ax4.set_title(f'Z Direction Component - {atom_name} Atoms\n'
                      f'(Positive = Forward, Negative = Backward)', fontsize=14)
        ax4.grid(True, alpha=0.3)
        ax4.legend(fontsize=11)
        
        mse_z = torch.mean((gt_gradients_z - pred_gradients_z) ** 2).item()
        mae_z = torch.mean(torch.abs(gt_gradients_z - pred_gradients_z)).item()
        ax4.text(0.02, 0.98, f'MSE: {mse_z:.6f}\nMAE: {mae_z:.6f}', 
                 transform=ax4.transAxes, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8), fontsize=10)
        
        # 子图5: 幅度误差分析
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
        ax5.set_title(f'Magnitude Error Analysis - {atom_name} Atoms', fontsize=14)
        ax5.grid(True, alpha=0.3)
        ax5.legend(fontsize=11)
        
        error_mean = torch.mean(error).item()
        error_std = torch.std(error).item()
        ax5.text(0.02, 0.98, f'Error Mean: {error_mean:.6f}\nError Std: {error_std:.6f}', 
                 transform=ax5.transAxes, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8), fontsize=10)
        
        # 子图6: X方向误差分析
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
        ax6.set_title(f'X Direction Error Analysis - {atom_name} Atoms', fontsize=14)
        ax6.grid(True, alpha=0.3)
        ax6.legend(fontsize=11)
        
        error_x_mean = torch.mean(error_x).item()
        error_x_std = torch.std(error_x).item()
        ax6.text(0.02, 0.98, f'Error Mean: {error_x_mean:.6f}\nError Std: {error_x_std:.6f}', 
                 transform=ax6.transAxes, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8), fontsize=10)
        
        # 子图7: Y方向误差分析
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
        ax7.set_title(f'Y Direction Error Analysis - {atom_name} Atoms', fontsize=14)
        ax7.grid(True, alpha=0.3)
        ax7.legend(fontsize=11)
        
        error_y_mean = torch.mean(error_y).item()
        error_y_std = torch.std(error_y).item()
        ax7.text(0.02, 0.98, f'Error Mean: {error_y_mean:.6f}\nError Std: {error_y_std:.6f}', 
                 transform=ax7.transAxes, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8), fontsize=10)
        
        # 子图8: Z方向误差分析
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
        ax8.set_title(f'Z Direction Error Analysis - {atom_name} Atoms', fontsize=14)
        ax8.grid(True, alpha=0.3)
        ax8.legend(fontsize=11)
        
        error_z_mean = torch.mean(error_z).item()
        error_z_std = torch.std(error_z).item()
        ax8.text(0.02, 0.98, f'Error Mean: {error_z_mean:.6f}\nError Std: {error_z_std:.6f}', 
                 transform=ax8.transAxes, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8), fontsize=10)
        
        plt.tight_layout()
        
        # 保存每个原子类型的图
        if save_path:
            atom_name_clean = atom_name
            base_name = f"field_1d_sample_{display_sample_idx}_atom_{atom_name_clean}"
            atom_save_path = os.path.join(os.path.dirname(save_path) or '.', f"{base_name}.png")
            
            # 确保保存目录存在
            save_dir = os.path.dirname(atom_save_path)
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
                print(f"Created directory: {save_dir}")
            
            plt.savefig(atom_save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Field 1D comparison (atom_type={atom_name}) saved to: {atom_save_path}")
        else:
            plt.show()
        
        # 保存该原子类型的统计信息
        all_results[atom_name] = {
            'save_path': atom_save_path if save_path else None,
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
    
    return {
        'all_results': all_results,
        'available_atom_types': available_atom_types
    }

def visualize_1d_gradient_field_generation(
    field_func,
    sample_idx: int = 0,
    atom_types: Union[int, List[int]] = 0,
    x_range: Optional[tuple] = None,
    n_points: int = 3000,
    y_coord: float = 0.0,
    z_coord: float = 0.0,
    save_path: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """可视化生成任务的一维梯度场（无ground truth对比）。
    
    Args:
        field_func: 梯度场函数
        sample_idx: 样本索引
        atom_types: 原子类型列表或单个原子类型（0=C, 1=H, 2=O, 3=N, 4=F）
        x_range: x 轴范围，None 时使用默认范围
        n_points: 采样点数
        y_coord: y 坐标固定值
        z_coord: z 坐标固定值
        save_path: 保存路径，None 时显示图像

    Returns:
        包含梯度场统计信息和数据的字典
    """
    device = next(field_func.parameters()).device if hasattr(field_func, 'parameters') else torch.device('cpu')
    
    # 确保 atom_types 是列表
    if isinstance(atom_types, int):
        atom_types = [atom_types]
    
    # 设置默认x范围
    if x_range is None:
        x_range = (-5.0, 5.0)  # 默认范围
        print(f"使用默认 x 轴范围: {x_range}")
    
    x = torch.linspace(x_range[0], x_range[1], n_points, device=device)
    query_points = torch.zeros(n_points, 3, device=device)
    query_points[:, 0], query_points[:, 1], query_points[:, 2] = x, y_coord, z_coord
    
    with torch.no_grad():
        pred_field = field_func(query_points.unsqueeze(0))
        # 确保预测场形状正确
        if pred_field.dim() == 4:  # [batch, n_points, n_atom_types, 3]
            pred_field = pred_field[0]  # 取第一个batch
    
    # 为每个原子类型创建单独的2×4图
    all_results = {}
    
    for atom_type in atom_types:
        # 扩展原子类型名称列表以支持更多元素
        atom_names = ["C", "H", "O", "N", "F", "S", "Cl", "Br"]
        if atom_type < len(atom_names):
            atom_name = atom_names[atom_type]
        else:
            atom_name = f"Type{atom_type}"
        
        # 获取梯度场数据
        pred_gradients_3d = pred_field[:, atom_type, :]
        pred_gradients = torch.norm(pred_gradients_3d, dim=1)
        pred_gradients_x, pred_gradients_y, pred_gradients_z = pred_gradients_3d[:, 0], pred_gradients_3d[:, 1], pred_gradients_3d[:, 2]
        
        # 创建2×4的子图
        fig, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8)) = plt.subplots(2, 4, figsize=(24, 12))
        
        # 子图1: 梯度场幅度
        ax1.plot(x.cpu().numpy(), pred_gradients.cpu().numpy(), 
                 label=f'Generated Field ({atom_name})', 
                 linewidth=2, color='red', alpha=0.8)
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=0.5)
        ax1.set_xlabel('X Position (Å)', fontsize=12)
        ax1.set_ylabel('Gradient Field Magnitude', fontsize=12)
        ax1.set_title(f'Gradient Field Magnitude - {atom_name} Atoms\n'
                      f'Sample {sample_idx}, Line: y={y_coord:.2f}, z={z_coord:.2f}', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=11)
        
        # 子图2: X方向分量
        ax2.plot(x.cpu().numpy(), pred_gradients_x.cpu().numpy(), 
                 label=f'Generated X ({atom_name})', 
                 linewidth=2, color='red', alpha=0.8)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=0.5)
        ax2.set_xlabel('X Position (Å)', fontsize=12)
        ax2.set_ylabel('X Component of Gradient Field', fontsize=12)
        ax2.set_title(f'X Direction Component - {atom_name} Atoms\n'
                      f'(Positive = Right, Negative = Left)', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=11)
        
        # 子图3: Y方向分量
        ax3.plot(x.cpu().numpy(), pred_gradients_y.cpu().numpy(), 
                 label=f'Generated Y ({atom_name})', 
                 linewidth=2, color='red', alpha=0.8)
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=0.5)
        ax3.set_xlabel('X Position (Å)', fontsize=12)
        ax3.set_ylabel('Y Component of Gradient Field', fontsize=12)
        ax3.set_title(f'Y Direction Component - {atom_name} Atoms\n'
                      f'(Positive = Up, Negative = Down)', fontsize=14)
        ax3.grid(True, alpha=0.3)
        ax3.legend(fontsize=11)
        
        # 子图4: Z方向分量
        ax4.plot(x.cpu().numpy(), pred_gradients_z.cpu().numpy(), 
                 label=f'Generated Z ({atom_name})', 
                 linewidth=2, color='red', alpha=0.8)
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=0.5)
        ax4.set_xlabel('X Position (Å)', fontsize=12)
        ax4.set_ylabel('Z Component of Gradient Field', fontsize=12)
        ax4.set_title(f'Z Direction Component - {atom_name} Atoms\n'
                      f'(Positive = Forward, Negative = Backward)', fontsize=14)
        ax4.grid(True, alpha=0.3)
        ax4.legend(fontsize=11)
        
        # 第二行子图：统计信息
        # 子图5: 梯度场幅度分布
        ax5.hist(pred_gradients.cpu().numpy(), bins=50, alpha=0.7, color='red', edgecolor='black')
        ax5.set_xlabel('Gradient Field Magnitude', fontsize=12)
        ax5.set_ylabel('Frequency', fontsize=12)
        ax5.set_title(f'Magnitude Distribution - {atom_name}', fontsize=14)
        ax5.grid(True, alpha=0.3)
        
        # 子图6: X分量分布
        ax6.hist(pred_gradients_x.cpu().numpy(), bins=50, alpha=0.7, color='red', edgecolor='black')
        ax6.set_xlabel('X Component Value', fontsize=12)
        ax6.set_ylabel('Frequency', fontsize=12)
        ax6.set_title(f'X Component Distribution - {atom_name}', fontsize=14)
        ax6.grid(True, alpha=0.3)
        
        # 子图7: Y分量分布
        ax7.hist(pred_gradients_y.cpu().numpy(), bins=50, alpha=0.7, color='red', edgecolor='black')
        ax7.set_xlabel('Y Component Value', fontsize=12)
        ax7.set_ylabel('Frequency', fontsize=12)
        ax7.set_title(f'Y Component Distribution - {atom_name}', fontsize=14)
        ax7.grid(True, alpha=0.3)
        
        # 子图8: Z分量分布
        ax8.hist(pred_gradients_z.cpu().numpy(), bins=50, alpha=0.7, color='red', edgecolor='black')
        ax8.set_xlabel('Z Component Value', fontsize=12)
        ax8.set_ylabel('Frequency', fontsize=12)
        ax8.set_title(f'Z Component Distribution - {atom_name}', fontsize=14)
        ax8.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图像
        if save_path:
            atom_save_path = save_path.replace('.png', f'_{atom_name}.png')
            plt.savefig(atom_save_path, dpi=300, bbox_inches='tight')
            print(f"1D field visualization saved: {atom_save_path}")
        else:
            plt.show()
        
        # 保存该原子类型的统计信息
        all_results[atom_name] = {
            'save_path': atom_save_path if save_path else None,
            'pred_gradients': pred_gradients.cpu().numpy(),
            'pred_gradients_x': pred_gradients_x.cpu().numpy(),
            'pred_gradients_y': pred_gradients_y.cpu().numpy(),
            'pred_gradients_z': pred_gradients_z.cpu().numpy(),
            'x_positions': x.cpu().numpy(),
            'magnitude_mean': pred_gradients.mean().item(),
            'magnitude_std': pred_gradients.std().item(),
            'magnitude_max': pred_gradients.max().item(),
            'magnitude_min': pred_gradients.min().item()
        }
        
        # 关闭图形以释放内存
        plt.close(fig)
    
    return {
        'all_results': all_results,
        'available_atom_types': atom_types
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

