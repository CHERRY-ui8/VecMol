import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import imageio.v2 as imageio
from omegaconf import OmegaConf
import os
import sys
from typing import Optional, Tuple, List, Dict, Any, Union
from pathlib import Path
from funcmol.utils.constants import PADDING_INDEX
from funcmol.utils.utils_nf import create_neural_field, load_neural_field
from funcmol.utils.gnf_converter import GNFConverter
from lightning import Fabric
from torch_geometric.utils import to_dense_batch
from funcmol.dataset.dataset_field import create_field_loaders
import hydra

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 禁用PyTorch Dynamo编译以避免torch_cluster兼容性问题
import torch._dynamo
torch._dynamo.config.suppress_errors = True

# 全局变量控制使用哪种场
# FIELD_OPTION = 1: 使用标准答案梯度场 (ground truth)
# FIELD_OPTION = 2: 使用神经网络预测的梯度场 (predicted)
FIELD_OPTION = 1

class MoleculeMetrics:
    """分子重建指标计算类"""
    
    @staticmethod
    def compute_rmsd(coords1: torch.Tensor, coords2: torch.Tensor) -> float:
        """计算两个坐标集之间的对称RMSD"""
        dist1 = torch.sqrt(torch.sum((coords1.unsqueeze(1) - coords2.unsqueeze(0))**2, dim=2) + 1e-8)
        dist2 = torch.sqrt(torch.sum((coords2.unsqueeze(1) - coords1.unsqueeze(0))**2, dim=2) + 1e-8)
        min_dist1 = torch.min(dist1, dim=1)[0]
        min_dist2 = torch.min(dist2, dim=1)[0]
        rmsd = (torch.mean(min_dist1) + torch.mean(min_dist2)) / 2
        return rmsd.item()
    
    @staticmethod
    def compute_reconstruction_loss(coords: torch.Tensor, points: torch.Tensor) -> float:
        """计算重建损失"""
        dist1 = torch.sum((coords.unsqueeze(1) - points.unsqueeze(0))**2, dim=2)
        min_dist_to_samples = torch.min(dist1 + 1e-8, dim=1)[0]
        min_dist_to_atoms = torch.min(dist1 + 1e-8, dim=0)[0]
        coverage_loss = torch.mean(min_dist_to_samples)
        clustering_loss = torch.mean(min_dist_to_atoms)
        total_loss = coverage_loss + 0.1 * clustering_loss
        return torch.sqrt(total_loss).item()
    
    @staticmethod
    def compute_kl_divergences(coords1: torch.Tensor, coords2: torch.Tensor, temperature: float = 0.1) -> Tuple[float, float]:
        """计算两个坐标集之间的双向KL散度"""
        dist_matrix = torch.sum((coords1.unsqueeze(1) - coords2.unsqueeze(0))**2, dim=2)
        p1_given_2 = torch.softmax(-dist_matrix / temperature, dim=0)
        p2_given_1 = torch.softmax(-dist_matrix / temperature, dim=1)
        eps = 1e-8
        kl_1to2 = torch.mean(torch.sum(p1_given_2 * torch.log(p1_given_2 / (1.0/coords2.shape[0] + eps) + eps), dim=0))
        kl_2to1 = torch.mean(torch.sum(p2_given_1 * torch.log(p2_given_1 / (1.0/coords1.shape[0] + eps) + eps), dim=1))
        return kl_1to2.item(), kl_2to1.item()

class MoleculeVisualizer:
    """分子结构可视化基类"""
    
    def __init__(self):
        self.atom_colors = {
            0: 'black',   # C
            1: 'gray',    # H
            2: 'red',     # O
            3: 'blue',    # N
            4: 'green',   # F
        }
        
        # 设置matplotlib样式
        plt.style.use('default')
        plt.rcParams['figure.dpi'] = 150
        plt.rcParams['savefig.dpi'] = 300
    
    def _get_atom_colors(self, atom_types: torch.Tensor) -> List[str]:
        """根据原子类型获取颜色列表"""
        return [self.atom_colors.get(int(atom_type), 'gray') for atom_type in atom_types]
    
    def _setup_3d_axis(self, ax: plt.Axes, coords_list: List[np.ndarray], margin: float = 2.0):
        """设置3D坐标轴的共同属性"""
        ax.set_xlabel('X (Å)')
        ax.set_ylabel('Y (Å)')
        ax.set_zlabel('Z (Å)')
        ax.grid(True, alpha=0.3)
        
        # 设置统一的坐标范围
        all_coords = np.vstack(coords_list)
        x_min, x_max = all_coords[:, 0].min() - margin, all_coords[:, 0].max() + margin
        y_min, y_max = all_coords[:, 1].min() - margin, all_coords[:, 1].max() + margin
        z_min, z_max = all_coords[:, 2].min() - margin, all_coords[:, 2].max() + margin
        
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)

class GNFVisualizer(MoleculeVisualizer):
    """GNF重建过程可视化类"""
    
    def __init__(self, output_dir: str = "gnf_reconstruction_results"):
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
        """可视化原始分子和重建分子的对比"""
        fig = plt.figure(figsize=(12, 6))
        
        # 原始分子
        ax1 = fig.add_subplot(121, projection='3d')
        orig_coords_np = orig_coords.detach().cpu().numpy()
        if orig_types is not None:
            colors = self._get_atom_colors(orig_types)
            for coord, color in zip(orig_coords_np, colors):
                ax1.scatter(coord[0], coord[1], coord[2], c=[color], marker='o', s=100, alpha=0.8)
        else:
            ax1.scatter(orig_coords_np[:, 0], orig_coords_np[:, 1], orig_coords_np[:, 2], 
                       c='blue', marker='o', s=100, label='Original', alpha=0.8)
        ax1.set_title("Original Molecule")
        
        # 重建分子
        ax2 = fig.add_subplot(122, projection='3d')
        recon_coords_np = recon_coords.detach().cpu().numpy()
        if recon_types is not None:
            colors = self._get_atom_colors(recon_types)
            for coord, color in zip(recon_coords_np, colors):
                ax2.scatter(coord[0], coord[1], coord[2], c=[color], marker='o', s=100, alpha=0.8)
        else:
            ax2.scatter(recon_coords_np[:, 0], recon_coords_np[:, 1], recon_coords_np[:, 2], 
                       c='red', marker='o', s=100, label='Reconstructed', alpha=0.8)
        ax2.set_title("Reconstructed Molecule")
        
        # 设置坐标轴
        for ax in [ax1, ax2]:
            self._setup_3d_axis(ax, [orig_coords_np, recon_coords_np])
        
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
        """可视化重建过程的单个步骤"""
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # 绘制原始分子
        coords_np = coords.detach().cpu().numpy()
        if coords_types is not None:
            colors = self._get_atom_colors(coords_types)
            for i, (coord, color) in enumerate(zip(coords_np, colors)):
                ax.scatter(coord[0], coord[1], coord[2], 
                          c=[color], marker='o', s=100, alpha=0.8,
                          label='Original' if i == 0 else "")
        else:
            ax.scatter(coords_np[:, 0], coords_np[:, 1], coords_np[:, 2], 
                      c='blue', marker='o', s=100, label='Original', alpha=0.8)
        
        # 绘制当前重建点
        points_np = current_points.detach().cpu().numpy()
        if points_types is not None:
            colors = self._get_atom_colors(points_types)
            for point, color in zip(points_np, colors):
                ax.scatter(point[0], point[1], point[2],
                          c=[color], marker='.', s=5, alpha=0.4)
        else:
            ax.scatter(points_np[:, 0], points_np[:, 1], points_np[:, 2], 
                      c='red', marker='.', s=5, label='Current Points', alpha=0.4)
        
        self._setup_3d_axis(ax, [coords_np, points_np])
        ax.set_title(f"Reconstruction Step {iteration}")
        ax.legend()
        
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
                                     field_option: int = 1,
                                     sample_idx: int = 1) -> Dict[str, Any]:
        """创建重建过程动画
        Args:
            gt_coords: 真实分子坐标
            gt_types: 真实原子类型
            converter: GNF转换器
            decoder: 解码器模型
            codes: 编码
            save_interval: 保存关键帧的间隔
            animation_name: 动画文件名
            field_option: 使用哪种场
                1: 使用标准答案梯度场 (ground truth)
                2: 使用神经网络预测的梯度场 (predicted)
            sample_idx: 要可视化的分子样本索引
        """
        device = gt_coords.device
        
        # 准备真实数据
        gt_mask = (gt_types[sample_idx] != PADDING_INDEX)  # [n_atoms]
        gt_valid_coords = gt_coords[sample_idx][gt_mask]  # [n_valid_atoms, 3]
        gt_valid_types = gt_types[sample_idx][gt_mask]  # [n_valid_atoms]
        
        # QM9数据集中所有可能的原子类型
        all_atom_types = list(range(5))  # [0, 1, 2, 3, 4] 对应 [C, H, O, N, F]
        
        # 为每种可能的原子类型初始化采样点
        z_dict = {}  # 存储每种原子类型的采样点
        for atom_type in all_atom_types:
            z_dict[atom_type] = torch.rand(converter.n_query_points, 3, device=device) * 2 - 1
        
        # 记录重建过程
        frame_paths = []
        metrics_history = {
            'iterations': [],
            'loss': [],
            'rmsd': [],
            'kl_1to2': [],
            'kl_2to1': []
        }
        
        # 重建迭代
        for i in range(converter.n_iter):
            # 计算梯度场并更新点位置
            with torch.no_grad():
                # 对每种可能的原子类型分别处理
                for atom_type in all_atom_types:
                    z = z_dict[atom_type]
                    z_batch = z.unsqueeze(0)  # [1, n_query_points, 3]
                    
                    if field_option == 1:  # 使用标准答案梯度场
                        # 只选择当前类型的原子
                        type_mask = gt_valid_types == atom_type
                        type_coords = gt_valid_coords[type_mask]
                        type_types = gt_valid_types[type_mask]
                        
                        if len(type_coords) > 0:  # 如果存在这种类型的原子
                            gt_field = converter.mol2gnf(
                                type_coords.unsqueeze(0),  # [1, n_type_atoms, 3]
                                type_types.unsqueeze(0),  # [1, n_type_atoms]
                                z_batch  # [1, n_query_points, 3]
                            )
                            field = gt_field[0]  # [n_query_points, n_atom_types, 3]
                            # 只使用当前类型的梯度场
                            grad = field[:, atom_type]  # [n_query_points, 3]
                            z_dict[atom_type] = z + converter.step_size * grad
                        # 如果不存在这种类型的原子,采样点保持不动
                    else:  # 使用神经网络预测的梯度场
                        field = decoder(z_batch, codes[sample_idx:sample_idx+1])[0]
                        # 只使用当前类型的梯度场
                        grad = field[:, atom_type]  # [n_query_points, 3]
                        z_dict[atom_type] = z + converter.step_size * grad
            
            # 保存关键帧
            if i % save_interval == 0 or i == converter.n_iter - 1:
                frame_path = os.path.join(self.output_dir, f"frame_{i:04d}.png")
                
                # 合并所有类型的点
                all_points = []
                all_types = []
                for atom_type in all_atom_types:
                    points = z_dict[atom_type]
                    # 使用converter的点合并逻辑
                    merged_points = torch.from_numpy(
                        converter._merge_points(points.detach().cpu().numpy())
                    ).to(device)
                    if len(merged_points) > 0:  # 只添加密度足够大的点
                        all_points.append(merged_points)
                        all_types.extend([atom_type] * len(merged_points))
                
                if all_points:  # 如果有点被保留
                    current_points = torch.cat(all_points, dim=0)
                    current_types = torch.tensor(all_types, device=device)
                else:  # 如果没有点被保留
                    current_points = torch.empty((0, 3), device=device)
                    current_types = torch.empty((0,), device=device)
                
                self.visualize_reconstruction_step(
                    gt_valid_coords, current_points, i, frame_path, 
                    gt_valid_types, current_types  # 添加类型信息用于可视化
                )
                frame_paths.append(frame_path)
                
                # 计算指标
                if len(current_points) > 0:
                    metrics_history['iterations'].append(i)
                    metrics_history['loss'].append(
                        self.metrics.compute_reconstruction_loss(gt_valid_coords, current_points)
                    )
                    metrics_history['rmsd'].append(
                        self.metrics.compute_rmsd(gt_valid_coords, current_points)
                    )
                    kl_1to2, kl_2to1 = self.metrics.compute_kl_divergences(
                        gt_valid_coords, current_points
                    )
                    metrics_history['kl_1to2'].append(kl_1to2)
                    metrics_history['kl_2to1'].append(kl_2to1)
        
        # 生成GIF
        gif_path = os.path.join(self.output_dir, f"{animation_name}.gif")
        with imageio.get_writer(gif_path, mode='I', duration=0.3) as writer:
            for frame_path in frame_paths:
                writer.append_data(imageio.imread(frame_path))
                os.remove(frame_path)
        
        # 合并最终结果
        final_points = []
        final_types = []
        for atom_type in all_atom_types:
            points = z_dict[atom_type]
            merged_points = torch.from_numpy(
                converter._merge_points(points.detach().cpu().numpy())
            ).to(device)
            if len(merged_points) > 0:
                final_points.append(merged_points)
                final_types.extend([atom_type] * len(merged_points))
        
        if final_points:
            final_points = torch.cat(final_points, dim=0)
            final_types = torch.tensor(final_types, device=device)
        else:
            final_points = torch.empty((0, 3), device=device)
            final_types = torch.empty((0,), device=device)
        
        # 保存最终对比图
        comparison_path = os.path.join(self.output_dir, f"{animation_name}_final.png")
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
                           output_dir: str = "reconstruction_results",
                           sample_idx: int = 0) -> Dict[str, Any]:
    """便捷函数：可视化分子重建过程
    Args:
        gt_coords: 真实分子坐标
        gt_types: 真实原子类型
        converter: GNF转换器
        decoder: 解码器模型
        codes: 编码
        output_dir: 输出目录
        sample_idx: 要可视化的分子样本索引
    """
    visualizer = GNFVisualizer(output_dir)
    return visualizer.create_reconstruction_animation(
        gt_coords, gt_types, converter, decoder, codes,
        sample_idx=sample_idx
    )

def main():
    """
    主函数：加载训练好的模型和数据，执行GNF可视化
    """
    def setup_environment():
        """初始化运行环境"""
        print("GNF Visualizer for QM9 Dataset")
        
        # 初始化Fabric
        fabric = Fabric(
            accelerator="auto",
            devices=1,
            precision="32-true",
            strategy="auto"
        )
        fabric.launch()
        
        # 设置设备
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        return fabric, device
    
    def load_config():
        """加载配置文件"""
        config_path = Path(__file__).parent / "configs" / "train_nf_qm9.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        # 使用hydra加载配置
        with hydra.initialize_config_dir(config_dir=str(config_path.parent), version_base=None):
            config = hydra.compose(config_name="train_nf_qm9")
        
        # 设置数据集路径
        config["dset"]["data_dir"] = str(project_root / "funcmol" / "dataset" / "data")
        print(f"Dataset directory: {config['dset']['data_dir']}")
        
        return config
    
    def load_model(fabric, config):
        """加载预训练模型"""
        # 查找最新的实验目录
        exp_dir = Path(__file__).parent / "exps" / "neural_field"
        if not exp_dir.exists():
            raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")
        
        exp_dirs = [d for d in exp_dir.iterdir() if d.is_dir() and d.name.startswith("nf_qm9_")]
        if not exp_dirs:
            raise FileNotFoundError("No experiment directories found")
        
        latest_exp_dir = max(exp_dirs, key=lambda x: x.stat().st_mtime)
        model_path = latest_exp_dir / "model.pt"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        print(f"Loading model from: {model_path}")
        
        # 创建并加载模型
        enc, dec = create_neural_field(config, fabric)
        
        # 完全禁用模型编译
        if hasattr(enc, '_orig_mod'):
            enc = enc._orig_mod
        if hasattr(dec, '_orig_mod'):
            dec = dec._orig_mod
        
        # 加载权重
        checkpoint = fabric.load(str(model_path))
        enc = load_neural_field(checkpoint, fabric, config)[0]
        dec = load_neural_field(checkpoint, fabric, config)[1]
        print("Model loaded successfully!")
        
        return enc, dec
    
    def create_converter(config, device):
        """创建GNF转换器"""
        # 从配置文件中读取GNF转换器参数
        gnf_config = config.get("converter", {}) or config.get("gnf_converter", {})
        sigma_ratios = gnf_config.get("sigma_ratios", None)
        if sigma_ratios is not None and not isinstance(sigma_ratios, dict):
            sigma_ratios = OmegaConf.to_container(sigma_ratios, resolve=True)
        
        return GNFConverter(
            sigma=gnf_config.get("sigma", 0.2),
            n_query_points=gnf_config.get("n_query_points", config["dset"]["n_points"]),
            n_iter=gnf_config.get("n_iter", 5000),
            step_size=gnf_config.get("step_size", 0.01),
            merge_threshold=gnf_config.get("merge_threshold", 5),
            device=device,
            sigma_ratios=sigma_ratios
        )
    
    def prepare_data(fabric, config, device):
        """准备数据"""
        # 创建数据加载器
        loader_val = create_field_loaders(config, split="val", fabric=fabric)
        
        # 获取一个批次的数据
        batch = next(iter(loader_val)).to(device)
        
        # 转换为稠密张量
        coords, _ = to_dense_batch(batch.pos, batch.batch, fill_value=0)
        atoms_channel, _ = to_dense_batch(batch.x, batch.batch, fill_value=PADDING_INDEX)
        
        return batch, coords, atoms_channel
    
    def run_visualization(visualizer, gt_coords, gt_types, converter, decoder, codes, sample_idx: int):
        """执行可视化"""
        try:
            results = visualizer.create_reconstruction_animation(
                gt_coords=gt_coords,
                gt_types=gt_types,
                converter=converter,
                decoder=decoder,
                codes=codes,
                save_interval=50,
                animation_name="recon",
                field_option=FIELD_OPTION,
                sample_idx=sample_idx
            )
            
            # 打印结果
            print("\n=== 重建结果 ===")
            print(f"RMSD: {results['final_rmsd']:.4f}")
            print(f"Reconstruction Loss: {results['final_loss']:.4f}")
            print(f"KL Divergence (orig->recon): {results['final_kl_1to2']:.4f}")
            print(f"KL Divergence (recon->orig): {results['final_kl_2to1']:.4f}")
            
            print("\n输出文件:")
            print(f"  GIF动画: {results['gif_path']}")
            print(f"  对比图: {results['comparison_path']}")
            
            return results
            
        except Exception as e:
            print(f"Error during visualization: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    # 1. 初始化环境
    fabric, device = setup_environment()
        
    # 2. 加载配置
    config = load_config()
        
    # 3. 加载模型
    encoder, decoder = load_model(fabric, config)
        
    # 4. 创建转换器
    converter = create_converter(config, device)
        
    # 5. 准备数据
    batch, gt_coords, gt_types = prepare_data(fabric, config, device)
        
    # 6. 生成编码
    print("Generating codes...")
    with torch.no_grad():
        codes = encoder(batch)
        
    # 7. 执行可视化
    print("Starting visualization...")
        
    # 可以通过命令行参数或环境变量来控制sample_idx
    sample_idx = int(os.environ.get("SAMPLE_IDX", "0"))
    print(f"Visualizing molecule sample {sample_idx}")
        
    output_dir = f"gnf_visualization_results_sample_{sample_idx}"
    visualizer = GNFVisualizer(output_dir)
        
    results = run_visualization(visualizer, gt_coords, gt_types, converter, decoder, codes, sample_idx=sample_idx)
    if results is not None:
        print("GNF visualization completed successfully!")
        
if __name__ == "__main__":
    main()