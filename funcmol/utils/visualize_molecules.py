import torch
import numpy as np
import os
from typing import Tuple, Optional
from funcmol.utils.gnf_converter import GNFConverter
from funcmol.utils.constants import ELEMENTS_HASH, ELEMENTS_HASH_INV

def visualize_molecule_comparison(
    original_coords,
    original_types,
    reconstructed_coords,
    reconstructed_types,
    save_path=None,
    title="Molecule Comparison"
):
    """Visualize original and reconstructed molecules side by side using PyMOL command line."""
    # 创建临时PDB文件
    original_pdb = 'original.pdb'
    reconstructed_pdb = 'reconstructed.pdb'
    pml_script = 'visualize.pml'
    
    # 确定会话文件路径
    session_path = None
    if save_path:
        session_dir = os.path.join(os.path.dirname(save_path).replace('visualizations', 'pymol_sessions'))
        os.makedirs(session_dir, exist_ok=True)
        session_name = os.path.basename(save_path).replace('.png', '.pse')
        session_path = os.path.join(session_dir, session_name)

    try:
        # 保存分子结构
        create_pdb(original_coords, original_types, original_pdb)
        create_pdb(reconstructed_coords, reconstructed_types, reconstructed_pdb)
        
        if save_path:
            # 创建PyMOL脚本
            with open(pml_script, 'w') as f:
                script_content = f'''
# 加载分子
load {original_pdb}, original
load {reconstructed_pdb}, reconstructed

# 对齐重建分子到原始分子
align reconstructed, original

# 设置显示参数
hide all
show spheres, original
show spheres, reconstructed
set sphere_scale, 0.3

# 为重建分子设置特定颜色和透明度
set transparency, 0.5, reconstructed
set transparency, 0, original

# 设置视图
orient

# 保存图像
png {save_path}, ray=1, width=1200, height=600
'''
                if session_path:
                    script_content += f'''
# 保存PyMOL会话文件
save {session_path}
'''
                script_content += '''
# 退出PyMOL
quit
'''
                f.write(script_content)
            
            # 使用命令行运行PyMOL
            import subprocess
            subprocess.run(['pymol', '-qc', pml_script], check=True)
            
    except Exception as e:
        print(f"Error in visualization: {str(e)}")
    
    finally:
        # 清理临时文件
        for f in [original_pdb, reconstructed_pdb, pml_script]:
            try:
                if os.path.exists(f):
                    os.remove(f)
            except:
                pass

def create_pdb(coords, atom_types, output_file):
    """Create a PDB file from coordinates and atom types."""
    with open(output_file, 'w') as f:
        # 转换为numpy数组
        if torch.is_tensor(coords):
            coords = coords.detach().cpu().numpy()
        if torch.is_tensor(atom_types):
            atom_types = atom_types.detach().cpu().numpy()
            
        # 确保维度正确
        if len(coords.shape) == 1:  # [3]
            coords = coords.reshape(1, 1, 3)  # [1, 1, 3]
            n_atoms = 1
        elif len(coords.shape) == 2:  # [N, 3]
            n_atoms = coords.shape[0]  # 使用第一个维度作为原子数量
            coords = coords.reshape(1, n_atoms, 3)  # [1, N, 3]
        elif len(coords.shape) == 3:  # [B, N, 3]
            n_atoms = coords.shape[1]
        else:
            raise ValueError(f"Unexpected coords shape: {coords.shape}")
            
        if len(atom_types.shape) == 1:  # [N]
            atom_types = atom_types.reshape(1, -1)  # [1, N]
        elif len(atom_types.shape) == 2:  # [B, N]
            pass
        else:
            raise ValueError(f"Unexpected atom_types shape: {atom_types.shape}")
            
        # 如果atom_types数量不足，用最后一个类型填充
        if atom_types.shape[1] < n_atoms:
            last_type = atom_types[0, -1]
            padding = np.full((1, n_atoms - atom_types.shape[1]), last_type)
            atom_types = np.concatenate([atom_types, padding], axis=1)
        
        # 写入PDB文件
        for i in range(n_atoms):
            coord = coords[0, i]
            atom_type = ELEMENTS_HASH_INV[atom_types[0, i]]
            f.write(f"HETATM{i+1:5d} {atom_type:4s} MOL A{i+1:4d}    {coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}  1.00  0.00\n")
        f.write("END\n")

def visualize_batch_comparison(
    original_coords: torch.Tensor,
    original_types: torch.Tensor,
    reconstructed_coords: torch.Tensor,
    reconstructed_types: Optional[torch.Tensor] = None,
    save_dir: str = "molecule_comparisons",
    prefix: str = "molecule"
) -> None:
    """
    使用PyMOL可视化一批分子的原始结构和重建结构对比
    
    Args:
        original_coords: 原始分子原子坐标 [batch_size, n_atoms, 3]
        original_types: 原始分子原子类型 [batch_size, n_atoms]
        reconstructed_coords: 重建分子原子坐标 [batch_size, n_atoms, 3]
        reconstructed_types: 重建分子原子类型 [batch_size, n_atoms]
        save_dir: 保存图片的目录
        prefix: 图片文件名前缀
    """
    os.makedirs(save_dir, exist_ok=True)
    
    batch_size = original_coords.size(0)
    for i in range(batch_size):
        save_path = os.path.join(save_dir, f"{prefix}_{i}.png")
        visualize_molecule_comparison(
            original_coords[i],
            original_types[i],
            reconstructed_coords[i],
            reconstructed_types[i] if reconstructed_types is not None else None,
            save_path=save_path,
            title=f"Molecule {i} Comparison"
        )

def visualize_reconstruction_process(
    original_coords: torch.Tensor,
    original_types: torch.Tensor,
    gnf_converter: GNFConverter,
    query_points: torch.Tensor,
    save_path: str = "reconstruction_process.png",
    n_steps: int = 5
) -> None:
    """
    使用PyMOL可视化重建过程中的中间状态
    
    Args:
        original_coords: 原始分子原子坐标 [n_atoms, 3]
        original_types: 原始分子原子类型 [n_atoms]
        gnf_converter: GNFConverter实例
        query_points: 查询点坐标 [n_points, 3]
        save_path: 保存图片的路径
        n_steps: 要可视化的中间步骤数
    """
    try:
        import pymol
        pymol.finish_launching(['pymol', '-qc'])
    except ImportError:
        raise ImportError("请先安装PyMOL: pip install pymol")

    # 计算梯度场
    vector_field = gnf_converter.mol2gnf(original_coords, query_points)
    
    # 创建原始分子的PDB文件
    def create_pdb(coords, types, filename):
        with open(filename, 'w') as f:
            f.write("TITLE     MOLECULE STRUCTURE\n")
            f.write("REMARK    GENERATED BY FUNCMOL\n")
            atom_type_map = {0: 'C', 1: 'H', 2: 'O', 3: 'N', 4: 'F'}
            for i, (coord, type_idx) in enumerate(zip(coords, types)):
                atom_type = atom_type_map.get(type_idx.item(), 'X')
                f.write(f"HETATM{i+1:5d} {atom_type:4s} MOL A{i+1:4d}    {coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}  1.00  0.00\n")
            f.write("END\n")

    # 创建梯度场点的PDB文件
    def create_field_pdb(points, magnitudes, filename):
        with open(filename, 'w') as f:
            f.write("TITLE     GRADIENT FIELD\n")
            f.write("REMARK    GENERATED BY FUNCMOL\n")
            for i, (point, mag) in enumerate(zip(points, magnitudes)):
                f.write(f"HETATM{i+1:5d} X    FLD A{i+1:4d}    {point[0]:8.3f}{point[1]:8.3f}{point[2]:8.3f}  1.00  {mag:4.2f}\n")
            f.write("END\n")

    # 创建并保存文件
    original_pdb = "original.pdb"
    field_pdb = "field.pdb"
    create_pdb(original_coords, original_types, original_pdb)
    field_magnitudes = torch.norm(vector_field, dim=-1).detach().cpu().numpy()
    create_field_pdb(query_points.detach().cpu().numpy(), field_magnitudes, field_pdb)

    # 加载到PyMOL
    pymol.cmd.load(original_pdb, "original")
    pymol.cmd.load(field_pdb, "field")

    # 设置显示样式
    pymol.cmd.hide("everything")
    pymol.cmd.show("spheres", "original")
    pymol.cmd.show("points", "field")
    
    # 设置原子颜色
    pymol.cmd.color("gray", "original and name C")
    pymol.cmd.color("white", "original and name H")
    pymol.cmd.color("red", "original and name O")
    pymol.cmd.color("blue", "original and name N")
    pymol.cmd.color("green", "original and name F")
    
    # 设置梯度场点的颜色
    pymol.cmd.color("red", "field")
    pymol.cmd.set("sphere_scale", 0.5, "original")
    pymol.cmd.set("point_size", 2, "field")
    
    # 设置视角
    pymol.cmd.orient()
    
    # 设置背景颜色
    pymol.cmd.bg_color("white")
    
    # 保存图片
    pymol.cmd.set("ray_opaque_background", "on")
    pymol.cmd.set("ray_trace_mode", 1)
    pymol.cmd.set("ray_shadows", "off")
    pymol.cmd.png(save_path, ray=1, width=1600, height=800)
    
    # 清理临时文件
    os.remove(original_pdb)
    os.remove(field_pdb)
    
    # 关闭PyMOL
    pymol.cmd.quit() 