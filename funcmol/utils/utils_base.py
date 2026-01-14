import getpass
import lightning as L
from omegaconf import OmegaConf
import os
import torch
# from wandb.integration.lightning.fabric import WandbLogger
from lightning.fabric.strategies import DDPStrategy
# import tensorboard
import subprocess
import tempfile
import numpy as np

from funcmol.utils.constants import PADDING_INDEX


def setup_fabric(config: dict, find_unused_parameters=False) -> L.Fabric:
    """
    Sets up and initializes a Lightning Fabric environment based on the provided configuration.

    Args:
        config (dict): Configuration dictionary containing the following keys:
            - "wandb" (bool): Whether to use Weights and Biases for logging.
            - "exp_name" (str): The name of the experiment.
            - "dirname" (str): The directory name for saving logs.
            - "seed" (int): The seed for random number generation.
        find_unused_parameters (bool, optional): Whether to find unused parameters in DDP strategy. Defaults to False.

    Returns:
        L.Fabric: An initialized Lightning Fabric object.
    """
    logger = None
    if config["wandb"]:
        logger = WandbLogger(
            project="funcmol",
            entity=getpass.getuser(),
            config=OmegaConf.to_container(config),
            name=config["exp_name"],
            dir=config["dirname"],
        )

    n_devs = torch.cuda.device_count()
    torch.set_default_dtype(torch.float32)
    torch.set_float32_matmul_precision("high")
    strat_ = "ddp" if n_devs > 1 else "auto"
    if strat_ == "ddp" and find_unused_parameters:
        strat_ = DDPStrategy(find_unused_parameters=True)
    fabric = L.Fabric(
        devices=n_devs, num_nodes=1, strategy=strat_, accelerator="gpu", loggers=[logger], precision="bf16-mixed"
    )
    fabric.launch()
    fabric.seed_everything(config["seed"])
    fabric.print(f"config: {config}")
    fabric.print(f"device_count: {torch.cuda.device_count()}, world_size: {fabric.world_size}")

    return fabric


def mol2xyz(sample: dict, atom_elements = ["C", "H", "O", "N", "F", "S", "Cl", "Br", "P", "I", "B"]):
    """
    Converts molecular data from a dictionary to an XYZ format string.

    Args:
        sample (dict): A dictionary containing molecular data with keys:
            - "atoms_channel": A tensor or array-like structure containing atom type indices.
            - "coords": A tensor or array-like structure containing atomic coordinates.
        atom_elements (list, optional): A list of element symbols corresponding to atom type indices.
                                        Defaults to ["C", "H", "O", "N", "F", "S", "Cl", "Br", "P", "I", "B"].

    Returns:
        str: A string in XYZ format representing the molecular structure.
    """
    n_atoms = sample["atoms_channel"].shape[-1]
    xyz_str = str(n_atoms) + "\n\n"
    for i in range(n_atoms):
        element = sample["atoms_channel"][0, i]
        element = atom_elements[int(element.item())]
        coords = sample["coords"][0, i, :]
        element = "C" if element == "CA" else element
        line = (
            element
            + "\t"
            + str(coords[0].item())
            + "\t"
            + str(coords[1].item())
            + "\t"
            + str(coords[2].item())
        )
        xyz_str += line + "\n"
    return xyz_str


def save_xyz(mols: list, out_dir: str, fabric = None, atom_elements = ["C", "H", "O", "N", "F", "S", "Cl", "Br", "P", "I", "B"]):
    def save_xyz(mols: list, out_dir: str, fabric=None, atom_elements=["C", "H", "O", "N", "F", "S", "Cl", "Br", "P", "I", "B"]):
        """
        Save a list of molecules in XYZ format to the specified output directory.

        Parameters:
        mols (list): A list of molecule objects to be saved.
        out_dir (str): The directory where the XYZ files will be saved.
        fabric (optional): An object with a print method for logging. Default is None.
        atom_elements (list, optional): A list of atom elements to be considered. Default is ["C", "H", "O", "N", "F", "S", "Cl", "Br", "P", "I", "B"].

        Returns:
        list: A list of strings, each representing a molecule in XYZ format.

        Notes:
        - The function attempts to convert each molecule in the input list to XYZ format.
        - If a molecule is not valid, it is skipped, and a message is printed.
        - The number of valid molecules is logged using the fabric object's print method.
        - Each valid molecule is saved as a separate XYZ file in the output directory, with filenames in the format "sample_XXXXX.xyz".
        """
    molecules_xyz = []
    for i in range(len(mols)):
        try:
            mol = mols[i]
            xyz_str = mol2xyz(mol, atom_elements=atom_elements)
            molecules_xyz.append(xyz_str)
        except Exception:
            print(">> molecule not valid")
            continue
    fabric.print(f">> n valid molecules: {len(molecules_xyz)} / {len(mols)}")
    for idx, mol_xyz in enumerate(molecules_xyz):
        with open(os.path.join(out_dir, f"sample_{idx:05d}.xyz"), "w") as f:
            f.write(mol_xyz)
    return molecules_xyz


def atomlistToRadius(atomList: list, hashing: dict, device: str = "cpu") -> torch.Tensor:
    """
    Convert a list of atom names to their corresponding radii.

    Args:
        atomList (list): A list of atom names.
        hashing (dict): A dictionary containing the radii information for each atom.
        device (str, optional): The device to store the resulting tensor on. Defaults to "cpu".

    Returns:
        torch.Tensor: A tensor containing the radii for each atom in the input list.
    """
    radius = []
    for singleAtomList in atomList:
        haTMP = []
        for i in singleAtomList:
            resname, atName = i.split("_")[0], i.split("_")[2]
            if resname in hashing and atName in hashing[resname]:
                haTMP += [hashing[resname][atName]]
            else:
                haTMP += [1.0]
                print("missing ", resname, atName)
        radius += [torch.tensor(haTMP, dtype=torch.float, device=device)]
    radius = torch.torch.nn.utils.rnn.pad_sequence(
        radius, batch_first=True, padding_value=PADDING_INDEX
    )
    return radius


def convert_xyzs_to_sdf(path_xyzs: str, fname: str = None, delete: bool = True, fabric = None) -> None:
    """
    Convert all .xyz files in a specified directory to a single .sdf file using Open Babel.

    Args:
        path_xyzs (str): The path to the directory containing .xyz files.
        fname (str, optional): The name of the output .sdf file. Defaults to "molecules_obabel.sdf".
        delete (bool, optional): Whether to delete the original .xyz files after conversion. Defaults to True.
        fabric: An object with a print method for logging messages. Defaults to None.

    Returns:
        None
    """
    fname = "molecules_obabel.sdf" if fname is None else fname
    fabric.print(f">> process .xyz files and save in .sdf in {path_xyzs}")
    # -h: add hydrogens (补齐缺少的H原子)
    cmd = f"obabel {path_xyzs}/*xyz -osdf -O {path_xyzs}/{fname} --title  end -h"
    os.system(cmd)
    if delete:
        os.system(f"rm {path_xyzs}/*.xyz")


def _check_obabel_available():
    """
    Check if obabel command is available in the system.
    
    Returns:
        bool: True if obabel is available, False otherwise.
    """
    try:
        result = subprocess.run(
            ['which', 'obabel'],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def _coords_types_to_xyz_string(coords, atom_types, ele_list):
    """
    Convert atomic coordinates and types to XYZ format string.
    
    Args:
        coords: [N, 3] numpy array of atomic coordinates
        atom_types: [N,] numpy array of atom type indices
        ele_list: list of element symbols
        
    Returns:
        str: XYZ format string
    """
    # Filter out padding atoms
    valid_mask = (atom_types >= 0) & (atom_types < len(ele_list))
    if not valid_mask.any():
        return None
    
    coords = coords[valid_mask]
    types = atom_types[valid_mask].astype(int)
    
    n_atoms = coords.shape[0]
    xyz_lines = [str(n_atoms), ""]  # XYZ format: first line is atom count, second is blank
    
    for (x, y, z), t in zip(coords, types):
        symbol = ele_list[int(t)]
        xyz_lines.append(f"{symbol:>2s} {x:>15.10f} {y:>15.10f} {z:>15.10f}")
    
    return "\n".join(xyz_lines) + "\n"


def add_bonds_with_openbabel(coords, atom_type, ele_list, fallback_to_xyz_to_sdf=None, add_hydrogens=False):
    """
    Use OpenBabel to add bonds to discrete atoms and generate SDF format string.
    
    This function takes atomic coordinates and types, creates a temporary XYZ file,
    uses obabel command-line tool to convert XYZ to SDF (which automatically infers bonds),
    and returns the SDF content as a string.
    
    Args:
        coords: [N, 3] numpy array of atomic coordinates
        atom_type: [N,] numpy array of atom type indices
        ele_list: list of element symbols, e.g., ["C", "H", "O", "N", "F"]
        fallback_to_xyz_to_sdf: Optional function to use as fallback if obabel is not available.
                                Should have signature: (coords, atom_type, ele_list) -> str
        add_hydrogens: Whether to automatically add missing hydrogen atoms (default: False)
        
    Returns:
        str: SDF format string with bond information, or empty string if conversion fails.
             If obabel is not available and fallback is provided, returns fallback result.
    """
    # Ensure numpy arrays
    coords = np.asarray(coords)
    types = np.asarray(atom_type)
    
    # Basic validation
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coords must be a [N,3] array")
    if types.ndim != 1 or types.shape[0] != coords.shape[0]:
        raise ValueError("atom_type must be a 1-D array with same length as coords")
    
    # Check if obabel is available
    if not _check_obabel_available():
        if fallback_to_xyz_to_sdf is not None:
            print("Warning: obabel not found, falling back to xyz_to_sdf (no bonds)")
            return fallback_to_xyz_to_sdf(coords, types, ele_list)
        else:
            print("Warning: obabel not found and no fallback provided, returning empty string")
            return ""
    
    # Create temporary directory for files
    temp_dir = None
    temp_xyz_path = None
    temp_sdf_path = None
    
    try:
        # Create temporary directory
        temp_dir = tempfile.mkdtemp(prefix="obabel_")
        temp_xyz_path = os.path.join(temp_dir, "temp_molecule.xyz")
        temp_sdf_path = os.path.join(temp_dir, "temp_molecule.sdf")
        
        # Convert to XYZ format string
        xyz_content = _coords_types_to_xyz_string(coords, types, ele_list)
        if xyz_content is None:
            if fallback_to_xyz_to_sdf is not None:
                return fallback_to_xyz_to_sdf(coords, types, ele_list)
            return ""
        
        
        # Write temporary XYZ file
        with open(temp_xyz_path, 'w') as f:
            f.write(xyz_content)
        
        # Convert XYZ to SDF using OpenBabel (infer bonds)
        # Note: OpenBabel's -h flag does NOT work reliably for XYZ input, even with two-step process
        # (XYZ->SDF->SDF with -h), because bond inference from XYZ coordinates is often incomplete
        # for single atoms or molecules with missing hydrogens. We use manual method instead.
        cmd = ['obabel', '-ixyz', temp_xyz_path, '-osdf', '-O', temp_sdf_path]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode != 0:
            print(f"Warning: obabel conversion failed: {result.stderr}")
            if fallback_to_xyz_to_sdf is not None:
                return fallback_to_xyz_to_sdf(coords, types, ele_list)
            return ""
        
        # Check if SDF file was created
        if not os.path.exists(temp_sdf_path):
            print("Warning: obabel did not create output SDF file")
            if fallback_to_xyz_to_sdf is not None:
                return fallback_to_xyz_to_sdf(coords, types, ele_list)
            return ""
        
        # Read SDF content
        with open(temp_sdf_path, 'r') as f:
            sdf_content = f.read()
        
        # Ensure SDF ends with $$$$ terminator
        if not sdf_content.strip().endswith("$$$$"):
            sdf_content = sdf_content.rstrip() + "\n$$$$\n"
        
        # If add_hydrogens is True, use manual method to add missing hydrogens
        # Comprehensive testing shows that OpenBabel's -h flag does NOT work reliably
        # even for SDF input with bond information (success rate only 16.7%).
        # The manual method (based on valency calculation) is much more reliable.
        if add_hydrogens:
            sdf_with_h = add_hydrogens_manually_from_sdf(sdf_content)
            if sdf_with_h is not None:
                sdf_content = sdf_with_h
            else:
                print("Warning: Failed to add hydrogens manually, continuing without H addition")
        
        return sdf_content
        
    except subprocess.TimeoutExpired:
        print("Warning: obabel conversion timed out")
        if fallback_to_xyz_to_sdf is not None:
            return fallback_to_xyz_to_sdf(coords, types, ele_list)
        return ""
    except Exception as e:
        print(f"Warning: Error in add_bonds_with_openbabel: {e}")
        if fallback_to_xyz_to_sdf is not None:
            return fallback_to_xyz_to_sdf(coords, types, ele_list)
        return ""


# 原子最大价态（常见元素）
MAX_VALENCY = {
    1: 1,   # H
    6: 4,   # C
    7: 3,   # N
    8: 2,   # O
    9: 1,   # F
    15: 3,  # P
    16: 2,  # S
    17: 1,  # Cl
    35: 1,  # Br
    53: 1,  # I
}

def get_bond_order_value(bond_type):
    """将 RDKit 键类型转换为数值"""
    from rdkit import Chem
    if bond_type == Chem.BondType.SINGLE:
        return 1
    elif bond_type == Chem.BondType.DOUBLE:
        return 2
    elif bond_type == Chem.BondType.TRIPLE:
        return 3
    elif bond_type == Chem.BondType.AROMATIC:
        return 1.5  # 芳香键通常视为 1.5
    else:
        return 1

def calculate_atom_valency(mol, atom_idx):
    """计算原子的当前价态（键级之和）"""
    valency = 0
    atom = mol.GetAtomWithIdx(atom_idx)
    for bond in atom.GetBonds():
        bond_order = get_bond_order_value(bond.GetBondType())
        valency += bond_order
    return valency

def get_atom_max_valency(atomic_num):
    """获取原子的最大价态"""
    return MAX_VALENCY.get(atomic_num, 4)  # 默认 4

def _calculate_hydrogen_positions_3d(atom_pos, existing_bond_vectors, num_h, bond_length=1.0):
    """
    计算H原子在3D空间中的位置，考虑已有键的方向和合理的键角
    
    Args:
        atom_pos: 重原子的位置 (Point3D)
        existing_bond_vectors: 已有键的方向向量列表（归一化的3D向量）
        num_h: 需要添加的H数量
        bond_length: H原子到重原子的距离（默认1.0 Å）
        
    Returns:
        H原子位置的列表，每个元素是 [x, y, z] 数组
    """
    # 使用文件顶部已导入的numpy
    atom_center = np.array([atom_pos.x, atom_pos.y, atom_pos.z])
    h_positions = []
    
    if num_h == 0:
        return h_positions
    
    # 计算已有键的总方向（用于确定主要方向）
    if existing_bond_vectors:
        # 计算已有键的平均方向（加权平均，考虑键的重要性）
        total_vec = np.sum(existing_bond_vectors, axis=0)
        total_vec_norm = np.linalg.norm(total_vec)
        if total_vec_norm > 0.01:
            main_direction = total_vec / total_vec_norm
        else:
            main_direction = np.array([1.0, 0.0, 0.0])  # 默认方向
    else:
        # 没有已有键，使用默认方向
        main_direction = np.array([1.0, 0.0, 0.0])
    
    # 根据已有键的数量和需要添加的H数量，确定几何构型
    total_bonds = len(existing_bond_vectors) + num_h
    
    if total_bonds == 1:
        # 线性：H在已有键的相反方向
        if existing_bond_vectors:
            h_direction = -existing_bond_vectors[0]
        else:
            h_direction = main_direction
        h_pos = atom_center + bond_length * h_direction
        h_positions.append(h_pos.tolist())
        
    elif total_bonds == 2:
        # 线性：两个原子在一条直线上
        if existing_bond_vectors:
            # 已有1个键，H在相反方向
            h_direction = -existing_bond_vectors[0]
        else:
            # 需要添加2个H，在相反方向
            h_direction = main_direction
            h_pos1 = atom_center + bond_length * h_direction
            h_pos2 = atom_center - bond_length * h_direction
            h_positions.append(h_pos1.tolist())
            h_positions.append(h_pos2.tolist())
            return h_positions
        
        h_pos = atom_center + bond_length * h_direction
        h_positions.append(h_pos.tolist())
        
    elif total_bonds == 3:
        # 三角锥：键角约120度
        if existing_bond_vectors:
            # 已有键的方向
            existing_dir = existing_bond_vectors[0]
            # 计算垂直于已有键的方向
            if abs(existing_dir[2]) < 0.9:
                perp1 = np.cross(existing_dir, np.array([0, 0, 1]))
            else:
                perp1 = np.cross(existing_dir, np.array([1, 0, 0]))
            perp1 = perp1 / np.linalg.norm(perp1)
            perp2 = np.cross(existing_dir, perp1)
            perp2 = perp2 / np.linalg.norm(perp2)
            
            # 在垂直于已有键的平面内均匀分布H
            for i in range(num_h):
                angle = 2 * np.pi * i / num_h
                h_direction = np.cos(angle) * perp1 + np.sin(angle) * perp2
                # 调整角度，使其与已有键形成约120度角
                h_direction = 0.5 * (-existing_dir) + 0.866 * h_direction  # cos(120°) ≈ -0.5, sin(120°) ≈ 0.866
                h_direction = h_direction / np.linalg.norm(h_direction)
                h_pos = atom_center + bond_length * h_direction
                h_positions.append(h_pos.tolist())
        else:
            # 需要添加3个H，形成三角锥
            # 使用标准三角锥几何（键角约109.5度）
            # 创建一个垂直于某个方向的平面
            base_normal = np.array([0, 0, 1])
            if np.abs(np.dot(main_direction, base_normal)) > 0.9:
                base_normal = np.array([1, 0, 0])
            
            perp1 = np.cross(main_direction, base_normal)
            perp1 = perp1 / np.linalg.norm(perp1)
            perp2 = np.cross(main_direction, perp1)
            perp2 = perp2 / np.linalg.norm(perp2)
            
            # 三角锥：一个H在上方，两个H在下方平面
            h1_direction = main_direction
            h_pos1 = atom_center + bond_length * h1_direction
            h_positions.append(h_pos1.tolist())
            
            for i in range(num_h - 1):
                angle = 2 * np.pi * i / (num_h - 1)
                h_direction = np.cos(angle) * perp1 + np.sin(angle) * perp2
                # 调整角度
                h_direction = -0.333 * main_direction + 0.943 * h_direction  # 约109.5度
                h_direction = h_direction / np.linalg.norm(h_direction)
                h_pos = atom_center + bond_length * h_direction
                h_positions.append(h_pos.tolist())
    
    elif total_bonds == 4:
        # 四面体：键角约109.5度
        if existing_bond_vectors:
            # 已有键的方向
            existing_dir = existing_bond_vectors[0]
            
            # 计算四面体的其他方向
            # 使用标准四面体几何
            if len(existing_bond_vectors) == 1:
                # 已有1个键，需要添加3个H形成四面体
                # 计算垂直于已有键的平面
                if abs(existing_dir[2]) < 0.9:
                    perp1 = np.cross(existing_dir, np.array([0, 0, 1]))
                else:
                    perp1 = np.cross(existing_dir, np.array([1, 0, 0]))
                perp1 = perp1 / np.linalg.norm(perp1)
                perp2 = np.cross(existing_dir, perp1)
                perp2 = perp2 / np.linalg.norm(perp2)
                
                # 四面体：3个H在垂直于已有键的平面内，形成等边三角形
                for i in range(num_h):
                    angle = 2 * np.pi * i / num_h
                    h_direction = np.cos(angle) * perp1 + np.sin(angle) * perp2
                    # 调整角度，使其与已有键形成约109.5度角
                    h_direction = -0.333 * existing_dir + 0.943 * h_direction
                    h_direction = h_direction / np.linalg.norm(h_direction)
                    h_pos = atom_center + bond_length * h_direction
                    h_positions.append(h_pos.tolist())
            elif len(existing_bond_vectors) == 2:
                # 已有2个键，需要添加2个H
                # 计算两个已有键的夹角平分线的垂直方向
                dir1 = existing_bond_vectors[0]
                dir2 = existing_bond_vectors[1]
                bisector = (dir1 + dir2) / np.linalg.norm(dir1 + dir2)
                
                # 计算垂直于平分线的方向
                if abs(bisector[2]) < 0.9:
                    perp = np.cross(bisector, np.array([0, 0, 1]))
                else:
                    perp = np.cross(bisector, np.array([1, 0, 0]))
                perp = perp / np.linalg.norm(perp)
                
                for i in range(num_h):
                    sign = 1 if i == 0 else -1
                    h_direction = -0.333 * bisector + sign * 0.943 * perp
                    h_direction = h_direction / np.linalg.norm(h_direction)
                    h_pos = atom_center + bond_length * h_direction
                    h_positions.append(h_pos.tolist())
            else:
                # 已有3个键，需要添加1个H
                # H在已有键的相反方向（四面体的第4个顶点）
                total_existing = np.sum(existing_bond_vectors, axis=0)
                h_direction = -total_existing / np.linalg.norm(total_existing)
                h_pos = atom_center + bond_length * h_direction
                h_positions.append(h_pos.tolist())
        else:
            # 需要添加4个H，形成标准四面体
            # 使用标准四面体顶点坐标
            tetra_vectors = [
                np.array([1.0, 1.0, 1.0]),
                np.array([1.0, -1.0, -1.0]),
                np.array([-1.0, 1.0, -1.0]),
                np.array([-1.0, -1.0, 1.0])
            ]
            for vec in tetra_vectors[:num_h]:
                vec = vec / np.linalg.norm(vec)
                h_pos = atom_center + bond_length * vec
                h_positions.append(h_pos.tolist())
    
    else:
        # 对于更多键的情况，使用更通用的方法
        # 在已有键的"对面"均匀分布H
        if existing_bond_vectors:
            # 计算已有键的平均方向
            avg_direction = np.mean(existing_bond_vectors, axis=0)
            avg_direction = avg_direction / np.linalg.norm(avg_direction)
            
            # 计算垂直于平均方向的方向
            if abs(avg_direction[2]) < 0.9:
                perp1 = np.cross(avg_direction, np.array([0, 0, 1]))
            else:
                perp1 = np.cross(avg_direction, np.array([1, 0, 0]))
            perp1 = perp1 / np.linalg.norm(perp1)
            perp2 = np.cross(avg_direction, perp1)
            perp2 = perp2 / np.linalg.norm(perp2)
            
            # 在垂直于平均方向的平面内均匀分布H
            for i in range(num_h):
                angle = 2 * np.pi * i / num_h
                h_direction = np.cos(angle) * perp1 + np.sin(angle) * perp2
                # 稍微偏向已有键的相反方向
                h_direction = -0.5 * avg_direction + 0.866 * h_direction
                h_direction = h_direction / np.linalg.norm(h_direction)
                h_pos = atom_center + bond_length * h_direction
                h_positions.append(h_pos.tolist())
        else:
            # 没有已有键，在3D空间中均匀分布
            # 使用球面均匀分布
            for i in range(num_h):
                # 使用球面坐标
                theta = np.arccos(1 - 2 * (i + 0.5) / num_h)  # 极角
                phi = 2 * np.pi * i * 0.618  # 方位角（使用黄金角度）
                h_direction = np.array([
                    np.sin(theta) * np.cos(phi),
                    np.sin(theta) * np.sin(phi),
                    np.cos(theta)
                ])
                h_pos = atom_center + bond_length * h_direction
                h_positions.append(h_pos.tolist())
    
    return h_positions

def add_hydrogens_manually_from_sdf(sdf_content):
    """
    从 SDF 内容手动添加 H 原子，基于价态计算
    
    1. 解析 SDF 得到分子对象（含键信息）
    2. 对每个重原子计算当前价态（键级之和）和最大价态
    3. 如果当前价态 < 最大价态，计算需要添加的 H 数量
    4. 对于没有键的原子（价态=0），直接使用最大价态添加 H
    5. 在重原子周围均匀分布添加 H 原子（距离约 1.0 Å）
    6. 转换回 SDF 格式
    
    Args:
        sdf_content: SDF 格式字符串（应包含键信息）
        
    Returns:
        添加了 H 的新 SDF 格式字符串，或 None
    """
    try:
        from rdkit import Chem
        from rdkit.Geometry import Point3D
        
        # 解析 SDF
        mol = Chem.MolFromMolBlock(sdf_content, sanitize=False)
        if mol is None:
            return None
        
        # 尝试 sanitize（可能帮助推断键）
        try:
            Chem.SanitizeMol(mol)
        except:
            pass
        
        # 创建新的分子对象
        new_mol = Chem.RWMol(mol)
        
        # 获取原始坐标
        conf = mol.GetConformer()
        if conf is None:
            return None
        
        # 计算每个原子需要添加的 H 数量
        h_to_add = []
        for atom_idx in range(mol.GetNumAtoms()):
            atom = mol.GetAtomWithIdx(atom_idx)
            atomic_num = atom.GetAtomicNum()
            
            # 跳过 H 原子
            if atomic_num == 1:
                continue
            
            # 计算当前价态和最大价态
            current_valency = calculate_atom_valency(mol, atom_idx)
            max_valency = get_atom_max_valency(atomic_num)
            
            # 如果当前价态为 0（没有键），直接使用最大价态
            if current_valency == 0:
                needed_h = max_valency
            else:
                # 计算需要添加的 H 数量（取整，确保非负）
                needed_h = max(0, int(round(max_valency - current_valency)))
            
            # 只添加非负数量的 H，并且确保不会超过最大价态
            if needed_h > 0:
                # 额外检查：确保添加H后不会超过最大价态
                # 这里 current_valency 已经包含了所有键（包括到H的键）
                # 所以如果 current_valency + needed_h > max_valency，说明计算有误
                if current_valency + needed_h <= max_valency:
                    h_to_add.append((atom_idx, needed_h))
                else:
                    # 如果计算出的 needed_h 会导致超过最大价态，不添加
                    print(f"Warning: Atom {atom_idx} ({atom.GetSymbol()}) already has valency {current_valency:.1f}, "
                          f"cannot add {needed_h} H (would exceed max valency {max_valency})")
        
        if not h_to_add:
            # 没有需要添加的 H，返回原始 SDF
            return sdf_content
        
        # 添加 H 原子
        h_indices = []
        for atom_idx, num_h in h_to_add:
            atom = new_mol.GetAtomWithIdx(atom_idx)
            atom_pos = conf.GetAtomPosition(atom_idx)
            
            for i in range(num_h):
                # 创建 H 原子
                h_atom = Chem.Atom(1)  # H 的原子序数是 1
                h_idx = new_mol.AddAtom(h_atom)
                h_indices.append(h_idx)
                
                # 添加键
                new_mol.AddBond(atom_idx, h_idx, Chem.BondType.SINGLE)
        
        # 转换为普通分子对象
        new_mol = new_mol.GetMol()
        
        # 添加 conformer 并设置坐标
        new_conf = Chem.Conformer(new_mol.GetNumAtoms())
        # 复制原始坐标
        for i in range(mol.GetNumAtoms()):
            pos = conf.GetAtomPosition(i)
            new_conf.SetAtomPosition(i, pos)
        
        # 为新添加的 H 设置坐标（考虑已有键的方向和合理的键角）
        h_idx_counter = 0
        for atom_idx, num_h in h_to_add:
            atom = mol.GetAtomWithIdx(atom_idx)
            atom_pos = conf.GetAtomPosition(atom_idx)
            
            # 获取已有键的方向向量
            existing_bond_vectors = []
            for bond in atom.GetBonds():
                other_atom = bond.GetOtherAtom(atom)
                other_pos = conf.GetAtomPosition(other_atom.GetIdx())
                # 计算键方向向量（从重原子指向其他原子）
                bond_vec = np.array([
                    other_pos.x - atom_pos.x,
                    other_pos.y - atom_pos.y,
                    other_pos.z - atom_pos.z
                ])
                # 归一化
                bond_len = np.linalg.norm(bond_vec)
                if bond_len > 0.01:  # 避免除零
                    bond_vec = bond_vec / bond_len
                    existing_bond_vectors.append(bond_vec)
            
            # 根据已有键的数量和需要添加的H数量，计算H的位置
            h_positions = _calculate_hydrogen_positions_3d(
                atom_pos, existing_bond_vectors, num_h, bond_length=1.0
            )
            
            for i, h_pos_3d in enumerate(h_positions):
                h_idx = mol.GetNumAtoms() + h_idx_counter
                h_pos = Point3D(h_pos_3d[0], h_pos_3d[1], h_pos_3d[2])
                new_conf.SetAtomPosition(h_idx, h_pos)
                h_idx_counter += 1
        
        new_mol.AddConformer(new_conf)
        
        # 转换回 SDF 格式
        # 如果分子无法被kekulize（如某些芳香环），尝试修复键类型后再生成SDF
        sdf_with_h = None
        try:
            # 先尝试sanitize
            Chem.SanitizeMol(new_mol)
            sdf_with_h = Chem.MolToMolBlock(new_mol)
        except (Chem.rdchem.KekulizeException, Chem.rdchem.AtomValenceException) as e:
            # 如果sanitize失败（kekulization或价态错误），尝试修复键类型
            # 将芳香键改为单键，这可能有助于某些情况
            try:
                # 创建可编辑的分子对象
                editable_mol = Chem.RWMol(new_mol)
                # 遍历所有键，将芳香键改为单键
                bonds_to_fix = []
                for bond in editable_mol.GetBonds():
                    if bond.GetBondType() == Chem.BondType.AROMATIC:
                        bonds_to_fix.append((bond.GetBeginIdx(), bond.GetEndIdx()))
                
                # 修复芳香键
                for begin_idx, end_idx in bonds_to_fix:
                    editable_mol.RemoveBond(begin_idx, end_idx)
                    editable_mol.AddBond(begin_idx, end_idx, Chem.BondType.SINGLE)
                
                # 尝试sanitize修复后的分子
                fixed_mol = editable_mol.GetMol()
                try:
                    Chem.SanitizeMol(fixed_mol)
                    sdf_with_h = Chem.MolToMolBlock(fixed_mol)
                except:
                    # 如果修复后还是无法sanitize，尝试直接生成SDF（不sanitize）
                    # 注意：MolToMolBlock可能会抛出异常
                    pass
            except Exception:
                pass
            
            # 如果修复键类型后还是失败，尝试直接生成SDF（可能会失败）
            if sdf_with_h is None:
                try:
                    # 最后尝试：直接生成SDF，即使无法sanitize
                    sdf_with_h = Chem.MolToMolBlock(new_mol)
                except Exception:
                    # 如果连生成SDF都失败，返回原始SDF（不添加H）
                    # 这样至少不会丢失原始信息
                    return sdf_content
        except Exception:
            # 其他未知错误，尝试直接生成SDF
            try:
                sdf_with_h = Chem.MolToMolBlock(new_mol)
            except Exception:
                # 如果失败，返回原始SDF
                return sdf_content
        
        if sdf_with_h:
            if not sdf_with_h.strip().endswith("$$$$"):
                sdf_with_h = sdf_with_h.rstrip() + "\n$$$$\n"
            return sdf_with_h
        
        return None
        
    except Exception as e:
        print(f"Warning: Failed to add hydrogens manually: {e}")
        return None


def xyz_to_sdf(coords, atom_type, ele_list):
    """
    Convert atomic coordinates and types to SDF format string (without bonds).
    
    Args:
        coords: [N, 3] numpy array of atomic coordinates
        atom_type: [N,] numpy array of atom type indices
        ele_list: list of element symbols, e.g., ["C", "H", "O", "N", "F"]
        
    Returns:
        str: SDF format string without bond information
    """
    import numpy as np
    
    # Ensure numpy arrays
    coords = np.asarray(coords)
    types = np.asarray(atom_type)

    # Basic validation
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coords must be a [N,3] array")
    if types.ndim != 1 or types.shape[0] != coords.shape[0]:
        raise ValueError("atom_type must be a 1-D array with same length as coords")

    # Filter out padding atoms (commonly -1)
    valid_mask = (types >= 0) & (types < len(ele_list))
    if not valid_mask.any():
        # No valid atoms -> return empty string
        return ""

    coords = coords[valid_mask]
    types = types[valid_mask].astype(int)

    n_atoms = coords.shape[0]
    n_bonds = 0

    lines = []
    # Header: simple 3-line header used in MOL/SDF files
    lines.append("Generated by xyz_to_sdf")
    lines.append("")
    lines.append("")

    # Counts line (V2000 format)
    counts_line = f"{n_atoms:>3}{n_bonds:>3}  0  0  0  0            999 V2000"
    lines.append(counts_line)

    # Atom block: x, y, z, symbol and placeholder fields
    for (x, y, z), t in zip(coords, types):
        symbol = ele_list[int(t)]
        # Standard fixed-width formatting used in MOL files
        atom_line = f"{x:10.4f}{y:10.4f}{z:10.4f} {symbol:<3s} 0  0  0  0  0  0  0  0  0  0"
        lines.append(atom_line)

    # No bond block for now (n_bonds == 0)
    # Footer
    lines.append("M  END")
    # SDF multi-record terminator
    lines.append("$$$$")

    sdf_string = "\n".join(lines) + "\n"
    return sdf_string