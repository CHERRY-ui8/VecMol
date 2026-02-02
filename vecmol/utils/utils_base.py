import getpass
import lightning as L
from omegaconf import OmegaConf
import os
import sys
import torch
# from wandb.integration.lightning.fabric import WandbLogger
from lightning.fabric.strategies import DDPStrategy
# import tensorboard
import subprocess
import tempfile
import numpy as np

from vecmol.utils.constants import PADDING_INDEX, BOND_LENGTHS_PM


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
            project="vecmol",
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
    """计算原子的当前价态（优先使用 RDKit 的总价，失败时按键级之和）"""
    atom = mol.GetAtomWithIdx(atom_idx)
    
    # 优先使用 RDKit 内部计算的价态
    try:
        valency = float(atom.GetTotalValence())
        # 对于某些未sanitize或信息不完整的分子，RDKit 可能返回 0 或负值，
        # 这种情况下回退到根据键级求和的旧实现。
        if valency > 0:
            return valency
    except Exception:
        pass
    
    # 回退：根据当前键级求和（与旧逻辑保持兼容）
    valency = 0.0
    for bond in atom.GetBonds():
        bond_order = get_bond_order_value(bond.GetBondType())
        valency += bond_order
    return valency

def get_atom_max_valency(atomic_num):
    """获取原子的最大价态"""
    return MAX_VALENCY.get(atomic_num, 4)  # 默认 4

def _calculate_hydrogen_positions_3d(atom_pos, existing_bond_vectors, num_h, atom_symbol='C', bond_length=None):
    """
    计算H原子在3D空间中的位置，考虑已有键的方向和合理的键角
    
    Args:
        atom_pos: 重原子的位置 (Point3D)
        existing_bond_vectors: 已有键的方向向量列表（归一化的3D向量）
        num_h: 需要添加的H数量
        atom_symbol: 重原子的元素符号（用于获取标准键长）
        bond_length: H原子到重原子的距离（单位：Å）。如果为None，则根据原子类型从BOND_LENGTHS_PM获取标准键长
        
    Returns:
        H原子位置的列表，每个元素是 [x, y, z] 数组。如果计算失败，返回空列表（调用者需要处理）
    """
    # 根据原子类型获取标准键长
    if bond_length is None:
        # 从BOND_LENGTHS_PM获取标准键长（单位：pm，需要转换为Å）
        if atom_symbol in BOND_LENGTHS_PM and 'H' in BOND_LENGTHS_PM[atom_symbol]:
            bond_length = BOND_LENGTHS_PM[atom_symbol]['H'] / 100.0  # pm转Å
        else:
            # 如果没有找到标准键长，使用默认值（根据原子类型）
            default_bond_lengths = {
                'C': 1.09,  # C-H
                'N': 1.01,  # N-H
                'O': 0.96,  # O-H
                'S': 1.34,  # S-H
                'P': 1.44,  # P-H (144 pm)
                'F': 0.92,  # F-H
                'Cl': 1.27, # Cl-H
                'Br': 1.41, # Br-H
            }
            bond_length = default_bond_lengths.get(atom_symbol, 1.0)  # 默认1.0 Å
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
        # 三键情况（例如 sp2 / 三角平面 或 三角锥）
        if existing_bond_vectors:
            n_existing = len(existing_bond_vectors)
            # 情形 A: 已有 2 个键，只缺 1 个 H（最常见：=CH-，sp2）
            # 目标：让新的 H 方向大致指向两个已有键的“对面”，
            # 即与 v1、v2 都约 120 度：h_dir ≈ -normalize(v1 + v2)
            if n_existing >= 2 and num_h == 1:
                v_sum = np.sum(existing_bond_vectors, axis=0)
                v_norm = np.linalg.norm(v_sum)
                if v_norm > 1e-2:
                    h_direction = -v_sum / v_norm
                else:
                    # 如果两个键几乎对消（数值退化），退回到任意一个键的反方向
                    h_direction = -existing_bond_vectors[0]
                h_pos = atom_center + bond_length * h_direction
                h_positions.append(h_pos.tolist())
            # 情形 B: 已有 1 个键，需要补 2 个 H
            # 目标：与已有键一起构成三角平面（或近似），三条键之间两两约 120 度。
            elif n_existing == 1 and num_h == 2:
                existing_dir = existing_bond_vectors[0]
                # 计算一个与已有键垂直的方向 u
                if abs(existing_dir[2]) < 0.9:
                    u = np.cross(existing_dir, np.array([0, 0, 1]))
                else:
                    u = np.cross(existing_dir, np.array([1, 0, 0]))
                u = u / np.linalg.norm(u)
                # 利用 v2 = -0.5*v1 + (√3/2)*u, v3 = -0.5*v1 - (√3/2)*u
                # 保证 v1,v2,v3 两两夹角约 120°
                h_dir1 = -0.5 * existing_dir + (np.sqrt(3) / 2.0) * u
                h_dir2 = -0.5 * existing_dir - (np.sqrt(3) / 2.0) * u
                h_dir1 = h_dir1 / np.linalg.norm(h_dir1)
                h_dir2 = h_dir2 / np.linalg.norm(h_dir2)
                for h_direction in (h_dir1, h_dir2):
                    h_pos = atom_center + bond_length * h_direction
                    h_positions.append(h_pos.tolist())
            else:
                # 其他较少见的组合，退回到原有逻辑：
                # 以已有键方向的平均向量作为“主方向”，在其垂直平面内均匀分布 H，
                # 并调节到与已有键约 120 度。
                avg_vec = np.sum(existing_bond_vectors, axis=0)
                if np.linalg.norm(avg_vec) < 1e-2:
                    existing_dir = existing_bond_vectors[0]
                else:
                    existing_dir = avg_vec / np.linalg.norm(avg_vec)
                # 计算垂直于已有键平均方向的两个基向量
                if abs(existing_dir[2]) < 0.9:
                    perp1 = np.cross(existing_dir, np.array([0, 0, 1]))
                else:
                    perp1 = np.cross(existing_dir, np.array([1, 0, 0]))
                perp1 = perp1 / np.linalg.norm(perp1)
                perp2 = np.cross(existing_dir, perp1)
                perp2 = perp2 / np.linalg.norm(perp2)
                # 在垂直平面内均匀分布 H，并使其与已有键形成约 120 度
                for i in range(num_h):
                    angle = 2 * np.pi * i / num_h
                    h_direction = np.cos(angle) * perp1 + np.sin(angle) * perp2
                    h_direction = 0.5 * (-existing_dir) + 0.866 * h_direction  # cos(120°) ≈ -0.5, sin(120°) ≈ 0.866
                    h_direction = h_direction / np.linalg.norm(h_direction)
                    h_pos = atom_center + bond_length * h_direction
                    h_positions.append(h_pos.tolist())
        else:
            # 没有已有键，需要添加3个H：使用标准三角锥几何（键角约109.5度）
            # 创建一个垂直于 main_direction 的平面
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
    5. 在重原子周围均匀分布添加 H 原子，使用标准键长（根据原子类型）
       - C-H: 1.09 Å, O-H: 0.96 Å, N-H: 1.01 Å, S-H: 1.34 Å 等
    6. 使用最大排斥原则计算H原子位置，确保合理的几何构型
    7. 转换回 SDF 格式
    
    Args:
        sdf_content: SDF 格式字符串（应包含键信息）
        
    Returns:
        添加了 H 的新 SDF 格式字符串，或 None
        
    注意：
    - 所有添加的H原子都会有有效的3D坐标（不会是0,0,0）
    - 如果坐标计算失败，会使用备用方法（基于最大排斥原则的球面分布）
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
        
        # 在补H之前排除单原子碎片（只有一个原子且没有键的连通分量）
        # 避免单原子碎片（如Cl、Br）被补H后无法识别
        # 保存原始分子的conformer（用于后续坐标复制）
        original_conf = mol.GetConformer() if mol.GetNumConformers() > 0 else None
        
        if mol.GetNumBonds() > 0 and mol.GetNumAtoms() > 1:
            # 获取所有连通分量
            mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
            if len(mol_frags) > 1:
                # 过滤掉单原子碎片（只有一个原子且没有键的连通分量）
                valid_frags = [frag for frag in mol_frags if frag.GetNumBonds() > 0]
                if len(valid_frags) > 0:
                    # 如果只有一个有效片段，使用它
                    if len(valid_frags) == 1:
                        # 找到这个有效片段在原始分子中的原子索引
                        frag_atom_indices = Chem.rdmolops.GetMolFrags(mol, asMols=False)
                        valid_frag_idx = None
                        for i, frag in enumerate(mol_frags):
                            if frag.GetNumBonds() > 0:
                                valid_frag_idx = i
                                break
                        
                        if valid_frag_idx is not None:
                            valid_atom_indices = frag_atom_indices[valid_frag_idx]
                            mol = valid_frags[0]
                            # 从原始分子复制conformer到新分子
                            if original_conf is not None:
                                new_conf = Chem.Conformer(mol.GetNumAtoms())
                                for new_idx, old_idx in enumerate(valid_atom_indices):
                                    pos = original_conf.GetAtomPosition(old_idx)
                                    new_conf.SetAtomPosition(new_idx, pos)
                                mol.RemoveAllConformers()
                                mol.AddConformer(new_conf)
                        else:
                            mol = valid_frags[0]
                    else:
                        # 如果有多个有效片段，合并它们（保留所有有键的片段）
                        new_mol = Chem.RWMol()
                        atom_map = {}  # 原始分子中的原子索引 -> 新分子中的原子索引
                        
                        # 获取所有有效片段的原子索引
                        frag_atom_indices = Chem.rdmolops.GetMolFrags(mol, asMols=False)
                        all_valid_atom_indices = []
                        for frag, atom_indices in zip(mol_frags, frag_atom_indices):
                            if frag.GetNumBonds() > 0:
                                all_valid_atom_indices.extend(atom_indices)
                        all_valid_atom_indices = sorted(all_valid_atom_indices)
                        
                        # 添加所有有效原子
                        for old_idx in all_valid_atom_indices:
                            atom = mol.GetAtomWithIdx(old_idx)
                            new_idx = new_mol.AddAtom(atom)
                            atom_map[old_idx] = new_idx
                        
                        # 添加所有有效片段的键
                        for frag, atom_indices in zip(mol_frags, frag_atom_indices):
                            if frag.GetNumBonds() > 0:
                                frag_atom_map = {old_idx: atom_map[old_idx] for old_idx in atom_indices}
                                for bond in frag.GetBonds():
                                    begin_old = bond.GetBeginAtomIdx()
                                    end_old = bond.GetEndAtomIdx()
                                    begin_orig = atom_indices[begin_old]
                                    end_orig = atom_indices[end_old]
                                    if begin_orig in frag_atom_map and end_orig in frag_atom_map:
                                        begin_new = frag_atom_map[begin_orig]
                                        end_new = frag_atom_map[end_orig]
                                        new_mol.AddBond(begin_new, end_new, bond.GetBondType())
                        
                        # 转换为普通分子对象
                        mol = new_mol.GetMol()
                        
                        # 从原始分子复制坐标
                        if original_conf is not None:
                            new_conf = Chem.Conformer(mol.GetNumAtoms())
                            for old_idx, new_idx in atom_map.items():
                                pos = original_conf.GetAtomPosition(old_idx)
                                new_conf.SetAtomPosition(new_idx, pos)
                            mol.AddConformer(new_conf)
                else:
                    # 如果没有有效片段（所有片段都是单原子碎片），返回None
                    return None
        
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
        
        if not h_to_add:
            # 没有需要添加的 H，返回原始 SDF
            return sdf_content
        
        # 添加 H 原子
        # 使用 h_map 记录每个 H 原子与其父原子的对应关系: [(parent_atom_idx, h_idx), ...]
        h_map = []
        original_atom_count = new_mol.GetNumAtoms()
        for atom_idx, num_h in h_to_add:
            atom = new_mol.GetAtomWithIdx(atom_idx)
            atom_pos = conf.GetAtomPosition(atom_idx)
            
            for i in range(num_h):
                # 创建 H 原子
                h_atom = Chem.Atom(1)  # H 的原子序数是 1
                h_idx = new_mol.AddAtom(h_atom)
                h_map.append((atom_idx, h_idx))  # 记录父原子索引和 H 原子索引
                
                # 添加键
                new_mol.AddBond(atom_idx, h_idx, Chem.BondType.SINGLE)
        
        # 转换为普通分子对象
        new_mol = new_mol.GetMol()
        
        # 删除所有旧的conformer（如果有），确保新conformer是ID=0
        # 这样MolToMolBlock就会使用我们的新conformer
        while new_mol.GetNumConformers() > 0:
            new_mol.RemoveConformer(0)
        
        # 添加 conformer 并设置坐标
        new_conf = Chem.Conformer(new_mol.GetNumAtoms())
        # 复制原始坐标
        for i in range(mol.GetNumAtoms()):
            pos = conf.GetAtomPosition(i)
            new_conf.SetAtomPosition(i, pos)
        
        # 为新添加的 H 设置坐标（考虑已有键的方向和合理的键角）
        # 使用 h_map 来正确映射每个 H 原子的索引
        h_map_idx = 0  # 用于遍历 h_map 的索引
        for atom_idx, num_h in h_to_add:
            atom = mol.GetAtomWithIdx(atom_idx)
            atom_symbol = atom.GetSymbol()
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
                atom_pos, existing_bond_vectors, num_h, atom_symbol=atom_symbol
            )
            
            # 如果坐标计算失败（返回空列表），使用备用方法
            if len(h_positions) < num_h:
                # 备用方法：基于最大排斥原则，在3D空间中均匀分布H
                atom_center = np.array([atom_pos.x, atom_pos.y, atom_pos.z])
                
                # 获取标准键长
                if atom_symbol in BOND_LENGTHS_PM and 'H' in BOND_LENGTHS_PM[atom_symbol]:
                    bond_length = BOND_LENGTHS_PM[atom_symbol]['H'] / 100.0
                else:
                    default_bond_lengths = {
                        'C': 1.09, 'N': 1.01, 'O': 0.96, 'S': 1.34,
                        'P': 1.44, 'F': 0.92, 'Cl': 1.27, 'Br': 1.41,
                    }
                    bond_length = default_bond_lengths.get(atom_symbol, 1.0)
                
                # 计算已有键的平均方向（用于确定主要方向）
                if existing_bond_vectors:
                    main_direction = np.mean(existing_bond_vectors, axis=0)
                    main_direction = main_direction / np.linalg.norm(main_direction)
                else:
                    main_direction = np.array([1.0, 0.0, 0.0])
                
                # 使用球面均匀分布（最大排斥）
                for i in range(num_h - len(h_positions)):
                    # 使用黄金角度螺旋分布，确保最大排斥
                    theta = np.arccos(1 - 2 * (i + 0.5) / num_h)  # 极角
                    phi = 2 * np.pi * i * 0.618  # 方位角（黄金角度）
                    h_direction = np.array([
                        np.sin(theta) * np.cos(phi),
                        np.sin(theta) * np.sin(phi),
                        np.cos(theta)
                    ])
                    # 如果已有键，稍微偏向已有键的相反方向
                    if existing_bond_vectors:
                        h_direction = -0.3 * main_direction + 0.954 * h_direction
                        h_direction = h_direction / np.linalg.norm(h_direction)
                    h_pos = atom_center + bond_length * h_direction
                    h_positions.append(h_pos.tolist())
            
            # 确保有足够的坐标
            if len(h_positions) != num_h:
                # 如果还是不够，使用简单的备用方法
                atom_center = np.array([atom_pos.x, atom_pos.y, atom_pos.z])
                if atom_symbol in BOND_LENGTHS_PM and 'H' in BOND_LENGTHS_PM[atom_symbol]:
                    bond_length = BOND_LENGTHS_PM[atom_symbol]['H'] / 100.0
                else:
                    bond_length = 1.0
                while len(h_positions) < num_h:
                    # 简单的均匀分布
                    angle = 2 * np.pi * len(h_positions) / num_h
                    h_direction = np.array([np.cos(angle), np.sin(angle), 0.0])
                    h_pos = atom_center + bond_length * h_direction
                    h_positions.append(h_pos.tolist())
            
            # 确保h_positions的数量等于num_h
            if len(h_positions) < num_h:
                # 使用默认位置填充
                atom_center = np.array([atom_pos.x, atom_pos.y, atom_pos.z])
                bond_length = 1.0
                while len(h_positions) < num_h:
                    angle = 2 * np.pi * len(h_positions) / num_h
                    h_direction = np.array([np.cos(angle), np.sin(angle), 0.0])
                    h_pos = atom_center + bond_length * h_direction
                    h_positions.append(h_pos.tolist())
            
            # 设置H原子坐标 - 使用 h_map 中的真实索引
            # 确保为这个原子的所有H设置坐标
            for i in range(num_h):
                if h_map_idx >= len(h_map):
                    break
                # 从 h_map 获取真实的 H 原子索引
                parent_atom_idx_in_map, h_idx = h_map[h_map_idx]
                # 验证父原子索引是否匹配
                if parent_atom_idx_in_map != atom_idx:
                    break
                
                # 获取对应的坐标
                if i < len(h_positions):
                    h_pos_3d = h_positions[i]
                else:
                    # 如果坐标不够，使用默认位置
                    atom_center = np.array([atom_pos.x, atom_pos.y, atom_pos.z])
                    angle = 2 * np.pi * i / num_h
                    h_direction = np.array([np.cos(angle), np.sin(angle), 0.0])
                    h_pos_3d = (atom_center + 1.0 * h_direction).tolist()
                
                h_pos = Point3D(float(h_pos_3d[0]), float(h_pos_3d[1]), float(h_pos_3d[2]))
                new_conf.SetAtomPosition(h_idx, h_pos)
                h_map_idx += 1
        
        # 验证所有H原子都被处理了
        if h_map_idx != len(h_map):
            # 为剩余的H原子设置默认坐标
            for remaining_idx in range(h_map_idx, len(h_map)):
                parent_idx, h_idx = h_map[remaining_idx]
                parent_pos = conf.GetAtomPosition(parent_idx)
                # 使用简单的默认位置
                h_pos = Point3D(parent_pos.x + 1.0, parent_pos.y, parent_pos.z)
                new_conf.SetAtomPosition(h_idx, h_pos)
        
        # 添加conformer到分子
        conf_id = new_mol.AddConformer(new_conf)
                
        # 转换回 SDF 格式
        # 如果分子无法被kekulize（如某些芳香环），尝试修复键类型后再生成SDF
        sdf_with_h = None
        
        try:
            # 先尝试sanitize
            Chem.SanitizeMol(new_mol)
            # 显式指定使用conformer ID=0（我们新添加的conformer）
            sdf_with_h = Chem.MolToMolBlock(new_mol, confId=conf_id)
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
                # 复制conformer到修复后的分子
                if new_mol.GetNumConformers() > 0:
                    fixed_mol.AddConformer(new_mol.GetConformer(conf_id))
                try:
                    Chem.SanitizeMol(fixed_mol)
                    sdf_with_h = Chem.MolToMolBlock(fixed_mol, confId=conf_id if fixed_mol.GetNumConformers() > 0 else -1)
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
                    sdf_with_h = Chem.MolToMolBlock(new_mol, confId=conf_id)
                except Exception:
                    # 如果连生成SDF都失败，返回原始SDF（不添加H）
                    # 这样至少不会丢失原始信息
                    return sdf_content
        except Exception:
            # 其他未知错误，尝试直接生成SDF
            try:
                sdf_with_h = Chem.MolToMolBlock(new_mol, confId=conf_id)
            except Exception:
                # 如果失败，返回原始SDF
                return sdf_content
        
        if sdf_with_h:
            if not sdf_with_h.strip().endswith("$$$$"):
                sdf_with_h = sdf_with_h.rstrip() + "\n$$$$\n"
            return sdf_with_h
        
        return None
        
    except Exception as e:
        return None


# 双键标准键长（单位：pm）
BONDS2_PM = {
    'C': {'C': 134, 'N': 129, 'O': 120, 'S': 160},
    'N': {'C': 129, 'N': 125, 'O': 121},
    'O': {'C': 120, 'N': 121, 'O': 121, 'P': 150},
    'P': {'O': 150, 'S': 186},
    'S': {'P': 186}
}

# 三键标准键长（单位：pm）
BONDS3_PM = {
    'C': {'C': 120, 'N': 116, 'O': 113},
    'N': {'C': 116, 'N': 110},
    'O': {'C': 113}
}

# 键判断容差（单位：pm）
BOND_MARGIN1 = 40  # 单键容差
BOND_MARGIN2 = 20  # 双键容差
BOND_MARGIN3 = 12  # 三键容差


def fix_missing_bonds_from_sdf(sdf_content, bond_length_ratio=1.3, max_iterations=10):
    """
    修复缺失的化学键：在OpenBabel处理之后、补H之前，遍历所有价态未饱和的原子，
    检查最近的原子，如果距离不超过标准键长的指定比例，就添加化学键。
    
    Args:
        sdf_content: SDF格式字符串（应包含键信息）
        bond_length_ratio: 标准键长的比例阈值（默认1.3，即距离≤标准键长×1.3时添加键）
        max_iterations: 最大迭代次数（避免无限循环）
        
    Returns:
        修复后的SDF格式字符串，或None（如果失败）
    """
    try:
        from rdkit import Chem
        from rdkit.Geometry import Point3D
        import numpy as np
        
        # 解析SDF
        mol = Chem.MolFromMolBlock(sdf_content, sanitize=False)
        if mol is None:
            return None
        
        # 检查是否有conformer（用于坐标计算）
        if mol.GetNumConformers() == 0:
            return sdf_content  # 没有坐标，无法计算距离
        
        # 迭代修复缺失的键
        for iteration in range(max_iterations):
            bonds_added = 0
            bonds_upgraded = 0
            
            # 重新获取conformer（因为分子可能已改变）
            if mol.GetNumConformers() == 0:
                break
            conf = mol.GetConformer()
            
            # 第一步：升级已有单键为双键（在添加缺失键之前）
            # 遍历所有单键，检查是否可以升级为双键
            bonds_to_upgrade = []
            for bond in mol.GetBonds():
                if bond.GetBondType() != Chem.BondType.SINGLE:
                    continue  # 只考虑单键
                
                begin_idx = bond.GetBeginAtomIdx()
                end_idx = bond.GetEndAtomIdx()
                begin_atom = mol.GetAtomWithIdx(begin_idx)
                end_atom = mol.GetAtomWithIdx(end_idx)
                begin_symbol = begin_atom.GetSymbol()
                end_symbol = end_atom.GetSymbol()
                
                # 检查两个原子的价态是否都未饱和
                begin_valency = calculate_atom_valency(mol, begin_idx)
                end_valency = calculate_atom_valency(mol, end_idx)
                begin_max_valency = get_atom_max_valency(begin_atom.GetAtomicNum())
                end_max_valency = get_atom_max_valency(end_atom.GetAtomicNum())
                
                # 如果升级为双键，新价态 = 当前价态 - 1 + 2 = 当前价态 + 1
                begin_new_valency = begin_valency + 1
                end_new_valency = end_valency + 1
                
                # 检查升级后价态是否超限
                if begin_new_valency > begin_max_valency or end_new_valency > end_max_valency:
                    continue  # 升级后价态超限，跳过
                
                # 检查是否有双键的标准键长数据
                standard_bond2_pm = None
                if begin_symbol in BONDS2_PM and end_symbol in BONDS2_PM[begin_symbol]:
                    standard_bond2_pm = BONDS2_PM[begin_symbol][end_symbol]
                elif end_symbol in BONDS2_PM and begin_symbol in BONDS2_PM[end_symbol]:
                    standard_bond2_pm = BONDS2_PM[end_symbol][begin_symbol]
                
                if standard_bond2_pm is None:
                    continue  # 没有双键标准键长数据，跳过
                
                # 计算距离
                begin_pos = conf.GetAtomPosition(begin_idx)
                end_pos = conf.GetAtomPosition(end_idx)
                begin_pos_array = np.array([begin_pos.x, begin_pos.y, begin_pos.z])
                end_pos_array = np.array([end_pos.x, end_pos.y, end_pos.z])
                distance = np.linalg.norm(end_pos_array - begin_pos_array)
                distance_pm = distance * 100.0
                
                # 检查距离是否在双键的容差范围内
                threshold2_pm = standard_bond2_pm * bond_length_ratio
                if distance_pm <= threshold2_pm:
                    # 计算相对偏差
                    relative_deviation = abs(distance_pm - standard_bond2_pm) / standard_bond2_pm
                    bonds_to_upgrade.append((begin_idx, end_idx, distance_pm, standard_bond2_pm, relative_deviation))
            
            # 按相对偏差排序，优先升级距离最接近标准双键键长的
            bonds_to_upgrade.sort(key=lambda x: x[4])
            
            # 升级单键为双键
            for begin_idx, end_idx, distance_pm, standard_bond2_pm, _ in bonds_to_upgrade:
                try:
                    rw_mol = Chem.RWMol(mol)
                    # 找到对应的键
                    bond = rw_mol.GetBondBetweenAtoms(begin_idx, end_idx)
                    if bond is not None and bond.GetBondType() == Chem.BondType.SINGLE:
                        # 删除旧键
                        rw_mol.RemoveBond(begin_idx, end_idx)
                        # 添加双键
                        rw_mol.AddBond(begin_idx, end_idx, Chem.BondType.DOUBLE)
                        mol = rw_mol.GetMol()
                        bonds_upgraded += 1
                except Exception as e:
                    # 如果升级失败，继续
                    continue
            
            # 第二步：添加缺失的键（原有逻辑）
            # 找到所有价态未饱和的原子
            unsaturated_atoms = []
            for atom_idx in range(mol.GetNumAtoms()):
                atom = mol.GetAtomWithIdx(atom_idx)
                atomic_num = atom.GetAtomicNum()
                current_valency = calculate_atom_valency(mol, atom_idx)
                max_valency = get_atom_max_valency(atomic_num)
                
                if current_valency < max_valency:
                    unsaturated_atoms.append(atom_idx)
            
            if not unsaturated_atoms and bonds_upgraded == 0:
                break  # 所有原子都已饱和且没有键需要升级，无需继续
            
            # 按未连通片段大小排序：小的片段优先处理
            # 获取所有连通分量
            try:
                frags = Chem.rdmolops.GetMolFrags(mol, asMols=False)
                # frags 是一个列表，每个元素是一个连通分量中的原子索引列表
                
                # 为每个原子找到它所在的连通分量大小
                atom_to_frag_size = {}
                for frag in frags:
                    frag_size = len(frag)
                    for atom_idx in frag:
                        atom_to_frag_size[atom_idx] = frag_size
                
                # 按连通分量大小排序，然后按原子索引排序（保持稳定性）
                unsaturated_atoms.sort(key=lambda idx: (atom_to_frag_size.get(idx, float('inf')), idx))
            except Exception:
                # 如果获取连通分量失败，保持原有顺序
                pass
            
            # 对每个价态未饱和的原子，尝试添加缺失的键
            for atom_idx in unsaturated_atoms:
                atom = mol.GetAtomWithIdx(atom_idx)
                atom_symbol = atom.GetSymbol()
                atomic_num = atom.GetAtomicNum()
                current_valency = calculate_atom_valency(mol, atom_idx)
                max_valency = get_atom_max_valency(atomic_num)
                
                # 获取已连接的原子索引
                connected_atoms = set()
                for bond in atom.GetBonds():
                    other_idx = bond.GetOtherAtomIdx(atom_idx)
                    connected_atoms.add(other_idx)
                
                # 获取原子坐标
                pos1 = conf.GetAtomPosition(atom_idx)
                pos1_array = np.array([pos1.x, pos1.y, pos1.z])
                
                # 找到最近的未连接且价态未饱和的原子
                nearest_atom_idx = None
                nearest_distance = float('inf')
                
                for other_idx in range(mol.GetNumAtoms()):
                    if other_idx == atom_idx or other_idx in connected_atoms:
                        continue
                    
                    other_atom = mol.GetAtomWithIdx(other_idx)
                    other_symbol = other_atom.GetSymbol()
                    other_atomic_num = other_atom.GetAtomicNum()
                    
                    # 只考虑重原子之间的键，以及重原子与H之间的键
                    if atomic_num == 1 and other_atomic_num == 1:
                        continue  # 不考虑H-H键
                    
                    # 检查目标原子的价态：如果已经饱和，跳过（至少需要能添加一个单键）
                    other_current_valency = calculate_atom_valency(mol, other_idx)
                    other_max_valency = get_atom_max_valency(other_atomic_num)
                    if other_current_valency >= other_max_valency:
                        continue  # 目标原子已价态饱和，跳过
                    
                    # 检查：如果两个原子都是孤立原子（都没有其他键），且距离异常近（<0.1Å），跳过
                    # 这可能是坐标错误导致的，不应该连接
                    if current_valency == 0 and other_current_valency == 0:
                        pos2 = conf.GetAtomPosition(other_idx)
                        pos2_array = np.array([pos2.x, pos2.y, pos2.z])
                        distance = np.linalg.norm(pos2_array - pos1_array)
                        if distance < 0.1:  # 距离小于0.1Å，很可能是坐标错误
                            continue  # 跳过这种异常近的孤立原子对
                    
                    # 计算距离
                    pos2 = conf.GetAtomPosition(other_idx)
                    pos2_array = np.array([pos2.x, pos2.y, pos2.z])
                    distance = np.linalg.norm(pos2_array - pos1_array)
                    
                    # 获取标准键长（单键）
                    standard_bond_length_pm = None
                    if atom_symbol in BOND_LENGTHS_PM and other_symbol in BOND_LENGTHS_PM[atom_symbol]:
                        standard_bond_length_pm = BOND_LENGTHS_PM[atom_symbol][other_symbol]
                    elif other_symbol in BOND_LENGTHS_PM and atom_symbol in BOND_LENGTHS_PM[other_symbol]:
                        standard_bond_length_pm = BOND_LENGTHS_PM[other_symbol][atom_symbol]
                    
                    if standard_bond_length_pm is None:
                        # 如果没有标准键长数据，使用默认阈值（2.0 Å）
                        if distance <= 2.0 * bond_length_ratio:
                            if distance < nearest_distance:
                                nearest_distance = distance
                                nearest_atom_idx = other_idx
                        continue
                    
                    standard_bond_length_angstrom = standard_bond_length_pm / 100.0
                    threshold = standard_bond_length_angstrom * bond_length_ratio
                    
                    if distance <= threshold and distance < nearest_distance:
                        nearest_distance = distance
                        nearest_atom_idx = other_idx
                
                # 如果找到最近的原子，尝试添加键
                # 注意：此时目标原子的价态已经检查过（未饱和），但需要再次检查以确保在迭代过程中没有变化
                if nearest_atom_idx is not None:
                    other_atom = mol.GetAtomWithIdx(nearest_atom_idx)
                    other_symbol = other_atom.GetSymbol()
                    other_atomic_num = other_atom.GetAtomicNum()
                    other_current_valency = calculate_atom_valency(mol, nearest_atom_idx)
                    other_max_valency = get_atom_max_valency(other_atomic_num)
                    
                    # 再次检查价态（可能在迭代过程中已变化）
                    if other_current_valency >= other_max_valency:
                        continue  # 目标原子已价态饱和，跳过
                    
                    # 根据距离判断候选键类型
                    # 注意：这里应该使用 bond_length_ratio 来判断，而不是固定的容差
                    candidate_bonds = []
                    distance_pm = nearest_distance * 100.0  # 转换为pm
                    
                    # 检查三键
                    if (atom_symbol in BONDS3_PM and other_symbol in BONDS3_PM[atom_symbol]) or \
                       (other_symbol in BONDS3_PM and atom_symbol in BONDS3_PM[other_symbol]):
                        standard_bond3_pm = BONDS3_PM.get(atom_symbol, {}).get(other_symbol) or \
                                           BONDS3_PM.get(other_symbol, {}).get(atom_symbol)
                        if standard_bond3_pm is not None:
                            # 使用 bond_length_ratio 而不是固定容差
                            threshold3_pm = standard_bond3_pm * bond_length_ratio
                            if distance_pm < threshold3_pm:
                                relative_deviation = abs(distance_pm - standard_bond3_pm) / standard_bond3_pm
                                candidate_bonds.append((3, standard_bond3_pm, relative_deviation))
                    
                    # 检查双键
                    if (atom_symbol in BONDS2_PM and other_symbol in BONDS2_PM[atom_symbol]) or \
                       (other_symbol in BONDS2_PM and atom_symbol in BONDS2_PM[other_symbol]):
                        standard_bond2_pm = BONDS2_PM.get(atom_symbol, {}).get(other_symbol) or \
                                           BONDS2_PM.get(other_symbol, {}).get(atom_symbol)
                        if standard_bond2_pm is not None:
                            # 使用 bond_length_ratio 而不是固定容差
                            threshold2_pm = standard_bond2_pm * bond_length_ratio
                            if distance_pm < threshold2_pm:
                                relative_deviation = abs(distance_pm - standard_bond2_pm) / standard_bond2_pm
                                candidate_bonds.append((2, standard_bond2_pm, relative_deviation))
                    
                    # 检查单键
                    if atom_symbol in BOND_LENGTHS_PM and other_symbol in BOND_LENGTHS_PM[atom_symbol]:
                        standard_bond1_pm = BOND_LENGTHS_PM[atom_symbol][other_symbol]
                    elif other_symbol in BOND_LENGTHS_PM and atom_symbol in BOND_LENGTHS_PM[other_symbol]:
                        standard_bond1_pm = BOND_LENGTHS_PM[other_symbol][atom_symbol]
                    else:
                        standard_bond1_pm = None
                    
                    if standard_bond1_pm is not None:
                        # 使用 bond_length_ratio 而不是固定容差
                        threshold1_pm = standard_bond1_pm * bond_length_ratio
                        if distance_pm < threshold1_pm:
                            relative_deviation = abs(distance_pm - standard_bond1_pm) / standard_bond1_pm
                            candidate_bonds.append((1, standard_bond1_pm, relative_deviation))
                    
                    # 如果没有候选键，跳过
                    if not candidate_bonds:
                        continue
                    
                    # 按相对偏差排序候选键类型
                    candidate_bonds.sort(key=lambda x: x[2])
                    
                    # 根据价态约束筛选合适的键类型（从键级高到低）
                    selected_bond_order = None
                    for bond_order, _, _ in reversed(candidate_bonds):  # 从高到低（三键→双键→单键）
                        new_valency_i = current_valency + bond_order
                        new_valency_j = other_current_valency + bond_order
                        
                        if new_valency_i <= max_valency and new_valency_j <= other_max_valency:
                            selected_bond_order = bond_order
                            break
                    
                    # 如果所有候选键类型都不满足价态约束，选择单键（最保守）
                    if selected_bond_order is None:
                        selected_bond_order = 1
                        # 但还是要检查单键是否满足价态约束
                        new_valency_i = current_valency + 1
                        new_valency_j = other_current_valency + 1
                        if new_valency_i > max_valency or new_valency_j > other_max_valency:
                            continue  # 即使单键也不满足，跳过
                    
                    # 添加键
                    try:
                        rw_mol = Chem.RWMol(mol)
                        if selected_bond_order == 1:
                            bond_type = Chem.BondType.SINGLE
                        elif selected_bond_order == 2:
                            bond_type = Chem.BondType.DOUBLE
                        elif selected_bond_order == 3:
                            bond_type = Chem.BondType.TRIPLE
                        else:
                            bond_type = Chem.BondType.SINGLE
                        
                        rw_mol.AddBond(atom_idx, nearest_atom_idx, bond_type)
                        mol = rw_mol.GetMol()
                        bonds_added += 1
                    except Exception as e:
                        # 如果添加键失败（例如键已存在），继续
                        continue
            
            # 如果本次迭代没有添加任何键且没有升级任何键，停止迭代
            if bonds_added == 0 and bonds_upgraded == 0:
                break
            # 调试信息：打印添加和升级的键数
            if iteration == 0 and (bonds_added > 0 or bonds_upgraded > 0):
                import warnings
                if bonds_upgraded > 0:
                    warnings.warn(f"fix_missing_bonds_from_sdf: 第1次迭代升级了 {bonds_upgraded} 个键为双键")
                if bonds_added > 0:
                    warnings.warn(f"fix_missing_bonds_from_sdf: 第1次迭代添加了 {bonds_added} 个键")
        
        # 转换回SDF格式
        try:
            sdf_block = Chem.MolToMolBlock(mol)
            if sdf_block:
                if not sdf_block.strip().endswith("$$$$"):
                    sdf_block = sdf_block.rstrip() + "\n$$$$\n"
                return sdf_block
            else:
                # 如果生成SDF失败，返回原始SDF
                import warnings
                warnings.warn("fix_missing_bonds_from_sdf: MolToMolBlock 返回空字符串，返回原始SDF")
                return sdf_content
        except Exception as e:
            # 如果转换失败，返回原始SDF
            import warnings
            warnings.warn(f"fix_missing_bonds_from_sdf: 转换SDF时出现异常: {e}，返回原始SDF")
            return sdf_content
        
        return sdf_content
        
    except Exception as e:
        # 如果出现任何错误，返回原始SDF
        import warnings
        warnings.warn(f"fix_missing_bonds_from_sdf 出现异常: {e}，返回原始SDF")
        return sdf_content


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
