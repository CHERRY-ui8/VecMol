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
    # -h: add hydrogens (fill in missing H atoms)
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


# Max valency for common elements
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
    """Convert RDKit bond type to numerical value"""
    from rdkit import Chem
    if bond_type == Chem.BondType.SINGLE:
        return 1
    elif bond_type == Chem.BondType.DOUBLE:
        return 2
    elif bond_type == Chem.BondType.TRIPLE:
        return 3
    elif bond_type == Chem.BondType.AROMATIC:
        return 1.5  # Aromatic bonds are usually considered as 1.5
    else:
        return 1

def calculate_atom_valency(mol, atom_idx):
    """Calculate the current valency of an atom (use RDKit's total valency first, fallback to sum of bond orders if fails)"""
    atom = mol.GetAtomWithIdx(atom_idx)
    
    # Use RDKit's internal calculated valency first
    try:
        valency = float(atom.GetTotalValence())
        # For some molecules that are not sanitized or have incomplete information, RDKit may return 0 or negative values,
        # Fall back to bond-order-sum implementation in this case.
        if valency > 0:
            return valency
    except Exception:
        pass
    
    # Fallback: sum of bond orders (compatible with old logic)
    valency = 0.0
    for bond in atom.GetBonds():
        bond_order = get_bond_order_value(bond.GetBondType())
        valency += bond_order
    return valency

def get_atom_max_valency(atomic_num):
    """Get the maximum valency of an atom"""
    return MAX_VALENCY.get(atomic_num, 4)  # default 4

def _calculate_hydrogen_positions_3d(atom_pos, existing_bond_vectors, num_h, atom_symbol='C', bond_length=None):
    """
    Calculate the position of H atoms in 3D space, considering the direction of existing bonds and reasonable bond angles
    
    Args:
        atom_pos: position of the heavy atom (Point3D)
        existing_bond_vectors: list of direction vectors of existing bonds (normalized 3D vectors)
        num_h: number of H atoms to add
        atom_symbol: symbol of the heavy atom (for getting standard bond length)
        bond_length: distance from H atom to heavy atom (in Å). If None, use standard bond length from BOND_LENGTHS_PM based on atom type
        
    Returns:
        List of H atom positions, each element is a [x, y, z] array. If calculation fails, return empty list (caller needs to handle)
    """
    # Get standard bond length based on atom type
    if bond_length is None:
        # Get standard bond length from BOND_LENGTHS_PM (in pm, need to convert to Å)
        if atom_symbol in BOND_LENGTHS_PM and 'H' in BOND_LENGTHS_PM[atom_symbol]:
            bond_length = BOND_LENGTHS_PM[atom_symbol]['H'] / 100.0  # pm to Å
        else:
            # If standard bond length is not found, use default value (based on atom type)
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
            bond_length = default_bond_lengths.get(atom_symbol, 1.0)  # default 1.0 Å
    # Use numpy imported at the top of the file
    atom_center = np.array([atom_pos.x, atom_pos.y, atom_pos.z])
    h_positions = []
    
    if num_h == 0:
        return h_positions
    
    # Calculate total direction of existing bonds (for determining main direction)
    if existing_bond_vectors:
        # Calculate average direction of existing bonds (weighted average, considering bond importance)
        total_vec = np.sum(existing_bond_vectors, axis=0)
        total_vec_norm = np.linalg.norm(total_vec)
        if total_vec_norm > 0.01:
            main_direction = total_vec / total_vec_norm
        else:
            main_direction = np.array([1.0, 0.0, 0.0])  # Default direction
    else:
        # No existing bonds, use default direction
        main_direction = np.array([1.0, 0.0, 0.0])
    
    # Determine geometric configuration based on number of existing bonds and number of H atoms to add
    total_bonds = len(existing_bond_vectors) + num_h
    
    if total_bonds == 1:
        # Linear: H in opposite direction of existing bonds
        if existing_bond_vectors:
            h_direction = -existing_bond_vectors[0]
        else:
            h_direction = main_direction
        h_pos = atom_center + bond_length * h_direction
        h_positions.append(h_pos.tolist())
        
    elif total_bonds == 2:
        # Linear: two atoms on a straight line
        if existing_bond_vectors:
            # Existing 1 bond, H in opposite direction
            h_direction = -existing_bond_vectors[0]
        else:
            # Need to add 2 H, in opposite direction
            h_direction = main_direction
            h_pos1 = atom_center + bond_length * h_direction
            h_pos2 = atom_center - bond_length * h_direction
            h_positions.append(h_pos1.tolist())
            h_positions.append(h_pos2.tolist())
            return h_positions
        
        h_pos = atom_center + bond_length * h_direction
        h_positions.append(h_pos.tolist())
        
    elif total_bonds == 3:
        # Three-bond case (e.g. sp2 / triangular plane or tetrahedron)
        if existing_bond_vectors:
            n_existing = len(existing_bond_vectors)
            # Case A: Existing 2 bonds, only missing 1 H (most common: =CH-, sp2)
            # Goal: make new H direction roughly point to the "opposite" side of the two existing bonds,
            # i.e. approximately 120° to v1 and v2: h_dir ≈ -normalize(v1 + v2)
            if n_existing >= 2 and num_h == 1:
                v_sum = np.sum(existing_bond_vectors, axis=0)
                v_norm = np.linalg.norm(v_sum)
                if v_norm > 1e-2:
                    h_direction = -v_sum / v_norm
                else:
                    # If two bonds almost cancel out (numerical degeneration), fall back to the opposite direction of any one bond
                    h_direction = -existing_bond_vectors[0]
                h_pos = atom_center + bond_length * h_direction
                h_positions.append(h_pos.tolist())
            # Case B: Existing 1 bond, need to add 2 H
            # Goal: form a triangular plane (or approximate) with the existing bonds, with two bonds between each other approximately 120°.
            elif n_existing == 1 and num_h == 2:
                existing_dir = existing_bond_vectors[0]
                # Calculate a direction perpendicular to the existing bond u
                if abs(existing_dir[2]) < 0.9:
                    u = np.cross(existing_dir, np.array([0, 0, 1]))
                else:
                    u = np.cross(existing_dir, np.array([1, 0, 0]))
                u = u / np.linalg.norm(u)
                # Use v2 = -0.5*v1 + (√3/2)*u, v3 = -0.5*v1 - (√3/2)*u
                # Ensure v1,v2,v3 are approximately 120° to each other
                h_dir1 = -0.5 * existing_dir + (np.sqrt(3) / 2.0) * u
                h_dir2 = -0.5 * existing_dir - (np.sqrt(3) / 2.0) * u
                h_dir1 = h_dir1 / np.linalg.norm(h_dir1)
                h_dir2 = h_dir2 / np.linalg.norm(h_dir2)
                for h_direction in (h_dir1, h_dir2):
                    h_pos = atom_center + bond_length * h_direction
                    h_positions.append(h_pos.tolist())
            else:
                # Other less common combinations, fall back to original logic:
                # Use average direction of existing bonds as "main direction", distribute H uniformly in the perpendicular plane,
                # and adjust to approximately 120° to the existing bonds.
                avg_vec = np.sum(existing_bond_vectors, axis=0)
                if np.linalg.norm(avg_vec) < 1e-2:
                    existing_dir = existing_bond_vectors[0]
                else:
                    existing_dir = avg_vec / np.linalg.norm(avg_vec)
                # Calculate two basis vectors perpendicular to the average direction of existing bonds
                if abs(existing_dir[2]) < 0.9:
                    perp1 = np.cross(existing_dir, np.array([0, 0, 1]))
                else:
                    perp1 = np.cross(existing_dir, np.array([1, 0, 0]))
                perp1 = perp1 / np.linalg.norm(perp1)
                perp2 = np.cross(existing_dir, perp1)
                perp2 = perp2 / np.linalg.norm(perp2)
                # Distribute H uniformly in the perpendicular plane, and make it approximately 120° to the existing bonds
                for i in range(num_h):
                    angle = 2 * np.pi * i / num_h
                    h_direction = np.cos(angle) * perp1 + np.sin(angle) * perp2
                    h_direction = 0.5 * (-existing_dir) + 0.866 * h_direction  # cos(120°) ≈ -0.5, sin(120°) ≈ 0.866
                    h_direction = h_direction / np.linalg.norm(h_direction)
                    h_pos = atom_center + bond_length * h_direction
                    h_positions.append(h_pos.tolist())
        else:
            # No existing bonds, need to add 3 H: use standard tetrahedron geometry (bond angle approximately 109.5°)
            # Create a plane perpendicular to main_direction
            base_normal = np.array([0, 0, 1])
            if np.abs(np.dot(main_direction, base_normal)) > 0.9:
                base_normal = np.array([1, 0, 0])
            
            perp1 = np.cross(main_direction, base_normal)
            perp1 = perp1 / np.linalg.norm(perp1)
            perp2 = np.cross(main_direction, perp1)
            perp2 = perp2 / np.linalg.norm(perp2)
            
            # Tetrahedron: one H above, two H below the plane
            h1_direction = main_direction
            h_pos1 = atom_center + bond_length * h1_direction
            h_positions.append(h_pos1.tolist())
            
            for i in range(num_h - 1):
                angle = 2 * np.pi * i / (num_h - 1)
                h_direction = np.cos(angle) * perp1 + np.sin(angle) * perp2
                # Adjust angle
                h_direction = -0.333 * main_direction + 0.943 * h_direction  # ~109.5 deg
                h_direction = h_direction / np.linalg.norm(h_direction)
                h_pos = atom_center + bond_length * h_direction
                h_positions.append(h_pos.tolist())
    
    elif total_bonds == 4:
        # Tetrahedron: bond angle approximately 109.5°
        if existing_bond_vectors:
            # Direction of existing bonds
            existing_dir = existing_bond_vectors[0]
            
            # Calculate other directions of the tetrahedron
            # Use standard tetrahedron geometry
            if len(existing_bond_vectors) == 1:
                # Existing 1 bond, need to add 3 H to form a tetrahedron
                # Calculate plane perpendicular to the existing bond
                if abs(existing_dir[2]) < 0.9:
                    perp1 = np.cross(existing_dir, np.array([0, 0, 1]))
                else:
                    perp1 = np.cross(existing_dir, np.array([1, 0, 0]))
                perp1 = perp1 / np.linalg.norm(perp1)
                perp2 = np.cross(existing_dir, perp1)
                perp2 = perp2 / np.linalg.norm(perp2)
                
                # Tetrahedron: 3 H in the plane perpendicular to the existing bond, forming an equilateral triangle
                for i in range(num_h):
                    angle = 2 * np.pi * i / num_h
                    h_direction = np.cos(angle) * perp1 + np.sin(angle) * perp2
                    # Adjust angle, make it approximately 109.5° to the existing bonds
                    h_direction = -0.333 * existing_dir + 0.943 * h_direction
                    h_direction = h_direction / np.linalg.norm(h_direction)
                    h_pos = atom_center + bond_length * h_direction
                    h_positions.append(h_pos.tolist())
            elif len(existing_bond_vectors) == 2:
                # Existing 2 bonds, need to add 2 H
                # Calculate direction perpendicular to the bisector of the two existing bonds
                dir1 = existing_bond_vectors[0]
                dir2 = existing_bond_vectors[1]
                bisector = (dir1 + dir2) / np.linalg.norm(dir1 + dir2)
                
                # Calculate direction perpendicular to the bisector
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
                # Existing 3 bonds, need to add 1 H
                # H in opposite direction of the existing bonds (4th vertex of the tetrahedron)
                total_existing = np.sum(existing_bond_vectors, axis=0)
                h_direction = -total_existing / np.linalg.norm(total_existing)
                h_pos = atom_center + bond_length * h_direction
                h_positions.append(h_pos.tolist())
        else:
            # Need to add 4 H to form a standard tetrahedron
            # Use standard tetrahedron vertex coordinates
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
        # For more bonds, use a more general method
        # Distribute H uniformly on the "opposite" side of the existing bonds
        if existing_bond_vectors:
            # Calculate average direction of existing bonds
            avg_direction = np.mean(existing_bond_vectors, axis=0)
            avg_direction = avg_direction / np.linalg.norm(avg_direction)
            
            # Calculate direction perpendicular to the average direction
            if abs(avg_direction[2]) < 0.9:
                perp1 = np.cross(avg_direction, np.array([0, 0, 1]))
            else:
                perp1 = np.cross(avg_direction, np.array([1, 0, 0]))
            perp1 = perp1 / np.linalg.norm(perp1)
            perp2 = np.cross(avg_direction, perp1)
            perp2 = perp2 / np.linalg.norm(perp2)
            
            # Distribute H uniformly in the plane perpendicular to the average direction
            for i in range(num_h):
                angle = 2 * np.pi * i / num_h
                h_direction = np.cos(angle) * perp1 + np.sin(angle) * perp2
                # Slightly towards the opposite direction of the existing bonds
                h_direction = -0.5 * avg_direction + 0.866 * h_direction
                h_direction = h_direction / np.linalg.norm(h_direction)
                h_pos = atom_center + bond_length * h_direction
                h_positions.append(h_pos.tolist())
        else:
            # No existing bonds, distribute H uniformly in 3D space
            # Use spherical uniform distribution
            for i in range(num_h):
                # Use spherical coordinates
                theta = np.arccos(1 - 2 * (i + 0.5) / num_h)  # Polar angle
                phi = 2 * np.pi * i * 0.618  # Azimuthal angle (using golden angle)
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
    Manually add H atoms from SDF content, based on valency calculation
    
    1. Parse SDF to get molecule object (containing bond information)
    2. For each heavy atom, calculate current valency (sum of bond orders) and maximum valency
    3. If current valency < maximum valency, calculate number of H to add
    4. For atoms with no bonds (valency=0), add H directly using maximum valency
    5. Add H atoms uniformly around heavy atoms, using standard bond lengths (based on atom type)
       - C-H: 1.09 Å, O-H: 0.96 Å, N-H: 1.01 Å, S-H: 1.34 Å, etc.
    6. Use maximum repulsion principle to calculate H atom positions, ensuring reasonable geometry
    7. Convert back to SDF format
    
    Args:
        sdf_content: SDF format string (should contain bond information)
        
    Returns:
        New SDF format string with added H, or None
        
    Note:
    - All added H atoms will have valid 3D coordinates (not 0,0,0)
    - If coordinate calculation fails, use fallback method (spherical distribution based on maximum repulsion principle)
    """
    try:
        from rdkit import Chem
        from rdkit.Geometry import Point3D
        
        # Parse SDF
        mol = Chem.MolFromMolBlock(sdf_content, sanitize=False)
        if mol is None:
            return None
        
        # Try sanitize (may help infer bonds)
        try:
            Chem.SanitizeMol(mol)
        except:
            pass
        
        # Before adding H, exclude single-atom fragments (connected components with only one atom and no bonds)
        # Avoid single-atom fragments (e.g. Cl, Br) being added H after being unidentifiable
        # Save original molecule's conformer (for subsequent coordinate copying)
        original_conf = mol.GetConformer() if mol.GetNumConformers() > 0 else None
        
        if mol.GetNumBonds() > 0 and mol.GetNumAtoms() > 1:
            # Get all connected components
            mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
            if len(mol_frags) > 1:
                # Filter out single-atom fragments (connected components with only one atom and no bonds)
                valid_frags = [frag for frag in mol_frags if frag.GetNumBonds() > 0]
                if len(valid_frags) > 0:
                    # If there is only one valid fragment, use it
                    if len(valid_frags) == 1:
                        # Find the atom indices of this valid fragment in the original molecule
                        frag_atom_indices = Chem.rdmolops.GetMolFrags(mol, asMols=False)
                        valid_frag_idx = None
                        for i, frag in enumerate(mol_frags):
                            if frag.GetNumBonds() > 0:
                                valid_frag_idx = i
                                break
                        
                        if valid_frag_idx is not None:
                            valid_atom_indices = frag_atom_indices[valid_frag_idx]
                            mol = valid_frags[0]
                            # Copy conformer from original molecule to new molecule
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
                        # If there are multiple valid fragments, merge them (keep all fragments with bonds)
                        new_mol = Chem.RWMol()
                        atom_map = {}  # Original molecule's atom indices -> new molecule's atom indices
                        
                        # Get all valid fragment's atom indices
                        frag_atom_indices = Chem.rdmolops.GetMolFrags(mol, asMols=False)
                        all_valid_atom_indices = []
                        for frag, atom_indices in zip(mol_frags, frag_atom_indices):
                            if frag.GetNumBonds() > 0:
                                all_valid_atom_indices.extend(atom_indices)
                        all_valid_atom_indices = sorted(all_valid_atom_indices)
                        
                        # Add all valid atoms
                        for old_idx in all_valid_atom_indices:
                            atom = mol.GetAtomWithIdx(old_idx)
                            new_idx = new_mol.AddAtom(atom)
                            atom_map[old_idx] = new_idx
                        
                        # Add all valid fragment's bonds
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
                        
                        # Convert to normal molecule object
                        mol = new_mol.GetMol()
                        
                        # Copy coordinates from original molecule
                        if original_conf is not None:
                            new_conf = Chem.Conformer(mol.GetNumAtoms())
                            for old_idx, new_idx in atom_map.items():
                                pos = original_conf.GetAtomPosition(old_idx)
                                new_conf.SetAtomPosition(new_idx, pos)
                            mol.AddConformer(new_conf)
                else:
                    # If there are no valid fragments (all fragments are single-atom fragments), return None
                    return None
        
        # Create new molecule object
        new_mol = Chem.RWMol(mol)
        
        # Get original coordinates
        conf = mol.GetConformer()
        if conf is None:
            return None
        
        # Calculate number of H to add for each atom
        h_to_add = []
        for atom_idx in range(mol.GetNumAtoms()):
            atom = mol.GetAtomWithIdx(atom_idx)
            atomic_num = atom.GetAtomicNum()
            
            # Skip H atoms
            if atomic_num == 1:
                continue
            
            # Calculate current valency and maximum valency
            current_valency = calculate_atom_valency(mol, atom_idx)
            max_valency = get_atom_max_valency(atomic_num)
            
            # If current valency is 0 (no bonds), use maximum valency
            if current_valency == 0:
                needed_h = max_valency
            else:
                # Calculate number of H to add (round to nearest integer, ensure non-negative)
                needed_h = max(0, int(round(max_valency - current_valency)))
            
            # Only add non-negative number of H, and ensure it does not exceed maximum valency
            if needed_h > 0:
                # Additional check: ensure adding H does not exceed maximum valency
                # Here current_valency already contains all bonds (including bonds to H)
                # So if current_valency + needed_h > max_valency, it means calculation is incorrect
                if current_valency + needed_h <= max_valency:
                    h_to_add.append((atom_idx, needed_h))
        
        if not h_to_add:
            # No H to add, return original SDF
            return sdf_content
        
        # Add H atoms
        # Use h_map to record the correspondence between each H atom and its parent atom: [(parent_atom_idx, h_idx), ...]
        h_map = []
        original_atom_count = new_mol.GetNumAtoms()
        for atom_idx, num_h in h_to_add:
            atom = new_mol.GetAtomWithIdx(atom_idx)
            atom_pos = conf.GetAtomPosition(atom_idx)
            
            for i in range(num_h):
                # Create H atom
                h_atom = Chem.Atom(1)  # Atomic number of H is 1
                h_idx = new_mol.AddAtom(h_atom)
                h_map.append((atom_idx, h_idx))  # Record parent atom index and H atom index
                
                # Add bond
                new_mol.AddBond(atom_idx, h_idx, Chem.BondType.SINGLE)
        
        # Convert to normal molecule object
        new_mol = new_mol.GetMol()
        
        # Delete all old conformers (if any), ensure new conformer is ID=0
        # So MolToMolBlock will use our new conformer
        while new_mol.GetNumConformers() > 0:
            new_mol.RemoveConformer(0)
        
        # Add conformer and set coordinates
        new_conf = Chem.Conformer(new_mol.GetNumAtoms())
        # Copy original coordinates
        for i in range(mol.GetNumAtoms()):
            pos = conf.GetAtomPosition(i)
            new_conf.SetAtomPosition(i, pos)
        
        # Set coordinates for new added H (consider direction of existing bonds and reasonable bond angles)
        # Use h_map to correctly map each H atom's index
        h_map_idx = 0  # Index for traversing h_map
        for atom_idx, num_h in h_to_add:
            atom = mol.GetAtomWithIdx(atom_idx)
            atom_symbol = atom.GetSymbol()
            atom_pos = conf.GetAtomPosition(atom_idx)
            
            # Get direction vectors of existing bonds
            existing_bond_vectors = []
            for bond in atom.GetBonds():
                other_atom = bond.GetOtherAtom(atom)
                other_pos = conf.GetAtomPosition(other_atom.GetIdx())
                # Calculate bond direction vector (from heavy atom to other atoms)
                bond_vec = np.array([
                    other_pos.x - atom_pos.x,
                    other_pos.y - atom_pos.y,
                    other_pos.z - atom_pos.z
                ])
                # Normalize
                bond_len = np.linalg.norm(bond_vec)
                if bond_len > 0.01:  # Avoid division by zero
                    bond_vec = bond_vec / bond_len
                    existing_bond_vectors.append(bond_vec)
            
            # Calculate H positions based on number of existing bonds and number of H to add
            h_positions = _calculate_hydrogen_positions_3d(
                atom_pos, existing_bond_vectors, num_h, atom_symbol=atom_symbol
            )
            
            # If coordinate calculation fails (returns empty list), use fallback method
            if len(h_positions) < num_h:
                # Fallback method: uniformly distribute H in 3D space based on maximum repulsion principle
                atom_center = np.array([atom_pos.x, atom_pos.y, atom_pos.z])
                
                # Get standard bond lengths
                if atom_symbol in BOND_LENGTHS_PM and 'H' in BOND_LENGTHS_PM[atom_symbol]:
                    bond_length = BOND_LENGTHS_PM[atom_symbol]['H'] / 100.0
                else:
                    default_bond_lengths = {
                        'C': 1.09, 'N': 1.01, 'O': 0.96, 'S': 1.34,
                        'P': 1.44, 'F': 0.92, 'Cl': 1.27, 'Br': 1.41,
                    }
                    bond_length = default_bond_lengths.get(atom_symbol, 1.0)
                
                # Calculate average direction of existing bonds (for determining main direction)
                if existing_bond_vectors:
                    main_direction = np.mean(existing_bond_vectors, axis=0)
                    main_direction = main_direction / np.linalg.norm(main_direction)
                else:
                    main_direction = np.array([1.0, 0.0, 0.0])
                
                # use spherical uniform distribution (maximum repulsion)
                for i in range(num_h - len(h_positions)):
                    # use golden angle spiral distribution, ensure maximum repulsion
                    theta = np.arccos(1 - 2 * (i + 0.5) / num_h)  # Polar angle
                    phi = 2 * np.pi * i * 0.618  # Azimuthal angle (golden angle)
                    h_direction = np.array([
                        np.sin(theta) * np.cos(phi),
                        np.sin(theta) * np.sin(phi),
                        np.cos(theta)
                    ])
                    # If there are existing bonds, slightly bias toward opposite of existing bonds
                    if existing_bond_vectors:
                        h_direction = -0.3 * main_direction + 0.954 * h_direction
                        h_direction = h_direction / np.linalg.norm(h_direction)
                    h_pos = atom_center + bond_length * h_direction   
                    h_positions.append(h_pos.tolist())
            
            # Ensure there are enough coordinates
            if len(h_positions) != num_h:
                # If still not enough, use simple fallback method
                atom_center = np.array([atom_pos.x, atom_pos.y, atom_pos.z])
                if atom_symbol in BOND_LENGTHS_PM and 'H' in BOND_LENGTHS_PM[atom_symbol]:
                    bond_length = BOND_LENGTHS_PM[atom_symbol]['H'] / 100.0
                else:
                    bond_length = 1.0
                while len(h_positions) < num_h:
                    # Simple uniform distribution
                    angle = 2 * np.pi * len(h_positions) / num_h
                    h_direction = np.array([np.cos(angle), np.sin(angle), 0.0])
                    h_pos = atom_center + bond_length * h_direction
                    h_positions.append(h_pos.tolist())
            
            # Ensure number of h_positions equals num_h
            if len(h_positions) < num_h:
                # Use default position filling
                atom_center = np.array([atom_pos.x, atom_pos.y, atom_pos.z])
                bond_length = 1.0
                while len(h_positions) < num_h:
                    angle = 2 * np.pi * len(h_positions) / num_h
                    h_direction = np.array([np.cos(angle), np.sin(angle), 0.0])
                    h_pos = atom_center + bond_length * h_direction
                    h_positions.append(h_pos.tolist())
            
            # Set H atom coordinates - use real indices in h_map
            # Ensure coordinates are set for all H atoms of this atom
            for i in range(num_h):
                if h_map_idx >= len(h_map):
                    break
                # Get real H atom index from h_map
                parent_atom_idx_in_map, h_idx = h_map[h_map_idx]
                # Verify if parent atom index matches
                if parent_atom_idx_in_map != atom_idx:
                    break
                
                # Get corresponding coordinates
                if i < len(h_positions):
                    h_pos_3d = h_positions[i]
                else:
                    # If coordinates are not enough, use default position
                    atom_center = np.array([atom_pos.x, atom_pos.y, atom_pos.z])
                    angle = 2 * np.pi * i / num_h
                    h_direction = np.array([np.cos(angle), np.sin(angle), 0.0])
                    h_pos_3d = (atom_center + 1.0 * h_direction).tolist()
                
                h_pos = Point3D(float(h_pos_3d[0]), float(h_pos_3d[1]), float(h_pos_3d[2]))
                new_conf.SetAtomPosition(h_idx, h_pos)
                h_map_idx += 1
        
        # Verify all H atoms are processed
        if h_map_idx != len(h_map):
            # Set default coordinates for remaining H atoms
            for remaining_idx in range(h_map_idx, len(h_map)):
                parent_idx, h_idx = h_map[remaining_idx]
                parent_pos = conf.GetAtomPosition(parent_idx)
                # Use simple default position
                h_pos = Point3D(parent_pos.x + 1.0, parent_pos.y, parent_pos.z)
                new_conf.SetAtomPosition(h_idx, h_pos)
        
        # Add conformer to molecule
        conf_id = new_mol.AddConformer(new_conf)
                
        # Convert back to SDF format
        # If molecule cannot be kekulized (e.g. some aromatic rings), try fixing bond types then write SDF
        sdf_with_h = None
        
        try:
            # Try sanitize
            Chem.SanitizeMol(new_mol)
            # Explicitly specify using conformer ID=0 (our new added conformer)
            sdf_with_h = Chem.MolToMolBlock(new_mol, confId=conf_id)
        except (Chem.rdchem.KekulizeException, Chem.rdchem.AtomValenceException) as e:
            # If sanitize fails (kekulization or valence error), try fixing bond types
            # Change aromatic bonds to single bonds
            try:
                # Create editable molecule object
                editable_mol = Chem.RWMol(new_mol)
                # Traverse all bonds, change aromatic bonds to single bonds
                bonds_to_fix = []
                for bond in editable_mol.GetBonds():
                    if bond.GetBondType() == Chem.BondType.AROMATIC:
                        bonds_to_fix.append((bond.GetBeginIdx(), bond.GetEndIdx()))
                
                # Fix aromatic bonds
                for begin_idx, end_idx in bonds_to_fix:
                    editable_mol.RemoveBond(begin_idx, end_idx)
                    editable_mol.AddBond(begin_idx, end_idx, Chem.BondType.SINGLE)
                
                # Try sanitize fixed molecule
                fixed_mol = editable_mol.GetMol()
                # Copy conformer to fixed molecule
                if new_mol.GetNumConformers() > 0:
                    fixed_mol.AddConformer(new_mol.GetConformer(conf_id))
                try:
                    Chem.SanitizeMol(fixed_mol)
                    sdf_with_h = Chem.MolToMolBlock(fixed_mol, confId=conf_id if fixed_mol.GetNumConformers() > 0 else -1)
                except:
                    # If still cannot sanitize after fixing, try generating SDF directly (without sanitize)
                    # Note: MolToMolBlock may throw exceptions
                    pass
            except Exception:
                pass
            
            # If still fails after fixing bond types, try generating SDF directly (may fail)
            if sdf_with_h is None:
                try:
                    # Last try: generate SDF directly, even if cannot sanitize
                    sdf_with_h = Chem.MolToMolBlock(new_mol, confId=conf_id)
                except Exception:
                    # If even generating SDF fails, return original SDF (without adding H)
                    return sdf_content
        except Exception:
            # Other unknown errors, try generating SDF directly
            try:
                sdf_with_h = Chem.MolToMolBlock(new_mol, confId=conf_id)
            except Exception:
                # If fails, return original SDF
                return sdf_content
        
        if sdf_with_h:
            if not sdf_with_h.strip().endswith("$$$$"):
                sdf_with_h = sdf_with_h.rstrip() + "\n$$$$\n"
            return sdf_with_h
        
        return None
        
    except Exception as e:
        return None


# Standard bond lengths for double bonds (unit: pm)
BONDS2_PM = {
    'C': {'C': 134, 'N': 129, 'O': 120, 'S': 160},
    'N': {'C': 129, 'N': 125, 'O': 121},
    'O': {'C': 120, 'N': 121, 'O': 121, 'P': 150},
    'P': {'O': 150, 'S': 186},
    'S': {'P': 186}
}

# Standard bond lengths for triple bonds (unit: pm)
BONDS3_PM = {
    'C': {'C': 120, 'N': 116, 'O': 113},
    'N': {'C': 116, 'N': 110},
    'O': {'C': 113}
}

# Bond judgment tolerance (unit: pm)
BOND_MARGIN1 = 40  # Single bond tolerance
BOND_MARGIN2 = 20  # Double bond tolerance
BOND_MARGIN3 = 12  # Triple bond tolerance


def fix_missing_bonds_from_sdf(sdf_content, bond_length_ratio=1.3, max_iterations=10):
    """
    Fix missing chemical bonds: after OpenBabel processing, before adding H, traverse all unsaturated atoms,
    check the nearest atom, if the distance is not greater than the specified ratio of the standard bond length, add a chemical bond.
    
    Args:
        sdf_content: SDF format string (should contain bond information)
        bond_length_ratio: standard bond length ratio threshold (default 1.3, i.e. add bond when distance ≤ standard bond length × 1.3)
        max_iterations: maximum number of iterations (to avoid infinite loop)
        
    Returns:
        Fixed SDF format string, or None (if fails)
    """
    try:
        from rdkit import Chem
        from rdkit.Geometry import Point3D
        import numpy as np
        
        # Parse SDF
        mol = Chem.MolFromMolBlock(sdf_content, sanitize=False)
        if mol is None:
            return None
        
        # Check if there is a conformer (for coordinate calculation)
        if mol.GetNumConformers() == 0:
            return sdf_content  # No coordinates, cannot calculate distance
        
        # Iteratively fix missing bonds
        for iteration in range(max_iterations):
            bonds_added = 0
            bonds_upgraded = 0
            
            # Re-get conformer (because molecule may have changed)
            if mol.GetNumConformers() == 0:
                break
            conf = mol.GetConformer()
            
            # Step 1: upgrade existing single bonds to double bonds (before adding missing bonds)
            # Traverse all single bonds, check if they can be upgraded to double bonds
            bonds_to_upgrade = []
            for bond in mol.GetBonds():
                if bond.GetBondType() != Chem.BondType.SINGLE:
                    continue  # Only consider single bonds
                
                begin_idx = bond.GetBeginAtomIdx()
                end_idx = bond.GetEndAtomIdx()
                begin_atom = mol.GetAtomWithIdx(begin_idx)
                end_atom = mol.GetAtomWithIdx(end_idx)
                begin_symbol = begin_atom.GetSymbol()
                end_symbol = end_atom.GetSymbol()
                
                # Check if the valencies of the two atoms are both unsaturated
                begin_valency = calculate_atom_valency(mol, begin_idx)
                end_valency = calculate_atom_valency(mol, end_idx)
                begin_max_valency = get_atom_max_valency(begin_atom.GetAtomicNum())
                end_max_valency = get_atom_max_valency(end_atom.GetAtomicNum())
                
                # If upgraded to double bond, new valency = current valency - 1 + 2 = current valency + 1
                begin_new_valency = begin_valency + 1
                end_new_valency = end_valency + 1
                
                # Check if the valency exceeds the limit after upgrading
                if begin_new_valency > begin_max_valency or end_new_valency > end_max_valency:
                    continue  # Valency exceeds the limit after upgrading, skip
                
                # Check if there is standard bond length data for double bonds
                standard_bond2_pm = None
                if begin_symbol in BONDS2_PM and end_symbol in BONDS2_PM[begin_symbol]:
                    standard_bond2_pm = BONDS2_PM[begin_symbol][end_symbol]
                elif end_symbol in BONDS2_PM and begin_symbol in BONDS2_PM[end_symbol]:
                    standard_bond2_pm = BONDS2_PM[end_symbol][begin_symbol]
                
                if standard_bond2_pm is None:
                    continue  # No standard bond length data for double bonds, skip
                
                # Calculate distance
                begin_pos = conf.GetAtomPosition(begin_idx)
                end_pos = conf.GetAtomPosition(end_idx)
                begin_pos_array = np.array([begin_pos.x, begin_pos.y, begin_pos.z])
                end_pos_array = np.array([end_pos.x, end_pos.y, end_pos.z])
                distance = np.linalg.norm(end_pos_array - begin_pos_array)
                distance_pm = distance * 100.0
                
                # Check if the distance is within the tolerance range for double bonds
                threshold2_pm = standard_bond2_pm * bond_length_ratio
                if distance_pm <= threshold2_pm:
                    # Calculate relative deviation
                    relative_deviation = abs(distance_pm - standard_bond2_pm) / standard_bond2_pm
                    bonds_to_upgrade.append((begin_idx, end_idx, distance_pm, standard_bond2_pm, relative_deviation))
            
            # Sort by relative deviation, prioritize upgrading bonds with the shortest distance to the standard double bond length
            bonds_to_upgrade.sort(key=lambda x: x[4])
            
            # Upgrade single bonds to double bonds
            for begin_idx, end_idx, distance_pm, standard_bond2_pm, _ in bonds_to_upgrade:
                try:
                    rw_mol = Chem.RWMol(mol)
                    # Find the corresponding bond
                    bond = rw_mol.GetBondBetweenAtoms(begin_idx, end_idx)
                    if bond is not None and bond.GetBondType() == Chem.BondType.SINGLE:
                        # Delete old bond
                        rw_mol.RemoveBond(begin_idx, end_idx)
                        # Add double bond
                        rw_mol.AddBond(begin_idx, end_idx, Chem.BondType.DOUBLE)
                        mol = rw_mol.GetMol()
                        bonds_upgraded += 1
                except Exception as e:
                    # If upgrade fails, continue
                    continue
            
            # Step 2: add missing bonds (original logic)
            # Find all unsaturated atoms
            unsaturated_atoms = []
            for atom_idx in range(mol.GetNumAtoms()):
                atom = mol.GetAtomWithIdx(atom_idx)
                atomic_num = atom.GetAtomicNum()
                current_valency = calculate_atom_valency(mol, atom_idx)
                max_valency = get_atom_max_valency(atomic_num)
                
                if current_valency < max_valency:
                    unsaturated_atoms.append(atom_idx)
            
            if not unsaturated_atoms and bonds_upgraded == 0:
                break  # All atoms are saturated and no bonds need to be upgraded, no need to continue
            
            # Sort by size of disconnected fragments: smaller fragments are processed first
            try:
                frags = Chem.rdmolops.GetMolFrags(mol, asMols=False)                
                # For each atom, find the size of the connected component it belongs to
                atom_to_frag_size = {}
                for frag in frags:
                    frag_size = len(frag)
                    for atom_idx in frag:
                        atom_to_frag_size[atom_idx] = frag_size
                
                # Sort by size of connected components, then by atom index (maintain stability)
                unsaturated_atoms.sort(key=lambda idx: (atom_to_frag_size.get(idx, float('inf')), idx))
            except Exception:
                # If getting connected components fails, keep original order
                pass
            
            # For each unsaturated atom, try to add missing bonds
            for atom_idx in unsaturated_atoms:
                atom = mol.GetAtomWithIdx(atom_idx)
                atom_symbol = atom.GetSymbol()
                atomic_num = atom.GetAtomicNum()
                current_valency = calculate_atom_valency(mol, atom_idx)
                max_valency = get_atom_max_valency(atomic_num)
                
                # Get connected atom indices
                connected_atoms = set()
                for bond in atom.GetBonds():
                    other_idx = bond.GetOtherAtomIdx(atom_idx)
                    connected_atoms.add(other_idx)
                
                # Get atom coordinates
                pos1 = conf.GetAtomPosition(atom_idx)
                pos1_array = np.array([pos1.x, pos1.y, pos1.z])
                
                # Find the nearest unsaturated atom that is not connected
                nearest_atom_idx = None
                nearest_distance = float('inf')
                
                for other_idx in range(mol.GetNumAtoms()):
                    if other_idx == atom_idx or other_idx in connected_atoms:
                        continue
                    
                    other_atom = mol.GetAtomWithIdx(other_idx)
                    other_symbol = other_atom.GetSymbol()
                    other_atomic_num = other_atom.GetAtomicNum()
                    
                    # Only consider bonds between heavy atoms, and heavy atoms and H
                    if atomic_num == 1 and other_atomic_num == 1:
                        continue  # Do not consider H-H bonds
                    
                    # Check the valency of the target atom: if it is saturated, skip (at least one single bond needs to be added)
                    other_current_valency = calculate_atom_valency(mol, other_idx)
                    other_max_valency = get_atom_max_valency(other_atomic_num)
                    if other_current_valency >= other_max_valency:
                        continue  # Target atom is saturated, skip
                    
                    # Check: if both atoms are isolated atoms (no other bonds), and the distance is extremely close (<0.1Å), skip
                    # This may be due to coordinate errors, and should not be connected
                    if current_valency == 0 and other_current_valency == 0:
                        pos2 = conf.GetAtomPosition(other_idx)
                        pos2_array = np.array([pos2.x, pos2.y, pos2.z])
                        distance = np.linalg.norm(pos2_array - pos1_array)
                        if distance < 0.1:  # Distance is less than 0.1Å, very likely due to coordinate errors
                            continue  # Skip this extremely close isolated atom pair
                    
                    # Calculate distance
                    pos2 = conf.GetAtomPosition(other_idx)
                    pos2_array = np.array([pos2.x, pos2.y, pos2.z])
                    distance = np.linalg.norm(pos2_array - pos1_array)
                    
                    # Get standard bond length (single bond)
                    standard_bond_length_pm = None
                    if atom_symbol in BOND_LENGTHS_PM and other_symbol in BOND_LENGTHS_PM[atom_symbol]:
                        standard_bond_length_pm = BOND_LENGTHS_PM[atom_symbol][other_symbol]
                    elif other_symbol in BOND_LENGTHS_PM and atom_symbol in BOND_LENGTHS_PM[other_symbol]:
                        standard_bond_length_pm = BOND_LENGTHS_PM[other_symbol][atom_symbol]
                    
                    if standard_bond_length_pm is None:
                        # If there is no standard bond length data, use default threshold (2.0 Å)
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
                
                # If found the nearest atom, try to add bond
                # Note: at this time, the valency of the target atom has been checked (unsaturated), but it needs to be checked again to ensure that it has not changed in the iteration process
                if nearest_atom_idx is not None:
                    other_atom = mol.GetAtomWithIdx(nearest_atom_idx)
                    other_symbol = other_atom.GetSymbol()
                    other_atomic_num = other_atom.GetAtomicNum()
                    other_current_valency = calculate_atom_valency(mol, nearest_atom_idx)
                    other_max_valency = get_atom_max_valency(other_atomic_num)
                    
                    # Check valency again (may have changed in the iteration process)
                    if other_current_valency >= other_max_valency:
                        continue  # Target atom is saturated, skip
                    
                    # Based on distance, determine candidate bond types
                    # Note: here should use bond_length_ratio to determine, not fixed tolerance
                    candidate_bonds = []
                    distance_pm = nearest_distance * 100.0  # Convert to pm
                    
                    # Check triple bonds
                    if (atom_symbol in BONDS3_PM and other_symbol in BONDS3_PM[atom_symbol]) or \
                       (other_symbol in BONDS3_PM and atom_symbol in BONDS3_PM[other_symbol]):
                        standard_bond3_pm = BONDS3_PM.get(atom_symbol, {}).get(other_symbol) or \
                                           BONDS3_PM.get(other_symbol, {}).get(atom_symbol)
                        if standard_bond3_pm is not None:
                            # Use bond_length_ratio instead of fixed tolerance
                            threshold3_pm = standard_bond3_pm * bond_length_ratio
                            if distance_pm < threshold3_pm:
                                relative_deviation = abs(distance_pm - standard_bond3_pm) / standard_bond3_pm
                                candidate_bonds.append((3, standard_bond3_pm, relative_deviation))
                    
                    # Check double bonds
                    if (atom_symbol in BONDS2_PM and other_symbol in BONDS2_PM[atom_symbol]) or \
                       (other_symbol in BONDS2_PM and atom_symbol in BONDS2_PM[other_symbol]):
                        standard_bond2_pm = BONDS2_PM.get(atom_symbol, {}).get(other_symbol) or \
                                           BONDS2_PM.get(other_symbol, {}).get(atom_symbol)
                        if standard_bond2_pm is not None:
                            # Use bond_length_ratio instead of fixed tolerance
                            threshold2_pm = standard_bond2_pm * bond_length_ratio
                            if distance_pm < threshold2_pm:
                                relative_deviation = abs(distance_pm - standard_bond2_pm) / standard_bond2_pm
                                candidate_bonds.append((2, standard_bond2_pm, relative_deviation))
                    
                    # Check single bonds
                    if atom_symbol in BOND_LENGTHS_PM and other_symbol in BOND_LENGTHS_PM[atom_symbol]:
                        standard_bond1_pm = BOND_LENGTHS_PM[atom_symbol][other_symbol]
                    elif other_symbol in BOND_LENGTHS_PM and atom_symbol in BOND_LENGTHS_PM[other_symbol]:
                        standard_bond1_pm = BOND_LENGTHS_PM[other_symbol][atom_symbol]
                    else:
                        standard_bond1_pm = None
                    
                    if standard_bond1_pm is not None:
                        # Use bond_length_ratio instead of fixed tolerance
                        threshold1_pm = standard_bond1_pm * bond_length_ratio
                        if distance_pm < threshold1_pm:
                            relative_deviation = abs(distance_pm - standard_bond1_pm) / standard_bond1_pm
                            candidate_bonds.append((1, standard_bond1_pm, relative_deviation))
                    
                    # If there are no candidate bonds, skip
                    if not candidate_bonds:
                        continue
                    
                    # Sort candidate bond types by relative deviation
                    candidate_bonds.sort(key=lambda x: x[2])
                    
                    # Filter suitable bond types based on valency constraints (from high to low bond order)
                    selected_bond_order = None
                    for bond_order, _, _ in reversed(candidate_bonds):  # high to low (triple -> double -> single)
                        new_valency_i = current_valency + bond_order
                        new_valency_j = other_current_valency + bond_order
                        
                        if new_valency_i <= max_valency and new_valency_j <= other_max_valency:
                            selected_bond_order = bond_order
                            break
                    
                    # If all candidate bond types do not satisfy valency constraints, select single bond (most conservative)
                    if selected_bond_order is None:
                        selected_bond_order = 1
                        # But still need to check if the single bond satisfies the valency constraint
                        new_valency_i = current_valency + 1
                        new_valency_j = other_current_valency + 1
                        if new_valency_i > max_valency or new_valency_j > other_max_valency:
                            continue  # Even if the single bond does not satisfy, skip
                    
                    # Add bond
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
                        # If adding bond fails (e.g. bond already exists), continue
                        continue
            
            # If this iteration does not add any bonds and does not upgrade any bonds, stop iteration
            if bonds_added == 0 and bonds_upgraded == 0:
                break
            # Debug information: print the number of added and upgraded bonds
            if iteration == 0 and (bonds_added > 0 or bonds_upgraded > 0):
                import warnings
                if bonds_upgraded > 0:
                    warnings.warn(f"fix_missing_bonds_from_sdf: First iteration upgraded {bonds_upgraded} bonds to double bonds")
                if bonds_added > 0:
                    warnings.warn(f"fix_missing_bonds_from_sdf: First iteration added {bonds_added} bonds")
        
        # Convert back to SDF format
        try:
            sdf_block = Chem.MolToMolBlock(mol)
            if sdf_block:
                if not sdf_block.strip().endswith("$$$$"):
                    sdf_block = sdf_block.rstrip() + "\n$$$$\n"
                return sdf_block
            else:
                # If generating SDF fails, return original SDF
                import warnings
                warnings.warn("fix_missing_bonds_from_sdf: MolToMolBlock returned empty string, returning original SDF")
                return sdf_content
        except Exception as e:
            # If conversion fails, return original SDF
            import warnings
            warnings.warn(f"fix_missing_bonds_from_sdf: Exception occurred when converting SDF: {e}, returning original SDF")
            return sdf_content
        
        return sdf_content
        
    except Exception as e:
        # If any error occurs, return original SDF
        import warnings
        warnings.warn(f"fix_missing_bonds_from_sdf: Exception occurred: {e}, returning original SDF")
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
