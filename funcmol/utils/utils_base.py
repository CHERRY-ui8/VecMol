import getpass
import lightning as L
from omegaconf import OmegaConf
import os
import torch
# from wandb.integration.lightning.fabric import WandbLogger
from lightning.fabric.strategies import DDPStrategy
# import tensorboard
from torch.utils.tensorboard import SummaryWriter
import subprocess
import tempfile
import shutil
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
    cmd = f"obabel {path_xyzs}/*xyz -osdf -O {path_xyzs}/{fname} --title  end"
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


def add_bonds_with_openbabel(coords, atom_type, ele_list, fallback_to_xyz_to_sdf=None):
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
        
        # Call obabel to convert XYZ to SDF
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
    finally:
        # Clean up temporary files
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                print(f"Warning: Failed to clean up temporary directory {temp_dir}: {e}")