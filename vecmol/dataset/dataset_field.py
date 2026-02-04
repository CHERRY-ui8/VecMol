import itertools
import math
import os
import random
import pickle
import lmdb
import threading
from typing import Optional

from omegaconf import DictConfig, OmegaConf
import torch
from torch_geometric.data import Data, Dataset, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_batch

from vecmol.models.decoder import get_grid
from vecmol.utils.constants import ELEMENTS_HASH, PADDING_INDEX
from vecmol.utils.gnf_converter import GNFConverter

# local storage for LMDB connections in multi-process environment
_thread_local = threading.local()


class FieldDataset(Dataset):
    """
    Initializes the dataset with the specified parameters.

    Args:
        gnf_converter (GNFConverter): GNFConverter instance for calculating target gradient field
        dset_name (str): Name of the dataset. Default is "qm9".
        data_dir (str): Directory where the dataset is stored. Default is "dataset/data".
        elements (list): List of elements to filter by. Default is None, which uses ELEMENTS_HASH.
        split (str): Dataset split to use. Must be one of ["train", "val", "test"]. Default is "train".
        rotate (bool): Whether to apply rotation. Default is False.
        grid_dim (int): Dimension of the grid. Default is 32.
        resolution (float): Resolution of the grid. Default is 0.25.
        n_points (int): Number of points to sample. Default is 4000.
        sample_full_grid (bool): Whether to sample the full grid. Default is False.
        targeted_sampling_ratio (int): Ratio for targeted sampling. Default is 2.
        cubes_around (int): Number of cubes around to consider. Default is 3.
    """
    def __init__(
        self,
        gnf_converter: GNFConverter,  # Receive GNFConverter instance
        dset_name: str = "qm9",
        data_dir: str = "dataset/data",
        elements: Optional[dict] = None,
        split: str = "train",
        rotate: bool = False,
        grid_dim: int = 32,
        resolution: float = 0.25,
        n_points: int = 4000,
        sample_full_grid: bool = False,
        targeted_sampling_ratio: int = 2,
        cubes_around: int = 3,
        atom_distance_threshold: float = 0.5,  # Only sample points within a certain distance (in Å) from the atom
    ):
        if elements is None:
            elements = ELEMENTS_HASH
        # Allow drugs_no_h as a valid dataset name (for datasets without hydrogen atoms)
        assert dset_name in ["qm9", "drugs", "drugs_no_h", "cremp"]
        assert split in ["train", "val", "test"]
        self.dset_name = dset_name
        self.data_dir = data_dir
        self.elements = elements
        self.split = split
        self.rotate = rotate
        self.resolution = resolution
        # Ensure n_points is not None, use default value if None
        if n_points is None:
            raise ValueError(f"n_points cannot be None. Please set dset.n_points or joint_finetune.n_points in config.")
        self.n_points = n_points
        self.sample_full_grid = sample_full_grid
        self.grid_dim = grid_dim
        # Remove scale_factor - no longer needed for normalization
        # self.scale_factor = 1 / (self.resolution * self.grid_dim / 2)

        self._read_data()
        self._filter_by_elements(elements)

        # Create increments based on real-world distances (Angstroms)
        # Each increment represents a step in Angstrom units
        self.increments = torch.tensor(
            list(itertools.product(list(range(-cubes_around, cubes_around+1)), repeat=3)),
            dtype=torch.float32
        ) * self.resolution  # Scale by resolution to get real distances
        
        # Allow targeted_sampling_ratio in validation/test to enable neighbor point sampling
        # Previously it was forced to 0 for non-train splits, but now we allow it for better evaluation
        self.targeted_sampling_ratio = targeted_sampling_ratio
        self.atom_distance_threshold = atom_distance_threshold
        
        # Set field_idxs based on data loading method
        if hasattr(self, 'use_lmdb') and self.use_lmdb:
            # In LMDB mode, use the length of keys
            self.field_idxs = torch.arange(len(self.keys))
        else:
            # In traditional mode, use the length of data
            self.field_idxs = torch.arange(len(self.data))
            
        self.discrete_grid, self.full_grid_high_res = get_grid(self.grid_dim, self.resolution)
        
        # Use the incoming GNFConverter instance
        self.gnf_converter = gnf_converter

        # if self.db is None:
        #     self._connect_db()
            

    def __del__(self):
        """Destructor to ensure database connection is closed"""
        if hasattr(self, 'db') and self.db is not None:
            self._close_db()

    def _read_data(self):
        fname = f"{self.split}_data"
        if self.dset_name == "cremp":
            fname = f"{self.split}_50_data"
        
        # Check if LMDB database exists
        lmdb_path = os.path.join(self.data_dir, self.dset_name, f"{fname}.lmdb")
        # NOTE: remove molid2idx definition
        keys_path = os.path.join(self.data_dir, self.dset_name, f"{fname}_keys.pt")
        
        if os.path.exists(lmdb_path) and os.path.exists(keys_path):
            # Use LMDB database
            self._use_lmdb_database(lmdb_path, keys_path)
        else:
            # Use traditional torch.load method
            self.data = torch.load(os.path.join(
                self.data_dir, self.dset_name, f"{fname}.pth"), weights_only=False
            )
            self.use_lmdb = False

    def _use_lmdb_database(self, lmdb_path, keys_path):
        """Use LMDB database to load data"""
        self.lmdb_path = lmdb_path
        self.keys = torch.load(keys_path)  # load keys file directly
        self.db = None
        self.use_lmdb = True
        
        print(f"  | Using LMDB database: {lmdb_path}")
        print(f"  | Database contains {len(self.keys)} molecules")

    def _connect_db(self):
        """Create read-only database connection - process safe version"""
        if self.db is None:
            # Use thread local storage to ensure each worker process has an independent connection
            # In spawn mode, each process has independent thread local storage
            if not hasattr(_thread_local, 'lmdb_connections'):
                _thread_local.lmdb_connections = {}
            
            # Create independent connection for each LMDB path
            if self.lmdb_path not in _thread_local.lmdb_connections:
                try:
                    _thread_local.lmdb_connections[self.lmdb_path] = lmdb.open(
                        self.lmdb_path,
                        map_size=10*(1024*1024*1024),   # 10GB
                        create=False,
                        subdir=True,
                        readonly=True,
                        lock=False,
                        readahead=False,
                        meminit=False,
                        max_readers=256,  # Increase maximum reader count, support more concurrent workers
                    )
                except Exception as e:
                    # If connection fails, clean up and retry
                    if self.lmdb_path in _thread_local.lmdb_connections:
                        try:
                            _thread_local.lmdb_connections[self.lmdb_path].close()
                        except:
                            pass
                        del _thread_local.lmdb_connections[self.lmdb_path]
                    raise
            
            self.db = _thread_local.lmdb_connections[self.lmdb_path]
            
            # Verify if connection is valid
            try:
                with self.db.begin() as txn:
                    txn.stat()
            except Exception:
                # Connection is invalid, clean up and recreate
                try:
                    self.db.close()
                except:
                    pass
                if self.lmdb_path in _thread_local.lmdb_connections:
                    del _thread_local.lmdb_connections[self.lmdb_path]
                # Recursively call reconnect
                self.db = None
                self._connect_db()

    def _close_db(self):
        """Close database connection"""
        if self.db is not None:
            self.db.close()
            self.db = None
            self.keys = None

    def _filter_by_elements(self, elements) -> None:
        if hasattr(self, 'use_lmdb') and self.use_lmdb:
            # Element filtering in convert_to_lmdb.py in LMDB mode
            return
        
        # Filtering in traditional mode
        filtered_data = []
        elements_ids = [ELEMENTS_HASH[element] for element in elements]

        for datum in self.data:
            atoms = datum["atoms_channel"][datum["atoms_channel"] != PADDING_INDEX]
            include = True
            for atom_id in atoms.unique():
                if int(atom_id.item()) not in elements_ids:
                    include = False
                    break
            if include:
                filtered_data.append(datum)
        if len(self.data) != len(filtered_data):
            print(
                f"  | filter data (elements): data reduced from {len(self.data)} to {len(filtered_data)}"
            )
            self.data = filtered_data

    def __len__(self):
        # Always use field_idxs to determine dataset size, this supports subset selection
        return self.field_idxs.size(0)

    def __getitem__(self, index) -> Data:
        if hasattr(self, 'use_lmdb') and self.use_lmdb:
            return self._getitem_lmdb(index)
        else:
            return self._getitem_traditional(index)

    def _getitem_lmdb(self, index) -> Data:
        """Data retrieval in LMDB mode - process safe version"""
        # Ensure database connection is established in worker process
        if self.db is None:
            self._connect_db()
        
        # Use field_idxs to get actual index
        idx = self.field_idxs[index].item()
        key = self.keys[idx]
        # Ensure key is bytes format, because LMDB needs bytes
        if isinstance(key, str):
            key = key.encode('utf-8')
        
        # Use more secure transaction processing
        try:
            with self.db.begin() as txn:
                sample_raw = pickle.loads(txn.get(key))
        except Exception as e:
            # If transaction fails, reconnect database
            print(f"LMDB transaction failed, reconnecting: {e}")
            self._close_db()
            self._connect_db()
            with self.db.begin() as txn:
                sample_raw = pickle.loads(txn.get(key))
        
        # preprocess molecule (handles rotation and centering)
        sample = self._preprocess_molecule(sample_raw)

        # sample points for field prediction
        xs, point_types = self._get_xs(sample)

        # get data from preprocessed sample
        coords = sample["coords"]
        atoms_channel = sample["atoms_channel"]

        # remove padding to handle variable-sized molecules
        valid_mask = atoms_channel != PADDING_INDEX
        coords = coords[valid_mask]
        atoms_channel = atoms_channel[valid_mask]

        # compute target gradient field (ground truth)
        # Reshape xs to [1, n_points, 3] for batch processing
        xs_batch = xs.unsqueeze(0)  # [1, n_points, 3]
        coords_batch = coords.unsqueeze(0)  # [1, n_atoms, 3]
        atoms_channel_batch = atoms_channel.unsqueeze(0)  # [1, n_atoms]
        
        with torch.no_grad():
            target_field = self.gnf_converter.mol2gnf(coords_batch, atoms_channel_batch, xs_batch)
            target_field = target_field.squeeze(0)  # [n_points, n_atom_types, 3]

        # create torch_geometric data object
        data = Data(
            pos=coords.float(),
            x=atoms_channel.long(),
            xs=xs.float(),
            target_field=target_field.float(), 
            point_types=point_types.long(),
            idx=torch.tensor([index], dtype=torch.long)
        )
        return data

    def _getitem_traditional(self, index) -> Data:
        """Data retrieval in traditional mode"""
        # get raw data
        idx = self.field_idxs[index]
        sample_raw = self.data[idx]
        
        # preprocess molecule (handles rotation and centering)
        sample = self._preprocess_molecule(sample_raw)

        # sample points for field prediction
        xs, point_types = self._get_xs(sample)

        # get data from preprocessed sample
        coords = sample["coords"]
        atoms_channel = sample["atoms_channel"]

        # remove padding to handle variable-sized molecules
        valid_mask = atoms_channel != PADDING_INDEX
        coords = coords[valid_mask]
        atoms_channel = atoms_channel[valid_mask]

        # compute target gradient field (ground truth)
        # Reshape xs to [1, n_points, 3] for batch processing
        xs_batch = xs.unsqueeze(0)  # [1, n_points, 3]
        coords_batch = coords.unsqueeze(0)  # [1, n_atoms, 3]
        atoms_channel_batch = atoms_channel.unsqueeze(0)  # [1, n_atoms]
        
        with torch.no_grad():
            target_field = self.gnf_converter.mol2gnf(coords_batch, atoms_channel_batch, xs_batch)
            target_field = target_field.squeeze(0)  # [n_points, n_atom_types, 3]

        # create torch_geometric data object
        data = Data(
            pos=coords.float(),
            x=atoms_channel.long(),
            xs=xs.float(),
            target_field=target_field.float(),
            point_types=point_types.long(),
            idx=torch.tensor([index], dtype=torch.long)
        )
        return data

    def _preprocess_molecule(self, sample_raw) -> dict:
        """
        Preprocesses a raw molecule sample by removing invalid values.

        Args:
            sample_raw (dict): The raw molecule sample.

        Returns:
            dict: The preprocessed molecule sample.
        """
        sample = {
            "coords": sample_raw["coords"],
            "atoms_channel": sample_raw["atoms_channel"],
        }

        if self.rotate:
            sample["coords"] = self._rotate_coords(sample)

        sample["coords"] = self._center_molecule(sample["coords"])

        return sample

    def _get_xs(self, sample) -> tuple:
        """
        Sample query points, return query point coordinates and point type markers
        
        Returns:
            tuple: (query_points, point_types)
                - query_points: [n_points, 3] query point coordinates
                - point_types: [n_points] point type markers, 0 represents grid point, 1 represents nearby point
        """
        mask = sample["atoms_channel"] != PADDING_INDEX
        coords = sample["coords"][mask]
        device = coords.device
        
        if self.sample_full_grid:
            query_points = self.full_grid_high_res.to(device)
            point_types = torch.zeros(len(query_points), dtype=torch.long, device=device)  # all grid points
            return query_points, point_types
        
        grid_size = len(self.full_grid_high_res) # grid_dim**3

        if self.targeted_sampling_ratio == 0:
            # Only sample from grid randomly
            sample_size = min(self.n_points * 2, grid_size)
            grid_indices = torch.randperm(grid_size, device=device)[:sample_size]
            grid_points = self.full_grid_high_res[grid_indices].to(device)
            unique_points = torch.unique(grid_points, dim=0)
            
            if unique_points.size(0) < self.n_points:
                remaining = self.n_points - unique_points.size(0)
                additional_indices = torch.randint(grid_size, (remaining,), device=device)
                additional_points = self.full_grid_high_res[additional_indices].to(device)
                unique_points = torch.cat([unique_points, additional_points], dim=0)
            
            indices = torch.randperm(unique_points.size(0), device=device)[:self.n_points]
            query_points = unique_points[indices]
            point_types = torch.zeros(self.n_points, dtype=torch.long, device=device)  # all grid points
            return query_points, point_types
        
        else:
            if self.targeted_sampling_ratio == 0:
                # Only sample grid points
                n_neighbor = 0
                n_grid = self.n_points
            else:
                n_neighbor = int(self.n_points / (1 + self.targeted_sampling_ratio))
                n_grid = self.n_points - n_neighbor
                
                # 1. Sample nearby points (points around atoms)
                rand_points = coords
                # Add small noise, noise size should be related to grid resolution
                noise = torch.randn_like(rand_points, device=device) * self.resolution / 4
                rand_points = (rand_points + noise)
                
                # Calculate the number of points to sample around each atom
                if n_neighbor > 0:
                    points_per_atom = max(1, n_neighbor // coords.size(0))
                    random_indices = torch.randperm(self.increments.size(0), device=device)[:points_per_atom]
                    
                    # Sample nearby points
                    neighbor_points = (rand_points.unsqueeze(1) + self.increments[random_indices].to(device)).view(-1, 3)
                    # Calculate bounds based on real-world distances
                    total_span = (self.grid_dim - 1) * self.resolution
                    half_span = total_span / 2
                    min_bound = -half_span
                    max_bound = half_span
                    neighbor_points = torch.clamp(neighbor_points, min_bound, max_bound).float()
                    
                    # If the number of sampled points exceeds the required number, randomly select
                    if neighbor_points.size(0) > n_neighbor:
                        indices = torch.randperm(neighbor_points.size(0), device=device)[:n_neighbor]
                        neighbor_points = neighbor_points[indices]
                    elif neighbor_points.size(0) < n_neighbor:
                        # If the number of sampled points is less than required, allow repeated sampling
                        remaining = n_neighbor - neighbor_points.size(0)
                        additional_indices = torch.randint(0, neighbor_points.size(0), (remaining,), device=device)
                        neighbor_points = torch.cat([neighbor_points, neighbor_points[additional_indices]], dim=0)
                else:
                    neighbor_points = torch.empty((0, 3), device=device)
                
                # 2. Sample grid points
                if n_grid > 0:
                    grid_indices = torch.randperm(grid_size, device=device)[:n_grid]
                    grid_points = self.full_grid_high_res[grid_indices].to(device)
                else:
                    grid_points = torch.empty((0, 3), device=device)
                
                # 3. Merge points
                query_points = torch.cat([grid_points, neighbor_points], dim=0)  # [n_points, 3]
                
                # 4. Create point type markers: 0 represents grid point, 1 represents nearby point
                point_types = torch.cat([
                    torch.zeros(n_grid, dtype=torch.long, device=device),  # grid point
                    torch.ones(n_neighbor, dtype=torch.long, device=device)  # nearby point
                ], dim=0)
                
                # 5. Randomly shuffle order (keep point type markers synchronized)
                perm = torch.randperm(self.n_points, device=device)
                query_points = query_points[perm]
                point_types = point_types[perm]
                
                return query_points, point_types

    def _center_molecule(self, coords) -> torch.Tensor:
        """
        Centers the molecule coordinates around the mean coordinate.

        Args:
            coords (torch.Tensor): The input molecule coordinates.

        Returns:
            torch.Tensor: The centered molecule coordinates.
        """
        mask = coords[:, 0] != PADDING_INDEX
        masked_coords = coords[mask]
        # get the center of the molecule
        max_coords = torch.max(masked_coords, dim=0).values
        min_coords = torch.min(masked_coords, dim=0).values
        center_coords = ((max_coords + min_coords) / 2).unsqueeze(0)
        masked_coords -= center_coords
        coords[mask] = masked_coords
        return coords

    def _scale_molecule(self, coords) -> torch.Tensor:
        """
        Scales the coordinates of a molecule.
        NOTE: This function is deprecated and will be removed in future versions.
        For now, it returns the original coordinates without scaling.

        Args:
            coords (torch.Tensor): A tensor containing the coordinates of the molecule.
                       The first dimension is assumed to be the batch size,
                       and the second dimension contains the coordinate values.

        Returns:
            torch.Tensor: The original coordinates tensor (no scaling applied).
        """
        # Return original coordinates without scaling
        return coords

    def _scale_batch_molecules(self, batch) -> torch.Tensor:
        """
        Scales the coordinates of molecules in a batch.
        NOTE: This function is deprecated and will be removed in future versions.
        For now, it returns the original coordinates without scaling.

        Args:
            batch (dict): A dictionary containing the batch data. It must include a key "coords"
                          which holds the coordinates of the molecules.

        Returns:
            torch.Tensor: The original coordinates of the molecules (no scaling applied).
        """
        # Return original coordinates without scaling
        return batch["coords"]

    def _rotate_coords(self, sample, rot_matrix=None) -> torch.Tensor:
        """
        Rotate the coordinates of a sample using a rotation matrix.

        Args:
            sample (dict): A dictionary containing the sample data, including the coordinates.
            rot_matrix (torch.Tensor, optional): The rotation matrix to use for rotation. If not provided, a random rotation matrix will be generated.

        Returns:
            torch.Tensor: The rotated coordinates.

        """
        if rot_matrix is None:
            rot_matrix = _random_rot_matrix()

        coords = sample["coords"]
        idx = sample["atoms_channel"] != PADDING_INDEX
        coords_masked = coords[idx]  # ignore value PADDING_INDEX
        coords_masked = torch.reshape(coords_masked, (-1, 3))

        # go to center of mass
        center_coords = torch.mean(coords_masked, dim=0)  # Calculate the average of the x, y, z axes (previously had index [0], i.e., only the average of the x axis)
        center_coords = center_coords.unsqueeze(0).tile((coords_masked.shape[0], 1))
        coords_masked = coords_masked - center_coords

        coords_rot = torch.einsum("ij, kj -> ki", rot_matrix, coords_masked)
        coords[: coords_rot.shape[0], :] = coords_rot

        return coords


def _random_rot_matrix() -> torch.Tensor:
    """Apply random rotation in each of hte x, y and z axis.
    First compute the 3D matrix for each rotation, then multiply them

    Returns:
        torch.Tensor: return rotation matrix (3x3)
    """
    theta = random.uniform(0, 2) * math.pi
    rot_x = torch.Tensor(
        [
            [1, 0, 0],
            [0, math.cos(theta), -math.sin(theta)],
            [0, math.sin(theta), math.cos(theta)],
        ]
    )
    theta = random.uniform(0, 2) * math.pi
    rot_y = torch.Tensor(
        [
            [math.cos(theta), 0, -math.sin(theta)],
            [0, 1, 0],
            [math.sin(theta), 0, math.cos(theta)],
        ]
    )
    theta = random.uniform(0, 2) * math.pi
    rot_z = torch.Tensor(
        [
            [math.cos(theta), -math.sin(theta), 0],
            [math.sin(theta), math.cos(theta), 0],
            [0, 0, 1],
        ]
    )

    return rot_z @ rot_y @ rot_x


################################################################################
# Batch field computation helper
def batch_compute_fields(gnf_converter, batch_data_list):
    """
    Batch compute target_field for multiple samples, improve performance
    
    Args:
        gnf_converter: GNFConverter instance
        batch_data_list: List containing multiple sample data, each element is a tuple of (coords, atoms_channel, query_points)
    
    Returns:
        list: List of target_field for each sample
    """
    if len(batch_data_list) == 0:
        return []
    
    # Collect data for all samples
    coords_list = []
    atoms_list = []
    query_points_list = []
    
    for coords, atoms_channel, query_points in batch_data_list:
        coords_list.append(coords)
        atoms_list.append(atoms_channel)
        query_points_list.append(query_points)
    
    # Batch compute: stack all samples into a batch
    # Note: Since the number of atoms in each sample may be different, we need to pad to the same length
    max_n_atoms = max(c.shape[0] for c in coords_list)
    max_n_points = max(q.shape[0] for q in query_points_list)
    
    batch_size = len(batch_data_list)
    device = query_points_list[0].device
    n_atom_types = gnf_converter.n_atom_types
    
    # Pad coords and atoms_channel
    padded_coords = torch.zeros(batch_size, max_n_atoms, 3, device=device)
    padded_atoms = torch.full((batch_size, max_n_atoms), PADDING_INDEX, device=device, dtype=torch.long)
    padded_query_points = torch.zeros(batch_size, max_n_points, 3, device=device)
    
    for i, (coords, atoms, q_points) in enumerate(zip(coords_list, atoms_list, query_points_list)):
        n_atoms = coords.shape[0]
        n_points = q_points.shape[0]
        padded_coords[i, :n_atoms] = coords
        padded_atoms[i, :n_atoms] = atoms
        padded_query_points[i, :n_points] = q_points
    
    # Batch compute field
    with torch.no_grad():
        batch_fields = gnf_converter.mol2gnf(padded_coords, padded_atoms, padded_query_points)
        # batch_fields: [batch_size, max_n_points, n_atom_types, 3]
    
    # Extract actual field for each sample (remove padded part)
    result_fields = []
    for i, (_, _, q_points) in enumerate(zip(coords_list, atoms_list, query_points_list)):
        n_points = q_points.shape[0]
        result_fields.append(batch_fields[i, :n_points])
    
    return result_fields


################################################################################
# create loaders
def create_field_loaders(
    config: dict,
    gnf_converter: GNFConverter,  # Receive GNFConverter instance
    split: str = "train",
    sample_full_grid = False,
):
    """
    Creates data loaders for training, validation, or testing datasets.

    Args:
        config (dict): Configuration dictionary containing dataset parameters.
        gnf_converter (GNFConverter): GNFConverter instance for computing target fields.
        split (str, optional): Dataset split to load. Options are "train", "val", or "test".
            Defaults to "train".
        sample_full_grid (bool, optional): Whether to sample the full grid. Defaults to False.

    Returns:
        DataLoader: Configured DataLoader for the specified dataset split.
    """

    # If joint fine-tuning is enabled, read sampling parameters from joint_finetune configuration, otherwise use default from dset
    joint_finetune_config = config.get("joint_finetune", {})
    joint_finetune_enabled = joint_finetune_config.get("enabled", False)
    
    # If joint fine-tuning is enabled and n_points is specified (not null), use the specified value, otherwise use default from dset.n_points
    n_points = config["dset"]["n_points"]
    if joint_finetune_enabled and "n_points" in joint_finetune_config and joint_finetune_config["n_points"] is not None:
        n_points = joint_finetune_config["n_points"]
    
    # Read targeted_sampling_ratio from configuration (ratio of grid points to nearby points)
    # Use joint_finetune configuration first, otherwise use default from dset
    if joint_finetune_enabled and "targeted_sampling_ratio" in joint_finetune_config and joint_finetune_config["targeted_sampling_ratio"] is not None:
        targeted_sampling_ratio = joint_finetune_config["targeted_sampling_ratio"]
    else:
        targeted_sampling_ratio = config["dset"].get("targeted_sampling_ratio", 2)
    
    # Read atom_distance_threshold from configuration
    # Use joint_finetune configuration first, otherwise use default from dset
    if joint_finetune_enabled and "atom_distance_threshold" in joint_finetune_config and joint_finetune_config["atom_distance_threshold"] is not None:
        atom_distance_threshold = joint_finetune_config["atom_distance_threshold"]
    else:
        atom_distance_threshold = config["dset"].get("atom_distance_threshold", 0.5)
    
    dset = FieldDataset(
        gnf_converter=gnf_converter,
        dset_name=config["dset"]["dset_name"],
        data_dir=config["dset"]["data_dir"],
        elements=config["dset"]["elements"],
        split=split,
        n_points=n_points,
        rotate=config["dset"]["data_aug"] if split == "train" else False,
        resolution=config["dset"]["resolution"],
        grid_dim=config["dset"]["grid_dim"],
        sample_full_grid=sample_full_grid,
        atom_distance_threshold=atom_distance_threshold,
        targeted_sampling_ratio=targeted_sampling_ratio,
    )

    loader = DataLoader(
        dset,
        batch_size=min(config["dset"]["batch_size"], len(dset)),
        num_workers=config["dset"]["num_workers"],
        shuffle=True if split == "train" else False,
        pin_memory=True,  # Accelerate GPU data transfer
        persistent_workers=True if config["dset"]["num_workers"] > 0 else False,  # keep workers to reduce startup cost
        drop_last=True,
    )
    print(f">> {split} set size: {len(dset)}")
    return loader


def create_gnf_converter(config: dict) -> GNFConverter:
    """
    Convenient function to create GNFConverter instance based on configuration
    
    Args:
        config (dict): Configuration dictionary containing gnf_converter parameters
        
    Returns:
        GNFConverter: Configured GNFConverter instance
    """
    # Force get converter configuration from config, no fallback
    if "converter" in config:
        gnf_config = config["converter"]
    elif "gnf_converter" in config:
        gnf_config = config["gnf_converter"]
    else:
        raise ValueError("GNF converter configuration not found in config! Must have either 'converter' or 'gnf_converter' key.")
    
    # Get the number of atom types from dataset config, no default
    dset_config = config["dset"]
    n_atom_types = dset_config["n_channels"]
    
    # Get gradient field method, no default
    gradient_field_method = gnf_config["gradient_field_method"]
    
    # Get method-specific configuration, no default
    method_configs = gnf_config["method_configs"]
    default_config = gnf_config["default_config"]
    
    # Select parameter config by method, force from configuration, no default
    if gradient_field_method in method_configs:
        method_config = method_configs[gradient_field_method]
        # Use method specific parameters, force get
        n_query_points = method_config["n_query_points"]
        step_size = method_config["step_size"]
        sig_sf = method_config["sig_sf"]
        sig_mag = method_config["sig_mag"]
        eps = method_config.get("eps", default_config["eps"])
        min_samples = method_config.get("min_samples", default_config["min_samples"])
    else:
        # Use default configuration, force get
        n_query_points = default_config["n_query_points"]
        step_size = default_config["step_size"]
        sig_sf = default_config["sig_sf"]
        sig_mag = default_config["sig_mag"]
        eps = default_config["eps"]
        min_samples = default_config["min_samples"]
    
    # Get other required parameters from configuration, no default
    sigma = gnf_config["sigma"]
    n_iter = gnf_config["n_iter"]
    temperature = gnf_config["temperature"]
    logsumexp_eps = gnf_config["logsumexp_eps"]
    inverse_square_strength = gnf_config["inverse_square_strength"]
    gradient_clip_threshold = gnf_config["gradient_clip_threshold"]
    sigma_ratios = gnf_config["sigma_ratios"]
    # Training specific gradient sampling parameters
    gradient_sampling_candidate_multiplier = gnf_config["gradient_sampling_candidate_multiplier"]
    # field variance sampling parameters
    field_variance_k_neighbors = gnf_config["field_variance_k_neighbors"]
    field_variance_weight = gnf_config["field_variance_weight"]
    
    # Early stopping related parameters (optional, default is disabled)
    enable_early_stopping = gnf_config["enable_early_stopping"]
    convergence_threshold = gnf_config["convergence_threshold"]
    min_iterations = gnf_config["min_iterations"]
    
    # Get the number of query_points for each atom type (optional)
    # If n_query_points_per_type in config, use it; else None and use unified n_query_points
    n_query_points_per_type = None
    if "n_query_points_per_type" in gnf_config:
        n_query_points_per_type = gnf_config["n_query_points_per_type"]
        # If None (null in configuration file), skip processing
        if n_query_points_per_type is not None:
            # Process OmegaConf DictConfig type, convert to Python dict
            if isinstance(n_query_points_per_type, DictConfig):
                n_query_points_per_type = OmegaConf.to_container(n_query_points_per_type, resolve=True)
            # Ensure it is a dictionary type
            if not isinstance(n_query_points_per_type, dict):
                raise ValueError(f"n_query_points_per_type must be a dictionary mapping atom symbols to query point counts, got {type(n_query_points_per_type)}")
        else:
            # If explicitly set to None/null, keep as None
            n_query_points_per_type = None
    
    # Get autoregressive clustering related parameters (optional)
    autoregressive_config = gnf_config["autoregressive_clustering"]
    enable_autoregressive_clustering = autoregressive_config["enable"]
    initial_min_samples = autoregressive_config["initial_min_samples"]
    min_samples_decay_factor = autoregressive_config["min_samples_decay_factor"]
    min_min_samples = autoregressive_config["min_min_samples"]
    max_clustering_iterations = autoregressive_config["max_clustering_iterations"]
    bond_length_tolerance = autoregressive_config["bond_length_tolerance"]
    bond_length_lower_tolerance = autoregressive_config.get("bond_length_lower_tolerance", 0.2)  # default 0.2
    enable_clustering_history = autoregressive_config["enable_clustering_history"]
    # Debug options (optional, default is False)
    debug_bond_validation = autoregressive_config.get("debug_bond_validation", False)
    # First N atoms exempt from bond length check (optional, default 3)
    n_initial_atoms_no_bond_check = autoregressive_config.get("n_initial_atoms_no_bond_check", 3)
    # Whether to enable bond length check (optional, default is True)
    enable_bond_validation = autoregressive_config.get("enable_bond_validation", True)
    
    # Get gradient batch size (optional, default is None, meaning process all points at once)
    gradient_batch_size = gnf_config.get("gradient_batch_size", None)
    
    # Get sampling range (optional, default is -7.0 and 7.0)
    sampling_range_min = gnf_config.get("sampling_range_min", -7.0)
    sampling_range_max = gnf_config.get("sampling_range_max", 7.0)
    
    return GNFConverter(
        sigma=sigma,
        n_query_points=n_query_points,
        n_iter=n_iter,
        step_size=step_size,
        eps=eps,
        min_samples=min_samples,
        sigma_ratios=sigma_ratios,
        gradient_field_method=gradient_field_method,
        temperature=temperature,
        logsumexp_eps=logsumexp_eps,
        inverse_square_strength=inverse_square_strength,
        gradient_clip_threshold=gradient_clip_threshold,
        sig_sf=sig_sf,
        sig_mag=sig_mag,
        gradient_sampling_candidate_multiplier=gradient_sampling_candidate_multiplier,
        field_variance_k_neighbors=field_variance_k_neighbors,  # Add field variance k nearest neighbors parameter
        field_variance_weight=field_variance_weight,  # Add field variance weight parameter
        n_atom_types=n_atom_types,  # Add number of atom types parameter
        enable_early_stopping=enable_early_stopping,
        convergence_threshold=convergence_threshold,
        min_iterations=min_iterations,
        n_query_points_per_type=n_query_points_per_type,  # Add number of query_points for each atom type
        enable_autoregressive_clustering=enable_autoregressive_clustering,
        initial_min_samples=initial_min_samples,
        min_samples_decay_factor=min_samples_decay_factor,
        min_min_samples=min_min_samples,
        max_clustering_iterations=max_clustering_iterations,
        bond_length_tolerance=bond_length_tolerance,
        bond_length_lower_tolerance=bond_length_lower_tolerance,
        enable_clustering_history=enable_clustering_history,
        debug_bond_validation=debug_bond_validation,
        gradient_batch_size=gradient_batch_size,
        n_initial_atoms_no_bond_check=n_initial_atoms_no_bond_check,
        enable_bond_validation=enable_bond_validation,
        sampling_range_min=sampling_range_min,
        sampling_range_max=sampling_range_max,
    )


def prepare_data_with_sample_idx(config, device, sample_idx, split="val"):
    """Prepare data containing specific sample.

    Args:
        config: Configuration object
        device: Compute device
        sample_idx: Index of the sample to load
        split: Dataset split, can be "train", "val", or "test", default is "val"

    Returns:
        Tuple[batch, coords, atoms_channel]: Data batch, coordinates and atom types
        batch: Single batch containing only the target sample (not the entire batch)
        coords: Coordinates of the target sample [1, n_atoms, 3]
        atoms_channel: Atom types of the target sample [1, n_atoms]
    """
    
    gnf_converter = create_gnf_converter(config)
    loader_val = create_field_loaders(config, gnf_converter, split=split)
    
    # Calculate how many batches need to be skipped to reach the target sample
    batch_size = config.dset.batch_size
    target_batch_idx = sample_idx // batch_size
    sample_in_batch = sample_idx % batch_size
    
    # Skip previous batches
    for i, batch in enumerate(loader_val):
        if i == target_batch_idx:
            batch = batch.to(device)
            
            # Extract the index range of the target sample in the batch
            batch_idx = batch.batch  # [N_total_atoms] Each atom belongs to which sample
            target_sample_mask = batch_idx == sample_in_batch
            
            # Extract data of the target sample
            target_pos = batch.pos[target_sample_mask]  # [n_atoms, 3]
            target_x = batch.x[target_sample_mask]  # [n_atoms]
            
            # Extract other attributes (if exist)
            target_data = {}
            if hasattr(batch, 'xs'):
                # xs is [N_total_points, 3], need to extract based on sample index
                # Note: xs may not be organized by samples, need to check its structure
                # If xs is organized by samples, need to handle accordingly
                # Here we assume xs is shared or needs special handling
                if batch.xs.shape[0] == batch.pos.shape[0]:
                    # If xs and pos have the same length, it means they are organized by atoms
                    target_data['xs'] = batch.xs[target_sample_mask]
                else:
                    # If the lengths are different, it may be organized by query points, need to keep all or split by samples
                    # Here we keep all, and may need to adjust based on actual usage
                    target_data['xs'] = batch.xs
            
            if hasattr(batch, 'target_field'):
                # Handling of target_field is similar to xs
                if batch.target_field.shape[0] == batch.pos.shape[0]:
                    target_data['target_field'] = batch.target_field[target_sample_mask]
                else:
                    target_data['target_field'] = batch.target_field
            
            if hasattr(batch, 'idx'):
                # idx is [batch_size], extract the index of the target sample
                target_data['idx'] = batch.idx[sample_in_batch:sample_in_batch+1] if batch.idx is not None else None
            
            # Create Data object containing only the target sample
            single_data = Data(
                pos=target_pos,
                x=target_x,
                batch=torch.zeros(target_pos.shape[0], dtype=torch.long, device=device),  # all atoms belong to sample 0
                **target_data
            )
            
            # Create Batch containing only the single sample
            single_batch = Batch.from_data_list([single_data])
            
            # Extract coordinates and atom types (for compatibility)
            coords, _ = to_dense_batch(single_batch.pos, single_batch.batch, fill_value=0)
            atoms_channel, _ = to_dense_batch(single_batch.x, single_batch.batch, fill_value=PADDING_INDEX)
            
            return single_batch, coords, atoms_channel
    
    raise IndexError(f"Sample index {sample_idx} is out of bounds for {split} set") 