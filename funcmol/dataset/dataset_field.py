import itertools
import math
import os
import random
from typing import Optional

from lightning import Fabric
import numpy as np
import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader

from funcmol.models.decoder import get_grid
from funcmol.utils.constants import ELEMENTS_HASH, PADDING_INDEX


class FieldDataset(Dataset):
    """
    Initializes the dataset with the specified parameters.

    Args:
        dset_name (str): Name of the dataset. Default is "qm9".
        data_dir (str): Directory where the dataset is stored. Default is "dataset/data".
        elements (list): List of elements to filter by. Default is None, which uses ELEMENTS_HASH.
        split (str): Dataset split to use. Must be one of ["train", "val", "test"]. Default is "train".
        rotate (bool): Whether to apply rotation. Default is False.
        radius (float): Radius for some operation. Default is 0.5.
        grid_dim (int): Dimension of the grid. Default is 32.
        resolution (float): Resolution of the grid. Default is 0.25.
        n_points (int): Number of points to sample. Default is 4000.
        sample_full_grid (bool): Whether to sample the full grid. Default is False.
        targeted_sampling_ratio (int): Ratio for targeted sampling. Default is 2.
        cubes_around (int): Number of cubes around to consider. Default is 3.
        debug_one_mol (bool): Whether to keep only the first molecule for debugging. Default is False.
        debug_subset (bool): Whether to use only the first 128 molecules for debugging. Default is False.
    """
    def __init__(
        self,
        dset_name: str = "qm9",
        data_dir: str = "dataset/data",
        elements: Optional[dict] = None,
        split: str = "train",
        rotate: bool = False,
        radius: float = 0.5,
        grid_dim: int = 32,
        resolution: float = 0.25,
        n_points: int = 4000,
        sample_full_grid: bool = False,
        targeted_sampling_ratio: int = 2,
        cubes_around: int = 3,
        debug_one_mol: bool = False,
        debug_subset: bool = False,
    ):
        if elements is None:
            elements = ELEMENTS_HASH
        assert dset_name in ["qm9", "drugs", "cremp"]
        assert split in ["train", "val", "test"]
        self.dset_name = dset_name
        self.data_dir = data_dir
        self.elements = elements
        self.split = split
        self.rotate = rotate
        self.fix_radius = radius
        self.resolution = resolution
        self.n_points = n_points
        self.sample_full_grid = sample_full_grid
        self.grid_dim = grid_dim

        self._read_data()
        self._filter_by_elements(elements)

        self.increments = torch.tensor(
            list(itertools.product(list(range(-cubes_around, cubes_around+1)), repeat=3))
        )
        self.targeted_sampling_ratio = targeted_sampling_ratio if split == "train" else 0
        self.field_idxs = torch.arange(len(self.data))
        self.discrete_grid, self.full_grid_high_res = get_grid(self.grid_dim)

    def _read_data(self):
        fname = f"{self.split}_data"
        if self.dset_name == "cremp":
            fname = f"{self.split}_50_data"
        self.data = torch.load(os.path.join(
            self.data_dir, self.dset_name, f"{fname}.pth"), weights_only=False
        )

    def _filter_by_elements(self, elements) -> None:
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
        return self.field_idxs.size(0)

    def __getitem__(self, index) -> Data:
        # get raw data
        idx = self.field_idxs[index]
        sample_raw = self.data[idx]
        
        # preprocess molecule (handles rotation and centering)
        sample = self._preprocess_molecule(sample_raw)

        # sample points for field prediction
        xs = self._get_xs(sample)

        # get data from preprocessed sample
        coords = sample["coords"]
        atoms_channel = sample["atoms_channel"]

        # remove padding to handle variable-sized molecules
        valid_mask = atoms_channel != PADDING_INDEX
        coords = coords[valid_mask]
        atoms_channel = atoms_channel[valid_mask]

        # create torch_geometric data object
        data = Data(
            pos=coords.float(),
            x=atoms_channel.long(),
            xs=xs.float(),
            idx=torch.tensor([index], dtype=torch.long)
        )
        return data

    def _preprocess_molecule(self, sample_raw) -> dict:
        """
        Preprocesses a raw molecule sample by removing invalid values and fixing
        the radius if necessary.

        Args:
            sample_raw (dict): The raw molecule sample.

        Returns:
            dict: The preprocessed molecule sample.
        """
        sample = {
            "coords": sample_raw["coords"],
            "atoms_channel": sample_raw["atoms_channel"],
            "radius": sample_raw["radius"]
        }
        if self.fix_radius > 0:
            sample["radius"].fill_(self.fix_radius)

        if self.rotate:
            sample["coords"] = self._rotate_coords(sample)

        sample["coords"] = self._center_molecule(sample["coords"])

        sample["coords"] = self._scale_molecule(sample["coords"])

        return sample

    def _get_xs(self, sample) -> torch.Tensor:
        mask = sample["atoms_channel"] != PADDING_INDEX
        coords = sample["coords"][mask]
        device = coords.device
        
        if self.sample_full_grid:
            return self.full_grid_high_res.to(device)
        
        grid_size = len(self.full_grid_high_res)
        if grid_size < self.n_points:
            raise ValueError(f"Grid size {grid_size} < n_points {self.n_points}")
        
        if self.targeted_sampling_ratio >= 1:
            # 1. 目标采样：原子周围的点
            rand_points = coords / self.resolution
            noise = torch.randn_like(rand_points, device=device) * (self.resolution / 4)
            rand_points = (rand_points + noise).floor().long()
            
            # 计算每个原子周围需要采样的点数
            points_per_atom = max(1, (self.n_points // coords.size(0)) // self.targeted_sampling_ratio)
            random_indices = torch.randperm(self.increments.size(0), device=device)[:points_per_atom]
            
            # 在原子周围采样
            rand_points = (rand_points.unsqueeze(1) + self.increments[random_indices].to(device)).view(-1, 3)
            min_bound = -(self.grid_dim // 2)
            max_bound = (self.grid_dim - 1) // 2
            rand_points = torch.clamp(rand_points, min_bound, max_bound).float() / (self.grid_dim // 2)
            
            # 2. 随机采样网格点
            grid_indices = torch.randperm(grid_size, device=device)[:self.n_points]
            grid_points = self.full_grid_high_res[grid_indices].to(device)
            
            # 3. 合并并去重
            all_points = torch.cat([rand_points, grid_points], dim=0)
            unique_points = torch.unique(all_points, dim=0)
            
            # 4. 如果点数不足，补充随机网格点
            if unique_points.size(0) < self.n_points:
                remaining = self.n_points - unique_points.size(0)
                used_mask = torch.zeros(grid_size, dtype=torch.bool, device=device)
                used_mask[grid_indices] = True
                available_indices = torch.nonzero(~used_mask).squeeze(-1)
                
                if len(available_indices) >= remaining:
                    additional_indices = available_indices[torch.randperm(len(available_indices))[:remaining]]
                else:
                    additional_indices = torch.randint(grid_size, (remaining,), device=device)
                
                additional_points = self.full_grid_high_res[additional_indices].to(device)
                unique_points = torch.cat([unique_points, additional_points], dim=0)
            
            # 5. 随机选择所需数量的点
            indices = torch.randperm(unique_points.size(0), device=device)[:self.n_points]
            return unique_points[indices]
        
        else:
            # 简单随机采样
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
            return unique_points[indices]

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

        This method scales the input coordinates by dividing the masked coordinates
        by the product of the resolution and half of the grid dimension. The scaling
        is applied only to the coordinates that are not equal to the PADDING_INDEX.

        Args:
            coords (torch.Tensor): A tensor containing the coordinates of the molecule.
                       The first dimension is assumed to be the batch size,
                       and the second dimension contains the coordinate values.

        Returns:
            torch.Tensor: The scaled coordinates tensor.
        """
        mask = coords[:, 0] != PADDING_INDEX
        masked_coords = coords[mask]
        masked_coords = masked_coords / (self.resolution * self.grid_dim / 2)
        coords[mask] = masked_coords
        return coords

    def _scale_batch_molecules(self, batch) -> torch.Tensor:
        """
        Scales the coordinates of molecules in a batch.

        Args:
            batch (dict): A dictionary containing the batch data. It must include a key "coords"
                          which holds the coordinates of the molecules.

        Returns:
            torch.Tensor: The scaled coordinates of the molecules.
        """
        coords = batch["coords"]
        mask = coords[:, :, 0] != PADDING_INDEX
        coords[mask] = coords[mask] / (self.resolution * self.grid_dim / 2)
        return coords

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
        center_coords = torch.mean(coords_masked, dim=0)[0]
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
# create loaders
def create_field_loaders(
    config: dict,
    split: str = "train",
    fabric = Fabric(),
    n_samples = None,
    sample_full_grid = False,
):
    """
    Creates data loaders for training, validation, or testing datasets.

    Args:
        config (dict): Configuration dictionary containing dataset parameters.
        split (str, optional): Dataset split to load. Options are "train", "val", or "test".
            Defaults to "train".
        fabric (Fabric, optional): Fabric object for distributed training.
            Defaults to a new Fabric instance.
        n_samples (int, optional): Number of samples to use for validation or testing.
            If None, defaults to 5000. Defaults to None.
        sample_full_grid (bool, optional): Whether to sample the full grid. Defaults to False.

    Returns:
        DataLoader: Configured DataLoader for the specified dataset split.
    """
    dset = FieldDataset(
        dset_name=config["dset"]["dset_name"],
        data_dir=config["dset"]["data_dir"],
        elements=config["dset"]["elements"],
        split=split,
        n_points=config["dset"]["n_points"],
        rotate=config["dset"]["data_aug"] if split == "train" else False,
        resolution=config["dset"]["resolution"],
        grid_dim=config["dset"]["grid_dim"],
        radius=config["dset"]["atomic_radius"],
        sample_full_grid=sample_full_grid,
        debug_one_mol=config.get("debug_one_mol", False),
        debug_subset=config.get("debug_subset", False),
    )

    if config.get("debug_one_mol", False):
        dset.data = [dset.data[0]]  # 只保留第一个分子，不复制
        dset.field_idxs = torch.arange(len(dset.data))
    elif config.get("debug_subset", False):
        dset.data = dset.data[:16]
        dset.field_idxs = torch.arange(len(dset.data))

    loader = DataLoader(
        dset,
        batch_size=min(config["dset"]["batch_size"], len(dset)),
        num_workers=config["dset"]["num_workers"],
        shuffle=True if split == "train" else False,
        pin_memory=True,
        drop_last=True,
    )
    fabric.print(f">> {split} set size: {len(dset)}")

    return fabric.setup_dataloaders(loader, use_distributed_sampler=(split == "train"))
