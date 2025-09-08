import itertools
import math
import os
import random
import pickle
import lmdb
import threading
from typing import Optional

from lightning import Fabric
import numpy as np
import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader

from funcmol.models.decoder import get_grid
from funcmol.utils.constants import ELEMENTS_HASH, PADDING_INDEX
from funcmol.utils.gnf_converter import GNFConverter

# 进程本地存储，用于在多进程环境下管理LMDB连接
_thread_local = threading.local()


class FieldDataset(Dataset):
    """
    Initializes the dataset with the specified parameters.

    Args:
        gnf_converter (GNFConverter): GNFConverter实例用于计算目标梯度场
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
        gnf_converter: GNFConverter,  # 接收GNFConverter实例
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
        
        self.targeted_sampling_ratio = targeted_sampling_ratio if split == "train" else 0
        
        # 根据数据加载方式设置field_idxs
        if hasattr(self, 'use_lmdb') and self.use_lmdb:
            # LMDB模式下，使用molid2idx的长度
            self.field_idxs = torch.arange(len(self.molid2idx))
        else:
            # 传统模式下，使用data的长度
            self.field_idxs = torch.arange(len(self.data))
            
        self.discrete_grid, self.full_grid_high_res = get_grid(self.grid_dim, self.resolution)
        
        # 使用传入的GNFConverter实例
        self.gnf_converter = gnf_converter

        # if self.db is None:
        #     self._connect_db()
            

    def __del__(self):
        """析构函数，确保数据库连接被关闭"""
        if hasattr(self, 'db') and self.db is not None:
            self._close_db()

    def _read_data(self):
        fname = f"{self.split}_data"
        if self.dset_name == "cremp":
            fname = f"{self.split}_50_data"
        
        # 检查是否存在LMDB数据库
        lmdb_path = os.path.join(self.data_dir, self.dset_name, f"{fname}.lmdb")
        molid2idx_path = os.path.join(self.data_dir, self.dset_name, f"{fname}_molid2idx.pt")
        
        if os.path.exists(lmdb_path) and os.path.exists(molid2idx_path):
            # 使用LMDB数据库
            self._use_lmdb_database(lmdb_path, molid2idx_path)
        else:
            # 使用传统的torch.load方式
            self.data = torch.load(os.path.join(
                self.data_dir, self.dset_name, f"{fname}.pth"), weights_only=False
            )
            self.use_lmdb = False

    def _use_lmdb_database(self, lmdb_path, molid2idx_path):
        """使用LMDB数据库加载数据"""
        self.lmdb_path = lmdb_path
        self.molid2idx = torch.load(molid2idx_path)
        self.db = None
        self.keys = None
        self.use_lmdb = True
        print(f"  | Using LMDB database: {lmdb_path}")
        print(f"  | Database contains {len(self.molid2idx)} molecules")

    def _connect_db(self):
        """建立只读数据库连接 - 进程安全版本"""
        if self.db is None:
            import os
            
            # 使用线程本地存储确保每个worker进程有独立的连接
            if not hasattr(_thread_local, 'lmdb_connections'):
                _thread_local.lmdb_connections = {}
            
            # 为每个LMDB路径创建独立的连接
            if self.lmdb_path not in _thread_local.lmdb_connections:
                _thread_local.lmdb_connections[self.lmdb_path] = lmdb.open(
                    self.lmdb_path,
                    map_size=10*(1024*1024*1024),   # 10GB
                    create=False,
                    subdir=True,
                    readonly=True,
                    lock=False,
                    readahead=False,
                    meminit=False,
                )
            
            self.db = _thread_local.lmdb_connections[self.lmdb_path]
            
            # 使用只读事务获取keys
            with self.db.begin() as txn:
                self.keys = list(txn.cursor().iternext(values=False))

    def _close_db(self):
        """关闭数据库连接"""
        if self.db is not None:
            self.db.close()
            self.db = None
            self.keys = None

    def _filter_by_elements(self, elements) -> None:
        if hasattr(self, 'use_lmdb') and self.use_lmdb:
            # LMDB模式下，过滤在__getitem__中进行
            self.elements_ids = [ELEMENTS_HASH[element] for element in elements]
            return
        
        # 传统模式下的过滤
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
        if hasattr(self, 'use_lmdb') and self.use_lmdb:
            if self.db is None:
                self._connect_db()
            return len(self.keys)
        return self.field_idxs.size(0)

    def __getitem__(self, index) -> Data:
        if hasattr(self, 'use_lmdb') and self.use_lmdb:
            return self._getitem_lmdb(index)
        else:
            return self._getitem_traditional(index)

    def _getitem_lmdb(self, index) -> Data:
        """LMDB模式下的数据获取 - 进程安全版本"""
        # 确保数据库连接在worker进程中建立
        if self.db is None:
            self._connect_db()
        
        key = self.keys[index]
        
        # 使用更安全的事务处理
        try:
            with self.db.begin() as txn:
                sample_raw = pickle.loads(txn.get(key))
        except Exception as e:
            # 如果事务失败，重新连接数据库
            print(f"LMDB transaction failed, reconnecting: {e}")
            self._close_db()
            self._connect_db()
            with self.db.begin() as txn:
                sample_raw = pickle.loads(txn.get(key))
        
        # 检查元素过滤
        if hasattr(self, 'elements_ids'):
            atoms = sample_raw["atoms_channel"][sample_raw["atoms_channel"] != PADDING_INDEX]
            include = True
            for atom_id in atoms.unique():
                if int(atom_id.item()) not in self.elements_ids:
                    include = False
                    break
            if not include:
                # 如果不符合元素要求，返回下一个有效的数据
                return self._getitem_lmdb((index + 1) % len(self))
        
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
            target_field=target_field.float(),  # 新增：添加目标梯度场
            idx=torch.tensor([index], dtype=torch.long)
        )
        return data

    def _getitem_traditional(self, index) -> Data:
        """传统模式下的数据获取"""
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
            target_field=target_field.float(),  # 新增：添加目标梯度场
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

        # 移除分子缩放，保持原始坐标
        # sample["coords"] = self._scale_molecule(sample["coords"]) # 原来的 funcmol 根本就没有用 _scale_molecule

        return sample

    def _get_xs(self, sample) -> torch.Tensor:
        mask = sample["atoms_channel"] != PADDING_INDEX
        coords = sample["coords"][mask]
        device = coords.device
        
        if self.sample_full_grid:
            return self.full_grid_high_res.to(device)
        
        grid_size = len(self.full_grid_high_res) # grid_dim**3

        if self.targeted_sampling_ratio >= 1:
            # 1. 目标采样：原子周围的点
            # 注意：coords 现在保持原始坐标，不再进行缩放
            # coords 的范围是真实的埃单位
            rand_points = coords
            # 添加小的噪声，噪声大小应该与网格分辨率相关
            noise = torch.randn_like(rand_points, device=device) * self.resolution / 4  # Scale noise by resolution
            rand_points = (rand_points + noise)  # .floor().long() TODO：如果这里floor()再long()，会全部变成-1,0,1
            
            # 计算每个原子周围需要采样的点数
            points_per_atom = max(1, (self.n_points // coords.size(0)) // self.targeted_sampling_ratio)
            random_indices = torch.randperm(self.increments.size(0), device=device)[:points_per_atom]
            
            # 在原子周围采样
            rand_points = (rand_points.unsqueeze(1) + self.increments[random_indices].to(device)).view(-1, 3)
            # Calculate bounds based on real-world distances
            total_span = (self.grid_dim - 1) * self.resolution
            half_span = total_span / 2
            min_bound = -half_span
            max_bound = half_span
            rand_points = torch.clamp(rand_points, min_bound, max_bound).float()
            
            # 2. 随机采样网格点
            grid_indices = torch.randperm(grid_size, device=device)[:self.n_points]
            grid_points = self.full_grid_high_res[grid_indices].to(device)
            
            # 3. 合并并去重
            all_points = torch.cat([rand_points, grid_points], dim=0)
            unique_points = torch.unique(all_points, dim=0)
            
            # 4. 随机选择所需数量的点
            indices = torch.randperm(unique_points.size(0), device=device)[:self.n_points]
            return unique_points[indices]
        
        else:
            # raise NotImplementedError("Targeted sampling ratio < 1 is not implemented") 
            # （在init里有写，如果split是val，则targeted_sampling_ratio=0，所以这个else是有用的，是用来处理val的）
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
        center_coords = torch.mean(coords_masked, dim=0)  # 计算x,y,z三个轴的平均值（之前有索引[0]，也就是只取x轴的平均值）
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
    gnf_converter: GNFConverter,  # 接收GNFConverter实例
    split: str = "train",
    fabric = Fabric(),
    n_samples = None,
    sample_full_grid = False,
):
    """
    Creates data loaders for training, validation, or testing datasets.

    Args:
        config (dict): Configuration dictionary containing dataset parameters.
        gnf_converter (GNFConverter): GNFConverter instance for computing target fields.
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
        gnf_converter=gnf_converter,  # 传入GNFConverter实例
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
        if hasattr(dset, 'data'):
            dset.data = [dset.data[0]]  # 只保留第一个分子，不复制
        dset.field_idxs = torch.arange(1)  # 只保留一个分子
    elif config.get("debug_subset", False):
        if hasattr(dset, 'data'):
            dset.data = dset.data[:128]
        # dset.field_idxs = torch.arange(min(128, len(dset.field_idxs)))  # 限制为128个分子
        dset.keys = torch.arange(min(32, len(dset.keys)))  # 限制为32个分子

    loader = DataLoader(
        dset,
        batch_size=min(config["dset"]["batch_size"], len(dset)),
        num_workers=config["dset"]["num_workers"],
        shuffle=True if split == "train" else False,
        pin_memory=False,
        drop_last=True,
    )
    fabric.print(f">> {split} set size: {len(dset)}")

    return fabric.setup_dataloaders(loader, use_distributed_sampler=True)  # 所有split都使用分布式采样器


def create_gnf_converter(config: dict, device: str = "cpu") -> GNFConverter:
    """
    根据配置创建GNFConverter实例的便捷函数。
    
    Args:
        config (dict): 配置字典，包含gnf_converter相关参数
        device (str): 设备类型，默认为"cpu"
        
    Returns:
        GNFConverter: 配置好的GNFConverter实例
    """
    gnf_config = config.get("converter", {}) or config.get("gnf_converter", {})
    
    # 检查配置是否存在
    if not gnf_config:
        raise ValueError("GNF converter configuration not found in config!")
    
    # 获取数据集配置中的原子类型数量
    dset_config = config.get("dset", {})
    n_atom_types = dset_config.get("n_channels", 5)  # 默认为5以保持向后兼容
    
    # 获取梯度场方法
    gradient_field_method = gnf_config.get("gradient_field_method")
    if gradient_field_method is None:
        raise ValueError("gradient_field_method not found in converter config!")
    
    # 获取方法特定的配置
    method_configs = gnf_config.get("method_configs", {})
    default_config = gnf_config.get("default_config", {})
    
    # 根据方法选择参数配置，移除所有硬编码默认值
    if gradient_field_method in method_configs:
        method_config = method_configs[gradient_field_method]
        # 使用方法特定的参数
        n_query_points = method_config.get("n_query_points")
        step_size = method_config.get("step_size")
        sig_sf = method_config.get("sig_sf")
        sig_mag = method_config.get("sig_mag")
        
        if n_query_points is None:
            raise ValueError(f"n_query_points not found in method_config for {gradient_field_method}")
        if step_size is None:
            raise ValueError(f"step_size not found in method_config for {gradient_field_method}")
        if sig_sf is None:
            raise ValueError(f"sig_sf not found in method_config for {gradient_field_method}")
        if sig_mag is None:
            raise ValueError(f"sig_mag not found in method_config for {gradient_field_method}")
    else:
        # 使用默认配置
        n_query_points = default_config.get("n_query_points")
        step_size = default_config.get("step_size")
        sig_sf = default_config.get("sig_sf")
        sig_mag = default_config.get("sig_mag")
        
        if n_query_points is None:
            raise ValueError("n_query_points not found in default_config")
        if step_size is None:
            raise ValueError("step_size not found in default_config")
        if sig_sf is None:
            raise ValueError("sig_sf not found in default_config")
        if sig_mag is None:
            raise ValueError("sig_mag not found in default_config")
    
    # 获取其他必需参数，移除所有硬编码默认值
    sigma = gnf_config.get("sigma")
    n_iter = gnf_config.get("n_iter")
    eps = gnf_config.get("eps")
    min_samples = gnf_config.get("min_samples")
    temperature = gnf_config.get("temperature")
    logsumexp_eps = gnf_config.get("logsumexp_eps")
    inverse_square_strength = gnf_config.get("inverse_square_strength")
    gradient_clip_threshold = gnf_config.get("gradient_clip_threshold")
    sigma_ratios = gnf_config.get("sigma_ratios")
    
    # 检查所有必需参数是否存在
    required_params = {
        "sigma": sigma,
        "n_iter": n_iter,
        "eps": eps,
        "min_samples": min_samples,
        "temperature": temperature,
        "logsumexp_eps": logsumexp_eps,
        "inverse_square_strength": inverse_square_strength,
        "gradient_clip_threshold": gradient_clip_threshold
    }
    
    missing_params = [param for param, value in required_params.items() if value is None]
    if missing_params:
        raise ValueError(f"Missing required parameters in converter config: {missing_params}")
    
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
        n_atom_types=n_atom_types,  # 添加原子类型数量参数
        device=device
    ) 