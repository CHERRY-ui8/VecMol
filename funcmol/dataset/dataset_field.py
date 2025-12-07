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
            # LMDB模式下，使用keys的长度
            self.field_idxs = torch.arange(len(self.keys))
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
        # NOTE: remove molid2idx definition
        keys_path = os.path.join(self.data_dir, self.dset_name, f"{fname}_keys.pt")
        
        if os.path.exists(lmdb_path) and os.path.exists(keys_path):
            # 使用LMDB数据库
            self._use_lmdb_database(lmdb_path, keys_path)
        else:
            # 使用传统的torch.load方式
            self.data = torch.load(os.path.join(
                self.data_dir, self.dset_name, f"{fname}.pth"), weights_only=False
            )
            self.use_lmdb = False

    def _use_lmdb_database(self, lmdb_path, keys_path):
        """使用LMDB数据库加载数据"""
        self.lmdb_path = lmdb_path
        self.keys = torch.load(keys_path)  # 直接加载keys文件
        self.db = None
        self.use_lmdb = True
        
        print(f"  | Using LMDB database: {lmdb_path}")
        print(f"  | Database contains {len(self.keys)} molecules")

    def _connect_db(self):
        """建立只读数据库连接 - 进程安全版本"""
        if self.db is None:
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

    def _close_db(self):
        """关闭数据库连接"""
        if self.db is not None:
            self.db.close()
            self.db = None
            self.keys = None

    def _filter_by_elements(self, elements) -> None:
        if hasattr(self, 'use_lmdb') and self.use_lmdb:
            # LMDB模式下，元素过滤已在convert_to_lmdb.py中完成
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
        # 确保key是bytes格式，因为LMDB需要bytes
        if isinstance(key, str):
            key = key.encode('utf-8')
        
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
        
        # NOTE: 过滤操作转移到了convert_to_lmdb.py中进行
        
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
            rand_points = (rand_points + noise)  # .floor().long() NOTE：如果这里floor()再long()，会全部变成-1,0,1
            
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
        debug_one_mol=config["debug_one_mol"],
        debug_subset=config["debug_subset"],
    )

    if config["debug_one_mol"]:
        if hasattr(dset, 'data'):
            dset.data = [dset.data[0]]  # 只保留第一个分子，不复制
        dset.field_idxs = torch.arange(1)  # 只保留一个分子
    elif config["debug_subset"]:
        if hasattr(dset, 'data'):
            dset.data = dset.data[:128]
        # dset.field_idxs = torch.arange(min(128, len(dset.field_idxs)))  # 限制为128个分子
        dset.keys = torch.arange(min(32, len(dset.keys)))  # 限制为32个分子

    loader = DataLoader(
        dset,
        batch_size=min(config["dset"]["batch_size"], len(dset)),
        num_workers=config["dset"]["num_workers"],
        shuffle=True if split == "train" else False,
        pin_memory=True,
        drop_last=True,
    )
    print(f">> {split} set size: {len(dset)}")
    return loader


def create_gnf_converter(config: dict) -> GNFConverter:
    """
    根据配置创建GNFConverter实例的便捷函数。
    
    Args:
        config (dict): 配置字典，包含gnf_converter相关参数
        
    Returns:
        GNFConverter: 配置好的GNFConverter实例
    """
    # 强制从config中获取converter配置，不允许fallback
    if "converter" in config:
        gnf_config = config["converter"]
    elif "gnf_converter" in config:
        gnf_config = config["gnf_converter"]
    else:
        raise ValueError("GNF converter configuration not found in config! Must have either 'converter' or 'gnf_converter' key.")
    
    # 获取数据集配置中的原子类型数量，不允许默认值
    dset_config = config["dset"]
    n_atom_types = dset_config["n_channels"]
    
    # 获取梯度场方法，不允许默认值
    gradient_field_method = gnf_config["gradient_field_method"]
    
    # 获取方法特定的配置，不允许默认值
    method_configs = gnf_config["method_configs"]
    default_config = gnf_config["default_config"]
    
    # 根据方法选择参数配置，强制从配置中获取，不允许默认值
    if gradient_field_method in method_configs:
        method_config = method_configs[gradient_field_method]
        # 使用方法特定的参数，强制获取
        n_query_points = method_config["n_query_points"]
        step_size = method_config["step_size"]
        sig_sf = method_config["sig_sf"]
        sig_mag = method_config["sig_mag"]
        eps = method_config.get("eps", default_config["eps"])
        min_samples = method_config.get("min_samples", default_config["min_samples"])
    else:
        # 使用默认配置，强制获取
        n_query_points = default_config["n_query_points"]
        step_size = default_config["step_size"]
        sig_sf = default_config["sig_sf"]
        sig_mag = default_config["sig_mag"]
        eps = default_config["eps"]
        min_samples = default_config["min_samples"]
    
    # 获取其他必需参数，强制从配置中获取，不允许默认值
    sigma = gnf_config["sigma"]
    n_iter = gnf_config["n_iter"]
    temperature = gnf_config["temperature"]
    logsumexp_eps = gnf_config["logsumexp_eps"]
    inverse_square_strength = gnf_config["inverse_square_strength"]
    gradient_clip_threshold = gnf_config["gradient_clip_threshold"]
    sigma_ratios = gnf_config["sigma_ratios"]
    # 训练版专有的梯度采样参数
    gradient_sampling_candidate_multiplier = gnf_config["gradient_sampling_candidate_multiplier"]
    # field变化率采样参数
    field_variance_k_neighbors = gnf_config["field_variance_k_neighbors"]
    field_variance_weight = gnf_config["field_variance_weight"]
    
    # 早停相关参数（可选，默认关闭早停）
    enable_early_stopping = gnf_config["enable_early_stopping"]
    convergence_threshold = gnf_config["convergence_threshold"]
    min_iterations = gnf_config["min_iterations"]
    
    # 获取每个原子类型的 query_points 数（可选）
    # 如果配置中提供了 n_query_points_per_type，则使用它；否则为 None，将使用统一的 n_query_points
    n_query_points_per_type = None
    if "n_query_points_per_type" in gnf_config:
        n_query_points_per_type = gnf_config["n_query_points_per_type"]
        # 如果为 None（配置文件中的 null），则跳过处理
        if n_query_points_per_type is not None:
            # 处理 OmegaConf 的 DictConfig 类型，转换为 Python dict
            if isinstance(n_query_points_per_type, DictConfig):
                n_query_points_per_type = OmegaConf.to_container(n_query_points_per_type, resolve=True)
            # 确保是字典类型
            if not isinstance(n_query_points_per_type, dict):
                raise ValueError(f"n_query_points_per_type must be a dictionary mapping atom symbols to query point counts, got {type(n_query_points_per_type)}")
        else:
            # 如果明确设置为 None/null，保持为 None
            n_query_points_per_type = None
    
    # 获取自回归聚类相关参数（可选）
    autoregressive_config = gnf_config["autoregressive_clustering"]
    enable_autoregressive_clustering = autoregressive_config["enable"]
    initial_min_samples = autoregressive_config["initial_min_samples"]
    min_samples_decay_factor = autoregressive_config["min_samples_decay_factor"]
    min_min_samples = autoregressive_config["min_min_samples"]
    max_clustering_iterations = autoregressive_config["max_clustering_iterations"]
    bond_length_tolerance = autoregressive_config["bond_length_tolerance"]
    enable_clustering_history = autoregressive_config["enable_clustering_history"]
    # 调试选项（可选，默认为False）
    debug_bond_validation = autoregressive_config.get("debug_bond_validation", False)
    
    # 获取梯度批次大小（可选，默认为None，表示一次性处理所有点）
    gradient_batch_size = gnf_config.get("gradient_batch_size", None)
    
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
        field_variance_k_neighbors=field_variance_k_neighbors,  # 添加field方差k最近邻参数
        field_variance_weight=field_variance_weight,  # 添加field方差权重参数
        n_atom_types=n_atom_types,  # 添加原子类型数量参数
        enable_early_stopping=enable_early_stopping,
        convergence_threshold=convergence_threshold,
        min_iterations=min_iterations,
        n_query_points_per_type=n_query_points_per_type,  # 添加每个原子类型的 query_points 数
        # 自回归聚类参数
        enable_autoregressive_clustering=enable_autoregressive_clustering,
        initial_min_samples=initial_min_samples,
        min_samples_decay_factor=min_samples_decay_factor,
        min_min_samples=min_min_samples,
        max_clustering_iterations=max_clustering_iterations,
        bond_length_tolerance=bond_length_tolerance,
        enable_clustering_history=enable_clustering_history,
        debug_bond_validation=debug_bond_validation,
        gradient_batch_size=gradient_batch_size,
    )


def prepare_data_with_sample_idx(config, device, sample_idx):
    """准备包含特定样本的数据。

    Args:
        config: 配置对象
        device: 计算设备
        sample_idx: 要加载的样本索引

    Returns:
        Tuple[batch, coords, atoms_channel]，数据批次、坐标和原子类型
        batch: 只包含目标样本的单个 batch（不再是整个 batch）
        coords: 目标样本的坐标 [1, n_atoms, 3]
        atoms_channel: 目标样本的原子类型 [1, n_atoms]
    """
    
    gnf_converter = create_gnf_converter(config)
    loader_val = create_field_loaders(config, gnf_converter, split="val")
    
    # 计算需要跳过多少个batch才能到达目标样本
    batch_size = config.dset.batch_size
    target_batch_idx = sample_idx // batch_size
    sample_in_batch = sample_idx % batch_size
    
    # 跳过前面的batch
    for i, batch in enumerate(loader_val):
        if i == target_batch_idx:
            batch = batch.to(device)
            
            # 提取目标样本在 batch 中的索引范围
            batch_idx = batch.batch  # [N_total_atoms] 每个原子属于哪个样本
            target_sample_mask = batch_idx == sample_in_batch
            
            # 提取目标样本的数据
            target_pos = batch.pos[target_sample_mask]  # [n_atoms, 3]
            target_x = batch.x[target_sample_mask]  # [n_atoms]
            
            # 提取其他属性（如果存在）
            target_data = {}
            if hasattr(batch, 'xs'):
                # xs 是 [N_total_points, 3]，需要根据样本索引提取
                # 注意：xs 可能不是按样本组织的，需要检查其结构
                # 如果 xs 是按样本组织的，需要相应处理
                # 这里先假设 xs 是共享的或者需要特殊处理
                if batch.xs.shape[0] == batch.pos.shape[0]:
                    # 如果 xs 和 pos 长度相同，说明是按原子组织的
                    target_data['xs'] = batch.xs[target_sample_mask]
                else:
                    # 如果长度不同，可能是按查询点组织的，需要保留全部或按样本分割
                    # 这里先保留全部，后续可能需要根据实际使用情况调整
                    target_data['xs'] = batch.xs
            
            if hasattr(batch, 'target_field'):
                # target_field 的处理类似 xs
                if batch.target_field.shape[0] == batch.pos.shape[0]:
                    target_data['target_field'] = batch.target_field[target_sample_mask]
                else:
                    target_data['target_field'] = batch.target_field
            
            if hasattr(batch, 'idx'):
                # idx 是 [batch_size]，提取目标样本的索引
                target_data['idx'] = batch.idx[sample_in_batch:sample_in_batch+1] if batch.idx is not None else None
            
            # 创建只包含目标样本的 Data 对象
            single_data = Data(
                pos=target_pos,
                x=target_x,
                batch=torch.zeros(target_pos.shape[0], dtype=torch.long, device=device),  # 所有原子都属于样本0
                **target_data
            )
            
            # 创建只包含单个样本的 Batch
            single_batch = Batch.from_data_list([single_data])
            
            # 提取坐标和原子类型（用于兼容性）
            coords, _ = to_dense_batch(single_batch.pos, single_batch.batch, fill_value=0)
            atoms_channel, _ = to_dense_batch(single_batch.x, single_batch.batch, fill_value=PADDING_INDEX)
            
            return single_batch, coords, atoms_channel
    
    raise IndexError(f"Sample index {sample_idx} is out of bounds for validation set") 