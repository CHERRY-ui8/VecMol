import os
import random
import pickle
import threading

import torch
import lmdb
from torch.utils.data import Dataset, Subset

# 进程本地存储，用于在多进程环境下管理LMDB连接
_thread_local = threading.local()


class CodeDataset(Dataset):
    """
    A dataset class for handling code files in a specified directory.

    Attributes:
        dset_name (str): The name of the dataset. Default is "qm9".
        split (str): The data split to use (e.g., "train", "test"). Default is "train".
        codes_dir (str): The directory where the code files are stored.
        num_augmentations (int): The number of augmentations to use. If None and split is "train", it defaults to the number of code files minus one.

    Methods:
        __len__(): Returns the number of samples in the current codes.
        __getitem__(index): Returns the code at the specified index.
        load_codes(index=None): Loads the codes from the specified index or a random index if None.
    """
    def __init__(
        self,
        dset_name: str = "qm9",
        split: str = "train",
        codes_dir: str = None,
        num_augmentations = None,  # 保留参数以兼容现有代码，但不再使用
    ):
        self.dset_name = dset_name
        self.split = split
        self.codes_dir = os.path.join(codes_dir, self.split)
        
        # 检查是否存在position_weights文件
        self.position_weights = None
        self.use_position_weights = False

        # 检查是否存在LMDB数据库
        lmdb_path = os.path.join(self.codes_dir, "codes.lmdb")
        keys_path = os.path.join(self.codes_dir, "codes_keys.pt")
        
        if os.path.exists(lmdb_path) and os.path.exists(keys_path):
            # 使用LMDB数据库
            self._use_lmdb_database(lmdb_path, keys_path)
        else:
            # 检查是否有多个 codes 文件
            list_codes = [
                f for f in os.listdir(self.codes_dir)
                if os.path.isfile(os.path.join(self.codes_dir, f)) and \
                f.startswith("codes") and f.endswith(".pt")
            ]
            
            # 如果有多个文件，要求使用 LMDB 格式
            if "codes.pt" in list_codes:
                # 单个 codes.pt 文件，向后兼容
                self.list_codes = ["codes.pt"]
                self.num_augmentations = 0
                self.use_lmdb = False
                self.load_codes()
                self._load_position_weights_files()
            else:
                # 查找所有 codes_XXX.pt 文件
                numbered_codes = [f for f in list_codes if f.startswith("codes_") and f.endswith(".pt")]
                if numbered_codes and len(numbered_codes) > 1:
                    # 多个文件，要求转换为 LMDB
                    raise RuntimeError(
                        f"Found {len(numbered_codes)} codes files. Multiple codes files require LMDB format.\n"
                        f"Please convert to LMDB format first:\n"
                        f"  python funcmol/dataset/convert_codes_to_lmdb.py --codes_dir {os.path.dirname(self.codes_dir)} --splits {self.split}\n"
                        f"Or ensure codes.lmdb and codes_keys.pt exist in: {self.codes_dir}"
                    )
                elif numbered_codes and len(numbered_codes) == 1:
                    # 单个 codes_XXX.pt 文件，向后兼容
                    self.list_codes = numbered_codes
                    self.num_augmentations = 0
                    self.use_lmdb = False
                    self.load_codes()
                    self._load_position_weights_files()
                else:
                    # 没有找到 codes 文件
                    raise FileNotFoundError(
                        f"No codes files found in {self.codes_dir}.\n"
                        f"Expected either:\n"
                        f"  - codes.lmdb and codes_keys.pt (LMDB format), or\n"
                        f"  - codes.pt or codes_XXX.pt (single file format)"
                    )

    def _use_lmdb_database(self, lmdb_path, keys_path):
        """使用LMDB数据库加载数据"""
        self.lmdb_path = lmdb_path
        self.keys = torch.load(keys_path, weights_only=False)  # 直接加载keys文件
        self.db = None
        self.use_lmdb = True
        
        print(f"  | Using LMDB database: {lmdb_path}")
        print(f"  | Database contains {len(self.keys)} codes")
        
        # 检查是否存在position_weights文件（LMDB格式）
        weights_lmdb_path = os.path.join(os.path.dirname(lmdb_path), "position_weights.lmdb")
        weights_keys_path = os.path.join(os.path.dirname(lmdb_path), "position_weights_keys.pt")
        if os.path.exists(weights_lmdb_path) and os.path.exists(weights_keys_path):
            self.position_weights_lmdb_path = weights_lmdb_path
            self.position_weights_keys_path = weights_keys_path
            self.position_weights_keys = torch.load(weights_keys_path, weights_only=False)
            self.position_weights_db = None
            self.use_position_weights = True
            print(f"  | Found position_weights LMDB database: {weights_lmdb_path}")
            print(f"  | Position weights database contains {len(self.position_weights_keys)} entries")
        else:
            # 检查是否存在position_weights文件（文件格式）
            self._load_position_weights_files()
    
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
                    readahead=True,  # 启用readahead以提高读取性能
                    meminit=False,
                    max_readers=256,  # 增加最大读取器数量，支持更多并发worker
                )
            
            self.db = _thread_local.lmdb_connections[self.lmdb_path]

    def __len__(self):
        if hasattr(self, 'use_lmdb') and self.use_lmdb:
            return len(self.keys)
        return self.curr_codes.shape[0]

    def __getitem__(self, index):
        if hasattr(self, 'use_lmdb') and self.use_lmdb:
            code = self._getitem_lmdb(index)
        else:
            code = self.curr_codes[index]
        
        # 如果存在position_weights，一起返回
        if self.use_position_weights:
            if hasattr(self, 'use_lmdb') and self.use_lmdb:
                position_weight = self._get_position_weight_lmdb(index)
            else:
                if hasattr(self, 'position_weights') and self.position_weights is not None:
                    position_weight = self.position_weights[index]
                else:
                    position_weight = None
            if position_weight is not None:
                return code, position_weight
            else:
                # 如果position_weight为None，只返回code（向后兼容）
                return code
        else:
            return code
    
    def _getitem_lmdb(self, index):
        """LMDB模式下的数据获取 - 进程安全版本（优化版）"""
        # 确保数据库连接在worker进程中建立
        if self.db is None:
            self._connect_db()
        
        key = self.keys[index]
        # 确保key是bytes格式，因为LMDB需要bytes
        if isinstance(key, str):
            key = key.encode('utf-8')
        
        # 优化：使用buffers=True避免额外的内存拷贝
        # 使用更安全的事务处理
        try:
            with self.db.begin(buffers=True) as txn:
                value = txn.get(key)
                if value is None:
                    raise KeyError(f"Key not found: {key}")
                # 直接反序列化，避免额外的内存拷贝
                code_raw = pickle.loads(value)
        except Exception as e:
            # 如果事务失败，重新连接数据库
            print(f"LMDB transaction failed, reconnecting: {e}")
            self._close_db()
            self._connect_db()
            with self.db.begin(buffers=True) as txn:
                value = txn.get(key)
                if value is None:
                    raise KeyError(f"Key not found: {key}")
                code_raw = pickle.loads(value)
        
        return code_raw
    
    def _close_db(self):
        """关闭数据库连接"""
        if self.db is not None:
            self.db = None
            # 注意：不关闭共享连接，让其他worker继续使用

    def _load_position_weights_files(self):
        """加载position_weights文件（文件格式，非LMDB）"""
        # 查找position_weights文件（支持多个augmentation版本）
        list_weights = [
            f for f in os.listdir(self.codes_dir)
            if os.path.isfile(os.path.join(self.codes_dir, f)) and \
            f.startswith("position_weights") and f.endswith(".pt")
        ]
        
        if list_weights:
            # 如果有多个position_weights文件（对应多个augmentation版本），需要合并
            # 按照augmentation顺序合并：position_weights_000.pt, position_weights_001.pt, ...
            numbered_weights = [f for f in list_weights if f.startswith("position_weights_") and f.endswith(".pt")]
            if numbered_weights:
                numbered_weights.sort()  # 按编号排序
                print(f"  | Found {len(numbered_weights)} position_weights files (augmentation versions)")
                
                # 合并所有augmentation版本的position_weights
                all_weights = []
                for weights_file in numbered_weights:
                    weights_path = os.path.join(self.codes_dir, weights_file)
                    weights = torch.load(weights_path, weights_only=False)
                    all_weights.append(weights)
                    print(f"  |   - {weights_file}: shape {weights.shape}")
                
                # 合并所有augmentation版本
                self.position_weights = torch.cat(all_weights, dim=0)
                self.use_position_weights = True
                print(f"  | Merged position_weights shape: {self.position_weights.shape}")
                
                # 验证长度是否匹配
                if hasattr(self, 'curr_codes'):
                    if len(self.position_weights) != len(self.curr_codes):
                        print(f"  | WARNING: Position weights length ({len(self.position_weights)}) != codes length ({len(self.curr_codes)})")
                        self.use_position_weights = False
            elif len(list_weights) == 1:
                # 单个position_weights文件（向后兼容）
                weights_path = os.path.join(self.codes_dir, list_weights[0])
                print(f"  | Loading position_weights from: {weights_path}")
                self.position_weights = torch.load(weights_path, weights_only=False)
                self.use_position_weights = True
                print(f"  | Position weights shape: {self.position_weights.shape}")
                
                # 验证长度是否匹配
                if hasattr(self, 'curr_codes'):
                    if len(self.position_weights) != len(self.curr_codes):
                        print(f"  | WARNING: Position weights length ({len(self.position_weights)}) != codes length ({len(self.curr_codes)})")
                        self.use_position_weights = False
            else:
                print(f"  | WARNING: Found {len(list_weights)} position_weights files, expected 1 or numbered files. Disabling position_weights.")
        else:
            print(f"  | No position_weights files found in {self.codes_dir}")
    
    def _get_position_weight_lmdb(self, index):
        """从LMDB获取position_weight"""
        if not self.use_position_weights:
            return None
        
        # 确保数据库连接在worker进程中建立
        if not hasattr(self, 'position_weights_db') or self.position_weights_db is None:
            self._connect_position_weights_db()
        
        key = self.position_weights_keys[index]
        if isinstance(key, str):
            key = key.encode('utf-8')
        
        try:
            with self.position_weights_db.begin(buffers=True) as txn:
                value = txn.get(key)
                if value is None:
                    return None
                weight = pickle.loads(value)
        except Exception as e:
            print(f"LMDB position_weights transaction failed: {e}")
            return None
        
        return weight
    
    def _connect_position_weights_db(self):
        """建立position_weights LMDB数据库连接"""
        if not hasattr(self, 'position_weights_lmdb_path'):
            return
        
        if not hasattr(_thread_local, 'lmdb_connections'):
            _thread_local.lmdb_connections = {}
        
        if self.position_weights_lmdb_path not in _thread_local.lmdb_connections:
            _thread_local.lmdb_connections[self.position_weights_lmdb_path] = lmdb.open(
                self.position_weights_lmdb_path,
                map_size=10*(1024*1024*1024),
                create=False,
                subdir=True,
                readonly=True,
                lock=False,
                readahead=True,
                meminit=False,
                max_readers=256,
            )
        
        self.position_weights_db = _thread_local.lmdb_connections[self.position_weights_lmdb_path]

    def load_codes(self, index=None) -> None:
        """
        Load codes from a single code file.

        Args:
            index (int, optional): Not used, kept for compatibility.

        Returns:
            None

        Side Effects:
            - Sets `self.curr_codes` to the loaded codes from the code file.
            - Prints the path of the loaded codes.
        """
        if len(self.list_codes) != 1:
            raise RuntimeError(f"load_codes expects exactly 1 file, got {len(self.list_codes)}")
        
        code_path = os.path.join(self.codes_dir, self.list_codes[0])
        print(">> loading codes: ", code_path)
        self.curr_codes = torch.load(code_path, weights_only=False)


def create_code_loaders(
    config: dict,
    split: str = None,
    # fabric = None,
):
    """
    Creates and returns a DataLoader for the specified dataset split.

    Args:
        config (dict): Configuration dictionary containing dataset parameters.
            - dset (dict): Dictionary with dataset-specific parameters.
                - dset_name (str): Name of the dataset.
            - codes_dir (str): Directory where the code files are stored.
            - num_augmentations (int): Number of augmentations to apply to the dataset.
            - debug (bool): Flag to indicate if debugging mode is enabled.
            - dset (dict): Dictionary with DataLoader parameters.
                - batch_size (int): Batch size for the DataLoader.
                - num_workers (int): Number of worker threads for data loading.
        split (str, optional): The dataset split to load (e.g., 'train', 'val', 'test'). Defaults to None.
        fabric: Fabric object for setting up the DataLoader.

    Returns:
        DataLoader: A PyTorch DataLoader object for the specified dataset split.
    """
    dset = CodeDataset(
        dset_name=config["dset"]["dset_name"],
        codes_dir=config["codes_dir"],
        split=split,
        num_augmentations=config.get("num_augmentations", None),
    )

    # reduce the dataset size for debugging
    if config["debug"] or split in ["val", "test"]:
        indexes = list(range(len(dset)))
        random.Random(0).shuffle(indexes)
        indexes = indexes[:5000]
        if len(dset) > len(indexes):
            dset = Subset(dset, indexes)  # Smaller training set for debugging

    loader = torch.utils.data.DataLoader(
        dset,
        batch_size=min(config["dset"]["batch_size"], len(dset)),
        num_workers=config["dset"]["num_workers"],
        shuffle=True if split == "train" else False,
        pin_memory=True,
        drop_last=True,
    )
    # fabric.print(f">> {split} set size: {len(dset)}")
    print(f">> {split} set size: {len(dset)}")

    return loader # fabric.setup_dataloaders(loader, use_distributed_sampler=(split == "train"))
