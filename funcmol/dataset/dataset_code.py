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

        # 检查是否存在LMDB数据库
        lmdb_path = os.path.join(self.codes_dir, "codes.lmdb")
        keys_path = os.path.join(self.codes_dir, "codes_keys.pt")
        
        if os.path.exists(lmdb_path) and os.path.exists(keys_path):
            # 使用LMDB数据库
            self._use_lmdb_database(lmdb_path, keys_path)
        else:
            # 使用传统方式加载
            # get list of codes
            self.list_codes = [
                f for f in os.listdir(self.codes_dir)
                if os.path.isfile(os.path.join(self.codes_dir, f)) and \
                f.startswith("codes") and f.endswith(".pt")
            ]
            
            # 优先使用 codes.pt（向后兼容）
            if "codes.pt" in self.list_codes:
                self.list_codes = ["codes.pt"]
                self.num_augmentations = 0
            else:
                # 查找所有 codes_XXX.pt 文件（新格式）
                numbered_codes = [f for f in self.list_codes if f.startswith("codes_") and f.endswith(".pt")]
                if numbered_codes:
                    # 按编号排序
                    numbered_codes.sort()
                    self.list_codes = numbered_codes
                    self.num_augmentations = len(numbered_codes)
                else:
                    # 兼容旧格式：如果有其他codes文件，使用第一个
                    self.list_codes.sort()
                    if self.list_codes:
                        self.list_codes = [self.list_codes[0]]
                    else:
                        raise FileNotFoundError(f"No codes files found in {self.codes_dir}")
                    self.num_augmentations = 0
            
            self.use_lmdb = False
            self.load_codes()

    def _use_lmdb_database(self, lmdb_path, keys_path):
        """使用LMDB数据库加载数据"""
        self.lmdb_path = lmdb_path
        self.keys = torch.load(keys_path, weights_only=False)  # 直接加载keys文件
        self.db = None
        self.use_lmdb = True
        
        print(f"  | Using LMDB database: {lmdb_path}")
        print(f"  | Database contains {len(self.keys)} codes")
    
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

    def __len__(self):
        if hasattr(self, 'use_lmdb') and self.use_lmdb:
            return len(self.keys)
        return self.curr_codes.shape[0]

    def __getitem__(self, index):
        if hasattr(self, 'use_lmdb') and self.use_lmdb:
            return self._getitem_lmdb(index)
        else:
            return self.curr_codes[index]
    
    def _getitem_lmdb(self, index):
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
                code_raw = pickle.loads(txn.get(key))
        except Exception as e:
            # 如果事务失败，重新连接数据库
            print(f"LMDB transaction failed, reconnecting: {e}")
            self._close_db()
            self._connect_db()
            with self.db.begin() as txn:
                code_raw = pickle.loads(txn.get(key))
        
        return code_raw
    
    def _close_db(self):
        """关闭数据库连接"""
        if self.db is not None:
            self.db = None
            # 注意：不关闭共享连接，让其他worker继续使用

    def load_codes(self, index=None) -> None:
        """
        Load codes from the available codes files.

        Args:
            index (int, optional): The index of the code to load. If None, loads all files and concatenates them.

        Returns:
            None

        Side Effects:
            - Sets `self.curr_codes` to the loaded codes from the codes file(s).
            - Prints the path(s) of the loaded codes.
        """
        if len(self.list_codes) == 1:
            # 只有一个文件，直接加载
            code_path = os.path.join(self.codes_dir, self.list_codes[0])
            print(">> loading codes: ", code_path)
            self.curr_codes = torch.load(code_path, weights_only=False)
        else:
            # 多个文件，合并加载
            all_codes = []
            for code_file in self.list_codes:
                code_path = os.path.join(self.codes_dir, code_file)
                print(">> loading codes: ", code_path)
                codes = torch.load(code_path, weights_only=False)
                all_codes.append(codes)
            # 合并所有codes
            self.curr_codes = torch.cat(all_codes, dim=0)
            print(f">> merged {len(self.list_codes)} codes files, total shape: {self.curr_codes.shape}")


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
        num_augmentations=config["num_augmentations"],
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
