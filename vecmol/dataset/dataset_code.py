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
        num_augmentations = None,  # 数据增强数量，用于查找对应的LMDB文件
        use_sharded_lmdb: bool = None,  # 是否使用分片LMDB。None表示自动检测，True强制使用分片，False强制使用完整LMDB
        load_position_weights: bool = True,  # 是否加载position_weights文件
    ):
        self.dset_name = dset_name
        self.split = split
        self.codes_dir = os.path.join(codes_dir, self.split)
        self.num_augmentations = num_augmentations
        self.load_position_weights = load_position_weights  # 保存配置
        
        # 检查是否存在position_weights文件
        self.position_weights = None
        self.use_position_weights = False

        # 检查是否存在LMDB数据库
        # 只有 train split 使用数据增强格式（codes_aug{num}.lmdb）
        # val/test split 始终使用默认格式（codes.lmdb），不需要数据增强
        if num_augmentations is not None and self.split == "train":
            lmdb_path = os.path.join(self.codes_dir, f"codes_aug{num_augmentations}.lmdb")
            keys_path = os.path.join(self.codes_dir, f"codes_aug{num_augmentations}_keys.pt")
        else:
            # val/test split 或未指定 num_augmentations 时，使用默认格式
            lmdb_path = os.path.join(self.codes_dir, "codes.lmdb")
            keys_path = os.path.join(self.codes_dir, "codes_keys.pt")
        
        # 根据配置决定是否使用分片LMDB
        # use_sharded_lmdb: None=自动检测, True=强制使用分片, False=强制使用完整LMDB
        lmdb_loaded = False
        
        if use_sharded_lmdb is False:
            # 强制使用完整LMDB，跳过分片检测
            if os.path.exists(lmdb_path) and os.path.exists(keys_path):
                self._use_lmdb_database(lmdb_path, keys_path)
                lmdb_loaded = True
        else:
            # 检查是否存在分片LMDB（更高效）
            # 检查两个可能的位置：
            # 1. 直接在codes_dir下：codes_dir/shard_info.pt
            # 2. 在sharded子目录下：codes_dir/sharded/shard_info.pt
            shard_info_path = os.path.join(self.codes_dir, "shard_info.pt")
            shard_info_path_sharded = os.path.join(self.codes_dir, "sharded", "shard_info.pt")
            
            shard_found = False
            if os.path.exists(shard_info_path):
                # 使用分片LMDB（直接在codes_dir下）
                self._use_sharded_lmdb_database(self.codes_dir, num_augmentations)
                shard_found = True
                lmdb_loaded = True
            elif os.path.exists(shard_info_path_sharded):
                # 使用分片LMDB（在sharded子目录下）
                sharded_dir = os.path.join(self.codes_dir, "sharded")
                self._use_sharded_lmdb_database(sharded_dir, num_augmentations)
                shard_found = True
                lmdb_loaded = True
            
            # 如果没有找到分片LMDB，尝试使用完整LMDB
            if not shard_found:
                if use_sharded_lmdb is True:
                    # 强制使用分片但未找到
                    # 对于val/test split，允许回退到完整LMDB（因为数据量通常较小，不需要分片）
                    if self.split in ["val", "test"]:
                        if os.path.exists(lmdb_path) and os.path.exists(keys_path):
                            print(f"Warning: use_sharded_lmdb=True but no sharded LMDB found for {self.split} split. "
                                  f"Falling back to full LMDB: {lmdb_path}")
                            self._use_lmdb_database(lmdb_path, keys_path)
                            lmdb_loaded = True
                        else:
                            raise FileNotFoundError(
                                f"use_sharded_lmdb=True but no sharded LMDB found in {self.codes_dir}, "
                                f"and full LMDB also not found at {lmdb_path}.\n"
                                f"Expected shard_info.pt in either:\n"
                                f"  - {self.codes_dir}/shard_info.pt, or\n"
                                f"  - {self.codes_dir}/sharded/shard_info.pt"
                            )
                    else:
                        # train split必须使用分片，如果找不到则报错
                        raise FileNotFoundError(
                            f"use_sharded_lmdb=True but no sharded LMDB found in {self.codes_dir}.\n"
                            f"Expected shard_info.pt in either:\n"
                            f"  - {self.codes_dir}/shard_info.pt, or\n"
                            f"  - {self.codes_dir}/sharded/shard_info.pt"
                        )
                elif os.path.exists(lmdb_path) and os.path.exists(keys_path):
                    # 使用单个LMDB数据库
                    self._use_lmdb_database(lmdb_path, keys_path)
                    lmdb_loaded = True
        
        # 如果LMDB都没有加载成功，继续后续的文件检测逻辑
        if not lmdb_loaded:
            # 检查是否有多个 codes 文件
            # 查找 codes_aug{num}_XXX.pt 格式的文件
            # 排除 keys 文件（codes_keys.pt, codes_aug*_keys.pt 等）
            list_codes = [
                f for f in os.listdir(self.codes_dir)
                if os.path.isfile(os.path.join(self.codes_dir, f)) and \
                f.startswith("codes") and f.endswith(".pt") and \
                not f.endswith("_keys.pt")  # 排除 keys 文件
            ]
            
            # 如果提供了num_augmentations 且是 train split，查找对应格式的文件
            if num_augmentations is not None and self.split == "train":
                numbered_codes = [f for f in list_codes if f.startswith(f"codes_aug{num_augmentations}_") and f.endswith(".pt")]
            else:
                # 向后兼容：查找所有格式的codes文件
                numbered_codes_aug = [f for f in list_codes if f.startswith("codes_aug") and f.endswith(".pt")]
                numbered_codes_old = [f for f in list_codes if f.startswith("codes_") and f.endswith(".pt") and not f.startswith("codes_aug")]
                single_code_old = ["codes.pt"] if "codes.pt" in list_codes else []
                
                if numbered_codes_aug:
                    numbered_codes = numbered_codes_aug
                elif numbered_codes_old:
                    numbered_codes = numbered_codes_old
                elif single_code_old:
                    numbered_codes = single_code_old
                else:
                    numbered_codes = []
            
            # 如果有多个文件，要求使用 LMDB 格式
            if numbered_codes and len(numbered_codes) > 1:
                # 多个文件，要求转换为 LMDB
                # 尝试从文件名推断数据增强数量
                inferred_num_aug = None
                if numbered_codes[0].startswith("codes_aug"):
                    # 从文件名中提取数据增强数量：codes_aug{num}_{idx}.pt
                    try:
                        prefix = numbered_codes[0].split("_")[1]  # 获取 "aug{num}" 部分
                        inferred_num_aug = int(prefix.replace("aug", ""))
                    except:
                        pass
                
                if inferred_num_aug is not None:
                    raise RuntimeError(
                        f"Found {len(numbered_codes)} codes files. Multiple codes files require LMDB format.\n"
                        f"Please convert to LMDB format first:\n"
                        f"  python vecmol/dataset/convert_codes_to_lmdb.py --codes_dir {os.path.dirname(self.codes_dir)} --num_augmentations {inferred_num_aug} --splits {self.split}\n"
                        f"Or ensure codes_aug{inferred_num_aug}.lmdb and codes_aug{inferred_num_aug}_keys.pt exist in: {self.codes_dir}"
                    )
                else:
                    raise RuntimeError(
                        f"Found {len(numbered_codes)} codes files. Multiple codes files require LMDB format.\n"
                        f"Please convert to LMDB format first:\n"
                        f"  python vecmol/dataset/convert_codes_to_lmdb.py --codes_dir {os.path.dirname(self.codes_dir)} --num_augmentations <num> --splits {self.split}\n"
                        f"Or ensure codes_aug{{num}}.lmdb and codes_aug{{num}}_keys.pt exist in: {self.codes_dir}"
                    )
            elif numbered_codes and len(numbered_codes) == 1:
                # 单个 codes 文件，向后兼容
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
                    f"  - codes_aug{{num}}.lmdb and codes_aug{{num}}_keys.pt (LMDB format with augmentation), or\n"
                    f"  - codes.lmdb and codes_keys.pt (LMDB format without augmentation, backward compatible), or\n"
                    f"  - codes.pt, codes_XXX.pt, or codes_aug{{num}}_XXX.pt (single file format)"
                )

    def _use_sharded_lmdb_database(self, codes_dir, num_augmentations):
        """使用分片LMDB数据库加载数据（更高效，减少多进程竞争）"""
        shard_info_path = os.path.join(codes_dir, "shard_info.pt")
        shard_info = torch.load(shard_info_path, weights_only=False)
        
        self.num_shards = shard_info['num_shards']
        self.total_samples = shard_info['total_samples']
        self.samples_per_shard = shard_info['samples_per_shard']
        self.shard_keys = shard_info['shard_keys']
        
        # 构建分片路径和keys
        self.shard_lmdb_paths = []
        self.shard_keys_list = []
        
        for shard_id in range(self.num_shards):
            if num_augmentations is not None:
                shard_lmdb_path = os.path.join(codes_dir, f"codes_aug{num_augmentations}_shard{shard_id}.lmdb")
                shard_keys_path = os.path.join(codes_dir, f"codes_aug{num_augmentations}_shard{shard_id}_keys.pt")
            else:
                shard_lmdb_path = os.path.join(codes_dir, f"codes_shard{shard_id}.lmdb")
                shard_keys_path = os.path.join(codes_dir, f"codes_shard{shard_id}_keys.pt")
            
            if os.path.exists(shard_lmdb_path) and os.path.exists(shard_keys_path):
                self.shard_lmdb_paths.append(shard_lmdb_path)
                self.shard_keys_list.append(torch.load(shard_keys_path, weights_only=False))
            else:
                raise FileNotFoundError(f"Shard {shard_id} not found: {shard_lmdb_path}")
        
        # 构建全局keys（用于兼容性）
        self.keys = []
        for shard_keys in self.shard_keys_list:
            self.keys.extend(shard_keys)
        
        # 初始化分片数据库连接（延迟加载）
        self.shard_dbs = [None] * self.num_shards
        self.use_lmdb = True
        self.use_sharded = True
        
        # 设置兼容的lmdb_path属性（用于缓存路径等）
        # 使用第一个分片的路径作为代表，或者使用codes_dir
        if self.shard_lmdb_paths:
            self.lmdb_path = self.shard_lmdb_paths[0]  # 用于兼容性，指向第一个分片
        else:
            self.lmdb_path = codes_dir  # 回退到目录路径
        
        print(f"  | Using SHARDED LMDB database: {self.num_shards} shards")
        print(f"  | Total samples: {self.total_samples}")
        print(f"  | Samples per shard: ~{self.samples_per_shard}")
        for i, path in enumerate(self.shard_lmdb_paths):
            shard_size = os.path.getsize(os.path.join(path, "data.mdb")) / (1024**3) if os.path.exists(os.path.join(path, "data.mdb")) else 0
            print(f"  |   Shard {i}: {len(self.shard_keys_list[i])} samples, {shard_size:.1f} GB")
    
    def _use_lmdb_database(self, lmdb_path, keys_path):
        """使用LMDB数据库加载数据"""
        self.lmdb_path = lmdb_path
        self.keys = torch.load(keys_path, weights_only=False)  # 直接加载keys文件
        self.db = None
        self.use_lmdb = True
        self.use_sharded = False  # 标记不是分片模式
        
        print(f"  | Using LMDB database: {lmdb_path}")
        print(f"  | Database contains {len(self.keys)} codes")
        
        # 如果配置中禁用了position_weights，跳过加载
        if not self.load_position_weights:
            print(f"  | Position weights loading disabled, skipping")
            return
        
        # 检查是否存在position_weights文件（LMDB格式）
        # 只有 train split 使用数据增强格式（position_weights_aug{num}.lmdb）
        # val/test split 使用默认格式（position_weights.lmdb 或 position_weights_v2.lmdb）
        dirname = os.path.dirname(lmdb_path)
        
        # 如果提供了num_augmentations 且是 train split，使用新格式：position_weights_aug{num}.lmdb
        if self.num_augmentations is not None and self.split == "train":
            weights_lmdb_path = os.path.join(dirname, f"position_weights_aug{self.num_augmentations}.lmdb")
            weights_keys_path = os.path.join(dirname, f"position_weights_aug{self.num_augmentations}_keys.pt")
            
            if os.path.exists(weights_lmdb_path) and os.path.exists(weights_keys_path):
                self.position_weights_lmdb_path = weights_lmdb_path
                self.position_weights_keys_path = weights_keys_path
                self.position_weights_keys = torch.load(weights_keys_path, weights_only=False)
                self.position_weights_db = None
                self.use_position_weights = True
                print(f"  | Found position_weights_aug{self.num_augmentations} LMDB database: {weights_lmdb_path}")
                print(f"  | Position weights database contains {len(self.position_weights_keys)} entries")
            else:
                # 如果找不到对应版本，检查是否存在position_weights文件（文件格式）
                print(f"  | Position weights LMDB not found for aug{self.num_augmentations}, trying file format...")
                self._load_position_weights_files()
        else:
            # 向后兼容：尝试旧格式（position_weights_v2.lmdb 和 position_weights.lmdb）
            weights_lmdb_path_v2 = os.path.join(dirname, "position_weights_v2.lmdb")
            weights_keys_path_v2 = os.path.join(dirname, "position_weights_v2_keys.pt")
            weights_lmdb_path_old = os.path.join(dirname, "position_weights.lmdb")
            weights_keys_path_old = os.path.join(dirname, "position_weights_keys.pt")
            
            # 优先检查新格式
            if os.path.exists(weights_lmdb_path_v2) and os.path.exists(weights_keys_path_v2):
                self.position_weights_lmdb_path = weights_lmdb_path_v2
                self.position_weights_keys_path = weights_keys_path_v2
                self.position_weights_keys = torch.load(weights_keys_path_v2, weights_only=False)
                self.position_weights_db = None
                self.use_position_weights = True
                print(f"  | Found position_weights_v2 LMDB database: {weights_lmdb_path_v2}")
                print(f"  | Position weights database contains {len(self.position_weights_keys)} entries")
            elif os.path.exists(weights_lmdb_path_old) and os.path.exists(weights_keys_path_old):
                # 回退到旧格式
                self.position_weights_lmdb_path = weights_lmdb_path_old
                self.position_weights_keys_path = weights_keys_path_old
                self.position_weights_keys = torch.load(weights_keys_path_old, weights_only=False)
                self.position_weights_db = None
                self.use_position_weights = True
                print(f"  | Found position_weights LMDB database: {weights_lmdb_path_old}")
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
                # 动态计算map_size：至少是文件大小的1.2倍，最大10TB
                # 对于大型LMDB文件（如drugs数据集574GB），需要足够大的map_size
                # 注意：map_size必须>=文件大小，否则LMDB无法正确映射，会导致性能严重下降
                try:
                    file_size = os.path.getsize(self.lmdb_path) if os.path.isfile(self.lmdb_path) else 0
                    # 如果是目录，尝试获取目录大小（LMDB subdir=True时）
                    if os.path.isdir(self.lmdb_path):
                        # 估算：检查data.mdb文件大小
                        data_file = os.path.join(self.lmdb_path, "data.mdb")
                        if os.path.exists(data_file):
                            file_size = os.path.getsize(data_file)
                    
                    # map_size至少是文件大小的1.2倍，最大10TB
                    # 注意：map_size必须>=文件大小，否则LMDB无法正确映射
                    if file_size > 0:
                        map_size = int(file_size * 1.2)  # 文件大小的1.2倍
                        map_size = min(map_size, 10 * 1024**4)  # 最大10TB
                    else:
                        # 如果无法获取文件大小，使用较大的默认值（针对大型数据集）
                        map_size = 1024 * (1024**3)  # 1TB，足够大多数情况
                except Exception:
                    # 如果出错，使用较大的默认值
                    map_size = 1024 * (1024**3)  # 1TB
                
                _thread_local.lmdb_connections[self.lmdb_path] = lmdb.open(
                    self.lmdb_path,
                    map_size=map_size,
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
        # 处理 curr_codes 可能是 list 或数组的情况
        if hasattr(self.curr_codes, 'shape'):
            return self.curr_codes.shape[0]
        else:
            return len(self.curr_codes)

    def __getitem__(self, index):
        if hasattr(self, 'use_lmdb') and self.use_lmdb:
            if hasattr(self, 'use_sharded') and self.use_sharded:
                code = self._getitem_sharded_lmdb(index)
            else:
                code = self._getitem_lmdb(index)
        else:
            code = self.curr_codes[index]
        
        # 确保 code 是 tensor，如果不是则转换或报错
        if not isinstance(code, torch.Tensor):
            if isinstance(code, str):
                # 如果 code 是字符串，可能是 key 被错误返回了，或者是数据损坏
                raise TypeError(
                    f"Expected tensor but got string at index {index}. "
                    f"This might indicate:\n"
                    f"  1. Data corruption in LMDB (stored value is string instead of tensor)\n"
                    f"  2. Key was returned instead of value\n"
                    f"  3. Wrong data format in codes file\n"
                    f"Code value (first 200 chars): {code[:200] if len(code) > 200 else code}\n"
                    f"Code type: {type(code)}"
                )
            # 尝试转换为 tensor（适用于 numpy array 等情况）
            try:
                if hasattr(code, '__array__'):
                    # 如果是 numpy array 或其他可转换为 tensor 的类型
                    code = torch.from_numpy(code.__array__()) if hasattr(code, '__array__') else torch.tensor(code)
                else:
                    code = torch.tensor(code)
            except (TypeError, ValueError, RuntimeError) as e:
                raise TypeError(
                    f"Cannot convert code at index {index} to tensor.\n"
                    f"  Type: {type(code)}\n"
                    f"  Value preview: {str(code)[:200] if not isinstance(code, (list, dict, torch.Tensor)) else 'complex object'}\n"
                    f"  Original error: {e}"
                ) from e
        
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
    
    def _getitem_sharded_lmdb(self, index):
        """从分片LMDB中获取数据"""
        # 确定数据在哪个分片
        shard_id = index // self.samples_per_shard
        shard_index = index % self.samples_per_shard
        
        # 确保分片ID有效
        if shard_id >= self.num_shards:
            raise IndexError(f"Index {index} out of range for {self.total_samples} samples")
        
        # 获取该分片的数据库连接
        if self.shard_dbs[shard_id] is None:
            self._connect_shard_db(shard_id)
        
        # 获取该分片的key
        key = self.shard_keys_list[shard_id][shard_index]
        if isinstance(key, str):
            key = key.encode('utf-8')
        
        # 从分片数据库读取
        try:
            with self.shard_dbs[shard_id].begin(buffers=True) as txn:
                value = txn.get(key)
                if value is None:
                    raise KeyError(f"Key not found in shard {shard_id}: {key}")
                code_raw = pickle.loads(value)
        except Exception as e:
            print(f"LMDB transaction failed for shard {shard_id}, reconnecting: {e}")
            self._connect_shard_db(shard_id, force_reconnect=True)
            with self.shard_dbs[shard_id].begin(buffers=True) as txn:
                value = txn.get(key)
                if value is None:
                    raise KeyError(f"Key not found in shard {shard_id}: {key}")
                code_raw = pickle.loads(value)
        
        return code_raw
    
    def _connect_shard_db(self, shard_id, force_reconnect=False):
        """建立分片数据库连接"""
        if self.shard_dbs[shard_id] is not None and not force_reconnect:
            return
        
        shard_lmdb_path = self.shard_lmdb_paths[shard_id]
        
        # 使用线程本地存储
        if not hasattr(_thread_local, 'lmdb_connections'):
            _thread_local.lmdb_connections = {}
        
        if shard_lmdb_path not in _thread_local.lmdb_connections or force_reconnect:
            # 计算map_size
            try:
                data_file = os.path.join(shard_lmdb_path, "data.mdb")
                if os.path.exists(data_file):
                    file_size = os.path.getsize(data_file)
                    map_size = int(file_size * 1.2)
                    map_size = min(map_size, 10 * 1024**4)  # 最大10TB
                else:
                    map_size = 100 * (1024**3)  # 默认100GB
            except Exception:
                map_size = 100 * (1024**3)  # 默认100GB
            
            _thread_local.lmdb_connections[shard_lmdb_path] = lmdb.open(
                shard_lmdb_path,
                map_size=map_size,
                create=False,
                subdir=True,
                readonly=True,
                lock=False,
                readahead=True,
                meminit=False,
                max_readers=256,
            )
        
        self.shard_dbs[shard_id] = _thread_local.lmdb_connections[shard_lmdb_path]
    
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
        # 排除 _keys.pt 文件（这些是索引文件，不是权重文件）
        list_weights = [
            f for f in os.listdir(self.codes_dir)
            if os.path.isfile(os.path.join(self.codes_dir, f)) and \
            f.startswith("position_weights") and f.endswith(".pt") and \
            not f.endswith("_keys.pt")
        ]
        
        if list_weights:
            # 如果有多个position_weights文件（对应多个augmentation版本），需要合并
            # NOTE：现在使用新格式：position_weights_v2_000.pt, position_weights_v2_001.pt, ...
            # 如果没有新格式，则使用旧格式：position_weights_000.pt, position_weights_001.pt, ...
            numbered_weights_v2 = [f for f in list_weights if f.startswith("position_weights_v2_") and f.endswith(".pt")]
            numbered_weights_old = [f for f in list_weights if f.startswith("position_weights_") and f.endswith(".pt") and not f.startswith("position_weights_v2_")]
            
            # 优先使用新格式（v2），如果没有则使用旧格式
            numbered_weights = numbered_weights_v2 if numbered_weights_v2 else numbered_weights_old
            
            if numbered_weights:
                numbered_weights.sort()  # 按编号排序
                print(f"  | Found {len(numbered_weights)} position_weights files (augmentation versions)")
                
                # 合并所有augmentation版本的position_weights
                all_weights = []
                for weights_file in numbered_weights:
                    weights_path = os.path.join(self.codes_dir, weights_file)
                    weights = torch.load(weights_path, weights_only=False)
                    # 确保加载的是tensor，而不是list或其他类型
                    if not isinstance(weights, torch.Tensor):
                        print(f"  | WARNING: {weights_file} is not a tensor (type: {type(weights)}), skipping")
                        continue
                    all_weights.append(weights)
                    print(f"  |   - {weights_file}: shape {weights.shape}")
                
                # 合并所有augmentation版本
                if len(all_weights) > 0:
                    self.position_weights = torch.cat(all_weights, dim=0)
                    self.use_position_weights = True
                    print(f"  | Merged position_weights shape: {self.position_weights.shape}")
                else:
                    print(f"  | WARNING: No valid position_weights files found, skipping position weights")
                    self.use_position_weights = False
                
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
            # 动态计算map_size（与codes LMDB相同逻辑）
            import os
            try:
                file_size = os.path.getsize(self.position_weights_lmdb_path) if os.path.isfile(self.position_weights_lmdb_path) else 0
                if os.path.isdir(self.position_weights_lmdb_path):
                    data_file = os.path.join(self.position_weights_lmdb_path, "data.mdb")
                    if os.path.exists(data_file):
                        file_size = os.path.getsize(data_file)
                
                if file_size > 0:
                    map_size = int(file_size * 1.2)  # 文件大小的1.2倍
                    map_size = min(map_size, 10 * 1024**4)  # 最大10TB
                else:
                    map_size = 1024 * (1024**3)  # 1TB默认值
            except Exception:
                map_size = 1024 * (1024**3)  # 1TB默认值
            
            _thread_local.lmdb_connections[self.position_weights_lmdb_path] = lmdb.open(
                self.position_weights_lmdb_path,
                map_size=map_size,
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
        
        # 检查文件名，防止误加载 keys 文件
        filename = os.path.basename(code_path)
        if filename.endswith("_keys.pt") or "keys" in filename.lower():
            raise ValueError(
                f"ERROR: Attempted to load a KEYS file as CODES file!\n"
                f"  File: {code_path}\n"
                f"  Keys files contain string identifiers, not tensor data.\n"
                f"  Please check your file list and ensure codes files are correctly named.\n"
                f"  Expected: codes.pt, codes_*.pt, or codes_aug*_*.pt\n"
                f"  Got: {filename}"
            )
        
        loaded_data = torch.load(code_path, weights_only=False)
        
        # 检查加载的数据格式
        if isinstance(loaded_data, torch.Tensor):
            # 正常情况：tensor 格式
            self.curr_codes = loaded_data
            print(f"  | Loaded codes as tensor: shape {self.curr_codes.shape}")
        elif isinstance(loaded_data, list):
            # 可能是旧版本格式：list of tensors
            if len(loaded_data) > 0 and isinstance(loaded_data[0], torch.Tensor):
                # 尝试将 list of tensors 转换为单个 tensor
                try:
                    self.curr_codes = torch.stack(loaded_data) if len(loaded_data) > 0 else torch.tensor([])
                    print(f"  | Converted list of {len(loaded_data)} tensors to single tensor: shape {self.curr_codes.shape}")
                except Exception as e:
                    raise RuntimeError(
                        f"Cannot convert list of tensors to single tensor. "
                        f"This might indicate inconsistent tensor shapes in the list. "
                        f"Error: {e}\n"
                        f"Please check the codes file format or regenerate codes files."
                    ) from e
            else:
                # 检查是否是 keys 文件被错误加载
                first_elem_type = type(loaded_data[0]) if len(loaded_data) > 0 else 'empty list'
                is_keys_file = (isinstance(loaded_data, list) and 
                               len(loaded_data) > 0 and 
                               isinstance(loaded_data[0], str) and
                               all(isinstance(x, str) for x in loaded_data[:10]))  # 检查前10个元素
                
                # 构建详细的错误信息
                error_msg = f"\n{'='*80}\n"
                error_msg += f"ERROR: Loaded file contains list of strings, not tensors!\n"
                error_msg += f"{'='*80}\n"
                error_msg += f"File path: {code_path}\n"
                error_msg += f"File size: {os.path.getsize(code_path) / (1024*1024):.2f} MB\n"
                error_msg += f"List length: {len(loaded_data)}\n"
                error_msg += f"Element type: {first_elem_type}\n"
                
                if len(loaded_data) > 0:
                    error_msg += f"First few elements: {loaded_data[:5]}\n"
                
                if is_keys_file:
                    error_msg += f"\n⚠️  DIAGNOSIS: This looks like a KEYS file, not a CODES file!\n"
                    error_msg += f"   Keys files contain string identifiers (e.g., ['0', '1', '2', ...])\n"
                    error_msg += f"   Codes files should contain tensors.\n\n"
                    error_msg += f"   Possible causes:\n"
                    error_msg += f"   1. Wrong file was loaded (keys file instead of codes file)\n"
                    error_msg += f"   2. File naming issue (codes file was saved with wrong name)\n"
                    error_msg += f"   3. File corruption or wrong format\n\n"
                    error_msg += f"   Expected codes file format:\n"
                    error_msg += f"     - Single tensor: torch.Tensor with shape [N, ...]\n"
                    error_msg += f"     - Or list of tensors: [torch.Tensor, torch.Tensor, ...]\n\n"
                    error_msg += f"   Check if you have:\n"
                    error_msg += f"     - codes.pt or codes_*.pt (should contain tensors)\n"
                    error_msg += f"     - codes_keys.pt (contains strings, this is what was loaded)\n"
                else:
                    error_msg += f"\n⚠️  This file contains unexpected data format.\n"
                    error_msg += f"   Expected: torch.Tensor or list of torch.Tensor\n"
                    error_msg += f"   Got: list of {first_elem_type}\n"
                
                error_msg += f"{'='*80}\n"
                raise TypeError(error_msg)
        else:
            raise TypeError(
                f"Unexpected codes format. Expected torch.Tensor or list of torch.Tensor, "
                f"got: {type(loaded_data)}\n"
                f"This might indicate:\n"
                f"  1. Codes file was saved in wrong format\n"
                f"  2. File corruption\n"
                f"  3. Wrong file was loaded\n"
                f"Please check the codes file or regenerate it."
            )


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
    # 获取数据增强数量
    # 只有 train split 使用数据增强，val/test split 不使用数据增强
    # 因此 val/test split 的 num_augmentations 始终为 None，使用默认的 codes.lmdb 格式
    if split == "train":
        # train split：优先使用配置中明确指定的 num_augmentations
        num_augmentations = config.get("num_augmentations", None)
        
        # 如果未提供num_augmentations，尝试从codes目录的文件名推断
        if num_augmentations is None and config.get("codes_dir") is not None:
            split_dir = os.path.join(config["codes_dir"], split)
            if os.path.exists(split_dir):
                # 首先尝试查找 LMDB 文件：codes_aug{num}.lmdb
                list_lmdb = [
                    f for f in os.listdir(split_dir)
                    if os.path.isfile(os.path.join(split_dir, f)) and \
                    f.startswith("codes_aug") and f.endswith(".lmdb")
                ]
                if list_lmdb:
                    # 从LMDB文件名中提取数据增强数量：codes_aug{num}.lmdb
                    try:
                        first_lmdb = sorted(list_lmdb)[0]
                        # 提取 "aug{num}" 部分
                        parts = first_lmdb.replace(".lmdb", "").split("_")
                        if len(parts) >= 2 and parts[1].startswith("aug"):
                            num_augmentations = int(parts[1].replace("aug", ""))
                            print(f">> Inferred num_augmentations={num_augmentations} from LMDB file: {first_lmdb}")
                    except Exception:
                        pass
                
                # 如果还没找到，尝试查找 codes_aug{num}_*.pt 格式的文件
                if num_augmentations is None:
                    list_codes = [
                        f for f in os.listdir(split_dir)
                        if os.path.isfile(os.path.join(split_dir, f)) and \
                        f.startswith("codes_aug") and f.endswith(".pt")
                    ]
                    if list_codes:
                        # 从第一个文件名中提取数据增强数量：codes_aug{num}_{idx}.pt
                        try:
                            first_file = sorted(list_codes)[0]
                            # 提取 "aug{num}" 部分
                            parts = first_file.split("_")
                            if len(parts) >= 2 and parts[1].startswith("aug"):
                                num_augmentations = int(parts[1].replace("aug", ""))
                                print(f">> Inferred num_augmentations={num_augmentations} from codes files in {split_dir}")
                        except Exception:
                            pass
    else:
        # val/test split：不使用数据增强，num_augmentations 为 None
        num_augmentations = None
        print(f">> {split} split: using default codes.lmdb format (no augmentation)")
    
    # 从配置中读取是否使用分片LMDB（可选参数）
    # 对于 train split：使用配置中的值（如果配置了 True，就使用 True）
    # 对于 val/test split：强制不使用分片LMDB（设置为 None 或 False），使用完整的 LMDB
    use_sharded_lmdb_config = config.get("use_sharded_lmdb", None)
    if split == "train":
        use_sharded_lmdb = use_sharded_lmdb_config
    else:
        # val/test split 不使用分片LMDB，设置为 None 以自动回退到完整LMDB
        use_sharded_lmdb = None
        if use_sharded_lmdb_config is True:
            print(f">> {split} split: use_sharded_lmdb is set to None (auto-detect, will fallback to full LMDB)")
    
    # 检查是否应该加载 position_weights
    position_weight_config = config.get("position_weight", {})
    position_weight_enabled = position_weight_config.get("enabled", False)
    # 根据配置决定是否加载 position_weights
    load_position_weights = position_weight_enabled
    
    dset = CodeDataset(
        dset_name=config["dset"]["dset_name"],
        codes_dir=config["codes_dir"],
        split=split,
        num_augmentations=num_augmentations,
        use_sharded_lmdb=use_sharded_lmdb,
        load_position_weights=load_position_weights,
    )

    # 如果在配置中显式关闭了 use_augmented_codes 或 position_weight.enabled=False，则不使用预计算的 position_weights
    # 这样可以避免在没有对应 LMDB / 文件时仍然尝试访问 position_weights_keys 等属性
    if not config.get("use_augmented_codes", False) or not position_weight_enabled:
        if isinstance(dset, CodeDataset):
            dset.use_position_weights = False
            if not position_weight_enabled:
                print(f">> Position weights disabled in config, skipping position_weights loading")

    # reduce the dataset size for debugging
    if config["debug"] or split in ["val", "test"]:
        indexes = list(range(len(dset)))
        random.Random(0).shuffle(indexes)
        indexes = indexes[:5000]
        if len(dset) > len(indexes):
            dset = Subset(dset, indexes)  # Smaller training set for debugging

    # DataLoader配置：优化性能设置
    # num_workers从config文件读取，不在此处限制
    loader_kwargs = {
        "batch_size": min(config["dset"]["batch_size"], len(dset)),
        "num_workers": config["dset"]["num_workers"],
        "shuffle": True if split == "train" else False,
        "pin_memory": True,  # 加速GPU数据传输
        "drop_last": True,
        "persistent_workers": True if config["dset"]["num_workers"] > 0 else False,  # 保持worker进程，减少启动开销
        # 移除prefetch_factor限制，使用PyTorch默认值（通常为2），让系统自动优化
    }
    
    # 如果配置中显式指定了prefetch_factor，则使用配置值（允许用户覆盖）
    if "prefetch_factor" in config["dset"]:
        loader_kwargs["prefetch_factor"] = config["dset"]["prefetch_factor"]
    
    loader = torch.utils.data.DataLoader(dset, **loader_kwargs)
    # fabric.print(f">> {split} set size: {len(dset)}")
    print(f">> {split} set size: {len(dset)}")

    return loader # fabric.setup_dataloaders(loader, use_distributed_sampler=(split == "train"))
