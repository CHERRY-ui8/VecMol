import os
import random
import pickle
import threading

import torch
import lmdb
from torch.utils.data import Dataset, Subset

# Process-local storage for LMDB connections in multi-process environment
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
        num_augmentations = None,  # Number of augmentations; used to find corresponding LMDB
        use_sharded_lmdb: bool = None,  # Use sharded LMDB: None=auto, True=force sharded, False=force full LMDB
        load_position_weights: bool = True,  # Whether to load position_weights
    ):
        self.dset_name = dset_name
        self.split = split
        self.codes_dir = os.path.join(codes_dir, self.split)
        self.num_augmentations = num_augmentations
        self.load_position_weights = load_position_weights  # Store config

        # Check for position_weights file
        self.position_weights = None
        self.use_position_weights = False

        # Check for LMDB database
        # Only train split uses augmentation format (codes_aug{num}.lmdb); val/test use default (codes.lmdb)
        if num_augmentations is not None and self.split == "train":
            lmdb_path = os.path.join(self.codes_dir, f"codes_aug{num_augmentations}.lmdb")
            keys_path = os.path.join(self.codes_dir, f"codes_aug{num_augmentations}_keys.pt")
        else:
            # val/test or no num_augmentations: use default format
            lmdb_path = os.path.join(self.codes_dir, "codes.lmdb")
            keys_path = os.path.join(self.codes_dir, "codes_keys.pt")
        
        # Whether to use sharded LMDB: None=auto, True=force sharded, False=force full LMDB
        lmdb_loaded = False

        if use_sharded_lmdb is False:
            # Force full LMDB, skip shard detection
            if os.path.exists(lmdb_path) and os.path.exists(keys_path):
                self._use_lmdb_database(lmdb_path, keys_path)
                lmdb_loaded = True
        else:
            # Check for sharded LMDB (more efficient)
            # Two possible locations: codes_dir/shard_info.pt or codes_dir/sharded/shard_info.pt
            shard_info_path = os.path.join(self.codes_dir, "shard_info.pt")
            shard_info_path_sharded = os.path.join(self.codes_dir, "sharded", "shard_info.pt")
            
            shard_found = False
            if os.path.exists(shard_info_path):
                # Use sharded LMDB (directly under codes_dir)
                self._use_sharded_lmdb_database(self.codes_dir, num_augmentations)
                shard_found = True
                lmdb_loaded = True
            elif os.path.exists(shard_info_path_sharded):
                # Use sharded LMDB (in sharded subdir)
                sharded_dir = os.path.join(self.codes_dir, "sharded")
                self._use_sharded_lmdb_database(sharded_dir, num_augmentations)
                shard_found = True
                lmdb_loaded = True
            
            # If no sharded LMDB found, try full LMDB
            if not shard_found:
                if use_sharded_lmdb is True:
                    # Forced sharded but not found; for val/test allow fallback to full LMDB
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
                        # train split must use sharded; error if not found
                        raise FileNotFoundError(
                            f"use_sharded_lmdb=True but no sharded LMDB found in {self.codes_dir}.\n"
                            f"Expected shard_info.pt in either:\n"
                            f"  - {self.codes_dir}/shard_info.pt, or\n"
                            f"  - {self.codes_dir}/sharded/shard_info.pt"
                        )
                elif os.path.exists(lmdb_path) and os.path.exists(keys_path):
                    # Use single LMDB database
                    self._use_lmdb_database(lmdb_path, keys_path)
                    lmdb_loaded = True
        
        # If LMDB not loaded, continue with file-based detection
        if not lmdb_loaded:
            # Check for multiple codes files (codes_aug{num}_XXX.pt); exclude _keys.pt
            list_codes = [
                f for f in os.listdir(self.codes_dir)
                if os.path.isfile(os.path.join(self.codes_dir, f)) and \
                f.startswith("codes") and f.endswith(".pt") and \
                not f.endswith("_keys.pt")
            ]

            # If num_augmentations and train split, look for matching format
            if num_augmentations is not None and self.split == "train":
                numbered_codes = [f for f in list_codes if f.startswith(f"codes_aug{num_augmentations}_") and f.endswith(".pt")]
            else:
                # Backward compat: find all codes file formats
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
            
            # Multiple files require LMDB format
            if numbered_codes and len(numbered_codes) > 1:
                # Try to infer num_augmentations from filename
                inferred_num_aug = None
                if numbered_codes[0].startswith("codes_aug"):
                    try:
                        prefix = numbered_codes[0].split("_")[1]  # "aug{num}" part
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
                # Single codes file (backward compat)
                self.list_codes = numbered_codes
                self.num_augmentations = 0
                self.use_lmdb = False
                self.load_codes()
                self._load_position_weights_files()
            else:
                # No codes files found
                raise FileNotFoundError(
                    f"No codes files found in {self.codes_dir}.\n"
                    f"Expected either:\n"
                    f"  - codes_aug{{num}}.lmdb and codes_aug{{num}}_keys.pt (LMDB format with augmentation), or\n"
                    f"  - codes.lmdb and codes_keys.pt (LMDB format without augmentation, backward compatible), or\n"
                    f"  - codes.pt, codes_XXX.pt, or codes_aug{{num}}_XXX.pt (single file format)"
                )

    def _use_sharded_lmdb_database(self, codes_dir, num_augmentations):
        """Load data from sharded LMDB (more efficient, less multi-process contention)."""
        shard_info_path = os.path.join(codes_dir, "shard_info.pt")
        shard_info = torch.load(shard_info_path, weights_only=False)
        
        self.num_shards = shard_info['num_shards']
        self.total_samples = shard_info['total_samples']
        self.samples_per_shard = shard_info['samples_per_shard']
        self.shard_keys = shard_info['shard_keys']
        
        # Build shard paths and keys
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
        
        # Build global keys (for compatibility)
        self.keys = []
        for shard_keys in self.shard_keys_list:
            self.keys.extend(shard_keys)
        
        # Initialize shard DB connections (lazy)
        self.shard_dbs = [None] * self.num_shards
        self.use_lmdb = True
        self.use_sharded = True
        
        # Set compatible lmdb_path (for cache path etc.); use first shard or codes_dir
        if self.shard_lmdb_paths:
            self.lmdb_path = self.shard_lmdb_paths[0]  # Compatibility: first shard
        else:
            self.lmdb_path = codes_dir  # Fallback to directory
        
        print(f"  | Using SHARDED LMDB database: {self.num_shards} shards")
        print(f"  | Total samples: {self.total_samples}")
        print(f"  | Samples per shard: ~{self.samples_per_shard}")
        for i, path in enumerate(self.shard_lmdb_paths):
            shard_size = os.path.getsize(os.path.join(path, "data.mdb")) / (1024**3) if os.path.exists(os.path.join(path, "data.mdb")) else 0
            print(f"  |   Shard {i}: {len(self.shard_keys_list[i])} samples, {shard_size:.1f} GB")
    
    def _use_lmdb_database(self, lmdb_path, keys_path):
        """Load data from LMDB database."""
        self.lmdb_path = lmdb_path
        self.keys = torch.load(keys_path, weights_only=False)  # Load keys file
        self.db = None
        self.use_lmdb = True
        self.use_sharded = False  # Not sharded mode
        
        print(f"  | Using LMDB database: {lmdb_path}")
        print(f"  | Database contains {len(self.keys)} codes")
        
        # Skip loading if position_weights disabled in config
        if not self.load_position_weights:
            print(f"  | Position weights loading disabled, skipping")
            return
        
        # Check for position_weights (LMDB): train uses position_weights_aug{num}.lmdb; val/test use default
        dirname = os.path.dirname(lmdb_path)

        # If num_augmentations and train split, use position_weights_aug{num}.lmdb
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
                # If no matching version, try file format
                print(f"  | Position weights LMDB not found for aug{self.num_augmentations}, trying file format...")
                self._load_position_weights_files()
        else:
            # Backward compat: try old format (position_weights_v2.lmdb, position_weights.lmdb)
            weights_lmdb_path_v2 = os.path.join(dirname, "position_weights_v2.lmdb")
            weights_keys_path_v2 = os.path.join(dirname, "position_weights_v2_keys.pt")
            weights_lmdb_path_old = os.path.join(dirname, "position_weights.lmdb")
            weights_keys_path_old = os.path.join(dirname, "position_weights_keys.pt")
            
            # Prefer new format first
            if os.path.exists(weights_lmdb_path_v2) and os.path.exists(weights_keys_path_v2):
                self.position_weights_lmdb_path = weights_lmdb_path_v2
                self.position_weights_keys_path = weights_keys_path_v2
                self.position_weights_keys = torch.load(weights_keys_path_v2, weights_only=False)
                self.position_weights_db = None
                self.use_position_weights = True
                print(f"  | Found position_weights_v2 LMDB database: {weights_lmdb_path_v2}")
                print(f"  | Position weights database contains {len(self.position_weights_keys)} entries")
            elif os.path.exists(weights_lmdb_path_old) and os.path.exists(weights_keys_path_old):
                # Fallback to old format
                self.position_weights_lmdb_path = weights_lmdb_path_old
                self.position_weights_keys_path = weights_keys_path_old
                self.position_weights_keys = torch.load(weights_keys_path_old, weights_only=False)
                self.position_weights_db = None
                self.use_position_weights = True
                print(f"  | Found position_weights LMDB database: {weights_lmdb_path_old}")
                print(f"  | Position weights database contains {len(self.position_weights_keys)} entries")
            else:
                # Check for position_weights file format
                self._load_position_weights_files()

    def _connect_db(self):
        """Create read-only DB connection (process-safe)."""
        if self.db is None:
            # Use thread-local storage so each worker has its own connection
            if not hasattr(_thread_local, 'lmdb_connections'):
                _thread_local.lmdb_connections = {}
            
            # Create one connection per LMDB path
            if self.lmdb_path not in _thread_local.lmdb_connections:
                # map_size: at least 1.2x file size, max 10TB; must be >= file size for correct mapping
                try:
                    file_size = os.path.getsize(self.lmdb_path) if os.path.isfile(self.lmdb_path) else 0
                    if os.path.isdir(self.lmdb_path):
                        data_file = os.path.join(self.lmdb_path, "data.mdb")
                        if os.path.exists(data_file):
                            file_size = os.path.getsize(data_file)
                    
                    if file_size > 0:
                        map_size = int(file_size * 1.2)
                        map_size = min(map_size, 10 * 1024**4)  # Max 10TB
                    else:
                        map_size = 1024 * (1024**3)  # 1TB default for large datasets
                except Exception:
                    map_size = 1024 * (1024**3)  # 1TB default on error
                
                _thread_local.lmdb_connections[self.lmdb_path] = lmdb.open(
                    self.lmdb_path,
                    map_size=map_size,
                    create=False,
                    subdir=True,
                    readonly=True,
                    lock=False,
                    readahead=True,  # Improve read performance
                    meminit=False,
                    max_readers=256,  # More concurrent workers
                )
            
            self.db = _thread_local.lmdb_connections[self.lmdb_path]

    def __len__(self):
        if hasattr(self, 'use_lmdb') and self.use_lmdb:
            return len(self.keys)
        # curr_codes may be list or array
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
        
        # Ensure code is tensor; convert or raise
        if not isinstance(code, torch.Tensor):
            if isinstance(code, str):
                # String may mean wrong key returned or corrupted data
                raise TypeError(
                    f"Expected tensor but got string at index {index}. "
                    f"This might indicate:\n"
                    f"  1. Data corruption in LMDB (stored value is string instead of tensor)\n"
                    f"  2. Key was returned instead of value\n"
                    f"  3. Wrong data format in codes file\n"
                    f"Code value (first 200 chars): {code[:200] if len(code) > 200 else code}\n"
                    f"Code type: {type(code)}"
                )
            # Try converting to tensor (e.g. numpy array)
            try:
                if hasattr(code, '__array__'):
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
        
        # Return with position_weights if present
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
                # If position_weight is None, return code only (backward compat)
                return code
        else:
            return code
    
    def _getitem_sharded_lmdb(self, index):
        """Get item from sharded LMDB."""
        shard_id = index // self.samples_per_shard
        shard_index = index % self.samples_per_shard

        if shard_id >= self.num_shards:
            raise IndexError(f"Index {index} out of range for {self.total_samples} samples")
        
        if self.shard_dbs[shard_id] is None:
            self._connect_shard_db(shard_id)

        key = self.shard_keys_list[shard_id][shard_index]
        if isinstance(key, str):
            key = key.encode('utf-8')

        # Read from shard DB
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
        """Create shard database connection."""
        if self.shard_dbs[shard_id] is not None and not force_reconnect:
            return

        shard_lmdb_path = self.shard_lmdb_paths[shard_id]

        if not hasattr(_thread_local, 'lmdb_connections'):
            _thread_local.lmdb_connections = {}
        
        if shard_lmdb_path not in _thread_local.lmdb_connections or force_reconnect:
            try:
                data_file = os.path.join(shard_lmdb_path, "data.mdb")
                if os.path.exists(data_file):
                    file_size = os.path.getsize(data_file)
                    map_size = int(file_size * 1.2)
                    map_size = min(map_size, 10 * 1024**4)  # Max 10TB
                else:
                    map_size = 100 * (1024**3)  # Default 100GB
            except Exception:
                map_size = 100 * (1024**3)  # Default 100GB
            
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
        """Get item from LMDB (process-safe, optimized)."""
        if self.db is None:
            self._connect_db()

        key = self.keys[index]
        if isinstance(key, str):
            key = key.encode('utf-8')

        try:
            with self.db.begin(buffers=True) as txn:
                value = txn.get(key)
                if value is None:
                    raise KeyError(f"Key not found: {key}")
                code_raw = pickle.loads(value)
        except Exception as e:
            # On transaction failure, reconnect
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
        """Close DB handle (shared connection left open for other workers)."""
        if self.db is not None:
            self.db = None

    def _load_position_weights_files(self):
        """Load position_weights from files (not LMDB). Exclude _keys.pt (index files)."""
        list_weights = [
            f for f in os.listdir(self.codes_dir)
            if os.path.isfile(os.path.join(self.codes_dir, f)) and \
            f.startswith("position_weights") and f.endswith(".pt") and \
            not f.endswith("_keys.pt")
        ]
        
        if list_weights:
            # New format: position_weights_v2_*.pt; fallback: position_weights_*.pt
            numbered_weights_v2 = [f for f in list_weights if f.startswith("position_weights_v2_") and f.endswith(".pt")]
            numbered_weights_old = [f for f in list_weights if f.startswith("position_weights_") and f.endswith(".pt") and not f.startswith("position_weights_v2_")]

            numbered_weights = numbered_weights_v2 if numbered_weights_v2 else numbered_weights_old

            if numbered_weights:
                numbered_weights.sort()
                print(f"  | Found {len(numbered_weights)} position_weights files (augmentation versions)")

                all_weights = []
                for weights_file in numbered_weights:
                    weights_path = os.path.join(self.codes_dir, weights_file)
                    weights = torch.load(weights_path, weights_only=False)
                    # Ensure loaded value is tensor
                    if not isinstance(weights, torch.Tensor):
                        print(f"  | WARNING: {weights_file} is not a tensor (type: {type(weights)}), skipping")
                        continue
                    all_weights.append(weights)
                    print(f"  |   - {weights_file}: shape {weights.shape}")
                
                if len(all_weights) > 0:
                    self.position_weights = torch.cat(all_weights, dim=0)
                    self.use_position_weights = True
                    print(f"  | Merged position_weights shape: {self.position_weights.shape}")
                else:
                    print(f"  | WARNING: No valid position_weights files found, skipping position weights")
                    self.use_position_weights = False
                
                if hasattr(self, 'curr_codes') and len(self.position_weights) != len(self.curr_codes):
                    print(f"  | WARNING: Position weights length ({len(self.position_weights)}) != codes length ({len(self.curr_codes)})")
                    self.use_position_weights = False
            elif len(list_weights) == 1:
                # Single position_weights file (backward compat)
                weights_path = os.path.join(self.codes_dir, list_weights[0])
                print(f"  | Loading position_weights from: {weights_path}")
                self.position_weights = torch.load(weights_path, weights_only=False)
                self.use_position_weights = True
                print(f"  | Position weights shape: {self.position_weights.shape}")
                
                if hasattr(self, 'curr_codes') and len(self.position_weights) != len(self.curr_codes):
                    print(f"  | WARNING: Position weights length ({len(self.position_weights)}) != codes length ({len(self.curr_codes)})")
                    self.use_position_weights = False
            else:
                print(f"  | WARNING: Found {len(list_weights)} position_weights files, expected 1 or numbered files. Disabling position_weights.")
        else:
            print(f"  | No position_weights files found in {self.codes_dir}")
    
    def _get_position_weight_lmdb(self, index):
        """Get position_weight from LMDB."""
        if not self.use_position_weights:
            return None

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
        """Create position_weights LMDB connection (same map_size logic as codes LMDB)."""
        if not hasattr(self, 'position_weights_lmdb_path'):
            return

        if not hasattr(_thread_local, 'lmdb_connections'):
            _thread_local.lmdb_connections = {}

        if self.position_weights_lmdb_path not in _thread_local.lmdb_connections:
            import os
            try:
                file_size = os.path.getsize(self.position_weights_lmdb_path) if os.path.isfile(self.position_weights_lmdb_path) else 0
                if os.path.isdir(self.position_weights_lmdb_path):
                    data_file = os.path.join(self.position_weights_lmdb_path, "data.mdb")
                    if os.path.exists(data_file):
                        file_size = os.path.getsize(data_file)

                if file_size > 0:
                    map_size = int(file_size * 1.2)
                    map_size = min(map_size, 10 * 1024**4)
                else:
                    map_size = 1024 * (1024**3)
            except Exception:
                map_size = 1024 * (1024**3)
            
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
        
        # Avoid loading keys file by mistake
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
        
        # Check loaded data format
        if isinstance(loaded_data, torch.Tensor):
            self.curr_codes = loaded_data
            print(f"  | Loaded codes as tensor: shape {self.curr_codes.shape}")
        elif isinstance(loaded_data, list):
            if len(loaded_data) > 0 and isinstance(loaded_data[0], torch.Tensor):
                # Convert list of tensors to single tensor
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
                first_elem_type = type(loaded_data[0]) if len(loaded_data) > 0 else 'empty list'
                is_keys_file = (isinstance(loaded_data, list) and 
                               len(loaded_data) > 0 and 
                               isinstance(loaded_data[0], str) and
                               all(isinstance(x, str) for x in loaded_data[:10]))

                # Build detailed error message
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
    # num_augmentations: only train uses augmentation; val/test use default codes.lmdb
    dset_cfg = config.get("dset", {})
    _dget = lambda c, k, default=None: c.get(k, default) if hasattr(c, "get") else getattr(c, k, default)
    if split == "train":
        num_augmentations = _dget(dset_cfg, "num_augmentations", None) or config.get("num_augmentations", None)

        if num_augmentations is None and (config.get("codes_dir") or _dget(dset_cfg, "codes_dir_no_aug")) is not None:
            codes_dir = config.get("codes_dir") or _dget(dset_cfg, "codes_dir_no_aug")
            split_dir = os.path.join(codes_dir, split)
            if os.path.exists(split_dir):
                list_lmdb = [
                    f for f in os.listdir(split_dir)
                    if os.path.isfile(os.path.join(split_dir, f)) and \
                    f.startswith("codes_aug") and f.endswith(".lmdb")
                ]
                if list_lmdb:
                    try:
                        first_lmdb = sorted(list_lmdb)[0]
                        parts = first_lmdb.replace(".lmdb", "").split("_")
                        if len(parts) >= 2 and parts[1].startswith("aug"):
                            num_augmentations = int(parts[1].replace("aug", ""))
                            print(f">> Inferred num_augmentations={num_augmentations} from LMDB file: {first_lmdb}")
                    except Exception:
                        pass
                
                if num_augmentations is None:
                    list_codes = [
                        f for f in os.listdir(split_dir)
                        if os.path.isfile(os.path.join(split_dir, f)) and \
                        f.startswith("codes_aug") and f.endswith(".pt")
                    ]
                    if list_codes:
                        try:
                            first_file = sorted(list_codes)[0]
                            parts = first_file.split("_")
                            if len(parts) >= 2 and parts[1].startswith("aug"):
                                num_augmentations = int(parts[1].replace("aug", ""))
                                print(f">> Inferred num_augmentations={num_augmentations} from codes files in {split_dir}")
                        except Exception:
                            pass
    else:
        num_augmentations = None
        print(f">> {split} split: using default codes.lmdb format (no augmentation)")
    
    # use_sharded_lmdb: train uses config value; val/test force full LMDB (None)
    use_sharded_lmdb_config = _dget(dset_cfg, "use_sharded_lmdb", None) or config.get("use_sharded_lmdb", None)
    if split == "train":
        use_sharded_lmdb = use_sharded_lmdb_config
    else:
        use_sharded_lmdb = None
        if use_sharded_lmdb_config is True:
            print(f">> {split} split: use_sharded_lmdb is set to None (auto-detect, will fallback to full LMDB)")
    
    # Single source: dset.position_weight; top-level config.position_weight overrides
    position_weight_config = config.get("position_weight") or (config.get("dset") or {}).get("position_weight", {})
    position_weight_enabled = position_weight_config.get("enabled", False)
    load_position_weights = position_weight_enabled
    
    dset = CodeDataset(
        dset_name=config["dset"]["dset_name"],
        codes_dir=config["codes_dir"],
        split=split,
        num_augmentations=num_augmentations,
        use_sharded_lmdb=use_sharded_lmdb,
        load_position_weights=load_position_weights,
    )

    # If use_augmented_codes or position_weight.enabled=False, do not use precomputed position_weights
    use_augmented_codes = _dget(dset_cfg, "use_augmented_codes", False) or config.get("use_augmented_codes", False)
    if not use_augmented_codes or not position_weight_enabled:
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

    # DataLoader: num_workers from config
    loader_kwargs = {
        "batch_size": min(config["dset"]["batch_size"], len(dset)),
        "num_workers": config["dset"]["num_workers"],
        "shuffle": True if split == "train" else False,
        "pin_memory": True,
        "drop_last": True,
        "persistent_workers": True if config["dset"]["num_workers"] > 0 else False,
    }

    if "prefetch_factor" in config["dset"]:
        loader_kwargs["prefetch_factor"] = config["dset"]["prefetch_factor"]
    
    loader = torch.utils.data.DataLoader(dset, **loader_kwargs)
    # fabric.print(f">> {split} set size: {len(dset)}")
    print(f">> {split} set size: {len(dset)}")

    return loader # fabric.setup_dataloaders(loader, use_distributed_sampler=(split == "train"))
