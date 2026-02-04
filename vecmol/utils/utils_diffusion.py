import os
import time
import fcntl
import torch
from tqdm import tqdm

from vecmol.utils.utils_nf import load_network, infer_codes, normalize_code


def compute_position_weights(
    atom_coords: torch.Tensor,
    grid_coords: torch.Tensor,
    batch_idx: torch.Tensor,
    radius: float = 3.0,
    weight_alpha: float = 0.5,
    device: torch.device = None
) -> torch.Tensor:
    """
    Compute position weights for grid coordinates based on the number of nearby atoms.
    
    For each grid coordinate, count the number of atoms within the specified radius,
    and compute weight as: weight = 1 + alpha * num_atoms
    
    Args:
        atom_coords: Atom coordinates [N_atoms, 3]
        grid_coords: Grid coordinates [B, n_grid, 3] or [n_grid, 3]
        batch_idx: Batch index for atoms [N_atoms] indicating which molecule each atom belongs to
        radius: Radius threshold in Angstroms for counting nearby atoms
        weight_alpha: Weight coefficient for linear weighting: weight = 1 + alpha * num_atoms
        device: Device to perform computation on. If None, uses atom_coords device.
    
    Returns:
        weights: Position weights [B, n_grid]
    """
    if device is None:
        device = atom_coords.device
    
    # Handle grid_coords shape: [n_grid, 3] -> [1, n_grid, 3]
    if grid_coords.dim() == 2:
        grid_coords = grid_coords.unsqueeze(0)  # [1, n_grid, 3]
    
    B = grid_coords.shape[0]
    n_grid = grid_coords.shape[1]
    
    # Get unique batch indices
    unique_batches = torch.unique(batch_idx)
    B_actual = len(unique_batches)
    
    if B_actual != B:
        # If grid_coords has batch dimension but batch_idx suggests different batch size,
        # we need to handle this case
        if B == 1:
            # Single batch case: expand grid_coords to match actual batch size
            grid_coords = grid_coords.expand(B_actual, -1, -1)  # [B_actual, n_grid, 3]
            B = B_actual
        else:
            raise ValueError(f"Batch size mismatch: grid_coords has {B} batches, but batch_idx suggests {B_actual} batches")
    
    # Initialize weights (will be updated based on nearby atoms)
    weights = torch.full((B, n_grid), 1.0, device=device, dtype=torch.float32)
    
    # Process each molecule in the batch
    for b_idx, batch_id in enumerate(unique_batches):
        # Get atoms belonging to this batch
        atom_mask = (batch_idx == batch_id)
        if not atom_mask.any():
            # No atoms for this batch, keep weights as 1.0 (base weight)
            continue
        
        batch_atom_coords = atom_coords[atom_mask]  # [N_atoms_b, 3]
        batch_grid_coords = grid_coords[b_idx]  # [n_grid, 3]
        
        # Compute pairwise distances between grid points and atoms
        # grid_coords: [n_grid, 3], atom_coords: [N_atoms_b, 3]
        # distances: [n_grid, N_atoms_b]
        distances = torch.cdist(batch_grid_coords, batch_atom_coords, p=2)  # [n_grid, N_atoms_b]
        
        # Count atoms within radius for each grid point
        nearby_mask = distances < radius  # [n_grid, N_atoms_b]
        num_nearby_atoms = nearby_mask.sum(dim=1).float()  # [n_grid]
        
        # Compute weights: weight = 1 + alpha * num_atoms
        batch_weights = 1.0 + weight_alpha * num_nearby_atoms  # [n_grid]
        weights[b_idx] = batch_weights
    
    return weights


def add_noise_to_code(codes: torch.Tensor, smooth_sigma: float = 0.1) -> torch.Tensor:
    """
    Adds Gaussian noise to the input codes.

    Args:
        codes (torch.Tensor): Input codes to which noise will be added.

    Returns:
        torch.Tensor: Codes with added noise.
        torch.Tensor: Noise added to the codes.
    """
    if smooth_sigma == 0.0:
        return codes
    noise = torch.empty(codes.shape, device=codes.device, dtype=codes.dtype).normal_(0, smooth_sigma)
    return codes + noise


def find_checkpoint_path(checkpoint_path_or_dir):
    """
    Find checkpoint file path from either a direct .ckpt file path or a directory.
    
    Args:
        checkpoint_path_or_dir (str): Path to checkpoint file (.ckpt) or directory containing checkpoint files
        
    Returns:
        str: Path to the checkpoint file
        
    Raises:
        FileNotFoundError: If no checkpoint file is found
    """
    import glob
    
    # Check if checkpoint_path_or_dir is a .ckpt file (Lightning format) or directory
    if checkpoint_path_or_dir.endswith('.ckpt'):
        # Direct Lightning checkpoint file
        checkpoint_path = checkpoint_path_or_dir
    else:
        # Directory path, try to find Lightning checkpoint file
        # Priority: 1) last.ckpt, 2) latest .ckpt file
        checkpoint_path = None
        
        # Try last.ckpt first (Lightning's default last checkpoint)
        last_ckpt_path = os.path.join(checkpoint_path_or_dir, "last.ckpt")
        if os.path.exists(last_ckpt_path):
            checkpoint_path = last_ckpt_path
            print(f"Found last.ckpt in directory")
        else:
            # Try to find the latest .ckpt file
            ckpt_files = glob.glob(os.path.join(checkpoint_path_or_dir, "*.ckpt"))
            if ckpt_files:
                # Sort by modification time, get the latest
                ckpt_files.sort(key=os.path.getmtime, reverse=True)
                checkpoint_path = ckpt_files[0]
                print(f"Found latest .ckpt file: {os.path.basename(checkpoint_path)}")
            else:
                raise FileNotFoundError(
                    f"No checkpoint file found in directory: {checkpoint_path_or_dir}\n"
                    f"Expected: last.ckpt or any .ckpt file"
                )
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    return checkpoint_path


def load_checkpoint_diffusion(
    model: torch.nn.Module,
    pretrained_path: str,
    optimizer: torch.optim.Optimizer = None,
):
    """
    Loads a checkpoint file and restores the model and optimizer states.
    Used for inference scenarios (e.g., sample_diffusion.py).

    Args:
        model (torch.nn.Module): The model to load the checkpoint into.
        pretrained_path (str): The path to the checkpoint file (.ckpt) or directory containing checkpoint files.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load the checkpoint into. Defaults to None.

    Returns:
        tuple: A tuple containing the loaded model, optimizer (if provided), and code_stats.
    """
    # Find checkpoint file if directory is provided
    checkpoint_path = find_checkpoint_path(pretrained_path)
    
    # Load Lightning checkpoint
    lightning_checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    # Convert Lightning checkpoint format to expected format
    # Handle code_stats: if None or empty dict, set to None (for normalize_codes=False case)
    code_stats = lightning_checkpoint.get("code_stats", None)
    # If code_stats is an empty dict (from old checkpoints without code_stats), treat as None
    if code_stats is not None and isinstance(code_stats, dict) and len(code_stats) == 0:
        code_stats = None
    
    checkpoint = {
        "vecmol_ema_state_dict": lightning_checkpoint.get("vecmol_ema_state_dict", {}),
        "code_stats": code_stats,
        "epoch": lightning_checkpoint.get("epoch", 0)
    }
    
    # Move model to CPU temporarily for loading, then back to original device
    original_device = next(model.parameters()).device
    model = model.cpu()
    
    if optimizer is not None:
        load_network(checkpoint, model, is_compile=True, sd="vecmol_ema_state_dict", net_name="denoiser")
        optimizer.load_state_dict(checkpoint["optimizer"])
        # Move model back to original device
        model = model.to(original_device)
        return model, optimizer, checkpoint["code_stats"]
    else:
        load_network(checkpoint, model, is_compile=True, sd="vecmol_ema_state_dict", net_name="denoiser")
        # Move model back to original device
        model = model.to(original_device)
        print(f">> loaded model trained for {checkpoint['epoch']} epochs")
        return model, checkpoint["code_stats"]


def compute_codes(
    loader_field: torch.utils.data.DataLoader,
    enc: torch.nn.Module,
    config_nf: dict,
    split: str,
    normalize_codes: bool,
    code_stats: dict=None,
) -> tuple:
    """
    Computes the codes using the provided encoder and data loader.

    Args:
        loader_field (torch.utils.data.DataLoader): DataLoader for the field data.
        enc (torch.nn.Module): Encoder model to generate codes.
        config_nf (dict): Configuration dictionary for the neural field.
        split (str): Data split identifier (e.g., 'train', 'val', 'test').
        normalize_codes (bool): Whether to normalize the codes.
        code_stats (dict, optional): Optional dictionary to store code statistics. Defaults to None.
    Returns:
        tuple: A tuple containing the generated codes and the code statistics.
    """
    codes = infer_codes(
        loader_field,
        enc,
        config_nf,
        to_cpu=True,
        code_stats=code_stats,
        n_samples=5_000,
    )
    if code_stats is None:
        code_stats = process_codes(codes, split, normalize_codes)
    else:
        get_stats(codes, message=f"====normalized codes {split}====")
    return codes, code_stats


def compute_code_stats_offline(
    loader: torch.utils.data.DataLoader,
    split: str,
    normalize_codes: bool,
    num_augmentations: int = None,
    max_samples: int = None,
) -> dict:
    """
    Computes statistics for codes offline.
    
    This function now supports caching to avoid recomputing statistics,
    and optional early stopping / subsampling to compute approximate statistics.
    Cache files are saved in the same directory as the codes data.

    Args:
        loader (torch.utils.data.DataLoader): DataLoader containing the dataset.
        split (str): The data split (e.g., 'train', 'val', 'test').
        normalize_codes (bool): Whether to normalize the codes.
        num_augmentations (int, optional): Number of augmentations, used for generating cache filename.
        max_samples (int, optional): If provided, stop after processing roughly this many scalar samples.

    Returns:
        dict: A dictionary containing the computed code statistics.
    """
    dataset = loader.dataset
    
    # If num_augmentations is not provided, try to get it from dataset
    if num_augmentations is None and hasattr(dataset, 'num_augmentations'):
        num_augmentations = dataset.num_augmentations
    
    # Determine cache file path
    cache_path = None
    if hasattr(dataset, 'use_lmdb') and dataset.use_lmdb:
        # LMDB mode: use LMDB path as cache directory
        # For sharded LMDB, lmdb_path points to the first shard, but the cache should be in the parent directory of the shard directory
        if hasattr(dataset, 'use_sharded') and dataset.use_sharded:
            # Sharded mode: use codes_dir as cache directory (parent directory of shard files)
            cache_dir = dataset.codes_dir
        else:
            # Single LMDB mode: use directory of LMDB path
            lmdb_path = dataset.lmdb_path
            cache_dir = os.path.dirname(lmdb_path)
        
        # Generate cache file name: based on split, normalize_codes and num_augmentations parameters
        if num_augmentations is not None:
            cache_filename = f"code_stats_{split}_aug{num_augmentations}_norm{normalize_codes}.pt"
        else:
            cache_filename = f"code_stats_{split}_norm{normalize_codes}.pt"  # 向后兼容
        cache_path = os.path.join(cache_dir, cache_filename)
    elif hasattr(dataset, 'codes_dir'):
        # Traditional mode: use codes_dir as cache directory
        cache_dir = dataset.codes_dir
        if num_augmentations is not None:
            cache_filename = f"code_stats_{split}_aug{num_augmentations}_norm{normalize_codes}.pt"
        else:
            cache_filename = f"code_stats_{split}_norm{normalize_codes}.pt"  # 向后兼容
        cache_path = os.path.join(cache_dir, cache_filename)
    
    # Try to load cache
    if cache_path and os.path.exists(cache_path):
        try:
            print(f"Loading cached code statistics from: {cache_path}")
            cached_stats = torch.load(cache_path, weights_only=False)
            print(f"Successfully loaded cached code statistics for {split} split")
            print(f"====cached codes {split}====")
            print(f"min: {cached_stats.get('min_normalized', 'N/A')}")
            print(f"max: {cached_stats.get('max_normalized', 'N/A')}")
            print(f"mean: {cached_stats.get('mean', 'N/A')}")
            print(f"std: {cached_stats.get('std', 'N/A')}")
            return cached_stats
        except Exception as e:
            print(f"Warning: Failed to load cached code statistics: {e}")
            print("Will recompute statistics...")
    
    # If cache does not exist, use file lock to ensure only one process computes statistics
    lock_path = None
    lock_file = None
    should_compute = False
    
    if cache_path:
        # Create lock file path
        lock_path = cache_path + ".lock"
        lock_dir = os.path.dirname(lock_path)
        os.makedirs(lock_dir, exist_ok=True)
        
        try:
            # Try to acquire file lock (non-blocking)
            lock_file = open(lock_path, 'w')
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            should_compute = True
            print(f"Acquired lock for computing code statistics (process will compute stats)")
        except (IOError, OSError):
            # Cannot acquire lock, other processes are computing
            if lock_file:
                lock_file.close()
            lock_file = None
            should_compute = False
            print(f"Another process is computing code statistics, waiting for cache file...")
            
            # Wait for other processes to complete computation (maximum wait time is 2 hours)
            max_wait_time = 7200  # 2小时
            wait_interval = 2  # Check every 2 seconds
            elapsed_time = 0
            
            while elapsed_time < max_wait_time:
                if os.path.exists(cache_path):
                    try:
                        print(f"Cache file found! Loading cached code statistics from: {cache_path}")
                        cached_stats = torch.load(cache_path, weights_only=False)
                        print(f"Successfully loaded cached code statistics for {split} split")
                        print(f"====cached codes {split}====")
                        print(f"min: {cached_stats.get('min_normalized', 'N/A')}")
                        print(f"max: {cached_stats.get('max_normalized', 'N/A')}")
                        print(f"mean: {cached_stats.get('mean', 'N/A')}")
                        print(f"std: {cached_stats.get('std', 'N/A')}")
                        return cached_stats
                    except Exception as e:
                        print(f"Warning: Failed to load cache file: {e}, continuing to wait...")
                
                time.sleep(wait_interval)
                elapsed_time += wait_interval
                if elapsed_time % 30 == 0:  # Print every 30 seconds
                    print(f"Still waiting for cache file... ({elapsed_time}s elapsed)")
            
            # If timeout, raise error
            raise RuntimeError(
                f"Timeout waiting for code statistics cache file. "
                f"Expected cache file: {cache_path}. "
                f"Please check if another process is still computing statistics."
            )
    
    # Only compute if lock is acquired or no cache path exists
    if not should_compute and cache_path:
        # If lock is not acquired, other processes are computing, we have returned in the waiting loop above
        # Here should not be reached, but for safety, check cache again
        if os.path.exists(cache_path):
            try:
                cached_stats = torch.load(cache_path, weights_only=False)
                return cached_stats
            except Exception:
                pass
        raise RuntimeError("Unexpected state: should not compute but no cache available")
    
    # Check if using LMDB mode
    if hasattr(dataset, 'use_lmdb') and dataset.use_lmdb:
        # LMDB mode: use streaming mode to compute statistics, avoid memory overflow
        print(f"Computing code statistics from LMDB database for {split} split (streaming mode)...")
        print(f"Total samples: {len(dataset)}")
        if max_samples is not None:
            print(f"[Approximate stats] Will stop after roughly {max_samples} scalar samples.")
        
        # Use online statistics algorithm (Welford's algorithm) to compute mean and standard deviation
        # Track max and min values simultaneously
        n_samples = 0
        mean = None
        M2 = None  # Intermediate variable for calculating variance
        max_val = None
        min_val = None
        
        # Create a temporary DataLoader for batch processing
        # Use smaller batch_size and single thread to avoid memory issues
        temp_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=min(loader.batch_size, 32),  # Limit batch size
            shuffle=False,
            num_workers=0,  # Single thread to avoid multi-process issues
            pin_memory=False
        )
        
        for batch in tqdm(temp_loader, desc=f"Computing stats for {split}"):
            # batch may be a single tensor or a list of tensors
            if isinstance(batch, (list, tuple)):
                batch_codes = torch.stack(batch) if len(batch) > 0 else None
            else:
                batch_codes = batch
            
            if batch_codes is None or batch_codes.numel() == 0:
                continue
            
            # Flatten batch for calculating statistics
            batch_flat = batch_codes.flatten()
            
            # Update max and min values
            batch_max = batch_flat.max()
            batch_min = batch_flat.min()
            
            if max_val is None:
                max_val = batch_max
                min_val = batch_min
            else:
                max_val = torch.max(max_val, batch_max)
                min_val = torch.min(min_val, batch_min)
            
            # Use Welford's online algorithm to update mean and variance
            batch_n = batch_flat.numel()
            
            # If max_samples is set, and this batch would significantly exceed the limit,
            # we still use this batch completely, but end the loop early after exceeding the limit.
            batch_mean = batch_flat.mean()
            
            if mean is None:
                # Initialize
                mean = batch_mean
                M2 = ((batch_flat - mean) ** 2).sum()
                n_samples = batch_n
            else:
                # Update statistics
                delta = batch_mean - mean
                n_samples_new = n_samples + batch_n
                mean = mean + delta * batch_n / n_samples_new
                
                # Update M2 (for calculating variance)
                M2 = M2 + ((batch_flat - batch_mean) ** 2).sum() + delta ** 2 * n_samples * batch_n / n_samples_new
                n_samples = n_samples_new
            
            # Free memory
            del batch_codes, batch_flat, batch_mean
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            if max_samples is not None and n_samples >= max_samples:
                print(f"[Approximate stats] Reached max_samples≈{max_samples} (actual scalar samples: {n_samples}), stopping early.")
                break
        
        # Calculate final statistics
        if n_samples == 0:
            raise ValueError(f"No samples found in dataset for {split} split")
        
        # Ensure all tensors are on the same device
        device = mean.device if isinstance(mean, torch.Tensor) else torch.device('cpu')
        if isinstance(max_val, torch.Tensor):
            max_val = max_val.to(device)
        if isinstance(min_val, torch.Tensor):
            min_val = min_val.to(device)
        if isinstance(M2, torch.Tensor):
            M2 = M2.to(device)
        
        # Calculate standard deviation, avoid division by zero error
        if n_samples > 1:
            std = torch.sqrt(M2 / n_samples)
            # If std is too small, set to a small positive value to avoid division by zero
            std = torch.clamp(std, min=1e-8)
        else:
            std = torch.tensor(1.0, device=device)
        
        # Print statistics
        print(f"====codes {split}====")
        print(f"min: {min_val.item()}")
        print(f"max: {max_val.item()}")
        print(f"mean: {mean.item()}")
        print(f"std: {std.item()}")
        print(f"Total samples processed: {n_samples}")
        
        # Build code_stats dictionary
        code_stats = {
            "mean": mean.item() if isinstance(mean, torch.Tensor) else mean,
            "std": std.item() if isinstance(std, torch.Tensor) else std,
        }
        
        if normalize_codes:
            # For normalized codes, we need to recompute statistics
            # But since we already have mean and std, normalized codes should be close to N(0,1)
            # We can estimate the range of normalized codes
            # Normalization formula: (x - mean) / std
            # Normalized min: (min_val - mean) / std
            # Normalized max: (max_val - mean) / std
            normalized_min = (min_val - mean) / std
            normalized_max = (max_val - mean) / std
            max_normalized = normalized_max.item() if isinstance(normalized_max, torch.Tensor) else normalized_max
            min_normalized = normalized_min.item() if isinstance(normalized_min, torch.Tensor) else normalized_min
        else:
            max_normalized = max_val.item() if isinstance(max_val, torch.Tensor) else max_val
            min_normalized = min_val.item() if isinstance(min_val, torch.Tensor) else min_val
        
        code_stats.update({
            "max_normalized": max_normalized,
            "min_normalized": min_normalized,
        })
        
        print(f"====normalized codes {split}====")
        print(f"min_normalized: {min_normalized}")
        print(f"max_normalized: {max_normalized}")
        
        # Save cache
        if cache_path:
            try:
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                # Save to temporary file, then atomically rename
                temp_cache_path = cache_path + ".tmp"
                torch.save(code_stats, temp_cache_path)
                os.rename(temp_cache_path, cache_path)
                print(f"Saved code statistics cache to: {cache_path}")
            except Exception as e:
                print(f"Warning: Failed to save code statistics cache: {e}")
        
        # Release file lock
        if lock_file:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                lock_file.close()
                # Delete lock file
                if os.path.exists(lock_path):
                    os.remove(lock_path)
            except Exception as e:
                print(f"Warning: Failed to release lock: {e}")
        
        return code_stats
    else:
        # Traditional mode: directly use curr_codes attribute
        if hasattr(dataset, 'curr_codes'):
            codes = dataset.curr_codes[:]
        else:
            raise AttributeError(
                f"Dataset does not have 'curr_codes' attribute and is not in LMDB mode. "
                f"Dataset type: {type(dataset)}"
            )
        
        code_stats = process_codes(codes, split, normalize_codes)
        
        # Save cache (traditional mode)
        if cache_path:
            try:
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                # Save to temporary file, then atomically rename
                temp_cache_path = cache_path + ".tmp"
                torch.save(code_stats, temp_cache_path)
                os.rename(temp_cache_path, cache_path)
                print(f"Saved code statistics cache to: {cache_path}")
            except Exception as e:
                print(f"Warning: Failed to save code statistics cache: {e}")
        
        # Release file lock
        if lock_file:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                lock_file.close()
                # Delete lock file
                if os.path.exists(lock_path):
                    os.remove(lock_path)
            except Exception as e:
                print(f"Warning: Failed to release lock: {e}")
        
        return code_stats


def process_codes(
    codes: torch.Tensor,
    split: str,
    normalize_codes: bool,
) -> dict:
    """
    Process the codes from the checkpoint.

    Args:
        checkpoint (dict): The checkpoint containing the codes.
        logger (object): The logger object for logging messages.
        device (torch.device): The device to use for processing the codes.
        is_filter (bool, optional): Whether to filter the codes. Defaults to False.

    Returns:
        tuple: A tuple containing the processed codes, statistics, and normalized codes.
    """
    max_val, min_val, mean, std = get_stats(
        codes,
        message=f"====codes {split}====",
    )
    code_stats = {
        "mean": mean,
        "std": std,
    }
    if normalize_codes:
        codes = normalize_code(codes, code_stats)
        max_normalized, min_normalized, _, _ = get_stats(
            codes,
            message=f"====normalized codes {split}====",
        )
    else:
        max_normalized, min_normalized = max_val, min_val
    code_stats.update({
        "max_normalized": max_normalized.item(),
        "min_normalized": min_normalized.item(),
    })
    return code_stats


def get_stats(
    codes: torch.Tensor,
    message: str = None,
):
    """
    Calculate statistics of the input codes.

    Args:
        codes_init (torch.Tensor): The input codes.
        message (str, optional): Additional message to log. Defaults to None.

    Returns:
        tuple: A tuple containing the calculated statistics:
            - max (torch.Tensor): The maximum values.
            - min (torch.Tensor): The minimum values.
            - mean (torch.Tensor): The mean values.
            - std (torch.Tensor): The standard deviation values.
            - (optional) codes (torch.Tensor): The filtered codes if `is_filter` is True.
    """
    if message is not None:
        print(message)
    max_val = codes.max()
    min_val = codes.min()
    mean = codes.mean()
    std = codes.std()
    print(f"min: {min_val.item()}")
    print(f"max: {max_val.item()}")
    print(f"mean: {mean.item()}")
    print(f"std: {std.item()}")
    print(f"codes size: {codes.shape}")
    return max_val, min_val, mean, std


def load_checkpoint_state_diffusion(model, checkpoint_path_or_dir):
    """
    Helper function to load checkpoint state for VecMol model.
    Automatically finds checkpoint file if a directory is provided.
    
    Args:
        model: The VecMol Lightning module
        checkpoint_path_or_dir (str): Path to the checkpoint file (.ckpt) or directory containing checkpoint files
        
    Returns:
        dict: Training state dictionary containing epoch, losses, and best_loss
    """
    # Find checkpoint file if directory is provided
    checkpoint_path = find_checkpoint_path(checkpoint_path_or_dir)
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Load Vecmol state
    # Use strict=True to ensure all keys match exactly
    if "vecmol" in state_dict:
        # Remove _orig_mod. prefix
        new_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict["vecmol"].items()}
        try:
            model.vecmol.load_state_dict(new_state_dict, strict=True)
            print("Loaded Vecmol state from 'vecmol' key (strict=True)")
        except RuntimeError as e:
            raise RuntimeError(
                f"Failed to load Vecmol state from 'vecmol' key with strict=True. "
                f"Original error: {str(e)}"
            ) from e
    elif "vecmol_state_dict" in state_dict:
        # Remove _orig_mod. prefix
        new_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict["vecmol_state_dict"].items()}
        try:
            model.vecmol.load_state_dict(new_state_dict, strict=True)
            print("Loaded Vecmol state from 'vecmol_state_dict' key (strict=True)")
        except RuntimeError as e:
            raise RuntimeError(
                f"Failed to load Vecmol state from 'vecmol_state_dict' key with strict=True. "
                f"Original error: {str(e)}"
            ) from e
    else:
        raise RuntimeError(
            "No Vecmol state found in checkpoint. "
            "Expected 'vecmol' or 'vecmol_state_dict' key in checkpoint."
        )
    
    # Load EMA state
    # ModelEma structure: model.vecmol_ema.module is the actual model
    # If ModelEma's module is wrapped by DDP, then model.vecmol_ema.module.module is the actual model
    # Checkpoint saves model.vecmol_ema.module.state_dict(), keys are net.xxx
    # Need to correctly map checkpoint keys to target model keys
    
    if "vecmol_ema" in state_dict or "vecmol_ema_state_dict" in state_dict:
        # Get EMA state_dict from checkpoint
        ema_state_dict = state_dict.get("vecmol_ema") or state_dict.get("vecmol_ema_state_dict")
        # Remove _orig_mod. prefix
        checkpoint_state_dict = {k.replace("_orig_mod.", ""): v for k, v in ema_state_dict.items()}
        
        # Get target model state_dict, check actual key format
        target_model = model.vecmol_ema.module
        target_state_dict = target_model.state_dict()
        
        # Map checkpoint keys to target model keys based on target model key format
        mapped_state_dict = {}
        for ckpt_key, ckpt_value in checkpoint_state_dict.items():
            # Try direct match
            if ckpt_key in target_state_dict:
                mapped_state_dict[ckpt_key] = ckpt_value
            # Try adding module. prefix (for DDP wrapped case)
            elif f"module.{ckpt_key}" in target_state_dict:
                mapped_state_dict[f"module.{ckpt_key}"] = ckpt_value
            # Try removing module. prefix (if checkpoint has but target doesn't)
            elif ckpt_key.startswith("module.") and ckpt_key[7:] in target_state_dict:
                mapped_state_dict[ckpt_key[7:]] = ckpt_value
            else:
                # If cannot match, record warning but continue (will raise error with strict=True)
                print(f"Warning: Cannot map checkpoint key '{ckpt_key}' to target model")
        
        # Verify all target model keys have corresponding checkpoint values
        missing_keys = set(target_state_dict.keys()) - set(mapped_state_dict.keys())
        if missing_keys:
            raise RuntimeError(
                f"Missing keys in checkpoint for ModelEma: {list(missing_keys)[:10]}... "
                f"(total {len(missing_keys)} missing keys)"
            )
        
        # Verify checkpoint has no extra keys
        unexpected_keys = set(mapped_state_dict.keys()) - set(target_state_dict.keys())
        if unexpected_keys:
            raise RuntimeError(
                f"Unexpected keys in checkpoint for ModelEma: {list(unexpected_keys)[:10]}... "
                f"(total {len(unexpected_keys)} unexpected keys)"
            )
        
        # Use strict=True for strict check
        try:
            target_model.load_state_dict(mapped_state_dict, strict=True)
            print("Loaded Vecmol EMA state with strict=True (all keys matched)")
        except RuntimeError as e:
            raise RuntimeError(
                f"Failed to load Vecmol EMA state with strict=True. "
                f"Original error: {str(e)}"
            ) from e
    
    # Load decoder state if joint fine-tuning is enabled
    # Use strict=True to ensure all keys match exactly
    if hasattr(model, 'joint_finetune_enabled') and model.joint_finetune_enabled:
        if hasattr(model, 'dec_module') and model.dec_module is not None:
            if "decoder_state_dict" in state_dict:
                decoder_state_dict = state_dict["decoder_state_dict"]
                # Handle _orig_mod. prefix if present
                new_state_dict = {k.replace("_orig_mod.", ""): v for k, v in decoder_state_dict.items()}
                try:
                    model.dec_module.load_state_dict(new_state_dict, strict=True)
                    print("Loaded decoder state from checkpoint (joint fine-tuning mode, strict=True)")
                except RuntimeError as e:
                    raise RuntimeError(
                        f"Failed to load decoder state with strict=True (joint fine-tuning mode). "
                        f"Original error: {str(e)}"
                    ) from e
            else:
                raise RuntimeError(
                    "Joint fine-tuning is enabled but no decoder_state_dict found in checkpoint. "
                    "Cannot continue without decoder state."
                )
    
    # Load training state
    training_state = {
        "epoch": state_dict.get("epoch", None),
        "train_losses": state_dict.get("train_losses", []),
        "val_losses": state_dict.get("val_losses", []),
        "best_loss": state_dict.get("best_loss", float("inf"))
    }
    
    return training_state
