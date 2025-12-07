import os
import torch
from tqdm import tqdm

from funcmol.utils.utils_nf import load_network, infer_codes, normalize_code


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


def load_checkpoint_fm(
    model: torch.nn.Module,
    pretrained_path: str,
    optimizer: torch.optim.Optimizer = None,
):
    """
    Loads a checkpoint file and restores the model and optimizer states.

    Args:
        model (torch.nn.Module): The model to load the checkpoint into.
        pretrained_path (str): The path to the directory containing the checkpoint file.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load the checkpoint into. Defaults to None.
        best_model (bool, optional): Whether to load the best model checkpoint or the regular checkpoint.
            Defaults to True.

    Returns:
        tuple: A tuple containing the loaded model, optimizer (if provided), and the number of epochs trained.
    """
    # Check if pretrained_path is a .ckpt file (Lightning format) or directory
    if pretrained_path.endswith('.ckpt'):
        # Load Lightning checkpoint directly
        lightning_checkpoint = torch.load(pretrained_path, map_location='cpu', weights_only=False)
        # Convert Lightning checkpoint format to expected format
        checkpoint = {
            "funcmol_ema_state_dict": lightning_checkpoint["funcmol_ema_state_dict"],
            "code_stats": lightning_checkpoint.get("code_stats", {}),
            "epoch": lightning_checkpoint.get("epoch", 0)
        }
    else:
        # Load from directory with checkpoint.pth.tar
        checkpoint = torch.load(os.path.join(pretrained_path, "checkpoint.pth.tar"))
    
    # Move model to CPU temporarily for loading, then back to original device
    original_device = next(model.parameters()).device
    model = model.cpu()
    
    if optimizer is not None:
        load_network(checkpoint, model, is_compile=True, sd="funcmol_ema_state_dict", net_name="denoiser")
        optimizer.load_state_dict(checkpoint["optimizer"])
        # Move model back to original device
        model = model.to(original_device)
        return model, optimizer, checkpoint["code_stats"]
    else:
        load_network(checkpoint, model, is_compile=True, sd="funcmol_ema_state_dict", net_name="denoiser")
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
        n_samples=100_000,
    )
    if code_stats is None:
        code_stats = process_codes(codes, split, normalize_codes)
    else:
        get_stats(codes, message=f"====normalized codes {split}====")
    return codes, code_stats


def compute_code_stats_offline(
    loader: torch.utils.data.DataLoader,
    split: str,
    normalize_codes: bool
) -> dict:
    """
    Computes statistics for codes offline.

    Args:
        loader (torch.utils.data.DataLoader): DataLoader containing the dataset.
        split (str): The data split (e.g., 'train', 'val', 'test').
        normalize_codes (bool): Whether to normalize the codes.

    Returns:
        dict: A dictionary containing the computed code statistics.
    """
    dataset = loader.dataset
    # 检查是否使用LMDB模式
    if hasattr(dataset, 'use_lmdb') and dataset.use_lmdb:
        # LMDB模式：使用流式处理计算统计信息，避免内存溢出
        print(f"Computing code statistics from LMDB database for {split} split (streaming mode)...")
        print(f"Total samples: {len(dataset)}")
        
        # 使用在线统计算法（Welford's algorithm）来计算均值和标准差
        # 同时跟踪最大值和最小值
        n_samples = 0
        mean = None
        M2 = None  # 用于计算方差的中间变量
        max_val = None
        min_val = None
        
        # 创建一个临时DataLoader用于分批处理
        # 使用较小的batch_size和单线程以避免内存问题
        temp_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=min(loader.batch_size, 32),  # 限制batch size
            shuffle=False,
            num_workers=0,  # 单线程避免多进程问题
            pin_memory=False
        )
        
        for batch in tqdm(temp_loader, desc=f"Computing stats for {split}"):
            # batch可能是单个tensor或tensor列表
            if isinstance(batch, (list, tuple)):
                batch_codes = torch.stack(batch) if len(batch) > 0 else None
            else:
                batch_codes = batch
            
            if batch_codes is None or batch_codes.numel() == 0:
                continue
            
            # 展平batch以便计算统计信息
            batch_flat = batch_codes.flatten()
            
            # 更新最大值和最小值
            batch_max = batch_flat.max()
            batch_min = batch_flat.min()
            
            if max_val is None:
                max_val = batch_max
                min_val = batch_min
            else:
                max_val = torch.max(max_val, batch_max)
                min_val = torch.min(min_val, batch_min)
            
            # 使用Welford's online algorithm更新均值和方差
            batch_n = batch_flat.numel()
            batch_mean = batch_flat.mean()
            
            if mean is None:
                # 初始化
                mean = batch_mean
                M2 = ((batch_flat - mean) ** 2).sum()
                n_samples = batch_n
            else:
                # 更新统计量
                delta = batch_mean - mean
                n_samples_new = n_samples + batch_n
                mean = mean + delta * batch_n / n_samples_new
                
                # 更新M2（用于计算方差）
                M2 = M2 + ((batch_flat - batch_mean) ** 2).sum() + delta ** 2 * n_samples * batch_n / n_samples_new
                n_samples = n_samples_new
            
            # 释放内存
            del batch_codes, batch_flat, batch_mean
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # 计算最终统计量
        if n_samples == 0:
            raise ValueError(f"No samples found in dataset for {split} split")
        
        # 确保所有tensor在同一设备上
        device = mean.device if isinstance(mean, torch.Tensor) else torch.device('cpu')
        if isinstance(max_val, torch.Tensor):
            max_val = max_val.to(device)
        if isinstance(min_val, torch.Tensor):
            min_val = min_val.to(device)
        if isinstance(M2, torch.Tensor):
            M2 = M2.to(device)
        
        # 计算标准差，避免除零错误
        if n_samples > 1:
            std = torch.sqrt(M2 / n_samples)
            # 如果std太小，设置为一个小的正值以避免除零
            std = torch.clamp(std, min=1e-8)
        else:
            std = torch.tensor(1.0, device=device)
        
        # 打印统计信息
        print(f"====codes {split}====")
        print(f"min: {min_val.item()}")
        print(f"max: {max_val.item()}")
        print(f"mean: {mean.item()}")
        print(f"std: {std.item()}")
        print(f"Total samples processed: {n_samples}")
        
        # 构建code_stats字典
        code_stats = {
            "mean": mean.item() if isinstance(mean, torch.Tensor) else mean,
            "std": std.item() if isinstance(std, torch.Tensor) else std,
        }
        
        if normalize_codes:
            # 对于归一化后的代码，我们需要重新计算统计信息
            # 但由于我们已经有了mean和std，归一化后的代码应该接近N(0,1)
            # 我们可以估算归一化后的范围
            # 归一化公式: (x - mean) / std
            # 归一化后的min: (min_val - mean) / std
            # 归一化后的max: (max_val - mean) / std
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
        
        return code_stats
    else:
        # 传统模式：直接使用curr_codes属性
        if hasattr(dataset, 'curr_codes'):
            codes = dataset.curr_codes[:]
        else:
            raise AttributeError(
                f"Dataset does not have 'curr_codes' attribute and is not in LMDB mode. "
                f"Dataset type: {type(dataset)}"
            )
        
        code_stats = process_codes(codes, split, normalize_codes)
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


def load_checkpoint_state_fm(model, checkpoint_path):
    """
    Helper function to load checkpoint state for FuncMol model.
    
    Args:
        model: The FuncMol Lightning module
        checkpoint_path (str): Path to the checkpoint file
        
    Returns:
        dict: Training state dictionary containing epoch, losses, and best_loss
    """
    print(f"Loading checkpoint from: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path)
    
    # Load Funcmol state
    if "funcmol" in state_dict:
        # 统一去掉 _orig_mod. 前缀
        new_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict["funcmol"].items()}
        model.funcmol.load_state_dict(new_state_dict)
        print("Loaded Funcmol state from 'funcmol' key")
    elif "funcmol_state_dict" in state_dict:
        # 统一去掉 _orig_mod. 前缀
        new_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict["funcmol_state_dict"].items()}
        model.funcmol.load_state_dict(new_state_dict)
        print("Loaded Funcmol state from 'funcmol_state_dict' key")
    else:
        print("No Funcmol state found in checkpoint")
    
    # Load EMA state
    if "funcmol_ema" in state_dict:
        # 统一去掉 _orig_mod. 前缀
        new_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict["funcmol_ema"].items()}
        model.funcmol_ema.load_state_dict(new_state_dict)
        print("Loaded Funcmol EMA state from 'funcmol_ema' key")
    elif "funcmol_ema_state_dict" in state_dict:
        # 统一去掉 _orig_mod. 前缀
        new_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict["funcmol_ema_state_dict"].items()}
        model.funcmol_ema.load_state_dict(new_state_dict)
        print("Loaded Funcmol EMA state from 'funcmol_ema_state_dict' key")
    
    # Load training state
    training_state = {
        "epoch": state_dict.get("epoch", None),
        "train_losses": state_dict.get("train_losses", []),
        "val_losses": state_dict.get("val_losses", []),
        "best_loss": state_dict.get("best_loss", float("inf"))
    }
    
    return training_state
