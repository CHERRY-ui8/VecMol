import os
import torch

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
    fabric = None,
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
        checkpoint = fabric.load(os.path.join(pretrained_path, "checkpoint.pth.tar"))
    
    # Move model to CPU temporarily for loading, then back to original device
    original_device = next(model.parameters()).device
    model = model.cpu()
    
    if optimizer is not None:
        load_network(checkpoint, model, fabric, is_compile=True, sd="funcmol_ema_state_dict", net_name="denoiser")
        optimizer.load_state_dict(checkpoint["optimizer"])
        # Move model back to original device
        model = model.to(original_device)
        return model, optimizer, checkpoint["code_stats"]
    else:
        load_network(checkpoint, model, fabric, is_compile=True, sd="funcmol_ema_state_dict", net_name="denoiser")
        # Move model back to original device
        model = model.to(original_device)
        fabric.print(f">> loaded model trained for {checkpoint['epoch']} epochs")
        return model, checkpoint["code_stats"]


def log_metrics(
    exp_name: str,
    epoch: int,
    train_loss: float,
    val_loss: float,
    best_res: float,
    time: float,
    fabric: object
):
    """
    Logs the metrics of a training experiment.

    Args:
        exp_name (str): The name of the experiment.
        epoch (int): The current epoch number.
        train_loss (float): The training loss value.
        val_loss (float): The validation loss value.
        best_res (float): The best result achieved so far.
        time (float): The time taken for the epoch.
        fabric (object): An object with a print method to output the log.

    Returns:
    None
    """
    str_ = f">> {exp_name} epoch: {epoch} ({time:.2f}s)\n"
    str_ += f"[train_loss] {train_loss:.2f} |"
    if val_loss is not None and best_res is not None:
        str_ += f" | [val_loss] {val_loss:.2f} (best: {best_res:.2f})"
    fabric.print(str_)

def compute_codes(
    loader_field: torch.utils.data.DataLoader,
    enc: torch.nn.Module,
    config_nf: dict,
    split: str,
    fabric: object,
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
        fabric (object): Fabric object for distributed training.
        normalize_codes (bool): Whether to normalize the codes.
        code_stats (dict, optional): Optional dictionary to store code statistics. Defaults to None.
    Returns:
        tuple: A tuple containing the generated codes and the code statistics.
    """
    codes = infer_codes(
        loader_field,
        enc,
        config_nf,
        fabric=fabric,
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
    # fabric: object,
    normalize_codes: bool
) -> dict:
    """
    Computes statistics for codes offline.

    Args:
        loader (torch.utils.data.DataLoader): DataLoader containing the dataset.
        split (str): The data split (e.g., 'train', 'val', 'test').
        fabric (object): An object representing the fabric used for processing.
        normalize_codes (bool): Whether to normalize the codes.

    Returns:
        dict: A dictionary containing the computed code statistics.
    """
    codes = loader.dataset.curr_codes[:]
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
    # fabric: object = None,
    message: str = None,
):
    """
    Calculate statistics of the input codes.

    Args:
        codes_init (torch.Tensor): The input codes.
        fabric (object, optional): The logger object for logging messages. Defaults to None.
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
