import os
import numpy as np
import torch
from collections import OrderedDict, defaultdict
from tqdm import tqdm
from torch import nn
from typing import cast
import shutil
from funcmol.utils.constants import PADDING_INDEX
from funcmol.utils.gnf_converter import GNFConverter
from torch_geometric.utils import to_dense_batch
from funcmol.models.encoder import CrossGraphEncoder
from funcmol.models.decoder import Decoder, get_atom_coords
from funcmol.utils.utils_base import convert_xyzs_to_sdf, save_xyz
from funcmol.utils.utils_vis import visualize_voxel_grid
import time
from omegaconf import OmegaConf
import random
from pathlib import Path
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    SummaryWriter = None


def get_latest_model_path(base_exp_dir: str = "exps/neural_field", model_prefix: str = "nf_qm9_") -> str:
    """
    自动获取最新的模型路径
    
    Args:
        base_exp_dir: 实验目录的基础路径
        model_prefix: 模型文件夹的前缀
        
    Returns:
        str: 最新模型的完整路径
    """
    exp_dir = Path(base_exp_dir)
    if not exp_dir.exists():
        raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")
    
    # 查找所有以指定前缀开头的目录
    exp_dirs = [d for d in exp_dir.iterdir() if d.is_dir() and d.name.startswith(model_prefix)]
    if not exp_dirs:
        raise FileNotFoundError(f"No experiment directories found with prefix '{model_prefix}'")
    
    # 选择最新的目录（按修改时间排序）
    latest_exp_dir = max(exp_dirs, key=lambda x: x.stat().st_mtime)
    model_path = latest_exp_dir / "model.pt"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    print(f"Using latest model from: {latest_exp_dir}")
    return str(latest_exp_dir)

def create_neural_field(config: dict, fabric: object) -> tuple:
    """
    Creates and compiles the Encoder and Decoder neural network models based on the provided configuration.

    Args:
        config (dict): A dictionary containing the configuration parameters for the models.
            Expected keys:
                - "decoder": A dictionary with the following keys:
                    - "code_dim" (int): The dimension of the bottleneck code.
                    - "hidden_dim" (int): The hidden dimension of the decoder.
                    - "coord_dim" (int): The coordinate dimension for the decoder.
                    - "n_layers" (int): The number of layers in the decoder.
                    - "input_scale" (float): The input scale for the decoder.
                - "dset": A dictionary with the following keys:
                    - "n_channels" (int): The number of input channels.
                    - "grid_dim" (int): The grid dimension of the dataset.
                - "encoder": A dictionary with the following keys:
                    - "level_channels" (list of int): The number of channels at each level of the encoder.
                    - "smaller" (bool, optional): A flag indicating whether to use a smaller encoder. Defaults to False.
        fabric (object): An object that provides utility functions such as printing and model compilation.

    Returns:
        tuple: A tuple containing the compiled Encoder and Decoder models.
    """
    # Initialize the encoder
    enc = CrossGraphEncoder(
        n_atom_types=config["dset"]["n_channels"],
        grid_size=config["dset"]["grid_size"],
        code_dim=config["encoder"]["code_dim"],
        hidden_dim=config["encoder"]["hidden_dim"],
        num_layers=config["encoder"]["num_layers"],
        k_neighbors=config["encoder"]["k_neighbors"],
        atom_k_neighbors=config["encoder"]["atom_k_neighbors"],
        anchor_spacing=config["dset"]["anchor_spacing"]
    )
    # Initialize the decoder
    dec = Decoder({
        "grid_size": config["dset"]["grid_size"],
        "anchor_spacing": config["dset"]["anchor_spacing"],
        "hidden_dim": config["decoder"]["hidden_dim"],
        "n_layers": config["decoder"]["n_layers"],
        "k_neighbors": config["decoder"]["k_neighbors"],
        "n_channels": config["dset"]["n_channels"],
        "code_dim": config["decoder"]["code_dim"]
    }, device=fabric.device)

    return enc, dec


def train_nf(
    config: dict,
    loader: torch.utils.data.DataLoader,
    dec: nn.Module,
    optim_dec: torch.optim.Optimizer,
    enc: nn.Module,
    optim_enc: torch.optim.Optimizer,
    criterion: nn.Module,
    fabric: object,
    gnf_converter: GNFConverter,
    metrics=None,
    epoch: int = 0,
    batch_train_losses=None, 
    global_step=0,
    tensorboard_writer=None,
) -> float:
    """
    Trains the neural field model for one epoch.

    Args:
        config (dict): Configuration dictionary.
        loader (DataLoader): DataLoader for training data.
        dec (nn.Module): Decoder network.
        optim_dec (Optimizer): Optimizer for decoder.
        enc (nn.Module): Encoder network.
        optim_enc (Optimizer): Optimizer for encoder.
        criterion (nn.Module): Loss function.
        fabric (object): Fabric object for distributed training.
        gnf_converter (GNFConverter): Instance of GNFConverter for field computation.
        metrics (dict, optional): Dictionary of metrics to track. Defaults to None.
        epoch (int, optional): Current epoch number. Defaults to 0.
        batch_train_losses (list, optional): List to store batch-wise losses. Defaults to None.
        global_step (int, optional): Global training step. Defaults to 0.
        tensorboard_writer (SummaryWriter, optional): TensorBoard writer for logging. Defaults to None.

    Returns:
        float: The computed loss value.
    """
    dec.train()
    enc.train()
    total_loss = 0.0
    max_grad_norm = 1.0
    
    # 创建进度条
    pbar = tqdm(enumerate(loader), total=len(loader), desc=f'Epoch {epoch}')
    running_loss = 0.0
    
    for i, data_batch in pbar:
        # 数据预处理
        data_batch = data_batch.to(fabric.device)
        B = data_batch.batch.max().item() + 1 # 可以直接从data_batch中获取B
        # 获取query_points
        query_points = data_batch.xs.to(fabric.device)
        if query_points.dim() == 2:
            n_points = config["dset"]["n_points"]
            query_points = query_points.view(B, n_points, 3)
        # 前向传播
        codes = enc(data_batch)
        pred_field = dec(query_points, codes)
        # 使用预计算的目标梯度场（从数据集中获取）
        target_field = data_batch.target_field.to(fabric.device)
        if target_field.dim() == 2: # TODO：看看这里的view有没有问题
            target_field = target_field.view(B, n_points, -1, 3)  # [B, n_points, n_atom_types, 3]
        elif target_field.dim() == 3:
            # 如果target_field是[n_points, n_atom_types, 3]，需要添加batch维度
            if target_field.shape[0] == n_points:
                target_field = target_field.unsqueeze(0).expand(B, -1, -1, -1)  # [B, n_points, n_atom_types, 3]
            else:
                # 如果已经是batch格式，确保维度正确
                target_field = target_field.view(B, n_points, -1, 3)
        
        # # 确保pred_field和target_field的维度匹配
        # if pred_field.shape != target_field.shape:
        #     fabric.print(f"[Warning] Shape mismatch: pred_field {pred_field.shape} vs target_field {target_field.shape}")
        #     # 尝试调整target_field的维度以匹配pred_field
        #     if target_field.shape[1] != pred_field.shape[1]:
        #         # 如果点数不匹配，取较小的那个
        #         min_points = min(pred_field.shape[1], target_field.shape[1])
        #         pred_field = pred_field[:, :min_points, :, :]
        #         target_field = target_field[:, :min_points, :, :]
        
        # 调试：输出5个query_point的梯度场大小（每2000个batch输出一次）
        if i % 2000 == 0 and fabric.global_rank == 0:
            b_idx = 0  # 只看第一个样本
            n_points = pred_field.shape[1]
            
            max_idx = min(n_points - 1, target_field.shape[1] - 1) if target_field.dim() >= 2 else 0
            if max_idx < 4:  # 如果点数太少，跳过调试
                fabric.print(f"[Debug] Not enough points for debugging (max_idx: {max_idx})")
                continue
                
            idxs = random.sample(range(max_idx + 1), min(5, max_idx + 1))
            norms = []
            for idx in idxs:
                # 取该query_point所有原子类型的3D向量，计算范数
                vec = pred_field[b_idx, idx]  # [n_atom_types, 3]
                norm = torch.norm(vec, dim=-1)  # [n_atom_types]
                norms.append(norm.detach().cpu().numpy())
            fabric.print(f"[Debug] 5个query_point的vector field范数: {norms}")

            target_norms = []
            rmsds = []
            for idx in idxs:
                # 确保target_field有正确的维度
                if target_field.dim() == 4:
                    target_vec = target_field[b_idx, idx]  # [n_atom_types, 3]
                elif target_field.dim() == 3:
                    target_vec = target_field[b_idx, idx]  # [n_atom_types, 3]
                else:
                    fabric.print(f"[Debug] Unexpected target_field dimension: {target_field.dim()}")
                    continue
                    
                target_norm = torch.norm(target_vec, dim=-1)  # [n_atom_types]
                target_norms.append(target_norm.detach().cpu().numpy())
                # 计算RMSD
                pred_vec = pred_field[b_idx, idx]  # [n_atom_types, 3]
                rmsd = compute_rmsd(pred_vec, target_vec)
                rmsds.append(rmsd.item() if hasattr(rmsd, 'item') else float(rmsd))
            fabric.print(f"[Debug] 5个query_point的target field标准答案范数: {target_norms}")
            fabric.print(f"[Debug] 5个query_point的vector field与target field RMSD: {rmsds}")

        # 计算损失
        loss = criterion(pred_field, target_field)
        
        # TensorBoard 日志记录
        if tensorboard_writer is not None and TENSORBOARD_AVAILABLE and fabric.global_rank == 0:
            current_step = global_step + i
            tensorboard_writer.add_scalar('Loss/Batch', loss.item(), current_step)
            tensorboard_writer.add_scalar('Loss/Running_Average', running_loss, current_step)
            
            # 记录学习率
            if hasattr(optim_dec, 'param_groups') and len(optim_dec.param_groups) > 0:
                tensorboard_writer.add_scalar('Learning_Rate/Decoder', optim_dec.param_groups[0]['lr'], current_step)
            if hasattr(optim_enc, 'param_groups') and len(optim_enc.param_groups) > 0:
                tensorboard_writer.add_scalar('Learning_Rate/Encoder', optim_enc.param_groups[0]['lr'], current_step)
            
            # 每100步记录一次更详细的指标
            if current_step % 100 == 0:
                # 记录预测场和目标场的统计信息
                pred_field_norm = torch.norm(pred_field, dim=-1).mean().item()
                target_field_norm = torch.norm(target_field, dim=-1).mean().item()
                tensorboard_writer.add_scalar('Field_Stats/Pred_Field_Norm', pred_field_norm, current_step)
                tensorboard_writer.add_scalar('Field_Stats/Target_Field_Norm', target_field_norm, current_step)
                
                # 记录模型参数统计信息
                for name, param in dec.named_parameters():
                    if param.grad is not None:
                        tensorboard_writer.add_histogram(f'Decoder_Params/{name}', param.data, current_step)
                        tensorboard_writer.add_histogram(f'Decoder_Grads/{name}', param.grad.data, current_step)
                
                for name, param in enc.named_parameters():
                    if param.grad is not None:
                        tensorboard_writer.add_histogram(f'Encoder_Params/{name}', param.data, current_step)
                        tensorboard_writer.add_histogram(f'Encoder_Grads/{name}', param.grad.data, current_step)
        
        # 反向传播
        optim_dec.zero_grad()
        optim_enc.zero_grad()
        fabric.backward(loss)
        
        # TensorBoard 记录梯度范数（在梯度裁剪前）
        if tensorboard_writer is not None and TENSORBOARD_AVAILABLE and fabric.global_rank == 0:
            current_step = global_step + i
            if current_step % 100 == 0:
                dec_grad_norm = torch.nn.utils.clip_grad_norm_(dec.parameters(), float('inf'), norm_type=2)
                enc_grad_norm = torch.nn.utils.clip_grad_norm_(enc.parameters(), float('inf'), norm_type=2)
                tensorboard_writer.add_scalar('Gradients/Decoder_Norm', dec_grad_norm, current_step)
                tensorboard_writer.add_scalar('Gradients/Encoder_Norm', enc_grad_norm, current_step)
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(dec.parameters(), max_grad_norm)
        torch.nn.utils.clip_grad_norm_(enc.parameters(), max_grad_norm)
        optim_dec.step()
        optim_enc.step()
        total_loss += loss.item() #  total loss 是 MSE 损失（预测的向量场与目标向量场之间的均方误差）
        
        # 更新运行中的loss平均值
        running_loss = (running_loss * i + loss.item()) / (i + 1)
        
        # 更新进度条
        pbar.set_postfix({
            'loss': f'{running_loss:.4f}',
            'batch': f'{i}/{len(loader)}'
        })
        
        # 更新指标
        if metrics is not None:
            metrics["loss"].update(loss)
        if batch_train_losses is not None:
            batch_train_losses.append({
                "total_loss": loss.item(),
            })
    
    # 关闭进度条
    pbar.close()
    return metrics["loss"].compute().item() if metrics is not None else total_loss / len(loader)


@torch.no_grad()
def eval_nf(
    loader: torch.utils.data.DataLoader,
    dec: nn.Module,
    enc: nn.Module,
    criterion: nn.Module,
    config: dict,
    gnf_converter: GNFConverter,
    metrics=None,
    fabric=None,
) -> float:
    """
    Evaluates the neural field model.

    Args:
        loader (DataLoader): DataLoader for evaluation data.
        dec (nn.Module): Decoder network.
        enc (nn.Module): Encoder network.
        criterion (nn.Module): Loss function.
        config (dict): Configuration dictionary.
        gnf_converter (GNFConverter): Instance of GNFConverter for field computation.
        metrics (dict, optional): Dictionary of metrics to track. Defaults to None.
        fabric (object, optional): Fabric object for distributed training. Defaults to None.

    Returns:
        float: The computed loss value.
    """
    dec.eval()
    enc.eval()
    if metrics is not None:
        for key in metrics.keys():
            metrics[key].reset()

    for i, data_batch in enumerate(loader):
        data_batch = data_batch.to(fabric.device)

        # 使用 to_dense_batch 转换数据格式以兼容现有函数
        coords, atom_mask = to_dense_batch(data_batch.pos, data_batch.batch, fill_value=0)
        atoms_channel, _ = to_dense_batch(data_batch.x, data_batch.batch, fill_value=PADDING_INDEX)
        B = coords.size(0)
        
        # Reshape query_points to [B, n_points, 3]
        query_points = data_batch.xs.to(fabric.device)
        if query_points.dim() == 2:
            n_points = config["dset"]["n_points"]
            query_points = query_points.view(B, n_points, 3)
        
        # 前向传播
        codes = enc(data_batch)
        pred_field = dec(query_points, codes)
        
        # 使用预计算的目标梯度场（从数据集中获取）
        target_field = data_batch.target_field.to(fabric.device)
        if target_field.dim() == 2:
            target_field = target_field.view(B, n_points, -1, 3)  # [B, n_points, n_atom_types, 3]
        elif target_field.dim() == 3:
            # 如果target_field是[n_points, n_atom_types, 3]，需要添加batch维度
            if target_field.shape[0] == n_points:
                target_field = target_field.unsqueeze(0).expand(B, -1, -1, -1)  # [B, n_points, n_atom_types, 3]
            else:
                # 如果已经是batch格式，确保维度正确
                target_field = target_field.view(B, n_points, -1, 3)
        
        # 确保pred_field和target_field的维度匹配
        if pred_field.shape != target_field.shape:
            fabric.print(f"[Warning] Shape mismatch: pred_field {pred_field.shape} vs target_field {target_field.shape}")
            # 尝试调整target_field的维度以匹配pred_field
            if target_field.shape[1] != pred_field.shape[1]:
                # 如果点数不匹配，取较小的那个
                min_points = min(pred_field.shape[1], target_field.shape[1])
                pred_field = pred_field[:, :min_points, :, :]
                target_field = target_field[:, :min_points, :, :]
        
        # 计算损失
        loss = criterion(pred_field, target_field)
        
        # 更新指标
        metrics["loss"].update(loss)
        
    return metrics["loss"].compute().item()


def set_requires_grad(module: nn.Module, tf: bool = False) -> None:
    """
    Set the requires_grad attribute of a module and its parameters.

    Args:
        module (nn.Module): The module for which requires_grad attribute needs to be set.
        tf (bool, optional): The value to set for requires_grad attribute. Defaults to False.
    """
    module.requires_grad = tf
    for param in module.parameters():
        param.requires_grad = tf

# TODO: 还没有改动的旧逻辑。这个函数在train_nf.py和train_fm.py中都被调用，但是train_nf.py中没有is_compile参数，train_fm.py中没有sd参数
def load_network(
    checkpoint: dict,
    net: nn.Module,
    fabric: object,
    net_name: str = "dec",
    is_compile: bool = True,
    sd: str = None,
) -> nn.Module:
    """
    Load a neural network's state dictionary from a checkpoint and update the network's parameters.

    Args:
        checkpoint (dict): A dictionary containing the checkpoint data.
        net (nn.Module): The neural network model to load the state dictionary into.
        fabric (object): An object with a print method for logging.
        net_name (str, optional): The key name for the network's state dictionary in the checkpoint. Defaults to "dec".
        is_compile (bool, optional): A flag indicating whether the network is compiled. Defaults to True.
        sd (str, optional): A specific key for the state dictionary in the checkpoint. If None, defaults to using net_name.

    Returns:
        nn.Module: The neural network model with the loaded state dictionary.
    """
    net_dict = net.state_dict()
    weight_first_layer_before = next(iter(net_dict.values())).sum()
    new_state_dict = OrderedDict()
    key = f"{net_name}_state_dict" if sd is None else sd
    for k, v in checkpoint[key].items():
        if sd is not None:
            k = k[17:] if k[:17] == "_orig_mod.module." else k
        else:
            k = k[10:] if k[:10] == "_orig_mod." and not is_compile else k  # remove compile prefix.
        new_state_dict[k] = v

    pretrained_dict = {k: v for k, v in new_state_dict.items() if k in net_dict}
    net_dict.update(pretrained_dict)
    net.load_state_dict(net_dict)

    # weight_first_layer_after = next(iter(net_dict.values())).sum()
    # assert (weight_first_layer_before != weight_first_layer_after).item(), "loading did not work"
    # 现在的 net_dict 是一个dict，net_dict.keys()的第0个元素是"layers.0.weight"，对应的是grid_coords，它本来就不会更新，所以这里的assert一定会报错
    fabric.print(f">> loaded {net_name}")

    return net


def load_optim_fabric(
    optim: torch.optim.Optimizer,
    checkpoint: dict,
    config: dict,
    fabric: object,
    net_name: str = "dec"
) -> torch.optim.Optimizer:
    """
    Loads the optimizer state from a checkpoint and updates the learning rate.
    Args:
        optim (torch.optim.Optimizer): The optimizer to be loaded.
        checkpoint (dict): The checkpoint containing the optimizer state.
        config (dict): Configuration dictionary containing learning rate information.
        fabric (object): An object with a print method for logging.
        net_name (str, optional): The name of the network. Defaults to "dec".
    Returns:
        torch.optim.Optimizer: The optimizer with the loaded state and updated learning rate.
    """
    fabric.print(f"optim_{net_name} ckpt", checkpoint[f"optim_{net_name}"]["param_groups"])
    optim.load_state_dict(checkpoint[f"optim_{net_name}"])
    for g in optim.param_groups:
        g["lr"] = config["dset"][f"lr_{net_name}"]
    fabric.print(f"Loaded optim_{net_name}")
    return optim


def save_checkpoint(
    epoch: int,
    config: dict,
    loss_tot: float,
    loss_min_tot: float,
    enc: nn.Module,
    dec: nn.Module,
    optim_enc: torch.optim.Optimizer,
    optim_dec: torch.optim.Optimizer,
    fabric: object,
)-> float:
    """
    Saves a model checkpoint if the current total loss is less than the minimum total loss.

    Args:
        epoch (int): The current epoch number.
        config (dict): Configuration dictionary containing model and training parameters.
        loss_tot (float): The current total loss.
        loss_min_tot (float): The minimum total loss encountered so far.
        enc (nn.Module): The encoder model.
        dec (nn.Module): The decoder model.
        optim_enc (torch.optim.Optimizer): The optimizer for the encoder.
        optim_dec (torch.optim.Optimizer): The optimizer for the decoder.
        fabric (object): An object responsible for saving the model state.

    Returns:
        float: The updated minimum total loss.
    """

    if loss_min_tot is None or loss_tot < loss_min_tot:
        if loss_min_tot is not None:
            loss_min_tot = loss_tot
        try:
            state = {
                "epoch": epoch,
                "dec_state_dict": dec.state_dict(),
                "enc_state_dict": enc.state_dict(),
                "optim_dec": optim_dec.state_dict(),
                "optim_enc": optim_enc.state_dict(),
                "config": config,
            }
            fabric.save(os.path.join(config["dirname"], "model.pt"), state)
            fabric.print(">> saved checkpoint")
        except Exception as e:
            fabric.print(f"Error saving checkpoint: {e}")
    return loss_min_tot


def auto_load_latest_checkpoint(
    config: dict,
    enc: nn.Module,
    dec: nn.Module,
    optim_enc: torch.optim.Optimizer,
    optim_dec: torch.optim.Optimizer,
    fabric: object
) -> tuple:
    """
    自动加载最新的checkpoint并返回训练状态
    
    Args:
        config (dict): 配置字典
        enc (nn.Module): 编码器
        dec (nn.Module): 解码器
        optim_enc (torch.optim.Optimizer): 编码器优化器
        optim_dec (torch.optim.Optimizer): 解码器优化器
        fabric (object): Fabric对象
        
    Returns:
        tuple: (enc, dec, optim_enc, optim_dec, start_epoch, train_losses, val_losses, best_loss)
    """
    try:
        # 查找所有以nf_qm9_开头的目录
        exp_dir = Path("exps/neural_field")
        if not exp_dir.exists():
            fabric.print(">> No exps/neural_field directory found, starting fresh training")
            return enc, dec, optim_enc, optim_dec, 0, [], [], float("inf")
        
        # 查找所有以nf_qm9_开头的目录
        exp_dirs = [d for d in exp_dir.iterdir() if d.is_dir() and d.name.startswith("nf_qm9_")]
        if not exp_dirs:
            fabric.print(">> No existing experiment directories found, starting fresh training")
            return enc, dec, optim_enc, optim_dec, 0, [], [], float("inf")
        
        # 查找有model.pt文件的目录
        valid_dirs = []
        for d in exp_dirs:
            model_path = d / "model.pt"
            if model_path.exists():
                valid_dirs.append(d)
        
        if not valid_dirs:
            fabric.print(">> No existing checkpoints found, starting fresh training")
            return enc, dec, optim_enc, optim_dec, 0, [], [], float("inf")
        
        # 选择最新的有checkpoint的目录
        latest_model_path = max(valid_dirs, key=lambda x: x.stat().st_mtime)
        checkpoint_path = latest_model_path / "model.pt"
        
        fabric.print(f">> Auto-loaded latest checkpoint: {latest_model_path}")
        
        # 加载checkpoint
        checkpoint = fabric.load(str(checkpoint_path))
        
        # 加载模型和优化器状态
        dec = load_network(checkpoint, dec, fabric, net_name="dec")
        optim_dec = load_optim_fabric(optim_dec, checkpoint, config, fabric, net_name="dec")
        
        enc = load_network(checkpoint, enc, fabric, net_name="enc")
        optim_enc = load_optim_fabric(optim_enc, checkpoint, config, fabric, net_name="enc")
        
        # 设置起始epoch
        start_epoch = checkpoint.get("epoch", 0) + 1
        fabric.print(f">> Resuming from epoch {start_epoch}")
        
        # 加载训练历史
        train_losses = []
        if os.path.exists(os.path.join(latest_model_path, "train_losses.npy")):
            train_losses = list(np.load(os.path.join(latest_model_path, "train_losses.npy")))
            fabric.print(f">> Loaded {len(train_losses)} previous training losses")
        
        val_losses = []
        if os.path.exists(os.path.join(latest_model_path, "val_losses.npy")):
            val_losses = list(np.load(os.path.join(latest_model_path, "val_losses.npy")))
            fabric.print(f">> Loaded {len(val_losses)} previous validation losses")
        
        # 更新最佳loss
        best_loss = checkpoint.get("best_loss", float("inf"))
        
        return enc, dec, optim_enc, optim_dec, start_epoch, train_losses, val_losses, best_loss
        
    except Exception as e:
        fabric.print(f"Error auto-loading checkpoint: {e}")
        fabric.print(">> Starting fresh training")
        return enc, dec, optim_enc, optim_dec, 0, [], [], float("inf")


def load_neural_field(nf_checkpoint: dict, fabric: object, config: dict = None) -> tuple:
    """
    Load and initialize the neural field encoder and decoder from a checkpoint.

    Args:
        nf_checkpoint (dict): The checkpoint containing the saved state of the neural field model.
        fabric (object): The fabric object used for setting up the modules.
        config (dict, optional): Configuration dictionary for initializing the encoder and decoder.
                                 If None, the configuration from the checkpoint will be used.

    Returns:
        tuple: A tuple containing the initialized encoder and decoder modules.
    """
    if config is None:
        config = nf_checkpoint["config"]
    
    # Initialize the decoder
    dec = Decoder({
        "grid_size": config["dset"]["grid_size"],
        "anchor_spacing": config["dset"]["anchor_spacing"],
        "hidden_dim": config["decoder"]["hidden_dim"],
        "n_layers": config["decoder"]["n_layers"],
        "k_neighbors": config["decoder"]["k_neighbors"],
        "n_channels": config["dset"]["n_channels"],
        "code_dim": config["decoder"]["code_dim"]
    }, device=fabric.device)
    dec = load_network(nf_checkpoint, dec, fabric, net_name="dec")
    dec = torch.compile(dec)
    dec.eval()
    
    enc = CrossGraphEncoder(
        n_atom_types=config["dset"]["n_channels"],
        grid_size=config["dset"]["grid_size"],
        code_dim=config["encoder"]["code_dim"],
        hidden_dim=config["encoder"]["hidden_dim"],
        num_layers=config["encoder"]["num_layers"],
        k_neighbors=config["encoder"]["k_neighbors"],
        atom_k_neighbors=config["encoder"]["atom_k_neighbors"],
        anchor_spacing=config["dset"]["anchor_spacing"]
    )
    enc = load_network(nf_checkpoint, enc, fabric, net_name="enc")
    # Disable torch.compile for encoder due to torch_cluster compatibility issues
    # enc = torch.compile(enc)
    enc.eval()

    dec = fabric.setup_module(dec)
    enc = fabric.setup_module(enc)

    return enc, dec


def normalize_code(codes: torch.Tensor, code_stats: dict) -> torch.Tensor:
    """
    Normalize codes using mean and std from code_stats.

    Args:
        codes (torch.Tensor): The input codes to be normalized.
        code_stats (dict): A dictionary containing 'mean' and 'std' for normalization.

    Returns:
        torch.Tensor: The normalized codes.
    """
    codes = (codes - code_stats["mean"]) / code_stats["std"]
    return codes


def compute_rmsd(coords1, coords2):
    """
    Calculate symmetric RMSD between two sets of coordinates.
    
    Args:
        coords1 (torch.Tensor): First set of coordinates
        coords2 (torch.Tensor): Second set of coordinates
        
    Returns:
        torch.Tensor: RMSD value
    """
    # 确保输入张量保持梯度
    coords1 = coords1.detach().requires_grad_(True)
    coords2 = coords2.detach().requires_grad_(True)
    
    # 计算距离（不是平方距离）
    dist1 = torch.sqrt(torch.sum((coords1.unsqueeze(1) - coords2.unsqueeze(0))**2, dim=2) + 1e-8)
    dist2 = torch.sqrt(torch.sum((coords2.unsqueeze(1) - coords1.unsqueeze(0))**2, dim=2) + 1e-8)
    
    # 对距离取min
    min_dist1 = torch.min(dist1, dim=1)[0]  # 对于coords1中的每个点，找到最近的coords2中的点
    min_dist2 = torch.min(dist2, dim=1)[0]  # 对于coords2中的每个点，找到最近的coords1中的点
    
    # 直接平均，不需要再开方
    rmsd = (torch.mean(min_dist1) + torch.mean(min_dist2)) / 2
    
    return rmsd


def create_tensorboard_writer(log_dir: str, experiment_name: str = None) -> SummaryWriter:
    """
    Create a TensorBoard SummaryWriter for logging training metrics.
    
    Args:
        log_dir (str): Directory to save TensorBoard logs
        experiment_name (str, optional): Name of the experiment. If None, uses timestamp.
        
    Returns:
        SummaryWriter: TensorBoard writer instance, or None if TensorBoard is not available
    """
    if not TENSORBOARD_AVAILABLE:
        print("Warning: TensorBoard is not available. Install with: pip install tensorboard")
        return None
    
    if experiment_name is None:
        import datetime
        experiment_name = f"experiment_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    log_path = os.path.join(log_dir, experiment_name)
    os.makedirs(log_path, exist_ok=True)
    
    writer = SummaryWriter(log_path)
    print(f"TensorBoard logs will be saved to: {log_path}")
    print(f"To view logs, run: tensorboard --logdir {log_path}")
    
    return writer

def infer_codes(
    loader: torch.utils.data.DataLoader,
    enc: torch.nn.Module,
    config: dict,
    fabric = None,
    to_cpu: bool = True,
    code_stats=None,
    n_samples=None,
) -> torch.Tensor:
    """
    Infer codes from a data loader using the specified encoder model.

    Args:
        loader (torch.utils.data.DataLoader): DataLoader providing batches of data
        enc (torch.nn.Module): Encoder model for inferring codes
        config (dict): Configuration dictionary containing model and dataset parameters
        fabric: Optional fabric object for distributed training (default: None)
        to_cpu (bool): Flag indicating whether to move inferred codes to CPU (default: True)
        code_stats (dict, optional): Statistics for code normalization (default: None)
        n_samples (int, optional): Number of samples to infer codes for. If None, infer for all samples (default: None)

    Returns:
        torch.Tensor: Tensor containing the inferred codes
    """
    enc.eval()
    codes_all = []
    
    if fabric is not None:
        fabric.print(f">> Inferring codes - batch size: {config['dset']['batch_size']}")
    
    total_samples = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            # Directly use encoder to process batch data
            codes = enc(batch)
            
            # Normalize codes if needed
            if code_stats is not None:
                codes = normalize_code(codes, code_stats)
            
            # Move to CPU if needed
            if to_cpu:
                codes = codes.cpu()
            
            # Gather codes in distributed environment
            if fabric is not None:
                codes = fabric.all_gather(codes).view(-1, config["decoder"]["code_dim"])
            
            codes_all.append(codes)
            total_samples += codes.size(0)
            
            # Stop if we've reached the specified number of samples
            if n_samples is not None and total_samples >= n_samples:
                break
    
    # Concatenate all codes
    codes = torch.cat(codes_all, dim=0)
    
    # Truncate to specified number if needed
    if n_samples is not None and codes.size(0) > n_samples:
        codes = codes[:n_samples]
    
    if fabric is not None:
        fabric.print(f">> Inference completed, generated {codes.size(0)} codes in total")
    
    return codes


def infer_codes_occs_batch(batch, enc, config, to_cpu=False, code_stats=None):
    """
    Infer codes for a batch of data.

    Args:
        batch (torch_geometric.data.Batch): Input data batch
        enc (torch.nn.Module): Encoder model
        config (dict): Configuration dictionary
        to_cpu (bool, optional): If True, move codes to CPU. Defaults to False
        code_stats (dict, optional): Statistics for code normalization. Defaults to None

    Returns:
        torch.Tensor: Inferred codes
    """
    # Directly use encoder to process batch
    codes = enc(batch)
    
    # Normalize codes if needed
    if code_stats is not None:
        codes = normalize_code(codes, code_stats)
    
    # Move to CPU if needed
    if to_cpu:
        codes = codes.cpu()
    
    return codes