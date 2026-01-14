import os
import numpy as np
import torch
import torch.nn.functional as F
from collections import OrderedDict
from tqdm import tqdm
from torch import nn
from funcmol.utils.constants import PADDING_INDEX
from funcmol.utils.gnf_converter import GNFConverter
from torch_geometric.utils import to_dense_batch
from funcmol.models.encoder import CrossGraphEncoder
from funcmol.models.decoder import Decoder
import time
from pathlib import Path
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    SummaryWriter = None



def create_neural_field(config: dict) -> tuple:
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
                - "dset": A dictionary with the following keys:
                    - "n_channels" (int): The number of input channels.
                    - "grid_dim" (int): The grid dimension of the dataset.
                - "encoder": A dictionary with the following keys:
                    - "level_channels" (list of int): The number of channels at each level of the encoder.
                    - "smaller" (bool, optional): A flag indicating whether to use a smaller encoder. Defaults to False.

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
        "radius": config["decoder"].get("radius", 3.0),
        "cutoff": config["decoder"].get("cutoff", None),  # 如果是None，则使用radius作为cutoff
        "n_channels": config["dset"]["n_channels"],
        "code_dim": config["decoder"]["code_dim"]
    })

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
    
    # 清理CUDA缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        fabric.print(f"CUDA memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        fabric.print(f"CUDA memory cached: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    
    # 创建进度条
    pbar = tqdm(enumerate(loader), total=len(loader), desc=f'Epoch {epoch}')
    running_loss = 0.0
    
    # 初始化时间跟踪变量
    last_time = time.time()
    last_step = global_step
    
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
        # if i % 2000 == 0 and fabric.global_rank == 0:
        #     b_idx = 0  # 只看第一个样本
        #     n_points = pred_field.shape[1]
        #     
        #     max_idx = min(n_points - 1, target_field.shape[1] - 1) if target_field.dim() >= 2 else 0
        #     if max_idx < 4:  # 如果点数太少，跳过调试
        #         fabric.print(f"[Debug] Not enough points for debugging (max_idx: {max_idx})")
        #         continue
        #         
        #     idxs = random.sample(range(max_idx + 1), min(5, max_idx + 1))
        #     norms = []
        #     for idx in idxs:
        #         # 取该query_point所有原子类型的3D向量，计算范数
        #         vec = pred_field[b_idx, idx]  # [n_atom_types, 3]
        #         norm = torch.norm(vec, dim=-1)  # [n_atom_types]
        #         norms.append(norm.detach().cpu().numpy())
        #     fabric.print(f"[Debug] 5个query_point的vector field范数: {norms}")

        #     target_norms = []
        #     rmsds = []
        #     for idx in idxs:
        #         # 确保target_field有正确的维度
        #         if target_field.dim() == 4:
        #             target_vec = target_field[b_idx, idx]  # [n_atom_types, 3]
        #         elif target_field.dim() == 3:
        #             target_vec = target_field[b_idx, idx]  # [n_atom_types, 3]
        #         else:
        #             fabric.print(f"[Debug] Unexpected target_field dimension: {target_field.dim()}")
        #             continue
        #             
        #         target_norm = torch.norm(target_vec, dim=-1)  # [n_atom_types]
        #         target_norms.append(target_norm.detach().cpu().numpy())
        #         # 计算RMSD
        #         pred_vec = pred_field[b_idx, idx]  # [n_atom_types, 3]
        #         rmsd = compute_rmsd(pred_vec, target_vec)
        #         rmsds.append(rmsd.item() if hasattr(rmsd, 'item') else float(rmsd))
        #     fabric.print(f"[Debug] 5个query_point的target field标准答案范数: {target_norms}")
        #     fabric.print(f"[Debug] 5个query_point的vector field与target field RMSD: {rmsds}")

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
            
            # 每10步记录一次训练速度指标
            if current_step % 10 == 0:
                # 记录训练速度（样本/秒）
                time_diff = time.time() - last_time
                step_diff = current_step - last_step
                if step_diff > 0 and time_diff > 0:
                    samples_per_sec = (step_diff * config.get('dset', {}).get('batch_size', 1)) / time_diff
                    tensorboard_writer.add_scalar('Training/Samples_per_Second', samples_per_sec, current_step)
                
                last_time = time.time()
                last_step = current_step
                
                # 记录GPU内存使用（如果可用）
                if torch.cuda.is_available():
                    gpu_memory_allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
                    gpu_memory_cached = torch.cuda.memory_reserved() / (1024**3)  # GB
                    tensorboard_writer.add_scalar('GPU/Memory_Allocated_GB', gpu_memory_allocated, current_step)
                    tensorboard_writer.add_scalar('GPU/Memory_Cached_GB', gpu_memory_cached, current_step)
            
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
                
                # 记录loss分布统计
                if len(batch_train_losses) > 0:
                    recent_losses = [l["total_loss"] for l in batch_train_losses[-100:]]
                    if recent_losses:
                        tensorboard_writer.add_scalar('Loss/Std_Dev', np.std(recent_losses), current_step)
                        tensorboard_writer.add_scalar('Loss/Min', np.min(recent_losses), current_step)
                        tensorboard_writer.add_scalar('Loss/Max', np.max(recent_losses), current_step)
        
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

    total_loss = 0.0
    num_batches = 0

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
            elif target_field.shape[0] == B * n_points:
                # 如果target_field是[B*n_points, n_atom_types, 3]，需要重新整形
                target_field = target_field.view(B, n_points, -1, 3)
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
        if metrics is not None:
            metrics["loss"].update(loss)
        
        # 累积损失用于分布式平均
        total_loss += loss.item()
        num_batches += 1
    
    # 分布式平均损失
    if fabric is not None:
        # 收集所有GPU的损失和批次数量
        loss_tensor = torch.tensor([total_loss, num_batches], device=fabric.device)
        gathered = fabric.all_gather(loss_tensor)
        
        # 计算全局平均损失
        total_loss_all = gathered[:, 0].sum().item()
        num_batches_all = gathered[:, 1].sum().item()
        avg_loss = total_loss_all / num_batches_all if num_batches_all > 0 else 0.0
        
        return avg_loss
    else:
        # 单GPU情况
        return total_loss / num_batches if num_batches > 0 else 0.0


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

# NOTE：这个函数要保留，load_nf和load_fm都调用到了这个函数，是load时的底层逻辑
# 它不能和load_neural_field函数合并
def load_network(
    checkpoint: dict,
    net: nn.Module,
    net_name: str = "dec",
    is_compile: bool = True,
    sd: str = None,
) -> nn.Module:
    """
    Load a neural network's state dictionary from a checkpoint and update the network's parameters.

    Args:
        checkpoint (dict): A dictionary containing the checkpoint data.
        net (nn.Module): The neural network model to load the state dictionary into.
        net_name (str, optional): The key name for the network's state dictionary in the checkpoint. Defaults to "dec".
        is_compile (bool, optional): A flag indicating whether the network is compiled. Defaults to True.
        sd (str, optional): A specific key for the state dictionary in the checkpoint. If None, defaults to using net_name.

    Returns:
        nn.Module: The neural network model with the loaded state dictionary.
    """
    net_dict = net.state_dict()
    # weight_first_layer_before = next(iter(net_dict.values())).sum()
    new_state_dict = OrderedDict()
    key = f"{net_name}_state_dict" if sd is None else sd
    
    # 之前fm的模型权重没有成功加载，pretrained_dict是空的
    if key not in checkpoint:
        print(f"Warning: key '{key}' not found in checkpoint!")
        return net
    
    for k, v in checkpoint[key].items():
        # 统一去掉 _orig_mod. 前缀
        if k.startswith("_orig_mod."):
            k = k[10:]  # 移除 "_orig_mod." 前缀
        
        # 处理其他前缀
        if sd is not None:
            # 处理不同的前缀
            if k.startswith("_orig_mod.module."):
                k = k[17:]  # 移除 "_orig_mod.module." 前缀
            elif k.startswith("module."):
                k = k[7:]   # 移除 "module." 前缀
        else:
            k = k[10:] if k.startswith("_orig_mod.") and not is_compile else k  # remove compile prefix.
        
        # 将enc.前缀转换为net.前缀（用于FuncMol模型）
        if k.startswith('enc.') and net_name == "denoiser":
            k = k.replace('enc.', 'net.', 1)
        
        new_state_dict[k] = v

    pretrained_dict = {k: v for k, v in new_state_dict.items() if k in net_dict}
    net_dict.update(pretrained_dict)
    net.load_state_dict(net_dict)

    # weight_first_layer_after = next(iter(net_dict.values())).sum()
    # assert (weight_first_layer_before != weight_first_layer_after).item(), "loading did not work"
    # 现在的 net_dict 是一个dict，net_dict.keys()的第0个元素是"layers.0.weight"，对应的是grid_coords，它本来就不会更新，所以这里的assert一定会报错
    
    print(f">> loaded {net_name}")

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
            # 保存前去掉多余的封装
            dec_state_dict = dec.module.state_dict() if hasattr(dec, "module") else dec.state_dict()
            enc_state_dict = enc.module.state_dict() if hasattr(enc, "module") else enc.state_dict()
            
            state = {
                "epoch": epoch,
                "dec_state_dict": dec_state_dict,
                "enc_state_dict": enc_state_dict,
                "optim_dec": optim_dec.state_dict(),
                "optim_enc": optim_enc.state_dict(),
                "config": config,
            }
            fabric.save(os.path.join(config["dirname"], "model.pt"), state)
            fabric.print(">> saved checkpoint")
        except Exception as e:
            fabric.print(f"Error saving checkpoint: {e}")
    return loss_min_tot


def load_neural_field(nf_checkpoint_or_path, config: dict = None) -> tuple:
    """
    Load and initialize the neural field encoder and decoder from a Lightning checkpoint.
    
    This function supports two calling patterns:
    1. load_neural_field(nf_checkpoint, fabric, config) - original pattern
    2. load_neural_field(model_path, fabric, config) - new pattern from load_model

    Args:
        nf_checkpoint_or_path: Either a dict containing the checkpoint data, or a string path to a .ckpt file.
        config (dict, optional): Configuration dictionary for initializing the encoder and decoder.
                                 If None, the configuration from the checkpoint will be used.

    Returns:
        tuple: A tuple containing the initialized encoder and decoder modules.
    """
    # Handle model_path string input (from load_model functionality)
    if isinstance(nf_checkpoint_or_path, str):
        model_path = Path(nf_checkpoint_or_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        print(f"Loading Lightning checkpoint from: {model_path}")
        
        # Load Lightning checkpoint
        checkpoint = torch.load(str(model_path), map_location='cpu', weights_only=False)
        
        # Extract config from checkpoint if not provided
        if config is None:
            config = checkpoint.get('hyper_parameters', {})
        
        # Create nf_checkpoint in expected format
        nf_checkpoint = {
            'enc_state_dict': checkpoint["enc_state_dict"],
            'dec_state_dict': checkpoint["dec_state_dict"],
            'config': config
        }
    else:
        # Original nf_checkpoint dict input
        nf_checkpoint = nf_checkpoint_or_path
    # 处理Lightning checkpoint格式
    if 'state_dict' in nf_checkpoint:
        # Lightning checkpoint格式
        state_dict = nf_checkpoint['state_dict']
        if config is None:
            config = nf_checkpoint.get('hyper_parameters', {})
        
        # 分离encoder和decoder的state dict
        enc_state_dict = {}
        dec_state_dict = {}
        
        for key, value in state_dict.items():
            if key.startswith('enc.'):
                # 移除'enc.'前缀
                new_key = key[4:]  # 去掉'enc.'
                enc_state_dict[new_key] = value
            elif key.startswith('dec.'):
                # 移除'dec.'前缀
                new_key = key[4:]  # 去掉'dec.'
                dec_state_dict[new_key] = value
        
        # 构建兼容格式
        nf_checkpoint = {
            'enc_state_dict': enc_state_dict,
            'dec_state_dict': dec_state_dict,
            'config': config
        }
    else:
        # 旧格式checkpoint（向后兼容）
        if config is None:
            config = nf_checkpoint["config"]
    
    # Initialize the decoder
    enc, dec = create_neural_field(config)
    dec = load_network(nf_checkpoint, dec, net_name="dec")
    # Disable torch.compile due to compatibility issues with torch_cluster
    # dec = torch.compile(dec)
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
    enc = load_network(nf_checkpoint, enc, net_name="enc")
    # Disable torch.compile for encoder due to torch_cluster compatibility issues
    # enc = torch.compile(enc)
    enc.eval()

    # Handle _orig_mod (from load_model functionality)
    if hasattr(enc, '_orig_mod'):
        enc = enc._orig_mod
    if hasattr(dec, '_orig_mod'):
        dec = dec._orig_mod

    # For Lightning modules, we don't need setup_module as Lightning handles device placement

    print("Model loaded successfully!")
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


def unnormalize_code(codes: torch.Tensor, code_stats: dict) -> torch.Tensor:
    """
    Unnormalize codes using mean and std from code_stats.
    This is the inverse operation of normalize_code.

    Args:
        codes (torch.Tensor): The normalized codes to be unnormalized.
        code_stats (dict): A dictionary containing 'mean' and 'std' for unnormalization.

    Returns:
        torch.Tensor: The unnormalized codes.
    """
    codes = codes * code_stats["std"] + code_stats["mean"]
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
        fabric: Optional fabric object for distributed training (can be Lightning module or Fabric, default: None)
        to_cpu (bool): Flag indicating whether to move inferred codes to CPU (default: True)
        code_stats (dict, optional): Statistics for code normalization (default: None)
        n_samples (int, optional): Number of samples to infer codes for. If None, infer for all samples (default: None)

    Returns:
        torch.Tensor: Tensor containing the inferred codes
    """
    enc.eval()
    codes_all = []
    
    if fabric is not None and hasattr(fabric, 'print'):
        fabric.print(f">> Inferring codes - batch size: {config['dset']['batch_size']}")
    elif fabric is not None:
        print(f">> Inferring codes - batch size: {config['dset']['batch_size']}")
    
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
            
            # Gather codes in distributed environment - 保持原始形状
            if fabric is not None and hasattr(fabric, 'all_gather'):
                codes = fabric.all_gather(codes)
                # 不进行reshape，保持原始形状 [B, n_grid, code_dim]
            
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
    
    if fabric is not None and hasattr(fabric, 'print'):
        fabric.print(f">> Inference completed, generated {codes.size(0)} codes in total")
    elif fabric is not None:
        print(f">> Inference completed, generated {codes.size(0)} codes in total")
    
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


def load_checkpoint_state_nf(model, checkpoint_path):
    """
    Helper function to load checkpoint state for Neural Field model from Lightning checkpoint.
    
    Args:
        model: The Neural Field Lightning module
        checkpoint_path (str): Path to the Lightning checkpoint file (.ckpt)
        
    Returns:
        dict: Training state dictionary containing epoch, losses, and best_loss
    """
    print(f"Loading Lightning checkpoint from: {checkpoint_path}")
    
    # Load checkpoint file
    state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Lightning checkpoint format (contains state_dict with enc.xxx and dec.xxx keys)
    if "state_dict" not in state_dict:
        raise ValueError("Invalid Lightning checkpoint format: missing 'state_dict' key")
    
    lightning_state_dict = state_dict["state_dict"]
    
    # First, try to use enc_state_dict and dec_state_dict if available (saved by on_save_checkpoint)
    if "enc_state_dict" in state_dict and "dec_state_dict" in state_dict:
        print("Found enc_state_dict and dec_state_dict in checkpoint (saved by on_save_checkpoint)")
        enc_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict["enc_state_dict"].items()}
        dec_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict["dec_state_dict"].items()}
    else:
        # Extract encoder and decoder state dicts from Lightning state_dict format
        enc_state_dict = {}
        dec_state_dict = {}
        
        for key, value in lightning_state_dict.items():
            if key.startswith("enc."):
                # Remove 'enc.' prefix
                new_key = key[4:]  # Remove 'enc.'
                # Also remove _orig_mod. prefix if present
                new_key = new_key.replace("_orig_mod.", "")
                enc_state_dict[new_key] = value
            elif key.startswith("dec."):
                # Remove 'dec.' prefix
                new_key = key[4:]  # Remove 'dec.'
                # Also remove _orig_mod. prefix if present
                new_key = new_key.replace("_orig_mod.", "")
                dec_state_dict[new_key] = value
    
    # Load encoder
    if enc_state_dict:
        model.enc.load_state_dict(enc_state_dict, strict=False)
        print(f"Loaded encoder state from Lightning checkpoint ({len(enc_state_dict)} parameters)")
    else:
        raise ValueError("No encoder state found in Lightning checkpoint")
    
    # Load decoder
    if dec_state_dict:
        model.dec.load_state_dict(dec_state_dict, strict=False)
        print(f"Loaded decoder state from Lightning checkpoint ({len(dec_state_dict)} parameters)")
    else:
        raise ValueError("No decoder state found in Lightning checkpoint")
    
    # Load training state
    # For Lightning checkpoints, epoch might be in different locations
    epoch = None
    if "epoch" in state_dict:
        epoch = state_dict["epoch"]
    elif "global_step" in state_dict:
        # Some Lightning checkpoints use global_step instead of epoch
        epoch = state_dict.get("global_step", None)
    
    training_state = {
        "epoch": epoch,
        "train_losses": state_dict.get("train_losses", []),
        "val_losses": state_dict.get("val_losses", []),
        "best_loss": state_dict.get("best_loss", float("inf"))
    }
    
    return training_state


def reshape_target_field(target_field, B, n_points, n_atom_types):
    """
    Reshape target_field to [B, n_points, n_atom_types, 3].
    Reused from train_nf_lt.py _process_batch logic.
    
    Args:
        target_field: Target field tensor with various possible shapes
        B: Batch size
        n_points: Number of query points
        n_atom_types: Number of atom types
    Returns:
        Reshaped target_field [B, n_points, n_atom_types, 3]
    """
    if target_field.dim() == 2:
        # [B*n_points*n_atom_types*3] -> [B, n_points, n_atom_types, 3]
        target_field = target_field.view(B, n_points, n_atom_types, 3)
    elif target_field.dim() == 3:
        if target_field.shape[0] == B * n_points:
            # [B*n_points, n_atom_types, 3] -> [B, n_points, n_atom_types, 3]
            target_field = target_field.view(B, n_points, n_atom_types, 3)
        elif target_field.shape[0] == n_points:
            # [n_points, n_atom_types, 3] -> [B, n_points, n_atom_types, 3]
            target_field = target_field.unsqueeze(0).expand(B, -1, -1, -1)
        else:
            # Assume it's already [B, n_points, n_atom_types, 3] or try to reshape
            target_field = target_field.view(B, n_points, n_atom_types, 3)
    elif target_field.dim() == 4:
        # Already in correct shape [B, n_points, n_atom_types, 3]
        pass
    else:
        raise ValueError(f"Unexpected target_field dimension: {target_field.dim()}, shape: {target_field.shape}")
    return target_field


def compute_decoder_field_loss(
    pred_field,
    target_field,
    use_cosine_loss=True,
    magnitude_loss_weight=0.1,
    valid_mask=None
):
    """
    Compute decoder field loss (cosine distance + magnitude or MSE).
    This is a generic function for computing decoder loss, independent of diffusion models.
    
    Args:
        pred_field: [B, n_points, n_atom_types, 3] predicted field
        target_field: [B, n_points, n_atom_types, 3] target field
        use_cosine_loss: Whether to use cosine distance loss (True) or MSE loss (False)
        magnitude_loss_weight: Weight for magnitude loss when use_cosine_loss=True
        valid_mask: Optional [B, n_points] boolean mask for valid points (None means all points are valid)
        
    Returns:
        loss: Scalar tensor with decoder field loss
    """
    device = pred_field.device
    B, n_points, n_atom_types, _ = pred_field.shape
    
    # Create valid mask if not provided
    if valid_mask is None:
        valid_mask = torch.ones(B, n_points, dtype=torch.bool, device=device)
    
    # Expand mask to match field dimensions
    valid_mask_expanded = valid_mask.unsqueeze(-1).unsqueeze(-1)  # [B, n_points, 1, 1]
    
    # Apply mask
    pred_field_masked = pred_field * valid_mask_expanded
    target_field_masked = target_field * valid_mask_expanded
    
    # Small epsilon for numerical stability
    eps = 1e-8
    
    if use_cosine_loss:
        # Cosine distance + magnitude loss
        # Compute norms
        pred_norm = torch.norm(pred_field_masked, dim=-1, keepdim=True)  # [B, n_points, n_atom_types, 1]
        target_norm = torch.norm(target_field_masked, dim=-1, keepdim=True)  # [B, n_points, n_atom_types, 1]
        
        # Compute cosine similarity
        dot_product = (pred_field_masked * target_field_masked).sum(dim=-1, keepdim=True)  # [B, n_points, n_atom_types, 1]
        
        # Handle zero vectors: if both vectors are zero (or very close to zero), cosine similarity should be 1 (perfect match)
        # Otherwise, compute normal cosine similarity
        both_zero = (pred_norm < eps) & (target_norm < eps)  # [B, n_points, n_atom_types, 1]
        denominator = pred_norm * target_norm + eps  # [B, n_points, n_atom_types, 1]
        cosine_sim_normal = dot_product / denominator  # [B, n_points, n_atom_types, 1]
        cosine_sim = torch.where(
            both_zero,
            torch.ones_like(cosine_sim_normal),  # Both zero vectors: perfect match (cosine_sim = 1)
            cosine_sim_normal  # Normal cosine similarity
        )
        
        # Cosine loss: 1 - cosine_similarity
        cosine_loss = (1 - cosine_sim) * valid_mask_expanded
        cosine_loss = cosine_loss.sum() / (valid_mask_expanded.sum() + eps)
        
        # Magnitude loss: MSE of norms
        magnitude_loss = F.mse_loss(
            pred_norm * valid_mask_expanded,
            target_norm * valid_mask_expanded,
            reduction='sum'
        )
        magnitude_loss = magnitude_loss / (valid_mask_expanded.sum() + eps)
        
        # Combined loss
        loss = cosine_loss + magnitude_loss_weight * magnitude_loss
    else:
        # Standard MSE loss
        loss = F.mse_loss(
            pred_field_masked,
            target_field_masked,
            reduction='sum'
        )
        loss = loss / (valid_mask_expanded.sum() + eps)
    
    return loss