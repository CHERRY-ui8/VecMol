import os
import torch
from collections import OrderedDict, defaultdict
from tqdm import tqdm
from torch import nn
import shutil
from funcmol.utils.constants import PADDING_INDEX
from funcmol.utils.gnf_converter import GNFConverter
from torch_geometric.utils import to_dense_batch
from funcmol.models.encoder import CrossGraphEncoder
from funcmol.models.decoder import Decoder, _normalize_coords, get_atom_coords
from funcmol.utils.utils_base import convert_xyzs_to_sdf, save_xyz
from funcmol.utils.utils_vis import visualize_voxel_grid
import time


def train_nf(
    config: dict,
    loader: torch.utils.data.DataLoader,
    dec: nn.Module,
    optim_dec: torch.optim.Optimizer,
    enc: nn.Module,
    optim_enc: torch.optim.Optimizer,
    criterion: nn.Module,
    fabric: object,
    metrics=None,
    field_maker=None,
    epoch: int = 0,
    batch_train_losses=None, 
    global_step=0, 
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
        metrics (dict, optional): Dictionary of metrics to track. Defaults to None.
        field_maker (object, optional): Field maker object. Defaults to None.
        epoch (int, optional): Current epoch number. Defaults to 0.
        batch_train_losses (list, optional): List to store batch-wise losses. Defaults to None.
        global_step (int, optional): Global training step. Defaults to 0.

    Returns:
        float: The computed loss value.
    """
    dec.train()
    enc.train()
    total_loss = 0.0
    max_grad_norm = 1.0
    
    if fabric.global_rank == 0:
        pbar = tqdm(total=len(loader), desc=f"Training Epoch {epoch}", 
                   bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}',
                   dynamic_ncols=True)
    else:
        pbar = None

    # 创建可视化目录
    if fabric.global_rank == 0:
        vis_dir = os.path.join(config["dirname"], "visualizations", f"epoch_{epoch}")
        os.makedirs(vis_dir, exist_ok=True)

    # 创建GNF转换器（按照项目习惯从yaml配置中读取参数）
    gnf_config = config.get("gnf_converter", {})
    gnf_converter = GNFConverter(
        sigma=gnf_config.get("sigma", 0.9),
        n_query_points=gnf_config.get("n_query_points", 2000),
        n_iter=gnf_config.get("n_iter", 1000),
        step_size=gnf_config.get("step_size", 0.001),
        merge_threshold=gnf_config.get("merge_threshold", 2),
        device=fabric.device
    )
    
    # 用于存储重建结果的变量
    recon_coords = None
    recon_types = None
    
    for i, data_batch in enumerate(loader):
        data_batch = data_batch.to(fabric.device)
        # 1. 准备数据
        # 使用 to_dense_batch 将 torch_geometric Batch 对象转换为填充后的稠密张量
        # 以便与代码库中其他需要固定大小输入的功能兼容
        coords, atom_mask = to_dense_batch(data_batch.pos, data_batch.batch, fill_value=0)
        atoms_channel, _ = to_dense_batch(data_batch.x, data_batch.batch, fill_value=PADDING_INDEX)
        B = coords.size(0)

        # Reshape query_points to [B, n_points, 3]
        query_points = data_batch.xs.to(fabric.device)
        if query_points.dim() == 2:
            n_points = config["dset"]["n_points"]
            query_points = query_points.view(B, n_points, 3)

        # 2. Encoder: 分子图 -> latent code (codes)
        codes = enc(data_batch)  # [B, n_grid, code_dim]

        # 3. Decoder: codes + query_points -> pred_field
        pred_field = dec(query_points, codes)  # [B, n_points, n_atom_types, 3]
        
        target_field = compute_vector_field(
            query_points, 
            coords, 
            atoms_channel, 
            n_atom_types=config["dset"]["n_channels"], 
            device=fabric.device
        )
        
        # 输出5个query_point的梯度场大小（每50个batch输出一次）
        if i % 500 == 0 and fabric.global_rank == 0:
            import random
            b_idx = 0  # 只看第一个样本
            n_points = pred_field.shape[1]
            idxs = random.sample(range(n_points), min(5, n_points))
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
                target_vec = target_field[b_idx, idx]  # [n_atom_types, 3]
                target_norm = torch.norm(target_vec, dim=-1)  # [n_atom_types]
                target_norms.append(target_norm.detach().cpu().numpy())
                # 计算RMSD
                pred_vec = pred_field[b_idx, idx]  # [n_atom_types, 3]
                rmsd = compute_rmsd(pred_vec, target_vec)
                rmsds.append(rmsd.item() if hasattr(rmsd, 'item') else float(rmsd))
            fabric.print(f"[Debug] 5个query_point的target field标准答案范数: {target_norms}")
            fabric.print(f"[Debug] 5个query_point的vector field与target field RMSD: {rmsds}")
            
        # 4. 计算目标矢量场
        target_field = compute_vector_field(
            query_points, 
            coords, 
            atoms_channel, 
            n_atom_types=config["dset"]["n_channels"], 
            device=fabric.device
        )  # [B, n_points, n_atom_types, 3]

        # 5. 计算神经场损失
        # 确保维度匹配
        assert pred_field.shape == target_field.shape, f"Shape mismatch: pred_field {pred_field.shape} vs target_field {target_field.shape}"
        field_loss = criterion(pred_field, target_field)
        
        # 6. 用 GNF转换器 重建分子（每个epoch重建一次，在最后一个batch执行）
        if i == len(loader) - 1:  # 只在每个epoch的最后一个batch执行重建
            # 直接使用现有的gnf2mol函数
            with torch.no_grad():
                recon_coords, recon_types = gnf_converter.gnf2mol(
                    pred_field.detach(),  # [B, n_points, n_atom_types, 3]
                    dec,  # 传入解码器
                    codes,  # 传入编码器的输出
                    atoms_channel  # 传入原子类型
                )
            
            # 7. 计算重建loss（RMSD）
            recon_loss = 0.0
            for b in range(B):
                mask = atoms_channel[b] != PADDING_INDEX
                gt_coords = coords[b, mask]
                pred_coords = recon_coords[b]
                if gt_coords.size(0) > 0 and pred_coords.size(0) > 0:
                    recon_loss += compute_rmsd(gt_coords, pred_coords)
            recon_loss = recon_loss / B
        else:
            recon_loss = 0.0
        
        # 8. 计算总损失（可以调整权重）
        field_weight = config.get("field_loss_weight", 1.0)
        recon_weight = config.get("recon_loss_weight", 1.0)
        loss = field_weight * field_loss + recon_weight * recon_loss

        # 9. 反向传播与优化
        optim_dec.zero_grad()
        optim_enc.zero_grad()
        fabric.backward(loss)
        torch.nn.utils.clip_grad_norm_(dec.parameters(), max_grad_norm)
        torch.nn.utils.clip_grad_norm_(enc.parameters(), max_grad_norm)
        optim_dec.step()
        optim_enc.step()
        
        # 10. 记录与可视化
        total_loss += loss.item()
        if metrics is not None:
            metrics["loss"].update(loss)
            metrics["field_loss"].update(field_loss)
            metrics["recon_loss"].update(recon_loss)
            
        # 记录每个batch的loss
        if batch_train_losses is not None:
            batch_train_losses.append({
                "total_loss": loss.item(),
                "field_loss": field_loss.item(),
                "recon_loss": recon_loss.item() if isinstance(recon_loss, torch.Tensor) else recon_loss,
            })
            
        # 每隔一定步数绘制loss曲线
        if fabric.global_rank == 0 and i == len(loader) - 1:  # 只在每个epoch的最后一个batch绘制
            try:
                from funcmol.train_nf import plot_loss_curve
                # 绘制最近的loss曲线
                recent_losses = [l["total_loss"] for l in batch_train_losses[-config.get("vis_every", 1000)*2:]]
                plot_loss_curve(
                    recent_losses,
                    None,
                    os.path.join(vis_dir, f"loss_curve_epoch_{epoch}.png"),
                    f"(Epoch {epoch})"
                )
                from funcmol.train_nf import plot_field_loss_curve
                plot_field_loss_curve(
                    batch_train_losses,
                    os.path.join(vis_dir, f"field_loss_curve_epoch_{epoch}.png"), 
                    f"(Epoch {epoch})"
                )
            except Exception as e:
                fabric.print(f"Error plotting loss curve: {str(e)}")
            
        if fabric.global_rank == 0:
            pbar.update(1)
            pbar.set_postfix({
                'batch': f'{i+1}/{len(loader)}',  # 显示当前batch数和总batch数
                'loss': f'{loss.item():.4f}', 
                'field_loss': f'{field_loss.item():.4f}',
                'recon_loss': f'{recon_loss.item():.4f}' if isinstance(recon_loss, torch.Tensor) else f'{recon_loss:.4f}',
                'avg_loss': f'{metrics["loss"].compute().item():.4f}'
            })
            pbar.refresh()  # 强制刷新进度条

        # 可视化分子结构（每个epoch的最后一个batch）
        if fabric.global_rank == 0 and i == len(loader) - 1 and recon_coords is not None:
            try:
                # 使用GNF转换器的可视化功能
                gnf_converter.visualize_conversion_results(
                    recon_coords,
                    recon_types,
                    coords,
                    atoms_channel,
                    save_dir=vis_dir,
                    sample_indices=list(range(min(B, 5)))
                )
            except Exception as e:
                fabric.print(f"Visualization failed: {str(e)}")

        fabric.barrier()

    if fabric.global_rank == 0:
        pbar.close()

    return metrics["loss"].compute().item()


@torch.no_grad()
def eval_nf(
    loader: torch.utils.data.DataLoader,
    dec: nn.Module,
    enc: nn.Module,
    criterion: nn.Module,
    config: dict,
    metrics=None,
    save_plot_png=False,
    fabric=None,
    field_maker=None,
    sample_full_grid=False,
) -> float:
    """
    Evaluates the neural field model.

    Args:
        loader (DataLoader): DataLoader for evaluation data.
        dec (nn.Module): Decoder network.
        enc (nn.Module): Encoder network.
        criterion (nn.Module): Loss function.
        config (dict): Configuration dictionary.
        metrics (dict, optional): Dictionary of metrics to track. Defaults to None.
        save_plot_png (bool, optional): Whether to save plots. Defaults to False.
        fabric (object, optional): Fabric object for distributed training. Defaults to None.
        field_maker (object, optional): Field maker object. Defaults to None.
        sample_full_grid (bool, optional): Whether to sample the full grid. Defaults to False.

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
        
        # 获取目标矢量场
        # occs_points, occs_grid = field_maker.forward(batch) # field_maker可能需要适配新的batch格式
        target_field = compute_vector_field(
            query_points, 
            coords, 
            atoms_channel, 
            n_atom_types=config["dset"]["n_channels"], 
            device=fabric.device
        )
        
        # 前向传播
        codes = enc(data_batch)
        pred_field = dec(query_points, codes)
        
        # 计算损失
        loss = criterion(pred_field, target_field)
        
        # 更新指标
        metrics["loss"].update(loss)
        
        # 保存可视化结果
        if save_plot_png and i == 0:
            save_voxel_eval_nf(config, fabric, pred_field, target_field)

    return metrics["loss"].compute().item()


def save_voxel_eval_nf(config, fabric, occs_pred, occs_gt=None, refine=True, i=0, mols_pred=[], mols_gt=[], mols_pred_dict=defaultdict(list), codes=None, codes_dict=defaultdict(list)):
    dirname_voxels = os.path.join(config["dirname"], "res")
    if os.path.exists(dirname_voxels):
        shutil.rmtree(dirname_voxels)
    os.makedirs(dirname_voxels, exist_ok=False)
    fabric.print(f">> saving images in {dirname_voxels}")

    occs_pred = occs_pred.permute(0, 2, 1).reshape(-1, occs_gt.size(2), config["dset"]["grid_dim"], config["dset"]["grid_dim"], config["dset"]["grid_dim"])
    if occs_gt is not None:
        occs_gt = occs_gt.permute(0, 2, 1).reshape(-1, occs_gt.size(2), config["dset"]["grid_dim"], config["dset"]["grid_dim"], config["dset"]["grid_dim"])

    for b in range(occs_pred.size(0)):
        # Predictions
        visualize_voxel_grid(occs_pred[b], os.path.join(dirname_voxels, f"./iter{i}_batch{b}_pred.png"), threshold=0.2, sparse=False)
        mol_init_pred = get_atom_coords(occs_pred[b].cpu(), rad=config["dset"]["atomic_radius"])
        if not refine:
            mol_init_pred["coords"] *= config["dset"]["resolution"]
        if mol_init_pred is not None:
            fabric.print("pred", mol_init_pred["coords"].size())
            if refine:
                mol_init_pred = _normalize_coords(mol_init_pred, config["dset"]["grid_dim"])
                num_coords = int(mol_init_pred["coords"].size(1))
                mols_pred_dict[num_coords].append(mol_init_pred)
                codes_dict[num_coords].append(codes[b])
            else:
                mols_pred.append(mol_init_pred)

        # Ground truth
        if occs_gt is not None:
            visualize_voxel_grid(occs_gt[b], os.path.join(dirname_voxels, f"./iter{i}_batch{b}_gt.png"), threshold=0.2, sparse=False)
            mol_init_gt = get_atom_coords(occs_gt[b].cpu(), rad=config["dset"]["atomic_radius"])
            mol_init_gt["coords"] *= config["dset"]["resolution"]
            if mol_init_gt is not None:
                fabric.print("gt", mol_init_gt["coords"].size())
                mols_gt.append(mol_init_gt)


def save_sdf_eval_nf(config, fabric, mols_pred, mols_gt=None, dec=None, mols_pred_dict=None, codes_dict=None, refine=True):
    dirname_voxels = os.path.join(config["dirname"], "res")

    # prediction
    if refine:
        mols_pred = dec._refine_coords(
            grouped_mol_inits=mols_pred_dict,
            grouped_codes=codes_dict,
            maxiter=200,
            grid_dim=config["dset"]["grid_dim"],
            resolution=config["dset"]["resolution"],
            fabric=fabric,
        )
    save_xyz(mols_pred, dirname_voxels, fabric, atom_elements=config["dset"]["elements"])
    convert_xyzs_to_sdf(dirname_voxels, fabric=fabric, delete=True, fname=f"molecules_obabel_pred_refine_{refine}.sdf")

    # ground truth
    if mols_gt is not None:
        save_xyz(mols_gt, dirname_voxels, fabric, atom_elements=config["dset"]["elements"])
        convert_xyzs_to_sdf(dirname_voxels, fabric=fabric, delete=True, fname="molecules_obabel_gt_refine_False.sdf")


def infer_codes(
    loader: torch.utils.data.DataLoader,
    enc: torch.nn.Module,
    config: dict,
    fabric = None,
    to_cpu: bool = False,
    field_maker = None,
    code_stats=None,
    n_samples=None,
) -> torch.Tensor:
    """
    Infers codes from a given data loader using a specified encoder model.

    Args:
        loader (torch.utils.data.DataLoader): DataLoader providing the batches of data.
        enc (torch.nn.Module): Encoder model used to infer codes.
        config (dict): Configuration dictionary containing model and dataset parameters.
        fabric: Optional fabric object for distributed training (default: None).
        to_cpu (bool): Flag indicating whether to move the inferred codes to CPU (default: False).
        field_maker: Optional field maker object for additional processing (default: None).
        code_stats: Optional object to collect code statistics (default: None).
        n_samples (int, optional): Number of samples to infer codes for. If None, infer codes for all samples (default: None).

    Returns:
        torch.Tensor: Tensor containing the inferred codes.
    """
    enc.eval()
    codes_all = []
    if fabric is not None:
        fabric.print(">> Inferring codes - batches of", config["dset"]["batch_size"])
    len_codes = 0
    for i, batch in tqdm(enumerate(loader)):
        with torch.no_grad():
            codes, _ = infer_codes_occs_batch(batch, enc, config, to_cpu, field_maker=field_maker, code_stats=code_stats)
        codes = fabric.all_gather(codes).view(-1, config["decoder"]["code_dim"])
        len_codes += codes.size(0)
        codes_all.append(codes)
        if n_samples is not None and len_codes >= n_samples:
            break
    return torch.cat(codes_all, dim=0)


def infer_codes_occs_batch(batch, enc, config, to_cpu=False, field_maker=None, code_stats=None):
    """
    Infer codes and occurrences for a batch of data.

    Args:
        batch (Tensor): The input batch of data.
        enc (Callable): The encoder function to generate codes from voxels.
        config (dict): Configuration dictionary.
        to_cpu (bool, optional): If True, move the codes to CPU. Defaults to False.
        field_maker (Callable, optional): A function to generate occurrences and voxels from the batch. Defaults to None.
        code_stats (dict, optional): Statistics for normalizing the codes. Defaults to None.

    Returns:
        Tuple[Tensor, Tensor]: A tuple containing the inferred codes and occurrences.
    """
    occs, voxels = field_maker.forward(batch)
    codes = enc(voxels)
    if code_stats is not None:
        codes = normalize_code(codes, code_stats)
    if to_cpu:
        codes = codes.cpu()
    return codes, occs


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

    weight_first_layer_after = next(iter(net_dict.values())).sum()
    assert weight_first_layer_before != weight_first_layer_after, "loading did not work"
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

    dec = Decoder({
        "grid_size": config["dset"]["grid_dim"],
        "hidden_dim": config["decoder"]["hidden_dim"],
        "n_layers": config["decoder"]["n_layers"],
        "k_neighbors": config["encoder"]["k_neighbors"],
        "n_channels": config["dset"]["n_channels"],
        "code_dim": config["decoder"]["code_dim"]
    })
    dec = load_network(nf_checkpoint, dec, fabric, net_name="dec")
    dec = torch.compile(dec)
    dec.eval()

    enc = Encoder(
        bottleneck_channel=config["decoder"]["code_dim"],
        in_channels=config["dset"]["n_channels"],
        level_channels=config["encoder"]["level_channels"],
        smaller=config["encoder"]["smaller"] if "smaller" in config["encoder"] else False,
    )
    enc = load_network(nf_checkpoint, enc, fabric, net_name="enc")
    enc = torch.compile(enc)
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


def compute_vector_field(
    xs,
    coords,
    atoms_channel,
    n_atom_types=5,
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Compute vector field using GNFConverter, treating each atom type independently.
    Each atom type has its own sigma parameter for better control of the gradient field.
    
    Args:
        xs: [batch_size, n_points, 3] - Query points
        coords: [batch_size, n_atoms, 3] - Atom coordinates
        atoms_channel: [batch_size, n_atoms] - Atom types
        n_atom_types: Number of atom types (default: 5 for QM9)
        
    Returns:
        vector_field: [batch_size, n_points, n_atom_types, 3] - Vector field for each atom type
    """
    batch_size, n_points, _ = xs.shape
    
    # 为不同类型的原子设置不同的 sigma 参数
    sigma_params = {
        0: 0.8,  # C - 碳原子，较小的 sigma 使梯度场更集中
        1: 1.2,  # H - 氢原子，较大的 sigma 使梯度场更分散
        2: 1.0,  # O - 氧原子，中等 sigma
        3: 0.9,  # N - 氮原子，略小于氧原子
        4: 1.1,  # F - 氟原子，略大于氧原子
    }
    
    vector_field = torch.zeros(batch_size, n_points, n_atom_types, 3, device=device)
    
    for b in range(batch_size):
        mask = atoms_channel[b] != PADDING_INDEX
        valid_coords = coords[b, mask]  # [n_valid_atoms, 3]
        valid_types = atoms_channel[b, mask].long()  # [n_valid_atoms]
        
        if valid_coords.size(0) == 0:
            continue
        
        # 对每种原子类型分别计算梯度场
        for t in range(n_atom_types):
            # 只选择当前类型的原子
            type_mask = (valid_types == t)
            if type_mask.sum() > 0:
                # 获取当前类型原子的坐标
                type_coords = valid_coords[type_mask]  # [n_type_atoms, 3]
                
                # 使用当前原子类型特定的 sigma 参数
                gnf_converter = GNFConverter(
                    # sigma=sigma_params[t],
                    sigma=0.9,
                    device=device
                ).to(device)
                
                # 计算当前类型原子的梯度场
                type_gradients = gnf_converter._compute_gnf(
                    type_coords,  # [n_type_atoms, 3]
                    xs[b],  # [n_points, 3]
                    version=4
                )  # [n_points, 3]
                
                # 将梯度场赋值给对应的通道
                vector_field[b, :, t, :] = type_gradients
    
    return vector_field


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
        grid_size=config["dset"]["grid_dim"],
        code_dim=config["decoder"]["code_dim"],
        hidden_dim=config["decoder"]["hidden_dim"],
        num_layers=config["encoder"]["num_layers"],
        k_neighbors=config["encoder"]["k_neighbors"]
    )
    n_params_enc = sum(p.numel() for p in enc.parameters() if p.requires_grad)
    fabric.print(f">> enc has {(n_params_enc/1e6):.02f}M parameters")

    # Initialize the decoder
    dec = Decoder({
        "grid_size": config["dset"]["grid_dim"],
        "hidden_dim": config["decoder"]["hidden_dim"],
        "n_layers": config["decoder"]["n_layers"],
        "k_neighbors": config["encoder"]["k_neighbors"],
        "n_channels": config["dset"]["n_channels"],
        "code_dim": config["decoder"]["code_dim"]
    }, device=fabric.device)
    n_params_dec = sum(p.numel() for p in dec.parameters() if p.requires_grad)
    fabric.print(f">> dec has {(n_params_dec/1e6):.02f}M parameters")

    # Compile the models
    fabric.print(">> compiling models...")
    dec = torch.compile(dec)
    enc = torch.compile(enc)
    fabric.print(">> models compiled")

    return enc, dec