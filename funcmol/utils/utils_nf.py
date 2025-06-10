import os
import torch
from collections import OrderedDict, defaultdict
from tqdm import tqdm
from torch import nn
import shutil
from funcmol.utils.constants import PADDING_INDEX
from funcmol.utils.gnf_converter import GNFConverter

from funcmol.models.decoder import Decoder, _normalize_coords, get_atom_coords
from funcmol.models.encoder import Encoder
from funcmol.utils.utils_base import convert_xyzs_to_sdf, save_xyz
from funcmol.utils.utils_vis import visualize_voxel_grid
import time


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
    enc = Encoder(
        bottleneck_channel=config["decoder"]["code_dim"],
        in_channels=config["dset"]["n_channels"],
        level_channels=config["encoder"]["level_channels"],
        smaller=config["encoder"]["smaller"] if "smaller" in config["encoder"] else False,
    )
    n_params_enc = sum(p.numel() for p in enc.parameters() if p.requires_grad)
    fabric.print(f">> enc has {(n_params_enc/1e6):.02f}M parameters")

    # 创建decoder配置
    decoder_config = {
        "grid_size": config["decoder"]["grid_size"],
        "hidden_dim": config["decoder"]["hidden_dim"],
        "n_layers": config["decoder"]["n_layers"],
        "k_neighbors": config["decoder"]["k_neighbors"],
        "n_channels": config["dset"]["n_channels"],
        "code_dim": config["decoder"]["code_dim"]
    }
    
    # 创建decoder
    dec = Decoder(decoder_config)
    n_params_dec = sum(p.numel() for p in dec.parameters() if p.requires_grad)
    fabric.print(f">> dec has {(n_params_dec/1e6):.02f}M parameters")

    fabric.print(">> compiling models...")
    dec = torch.compile(dec)
    enc = torch.compile(enc)
    fabric.print(">> models compiled")

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
    metrics=None,
    field_maker=None,
    epoch: int = 0,
) -> float:
    """
    Trains the neural field model.

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

    Returns:
        float: The computed loss value.
    """
    # 添加调试信息
    fabric.print(f">> Starting train_nf function")
    fabric.print(f">> Global rank: {fabric.global_rank}")
    fabric.print(f">> Config dirname: {config['dirname']}")
    
    enc.train()
    dec.train()
    if metrics is not None:
        for key in metrics.keys():
            metrics[key].reset()

    # 创建保存可视化结果的目录
    vis_dir = os.path.join(config["dirname"], "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    fabric.print(f">> Created visualization directory: {vis_dir}")
    
    # 创建日志文件
    log_dir = os.path.join(config["dirname"], "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"gradient_field_epoch_{epoch}.log")
    fabric.print(f">> Created log directory: {log_dir}")
    fabric.print(f">> Created log file: {log_file}")
    
    # 确保日志文件存在
    if fabric.global_rank == 0:
        try:
            with open(log_file, 'w') as f:
                f.write(f"=== Training Log for Epoch {epoch} ===\n")
                f.write(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            fabric.print(f">> Successfully wrote initial content to log file")
        except Exception as e:
            fabric.print(f">> Error writing to log file: {e}")
    
    # 创建进度条，只在主进程显示
    if fabric.global_rank == 0:
        pbar = tqdm(total=len(loader), desc="Training", leave=True)
    total_loss = 0.0
    
    # 设置梯度裁剪阈值，之前训练时，梯度爆炸，所以添加梯度裁剪
    max_grad_norm = 1.0
    
    for i, batch in enumerate(loader):
        # 获取查询点
        query_points = batch["xs"].to(fabric.device)  # [16, 500, 3]
        
        # 获取目标矢量场
        occs_points, occs_grid = field_maker.forward(batch)  # [16, 500, 5], [16, 15, 32, 32, 32]
        coords = batch["coords"].to(fabric.device)  # [16, n_atoms, 3]
        atoms_channel = batch["atoms_channel"].to(fabric.device)  # [16, n_atoms]
        target_field = compute_vector_field(query_points, coords, atoms_channel, n_atom_types=config["dset"]["n_channels"] // 3, device=fabric.device)  # [16, 500, 5, 3]
        
        # 前向传播
        pred_field = dec(query_points)  # [16, 500, 5, 3]
        
        # 记录梯度场信息到日志文件
        if fabric.global_rank == 0 and i % 100 == 0:  # 每100个batch记录一次
            try:
                with open(log_file, 'a') as f:
                    f.write(f"\n=== Gradient Field Information (Epoch {epoch}, Batch {i}) ===\n")
                    f.write(f"Query points shape: {query_points.shape}\n")
                    f.write(f"Target field shape: {target_field.shape}\n")
                    f.write(f"Predicted field shape: {pred_field.shape}\n")
                    
                    # 记录数值范围
                    f.write("\nValue ranges:\n")
                    f.write(f"Query points range: [{query_points.min().item():.3f}, {query_points.max().item():.3f}]\n")
                    f.write(f"Target field range: [{target_field.min().item():.3f}, {target_field.max().item():.3f}]\n")
                    f.write(f"Predicted field range: [{pred_field.min().item():.3f}, {pred_field.max().item():.3f}]\n")
                    
                    # 记录范数
                    f.write("\nNorms:\n")
                    f.write(f"Target field norm: {target_field.norm().item():.3f}\n")
                    f.write(f"Predicted field norm: {pred_field.norm().item():.3f}\n")
                    
                    # 记录NaN和Inf检查
                    f.write("\nNaN/Inf check:\n")
                    f.write(f"Target field NaN count: {torch.isnan(target_field).sum().item()}\n")
                    f.write(f"Target field Inf count: {torch.isinf(target_field).sum().item()}\n")
                    f.write(f"Predicted field NaN count: {torch.isnan(pred_field).sum().item()}\n")
                    f.write(f"Predicted field Inf count: {torch.isinf(pred_field).sum().item()}\n")
                    
                    # 记录一些具体的值
                    f.write("\nSample values (first batch, first point, first atom type):\n")
                    f.write(f"Target field: {target_field[0, 0, 0].tolist()}\n")
                    f.write(f"Predicted field: {pred_field[0, 0, 0].tolist()}\n")
                    f.write("================================\n")
                fabric.print(f">> Wrote gradient field info to log file at batch {i}")
            except Exception as e:
                fabric.print(f">> Error writing to log file: {e}")
        
        # 计算梯度场损失
        field_loss = criterion(pred_field, target_field)
        
        # 计算重建损失（使用梯度场的差异来近似）
        batch_size = coords.size(0)
        reconstruction_loss = 0.0
        
        for b in range(batch_size):
            # 获取当前样本的有效原子
            mask = atoms_channel[b] != PADDING_INDEX
            valid_coords = coords[b, mask]  # [n_valid_atoms, 3]
            valid_types = atoms_channel[b, mask]  # [n_valid_atoms]
            
            if valid_coords.size(0) == 0:
                continue
            
            # 每隔一定轮次可视化分子结构
            if fabric.global_rank == 0 and i % config.get("vis_every", 1000) == 0:  # 增加间隔到1000
                try:
                    fabric.print(f"\nAttempting visualization for batch {b}, iteration {i}")
                    fabric.print(f"Original coords shape: {valid_coords.shape}")
                    fabric.print(f"Original types shape: {valid_types.shape}")
                    fabric.print(f"Predicted field shape: {pred_field[b].shape}")
                    
                    # 清理GPU缓存
                    torch.cuda.empty_cache()
                    
                    # 使用GNFConverter重建分子结构
                    gnf_converter = GNFConverter(
                        sigma=1.0,
                        n_query_points=200,  # 减少查询点数量
                        n_iter=500,  # 减少迭代次数
                        step_size=0.005,
                        merge_threshold=0.2,
                        device=fabric.device
                    )
                    
                    # 确保输入维度正确
                    pred_field_b = pred_field[b].unsqueeze(0)  # [1, n_points, n_atom_types, 3]
                    valid_types_b = valid_types.unsqueeze(0)  # [1, n_valid_atoms]
                    
                    fabric.print(f"Input field shape: {pred_field_b.shape}")
                    fabric.print(f"Input types shape: {valid_types_b.shape}")
                    
                    # 重建分子结构
                    with torch.amp.autocast(device_type='cuda'):  # 移除 device_type 和 dtype 参数
                        reconstructed_coords, reconstructed_types = gnf_converter.gnf2mol(
                            pred_field_b,
                            valid_types_b
                        )
                    
                    fabric.print(f"Reconstructed coords shape: {reconstructed_coords.shape}")
                    fabric.print(f"Reconstructed types shape: {reconstructed_types.shape}")
                    
                    # 创建可视化目录
                    vis_dir = os.path.join(config["dirname"], "visualizations")
                    os.makedirs(vis_dir, exist_ok=True)
                    
                    # 保存可视化结果
                    save_path = os.path.join(vis_dir, f"molecule_epoch{epoch:04d}_batch{i:04d}_sample{b:02d}.png")
                    
                    # 使用try-finally确保PyMOL正确关闭
                    try:
                        from funcmol.utils.visualize_molecules import visualize_molecule_comparison
                        visualize_molecule_comparison(
                            valid_coords,
                            valid_types,
                            reconstructed_coords,
                            reconstructed_types,
                            save_path=save_path
                        )
                        fabric.print(f"Visualization saved to: {save_path}")
                    except Exception as e:
                        fabric.print(f"Visualization failed: {str(e)}")
                    finally:
                        # 确保PyMOL进程被终止
                        import psutil
                        for proc in psutil.process_iter(['pid', 'name']):
                            if 'pymol' in proc.info['name'].lower():
                                try:
                                    proc.kill()
                                except:
                                    pass
                    
                except Exception as e:
                    fabric.print(f"Error during visualization: {str(e)}")
                    continue

            # 计算每个原子位置处的预测梯度场
            atom_gradients = pred_field[b]  # [n_points, n_atom_types, 3]
            target_gradients = target_field[b]  # [n_points, n_atom_types, 3]
            
            # 计算原子位置到查询点的距离
            dist = torch.norm(query_points[b].unsqueeze(1) - valid_coords.unsqueeze(0), dim=2)  # [n_points, n_atoms]
            
            # 使用距离的倒数作为权重
            weights = 1.0 / (dist + 1e-8)  # [n_points, n_atoms]
            # 添加钳位操作，防止权重过大
            weights = torch.clamp(weights, max=1e4) # 将最大权重限制为10000
            weights = weights / weights.sum(dim=0, keepdim=True)  # 归一化权重
            
            # 计算梯度场差异
            grad_diff = torch.norm(atom_gradients - target_gradients, dim=-1)  # [n_points, n_atom_types]
            
            # 扩展权重维度以匹配梯度场差异
            weights = weights.unsqueeze(-1)  # [n_points, n_atoms, 1]
            
            # 计算加权梯度场差异
            weighted_grad_diff = torch.sum(weights * grad_diff.unsqueeze(1), dim=0)  # [n_atoms, n_atom_types]
            
            # 取最小差异作为重建损失
            reconstruction_loss += torch.min(weighted_grad_diff).mean()
        
        # 平均重建损失
        reconstruction_loss = reconstruction_loss / batch_size
        
        # 总损失 = 重建损失 + 梯度场损失
        # loss = reconstruction_loss + field_loss
        loss = field_loss
        
        # 检查损失是否为NaN或Inf
        if torch.isnan(loss) or torch.isinf(loss):
            fabric.print(f"WARNING: Loss is NaN or Inf at epoch {epoch}, batch {i}. Skipping update.")
            fabric.print(f"  reconstruction_loss: {reconstruction_loss.item():.4f}")
            fabric.print(f"  field_loss: {field_loss.item():.4f}")
            fabric.print(f"  pred_field norm: {pred_field.norm().item():.4f}")
            fabric.print(f"  target_field norm: {target_field.norm().item():.4f}")
            torch.cuda.empty_cache()
            continue

        fabric.print(f"DEBUG: Batch {i}, Original Loss: {loss.item():.4f}")
        
        total_loss += loss.item()
        
        # 更新进度条（只在主进程）
        if fabric.global_rank == 0:
            pbar.update(1)
            pbar.set_postfix({
                'loss': f'{loss:.4f}',
                'recon_loss': f'{reconstruction_loss:.4f}',
                'field_loss': f'{field_loss:.4f}',
                'avg_loss': f'{metrics["loss"].compute().item():.4f}'
            })
        
        # 反向传播
        optim_dec.zero_grad()
        optim_enc.zero_grad()
        fabric.backward(loss)
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(dec.parameters(), max_grad_norm)
        torch.nn.utils.clip_grad_norm_(enc.parameters(), max_grad_norm)
        
        optim_dec.step()
        optim_enc.step()
        
        # 更新指标
        # 钳位损失值，防止初期大损失影响平均显示
        clipped_loss = torch.clamp(loss, max=1000.0) # 将损失值最大限制为1000
        fabric.print(f"DEBUG: Batch {i}, Clipped Loss: {clipped_loss.item():.4f}")
        metrics["loss"].update(clipped_loss)
        
        # 确保所有进程同步
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

    for i, batch in enumerate(loader):
        # 获取查询点
        query_points = batch["xs"].to(fabric.device)
        
        # 获取目标矢量场
        occs_points, occs_grid = field_maker.forward(batch)
        coords = batch["coords"].to(fabric.device)
        atoms_channel = batch["atoms_channel"].to(fabric.device)
        target_field = compute_vector_field(query_points, coords, atoms_channel, n_atom_types=config["dset"]["n_channels"] // 3, device=fabric.device)
        
        # 前向传播
        pred_field = dec(query_points)
        
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

    dec = Decoder(
        n_channels=config["dset"]["n_channels"],
        grid_dim=config["dset"]["grid_dim"],
        hidden_dim=config["decoder"]["hidden_dim"],
        code_dim=config["decoder"]["code_dim"],
        coord_dim=config["decoder"]["coord_dim"],
        n_layers=config["decoder"]["n_layers"],
        input_scale=config["decoder"]["input_scale"],
        fabric=fabric
    )
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
                    sigma=sigma_params[t],
                    device=device
                ).to(device)
                
                # 计算当前类型原子的梯度场
                type_gradients = gnf_converter._compute_gnf(
                    type_coords,  # [n_type_atoms, 3]
                    xs[b],  # [500, 3]
                    version=1
                )  # [500, 3]
                
                # 将梯度场赋值给对应的通道
                vector_field[b, :, t, :] = type_gradients
    
    return vector_field