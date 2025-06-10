import time
import hydra
import os
import torch
import torchmetrics
from torch import nn
from tqdm import tqdm
import wandb
from funcmol.utils.utils_base import setup_fabric
from funcmol.utils.utils_nf import (
    create_neural_field, train_nf, eval_nf, save_checkpoint, load_network, load_optim_fabric, compute_vector_field
)
from funcmol.dataset.dataset_field import create_field_loaders
from funcmol.dataset.field_maker import FieldMaker
from funcmol.utils.constants import ELEMENTS_HASH, PADDING_INDEX
from funcmol.utils.gnf_converter import GNFConverter, MolecularStructure
from funcmol.utils.visualize_molecules import visualize_molecule_comparison
from lightning import Fabric


# 使用 Hydra 装饰器，读取位于 configs/train_nf_drugs.yaml 的配置文件，并传递为 config
@hydra.main(config_path="configs", config_name="train_nf_qm9", version_base=None)
def main(config):
    # 设置CUDA内存分配策略
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # 初始化fabric（单卡模式）
    fabric = Fabric(
        accelerator="auto",
        devices=1,  # 使用单卡
        precision="bf16-mixed"
    )
    fabric.launch()
    
    fabric.print(">> start training the neural field", config["exp_name"])
    
    field_maker = FieldMaker(config)
    field_maker = field_maker.to(fabric.device)

    ##############################
    # data loaders
    loader_train = create_field_loaders(config, split="train", fabric=fabric)
    loader_val = create_field_loaders(config, split="val", fabric=fabric) if fabric.global_rank == 0 else None
    # 只有主进程（global_rank == 0）加载验证集

    # model
    enc, dec = create_neural_field(config, fabric)
    criterion = nn.MSELoss()

    # optimizers
    optim_enc = torch.optim.Adam([{"params": enc.parameters(), "lr": config["dset"]["lr_enc"]}])
    optim_dec = torch.optim.Adam([{"params": dec.parameters(), "lr": config["dset"]["lr_dec"]}])

    # 设置设备
    dec, optim_dec = fabric.setup(dec, optim_dec)
    enc, optim_enc = fabric.setup(enc, optim_enc)
    
    # 设置 DDP 策略
    if hasattr(fabric.strategy, "ddp_kwargs"):
        fabric.strategy.ddp_kwargs["find_unused_parameters"] = True

    # reload
    if config["reload_model_path"] is not None:
        try:
            checkpoint = fabric.load(os.path.join(config["reload_model_path"], "model.pt"))
            fabric.print(f">> loaded checkpoint: {config['reload_model_path']}")

            dec = load_network(checkpoint, dec, fabric, net_name="dec")
            optim_dec = load_optim_fabric(optim_dec, checkpoint, config, fabric, net_name="dec")

            enc = load_network(checkpoint, enc, fabric, net_name="enc")
            optim_enc = load_optim_fabric(optim_enc, checkpoint, config, fabric, net_name="enc")
        except Exception as e:
            fabric.print(f"Error loading checkpoint: {e}")

    # Metrics
    metrics = {
        "loss": torchmetrics.MeanMetric().to(fabric.device),
        "miou": torchmetrics.classification.BinaryJaccardIndex().to(fabric.device),
    }
    metrics_val = {
        "loss": torchmetrics.MeanMetric(sync_on_compute=False).to(fabric.device),
        "miou": torchmetrics.classification.BinaryJaccardIndex(sync_on_compute=False).to(fabric.device),
    }
    # `miou` 是 **mean Intersection over Union（平均交并比）**，是语义分割（semantic segmentation）任务中最常见的评价指标之一
    # "macro"（对每一类平均）
    
    ##############################
    # start training the neural field
    best_loss = None  # save checkpoint each time save_checkpoint is called

    for epoch in range(config["n_epochs"]):
        start_time = time.time()

        adjust_learning_rate(optim_enc, optim_dec, epoch, config)

        # train
        loss_train, miou_train = train_nf(
            config,
            loader_train,
            dec,
            optim_dec,
            enc,
            optim_enc,
            criterion,
            fabric,
            metrics=metrics,
            field_maker=field_maker,
            epoch=epoch
        )

        # val
        loss_val, miou_val = None, None
        if (epoch % config["eval_every"] == 0 or epoch == config["n_epochs"] - 1) and epoch > 0:
            # Master rank performs evaluation and checkpointing
            if fabric.global_rank == 0:
                loss_val, miou_val = eval_nf(
                    loader_val,
                    dec,
                    enc,
                    criterion,
                    config,
                    metrics=metrics_val,
                    fabric=fabric,
                    field_maker=field_maker
                )
                save_checkpoint(
                    epoch, config, loss_val, best_loss, enc, dec, optim_enc, optim_dec, fabric)
            else:
                fabric.barrier()
        # log
        elapsed_time = time.time() - start_time
        log_epoch(config, epoch, loss_train, miou_train, loss_val, miou_val, elapsed_time, fabric)


def adjust_learning_rate(optim_enc, optim_dec, epoch, config):
    """
    Adjusts the learning rate for the encoder and decoder optimizers based on the current epoch and configuration.

    Parameters:
    optim_enc (torch.optim.Optimizer): The optimizer for the encoder.
    optim_dec (torch.optim.Optimizer): The optimizer for the decoder.
    epoch (int): The current epoch number.
    config (dict): Configuration dictionary containing learning rate settings and decay milestones.

    Returns:
    None
    """
    # Handcoded by now. Can be improved.
    if "lr_decay" not in config or config["lr_decay"] is None or not config["lr_decay"]:
        return
    lr_enc = config["dset"]["lr_enc"]
    lr_dec = config["dset"]["lr_dec"]
    for milestone in [80]:
        lr_enc *= 0.1 if epoch >= milestone else 1.0
        lr_dec *= 0.1 if epoch >= milestone else 1.0

    for param_group in optim_enc.param_groups:
        param_group["lr"] = lr_enc
    for param_group in optim_dec.param_groups:
        param_group["lr"] = lr_dec
    print("epoch", epoch, "lr_enc", lr_enc, "lr_dec", lr_dec)


def log_epoch(
    config: dict,
    epoch: int,
    loss_train: float,
    miou_train: float,
    loss_val: float,
    miou_val: float,
    elapsed_time: float,
    fabric: object
    ) -> None:
    """
    Logs the training and validation metrics for a given epoch.

    Args:
        config (dict): Configuration dictionary containing dataset and experiment details.
        epoch (int): The current epoch number.
        loss_train (float): Training loss for the current epoch.
        miou_train (float): Training mean Intersection over Union (mIoU) for the current epoch.
        loss_val (float): Validation loss for the current epoch.
        miou_val (float): Validation mean Intersection over Union (mIoU) for the current epoch.
        elapsed_time (float): Time elapsed since the start of training in seconds.
        fabric (object): An object for logging metrics, such as a Weights and Biases (wandb) logger.

    Returns:
        None
    """
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    log = f"\nEpoch {epoch}/{config['n_epochs']} | " +\
        f"Train Loss: {loss_train:.4f} | Train mIoU: {miou_train:.4f}"
    if loss_val is not None:
        log += f" | Val Loss: {loss_val:.4f} | Val mIoU: {miou_val:.4f}"
    log += f" | Time: {int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}"

    if config["wandb"]:
        fabric.log_dict({
            "trainer/global_step": epoch,
            "train_loss": loss_train,
            "train_miou": miou_train,
            "val_loss": loss_val,
            "val_miou": miou_val
        })
    fabric.print(log)


def compute_rmsd(coords1, coords2):
    """Calculate symmetric RMSD between two sets of coordinates"""
    # 先计算距离（不是平方距离）
    dist1 = torch.sqrt(torch.sum((coords1.unsqueeze(1) - coords2.unsqueeze(0))**2, dim=2) + 1e-8)
    dist2 = torch.sqrt(torch.sum((coords2.unsqueeze(1) - coords1.unsqueeze(0))**2, dim=2) + 1e-8)
    
    # 对距离取min
    min_dist1 = torch.min(dist1, dim=1)[0]  # 对于coords1中的每个点，找到最近的coords2中的点
    min_dist2 = torch.min(dist2, dim=1)[0]  # 对于coords2中的每个点，找到最近的coords1中的点
    
    # 直接平均，不需要再开方
    rmsd = (torch.mean(min_dist1) + torch.mean(min_dist2)) / 2
    
    return rmsd

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
) -> tuple:
    enc.train()
    dec.train()
    if metrics is not None:
        for key in metrics.keys():
            metrics[key].reset()

    # 创建保存可视化结果的目录
    vis_dir = os.path.join(config["dirname"], "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

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
                    with torch.cuda.amp.autocast():  # 移除 device_type 和 dtype 参数
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
        loss = reconstruction_loss + field_loss
        
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
        # 计算预测和目标的二值化结果用于 mIoU
        pred_binary = (pred_field.norm(dim=-1) > 0.5).float()
        target_binary = (target_field.norm(dim=-1) > 0.5).float()
        metrics["miou"].update(pred_binary, target_binary)
        
        # 确保所有进程同步
        fabric.barrier()

    if fabric.global_rank == 0:
        pbar.close()

    return metrics["loss"].compute().item(), metrics["miou"].compute().item()


if __name__ == "__main__":
    main()
