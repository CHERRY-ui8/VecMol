import time
import hydra
import os
import torch
import torchmetrics
from torch import nn
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt
import numpy as np
from funcmol.utils.utils_base import setup_fabric
from funcmol.utils.utils_nf import (
    create_neural_field, train_nf, eval_nf, save_checkpoint, load_network, load_optim_fabric, compute_vector_field
)
from funcmol.models.encoder import CrossGraphEncoder
from funcmol.dataset.dataset_field import create_field_loaders
from funcmol.dataset.field_maker import FieldMaker
from funcmol.utils.constants import ELEMENTS_HASH, PADDING_INDEX
from funcmol.utils.gnf_converter import GNFConverter, MolecularStructure
from lightning import Fabric
torch._dynamo.config.suppress_errors = True


def plot_loss_curve(train_losses, val_losses, save_path):
    """
    Plot training and validation loss curves.
    
    Args:
        train_losses (list): List of training losses
        val_losses (list): List of validation losses
        save_path (str): Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', color='blue')
    if val_losses:
        plt.plot(val_losses, label='Validation Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()


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
    enc = CrossGraphEncoder(
        n_atom_types=config["dset"]["n_channels"],
        grid_size=config["dset"]["grid_dim"],
        code_dim=config["decoder"]["code_dim"],
        hidden_dim=config["decoder"]["hidden_dim"],
        num_layers=config["encoder"]["num_layers"],
        k_neighbors=config["encoder"]["k_neighbors"]
    )
    dec = create_neural_field(config, fabric)[1]
    
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
    }
    metrics_val = {
        "loss": torchmetrics.MeanMetric(sync_on_compute=False).to(fabric.device),
    }
    
    ##############################
    # start training the neural field
    best_loss = None  # save checkpoint each time save_checkpoint is called
    
    # 用于记录loss的列表
    train_losses = []
    val_losses = []

    for epoch in range(config["n_epochs"]):
        start_time = time.time()

        adjust_learning_rate(optim_enc, optim_dec, epoch, config)

        # train
        loss_train = train_nf(
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
        train_losses.append(loss_train)

        # val
        loss_val = None
        if (epoch % config["eval_every"] == 0 or epoch == config["n_epochs"] - 1):
            # Master rank performs evaluation and checkpointing
            if fabric.global_rank == 0:
                loss_val = eval_nf(
                    loader_val,
                    dec,
                    enc,
                    criterion,
                    config,
                    metrics=metrics_val,
                    fabric=fabric,
                    field_maker=field_maker
                )
                val_losses.append(loss_val)
                save_checkpoint(
                    epoch, config, loss_val, best_loss, enc, dec, optim_enc, optim_dec, fabric)
            else:
                fabric.barrier()
        
        # 在每个epoch结束时绘制loss曲线
        if fabric.global_rank == 0:
            plot_loss_curve(
                train_losses, 
                val_losses, 
                os.path.join(config["dirname"], f"loss_curve_epoch_{epoch}.png")
            )
            
        # log
        elapsed_time = time.time() - start_time
        log_epoch(config, epoch, loss_train, loss_val, elapsed_time, fabric)


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
    loss_val: float,
    elapsed_time: float,
    fabric: object
    ) -> None:
    """
    Logs the training and validation metrics for a given epoch.

    Args:
        config (dict): Configuration dictionary containing dataset and experiment details.
        epoch (int): The current epoch number.
        loss_train (float): Training loss for the current epoch.
        loss_val (float): Validation loss for the current epoch.
        elapsed_time (float): Time elapsed since the start of training in seconds.
        fabric (object): An object for logging metrics, such as a Weights and Biases (wandb) logger.

    Returns:
        None
    """
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    log = f"\nEpoch {epoch}/{config['n_epochs']} | " +\
        f"Train Loss: {loss_train:.4f}"
    if loss_val is not None:
        log += f" | Val Loss: {loss_val:.4f}"
    log += f" | Time: {int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}"

    if config["wandb"]:
        fabric.log_dict({
            "trainer/global_step": epoch,
            "train_loss": loss_train,
            "val_loss": loss_val
        })
    fabric.print(log)


def compute_rmsd(coords1, coords2):
    """Calculate symmetric RMSD between two sets of coordinates"""
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
    """
    Train the neural field model for one epoch.
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

    # 预先创建GNFConverter实例
    gnf_converter = GNFConverter(
        sigma=config["gnf_converter"]["sigma"],
        n_query_points=config["gnf_converter"]["n_query_points"],
        n_iter=config["gnf_converter"]["n_iter"],
        step_size=config["gnf_converter"]["step_size"],
        merge_threshold=config["gnf_converter"]["merge_threshold"],
        device=fabric.device
    )

    for i, batch in enumerate(loader):
        # 1. 准备数据
        coords = batch["coords"].to(fabric.device)  # [B, n_atoms, 3]
        atoms_channel = batch["atoms_channel"].to(fabric.device)  # [B, n_atoms]
        query_points = batch["xs"].to(fabric.device)  # [B, n_points, 3]
        B, n_atoms, _ = coords.shape

        # 2. Encoder: 分子图 -> latent code (codes)
        codes_list = []
        for b in range(B):
            code = enc(coords[b], atoms_channel[b])  # [1, n_grid, code_dim]
            codes_list.append(code)
        codes = torch.cat(codes_list, dim=0)  # [B, n_grid, code_dim]

        # 3. Decoder: codes + query_points -> pred_field
        pred_field = dec(query_points, codes)  # [B, n_points, n_atom_types, 3]

        # 4. 用 GNFConverter 重建分子
        with torch.no_grad():  # 在GNF转换时不计算梯度
            recon_coords, recon_types = gnf_converter.gnf2mol(pred_field.detach())  # [B, n_atoms', 3], [B, n_atoms']

        # 5. 计算重建loss（RMSD或类似指标）
        loss = 0.0
        for b in range(B):
            mask = atoms_channel[b] != PADDING_INDEX
            gt_coords = coords[b, mask]
            pred_coords = recon_coords[b]
            if gt_coords.size(0) > 0 and pred_coords.size(0) > 0:
                loss += compute_rmsd(gt_coords, pred_coords)
        loss = loss / B

        # 6. 反向传播与优化
        optim_dec.zero_grad()
        optim_enc.zero_grad()
        fabric.backward(loss)
        torch.nn.utils.clip_grad_norm_(dec.parameters(), max_grad_norm)
        torch.nn.utils.clip_grad_norm_(enc.parameters(), max_grad_norm)
        optim_dec.step()
        optim_enc.step()

        # 7. 记录与可视化
        total_loss += loss.item()
        if metrics is not None:
            metrics["loss"].update(loss)
        if fabric.global_rank == 0:
            pbar.update(1)
            pbar.set_postfix({
                'batch': f'{i+1}/{len(loader)}',  # 显示当前batch数和总batch数
                'loss': f'{loss.item():.4f}', 
                'avg_loss': f'{metrics["loss"].compute().item():.4f}'
            })
            pbar.refresh()  # 强制刷新进度条

        # 可视化
        if fabric.global_rank == 0 and i % config.get("vis_every", 1000) == 0:
            try:
                for b in range(B):
                    mask = atoms_channel[b] != PADDING_INDEX
                    valid_coords = coords[b, mask]
                    valid_types = atoms_channel[b, mask]
                    save_path = os.path.join(vis_dir, f"sample{b:02d}_epoch{epoch:04d}_batch{i:04d}.png")
                    from funcmol.utils.visualize_molecules import visualize_molecule_comparison
                    visualize_molecule_comparison(
                        valid_coords,
                        valid_types,
                        recon_coords[b],
                        recon_types[b],
                        save_path=save_path
                    )
            except Exception as e:
                fabric.print(f"Visualization failed: {str(e)}")

        fabric.barrier()

    if fabric.global_rank == 0:
        pbar.close()

    return metrics["loss"].compute().item()


if __name__ == "__main__":
    main()
