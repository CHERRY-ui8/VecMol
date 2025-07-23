import sys
sys.path.append("..")

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
    create_neural_field, train_nf, eval_nf, save_checkpoint, load_network, load_optim_fabric
)
from funcmol.dataset.dataset_field import create_field_loaders
from lightning import Fabric
from omegaconf import OmegaConf


def plot_loss_curve(train_losses, val_losses, save_path, title_suffix=""):
    """
    Plot training and validation loss curves.
    
    Args:
        train_losses (list): List of training losses
        val_losses (list): List of validation losses
        save_path (str): Path to save the plot
        title_suffix (str): Additional information to add to the plot title
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', color='blue')
    if val_losses:
        plt.plot(val_losses, label='Validation Loss', color='red')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title(f'Training and Validation Loss {title_suffix}')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()


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
    
    ##############################
    # data loaders
    loader_train = create_field_loaders(config, split="train", fabric=fabric)
    if isinstance(loader_train, list):
        loader_train = loader_train[0]
    loader_val = create_field_loaders(config, split="val", fabric=fabric) if fabric.global_rank == 0 else None
    if isinstance(loader_val, list):
        loader_val = loader_val[0]
    # 只有主进程（global_rank == 0）加载验证集

    # model
    enc, dec = create_neural_field(config, fabric)
    print('Num of params in encoder:', sum(p.numel() for p in enc.parameters()))
    print('Num of params in decoder:', sum(p.numel() for p in dec.parameters()))
    criterion = nn.MSELoss()

    # optimizers
    optim_enc = torch.optim.Adam([{"params": enc.parameters(), "lr": config["dset"]["lr_enc"]}])
    optim_dec = torch.optim.Adam([{"params": dec.parameters(), "lr": config["dset"]["lr_dec"]}])

    # 设置设备
    dec, optim_dec = fabric.setup(dec, optim_dec)
    enc, optim_enc = fabric.setup(enc, optim_enc)
    
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
        "recon_loss": torchmetrics.MeanMetric().to(fabric.device),
    }
    metrics_val = {
        "loss": torchmetrics.MeanMetric(sync_on_compute=False).to(fabric.device), 
        "recon_loss": torchmetrics.MeanMetric(sync_on_compute=False).to(fabric.device),
    }
    
    ##############################
    # start training the neural field
    best_loss = float("inf")
    start_epoch = 0
    
    train_losses = []  # 记录每个epoch的训练loss
    val_losses = []    # 记录每个epoch的验证loss
    batch_losses = []  # 记录每个batch的训练loss
    global_step = 0
    
    # 实例化gnf_converter
    gnf_config = config.get("converter")
    from funcmol.utils.gnf_converter import GNFConverter
    if gnf_config is not None and not isinstance(gnf_config, dict):
        gnf_config = OmegaConf.to_container(gnf_config, resolve=True)
    if isinstance(gnf_config, list):
        gnf_config = gnf_config[0]
    assert isinstance(gnf_config, dict), f"gnf_config should be dict, got {type(gnf_config)}"
    gnf_converter = GNFConverter(**gnf_config)
    
    for epoch in range(start_epoch, config["n_epochs"]):
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
            gnf_converter,
            metrics=metrics,
            epoch=epoch,
            batch_train_losses=batch_losses,  # 记录每个batch的loss
            global_step=global_step
        )
        train_losses.append(loss_train)
        global_step += len(loader_train)

        # val
        if (epoch % config["eval_every"] == 0 or epoch == config["n_epochs"] - 1):
            if fabric.global_rank == 0 and loader_val is not None:
                loss_val = eval_nf(
                    loader_val,
                    dec,
                    enc,
                    criterion,
                    config,
                    gnf_converter,
                    metrics=metrics_val,
                    fabric=fabric
                )
                val_losses.append(loss_val)
                # 保存checkpoint
                save_checkpoint(
                    epoch, config, loss_val, best_loss, enc, dec, optim_enc, optim_dec, fabric
                )
                # 绘制loss曲线
                if len(batch_losses) > 0:
                    # 绘制每个batch的loss曲线
                    plot_loss_curve(
                        [l["total_loss"] for l in batch_losses],
                        None,
                        os.path.join(config["dirname"], f"loss_curve_batches.png"),
                        "(Batch Level)"
                    )
                # 绘制每个epoch的loss曲线
                plot_loss_curve(
                    train_losses,
                    val_losses,
                    os.path.join(config["dirname"], f"loss_curve_epochs.png"),
                    "(Epoch Level)"
                )
                # 保存loss数据
                np.save(os.path.join(config["dirname"], "train_losses.npy"), np.array(train_losses))
                np.save(os.path.join(config["dirname"], "val_losses.npy"), np.array(val_losses))
                if len(batch_losses) > 0:
                    np.save(
                        os.path.join(config["dirname"], "batch_losses.npy"),
                        np.array([l["total_loss"] for l in batch_losses])
                    )
            else:
                fabric.barrier()
        
        # log
        elapsed_time = time.time() - start_time
        log_epoch(config, epoch, loss_train, loss_val if loss_val is not None else 0.0, elapsed_time, fabric)


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
    for milestone in [500]: # TODO：这里的80是hardcoded的，可以改进
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


if __name__ == "__main__":
    main()
