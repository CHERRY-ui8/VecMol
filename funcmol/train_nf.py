import sys
sys.path.append("..")

import os
import time
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchmetrics
from lightning import Fabric
from omegaconf import OmegaConf
import hydra
from pathlib import Path
import numpy as np

# 在导入torch之前设置GPU
# TODO：手动指定要使用的GPU（0, 1, 或 2）
gpu_id = 1  # 修改这里来选择GPU：0, 1, 或 2
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
print(f"Setting CUDA_VISIBLE_DEVICES={gpu_id}")

from funcmol.utils.utils_nf import train_nf, create_neural_field, eval_nf
from funcmol.utils.utils_nf import load_network, load_optim_fabric, save_checkpoint, auto_load_latest_checkpoint
from funcmol.utils.utils_nf import create_tensorboard_writer
from funcmol.dataset.dataset_field import create_field_loaders, create_gnf_converter
from funcmol.utils.gnf_converter import GNFConverter


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
    
    # 初始化fabric
    fabric = Fabric(
        accelerator="gpu",
        devices=1,  # 使用cuda:0（通过CUDA_VISIBLE_DEVICES设置）
        precision="bf16-mixed"
    )
    fabric.launch()
    
    fabric.print(">> start training the neural field", config["exp_name"])
    fabric.print(f">> Using GPU {gpu_id}")
    
    # 添加调试信息
    fabric.print(f">> CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
    fabric.print(f">> Torch device count: {torch.cuda.device_count()}")
    fabric.print(f">> Torch current device: {torch.cuda.current_device()}")
    fabric.print(f">> Fabric device: {fabric.device}")
    
    # 创建TensorBoard writer
    tensorboard_writer = None
    if fabric.global_rank == 0:  # 只在主进程中创建tensorboard writer
        tensorboard_writer = create_tensorboard_writer(
            log_dir=config["dirname"],
            experiment_name=config["exp_name"]
        )
    
    ##############################
    # data loaders
    # 创建GNFConverter实例用于数据加载
    data_gnf_converter = create_gnf_converter(config, device="cpu")
    
    loader_train = create_field_loaders(config, data_gnf_converter, split="train", fabric=fabric)
    if isinstance(loader_train, list):
        loader_train = loader_train[0]
    loader_val = create_field_loaders(config, data_gnf_converter, split="val", fabric=fabric) if fabric.global_rank == 0 else None
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
    
    # 自动检测并加载最新的checkpoint（如果启用了auto_resume且没有指定reload_model_path）
    elif config.get("auto_resume", True) and config["reload_model_path"] is None:
        fabric.print(">> Auto-resume enabled, looking for latest checkpoint...")
        enc, dec, optim_enc, optim_dec, start_epoch, train_losses, val_losses, best_loss = auto_load_latest_checkpoint(
            config, enc, dec, optim_enc, optim_dec, fabric
        )
    
    # 从头开始训练（如果禁用了auto_resume）
    else:
        fabric.print(">> Auto-resume disabled, starting fresh training")
        start_epoch = 0
        train_losses = []
        val_losses = []
        best_loss = float("inf")

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
    
    train_losses = []  # 记录每个epoch的训练loss
    val_losses = []    # 记录每个epoch的验证loss
    batch_losses = []  # 记录每个batch的训练loss
    global_step = 0
    
    # 实例化gnf_converter
    gnf_config = config.get("converter")
    if gnf_config is not None and not isinstance(gnf_config, dict):
        gnf_config = OmegaConf.to_container(gnf_config, resolve=True)
    if isinstance(gnf_config, list):
        gnf_config = gnf_config[0]
    assert isinstance(gnf_config, dict), f"gnf_config should be dict, got {type(gnf_config)}"
    
    # 获取方法特定参数
    method = gnf_config.get("gradient_field_method", "softmax")
    method_config = gnf_config.get("method_configs", {}).get(method, gnf_config.get("default_config", {}))
    
    # 构建参数，优先使用方法特定配置
    gnf_converter_params = {
        'sigma': gnf_config.get("sigma", 1.0),
        'n_query_points': method_config.get("n_query_points", 2500),
        'n_iter': gnf_config.get("n_iter", 2000),
        'step_size': method_config.get("step_size", 0.01),
        'eps': gnf_config.get("eps", 0.01),
        'min_samples': gnf_config.get("min_samples", 10),
        'sigma_ratios': gnf_config.get("sigma_ratios", {'C': 0.9, 'H': 1.3, 'O': 1.1, 'N': 1.0, 'F': 1.2}),
        'gradient_field_method': method,
        'temperature': gnf_config.get("temperature", 0.01),
        'logsumexp_eps': gnf_config.get("logsumexp_eps", 1e-8),
        'inverse_square_strength': gnf_config.get("inverse_square_strength", 1.0),
        'gradient_clip_threshold': gnf_config.get("gradient_clip_threshold", 0.3),
        'sig_sf': method_config.get("sig_sf", 0.1),
        'sig_mag': method_config.get("sig_mag", 0.45),
        'gradient_sampling_candidate_multiplier': gnf_config.get("gradient_sampling_candidate_multiplier", 10),
        'gradient_sampling_temperature': gnf_config.get("gradient_sampling_temperature", 0.1),
        'device': "cuda" if torch.cuda.is_available() else "cpu"
    }
    
    gnf_converter = GNFConverter(**gnf_converter_params)
    
    fabric.print(f">> Training loop will start from epoch {start_epoch}")
    
    for epoch in range(start_epoch, config["n_epochs"]):
        start_time = time.time()
        
        fabric.print(f">> Current epoch in loop: {epoch}")

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
            global_step=global_step,
            tensorboard_writer=tensorboard_writer  # 传递tensorboard_writer
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
                
                # TensorBoard记录验证loss
                if tensorboard_writer is not None:
                    tensorboard_writer.add_scalar('Loss/Validation', loss_val, epoch)
                
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
    
    # 关闭TensorBoard writer
    if tensorboard_writer is not None:
        tensorboard_writer.close()
        fabric.print(">> TensorBoard writer closed")


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
    for milestone in [500]: # TODO：这里的数字是hardcoded的，可以改进
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
