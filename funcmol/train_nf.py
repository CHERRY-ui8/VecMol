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

# 在导入torch之前设置GPU，如果要多卡训练，就注释掉这一段
# TODO：手动指定要使用的GPU（0, 1, 或 2）
gpu_id = "0,1,2,3,4,5,6,7"  # 修改这里的数字来指定GPU编号，使用字符串格式，多卡用逗号分隔
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
print(f"Setting CUDA_VISIBLE_DEVICES={gpu_id}")

from funcmol.utils.utils_nf import train_nf, create_neural_field, eval_nf
from funcmol.utils.utils_nf import load_network, load_optim_fabric, save_checkpoint
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
    ##### 多卡模式
    # 设置CUDA内存分配策略
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # 减少PyTorch和Lightning的详细输出
    os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'OFF'  # 关闭分布式调试信息
    os.environ['LIGHTNING_CLI_USAGE'] = 'OFF'  # 关闭Lightning CLI使用信息
    
    # 设置NCCL环境变量以避免通信问题
    os.environ['NCCL_DEBUG'] = 'WARN'  # 只显示警告和错误，不显示INFO
    os.environ['NCCL_TIMEOUT'] = '1800'  # 30分钟超时
    os.environ['NCCL_IB_DISABLE'] = '1'  # 禁用InfiniBand
    os.environ['NCCL_P2P_DISABLE'] = '1'  # 禁用P2P通信
    os.environ['NCCL_SOCKET_IFNAME'] = 'lo'  # 使用本地回环接口
    os.environ['TORCH_NCCL_BLOCKING_WAIT'] = '1'  # 使用新的推荐设置
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # 禁用同步CUDA操作以提高性能
    
    # 添加更多NCCL稳定性设置
    os.environ['NCCL_ASYNC_ERROR_HANDLING'] = '1'  # 启用异步错误处理
    os.environ['NCCL_BUFFSIZE'] = '2097152'  # 增加缓冲区大小
    os.environ['NCCL_NTHREADS'] = '4'  # 设置线程数
    
    # 获取可用的GPU数量
    num_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {num_gpus}")
    
    # 初始化fabric，支持多卡训练
    from lightning.fabric.strategies import DDPStrategy
    ddp_strategy = DDPStrategy(find_unused_parameters=True)  # 必须设置为True，因为有未使用的参数
    
    fabric = Fabric(
        accelerator="gpu",
        devices=num_gpus,  # 使用所有可用的GPU
        strategy=ddp_strategy,    # 使用分布式数据并行策略
        precision="bf16-mixed"
    )

    ##### 单卡模式
    # os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # # 初始化fabric
    # fabric = Fabric(
    #     accelerator="gpu",
    #     devices=1,  # 使用cuda:0（通过CUDA_VISIBLE_DEVICES设置）
    #     precision="bf16-mixed"
    # )
    fabric.launch()
    
    fabric.print(">> start training the neural field", config["exp_name"])
    ##### 多卡模式
    fabric.print(f">> Using {num_gpus} GPUs with DDP strategy")
    ##### 单卡模式
    # fabric.print(f">> Using GPU {gpu_id}")
    
    # 添加调试信息
    fabric.print(f">> Torch device count: {torch.cuda.device_count()}")
    fabric.print(f">> Torch current device: {torch.cuda.current_device()}")
    fabric.print(f">> Fabric device: {fabric.device}")
    fabric.print(f">> Global rank: {fabric.global_rank}")
    fabric.print(f">> Local rank: {fabric.local_rank}")
    fabric.print(f">> World size: {fabric.world_size}")
    
    # 创建TensorBoard writer（只在主进程中创建）
    tensorboard_writer = None
    if fabric.global_rank == 0:
        tensorboard_writer = create_tensorboard_writer(
            log_dir=config["dirname"],
            experiment_name=config["exp_name"]
        )
    
    ##############################
    # data loaders
    # 创建GNFConverter实例用于数据加载
    data_gnf_converter = create_gnf_converter(config)
    
    # 为多卡训练设置数据加载器
    loader_train = create_field_loaders(config, data_gnf_converter, split="train")
    
    # 验证集只在主进程中加载
    loader_val = None
    if fabric.global_rank == 0:
        loader_val = create_field_loaders(config, data_gnf_converter, split="val")

    # model
    enc, dec = create_neural_field(config, fabric)
    if fabric.global_rank == 0:
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
            if fabric.global_rank == 0:
                fabric.print(f">> loaded checkpoint: {config['reload_model_path']}")

            dec = load_network(checkpoint, dec, fabric, net_name="dec")
            optim_dec = load_optim_fabric(optim_dec, checkpoint, config, fabric, net_name="dec")

            enc = load_network(checkpoint, enc, fabric, net_name="enc")
            optim_enc = load_optim_fabric(optim_enc, checkpoint, config, fabric, net_name="enc")
        except Exception as e:
            if fabric.global_rank == 0:
                fabric.print(f"Error loading checkpoint: {e}")
        
    # 从头开始训练（如果禁用了auto_resume）
    else:
        if fabric.global_rank == 0:
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
    
    # 检查必需参数是否存在，移除所有硬编码默认值
    required_params = {
        'sigma': gnf_config.get("sigma"),
        'n_iter': gnf_config.get("n_iter"),
        'eps': gnf_config.get("eps"),
        'min_samples': gnf_config.get("min_samples"),
        'sigma_ratios': gnf_config.get("sigma_ratios"),
        'temperature': gnf_config.get("temperature"),
        'logsumexp_eps': gnf_config.get("logsumexp_eps"),
        'inverse_square_strength': gnf_config.get("inverse_square_strength"),
        'gradient_clip_threshold': gnf_config.get("gradient_clip_threshold"),
        'gradient_sampling_candidate_multiplier': gnf_config.get("gradient_sampling_candidate_multiplier"),
        'gradient_sampling_temperature': gnf_config.get("gradient_sampling_temperature")
    }
    
    # 检查方法特定参数
    method_required_params = {
        'n_query_points': method_config.get("n_query_points"),
        'step_size': method_config.get("step_size"),
        'sig_sf': method_config.get("sig_sf"),
        'sig_mag': method_config.get("sig_mag")
    }
    
    # 检查缺失的参数
    missing_params = [param for param, value in required_params.items() if value is None]
    missing_method_params = [param for param, value in method_required_params.items() if value is None]
    
    if missing_params:
        raise ValueError(f"Missing required parameters in gnf_config: {missing_params}")
    if missing_method_params:
        raise ValueError(f"Missing required parameters in method_config for {method}: {missing_method_params}")
    
    # 构建参数，完全依赖配置文件
    gnf_converter_params = {
        'sigma': required_params['sigma'],
        'n_query_points': method_required_params['n_query_points'],
        'n_iter': required_params['n_iter'],
        'step_size': method_required_params['step_size'],
        'eps': required_params['eps'],
        'min_samples': required_params['min_samples'],
        'sigma_ratios': required_params['sigma_ratios'],
        'gradient_field_method': method,
        'temperature': required_params['temperature'],
        'logsumexp_eps': required_params['logsumexp_eps'],
        'inverse_square_strength': required_params['inverse_square_strength'],
        'gradient_clip_threshold': required_params['gradient_clip_threshold'],
        'sig_sf': method_required_params['sig_sf'],
        'sig_mag': method_required_params['sig_mag'],
        'gradient_sampling_candidate_multiplier': required_params['gradient_sampling_candidate_multiplier'],
        'gradient_sampling_temperature': required_params['gradient_sampling_temperature'],
        'n_atom_types': config["dset"]["n_channels"],  # 从数据集配置获取原子类型数量
        'device': "cuda" if torch.cuda.is_available() else "cpu"
    }
    
    gnf_converter = GNFConverter(**gnf_converter_params)
    
    if fabric.global_rank == 0:
        fabric.print(f">> Training loop will start from epoch {start_epoch}")
    
    for epoch in range(start_epoch, config["n_epochs"]):
        start_time = time.time()
        
        if fabric.global_rank == 0:
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
        loss_val = None  # 初始化验证损失
        if (epoch % config["eval_every"] == 0 or epoch == config["n_epochs"] - 1):
            if loader_val is not None:  # 所有GPU都参与验证
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
                
                # 只在rank 0进行日志记录和保存
                if fabric.global_rank == 0:
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
                # 验证完成后，所有GPU同步
                try:
                    fabric.barrier()
                except Exception as e:
                    print(f"Warning: Barrier operation failed: {e}")
                    # 继续训练，不因为barrier失败而停止
        
        # log
        elapsed_time = time.time() - start_time
        log_epoch(config, epoch, loss_train, loss_val if loss_val is not None else 0.0, elapsed_time, fabric)
    
    # 关闭TensorBoard writer
    if tensorboard_writer is not None:
        tensorboard_writer.close()
        if fabric.global_rank == 0:
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
    for milestone in [500]: # TODO：这里的数字是hardcoded的
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
