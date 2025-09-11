import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import time
import hydra
from omegaconf import OmegaConf
import torch
import torchmetrics
from tqdm import tqdm

from funcmol.models.funcmol import create_funcmol
from funcmol.utils.utils_fm import (
    add_noise_to_code, load_checkpoint_fm, log_metrics,
    compute_code_stats_offline, compute_codes
)
from funcmol.models.adamw import AdamW
from funcmol.models.ema import ModelEma
from funcmol.utils.utils_base import setup_fabric
from funcmol.utils.utils_nf import infer_codes_occs_batch, load_neural_field, normalize_code, create_tensorboard_writer
from funcmol.dataset.dataset_code import create_code_loaders
from funcmol.dataset.dataset_field import create_field_loaders, create_gnf_converter


@hydra.main(config_path="configs", config_name="train_fm_qm9", version_base=None)
def main(config):
    fabric = setup_fabric(config)

    exp_name, dirname = config["exp_name"], config["dirname"]
    if not config["on_the_fly"]:
        codes_dir = config["codes_dir"]

    config = OmegaConf.to_container(config)
    
    # 在分布式训练中，只有rank 0创建目录，其他进程使用相同目录
    if fabric.global_rank == 0:
        config["exp_name"], config["dirname"] = exp_name, dirname
        # 确保目录存在
        os.makedirs(config["dirname"], exist_ok=True)
        os.makedirs(os.path.join(config["dirname"], "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(config["dirname"], "samples"), exist_ok=True)
    else:
        # 其他进程使用rank 0的目录
        config["exp_name"] = exp_name
        config["dirname"] = dirname
    
    if not config["on_the_fly"]:
        config["codes_dir"] = codes_dir # 保护codes_dir，确保在配置对象转换过程中不会丢失关键参数
    
    # 只有rank 0打印目录信息
    if fabric.global_rank == 0:
        fabric.print(">> saving experiments in:", config["dirname"])

    ##############################
    # load pretrained neural field
    nf_checkpoint = fabric.load(os.path.join(config["nf_pretrained_path"], "model.pt"))
    enc, dec = load_neural_field(nf_checkpoint, fabric)
    dec_module = dec.module if hasattr(dec, "module") else dec

    ##############################
    # code loaders
    config_nf = nf_checkpoint["config"]
    config_nf["debug"] = config["debug"]
    config_nf["dset"]["batch_size"] = config["dset"]["batch_size"]

    if config["on_the_fly"]:
        # 创建GNFConverter实例用于数据加载
        gnf_converter = create_gnf_converter(config, device="cpu")
        
        loader_train = create_field_loaders(config, gnf_converter, split="train", fabric=fabric)
        loader_val = create_field_loaders(config, gnf_converter, split="val", fabric=fabric) if fabric.global_rank == 0 else None
        _, code_stats = compute_codes(
            loader_train, enc, config_nf, "train", fabric, config["normalize_codes"],
            code_stats=None
        )
    else:
        loader_train = create_code_loaders(config, split="train", fabric=fabric)
        loader_val = create_code_loaders(config, split="val", fabric=fabric) if fabric.global_rank == 0 else None
        code_stats = compute_code_stats_offline(loader_train, "train", fabric, config["normalize_codes"])
    dec_module.set_code_stats(code_stats)

    config["num_iterations"] = config["num_epochs"] * len(loader_train)

    ##############################
    # create Funcmol, optimizer, criterion, EMA, metrics
    funcmol = create_funcmol(config, fabric)
    criterion = torch.nn.MSELoss(reduction="mean")
    optimizer = AdamW(funcmol.parameters(), lr=config["lr"], weight_decay=config["wd"])
    optimizer.zero_grad()

    with torch.no_grad():
        funcmol_ema = ModelEma(funcmol, decay=config["ema_decay"])
        # Disable torch.compile due to compatibility issues with torch_cluster
        # funcmol_ema = torch.compile(funcmol_ema)

    if config["reload_model_path"] is not None:
        fabric.print(f">> loading checkpoint from {config['reload_model_path']}")
        funcmol, optimizer, code_stats = load_checkpoint_fm(funcmol, config["reload_model_path"], optimizer, fabric=fabric)
        dec_module.set_code_stats(code_stats)
        with torch.no_grad():
            if config["reload_model_path"] is not None:
                funcmol_ema, _ = load_checkpoint_fm(funcmol_ema, config["reload_model_path"], fabric=fabric)
    funcmol, optimizer = fabric.setup(funcmol, optimizer)

    ##############################
    # TensorBoard writer
    tensorboard_writer = None
    if fabric.global_rank == 0:
        tensorboard_writer = create_tensorboard_writer(
            log_dir=config["dirname"],
            experiment_name=config["exp_name"]
        )
    
    ##############################
    # metrics
    metrics = torchmetrics.MeanMetric().to(fabric.device)
    metrics_val = torchmetrics.MeanMetric(sync_on_compute=False).to(fabric.device)

    ##############################
    # start training
    fabric.print(">> start training the denoiser", config["exp_name"])
    best_res = 1e10
    acc_iter = 0

    for epoch in range(0, config["num_epochs"]):
        t0 = time.time()
        
        # 打印epoch信息
        fabric.print(f"\n>> Starting Epoch {epoch+1}/{config['num_epochs']}")
        fabric.print(f">> Total steps per epoch: {len(loader_train)}")
        fabric.print(f">> Global step range: {acc_iter} to {acc_iter + len(loader_train) - 1}")

        # train
        train_loss, acc_iter = train_denoiser(
            loader_train,
            enc,
            dec_module,
            funcmol,
            criterion,
            optimizer,
            metrics,
            config,
            funcmol_ema,
            acc_iter,
            fabric
        )

        # eval
        val_loss = None
        if (epoch + 1) % 5 == 0:
            with fabric.rank_zero_first():
                if fabric.global_rank == 0:
                    val_loss = val_denoiser(
                        loader_val,
                        enc,
                        dec_module,
                        funcmol_ema,
                        criterion,
                        metrics_val,
                        config
                    )
                    if val_loss < best_res:
                        best_res = val_loss

        # save checkpoint
        if ((epoch + 1) % config["save_every"] == 0 or epoch == config["num_epochs"] - 1) and epoch != 0:
            # save checkpoint with epoch number to avoid overwriting
            try:
                state = {
                    "epoch": epoch + 1,
                    "config": config,
                    "state_dict_ema": funcmol_ema.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "code_stats": dec_module.code_stats
                }
                # Create checkpoints directory if it doesn't exist
                checkpoints_dir = os.path.join(config["dirname"], "checkpoints")
                os.makedirs(checkpoints_dir, exist_ok=True)
                
                # Save with epoch number to avoid overwriting
                checkpoint_path = os.path.join(checkpoints_dir, f"checkpoint_epoch_{epoch+1:06d}.pth.tar")
                fabric.save(checkpoint_path, state)
                fabric.print(f">> Checkpoint saved successfully at epoch {epoch + 1}: {checkpoint_path}")
                
                # Also save as latest checkpoint for easy access
                latest_checkpoint_path = os.path.join(checkpoints_dir, "checkpoint_latest.pth.tar")
                fabric.save(latest_checkpoint_path, state)
                fabric.print(f">> Latest checkpoint updated: {latest_checkpoint_path}")
                
            except Exception as e:
                fabric.print(f"Error saving checkpoint: {e}")
                fabric.print(">> Training will continue but checkpoint was not saved")


        # log metrics
        log_metrics(
            config["exp_name"],
            epoch,
            train_loss,
            val_loss,
            best_res,
            time.time() - t0,
            fabric,
        )

        # TensorBoard logging
        if tensorboard_writer is not None:
            tensorboard_writer.add_scalar('Loss/Train', train_loss, epoch)
            if val_loss is not None:
                tensorboard_writer.add_scalar('Loss/Validation', val_loss, epoch)
            tensorboard_writer.add_scalar('Training/Epoch_Time', time.time() - t0, epoch)
            tensorboard_writer.add_scalar('Training/Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
            
        if config["wandb"]:
            fabric.log_dict({
                "trainer/global_step": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
            })
    
    # Close TensorBoard writer
    if tensorboard_writer is not None:
        tensorboard_writer.close()
        fabric.print(">> TensorBoard writer closed")


def train_denoiser(
    loader: torch.utils.data.DataLoader,
    enc: torch.nn.Module,
    dec_module: torch.nn.Module,
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    metrics: torchmetrics.MeanMetric,
    config: dict,
    model_ema: ModelEma=None,
    acc_iter: int = 0,
    fabric: object = None,
) -> tuple:
    """
    Train a denoising model using the provided data loader, model, and training configuration.

    Args:
        loader (torch.utils.data.DataLoader): DataLoader for the training data.
        enc (torch.nn.Module): Encoder module.
        dec_module (torch.nn.Module): Decoder module.
        model (torch.nn.Module): The denoising model to be trained.
        criterion (torch.nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer for model parameters.
        metrics (torchmetrics.MeanMetric): Metric to track the training performance.
        config (dict): Configuration dictionary containing training parameters.
        model_ema (ModelEma, optional): Exponential moving average model. Defaults to None.
        acc_iter (int, optional): Accumulated iteration count. Defaults to 0.
        fabric (object, optional): Fabric object for distributed training. Defaults to None.

    Returns:
        tuple: A tuple containing the computed metric value and the updated accumulated iteration count.
    """
    metrics.reset()
    model.train()
    
    # 创建tqdm进度条
    pbar = tqdm(loader, desc="Training", disable=not fabric.global_rank == 0)
    
    for batch_idx, batch in enumerate(pbar):
        adjust_learning_rate(optimizer, acc_iter, config)
        acc_iter += 1

        if config["on_the_fly"]:
            with torch.no_grad():
                codes, _ = infer_codes_occs_batch(
                    batch, enc, config, to_cpu=False,
                    code_stats=dec_module.code_stats if config["normalize_codes"] else None
                )
        else:
            codes = normalize_code(batch, dec_module.code_stats)

        smooth_codes = add_noise_to_code(codes, smooth_sigma=config["smooth_sigma"])
        
        # 添加denoiser训练调试信息 - 基于epoch内的step数
        debug_frequency = max(1, len(loader) // 10)  # 每个epoch打印10次调试信息
        if batch_idx % debug_frequency == 0:
            fabric.print(f"[TRAIN DEBUG] Global Step {acc_iter}, Epoch Step {batch_idx}/{len(loader)}:")
            fabric.print(f"  codes - min: {codes.min().item():.6f}, max: {codes.max().item():.6f}, mean: {codes.mean().item():.6f}, std: {codes.std().item():.6f}")
            fabric.print(f"  smooth_codes - min: {smooth_codes.min().item():.6f}, max: {smooth_codes.max().item():.6f}, mean: {smooth_codes.mean().item():.6f}, std: {smooth_codes.std().item():.6f}")
            
            # 检查denoiser输出
            with torch.no_grad():
                model.eval()
                denoiser_output = model(smooth_codes)
                model.train()
                fabric.print(f"  denoiser_output - min: {denoiser_output.min().item():.6f}, max: {denoiser_output.max().item():.6f}, mean: {denoiser_output.mean().item():.6f}, std: {denoiser_output.std().item():.6f}")
                
                # 检查是否有异常值
                if torch.isnan(denoiser_output).any():
                    fabric.print("  WARNING: denoiser_output contains NaN values!")
                if torch.isinf(denoiser_output).any():
                    fabric.print("  WARNING: denoiser_output contains Inf values!")
                if denoiser_output.abs().max().item() > 100.0:
                    fabric.print(f"  WARNING: denoiser_output has very large values (max abs: {denoiser_output.abs().max().item():.2f})")
        
        loss = compute_loss(codes, smooth_codes, model, criterion)

        optimizer.zero_grad()
        fabric.backward(loss)
        
        # 梯度裁剪 - 防止梯度爆炸，稳定训练
        max_grad_norm = config.get("max_grad_norm", 1.0)  # 从配置中获取，默认为1.0
        
        # 添加梯度监控 - 基于epoch内的step数
        if batch_idx % debug_frequency == 0:
            total_grad_norm = 0.0
            for param in model.parameters():
                if param.grad is not None:
                    param_grad_norm = param.grad.data.norm(2).item()
                    total_grad_norm += param_grad_norm ** 2
            total_grad_norm = total_grad_norm ** 0.5
            fabric.print(f"  grad_norm before clipping: {total_grad_norm:.6f}")
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        # 记录裁剪后的梯度范数
        if batch_idx % debug_frequency == 0:
            total_grad_norm_after = 0.0
            for param in model.parameters():
                if param.grad is not None:
                    param_grad_norm = param.grad.data.norm(2).item()
                    total_grad_norm_after += param_grad_norm ** 2
            total_grad_norm_after = total_grad_norm_after ** 0.5
            fabric.print(f"  grad_norm after clipping: {total_grad_norm_after:.6f}")
        
        optimizer.step()

        model_ema.update(model)
        metrics.update(loss)
        
        # 更新tqdm进度条
        current_loss = loss.item()
        pbar.set_postfix({
            'loss': f'{current_loss:.4f}',
            'global_step': acc_iter,
            'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
        })

    pbar.close()
    return metrics.compute().item(), acc_iter


@torch.no_grad()
def val_denoiser(
    loader: torch.utils.data.DataLoader,
    enc: torch.nn.Module,
    dec_module: torch.nn.Module,
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    metrics: torchmetrics.MeanMetric,
    config: dict,
) -> float:
    """
    Validate the denoising model on the given data loader.

    Args:
        loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
        enc (torch.nn.Module): Encoder module.
        dec_module (torch.nn.Module): Decoder module.
        model (torch.nn.Module): Denoising model.
        criterion (torch.nn.Module): Loss function.
        metrics (torchmetrics.MeanMetric): Metric to compute the mean loss.
        config (dict): Configuration dictionary containing various settings.

    Returns:
        float: Computed mean loss over the validation dataset.
    """
    enc = enc.module
    model = model.module
    model.eval()
    metrics.reset()
    for batch in loader:
        if config["on_the_fly"]:
            codes, _ = infer_codes_occs_batch(
                batch, enc, config, to_cpu=False,
                code_stats=dec_module.code_stats if config["normalize_codes"] else None
            )
        else:
            codes = normalize_code(batch, dec_module.code_stats)
        smooth_codes = add_noise_to_code(codes, smooth_sigma=config["smooth_sigma"])
        loss = compute_loss(codes, smooth_codes, model, criterion)
        metrics.update(loss)
    return metrics.compute().item()


def compute_loss(
    codes: torch.Tensor,
    smooth_codes: torch.Tensor,
    model: torch.nn.Module,
    criterion: torch.nn.Module,
) -> torch.Tensor:
    """
    Computes the loss between the predicted outputs of the model and the target codes.
    Args:
        codes (torch.Tensor): The target tensor containing the true codes.
        smooth_codes (torch.Tensor): The input tensor to be fed into the model.
        model (torch.nn.Module): The neural network model used for prediction.
        criterion (torch.nn.Module): The loss function used to compute the loss.
    Returns:
        torch.Tensor: The computed loss value.
    """
    return criterion(model(smooth_codes), codes)


def adjust_learning_rate(optimizer: torch.optim.Optimizer, iteration: int, config: dict) -> float:
    """
    Adjusts the learning rate based on the current iteration and configuration.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer object.
        iteration (int): The current iteration.
        config (dict): The configuration dictionary containing the learning rate and other parameters.

    Returns:
        float: The adjusted learning rate.
    """
    if config["use_lr_schedule"] <= 0:
        return config["lr"]
    else:
        if iteration < config["num_warmup_iter"]:
            lr_ratio = config["lr"] * float((iteration + 1) / config["num_warmup_iter"])
        else:
            # decay proportionally with the square root of the iteration
            lr_ratio = 1 - (iteration - config["num_warmup_iter"] + 1) / (config["num_iterations"] - config["num_warmup_iter"] + 1)
            lr_ratio = lr_ratio ** .5
        lr = config["lr"] * lr_ratio
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        return lr


if __name__ == "__main__":
    main()
