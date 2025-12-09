import sys
sys.path.append("..")

# Standard libraries
import os
import numpy as np
    
# PyTorch and related libraries
import torch
import torch.nn as nn
torch.set_float32_matmul_precision('medium')

# PyTorch Lightning
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.strategies import DDPStrategy

# Configuration management
from omegaconf import OmegaConf
import hydra

# Set GPU environment
# os.environ['CUDA_VISIBLE_DEVICES'] = "0,2,3,4,5"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

from funcmol.models.element_predictor import create_element_predictor
from funcmol.utils.utils_fm import (
    compute_code_stats_offline, compute_codes,
    load_checkpoint_state_fm
)
from funcmol.utils.utils_nf import infer_codes_occs_batch, load_neural_field
from funcmol.dataset.dataset_code import create_code_loaders
from funcmol.dataset.dataset_field import create_field_loaders, create_gnf_converter
from funcmol.models.adamw import AdamW
from funcmol.models.ema import ModelEma


class PredictorLightningModule(pl.LightningModule):
    """
    PyTorch Lightning module for training Element Predictor only
    """
    def __init__(self, config, enc, code_stats, field_loaders=None):
        super().__init__()
        self.config = config
        self.enc = enc
        self.code_stats = code_stats
        self.field_loaders = field_loaders  # 用于on_the_fly=False时获取分子信息
        self.save_hyperparameters(ignore=['enc', 'code_stats', 'field_loaders'])
        
        # Create element predictor
        n_atom_types = config["dset"]["n_channels"]
        self.predictor = create_element_predictor(config, n_atom_types)
        
        # Initialize EMA for predictor
        predictor_config = config.get("predictor", {})
        predictor_ema_decay = predictor_config.get("ema_decay")
        # 如果 decay 为 False，则禁用 EMA
        self.use_ema = predictor_ema_decay is not False
        
        with torch.no_grad():
            if self.use_ema:
                self.predictor_ema = ModelEma(self.predictor, decay=predictor_ema_decay)
            else:
                self.predictor_ema = None
        
        # Set up loss function
        # 使用BCEWithLogitsLoss代替BCELoss，因为：
        # 1. 数值更稳定（结合了sigmoid和BCE）
        # 2. 与bf16-mixed精度兼容
        # 3. predictor现在输出logits而不是概率
        self.bce_criterion = nn.BCEWithLogitsLoss(reduction="mean")
        
        # Track losses for plotting
        self.train_losses = []
        self.val_losses = []
        self.best_loss = float("inf")
        
        # Freeze encoder
        if self.enc is not None:
            for param in self.enc.parameters():
                param.requires_grad = False
        
        # 用于on_the_fly=False时迭代field_loaders获取分子信息
        self._field_train_iter = None
        self._field_val_iter = None

    def _process_batch(self, batch):
        """
        Process a batch and return codes
        
        Args:
            batch: Data batch
                - 如果on_the_fly=True: PyTorch Geometric Batch对象
                - 如果on_the_fly=False: codes tensor [B, grid_size³, code_dim]
            
        Returns:
            codes: [B, grid_size³, code_dim]
        """
        if self.config["on_the_fly"]:
            # 从encoder计算codes
            with torch.no_grad():
                codes = infer_codes_occs_batch(
                    batch, self.enc, self.config, to_cpu=False,
                    code_stats=self.code_stats if self.config["normalize_codes"] else None
                )
        else:
            # 直接使用预先保存的codes，但需要normalize
            codes = batch
            if self.config["normalize_codes"] and self.code_stats is not None:
                from funcmol.utils.utils_nf import normalize_code
                codes = normalize_code(codes, self.code_stats)
        return codes
    
    def _get_element_existence(self, batch, codes_batch_size=None):
        """
        从batch中计算ground truth的元素存在性
        
        Args:
            batch: 
                - 如果on_the_fly=True: PyTorch Geometric Batch对象
                - 如果on_the_fly=False: codes tensor，此时需要从field_loaders获取分子信息
            codes_batch_size: 当on_the_fly=False时，codes batch的大小
        
        Returns:
            element_existence: [B, n_atom_types] - 每个分子中每种元素是否存在（0或1）
        """
        n_atom_types = self.config["dset"]["n_channels"]
        
        if self.config["on_the_fly"]:
            # 从PyTorch Geometric Batch中获取分子信息
            B = batch.num_graphs  # batch中的分子数量
            device = batch.x.device
            
            # 初始化元素存在性矩阵 [B, n_atom_types]
            element_existence = torch.zeros(B, n_atom_types, device=device)
            
            # 对每个分子，检查哪些原子类型存在
            for b in range(B):
                # 获取当前分子的原子类型
                molecule_mask = (batch.batch == b)
                atom_types = batch.x[molecule_mask]  # [n_atoms_in_molecule]
                
                # 统计存在的原子类型
                unique_types = torch.unique(atom_types)
                # 确保类型索引在有效范围内
                valid_types = unique_types[(unique_types >= 0) & (unique_types < n_atom_types)]
                element_existence[b, valid_types] = 1.0
        else:
            # 从field_loaders获取对应的分子信息
            # 注意：这里假设codes和field_loaders的顺序是对应的
            if codes_batch_size is None:
                codes_batch_size = batch.shape[0] if isinstance(batch, torch.Tensor) else len(batch)
            
            # 获取对应的分子batch
            if self.training:
                if self._field_train_iter is None:
                    self._field_train_iter = iter(self.field_loaders[0])
                try:
                    mol_batch = next(self._field_train_iter)
                except StopIteration:
                    # 重新开始迭代
                    self._field_train_iter = iter(self.field_loaders[0])
                    mol_batch = next(self._field_train_iter)
            else:
                if self._field_val_iter is None:
                    self._field_val_iter = iter(self.field_loaders[1])
                try:
                    mol_batch = next(self._field_val_iter)
                except StopIteration:
                    # 重新开始迭代
                    self._field_val_iter = iter(self.field_loaders[1])
                    mol_batch = next(self._field_val_iter)
            
            # 从分子batch中计算元素存在性
            B = mol_batch.num_graphs
            device = mol_batch.x.device
            
            element_existence = torch.zeros(B, n_atom_types, device=device)
            
            for b in range(B):
                molecule_mask = (mol_batch.batch == b)
                atom_types = mol_batch.x[molecule_mask]
                
                unique_types = torch.unique(atom_types)
                valid_types = unique_types[(unique_types >= 0) & (unique_types < n_atom_types)]
                element_existence[b, valid_types] = 1.0
        
        return element_existence
    
    def training_step(self, batch, batch_idx):
        """Training step logic"""
        # 获取codes
        codes = self._process_batch(batch)
        
        # 获取ground truth的元素存在性
        codes_batch_size = codes.shape[0] if isinstance(codes, torch.Tensor) else len(codes)
        gt_element_existence = self._get_element_existence(batch, codes_batch_size=codes_batch_size)
        
        # 确保gt_element_existence与codes在同一设备上
        if isinstance(codes, torch.Tensor):
            gt_element_existence = gt_element_existence.to(codes.device)
        
        # 预测元素存在性
        pred_element_existence = self.predictor(codes)
        
        # 计算BCE loss
        loss = self.bce_criterion(pred_element_existence, gt_element_existence)
        
        # 更新predictor的EMA
        if self.use_ema:
            # 在DDP模式下，需要获取unwrapped模型
            predictor_model = self.predictor.module if hasattr(self.predictor, 'module') else self.predictor
            self.predictor_ema.update(predictor_model)
        
        # Log metrics
        self.log('train_loss', loss, batch_size=len(batch),
                 on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        # Store loss for plotting
        self.train_losses.append(loss.item())
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step logic"""
        # 获取codes
        codes = self._process_batch(batch)
        
        # 获取ground truth的元素存在性
        codes_batch_size = codes.shape[0] if isinstance(codes, torch.Tensor) else len(codes)
        gt_element_existence = self._get_element_existence(batch, codes_batch_size=codes_batch_size)
        
        # 确保gt_element_existence与codes在同一设备上
        if isinstance(codes, torch.Tensor):
            gt_element_existence = gt_element_existence.to(codes.device)
        
        # 使用EMA模型预测元素存在性（如果启用），否则使用普通模型
        if self.use_ema:
            pred_element_existence = self.predictor_ema(codes)
        else:
            pred_element_existence = self.predictor(codes)
        
        # 调试：检查形状是否匹配
        if pred_element_existence.shape != gt_element_existence.shape:
            raise ValueError(
                f"Shape mismatch in validation_step: "
                f"pred_element_existence.shape={pred_element_existence.shape}, "
                f"gt_element_existence.shape={gt_element_existence.shape}, "
                f"codes.shape={codes.shape if isinstance(codes, torch.Tensor) else 'N/A'}"
            )
        
        # 确保数据类型正确
        # 注意：pred_element_existence现在是logits，不需要clamp
        gt_element_existence = gt_element_existence.float()
        
        # 计算BCE loss（使用BCEWithLogitsLoss，内部会处理sigmoid）
        loss = self.bce_criterion(pred_element_existence, gt_element_existence)
        
        # Log metrics
        self.log('val_loss', loss, batch_size=len(batch),
                 on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        return loss
    
    def on_train_epoch_start(self):
        """Called at the beginning of training epoch"""
        # Set model to training mode
        self.predictor.train()
        
        # Clean up GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def on_validation_epoch_start(self):
        """Called at the beginning of validation epoch"""
        # Set model to evaluation mode
        self.predictor.eval()
    
    def on_validation_epoch_end(self):
        """Called at the end of validation epoch"""
        # Update best loss
        val_loss = self.trainer.callback_metrics.get('val_loss')
        if val_loss is not None:
            if hasattr(val_loss, 'item'):
                val_loss_scalar = val_loss.item()
            else:
                val_loss_scalar = float(val_loss)
            
            if val_loss_scalar < self.best_loss:
                self.best_loss = val_loss_scalar
            
            self.val_losses.append(val_loss_scalar)
    
    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers"""
        # Create optimizer (only for predictor)
        optimizer = AdamW(self.predictor.parameters(), 
                         lr=self.config["lr"], 
                         weight_decay=self.config["wd"])
        
        # Create learning rate scheduler if needed
        if self.config.get("use_lr_schedule", 0) > 0:
            # Custom scheduler based on iteration
            scheduler = {
                "scheduler": torch.optim.lr_scheduler.LambdaLR(
                    optimizer, 
                    lr_lambda=self._get_lr_ratio
                ),
                "interval": "step",
                "frequency": 1
            }
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        
        return optimizer
    
    def _get_lr_ratio(self, iteration):
        """Calculate learning rate ratio based on iteration"""
        if iteration < self.config["num_warmup_iter"]:
            # Linear Warmup
            return float((iteration + 1) / self.config["num_warmup_iter"])
        else:
            total_iterations = self.config.get("num_iterations", 50000)
            warmup_iters = self.config["num_warmup_iter"]
            min_ratio = 0.1 # Target minimum learning rate ratio
            
            # Calculate progress within the decay phase [0.0 to 1.0]
            decay_period = total_iterations - warmup_iters
            current_decay_step = iteration - warmup_iters
            
            if decay_period <= 0:
                return min_ratio

            progress_decay = current_decay_step / decay_period
            
            # Calculate square root decay factor [1.0 to 0.0]
            decay_factor = max(0.0, 1.0 - progress_decay) ** 0.5
            
            # Map decay factor to the target range [1.0, 0.1]
            lr_ratio = min_ratio + (1.0 - min_ratio) * decay_factor
            
            return max(lr_ratio, min_ratio)

    def on_save_checkpoint(self, checkpoint):
        # 保存predictor状态
        # 在DDP模式下，predictor可能被包装，需要获取unwrapped模型
        predictor_model = self.predictor.module if hasattr(self.predictor, "module") else self.predictor
        predictor_state_dict = predictor_model.state_dict()
        
        checkpoint["predictor_state_dict"] = predictor_state_dict
        
        # 保存EMA状态（如果启用）
        if self.use_ema:
            # predictor_ema是ModelEma对象，它的内部模型是self.module
            # 获取EMA模型的内部模型状态（不包含ModelEma的包装）
            predictor_ema_state_dict = self.predictor_ema.module.state_dict()
            checkpoint["predictor_ema_state_dict"] = predictor_ema_state_dict
        checkpoint["code_stats"] = self.code_stats
        checkpoint["train_losses"] = self.train_losses
        checkpoint["val_losses"] = self.val_losses
        checkpoint["best_loss"] = self.best_loss
    
    def on_load_checkpoint(self, checkpoint):
        """Custom checkpoint loading logic"""
        if "train_losses" in checkpoint:
            self.train_losses = checkpoint["train_losses"]
        if "val_losses" in checkpoint:
            self.val_losses = checkpoint["val_losses"]
        if "best_loss" in checkpoint:
            self.best_loss = checkpoint["best_loss"]


@hydra.main(config_path="configs", config_name="train_predictor_qm9", version_base=None)
def main_hydra(config):
    """Entry point for Hydra configuration system"""
    main(config)


def main(config):
    """Main function to set up and run the training"""
    # Get available GPU count
    num_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {num_gpus}")
    
    # Resolve config variables before converting to container
    config = OmegaConf.to_container(config, resolve=True)
    
    # Create directories
    os.makedirs(config["dirname"], exist_ok=True)
    
    print(">> saving experiments in:", config["dirname"])
    
    ##############################
    # Load pretrained neural field encoder
    nf_pretrained_path = config["nf_pretrained_path"]
    
    print(f">> Loading Lightning checkpoint from: {nf_pretrained_path}")
    checkpoint = torch.load(nf_pretrained_path, map_location='cpu', weights_only=False)
    
    # Extract config from checkpoint
    nf_config = checkpoint.get("hyper_parameters", {})
    
    # Create a checkpoint dict in the format expected by load_neural_field
    nf_checkpoint = {
        "config": nf_config,
        "enc_state_dict": checkpoint["enc_state_dict"],
        "dec_state_dict": checkpoint["dec_state_dict"]
    }
    
    # Load neural field encoder (decoder not needed for predictor training)
    enc, _ = load_neural_field(nf_checkpoint, None)
    
    # Freeze encoder
    enc.eval()
    for param in enc.parameters():
        param.requires_grad = False
    
    ##############################
    # Code loaders
    config_nf = nf_config
    config_nf["debug"] = config["debug"]
    config_nf["dset"]["batch_size"] = config["dset"]["batch_size"]
    
    # Create data loaders
    try:
        if config["on_the_fly"]:
            # Create GNFConverter instance for data loading
            gnf_converter = create_gnf_converter(config)
            
            loader_train = create_field_loaders(config, gnf_converter, split="train")
            loader_val = create_field_loaders(config, gnf_converter, split="val")
            
            # Compute codes for normalization
            _, code_stats = compute_codes(
                loader_train, enc, config_nf, "train", config["normalize_codes"],
                code_stats=None
            )
            
            # field_loaders用于获取分子信息（on_the_fly=True时，batch本身就包含分子信息）
            field_loaders = None
        else:
            # 使用预先保存的codes
            loader_train = create_code_loaders(config, split="train")
            loader_val = create_code_loaders(config, split="val")
            
            code_stats = compute_code_stats_offline(loader_train, "train", config["normalize_codes"])
            
            # 为了获取分子信息以计算元素存在性，仍然需要加载field数据
            # 但不需要重新计算codes
            gnf_converter = create_gnf_converter(config)
            field_loader_train = create_field_loaders(config, gnf_converter, split="train")
            field_loader_val = create_field_loaders(config, gnf_converter, split="val")
            field_loaders = (field_loader_train, field_loader_val)
        
        # Check if loaders are empty
        if not loader_train or len(loader_train) == 0:
            raise ValueError("Training data loader is empty")
            
        print(f"Created data loaders - Train: {len(loader_train)} batches, Val: {len(loader_val) if loader_val else 0} batches")
    except Exception as e:
        print(f"Error creating data loaders: {e}")
        raise
    
    # Calculate number of iterations
    config["num_iterations"] = config["num_epochs"] * len(loader_train)
    
    # Initialize Lightning model
    model = PredictorLightningModule(config, enc, code_stats, field_loaders=field_loaders)
    
    # Load checkpoint if specified
    if config["reload_model_path"] is not None:
        try:
            checkpoint_path = os.path.join(config["reload_model_path"], "checkpoint_latest.pth.tar")
            training_state = load_checkpoint_state_fm(model, checkpoint_path)
            
            # Apply training state
            if training_state["epoch"] is not None:
                print(f"Resuming from epoch {training_state['epoch']}")
            
            model.train_losses = training_state["train_losses"]
            model.val_losses = training_state["val_losses"]
            model.best_loss = training_state["best_loss"]
            
            # 如果checkpoint中有predictor的状态，加载它
            if "predictor_state_dict" in training_state:
                try:
                    predictor_state_dict = training_state["predictor_state_dict"]
                    if hasattr(model.predictor, "module"):
                        model.predictor.module.load_state_dict(predictor_state_dict)
                    else:
                        model.predictor.load_state_dict(predictor_state_dict)
                    print("Successfully loaded predictor state from checkpoint")
                except Exception as e:
                    print(f"Warning: Failed to load predictor state: {e}")
            
            if "predictor_ema_state_dict" in training_state and model.use_ema:
                try:
                    predictor_ema_state_dict = training_state["predictor_ema_state_dict"]
                    if hasattr(model.predictor_ema, "module"):
                        model.predictor_ema.module.load_state_dict(predictor_ema_state_dict)
                    else:
                        model.predictor_ema.load_state_dict(predictor_ema_state_dict)
                    print("Successfully loaded predictor EMA state from checkpoint")
                except Exception as e:
                    print(f"Warning: Failed to load predictor EMA state: {e}")
                
            print(f"Successfully loaded checkpoint from: {config['reload_model_path']}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
    
    # Configure callbacks
    callbacks = [
        ModelCheckpoint(
            filename="model-{epoch:02d}",
            every_n_epochs=config.get("ckpt_every_n_epochs", 10),
            monitor="val_loss",
            mode="min",
            save_last=True,
            save_top_k=-1  # 保存所有checkpoint，不限制数量
        ),
        LearningRateMonitor(logging_interval="epoch"),
        EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=config.get("early_stopping_patience", 10),
            verbose=True,
            min_delta=1e-5  # 最小改善阈值
        ),
    ]
    
    # Configure logger
    logger = TensorBoardLogger(
        save_dir=config["dirname"],
        default_hp_metric=False
    )
    
    # Configure trainer
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=num_gpus,
        max_epochs=config["num_epochs"],
        callbacks=callbacks,
        logger=logger,
        strategy='auto' if num_gpus == 1 else DDPStrategy(find_unused_parameters=True),
        precision="bf16-mixed",
        enable_checkpointing=True,
        check_val_every_n_epoch=config.get("check_val_every_n_epoch", 1),  # 验证频率，默认每个epoch验证一次
        # Use built-in gradient clipping
        gradient_clip_val=config.get("max_grad_norm", 1.0),
        gradient_clip_algorithm="norm",
    )
    
    # Train the model
    trainer.fit(model, loader_train, loader_val)
    
    if trainer.is_global_zero:
        # Save loss data
        np.save(os.path.join(config["dirname"], "train_losses.npy"), np.array(model.train_losses))
        np.save(os.path.join(config["dirname"], "val_losses.npy"), np.array(model.val_losses))
    
    print("Training completed.")


if __name__ == "__main__":
    main_hydra()

