import sys
sys.path.append("..")

# Standard libraries
import os

# Data visualization and processing
import matplotlib.pyplot as plt
import numpy as np
    
# PyTorch and related libraries
import torch
import torch.nn as nn
torch.set_float32_matmul_precision('medium')

# PyTorch Lightning
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.strategies import DDPStrategy

# Configuration management
from omegaconf import OmegaConf
import hydra

# Set GPU environment
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3,4,5,6,7"

from funcmol.models.funcmol import create_funcmol
from funcmol.utils.utils_fm import (
    add_noise_to_code,
    compute_code_stats_offline, compute_codes
)
from funcmol.models.adamw import AdamW
from funcmol.models.ema import ModelEma
from funcmol.utils.utils_nf import infer_codes_occs_batch, load_neural_field, normalize_code
from funcmol.dataset.dataset_code import create_code_loaders
from funcmol.dataset.dataset_field import create_field_loaders, create_gnf_converter


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


class FuncmolLightningModule(pl.LightningModule):
    """
    PyTorch Lightning module for training Funcmol (denoiser)
    """
    def __init__(self, config, enc, dec_module, code_stats):
        super().__init__()
        self.config = config
        self.enc = enc
        self.dec_module = dec_module
        self.code_stats = code_stats
        self.save_hyperparameters(ignore=['enc', 'dec_module', 'code_stats'])
        
        # Create Funcmol model
        self.funcmol = create_funcmol(config, None)
        
        # Set up loss function
        self.criterion = nn.MSELoss(reduction="mean")
        
        # Initialize EMA model
        with torch.no_grad():
            self.funcmol_ema = ModelEma(self.funcmol, decay=config["ema_decay"])
        
        # Track losses for plotting
        self.train_losses = []
        self.val_losses = []
        self.best_loss = float("inf")
        
    def _process_batch(self, batch):
        """
        Process a batch and return codes and smooth_codes
        
        Args:
            batch: Data batch
            
        Returns:
            tuple: (codes, smooth_codes)
        """
        if self.config["on_the_fly"]:
            with torch.no_grad():
                codes, _ = infer_codes_occs_batch(
                    batch, self.enc, self.config, to_cpu=False,
                    code_stats=self.code_stats if self.config["normalize_codes"] else None
                )
        else:
            codes = normalize_code(batch, self.code_stats)
        
        smooth_codes = add_noise_to_code(codes, smooth_sigma=self.config["smooth_sigma"])
        
        return codes, smooth_codes
    
    def training_step(self, batch, batch_idx):
        """Training step logic"""
        # Get codes and smooth codes
        codes, smooth_codes = self._process_batch(batch)
        
        # Forward pass through Funcmol
        pred_codes = self.funcmol(smooth_codes)
        
        # Calculate loss
        loss = self.criterion(pred_codes, codes)
        
        # Update EMA model
        self.funcmol_ema.update(self.funcmol)
        
        # Log metrics
        self.log('train_loss', loss, batch_size=len(batch),
                 on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        # Store loss for plotting
        self.train_losses.append(loss.item())
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step logic"""
        # Get codes and smooth codes
        codes, smooth_codes = self._process_batch(batch)
        
        # Forward pass through EMA model
        pred_codes = self.funcmol_ema(smooth_codes)
        
        # Calculate loss
        loss = self.criterion(pred_codes, codes)
        
        # Log metrics
        self.log('val_loss', loss, batch_size=len(batch),
                 on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        return loss
    
    def on_train_epoch_start(self):
        """Called at the beginning of training epoch"""
        # Set model to training mode
        self.funcmol.train()
        
        # Clean up GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def on_validation_epoch_start(self):
        """Called at the beginning of validation epoch"""
        # Set model to evaluation mode
        self.funcmol.eval()
    
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
        # Create optimizer
        optimizer = AdamW(self.funcmol.parameters(), 
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
            return float((iteration + 1) / self.config["num_warmup_iter"])
        else:
            # Decay proportionally with the square root of the iteration
            lr_ratio = 1 - (iteration - self.config["num_warmup_iter"] + 1) / (self.config["num_iterations"] - self.config["num_warmup_iter"] + 1)
            return lr_ratio ** 0.5
    
    def on_save_checkpoint(self, checkpoint):
        """Custom checkpoint saving logic"""
        # Save Funcmol state
        checkpoint["funcmol_state_dict"] = self.funcmol.state_dict()
        checkpoint["funcmol_ema_state_dict"] = self.funcmol_ema.state_dict()
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

# @hydra.main(config_path="configs", config_name="train_fm_drugs", version_base=None)
@hydra.main(config_path="configs", config_name="train_fm_qm9", version_base=None)
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
    # Load pretrained neural field
    nf_pretrained_path = config["nf_pretrained_path"]
    
    print(f">> Loading Lightning checkpoint from: {nf_pretrained_path}")
    checkpoint = torch.load(nf_pretrained_path, map_location='cpu', weights_only=False)
    
    # Extract config from checkpoint
    nf_config = checkpoint.get("hyper_parameters", {})
    
    # Create a minimal dummy fabric object for compatibility with data loaders
    class DummyFabric:
        def print(self, msg):
            print(msg)
        def all_gather(self, tensor):
            return tensor
        def setup_dataloaders(self, loader, use_distributed_sampler=True):
            return loader
    
    dummy_fabric = DummyFabric()
    
    # Create a checkpoint dict in the format expected by load_neural_field
    nf_checkpoint = {
        "config": nf_config,
        "enc_state_dict": checkpoint["enc_state_dict"],
        "dec_state_dict": checkpoint["dec_state_dict"]
    }
    
    # Load neural field models using the existing function
    enc, dec = load_neural_field(nf_checkpoint, None)  # Pass None for Lightning compatibility
    
    dec_module = dec.module if hasattr(dec, "module") else dec
    
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
            
            loader_train = create_field_loaders(config, gnf_converter, split="train", fabric=dummy_fabric, use_fabric=False)
            loader_val = create_field_loaders(config, gnf_converter, split="val", fabric=dummy_fabric, use_fabric=False)
            
            # Handle cases where loaders are returned as lists
            if isinstance(loader_train, list) and len(loader_train) > 0:
                loader_train = loader_train[0]
            if isinstance(loader_val, list) and len(loader_val) > 0:
                loader_val = loader_val[0]
            
            # Compute codes for normalization
            _, code_stats = compute_codes(
                loader_train, enc, config_nf, "train", dummy_fabric, config["normalize_codes"],
                code_stats=None
            )
        else:
            loader_train = create_code_loaders(config, split="train", fabric=dummy_fabric)
            loader_val = create_code_loaders(config, split="val", fabric=dummy_fabric)
            
            # Handle cases where loaders are returned as lists
            if isinstance(loader_train, list) and len(loader_train) > 0:
                loader_train = loader_train[0]
            if isinstance(loader_val, list) and len(loader_val) > 0:
                loader_val = loader_val[0]
            
            code_stats = compute_code_stats_offline(loader_train, "train", dummy_fabric, config["normalize_codes"])
        
        # Check if loaders are empty
        if not loader_train or len(loader_train) == 0:
            raise ValueError("Training data loader is empty")
            
        print(f"Created data loaders - Train: {len(loader_train)} batches, Val: {len(loader_val) if loader_val else 0} batches")
    except Exception as e:
        print(f"Error creating data loaders: {e}")
        raise
    
    dec_module.set_code_stats(code_stats)
    
    # Calculate number of iterations
    config["num_iterations"] = config["num_epochs"] * len(loader_train)
    
    # Initialize Lightning model
    model = FuncmolLightningModule(config, enc, dec_module, code_stats)
    
    # Load checkpoint if specified
    if config["reload_model_path"] is not None:
        try:
            def load_checkpoint_state(model, checkpoint_path):
                """Helper function to load checkpoint state"""
                print(f"Loading checkpoint from: {checkpoint_path}")
                state_dict = torch.load(checkpoint_path)
                
                # Load Funcmol state
                if "funcmol" in state_dict:
                    model.funcmol.load_state_dict(state_dict["funcmol"])
                    print("Loaded Funcmol state from 'funcmol' key")
                elif "funcmol_state_dict" in state_dict:
                    model.funcmol.load_state_dict(state_dict["funcmol_state_dict"])
                    print("Loaded Funcmol state from 'funcmol_state_dict' key")
                else:
                    print("No Funcmol state found in checkpoint")
                
                # Load EMA state
                if "funcmol_ema" in state_dict:
                    model.funcmol_ema.load_state_dict(state_dict["funcmol_ema"])
                    print("Loaded Funcmol EMA state from 'funcmol_ema' key")
                elif "funcmol_ema_state_dict" in state_dict:
                    model.funcmol_ema.load_state_dict(state_dict["funcmol_ema_state_dict"])
                    print("Loaded Funcmol EMA state from 'funcmol_ema_state_dict' key")
                
                # Load training state
                training_state = {
                    "epoch": state_dict.get("epoch", None),
                    "train_losses": state_dict.get("train_losses", []),
                    "val_losses": state_dict.get("val_losses", []),
                    "best_loss": state_dict.get("best_loss", float("inf"))
                }
                
                return training_state
            
            checkpoint_path = os.path.join(config["reload_model_path"], "checkpoint_latest.pth.tar")
            training_state = load_checkpoint_state(model, checkpoint_path)
            
            # Apply training state
            if training_state["epoch"] is not None:
                print(f"Resuming from epoch {training_state['epoch']}")
            
            model.train_losses = training_state["train_losses"]
            model.val_losses = training_state["val_losses"]
            model.best_loss = training_state["best_loss"]
                
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
        check_val_every_n_epoch=5,  # Validate every 5 epochs
        # Use built-in gradient clipping
        gradient_clip_val=config.get("max_grad_norm", 1.0),
        gradient_clip_algorithm="norm",
    )
    
    # Train the model
    trainer.fit(model, loader_train, loader_val)
    
    # Save final plots
    if trainer.is_global_zero:
        # Plot loss curves
        if len(model.train_losses) > 0:
            plot_loss_curve(
                model.train_losses,
                model.val_losses,
                os.path.join(config["dirname"], "loss_curve_epochs.png"),
                "(Epoch Level)"
            )
        
        # Save loss data
        np.save(os.path.join(config["dirname"], "train_losses.npy"), np.array(model.train_losses))
        np.save(os.path.join(config["dirname"], "val_losses.npy"), np.array(model.val_losses))
    
    print("Training completed.")


if __name__ == "__main__":
    main_hydra()