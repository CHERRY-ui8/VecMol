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

from funcmol.utils.utils_nf import create_neural_field, load_checkpoint_state_nf
from funcmol.dataset.dataset_field import create_field_loaders, create_gnf_converter

# TODO: set gpus based on server id
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3,4,5,6,7"
# os.environ['CUDA_VISIBLE_DEVICES'] = "0,2,3,4,5"
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"


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

class NeuralFieldLightningModule(pl.LightningModule):
    """
    PyTorch Lightning module for training Neural Fields
    """
    def __init__(self, config, gnf_converter):
        super().__init__()
        self.config = config
        self.gnf_converter = gnf_converter
        self.save_hyperparameters(config)
        
        # Create neural field models
        self.enc, self.dec = self._create_models()
        
        # Loss weighting settings
        loss_weight_config = config.get("loss_weighting", {})
        self.loss_weighting_enabled = loss_weight_config.get("enabled", False)
        
        # Set up loss function based on weighting mode
        if self.loss_weighting_enabled:
            # Use reduction='none' to apply weights element-wise
            self.criterion = nn.MSELoss(reduction='none')
            self.atom_distance_weight = loss_weight_config.get("atom_distance_weight", 1.0)
            self.atom_distance_scale = loss_weight_config.get("atom_distance_scale", 1.0)  # 距离衰减尺度
            print(f"Loss weighting enabled: atom_distance={self.atom_distance_weight}")
        else:
            # Standard MSE loss for non-weighted mode
            self.criterion = nn.MSELoss()
            print("Loss weighting disabled, using standard MSE loss")
        
        # Fine-tuning settings: freeze encoder and only train decoder
        finetune_config = config.get("finetune_decoder", {})
        self.finetune_enabled = finetune_config.get("enabled", False)
        
        if self.finetune_enabled:
            # Freeze encoder parameters
            freeze_encoder = finetune_config.get("freeze_encoder", True)
            if freeze_encoder:
                for param in self.enc.parameters():
                    param.requires_grad = False
                self.enc.eval()  # Set encoder to eval mode
                print("Encoder frozen for fine-tuning (requires_grad=False)")
            
            # Code augmentation settings for fine-tuning
            finetune_code_aug = finetune_config.get("code_augmentation", {})
            self.code_aug_enabled = finetune_code_aug.get("enabled", True)
            self.code_aug_noise_std = finetune_code_aug.get("noise_std", 0.01)
            print(f"Fine-tuning mode enabled: training decoder only with code augmentation (noise_std={self.code_aug_noise_std})")
        else:
            # Code augmentation settings (for training robustness)
            code_aug_config = config.get("code_augmentation", {})
            self.code_aug_enabled = code_aug_config.get("enabled", False)
            self.code_aug_noise_std = code_aug_config.get("noise_std", 0.01)
            
            if self.code_aug_enabled:
                print(f"Code augmentation enabled with noise_std={self.code_aug_noise_std}")
        
        # Track losses for plotting
        # self.train_losses = []
        # self.val_losses = []
        # self.batch_losses = []
        # self.best_loss = float("inf")
        
    def _create_models(self):
        """Create encoder and decoder neural field models"""
        try:
            # Create encoder and decoder using utility function
            enc, dec = create_neural_field(self.config)
                        
            return enc, dec
        
        except Exception as e:
            error_msg = f"Error creating neural field models: {str(e)}"
            print(f"ERROR: {error_msg}")
            raise RuntimeError(error_msg)
    
    
    def forward(self, batch):
        """
        Forward pass through the neural field model
        
        Args:
            batch: PyTorch Geometric batch object
            
        Returns:
            torch.Tensor: Predicted field
        """
        # Get query points from batch and ensure they're on the correct device
        query_points = batch.xs
        
        # Get codes from encoder
        # In fine-tuning mode, encoder is in eval mode, so use torch.no_grad() for efficiency
        if self.finetune_enabled:
            with torch.no_grad():
                codes = self.enc(batch)
        else:
            codes = self.enc(batch)
        
        # Apply code augmentation (Gaussian noise) during training only
        if self.training and self.code_aug_enabled:
            noise = torch.randn_like(codes) * self.code_aug_noise_std
            codes = codes + noise
        
        # Check and reshape query points if needed
        if query_points.dim() == 2:
            B = len(batch)
            n_points = self.config["dset"]["n_points"]
            query_points = query_points.view(B, n_points, 3)
            
        # Get predicted field from decoder
        pred_field = self.dec(query_points, codes)
        
        return pred_field
    
    def _compute_loss_weights(self, batch, query_points, target_field):
        """
        Compute loss weights for each query point based on:
        Distance to nearest atom (closer = higher weight)
        
        Args:
            batch: PyTorch Geometric batch object
            query_points: [B, n_points, 3] query point coordinates
            target_field: [B, n_points, n_atom_types, 3] target field values (unused, kept for API compatibility)
            
        Returns:
            weights: [B, n_points] weight tensor for each query point
        """
        B, n_points, _ = query_points.shape
        device = query_points.device
        weights = torch.ones(B, n_points, device=device)
        
        # Get atom positions from batch
        atom_positions = batch.pos  # [N_total_atoms, 3]
        batch_idx = batch.batch  # [N_total_atoms] - which sample each atom belongs to
        
        # Compute weights for each sample in batch
        for b in range(B):
            # Get query points for this sample
            sample_query_points = query_points[b]  # [n_points, 3]
            
            # Get atom positions for this sample
            sample_atom_mask = batch_idx == b
            sample_atoms = atom_positions[sample_atom_mask]  # [n_atoms, 3]
            
            if len(sample_atoms) == 0:
                continue
            
            # Compute distance to nearest atom for each query point
            # sample_query_points: [n_points, 3], sample_atoms: [n_atoms, 3]
            # Compute pairwise distances: [n_points, n_atoms]
            distances = torch.cdist(sample_query_points, sample_atoms)  # [n_points, n_atoms]
            min_distances = distances.min(dim=-1)[0]  # [n_points] - distance to nearest atom
            
            # Convert distance to weight: closer atoms = higher weight
            # Use exponential decay: weight = exp(-distance / scale)
            atom_distance_weights = torch.exp(-min_distances / self.atom_distance_scale)
            
            # Apply weight scaling
            combined_weights = self.atom_distance_weight * atom_distance_weights
            
            # Normalize weights to have mean=1 (preserve overall loss scale)
            weight_mean = combined_weights.mean()
            if weight_mean > 0:
                combined_weights = combined_weights / weight_mean
            
            weights[b] = combined_weights
        
        return weights
    
    def _process_batch(self, batch):
        """
        Process a batch and return the predicted and target fields
        
        Args:
            batch: PyTorch Geometric batch object
            
        Returns:
            tuple: (pred_field, target_field) with matching dimensions [B, n_points, n_atom_types, 3]
        """
        # Get predictions from forward pass
        pred_field = self(batch)  # [B, n_points, n_atom_types, 3]
        
        # Get target field from batch
        target_field = batch.target_field
        
        # Get batch size and number of points
        B = len(batch)
        n_points = self.config["dset"]["n_points"]
        n_atom_types = self.config["dset"]["n_channels"]
        
        # Reshape target_field to [B, n_points, n_atom_types, 3]
        if target_field.dim() == 2:
            # [B*n_points*n_atom_types*3] -> [B, n_points, n_atom_types, 3]
            target_field = target_field.view(B, n_points, n_atom_types, 3)
        elif target_field.dim() == 3:
            if target_field.shape[0] == B * n_points:
                # [B*n_points, n_atom_types, 3] -> [B, n_points, n_atom_types, 3]
                target_field = target_field.view(B, n_points, n_atom_types, 3)
            elif target_field.shape[0] == n_points:
                # [n_points, n_atom_types, 3] -> [B, n_points, n_atom_types, 3]
                target_field = target_field.unsqueeze(0).expand(B, -1, -1, -1)
            else:
                # Assume it's already [B, n_points, n_atom_types, 3] or try to reshape
                target_field = target_field.view(B, n_points, n_atom_types, 3)
        elif target_field.dim() == 4:
            # Already in correct shape [B, n_points, n_atom_types, 3]
            pass
        else:
            raise ValueError(f"Unexpected target_field dimension: {target_field.dim()}, shape: {target_field.shape}")
        
        # Ensure pred_field has the same shape
        if pred_field.shape != target_field.shape:
            pred_field = pred_field.view(B, n_points, n_atom_types, 3)
        
        return pred_field, target_field
    
    def training_step(self, batch, batch_idx):
        """Training step logic"""
        # Get predictions and targets
        pred_field, target_field = self._process_batch(batch)
        
        # Calculate loss with optional weighting
        if self.loss_weighting_enabled:
            # Get query points
            query_points = batch.xs
            B = len(batch)
            n_points = self.config["dset"]["n_points"]
            if query_points.dim() == 2:
                query_points = query_points.view(B, n_points, 3)
            
            # Compute weights for each query point
            weights = self._compute_loss_weights(batch, query_points, target_field)  # [B, n_points]
            
            # Compute element-wise MSE loss
            # pred_field: [B, n_points, n_atom_types, 3]
            # target_field: [B, n_points, n_atom_types, 3]
            elementwise_loss = self.criterion(pred_field, target_field)  # [B, n_points, n_atom_types, 3]
            
            # Average over atom_types and spatial dimensions, keep batch and point dimensions
            pointwise_loss = elementwise_loss.mean(dim=(-2, -1))  # [B, n_points]
            
            # Apply weights
            weighted_loss = pointwise_loss * weights  # [B, n_points]
            
            # Average over batch and points
            loss = weighted_loss.mean()
        else:
            # Standard MSE loss
            loss = self.criterion(pred_field, target_field)
        
        # Update metrics
        # self.train_metrics["loss"](loss.detach())
        # self.train_metrics["recon_loss"](loss.detach())
        
        # Log metrics
        self.log('train_loss', loss, batch_size=len(batch),
                 on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        # Store batch loss for later visualization
        # batch_loss_item = {"total_loss": loss.item()}
        # self.batch_losses.append(batch_loss_item)
        
        # Clean up GPU memory periodically
        # if torch.cuda.is_available() and batch_idx % 50 == 0:
        #     torch.cuda.empty_cache()
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step logic"""
        # Get predictions and targets
        pred_field, target_field = self._process_batch(batch)
        
        # Calculate loss with optional weighting
        if self.loss_weighting_enabled:
            # Get query points
            query_points = batch.xs
            B = len(batch)
            n_points = self.config["dset"]["n_points"]
            if query_points.dim() == 2:
                query_points = query_points.view(B, n_points, 3)
            
            # Compute weights for each query point
            weights = self._compute_loss_weights(batch, query_points, target_field)  # [B, n_points]
            
            # Compute element-wise MSE loss
            elementwise_loss = self.criterion(pred_field, target_field)  # [B, n_points, n_atom_types, 3]
            
            # Average over atom_types and spatial dimensions, keep batch and point dimensions
            pointwise_loss = elementwise_loss.mean(dim=(-2, -1))  # [B, n_points]
            
            # Apply weights
            weighted_loss = pointwise_loss * weights  # [B, n_points]
            
            # Average over batch and points
            loss = weighted_loss.mean()
        else:
            # Standard MSE loss
            loss = self.criterion(pred_field, target_field)
        
        # Log metrics
        self.log('val_loss', loss, batch_size=len(batch),
                 on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        return loss
    
    def on_train_epoch_start(self):
        """Called at the beginning of training epoch"""
        # Set models to training mode
        # In fine-tuning mode, keep encoder in eval mode
        if self.finetune_enabled:
            self.enc.eval()  # Keep encoder in eval mode when frozen
        else:
            self.enc.train()
        self.dec.train()
        
        # Clean up GPU memory
        # if torch.cuda.is_available():
        #     torch.cuda.empty_cache()
            # if self.trainer.is_global_zero:
            #     print(f"CUDA memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            #     print(f"CUDA memory cached: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    
    def on_validation_epoch_start(self):
        """Called at the beginning of validation epoch"""
        # Set models to evaluation mode
        self.enc.eval()
        self.dec.eval()
    
    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers"""
        # In fine-tuning mode, only optimize decoder parameters
        if self.finetune_enabled:
            # Only optimize decoder parameters
            optimizer = torch.optim.Adam(
                [{"params": self.dec.parameters(), "lr": self.config["dset"]["lr_dec"]}]
            )
            print("Optimizer configured for fine-tuning: only decoder parameters will be updated")
        else:
            # Create a single optimizer with different parameter groups for encoder and decoder
            optimizer = torch.optim.Adam([
                {"params": self.enc.parameters(), "lr": self.config["dset"]["lr_enc"]},
                {"params": self.dec.parameters(), "lr": self.config["dset"]["lr_dec"]}
            ])
        
        # Create learning rate scheduler if needed
        if "lr_decay" in self.config and self.config["lr_decay"]:
            # Default milestone if not specified in config
            milestones = self.config.get("lr_milestones", [500])
            gamma = self.config.get("lr_gamma", 0.1)
                
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=milestones, gamma=gamma
            )
            
            # Configure the scheduler with proper interval
            scheduler_config = {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
                "monitor": "val_loss"
            }
            
            return {"optimizer": optimizer, "lr_scheduler": scheduler_config}
        
        return optimizer
    
    def on_save_checkpoint(self, checkpoint):        
        # 保存前去掉多余的封装（但是lightning模式理论上会自动处理好，不需要去除，只是以防万一）
        enc_state_dict = self.enc.module.state_dict() if hasattr(self.enc, "module") else self.enc.state_dict()
        dec_state_dict = self.dec.module.state_dict() if hasattr(self.dec, "module") else self.dec.state_dict()
        
        checkpoint["enc_state_dict"] = enc_state_dict
        checkpoint["dec_state_dict"] = dec_state_dict
                    

@hydra.main(config_path="configs", config_name="train_nf_qm9", version_base=None)
# @hydra.main(config_path="configs", config_name="train_nf_drugs", version_base=None)
def main_hydra(config):
    """Entry point for Hydra configuration system"""
    main(config)

def main(config):
    """Main function to set up and run the training"""    
    # Get available GPU count
    num_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {num_gpus}")
    
    # Setup GNF Converter for data processing
    data_gnf_converter = create_gnf_converter(config)
    
    # Create data loaders
    try:
        loader_train = create_field_loaders(config, data_gnf_converter, split="train")
        loader_val = create_field_loaders(config, data_gnf_converter, split="val")
        
        # # Handle cases where loaders are returned as lists
        # if loader_train is not None and isinstance(loader_train, list):
        #     print(f"Train loader returned as list with {len(loader_train)} items, using first item")
        #     loader_train = loader_train[0]
        # if loader_val is not None and isinstance(loader_val, list):
        #     print(f"Val loader returned as list with {len(loader_val)} items, using first item")
        #     loader_val = loader_val[0]
            
        # Check if loaders are empty
        if not loader_train or len(loader_train) == 0:
            raise ValueError("Training data loader is empty")
            
        print(f"Created data loaders - Train: {len(loader_train) if loader_train else 0} batches, Val: {len(loader_val) if loader_val else 0} batches")
    except Exception as e:
        print(f"Error creating data loaders: {e}")
        raise
    
    # Create GNF Converter for model training
    gnf_converter = create_gnf_converter(config)
    
    # Initialize Lightning model
    model = NeuralFieldLightningModule(config, gnf_converter)
    
    # Load checkpoint if specified
    if config["reload_model_path"] is not None:
        try:
            checkpoint_path = config["reload_model_path"]
            
            # Verify checkpoint file exists
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
            
            if not checkpoint_path.endswith('.ckpt'):
                raise ValueError(f"Checkpoint path must be a .ckpt file, got: {checkpoint_path}")
            
            print(f"Loading Lightning checkpoint file: {checkpoint_path}")
            training_state = load_checkpoint_state_nf(model, checkpoint_path)
            
            # Apply training state
            if training_state["epoch"] is not None:
                print(f"Resuming from epoch {training_state['epoch']}")
            
            model.train_losses = training_state["train_losses"]
            model.val_losses = training_state["val_losses"]
            model.best_loss = training_state["best_loss"]
            
            # Ensure encoder remains frozen if fine-tuning is enabled
            finetune_config = config.get("finetune_decoder", {})
            if finetune_config.get("enabled", False):
                freeze_encoder = finetune_config.get("freeze_encoder", True)
                if freeze_encoder:
                    for param in model.enc.parameters():
                        param.requires_grad = False
                    model.enc.eval()
                    print("Encoder re-frozen after checkpoint loading (fine-tuning mode)")
                
            print(f"Successfully loaded checkpoint from: {config['reload_model_path']}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            raise
    
    # Create directories for output
    os.makedirs(config["dirname"], exist_ok=True)
    
    # Configure callbacks
    callbacks = [
        ModelCheckpoint(
            # dirpath=config["dirname"],
            # filename="model-{epoch:02d}-{val_loss:.4f}",
            filename="model-{epoch:02d}",
            every_n_epochs=config["ckpt_every_n_epochs"],
            monitor="val_loss",
            mode="min",
            save_last=True,
            save_top_k=-1  # 每ckpt_every_n_epochs保存所有checkpoint，不限制数量
        ),
        LearningRateMonitor(logging_interval="epoch"),
        # ProgressBarCallback()
    ]
    
    # Configure logger
    logger = TensorBoardLogger(
        save_dir=config["dirname"],
        # name=config["exp_name"],
        default_hp_metric=False
    )
    
    # Configure trainer
    
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=num_gpus,
        max_epochs=config["n_epochs"],
        callbacks=callbacks,
        logger=logger,
        strategy='auto' if num_gpus == 1 else DDPStrategy(find_unused_parameters=True),
        precision="bf16-mixed",
        # log_every_n_steps=50,
        enable_checkpointing=True,
        check_val_every_n_epoch=config["eval_every"],
        # Use built-in gradient clipping
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        # limit_train_batches=0.001,
        # limit_val_batches=0.01,
    )
    

    # empty iterate loader_val
    # for batch in loader_train:
    #     print(batch)
    
    
    # Train the model
    trainer.fit(model, loader_train, loader_val)
    
    # Save final model and plots
    # if trainer.is_global_zero:
    #     # Save the model
    #     torch.save({
    #         'enc': model.enc.state_dict(),
    #         'dec': model.dec.state_dict()
    #     }, os.path.join(config["dirname"], "final_model.pt"))
    # print(f"Training completed. Models saved to {config['dirname']}")
    print("Training done.")


if __name__ == "__main__":
    main_hydra()
