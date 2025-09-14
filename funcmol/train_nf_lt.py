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
# import torchmetrics
torch.set_float32_matmul_precision('medium')
# PyTorch Lightning
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.strategies import DDPStrategy
# from pytorch_lightning.utilities.rank_zero import rank_zero_only

# Configuration management
from omegaconf import OmegaConf
import hydra

# TODO: set gpus based on server id
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3,4,5,6,7"
# os.environ['CUDA_VISIBLE_DEVICES'] = "0,2,3,4,5"
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"


from funcmol.utils.utils_nf import create_neural_field
from funcmol.dataset.dataset_field import create_field_loaders, create_gnf_converter
from funcmol.utils.gnf_converter import GNFConverter


def create_training_gnf_converter(config, device="cuda"):
    """
    Create a GNF converter instance for training based on configuration.
    
    Args:
        config (dict): Configuration dictionary containing GNF parameters
        device (str): Device to place the converter on
        
    Returns:
        GNFConverter: Configured GNF converter instance ready for training
    """
    # Extract configuration
    gnf_config = config.get("converter")
    if gnf_config is not None and not isinstance(gnf_config, dict):
        gnf_config = OmegaConf.to_container(gnf_config, resolve=True)
    if isinstance(gnf_config, list):
        gnf_config = gnf_config[0]
    assert isinstance(gnf_config, dict), f"gnf_config should be dict, got {type(gnf_config)}"
    
    # Get method-specific parameters
    method = gnf_config.get("gradient_field_method", "softmax")
    method_config = gnf_config.get("method_configs", {}).get(method, gnf_config.get("default_config", {}))
    
    # Define required parameters
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
    
    # Define method-specific required parameters
    method_required_params = {
        'n_query_points': method_config.get("n_query_points"),
        'step_size': method_config.get("step_size"),
        'sig_sf': method_config.get("sig_sf"),
        'sig_mag': method_config.get("sig_mag")
    }
    
    # Check for missing parameters
    missing_params = [param for param, value in required_params.items() if value is None]
    missing_method_params = [param for param, value in method_required_params.items() if value is None]
    
    if missing_params:
        raise ValueError(f"Missing required parameters in gnf_config: {missing_params}")
    if missing_method_params:
        raise ValueError(f"Missing required parameters in method_config for {method}: {missing_method_params}")
    
    # Build parameters for GNF Converter
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
        'n_atom_types': config["dset"]["n_channels"]
    }
    
    # Create and return the GNF converter
    return GNFConverter(**gnf_converter_params)


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


# @rank_zero_only
# def print_model_sizes(enc, dec):
#     """Print the number of parameters in the encoder and decoder models."""
#     print('Num of params in encoder:', sum(p.numel() for p in enc.parameters()))
#     print('Num of params in decoder:', sum(p.numel() for p in dec.parameters()))


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
        
        # Set up loss function
        self.criterion = nn.MSELoss()
        
        # Track losses for plotting
        # self.train_losses = []
        # self.val_losses = []
        # self.batch_losses = []
        # self.best_loss = float("inf")
        
    def _create_models(self):
        """Create encoder and decoder neural field models"""
        try:
            # Create encoder and decoder using utility function
            enc, dec = create_neural_field(self.config, self)
            
            # Print model sizes only for rank 0
            # print_model_sizes(enc, dec)
            
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
        codes = self.enc(batch)
        
        # Check and reshape query points if needed
        if query_points.dim() == 2:
            B = len(batch)
            n_points = self.config["dset"]["n_points"]
            query_points = query_points.view(B, n_points, 3)
            
        # Get predicted field from decoder
        pred_field = self.dec(query_points, codes)
        
        return pred_field
    
    def _process_batch(self, batch):
        """
        Process a batch and return the predicted and target fields
        
        Args:
            batch: PyTorch Geometric batch object
            
        Returns:
            tuple: (pred_field, target_field) with matching dimensions
        """
        # Get predictions from forward pass
        pred_field = self(batch)
        
        # Get target field from batch
        target_field = batch.target_field
        
        # Ensure correct shapes for both fields
        pred_field = pred_field.view_as(target_field)
        
        return pred_field, target_field
    
    def training_step(self, batch, batch_idx):
        """Training step logic"""
        # Get predictions and targets
        pred_field, target_field = self._process_batch(batch)
        
        # Calculate MSE loss
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
        
        # Calculate loss
        loss = self.criterion(pred_field, target_field)
        
        # Log metrics
        self.log('val_loss', loss, batch_size=len(batch),
                 on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        return loss
    
    def on_train_epoch_start(self):
        """Called at the beginning of training epoch"""
        # Set models to training mode
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
    #     """Custom checkpoint saving logic"""
    #     checkpoint["train_losses"] = self.train_losses
    #     checkpoint["val_losses"] = self.val_losses
    #     checkpoint["batch_losses"] = self.batch_losses
    #     checkpoint["best_loss"] = self.best_loss
        
        # Save encoder and decoder state separately for compatibility with original code
        checkpoint["enc_state_dict"] = self.enc.state_dict()
        checkpoint["dec_state_dict"] = self.dec.state_dict()
        
    # def on_load_checkpoint(self, checkpoint):
    #     """Custom checkpoint loading logic"""
    #     if "train_losses" in checkpoint:
    #         self.train_losses = checkpoint["train_losses"]
    #     if "val_losses" in checkpoint:
    #         self.val_losses = checkpoint["val_losses"]
    #     if "batch_losses" in checkpoint:
    #         self.batch_losses = checkpoint["batch_losses"]    
    #     if "best_loss" in checkpoint:
    #         self.best_loss = checkpoint["best_loss"]
            
    def get_progress_bar_dict(self):
        """Customize the progress bar information"""
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        
        # Add learning rates to progress bar
        for i, optimizer in enumerate(self.trainer.optimizers):
            for j, param_group in enumerate(optimizer.param_groups):
                name = f"lr_{i}_{j}" if j > 0 else f"lr_{i}"
                items[name] = param_group["lr"]
        
        return items


@hydra.main(config_path="configs", config_name="train_nf_qm9", version_base=None)
# @hydra.main(config_path="configs", config_name="train_nf_drugs", version_base=None)
def main_hydra(config):
    """Entry point for Hydra configuration system"""
    main(config)

def main(config):
    """Main function to set up and run the training"""
    # Set environment variables
    # os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    # os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'OFF'
    # os.environ['LIGHTNING_CLI_USAGE'] = 'OFF'
    
    # # Set NCCL environment variables to avoid communication issues
    # os.environ['NCCL_DEBUG'] = 'WARN'
    # os.environ['NCCL_TIMEOUT'] = '1800'
    # os.environ['NCCL_IB_DISABLE'] = '1'
    # os.environ['NCCL_P2P_DISABLE'] = '1'
    # os.environ['NCCL_SOCKET_IFNAME'] = 'lo'
    # os.environ['TORCH_NCCL_BLOCKING_WAIT'] = '1'
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
    # os.environ['NCCL_ASYNC_ERROR_HANDLING'] = '1'
    # os.environ['NCCL_BUFFSIZE'] = '2097152'
    # os.environ['NCCL_NTHREADS'] = '4'
    
    # Get available GPU count
    num_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {num_gpus}")
    
    # Setup GNF Converter for data processing
    data_gnf_converter = create_gnf_converter(config)
    
    # Create data loaders
    try:
        loader_train = create_field_loaders(config, data_gnf_converter, split="train", use_fabric=False)
        loader_val = create_field_loaders(config, data_gnf_converter, split="val", use_fabric=False)
        
        # Handle cases where loaders are returned as lists
        if isinstance(loader_train, list):
            print(f"Train loader returned as list with {len(loader_train)} items, using first item")
            loader_train = loader_train[0]
        if isinstance(loader_val, list):
            print(f"Val loader returned as list with {len(loader_val)} items, using first item")
            loader_val = loader_val[0]
            
        # Check if loaders are empty
        if not loader_train or len(loader_train) == 0:
            raise ValueError("Training data loader is empty")
            
        print(f"Created data loaders - Train: {len(loader_train)} batches, Val: {len(loader_val) if loader_val else 0} batches")
    except Exception as e:
        print(f"Error creating data loaders: {e}")
        raise
    
    # Create GNF Converter for model training
    # TODO: check if training and data converters can be unified
    gnf_converter = create_training_gnf_converter(config, 
                        device="cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize Lightning model
    model = NeuralFieldLightningModule(config, gnf_converter)
    
    # Load checkpoint if specified
    if config["reload_model_path"] is not None:
        try:
            def load_checkpoint_state(model, checkpoint_path):
                """Helper function to load checkpoint state"""
                print(f"Loading checkpoint from: {checkpoint_path}")
                state_dict = torch.load(checkpoint_path)
                
                # Load encoder state dict
                if "enc" in state_dict:
                    model.enc.load_state_dict(state_dict["enc"])
                    print("Loaded encoder state from 'enc' key")
                elif "enc_state_dict" in state_dict:
                    model.enc.load_state_dict(state_dict["enc_state_dict"])
                    print("Loaded encoder state from 'enc_state_dict' key")
                else:
                    print("No encoder state found in checkpoint")
                
                # Load decoder state dict
                if "dec" in state_dict:
                    model.dec.load_state_dict(state_dict["dec"])
                    print("Loaded decoder state from 'dec' key")
                elif "dec_state_dict" in state_dict:
                    model.dec.load_state_dict(state_dict["dec_state_dict"])
                    print("Loaded decoder state from 'dec_state_dict' key")
                else:
                    print("No decoder state found in checkpoint")
                
                # Load training state
                training_state = {
                    "epoch": state_dict.get("epoch", None),
                    "train_losses": state_dict.get("train_losses", []),
                    "val_losses": state_dict.get("val_losses", []),
                    "best_loss": state_dict.get("best_loss", float("inf"))
                }
                
                return training_state
            
            checkpoint_path = os.path.join(config["reload_model_path"], "model.pt")
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
            save_top_k=-1  # 保存所有checkpoint，不限制数量
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
