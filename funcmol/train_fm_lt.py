# ========== 共享服务器线程限制配置（中和配置：80%情况最优）==========
# 在 import torch 之前设置环境变量，限制底层库的线程数
import os
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"
# ==========================================

import sys
sys.path.append("..")

# Set GPU environment
# os.environ['CUDA_VISIBLE_DEVICES'] = "1"
# os.environ['CUDA_VISIBLE_DEVICES'] = "4,5,6,7"
os.environ['CUDA_VISIBLE_DEVICES'] = "0,2,3,4,5"

# Data visualization and processing
import matplotlib.pyplot as plt
import numpy as np
import pickle
    
# PyTorch and related libraries
import torch
# 限制 PyTorch 自己的线程数（中和配置：80%情况最优）
torch.set_num_threads(2)
torch.set_num_interop_threads(2)

import torch.nn as nn
import torch.nn.functional as F
torch.set_float32_matmul_precision('medium')

# PyTorch Lightning
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.strategies import DDPStrategy

# Configuration management
from omegaconf import OmegaConf
import hydra

from funcmol.models.funcmol import create_funcmol
from funcmol.utils.utils_fm import (
    add_noise_to_code,
    compute_code_stats_offline, compute_codes,
    load_checkpoint_state_fm,
    compute_position_weights
)
from funcmol.models.adamw import AdamW
from funcmol.models.ema import ModelEma
from funcmol.utils.utils_nf import (
    infer_codes_occs_batch, 
    load_neural_field, 
    normalize_code,
    reshape_target_field,
    compute_decoder_field_loss
)
from funcmol.dataset.dataset_code import create_code_loaders
from funcmol.dataset.dataset_field import create_field_loaders, create_gnf_converter
from funcmol.models.encoder import create_grid_coords
from funcmol.utils.constants import PADDING_INDEX


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
    def __init__(self, config, enc, dec_module, code_stats, field_loader_train=None, field_loader_val=None):
        super().__init__()
        self.config = config
        self.enc = enc
        self.dec_module = dec_module
        self.code_stats = code_stats
        self.save_hyperparameters(ignore=['enc', 'dec_module', 'code_stats', 'field_loader_train', 'field_loader_val'])
        
        # Store field loaders for position weight computation when on_the_fly=False
        self.field_loader_train = field_loader_train
        self.field_loader_val = field_loader_val
        self._field_dataset_train = field_loader_train.dataset if field_loader_train is not None else None
        self._field_dataset_val = field_loader_val.dataset if field_loader_val is not None else None
        
        # Create Funcmol model
        self.funcmol = create_funcmol(config)
        
        # Set up loss function
        self.criterion = nn.MSELoss(reduction="mean")
        
        # Initialize EMA model
        with torch.no_grad():
            self.funcmol_ema = ModelEma(self.funcmol, decay=config["ema_decay"])

        # Joint fine-tuning configuration
        joint_finetune_config = config.get("joint_finetune", {})
        self.joint_finetune_enabled = joint_finetune_config.get("enabled", False)
        self.decoder_loss_weight = joint_finetune_config.get("decoder_loss_weight", 1.0)
        self.decoder_lr = joint_finetune_config.get("decoder_lr", None)
        
        # New configuration for enhanced joint fine-tuning
        self.num_timesteps_for_decoder = joint_finetune_config.get("num_timesteps_for_decoder", 50)
        # atom_distance_threshold: 优先从 joint_finetune 读取，否则从 dset 读取
        if "atom_distance_threshold" in joint_finetune_config and joint_finetune_config["atom_distance_threshold"] is not None:
            self.atom_distance_threshold = joint_finetune_config["atom_distance_threshold"]
        else:
            self.atom_distance_threshold = config.get("dset", {}).get("atom_distance_threshold", 0.5)
        self.magnitude_loss_weight = joint_finetune_config.get("magnitude_loss_weight", 0.1)
        self.use_cosine_loss = joint_finetune_config.get("use_cosine_loss", False)  # 默认使用MSE loss
        
        # atom_distance_scale: 用于 loss weighting 的指数衰减
        # 优先从 joint_finetune 读取，否则从 loss_weighting 读取，最后使用默认值
        if "atom_distance_scale" in joint_finetune_config and joint_finetune_config["atom_distance_scale"] is not None:
            self.atom_distance_scale = joint_finetune_config["atom_distance_scale"]
        else:
            loss_weighting_config = config.get("loss_weighting", {})
            self.atom_distance_scale = loss_weighting_config.get("atom_distance_scale", 0.5)
        
        # Set up decoder loss function for joint fine-tuning
        if self.joint_finetune_enabled:
            # Keep MSE loss for magnitude loss component
            self.decoder_criterion = nn.MSELoss(reduction="mean")
            print(f"Joint fine-tuning enabled: decoder_loss_weight={self.decoder_loss_weight}")
            if self.decoder_lr is not None:
                print(f"Decoder learning rate: {self.decoder_lr}")
            else:
                print(f"Decoder will use denoiser learning rate: {config['lr']}")
            if self.use_cosine_loss:
                print(f"Using cosine distance + magnitude loss (magnitude_weight={self.magnitude_loss_weight})")
            else:
                print(f"Using MSE loss for decoder training")
            print(f"Decoder training: randomly sampling one timestep from [0, {self.num_timesteps_for_decoder}) for slight noise perturbation, then denoising")
            print(f"Query points sampling: targeted_sampling_ratio from config, atom_distance_threshold={self.atom_distance_threshold}Å, atom_distance_scale={self.atom_distance_scale}")

        # Field loss finetune configuration (decoder frozen)
        field_loss_finetune_config = config.get("field_loss_finetune", {})
        self.field_loss_finetune_enabled = field_loss_finetune_config.get("enabled", False)
        self.field_loss_weight = field_loss_finetune_config.get("field_loss_weight", 1.0)
        self.num_timesteps_for_field_loss = field_loss_finetune_config.get("num_timesteps_for_field_loss", 5)
        
        # Get atom_distance_scale from config (for exp decay weighting)
        # Check if it's in loss_weighting config (from neural field training) or use default
        loss_weighting_config = config.get("loss_weighting", {})
        self.field_loss_atom_distance_scale = loss_weighting_config.get("atom_distance_scale", 0.5)
        
        if self.field_loss_finetune_enabled:
            print(f"Field loss finetune enabled: field_loss_weight={self.field_loss_weight}")
            print(f"Decoder will remain frozen (not updated)")
            print(f"Field loss: each sample independently samples a timestep from [0, {self.num_timesteps_for_field_loss})")
            print(f"Using MSE loss with exp decay weighting (atom_distance_scale={self.field_loss_atom_distance_scale})")
            print(f"Query points sampling: using dataset's _get_xs method (mix of grid and neighbor points)")

        self._freeze_nf()
        
        # Position weight configuration
        self.position_weight_config = config.get("position_weight", {})
        self.use_position_weight = self.position_weight_config.get("enabled", False)
        
        # Position weight augmentation info (for modulo mapping when codes are augmented)
        # These are set from config in main() when datasets are created
        self.num_augmentations_train = config.get("num_augmentations_train")
        self.num_augmentations_val = config.get("num_augmentations_val")
        self.field_dataset_size_train = config.get("field_dataset_size_train")
        self.field_dataset_size_val = config.get("field_dataset_size_val")
        
        # Track losses for plotting # TODO: remove these
        self.train_losses = []
        self.val_losses = []
        self.best_loss = float("inf")

    def _freeze_nf(self):
        """Freeze the neural field"""
        if self.enc is not None:
            for param in self.enc.parameters():
                param.requires_grad = False
            self.enc.eval()
            print("Encoder frozen (always frozen)")
        
        # Freeze or unfreeze decoder based on joint fine-tuning configuration
        # Note: field_loss_finetune keeps decoder frozen (only provides supervision signal)
        if self.dec_module is not None:
            if self.joint_finetune_enabled:
                # Unfreeze decoder for joint fine-tuning
                for param in self.dec_module.parameters():
                    param.requires_grad = True
                print("Decoder unfrozen for joint fine-tuning")
            else:
                # Freeze decoder (default behavior, also for field_loss_finetune)
                for param in self.dec_module.parameters():
                    param.requires_grad = False
                self.dec_module.eval()  # Set to eval mode when frozen
                if self.field_loss_finetune_enabled:
                    print("Decoder frozen (field loss finetune mode - decoder provides supervision only)")
                else:
                    print("Decoder frozen")
        
    def _process_batch(self, batch):
        """
        Process a batch and return codes, smooth_codes, and position_weights (if available)
        
        Args:
            batch: Data batch (may be tuple of (codes, position_weights) if position_weights are pre-computed)
            
        Returns:
            tuple: (codes, smooth_codes, position_weights) or (codes, smooth_codes, None)
        """
        # Check if batch contains position_weights (from dataset)
        # DataLoader会将多个样本的返回值组合：如果dataset返回tuple，DataLoader返回tuple of lists
        position_weights = None
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            # Dataset returns (codes, position_weights), DataLoader返回(codes_list, position_weights_list)
            codes_list, position_weights_list = batch
            # Stack codes
            if isinstance(codes_list, list):
                codes = torch.stack(codes_list)
            else:
                codes = codes_list
            # Stack position_weights
            if isinstance(position_weights_list, list):
                position_weights = torch.stack(position_weights_list)
            else:
                position_weights = position_weights_list
        else:
            # Dataset returns only codes
            if isinstance(batch, list):
                codes = torch.stack(batch)
            else:
                codes = batch
        
        if self.config["on_the_fly"]:
            with torch.no_grad():
                codes = infer_codes_occs_batch(
                    batch, self.enc, self.config, to_cpu=False,
                    code_stats=self.code_stats if self.config["normalize_codes"] else None
                )
        else:
            # Only normalize codes if normalize_codes is enabled and code_stats is available
            if self.config["normalize_codes"] and self.code_stats is not None:
                codes = normalize_code(codes, self.code_stats)
        
        with torch.no_grad():
            smooth_codes = add_noise_to_code(codes, smooth_sigma=self.config["smooth_sigma"])
        
        # Move position_weights to the same device as codes
        if position_weights is not None:
            position_weights = position_weights.to(codes.device)
        
        return codes, smooth_codes, position_weights
    
    def _compute_position_weights(self, batch, codes, batch_idx=None, is_training=True, precomputed_weights=None):
        """
        Get position weights for codes. First try to use precomputed weights from dataset,
        otherwise compute on-the-fly (for on_the_fly=True mode).
        
        Args:
            batch: Data batch (PyTorch Geometric Batch if on_the_fly=True, or codes tensor if False)
            codes: Codes tensor [B, n_grid, code_dim]
            batch_idx: Batch index in DataLoader (required for on_the_fly=False mode)
            is_training: Whether this is training step (True) or validation step (False)
            precomputed_weights: Pre-computed position weights from dataset [B, n_grid] or None
            
        Returns:
            position_weights: [B, n_grid] or None if position weighting is disabled
        """
        if not self.use_position_weight:
            return None
        
        # If precomputed weights are available, use them (much faster!)
        if precomputed_weights is not None:
            return precomputed_weights
        
        device = codes.device
        B = codes.shape[0]
        grid_size = self.config["dset"]["grid_size"]
        anchor_spacing = self.config["dset"]["anchor_spacing"]
        radius = self.position_weight_config.get("radius", 3.0)
        weight_alpha = self.position_weight_config.get("alpha", 0.5)
        
        # Get grid coordinates
        grid_coords = create_grid_coords(1, grid_size, device=device, anchor_spacing=anchor_spacing)
        grid_coords = grid_coords.squeeze(0)  # [n_grid, 3]
        
        if self.config["on_the_fly"]:
            # Extract atom coordinates from batch (PyTorch Geometric Batch)
            atom_coords = batch.pos  # [N_total_atoms, 3]
            batch_idx_atoms = batch.batch  # [N_total_atoms]
            
            # Compute position weights
            position_weights = compute_position_weights(
                atom_coords=atom_coords,
                grid_coords=grid_coords,
                batch_idx=batch_idx_atoms,
                radius=radius,
                weight_alpha=weight_alpha,
                device=device
            )
        else:
            # For on_the_fly=False, get atom coordinates from FieldDataset using index mapping
            # Since codes and molecules have matching order (shuffle disabled), we can use batch_idx
            # to calculate the dataset indices for this batch
            
            # Select the appropriate dataset (train or val)
            field_dataset = self._field_dataset_train if is_training else self._field_dataset_val
            
            if field_dataset is None:
                # If field dataset is not available, return None (position weighting disabled)
                return None
            
            if batch_idx is None:
                # batch_idx is required for on_the_fly=False mode
                print("[WARNING] batch_idx is required for position weighting in on_the_fly=False mode")
                return None
            
            try:
                # Calculate the dataset indices for this batch
                batch_size = self.config["dset"]["batch_size"]
                start_idx = batch_idx * batch_size
                end_idx = start_idx + B  # Use actual batch size (B) in case of last batch
                
                # Get augmentation info for modulo mapping
                num_augmentations = self.num_augmentations_train if is_training else self.num_augmentations_val
                field_dataset_size = self.field_dataset_size_train if is_training else self.field_dataset_size_val
                
                # If codes are augmented (num_augmentations > 1), use modulo mapping
                # codes[i] corresponds to field_dataset[i % field_dataset_size]
                if num_augmentations is not None and num_augmentations > 1 and field_dataset_size is not None:
                    # Use modulo mapping: codes index -> field dataset index
                    # codes[start_idx...end_idx] -> field_dataset[start_idx % field_dataset_size ... end_idx % field_dataset_size]
                    # But we need to handle the mapping for each code in the batch
                    all_atom_coords = []
                    all_batch_indices = []
                    
                    for code_idx in range(start_idx, end_idx):
                        # Map codes index to field dataset index using modulo
                        field_idx = code_idx % field_dataset_size
                        
                        try:
                            # Get molecule data from FieldDataset using modulo-mapped index
                            mol_data = field_dataset[field_idx]
                            
                            # Extract atom coordinates
                            if hasattr(mol_data, 'pos'):
                                atom_coords = mol_data.pos  # [N_atoms, 3]
                                all_atom_coords.append(atom_coords)
                                # Create batch index for this molecule (within the current batch)
                                batch_idx_for_mol = len(all_batch_indices)
                                num_atoms = atom_coords.shape[0]
                                all_batch_indices.append(torch.full((num_atoms,), batch_idx_for_mol, dtype=torch.long))
                            else:
                                print(f"[WARNING] Molecule at field index {field_idx} (codes index {code_idx}) does not have 'pos' attribute")
                                return None
                        except Exception as e:
                            print(f"[WARNING] Failed to get molecule at field index {field_idx} (codes index {code_idx}): {e}")
                            return None
                else:
                    # No augmentation or augmentation info not available, use direct mapping
                    # Verify dataset length matches
                    if start_idx >= len(field_dataset):
                        print(f"[WARNING] batch_idx {batch_idx} is out of range for dataset (size: {len(field_dataset)})")
                        return None
                    
                    # Clamp end_idx to dataset size
                    end_idx = min(end_idx, len(field_dataset))
                    
                    # Get molecule data for this batch from FieldDataset
                    all_atom_coords = []
                    all_batch_indices = []
                    
                    for i in range(start_idx, end_idx):
                        try:
                            # Get molecule data from FieldDataset
                            mol_data = field_dataset[i]
                            
                            # Extract atom coordinates
                            if hasattr(mol_data, 'pos'):
                                atom_coords = mol_data.pos  # [N_atoms, 3]
                                all_atom_coords.append(atom_coords)
                                # Create batch index for this molecule (within the current batch)
                                batch_idx_for_mol = len(all_batch_indices)
                                num_atoms = atom_coords.shape[0]
                                all_batch_indices.append(torch.full((num_atoms,), batch_idx_for_mol, dtype=torch.long))
                            else:
                                print(f"[WARNING] Molecule at index {i} does not have 'pos' attribute")
                                return None
                        except Exception as e:
                            print(f"[WARNING] Failed to get molecule at index {i}: {e}")
                            return None
                
                if not all_atom_coords:
                    print("[WARNING] No atom coordinates retrieved from FieldDataset")
                    return None
                
                # Concatenate all atom coordinates
                atom_coords = torch.cat(all_atom_coords, dim=0).to(device)  # [N_total_atoms, 3]
                batch_idx_atoms = torch.cat(all_batch_indices, dim=0).to(device)  # [N_total_atoms]
                
                # Verify we got the correct number of molecules
                actual_num_molecules = len(all_atom_coords)
                if actual_num_molecules != B:
                    print(f"[WARNING] Expected {B} molecules, got {actual_num_molecules}")
                    # Adjust position_weights shape if needed
                    # This can happen in the last batch if drop_last=False
                    if actual_num_molecules < B:
                        # If we got fewer molecules, we can only compute weights for those
                        # This should not happen if drop_last=True, but handle it gracefully
                        print(f"[WARNING] Batch size mismatch: codes has {B} samples but only {actual_num_molecules} molecules retrieved")
                        # We'll compute weights for available molecules and pad if needed
                        # But this is an error condition, so return None
                        return None
                
                # Compute position weights
                position_weights = compute_position_weights(
                    atom_coords=atom_coords,
                    grid_coords=grid_coords,
                    batch_idx=batch_idx_atoms,
                    radius=radius,
                    weight_alpha=weight_alpha,
                    device=device
                )
                
                return position_weights
                
            except Exception as e:
                print(f"[WARNING] Failed to compute position weights: {e}")
                import traceback
                traceback.print_exc()
                return None
        
        return position_weights
    
    def _get_multi_timestep_codes(self, x_0):
        """
        Get slightly perturbed and denoised codes by randomly sampling one timestep from [0, num_timesteps_for_decoder),
        adding noise, then denoising. This ensures codes are only slightly perturbed (using small timesteps),
        and after denoising, the codes should be close to the original x_0, so decoder can 
        still generate fields similar to ground truth.
        
        Randomly samples a timestep for each sample in the batch from [0, num_timesteps_for_decoder)
        to avoid OOM when num_timesteps_for_decoder is large. Each sample has its own independent timestep.
        
        Args:
            x_0: Ground truth codes [B, n_grid, code_dim]
            
        Returns:
            denoised_codes: [B, n_grid, code_dim] tensor containing denoised codes.
                Each sample uses its own randomly sampled timestep from [0, num_timesteps_for_decoder).
        """
        from funcmol.models.ddpm import q_sample, p_sample_x0
        
        device = x_0.device
        B = x_0.shape[0]
        diffusion_consts = self.funcmol.diffusion_consts
        
        # Randomly sample a timestep for each sample in the batch from [0, num_timesteps_for_decoder)
        # This allows using large num_timesteps_for_decoder without OOM
        # Each sample has its own independent timestep, similar to denoiser loss sampling
        t = torch.randint(0, self.num_timesteps_for_decoder, (B,), device=device).long()
        
        # Add noise to x_0 to get x_t (forward diffusion)
        # Each sample uses its own timestep t[i]
        noise = torch.randn_like(x_0)
        x_t = q_sample(x_0, t, diffusion_consts, noise)
        
        # Denoise using the model (predict x0)
        # Note: We need gradients for decoder loss, so don't use torch.no_grad()
        if self.funcmol.diffusion_method == "new_x0":
            # Model predicts x0 directly
            predicted_x0 = self.funcmol.net(x_t, t)
        else:
            # For "new" method, we need to use p_sample_x0 to get denoised codes
            predicted_x0 = p_sample_x0(self.funcmol.net, x_t, t, diffusion_consts, clip_denoised=False)
        
        return predicted_x0
    
    def _get_predicted_codes_from_diffusion(self, x_0):
        """
        Get predicted codes from diffusion model by adding noise and then denoising.
        This is used for field loss finetune to compare predicted codes with ground truth codes.
        
        Randomly samples a timestep for each sample in the batch from [0, num_timesteps_for_field_loss)
        to avoid OOM when num_timesteps_for_field_loss is large. Each sample has its own independent timestep.
        
        Args:
            x_0: Ground truth codes [B, n_grid, code_dim]
            
        Returns:
            predicted_codes: [B, n_grid, code_dim] - predicted codes, each sample uses its own randomly sampled timestep
        """
        from funcmol.models.ddpm import q_sample, p_sample_x0
        
        device = x_0.device
        B = x_0.shape[0]
        diffusion_consts = self.funcmol.diffusion_consts
        
        # Randomly sample a timestep for each sample in the batch from [0, num_timesteps_for_field_loss)
        # This allows using large num_timesteps_for_field_loss without OOM
        # Each sample has its own independent timestep, similar to denoiser loss sampling
        t = torch.randint(0, self.num_timesteps_for_field_loss, (B,), device=device).long()
        
        # Add noise to x_0 to get x_t (forward diffusion)
        # Each sample uses its own timestep t[i]
        noise = torch.randn_like(x_0)
        x_t = q_sample(x_0, t, diffusion_consts, noise)
        
        # Denoise using the model (predict x0)
        # Note: We need gradients for field loss, so don't use torch.no_grad()
        if self.funcmol.diffusion_method == "new_x0":
            # Model predicts x0 directly
            predicted_x0 = self.funcmol.net(x_t, t)
        else:
            # For "new" method, we need to use p_sample_x0 to get denoised codes
            predicted_x0 = p_sample_x0(self.funcmol.net, x_t, t, diffusion_consts, clip_denoised=False)
        
        return predicted_x0
    
    def _filter_query_points_near_atoms(self, query_points, atom_coords, target_field):
        """
        Filter query points to keep only those within atom_distance_threshold of atoms.
        
        Args:
            query_points: [B, n_points, 3] query points
            atom_coords: [B, n_atoms, 3] atom coordinates (or [N_total_atoms, 3] with batch_idx)
            target_field: [B, n_points, n_atom_types, 3] target field
            batch_idx: Optional [N_total_atoms] batch indices for atoms (if atom_coords is flattened)
            
        Returns:
            filtered_query_points: [B, n_filtered_points, 3] (may have different n_filtered_points per batch)
            filtered_target_field: [B, n_filtered_points, n_atom_types, 3]
            valid_mask: [B, n_points] boolean mask indicating which points were kept
        """
        device = query_points.device
        B = query_points.shape[0]
        n_points = query_points.shape[1]
        
        # Handle atom_coords shape: could be [B, n_atoms, 3] or [N_total_atoms, 3] with batch_idx
        if atom_coords.dim() == 2:
            # Flattened format: need batch_idx to separate
            # This case should be handled by caller, but we'll handle it here for robustness
            raise ValueError("atom_coords should be [B, n_atoms, 3], not flattened. Please reshape before calling this function.")
        
        # atom_coords is [B, n_atoms, 3]
        valid_masks = []
        filtered_query_points_list = []
        filtered_target_field_list = []
        
        for b in range(B):
            # Get query points and atoms for this batch
            batch_query_points = query_points[b]  # [n_points, 3]
            batch_atoms = atom_coords[b]  # [n_atoms, 3]
            
            if batch_atoms.shape[0] == 0:
                # No atoms in this batch, skip all points
                valid_masks.append(torch.zeros(n_points, dtype=torch.bool, device=device))
                filtered_query_points_list.append(torch.empty((0, 3), device=device))
                filtered_target_field_list.append(torch.empty((0, target_field.shape[2], 3), device=device))
                continue
            
            # Compute distances from each query point to all atoms
            # batch_query_points: [n_points, 3], batch_atoms: [n_atoms, 3]
            distances = torch.cdist(batch_query_points, batch_atoms)  # [n_points, n_atoms]
            
            # Find minimum distance for each query point
            min_distances = distances.min(dim=-1)[0]  # [n_points]
            
            # Keep only points within threshold
            valid_mask = min_distances < self.atom_distance_threshold  # [n_points]
            
            if valid_mask.sum() == 0:
                # No points within threshold, use a fallback: keep at least the closest point to each atom
                # This ensures we don't lose all points
                for atom_idx in range(batch_atoms.shape[0]):
                    atom_distances = distances[:, atom_idx]  # [n_points]
                    closest_point_idx = atom_distances.argmin()
                    valid_mask[closest_point_idx] = True
            
            valid_masks.append(valid_mask)
            
            # Filter query points and target field
            filtered_query_points_list.append(batch_query_points[valid_mask])  # [n_filtered, 3]
            filtered_target_field_list.append(target_field[b][valid_mask])  # [n_filtered, n_atom_types, 3]
        
        # Note: Different batches may have different numbers of filtered points
        # We'll pad to the maximum length for batching, or handle variable lengths
        max_filtered = max([fp.shape[0] for fp in filtered_query_points_list] + [1])  # At least 1
        
        # Pad filtered results to same length
        padded_query_points = []
        padded_target_field = []
        for b in range(B):
            n_filtered = filtered_query_points_list[b].shape[0]
            if n_filtered < max_filtered:
                # Pad with zeros (these will be masked out in loss calculation)
                padding_size = max_filtered - n_filtered
                query_padding = torch.zeros((padding_size, 3), device=device)
                field_padding = torch.zeros((padding_size, target_field.shape[2], 3), device=device)
                padded_query_points.append(torch.cat([filtered_query_points_list[b], query_padding], dim=0))
                padded_target_field.append(torch.cat([filtered_target_field_list[b], field_padding], dim=0))
            else:
                padded_query_points.append(filtered_query_points_list[b])
                padded_target_field.append(filtered_target_field_list[b])
        
        filtered_query_points = torch.stack(padded_query_points, dim=0)  # [B, max_filtered, 3]
        filtered_target_field = torch.stack(padded_target_field, dim=0)  # [B, max_filtered, n_atom_types, 3]
        
        # Also create a mask for the padded points (to ignore them in loss)
        n_filtered_per_batch = torch.tensor([fp.shape[0] for fp in filtered_query_points_list], device=device)
        padded_valid_mask = torch.arange(max_filtered, device=device).unsqueeze(0) < n_filtered_per_batch.unsqueeze(1)  # [B, max_filtered]
        
        return filtered_query_points, filtered_target_field, padded_valid_mask
    
    
    def _compute_decoder_loss(self, codes, batch, batch_idx, is_training=True):
        """
        Compute decoder reconstruction loss for joint fine-tuning.
        Uses single timestep codes (randomly sampled from [0, num_timesteps_for_decoder)),
        query point filtering, and cosine+magnitude loss.
        
        Args:
            codes: Codes tensor [B, n_grid, code_dim] (single timestep codes, randomly sampled)
            batch: Data batch (PyTorch Geometric Batch if on_the_fly=True, or codes tensor if False)
            batch_idx: Batch index in DataLoader
            is_training: Whether this is training step (True) or validation step (False)
            
        Returns:
            decoder_loss: Scalar tensor with decoder reconstruction loss
        """
        if not self.joint_finetune_enabled:
            return None
        
        device = codes.device
        # codes is now always [B, n_grid, code_dim] (single timestep, randomly sampled)
        B = codes.shape[0]
        
        n_points = self.config["dset"]["n_points"]
        n_atom_types = self.config["dset"]["n_channels"]
        
        # Get query points and target field
        if self.config["on_the_fly"]:
            # on_the_fly=True: batch contains molecule data with target_field
            if not hasattr(batch, 'xs') or not hasattr(batch, 'target_field'):
                print("[WARNING] Batch does not contain xs or target_field, skipping decoder loss")
                return None
            
            query_points = batch.xs  # [N_total_points, 3] or [B, n_points, 3]
            target_field = batch.target_field
            
            # Get atom coordinates for filtering
            atom_coords = batch.pos  # [N_total_atoms, 3]
            batch_idx_atoms = batch.batch  # [N_total_atoms]
            
            # Reshape query_points to [B, n_points, 3]
            if query_points.dim() == 2:
                # [N_total_points, 3] -> [B, n_points, 3]
                query_points = query_points.view(B, n_points, 3)
            elif query_points.dim() == 3:
                # Already [B, n_points, 3]
                if query_points.shape[0] != B:
                    query_points = query_points.view(B, n_points, 3)
            else:
                print(f"[WARNING] Unexpected query_points shape: {query_points.shape}")
                return None
            
            # Reshape target_field using utility function
            try:
                target_field = reshape_target_field(target_field, B, n_points, n_atom_types)
            except Exception as e:
                print(f"[WARNING] Failed to reshape target_field: {e}")
                return None
            
            # Reshape atom_coords to [B, n_atoms, 3]
            atom_coords_list = []
            for b in range(B):
                mask = batch_idx_atoms == b
                batch_atoms = atom_coords[mask]  # [n_atoms_b, 3]
                atom_coords_list.append(batch_atoms)
            # Pad to same length
            max_atoms = max([a.shape[0] for a in atom_coords_list] + [1])
            padded_atom_coords = []
            for b in range(B):
                n_atoms_b = atom_coords_list[b].shape[0]
                if n_atoms_b < max_atoms:
                    padding = torch.zeros((max_atoms - n_atoms_b, 3), device=device)
                    padded_atom_coords.append(torch.cat([atom_coords_list[b], padding], dim=0))
                else:
                    padded_atom_coords.append(atom_coords_list[b])
            atom_coords = torch.stack(padded_atom_coords, dim=0)  # [B, max_atoms, 3]
            
        else:
            # on_the_fly=False: need to get molecule data from field loader
            field_dataset = self._field_dataset_train if is_training else self._field_dataset_val
            
            if field_dataset is None:
                print("[WARNING] Field dataset not available for decoder loss computation")
                return None
            
            if batch_idx is None:
                print("[WARNING] batch_idx is required for decoder loss in on_the_fly=False mode")
                return None
            
            try:
                # Calculate dataset indices for this batch
                batch_size = self.config["dset"]["batch_size"]
                start_idx = batch_idx * batch_size
                end_idx = start_idx + B
                
                # Get augmentation info for modulo mapping
                num_augmentations = self.num_augmentations_train if is_training else self.num_augmentations_val
                field_dataset_size = self.field_dataset_size_train if is_training else self.field_dataset_size_val
                
                # Collect query points, target fields, and atom coordinates from field dataset
                all_query_points = []
                all_target_fields = []
                all_atom_coords = []
                
                for code_idx in range(start_idx, end_idx):
                    # Map codes index to field dataset index using modulo
                    if num_augmentations is not None and num_augmentations > 1 and field_dataset_size is not None:
                        field_idx = code_idx % field_dataset_size
                    else:
                        field_idx = code_idx
                    
                    if field_idx >= len(field_dataset):
                        print(f"[WARNING] Field index {field_idx} out of range (dataset size: {len(field_dataset)})")
                        return None
                    
                    try:
                        mol_data = field_dataset[field_idx]
                        
                        if hasattr(mol_data, 'xs') and hasattr(mol_data, 'target_field'):
                            all_query_points.append(mol_data.xs)  # [n_points, 3]
                            all_target_fields.append(mol_data.target_field)  # [n_points, n_atom_types, 3]
                            
                            # Get atom coordinates
                            if hasattr(mol_data, 'pos'):
                                all_atom_coords.append(mol_data.pos)  # [n_atoms, 3]
                            else:
                                print(f"[WARNING] Molecule at index {field_idx} missing pos (atom coordinates)")
                                return None
                        else:
                            print(f"[WARNING] Molecule at index {field_idx} missing xs or target_field")
                            return None
                    except Exception as e:
                        print(f"[WARNING] Failed to get molecule at field index {field_idx}: {e}")
                        return None
                
                if len(all_query_points) != B:
                    print(f"[WARNING] Expected {B} molecules, got {len(all_query_points)}")
                    return None
                
                # Stack query points and target fields
                query_points = torch.stack(all_query_points).to(device)  # [B, n_points, 3]
                target_field = torch.stack(all_target_fields).to(device)  # [B, n_points, n_atom_types, 3]
                
                # Pad atom coordinates to same length
                max_atoms = max([a.shape[0] for a in all_atom_coords] + [1])
                padded_atom_coords = []
                for b in range(B):
                    n_atoms_b = all_atom_coords[b].shape[0]
                    if n_atoms_b < max_atoms:
                        padding = torch.zeros((max_atoms - n_atoms_b, 3), device=device)
                        padded_atom_coords.append(torch.cat([all_atom_coords[b].to(device), padding], dim=0))
                    else:
                        padded_atom_coords.append(all_atom_coords[b].to(device))
                atom_coords = torch.stack(padded_atom_coords, dim=0)  # [B, max_atoms, 3]
                
            except Exception as e:
                print(f"[WARNING] Failed to compute decoder loss: {e}")
                import traceback
                traceback.print_exc()
                return None
        
        # Use query points directly (already sampled according to targeted_sampling_ratio)
        # No need to filter again
        filtered_query_points = query_points
        filtered_target_field = target_field
        n_filtered = filtered_query_points.shape[1]  # n_points
        # Create a mask of all True (all points are valid)
        padded_valid_mask = torch.ones(B, n_filtered, dtype=torch.bool, device=device)
        
        # Generate predicted field using decoder
        # codes is [B, n_grid, code_dim] (single timestep, randomly sampled)
        try:
            pred_field = self.dec_module(filtered_query_points, codes)  # [B, n_filtered, n_atom_types, 3]
        except Exception as e:
            print(f"[WARNING] Failed to generate field from decoder: {e}")
            return None
        
        # Ensure shapes match
        if pred_field.shape != filtered_target_field.shape:
            pred_field = pred_field.view(B, n_filtered, n_atom_types, 3)
            if pred_field.shape != filtered_target_field.shape:
                print(f"[WARNING] Shape mismatch after reshape: pred_field {pred_field.shape} vs target_field {filtered_target_field.shape}")
                return None
        
        # Apply mask to ignore padded points
        padded_valid_mask_expanded = padded_valid_mask.unsqueeze(-1).unsqueeze(-1)  # [B, n_filtered, 1, 1]
        pred_field = pred_field * padded_valid_mask_expanded
        filtered_target_field_masked = filtered_target_field * padded_valid_mask_expanded
        
        # Compute loss using utility function
        decoder_loss = compute_decoder_field_loss(
            pred_field,
            filtered_target_field,
            use_cosine_loss=self.use_cosine_loss,
            magnitude_loss_weight=self.magnitude_loss_weight,
            valid_mask=padded_valid_mask
        )
        
        return decoder_loss
    
    def _compute_field_loss_finetune(self, x_0, batch, batch_idx, is_training=True):
        """
        Compute field loss for field loss finetune (decoder frozen mode).
        Compares fields generated from diffusion-predicted codes vs ground truth codes.
        Uses dataset's _get_xs method to sample query points (mix of grid and neighbor points).
        
        Args:
            x_0: Ground truth codes [B, n_grid, code_dim]
            batch: Data batch (PyTorch Geometric Batch if on_the_fly=True, or codes tensor if False)
            batch_idx: Batch index in DataLoader
            is_training: Whether this is training step (True) or validation step (False)
            
        Returns:
            field_loss: Scalar tensor with field loss, or None if computation fails
        """
        if not self.field_loss_finetune_enabled:
            return None
        
        device = x_0.device
        B = x_0.shape[0]
        
        # Get predicted codes from diffusion
        # Now returns single timestep codes [B, n_grid, code_dim] (randomly sampled from [0, num_timesteps_for_field_loss))
        predicted_codes = self._get_predicted_codes_from_diffusion(x_0)  # [B, n_grid, code_dim]
        
        # Get field dataset to access _get_xs method
        field_dataset = self._field_dataset_train if is_training else self._field_dataset_val
        
        if field_dataset is None:
            print("[WARNING] Field dataset not available for field loss computation")
            return None
        
        try:
            # Collect query points and atom coordinates using dataset's _get_xs method
            all_query_points = []
            all_atom_coords = []  # Store atom coordinates for weight computation
            
            if self.config["on_the_fly"]:
                # on_the_fly=True: batch contains molecule data (PyTorch Geometric Batch)
                if not hasattr(batch, 'pos') or not hasattr(batch, 'x'):
                    print("[WARNING] Batch does not contain pos or x, skipping field loss")
                    return None
                
                # Extract atom coordinates and types for each molecule in batch
                atom_coords = batch.pos  # [N_total_atoms, 3]
                atom_types = batch.x  # [N_total_atoms]
                batch_idx_atoms = batch.batch  # [N_total_atoms]
                
                for b in range(B):
                    # Get atoms for this molecule
                    mask = batch_idx_atoms == b
                    coords_b = atom_coords[mask]  # [n_atoms_b, 3]
                    atoms_channel_b = atom_types[mask]  # [n_atoms_b]
                    
                    # Store atom coordinates for weight computation
                    all_atom_coords.append(coords_b.to(device))
                    
                    # Construct sample dict for _get_xs method
                    sample = {
                        "coords": coords_b,
                        "atoms_channel": atoms_channel_b
                    }
                    
                    # Use dataset's _get_xs method to sample query points
                    query_points_b, _ = field_dataset._get_xs(sample)  # [n_points, 3]
                    all_query_points.append(query_points_b.to(device))
            else:
                # on_the_fly=False: need to get data from field dataset
                if batch_idx is None:
                    print("[WARNING] batch_idx is required for field loss in on_the_fly=False mode")
                    return None
                
                # Calculate dataset indices for this batch
                batch_size = self.config["dset"]["batch_size"]
                start_idx = batch_idx * batch_size
                end_idx = start_idx + B
                
                # Get augmentation info for modulo mapping
                num_augmentations = self.num_augmentations_train if is_training else self.num_augmentations_val
                field_dataset_size = self.field_dataset_size_train if is_training else self.field_dataset_size_val
                
                for code_idx in range(start_idx, end_idx):
                    # Map codes index to field dataset index using modulo
                    if num_augmentations is not None and num_augmentations > 1 and field_dataset_size is not None:
                        field_idx = code_idx % field_dataset_size
                    else:
                        field_idx = code_idx
                    
                    if field_idx >= len(field_dataset):
                        print(f"[WARNING] Field index {field_idx} out of range (dataset size: {len(field_dataset)})")
                        return None
                    
                    try:
                        # Get raw sample from dataset (before preprocessing)
                        # We need to access the raw data to call _get_xs
                        if hasattr(field_dataset, 'use_lmdb') and field_dataset.use_lmdb:
                            # LMDB mode: need to get raw sample
                            idx = field_dataset.field_idxs[field_idx].item()
                            key = field_dataset.keys[idx]
                            if isinstance(key, str):
                                key = key.encode('utf-8')
                            with field_dataset.db.begin() as txn:
                                sample_raw = pickle.loads(txn.get(key))
                        else:
                            # Traditional mode: get from data list
                            idx = field_dataset.field_idxs[field_idx].item()
                            sample_raw = field_dataset.data[idx]
                        
                        # Preprocess molecule (handles rotation and centering)
                        sample = field_dataset._preprocess_molecule(sample_raw)
                        
                        # Extract atom coordinates for weight computation
                        mask = sample["atoms_channel"] != PADDING_INDEX
                        coords = sample["coords"][mask]  # [n_atoms, 3]
                        all_atom_coords.append(coords.to(device))
                        
                        # Use dataset's _get_xs method to sample query points
                        # This will use the same sampling strategy as dataset (mix of grid and neighbor points)
                        query_points, _ = field_dataset._get_xs(sample)  # [n_points, 3]
                        all_query_points.append(query_points.to(device))
                        
                    except Exception as e:
                        print(f"[WARNING] Failed to get query points for field index {field_idx}: {e}")
                        import traceback
                        traceback.print_exc()
                        return None
            
            if len(all_query_points) != B:
                print(f"[WARNING] Expected {B} molecules, got {len(all_query_points)}")
                return None
            
            # Stack query points
            query_points = torch.stack(all_query_points, dim=0)  # [B, n_points, 3]
            n_points = query_points.shape[1]
            
            # Compute loss weights based on distance to nearest atom (exp decay)
            # weights: [B, n_points]
            weights = torch.ones(B, n_points, device=device)
            for b in range(B):
                sample_query_points = query_points[b]  # [n_points, 3]
                sample_atoms = all_atom_coords[b]  # [n_atoms, 3]
                
                if len(sample_atoms) == 0:
                    continue
                
                # Compute distance to nearest atom for each query point
                distances = torch.cdist(sample_query_points, sample_atoms)  # [n_points, n_atoms]
                min_distances = distances.min(dim=-1)[0]  # [n_points] - distance to nearest atom
                
                # Convert distance to weight: closer atoms = higher weight
                # Use exponential decay: weight = exp(-distance / scale)
                sample_weights = torch.exp(-min_distances / self.field_loss_atom_distance_scale)
                weights[b] = sample_weights
            
        except Exception as e:
            print(f"[WARNING] Failed to compute field loss: {e}")
            import traceback
            traceback.print_exc()
            return None
        
        # Compute field loss for the single randomly sampled timestep
        # Generate fields using decoder
        # Ground truth field from ground truth codes (no gradients needed)
        with torch.no_grad():
            gt_field = self.dec_module(query_points, x_0)  # [B, n_points, n_atom_types, 3]
        
        # Predicted field from diffusion-predicted codes (need gradients for this)
        pred_field = self.dec_module(query_points, predicted_codes)  # [B, n_points, n_atom_types, 3]
        
        # Ensure shapes match
        if pred_field.shape != gt_field.shape:
            pred_field = pred_field.view(B, n_points, -1, 3)
            gt_field = gt_field.view(B, n_points, -1, 3)
            if pred_field.shape != gt_field.shape:
                print(f"[WARNING] Shape mismatch: pred_field {pred_field.shape} vs gt_field {gt_field.shape}")
                return None
        
        # Compute element-wise MSE loss
        elementwise_loss = (pred_field - gt_field) ** 2  # [B, n_points, n_atom_types, 3]
        # Average over atom_types and spatial dimensions, keep batch and point dimensions
        pointwise_loss = elementwise_loss.mean(dim=(-2, -1))  # [B, n_points]
        
        # Apply weights (exp decay based on distance to nearest atom)
        weighted_loss = pointwise_loss * weights  # [B, n_points]
        
        # Average over batch and points
        field_loss = weighted_loss.mean()
        
        return field_loss
    
    def training_step(self, batch, batch_idx):
        """Training step logic"""
        # 添加调试信息
        # if batch_idx == 0:  # 只在第一个batch打印
        #     print(f"[DEBUG] Using diffusion_method: {self.funcmol.diffusion_method}")
        
        if self.funcmol.diffusion_method == "new" or self.funcmol.diffusion_method == "new_x0":
            # DDPM训练
            return self._training_step_ddpm(batch, batch_idx)
        else:
            # 原有训练方法
            return self._training_step_original(batch, batch_idx)

    def _training_step_ddpm(self, batch, batch_idx):
        """DDPM训练步骤"""
        # 获取codes和position_weights（如果已预计算）
        codes, _, position_weights_precomputed = self._process_batch(batch)
        
        # 获取位置权重（优先使用预计算的，否则重新计算）
        position_weights = self._compute_position_weights(
            batch, codes, batch_idx=batch_idx, is_training=True,
            precomputed_weights=position_weights_precomputed
        )
        
        # 已移除：对codes的数据增强（使用bilinear插值）
        # 注意：在infer_codes.py中对分子坐标的增强（生成10个pt文件）仍然保留
        # 这里移除的是训练时对已生成codes的插值增强，避免bilinear插值误差
        # if self.training and self.config.get("use_data_augmentation", True):
        #     grid_size = self.config["dset"]["grid_size"]
        #     anchor_spacing = self.config["dset"]["anchor_spacing"]
        #     apply_rotation = self.config.get("data_augmentation", {}).get("apply_rotation", True)
        #     apply_translation = self.config.get("data_augmentation", {}).get("apply_translation", False)
        #     
        #     codes = augment_codes(
        #         codes, 
        #         grid_size=grid_size, 
        #         anchor_spacing=anchor_spacing,
        #         apply_rotation=apply_rotation,
        #         apply_translation=apply_translation
        #     )
        
        # 添加调试信息
        # if batch_idx == 0:  # 只在第一个batch打印
        #     print(f"[DEBUG] DDPM training step - input shape: {codes.shape}")
        #     print(f"[DEBUG] Using diffusion constants: {list(self.funcmol.diffusion_consts.keys())}")
        
        # DDPM训练步骤 - 直接使用3维输入 [B, N*N*N, code_dim]
        denoiser_loss = self.funcmol.train_ddpm_step(codes, position_weights=position_weights)
        
        # Compute decoder loss if joint fine-tuning is enabled
        if self.joint_finetune_enabled:
            # Get single-timestep codes for decoder training (randomly sampled from [0, num_timesteps_for_decoder))
            single_timestep_codes = self._get_multi_timestep_codes(codes)  # [B, n_grid, code_dim]
            
            # Compute decoder loss using single-timestep codes
            decoder_loss = self._compute_decoder_loss(single_timestep_codes, batch, batch_idx, is_training=True)
            if decoder_loss is not None:
                total_loss = denoiser_loss + self.decoder_loss_weight * decoder_loss
                # Log decoder loss separately
                self.log('train_decoder_loss', decoder_loss, batch_size=len(batch),
                         on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
            else:
                # If decoder loss computation failed, use only denoiser loss
                print("[WARNING] Decoder loss computation failed, using only denoiser loss")
                total_loss = denoiser_loss
        else:
            # Standard training: only denoiser loss
            total_loss = denoiser_loss
        
        # Compute field loss if field loss finetune is enabled
        if self.field_loss_finetune_enabled:
            # codes here is ground truth codes (x_0)
            field_loss = self._compute_field_loss_finetune(codes, batch, batch_idx, is_training=True)
            if field_loss is not None:
                total_loss = total_loss + self.field_loss_weight * field_loss
                # Log field loss separately
                self.log('train_field_loss', field_loss, batch_size=len(batch),
                         on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
            else:
                # If field loss computation failed, continue with existing total_loss
                print("[WARNING] Field loss computation failed, continuing without field loss")
        
        # Update EMA model
        self.funcmol_ema.update(self.funcmol)
        
        # Log metrics
        # on_step=True: log every N steps (controlled by Trainer's log_every_n_steps=200)
        # on_epoch=True: also log at the end of each epoch (for epoch-level statistics)
        self.log('train_loss', total_loss, batch_size=len(batch),
                 on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train_denoiser_loss', denoiser_loss, batch_size=len(batch),
                 on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        
        # Store loss for plotting
        self.train_losses.append(total_loss.item())
        
        return total_loss

    def _training_step_original(self, batch, batch_idx):
        """原有训练方法"""
        # Get codes, smooth_codes, and position_weights (if precomputed)
        codes, smooth_codes, position_weights_precomputed = self._process_batch(batch)
        
        # Forward pass through Funcmol
        pred_codes = self.funcmol(smooth_codes)
        
        # Calculate loss with position weights if enabled
        if self.use_position_weight:
            position_weights = self._compute_position_weights(
                batch, codes, batch_idx=batch_idx, is_training=True,
                precomputed_weights=position_weights_precomputed
            )
            if position_weights is not None:
                # Apply position weights to MSE loss
                squared_diff = (pred_codes - codes) ** 2  # [B, n_grid, code_dim]
                squared_diff_per_pos = squared_diff.mean(dim=-1)  # [B, n_grid]
                weighted_loss_per_pos = position_weights * squared_diff_per_pos  # [B, n_grid]
                denoiser_loss = weighted_loss_per_pos.mean()
            else:
                # Fallback to regular loss if weights cannot be computed
                denoiser_loss = self.criterion(pred_codes, codes)
        else:
            # Calculate loss without position weights
            denoiser_loss = self.criterion(pred_codes, codes)
        
        # Compute decoder loss if joint fine-tuning is enabled
        if self.joint_finetune_enabled:
            decoder_loss = self._compute_decoder_loss(codes, batch, batch_idx, is_training=True)
            if decoder_loss is not None:
                # Both denoiser and decoder are trained, combine losses
                total_loss = denoiser_loss + self.decoder_loss_weight * decoder_loss
                # Log decoder loss separately
                self.log('train_decoder_loss', decoder_loss, batch_size=len(batch),
                         on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
            else:
                # If decoder loss computation failed, use only denoiser loss
                print("[WARNING] Decoder loss computation failed, using only denoiser loss")
                total_loss = denoiser_loss
        else:
            # Standard training: only denoiser loss
            total_loss = denoiser_loss
        
        # Update EMA model
        self.funcmol_ema.update(self.funcmol)
        
        # Log metrics
        # on_step=True: log every N steps (controlled by Trainer's log_every_n_steps=200)
        # on_epoch=True: also log at the end of each epoch (for epoch-level statistics)
        self.log('train_loss', total_loss, batch_size=len(batch),
                 on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train_denoiser_loss', denoiser_loss, batch_size=len(batch),
                 on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        
        # Store loss for plotting
        self.train_losses.append(total_loss.item())
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step logic"""
        if self.funcmol.diffusion_method == "new" or self.funcmol.diffusion_method == "new_x0":
            # DDPM验证
            return self._validation_step_ddpm(batch, batch_idx)
        else:
            # 原有验证方法
            return self._validation_step_original(batch, batch_idx)

    def _validation_step_ddpm(self, batch, batch_idx):
        """DDPM验证步骤"""
        # 获取codes和position_weights（如果已预计算）
        codes, _, position_weights_precomputed = self._process_batch(batch)
        
        # 获取位置权重（优先使用预计算的，否则重新计算）
        position_weights = self._compute_position_weights(
            batch, codes, batch_idx=batch_idx, is_training=False,
            precomputed_weights=position_weights_precomputed
        )
        
        # DDPM验证步骤 - 直接使用3维输入 [B, N*N*N, code_dim]
        denoiser_loss = self.funcmol.train_ddpm_step(codes, position_weights=position_weights)
        
        # Compute decoder loss if joint fine-tuning is enabled
        if self.joint_finetune_enabled:
            # Get single-timestep codes for decoder training (randomly sampled from [0, num_timesteps_for_decoder))
            single_timestep_codes = self._get_multi_timestep_codes(codes)  # [B, n_grid, code_dim]
            
            # Compute decoder loss using single-timestep codes
            decoder_loss = self._compute_decoder_loss(single_timestep_codes, batch, batch_idx, is_training=False)
            if decoder_loss is not None:
                # Both denoiser and decoder are trained, combine losses
                total_loss = denoiser_loss + self.decoder_loss_weight * decoder_loss
                # Log decoder loss separately
                self.log('val_decoder_loss', decoder_loss, batch_size=len(batch),
                         on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
            else:
                # If decoder loss computation failed, use only denoiser loss
                total_loss = denoiser_loss
        else:
            # Standard validation: only denoiser loss
            total_loss = denoiser_loss
        
        # Compute field loss if field loss finetune is enabled
        if self.field_loss_finetune_enabled:
            # codes here is ground truth codes (x_0)
            field_loss = self._compute_field_loss_finetune(codes, batch, batch_idx, is_training=False)
            if field_loss is not None:
                total_loss = total_loss + self.field_loss_weight * field_loss
                # Log field loss separately
                self.log('val_field_loss', field_loss, batch_size=len(batch),
                         on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
            else:
                # If field loss computation failed, continue with existing total_loss
                pass  # Don't print warning in validation to avoid spam
        
        # Log metrics
        self.log('val_loss', total_loss, batch_size=len(batch),
                 on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_denoiser_loss', denoiser_loss, batch_size=len(batch),
                 on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        
        return total_loss

    def _validation_step_original(self, batch, batch_idx):
        """原有验证方法"""
        # Get codes, smooth_codes, and position_weights (if precomputed)
        codes, smooth_codes, position_weights_precomputed = self._process_batch(batch)
        
        # Forward pass through EMA model
        pred_codes = self.funcmol_ema(smooth_codes)
        
        # Calculate loss with position weights if enabled
        if self.use_position_weight:
            position_weights = self._compute_position_weights(
                batch, codes, batch_idx=batch_idx, is_training=False,
                precomputed_weights=position_weights_precomputed
            )
            if position_weights is not None:
                # Apply position weights to MSE loss
                squared_diff = (pred_codes - codes) ** 2  # [B, n_grid, code_dim]
                squared_diff_per_pos = squared_diff.mean(dim=-1)  # [B, n_grid]
                weighted_loss_per_pos = position_weights * squared_diff_per_pos  # [B, n_grid]
                denoiser_loss = weighted_loss_per_pos.mean()
            else:
                # Fallback to regular loss if weights cannot be computed
                denoiser_loss = self.criterion(pred_codes, codes)
        else:
            # Calculate loss without position weights
            denoiser_loss = self.criterion(pred_codes, codes)
        
        # Compute decoder loss if joint fine-tuning is enabled
        if self.joint_finetune_enabled:
            # Get single-timestep codes for decoder training (randomly sampled from [0, num_timesteps_for_decoder))
            single_timestep_codes = self._get_multi_timestep_codes(codes)  # [B, n_grid, code_dim]
            
            # Compute decoder loss using single-timestep codes
            decoder_loss = self._compute_decoder_loss(single_timestep_codes, batch, batch_idx, is_training=False)
            if decoder_loss is not None:
                # Both denoiser and decoder are trained, combine losses
                total_loss = denoiser_loss + self.decoder_loss_weight * decoder_loss
                # Log decoder loss separately
                self.log('val_decoder_loss', decoder_loss, batch_size=len(batch),
                         on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
            else:
                # If decoder loss computation failed, use only denoiser loss
                total_loss = denoiser_loss
        else:
            # Standard validation: only denoiser loss
            total_loss = denoiser_loss
        
        # Log metrics
        self.log('val_loss', total_loss, batch_size=len(batch),
                 on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_denoiser_loss', denoiser_loss, batch_size=len(batch),
                 on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        
        return total_loss
    
    # ############# function for debug ##############
    # def on_before_optimizer_step(self, optimizer):
    #     """Print gradient norm before clipping"""
    #     # Calculate total gradient norm
    #     total_norm = 0.0
    #     for param in self.parameters():
    #         if param.grad is not None:
    #             param_norm = param.grad.data.norm(2)
    #             total_norm += param_norm.item() ** 2
    #     total_norm = total_norm ** (1. / 2)
        
    #     # Print gradient info every 10 steps
    #     if self.global_step % 10 == 0:
    #         print(f"Step {self.global_step}: Gradient norm before clipping = {total_norm:.6f}")
    
    # ############# function for debug ############## 
    # def on_after_backward(self):
    #     """Print gradient norm after backward pass"""
    #     # Print gradient info every 20 steps
    #     if self.global_step % 20 == 0:
    #         total_norm = 0.0
    #         for param in self.parameters():
    #             if param.grad is not None:
    #                 param_norm = param.grad.data.norm(2)
    #                 total_norm += param_norm.item() ** 2
    #         total_norm = total_norm ** (1. / 2)
    #         print(f"Step {self.global_step}: Gradient norm after backward = {total_norm:.6f}")
    
    def on_train_epoch_start(self):
        """Called at the beginning of training epoch"""
        # Set model to training mode
        self.funcmol.train()
        
        # Set decoder to training mode if joint fine-tuning is enabled
        if self.joint_finetune_enabled and self.dec_module is not None:
            self.dec_module.train()
        
        # Clean up GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def on_validation_epoch_start(self):
        """Called at the beginning of validation epoch"""
        # Set model to evaluation mode
        self.funcmol.eval()
        
        # Set decoder to evaluation mode
        if self.dec_module is not None:
            self.dec_module.eval()
    
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
        # Create optimizer with different parameter groups for joint fine-tuning
        if self.joint_finetune_enabled:
            param_groups = [
                {"params": self.funcmol.parameters(), "lr": self.config["lr"], "weight_decay": self.config["wd"]}
            ]
            
            # Add decoder parameters with optional separate learning rate
            decoder_lr = self.decoder_lr if self.decoder_lr is not None else self.config["lr"]
            param_groups.append(
                {"params": self.dec_module.parameters(), "lr": decoder_lr, "weight_decay": self.config["wd"]}
            )
            
            optimizer = AdamW(param_groups)
            print(f"Optimizer configured for joint fine-tuning:")
            print(f"  - Denoiser LR: {self.config['lr']}")
            print(f"  - Decoder LR: {decoder_lr}")
        else:
            # Standard training: only optimize denoiser
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
        # 保存前去掉多余的封装（但是lightning模式理论上会自动处理好，不需要去除，只是以防万一）
        funcmol_state_dict = self.funcmol.module.state_dict() if hasattr(self.funcmol, "module") else self.funcmol.state_dict()
        funcmol_ema_state_dict = self.funcmol_ema.module.state_dict() if hasattr(self.funcmol_ema, "module") else self.funcmol_ema.state_dict()
        
        checkpoint["funcmol_state_dict"] = funcmol_state_dict
        checkpoint["funcmol_ema_state_dict"] = funcmol_ema_state_dict
        
        # Save decoder state if joint fine-tuning is enabled
        if self.joint_finetune_enabled and self.dec_module is not None:
            decoder_state_dict = self.dec_module.module.state_dict() if hasattr(self.dec_module, "module") else self.dec_module.state_dict()
            checkpoint["decoder_state_dict"] = decoder_state_dict
            print("Saved decoder state in checkpoint (joint fine-tuning mode)")
        
        checkpoint["code_stats"] = self.code_stats
        checkpoint["train_losses"] = self.train_losses
        checkpoint["val_losses"] = self.val_losses
        checkpoint["best_loss"] = self.best_loss
    
    def on_load_checkpoint(self, checkpoint):
        """Custom checkpoint loading logic"""
        # Load decoder state if available and joint fine-tuning is enabled
        if self.joint_finetune_enabled and self.dec_module is not None:
            if "decoder_state_dict" in checkpoint:
                decoder_state_dict = checkpoint["decoder_state_dict"]
                # Handle _orig_mod. prefix if present
                new_state_dict = {k.replace("_orig_mod.", ""): v for k, v in decoder_state_dict.items()}
                try:
                    self.dec_module.load_state_dict(new_state_dict, strict=True)
                    print("Loaded decoder state from checkpoint (joint fine-tuning mode)")
                except Exception as e:
                    print(f"Warning: Failed to load decoder state: {e}")
                    print("Continuing with decoder initialized from neural field checkpoint")
            else:
                print("Warning: Joint fine-tuning enabled but no decoder_state_dict found in checkpoint")
                print("Decoder will use weights from neural field checkpoint")
        
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
    
    # # Create a minimal dummy fabric object for compatibility with data loaders
    # class DummyFabric:  # TODO: remove fabric
    #     def print(self, msg):
    #         print(msg)
    #     def all_gather(self, tensor):
    #         return tensor
    #     def setup_dataloaders(self, loader, use_distributed_sampler=True):
    #         return loader
    
    # dummy_fabric = DummyFabric()
    
    # Create a checkpoint dict in the format expected by load_neural_field
    nf_checkpoint = {
        "config": nf_config,
        "enc_state_dict": checkpoint["enc_state_dict"],
        "dec_state_dict": checkpoint["dec_state_dict"]
    }
    
    # Load neural field models using the existing function
    enc, dec = load_neural_field(nf_checkpoint, None)  # Pass None for Lightning compatibility
    
    dec_module = dec.module if hasattr(dec, "module") else dec
    
    # Note: decoder 默认使用 neural field checkpoint 中的 decoder
    # 如果从 reload_model_path 加载的 checkpoint 中包含 decoder_state_dict，则会在 on_load_checkpoint 中自动加载
    
    ##############################
    # Code loaders
    config_nf = nf_config
    config_nf["debug"] = config["debug"]
    config_nf["dset"]["batch_size"] = config["dset"]["batch_size"]
    
    # Load checkpoint config if specified (to get codes_dir and grid_size from checkpoint)
    checkpoint_config = None
    checkpoint_codes_dir = None
    checkpoint_code_stats = None
    if config["reload_model_path"] is not None:
        try:
            checkpoint_path = config["reload_model_path"]
            if os.path.isdir(checkpoint_path):
                from funcmol.utils.utils_fm import find_checkpoint_path
                checkpoint_path = find_checkpoint_path(checkpoint_path)
            
            print(f">> Loading checkpoint config from: {checkpoint_path}")
            checkpoint_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            checkpoint_hparams = checkpoint_dict.get("hyper_parameters", {})
            if checkpoint_hparams and "config" in checkpoint_hparams:
                checkpoint_config = checkpoint_hparams["config"]
                print(f">> Checkpoint config loaded: grid_size={checkpoint_config.get('dset', {}).get('grid_size', 'N/A')}")
                # Get codes_dir from checkpoint
                use_augmented_codes_ckpt = checkpoint_config.get("use_augmented_codes", False)
                if use_augmented_codes_ckpt:
                    checkpoint_codes_dir = checkpoint_config.get("codes_dir_with_aug")
                else:
                    checkpoint_codes_dir = checkpoint_config.get("codes_dir_no_aug") or checkpoint_config.get("codes_dir")
                if checkpoint_codes_dir:
                    print(f">> Found codes_dir in checkpoint: {checkpoint_codes_dir}")
            
            # Try to load code_stats from checkpoint
            checkpoint_code_stats = checkpoint_dict.get("code_stats", None)
            if checkpoint_code_stats:
                print(f">> Found code_stats in checkpoint:")
                print(f">>   mean: {checkpoint_code_stats.get('mean', 'N/A')}")
                print(f">>   std: {checkpoint_code_stats.get('std', 'N/A')}")
                print(f">>   max_normalized: {checkpoint_code_stats.get('max_normalized', 'N/A')}")
                print(f">>   min_normalized: {checkpoint_code_stats.get('min_normalized', 'N/A')}")
                print(f">> Will use code_stats from checkpoint instead of loading from codes directory")
            else:
                print(f">> No code_stats found in checkpoint, will load from codes directory or recompute")
        except Exception as e:
            print(f">> Warning: Failed to load checkpoint config: {e}")
            print(">> Will use current config instead")
    
    # 根据配置选择使用数据增强的codes还是原始codes
    if not config["on_the_fly"]:
        use_augmented_codes = config.get("use_augmented_codes", False)
        
        # 优先使用配置文件中的设置（如果明确指定了）
        # 只有当配置文件中没有明确设置时，才使用checkpoint中的codes_dir
        if use_augmented_codes:
            codes_dir = config.get("codes_dir_with_aug")
            if codes_dir is None:
                # 如果配置文件中没有设置，尝试使用checkpoint中的
                if checkpoint_codes_dir is not None:
                    codes_dir = checkpoint_codes_dir
                    print(f">> 使用checkpoint中的codes_dir: {codes_dir}")
                else:
                    raise ValueError(
                        "use_augmented_codes=True 但未指定 codes_dir_with_aug。\n"
                        "请在配置文件中设置 codes_dir_with_aug 路径。"
                    )
            else:
                print(f">> 使用数据增强的codes: {codes_dir}")
        else:
            codes_dir = config.get("codes_dir_no_aug")
            if codes_dir is None:
                # 如果配置文件中没有设置，尝试使用旧的codes_dir或checkpoint中的
                codes_dir = config.get("codes_dir")
                if codes_dir is None:
                    # 最后尝试使用checkpoint中的
                    if checkpoint_codes_dir is not None:
                        codes_dir = checkpoint_codes_dir
                        print(f">> 使用checkpoint中的codes_dir: {codes_dir}")
                    else:
                        raise ValueError(
                            "use_augmented_codes=False 但未指定 codes_dir_no_aug 或 codes_dir。\n"
                            "请在配置文件中设置 codes_dir_no_aug 路径。"
                        )
                else:
                    print(f">> 使用兼容的codes_dir: {codes_dir}")
            else:
                print(f">> 使用原始codes（无数据增强）: {codes_dir}")
        
        # 设置config中的codes_dir，供create_code_loaders使用
        config["codes_dir"] = codes_dir
    
    # Create data loaders
    field_loader_train = None
    field_loader_val = None
    
    try:
        if config["on_the_fly"]:
            # Create GNFConverter instance for data loading
            gnf_converter = create_gnf_converter(config)
            
            loader_train = create_field_loaders(config, gnf_converter, split="train")
            loader_val = create_field_loaders(config, gnf_converter, split="val")
            
            # Store field loaders for position weight computation
            field_loader_train = loader_train
            field_loader_val = loader_val
            
            # # Handle cases where loaders are returned as lists
            # if isinstance(loader_train, list) and len(loader_train) > 0:
            #     loader_train = loader_train[0]
            # if isinstance(loader_val, list) and len(loader_val) > 0:
            #     loader_val = loader_val[0]
            
            # 优先使用checkpoint中的code_stats（如果存在）
            if checkpoint_code_stats is not None:
                print(f">> Using code_stats from checkpoint (resuming training)")
                code_stats = checkpoint_code_stats
            else:
                # 尝试从指定路径加载code_stats
                code_stats_path = config.get("code_stats_path", None)
                if code_stats_path is not None and os.path.exists(code_stats_path):
                    try:
                        print(f">> Loading code_stats from specified path: {code_stats_path}")
                        code_stats = torch.load(code_stats_path, map_location='cpu', weights_only=False)
                        print(f">> Successfully loaded code_stats from file:")
                        print(f">>   mean: {code_stats.get('mean', 'N/A')}")
                        print(f">>   std: {code_stats.get('std', 'N/A')}")
                        print(f">>   max_normalized: {code_stats.get('max_normalized', 'N/A')}")
                        print(f">>   min_normalized: {code_stats.get('min_normalized', 'N/A')}")
                    except Exception as e:
                        print(f">> Warning: Failed to load code_stats from {code_stats_path}: {e}")
                        print(f">> Will compute code_stats from data instead")
                        code_stats = None
                else:
                    code_stats = None
                
                # 如果仍未加载到code_stats，则重新计算
                if code_stats is None:
                    # Compute codes for normalization
                    print(f">> Computing code_stats from data (new training or checkpoint has no code_stats)")
                    _, code_stats = compute_codes(
                        loader_train, enc, config_nf, "train", config["normalize_codes"],
                        code_stats=None
                    )
        else:
            loader_train = create_code_loaders(config, split="train")
            loader_val = create_code_loaders(config, split="val")
            
            # Check if field loaders are needed (for position weighting or field loss finetune)
            position_weight_config = config.get("position_weight", {})
            field_loss_finetune_config = config.get("field_loss_finetune", {})
            joint_finetune_config = config.get("joint_finetune", {})
            joint_finetune_enabled = joint_finetune_config.get("enabled", False)
            need_field_loaders = (
                position_weight_config.get("enabled", False) or 
                field_loss_finetune_config.get("enabled", False) or
                joint_finetune_enabled
            )
            
            if need_field_loaders:
                reason = []
                if position_weight_config.get("enabled", False):
                    reason.append("position weighting")
                if field_loss_finetune_config.get("enabled", False):
                    reason.append("field loss finetune")
                if joint_finetune_enabled:
                    reason.append("joint finetune")
                reason_str = " and ".join(reason)
                print(f">> {reason_str.capitalize()} enabled for on_the_fly=False mode")
                print(">> Creating field loaders...")
                try:
                    gnf_converter = create_gnf_converter(config)
                    field_loader_train = create_field_loaders(config, gnf_converter, split="train")
                    field_loader_val = create_field_loaders(config, gnf_converter, split="val")
                    
                    # Verify dataset lengths match
                    code_dataset_train = loader_train.dataset
                    code_dataset_val = loader_val.dataset if loader_val else None
                    field_dataset_train = field_loader_train.dataset
                    field_dataset_val = field_loader_val.dataset if field_loader_val else None
                    
                    len_codes_train = len(code_dataset_train)
                    len_field_train = len(field_dataset_train)
                    
                    # Calculate num_augmentations if codes length is a multiple of field length
                    num_augmentations_train = None
                    if len_codes_train > 0 and len_field_train > 0:
                        if len_codes_train % len_field_train == 0:
                            num_augmentations_train = len_codes_train // len_field_train
                            print(f">> Train split: codes dataset has {num_augmentations_train}x samples (augmentations)")
                            print(f">>   Codes dataset: {len_codes_train} samples")
                            print(f">>   Field dataset: {len_field_train} samples")
                            print(f">>   Using modulo mapping: codes[i] -> field[i % {len_field_train}]")
                        else:
                            print(f">> WARNING: Codes dataset length ({len_codes_train}) is not a multiple of field dataset length ({len_field_train})")
                            print(f">>   Cannot use modulo mapping")
                            num_augmentations_train = None
                    
                    if len_codes_train == len_field_train:
                        print(f">> Train dataset lengths match: {len_codes_train} samples (no augmentation)")
                        num_augmentations_train = 1
                    
                    if code_dataset_val and field_dataset_val:
                        len_codes_val = len(code_dataset_val)
                        len_field_val = len(field_dataset_val)
                        
                        # Calculate num_augmentations for val split
                        if len_codes_val > 0 and len_field_val > 0:
                            if len_codes_val % len_field_val == 0:
                                num_augmentations_val = len_codes_val // len_field_val
                                print(f">> Val split: codes dataset has {num_augmentations_val}x samples (augmentations)")
                                print(f">>   Codes dataset: {len_codes_val} samples")
                                print(f">>   Field dataset: {len_field_val} samples")
                                print(f">>   Using modulo mapping: codes[i] -> field[i % {len_field_val}]")
                            else:
                                print(f">> WARNING: Val codes dataset length ({len_codes_val}) is not a multiple of field dataset length ({len_field_val})")
                                num_augmentations_val = None
                        else:
                            num_augmentations_val = None
                        
                        if len_codes_val == len_field_val:
                            print(f">> Val dataset lengths match: {len_codes_val} samples (no augmentation)")
                            num_augmentations_val = 1
                    else:
                        num_augmentations_val = None
                    
                    print(">> Field loaders created successfully")
                    print(f">>   field_loader_train: {field_loader_train is not None}")
                    print(f">>   field_loader_val: {field_loader_val is not None}")
                    print(">> Will use index-based mapping (codes and molecules have matching order)")
                    
                    # Store num_augmentations and field dataset sizes in config for later use in model
                    config["num_augmentations_train"] = num_augmentations_train
                    config["field_dataset_size_train"] = len_field_train
                    if 'num_augmentations_val' in locals():
                        config["num_augmentations_val"] = num_augmentations_val
                    if 'len_field_val' in locals():
                        config["field_dataset_size_val"] = len_field_val
                except Exception as e:
                    print(f">> Warning: Failed to create field loaders: {e}")
                    if position_weight_config.get("enabled", False):
                        print(">> Position weighting will be disabled for on_the_fly=False mode")
                    if field_loss_finetune_config.get("enabled", False):
                        print(">> Field loss finetune will be disabled (field dataset not available)")
                    if joint_finetune_enabled:
                        print(">> Joint finetune will be disabled (field dataset not available)")
                    print(">> Consider using on_the_fly=True mode")
                    import traceback
                    traceback.print_exc()
            
            # 获取数据增强数量（如果使用数据增强的codes）
            # 优先使用配置中明确指定的 num_augmentations
            # 如果没有指定，会由 create_code_loaders 自动推断
            num_augmentations = config.get("num_augmentations", None)
            
            # 如果配置中指定了 num_augmentations，打印信息
            if num_augmentations is not None:
                print(f">> Using specified num_augmentations={num_augmentations} for codes loading")
                print(f">> Will load: codes_aug{num_augmentations}.lmdb and position_weights_aug{num_augmentations}.lmdb")
            else:
                print(f">> num_augmentations not specified in config, will auto-infer from codes directory")
            
            # 优先使用checkpoint中的code_stats（如果存在）
            if checkpoint_code_stats is not None:
                print(f">> Using code_stats from checkpoint (resuming training)")
                code_stats = checkpoint_code_stats
            else:
                # 尝试从指定路径加载code_stats
                code_stats_path = config.get("code_stats_path", None)
                if code_stats_path is not None and os.path.exists(code_stats_path):
                    try:
                        print(f">> Loading code_stats from specified path: {code_stats_path}")
                        code_stats = torch.load(code_stats_path, map_location='cpu', weights_only=False)
                        print(f">> Successfully loaded code_stats from file:")
                        print(f">>   mean: {code_stats.get('mean', 'N/A')}")
                        print(f">>   std: {code_stats.get('std', 'N/A')}")
                        print(f">>   max_normalized: {code_stats.get('max_normalized', 'N/A')}")
                        print(f">>   min_normalized: {code_stats.get('min_normalized', 'N/A')}")
                    except Exception as e:
                        print(f">> Warning: Failed to load code_stats from {code_stats_path}: {e}")
                        print(f">> Will compute code_stats from codes directory instead")
                        code_stats = None
                else:
                    code_stats = None
                
                # 如果仍未加载到code_stats，则重新计算
                if code_stats is None:
                    # NOTE: 这里可以通过 max_samples 近似统计，加速大规模 codes 的统计过程
                    # 例如使用前约 200000w 样本来估计 mean/std (200000w个float)
                    print(f">> Computing code_stats from codes directory (new training or checkpoint has no code_stats)")
                    code_stats = compute_code_stats_offline(
                        loader_train,
                        "train",
                        config["normalize_codes"],
                        num_augmentations=num_augmentations,
                        max_samples=None,
                    )
        
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
    
    # Use checkpoint config for model initialization if available, otherwise use current config
    model_config = checkpoint_config if checkpoint_config is not None else config
    if checkpoint_config is not None:
        # Merge checkpoint config with current config (current config takes precedence for non-model params)
        model_config = config.copy()
        # Update dset config from checkpoint (especially grid_size and anchor_spacing)
        if "dset" in checkpoint_config:
            model_config["dset"] = config["dset"].copy()
            # Override grid_size and anchor_spacing from checkpoint
            model_config["dset"]["grid_size"] = checkpoint_config["dset"]["grid_size"]
            model_config["dset"]["anchor_spacing"] = checkpoint_config["dset"]["anchor_spacing"]
            print(f">> Using grid_size={model_config['dset']['grid_size']} and anchor_spacing={model_config['dset']['anchor_spacing']} from checkpoint")
    
    # Initialize Lightning model
    print(f">> Initializing model with field_loader_train={field_loader_train is not None}, field_loader_val={field_loader_val is not None}")
    model = FuncmolLightningModule(model_config, enc, dec_module, code_stats, 
                                   field_loader_train=field_loader_train, 
                                   field_loader_val=field_loader_val)
    print(f">> Model initialized: _field_dataset_train={model._field_dataset_train is not None}, _field_dataset_val={model._field_dataset_val is not None}")
    
    # Load checkpoint if specified
    if config["reload_model_path"] is not None:
        try:
            training_state = load_checkpoint_state_fm(model, config["reload_model_path"])
            
            # Apply training state
            if training_state["epoch"] is not None:
                print(f"Resuming from epoch {training_state['epoch']}")
            
            model.train_losses = training_state["train_losses"]
            model.val_losses = training_state["val_losses"]
            model.best_loss = training_state["best_loss"]
                
            print(f"Successfully loaded checkpoint from: {config['reload_model_path']}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            import traceback
            traceback.print_exc()
            print(f"\nNote: If you want to start training from scratch, set reload_model_path: null")
    
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
        # name="tensorboard",
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
        check_val_every_n_epoch=1,  # Validate every epoch (save checkpoint every epoch)
        log_every_n_steps=200,  # Log metrics every N training steps (in addition to epoch-level logging)
        # Use built-in gradient clipping
        gradient_clip_val=config.get("max_grad_norm", 1.0),
        gradient_clip_algorithm="norm",
        # overfit_batches=1, TODO：可以通过修改这里，开启调试模式
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