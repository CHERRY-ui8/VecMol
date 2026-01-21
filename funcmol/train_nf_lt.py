import sys
sys.path.append("..")

# Standard libraries
import os

# Set GPU environment BEFORE importing torch (must be before any CUDA initialization)
# TODO: set gpus based on server id
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3,4,5,6"
# os.environ['CUDA_VISIBLE_DEVICES'] = "0,2,3,4,5"
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"

# Data visualization and processing
import matplotlib.pyplot as plt

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
from omegaconf import OmegaConf, DictConfig
import hydra

from funcmol.utils.utils_nf import (
    create_neural_field, 
    load_checkpoint_state_nf,
    compute_decoder_field_loss
)
from funcmol.utils.utils_fm import find_checkpoint_path
from funcmol.dataset.dataset_field import create_field_loaders, create_gnf_converter
from funcmol.models.funcmol import create_funcmol


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
            self.atom_distance_scale = loss_weight_config.get("atom_distance_scale", 1.0)  # 距离衰减尺度
        else:
            # Standard MSE loss for non-weighted mode
            self.criterion = nn.MSELoss()
            print("Loss weighting disabled, using standard MSE loss")
        
        # Fine-tuning settings: freeze encoder and only train decoder
        finetune_config = config.get("finetune_decoder", {})
        self.finetune_enabled = finetune_config.get("enabled", False)
        
        # Option to use denoiser for generating codes instead of encoder
        self.use_denoiser_for_codes = finetune_config.get("use_denoiser_for_codes", False)
        self.denoiser = None
        
        # On-the-fly mode: use encoder to compute codes in real-time during fine-tuning
        # This is the default when finetune_enabled=True and use_denoiser_for_codes is False
        self.use_on_the_fly_codes = self.finetune_enabled and not self.use_denoiser_for_codes
        if self.use_on_the_fly_codes:
            print(f">> On-the-fly code generation enabled: codes will be computed from encoder in real-time during fine-tuning")
        
        # Configuration for use_denoiser_for_codes mode
        if self.use_denoiser_for_codes:
            self.sample_near_atoms_only = finetune_config.get("sample_near_atoms_only", True)
            # atom_distance_threshold 现在从 dset 配置中读取，不再从 finetune_config 读取
            self.atom_distance_threshold = config.get("dset", {}).get("atom_distance_threshold", 0.5)
            self.use_cosine_loss = finetune_config.get("use_cosine_loss", False)  # 默认使用MSE loss
            self.magnitude_loss_weight = finetune_config.get("magnitude_loss_weight", 0.1)
            self.n_points = finetune_config.get("n_points", None)  # If None, use dset.n_points
            self.max_timestep_for_decoder = finetune_config.get("max_timestep_for_decoder", 5)  # 使用很小的timestep范围（0到max_timestep_for_decoder-1）进行轻微加噪
            
            print(f"use_denoiser_for_codes mode enabled:")
            print(f"  - sample_near_atoms_only: {self.sample_near_atoms_only}")
            print(f"  - atom_distance_threshold: {self.atom_distance_threshold}Å")
            print(f"  - use_cosine_loss: {self.use_cosine_loss}")
            if self.use_cosine_loss:
                print(f"  - magnitude_loss_weight: {self.magnitude_loss_weight}")
            if self.n_points is not None:
                print(f"  - n_points: {self.n_points}")
            print(f"  - max_timestep_for_decoder: {self.max_timestep_for_decoder} (using small timesteps t=0 to {self.max_timestep_for_decoder-1} for slight noise perturbation, then denoising)")
        
        if self.use_denoiser_for_codes:
            # Load denoiser model for code generation
            denoiser_path = finetune_config.get("denoiser_checkpoint_path", None)
            if denoiser_path is None:
                raise ValueError("use_denoiser_for_codes=True requires denoiser_checkpoint_path to be specified")
            
            # Find checkpoint file if directory is provided
            checkpoint_path = find_checkpoint_path(denoiser_path)
            print(f">> Loading denoiser from: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            
            # Extract config from checkpoint (Lightning saves in hyper_parameters, may be nested under 'config')
            hyper_params = checkpoint.get("hyper_parameters", {})
            denoiser_config = hyper_params.get("config", hyper_params)
            
            # Convert OmegaConf to dict if needed
            try:
                if isinstance(denoiser_config, DictConfig):
                    denoiser_config = OmegaConf.to_container(denoiser_config, resolve=True)
            except:
                pass
            
            # Ensure required keys exist (use current config as fallback, then defaults)
            denoiser_config.setdefault("smooth_sigma", config.get("smooth_sigma", 0.0))
            denoiser_config.setdefault("diffusion_method", config.get("diffusion_method", "new_x0"))
            denoiser_config.setdefault("denoiser", config.get("denoiser", {}))
            denoiser_config.setdefault("ddpm", config.get("ddpm", {"num_timesteps": 1000, "use_time_weight": True}))
            denoiser_config.setdefault("decoder", config.get("decoder", {}))
            denoiser_config.setdefault("dset", config.get("dset", {}))
            
            # Create denoiser model
            self.denoiser = create_funcmol(denoiser_config)
            
            # Load denoiser state dict
            if "funcmol_state_dict" in checkpoint:
                state_dict = checkpoint["funcmol_state_dict"]
            elif "state_dict" in checkpoint:
                state_dict = {k[8:]: v for k, v in checkpoint["state_dict"].items() if k.startswith("funcmol.")}
            else:
                raise ValueError(f"Could not find funcmol_state_dict in checkpoint: {checkpoint_path}")
            
            # Handle _orig_mod. prefix if present (from torch.compile)
            new_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
            self.denoiser.load_state_dict(new_state_dict, strict=False)
            self.denoiser.eval()
            for param in self.denoiser.parameters():
                param.requires_grad = False
            
            # CRITICAL: Load code_stats from denoiser checkpoint for code normalization
            # Denoiser was trained on normalized codes, so we must normalize encoder-generated codes
            self.denoiser_code_stats = checkpoint.get("code_stats", None)
            denoiser_normalize_codes = denoiser_config.get("normalize_codes", False)
            
            if denoiser_normalize_codes:
                if self.denoiser_code_stats is None:
                    print(f">> WARNING: denoiser was trained with normalize_codes=True but code_stats not found in checkpoint!")
                    print(f">>   This may cause codes mismatch. Denoiser expects normalized codes.")
                    self.denoiser_code_stats = None
                else:
                    print(f">> Loaded code_stats from denoiser checkpoint for code normalization")
                    # Handle both tensor and scalar types for mean/std
                    mean_info = 'N/A'
                    std_info = 'N/A'
                    if 'mean' in self.denoiser_code_stats:
                        mean_val = self.denoiser_code_stats['mean']
                        if hasattr(mean_val, 'shape'):
                            mean_info = str(mean_val.shape)
                        else:
                            mean_info = f"scalar ({type(mean_val).__name__})"
                    if 'std' in self.denoiser_code_stats:
                        std_val = self.denoiser_code_stats['std']
                        if hasattr(std_val, 'shape'):
                            std_info = str(std_val.shape)
                        else:
                            std_info = f"scalar ({type(std_val).__name__})"
                    print(f">>   code_stats mean: {mean_info}")
                    print(f">>   code_stats std: {std_info}")
            else:
                self.denoiser_code_stats = None
                print(f">> Denoiser was trained without code normalization (normalize_codes=False)")
            
            print(f">> Denoiser loaded successfully. Will use slightly perturbed and denoised codes (small timestep noise + denoising) for decoder training.")
        
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
    
    
    def _get_denoised_code(self, x_0):
        """
        Get slightly perturbed and denoised code by adding noise using a small timestep, then denoising.
        This ensures codes are only slightly perturbed (using small timestep like t=0,1,2,3,4),
        and after denoising, the codes should be close to the original x_0, so decoder can 
        still generate fields similar to ground truth.
        
        Args:
            x_0: Ground truth codes [B, n_grid, code_dim]
            
        Returns:
            denoised_code: [B, n_grid, code_dim] tensor containing denoised code after slight perturbation.
                This is a code that was slightly perturbed (using small timestep) and then denoised,
                so it should be close to the original x_0.
        """
        from funcmol.models.ddpm import q_sample, extract
        
        device = x_0.device
        B = x_0.shape[0]
        diffusion_consts = self.denoiser.diffusion_consts
        
        # Use a small timestep (randomly sample from 0 to max_timestep_for_decoder-1)
        # These are timesteps close to 0, where noise is minimal, ensuring only slight perturbation
        t_val = torch.randint(0, self.max_timestep_for_decoder, (B,), device=device, dtype=torch.long)
        t = t_val  # [B]
        
        # Add noise to x_0 to get x_t (forward diffusion with small timestep)
        noise = torch.randn_like(x_0)
        x_t = q_sample(x_0, t, diffusion_consts, noise)
        
        # Denoise using the model (predict x0)
        # CRITICAL: We want the predicted x_0 directly, not x_{t-1}
        with torch.no_grad():
            if self.denoiser.diffusion_method == "new_x0":
                # Model directly predicts x0
                predicted_x0 = self.denoiser.net(x_t, t)
            else:
                # For "new" method, model predicts noise (epsilon)
                # We need to convert from noise prediction to x0 prediction
                # Formula: x_0 = (x_t - √(1 - α̅_t) · ε_θ) / √(α̅_t)
                predicted_noise = self.denoiser.net(x_t, t)  # [B, n_grid, code_dim]
                
                # Extract diffusion constants
                sqrt_alphas_cumprod_t = extract(diffusion_consts["sqrt_alphas_cumprod"], t, x_t)
                sqrt_one_minus_alphas_cumprod_t = extract(diffusion_consts["sqrt_one_minus_alphas_cumprod"], t, x_t)
                
                # Convert noise prediction to x0 prediction
                predicted_x0 = (x_t - sqrt_one_minus_alphas_cumprod_t * predicted_noise) / sqrt_alphas_cumprod_t
        
        return predicted_x0  # [B, n_grid, code_dim]
    
    def _prepare_code_stats(self, code_stats_dict, device, dtype):
        """Prepare code_stats dictionary, converting scalars to tensors if needed."""
        if code_stats_dict is None:
            return None
        prepared = {}
        for key in ['mean', 'std']:
            if key in code_stats_dict:
                val = code_stats_dict[key]
                if not isinstance(val, torch.Tensor):
                    prepared[key] = torch.tensor(val, device=device, dtype=dtype)
                else:
                    prepared[key] = val.to(device=device, dtype=dtype)
        return prepared
    
    def forward(self, batch, codes=None):
        """
        Forward pass through the neural field model
        
        Args:
            batch: PyTorch Geometric batch object
            codes: Optional precomputed codes [B, n_grid, code_dim]. If None, codes will be generated.
            
        Returns:
            torch.Tensor: Predicted field
        """
        # Get query points from batch and ensure they're on the correct device
        query_points = batch.xs
        
        # Get codes from encoder or denoiser
        if codes is not None:
            # Use provided codes (should not happen in normal flow)
            pass  # codes already provided
        elif self.use_denoiser_for_codes:
            # Use denoiser to generate codes from ground truth codes (via encoder)
            # First, get ground truth codes from encoder
            with torch.no_grad():
                x_0 = self.enc(batch)  # [B, n_grid, code_dim]
            
            # CRITICAL: Normalize codes before passing to denoiser
            # Denoiser was trained on normalized codes, so encoder-generated codes must be normalized
            if hasattr(self, 'denoiser_code_stats') and self.denoiser_code_stats is not None:
                from funcmol.utils.utils_nf import normalize_code
                code_stats = self._prepare_code_stats(self.denoiser_code_stats, x_0.device, x_0.dtype)
                x_0 = normalize_code(x_0, code_stats)
            
            # Get denoised code from final timestep using denoiser
            codes = self._get_denoised_code(x_0)  # [B, n_grid, code_dim]
            
            # CRITICAL: Unnormalize codes after denoising (decoder expects unnormalized codes)
            if hasattr(self, 'denoiser_code_stats') and self.denoiser_code_stats is not None:
                from funcmol.utils.utils_nf import unnormalize_code
                code_stats = self._prepare_code_stats(self.denoiser_code_stats, codes.device, codes.dtype)
                codes = unnormalize_code(codes, code_stats)
        else:
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
            # Use n_points from finetune config if available, otherwise use dset.n_points
            if self.use_denoiser_for_codes and self.n_points is not None:
                n_points = self.n_points
            else:
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
            sample_weights = torch.exp(-min_distances / self.atom_distance_scale)
            
            weights[b] = sample_weights
        
        return weights
    
    def _compute_separate_loss(self, pred_field, target_field, point_types, 
                               use_cosine_loss=False, magnitude_loss_weight=0.1,
                               use_weighting=False, batch=None, query_points=None):
        """
        分别计算grid点和邻近点的loss
        
        Args:
            pred_field: [B, n_points, n_atom_types, 3] 预测的field
            target_field: [B, n_points, n_atom_types, 3] 目标field
            point_types: [B, n_points] 或 [N_total_points] 点类型标记，0=grid点，1=邻近点
            use_cosine_loss: 是否使用cosine loss
            magnitude_loss_weight: magnitude loss的权重
            use_weighting: 是否使用距离加权
            batch: batch对象（用于计算权重）
            query_points: [B, n_points, 3] query点坐标（用于计算权重）
            
        Returns:
            loss: 总loss（标量）
            loss_dict: 包含各种loss的字典
        """
        B, n_points, n_atom_types, _ = pred_field.shape
        device = pred_field.device
        
        # Reshape point_types if needed
        # In PyTorch Geometric batch, point_types might be [N_total_points] or [B, n_points]
        if point_types.dim() == 1:
            # [N_total_points] -> [B, n_points]
            # Need to handle batching: each sample has n_points points
            if point_types.shape[0] == B * n_points:
                point_types = point_types.view(B, n_points)
            elif point_types.shape[0] == n_points:
                # Single sample case, expand to batch
                point_types = point_types.unsqueeze(0).expand(B, -1)
            else:
                raise ValueError(f"Unexpected point_types shape: {point_types.shape}, expected {B * n_points} or {n_points}")
        elif point_types.dim() == 2:
            # Already [B, n_points] or might need reshaping
            if point_types.shape[0] == B and point_types.shape[1] == n_points:
                pass  # Already correct shape
            else:
                raise ValueError(f"Unexpected point_types 2D shape: {point_types.shape}, expected [{B}, {n_points}]")
        else:
            raise ValueError(f"Unexpected point_types shape: {point_types.shape}")
        
        # Create masks for grid and neighbor points
        grid_mask = (point_types == 0)  # [B, n_points]
        neighbor_mask = (point_types == 1)  # [B, n_points]
        
        # Get loss weights if needed
        if use_weighting and batch is not None and query_points is not None:
            weights = self._compute_loss_weights(batch, query_points, target_field)  # [B, n_points]
        else:
            weights = torch.ones(B, n_points, device=device)
        
        # Compute element-wise loss
        if use_cosine_loss:
            # Use cosine distance + magnitude loss
            from funcmol.utils.utils_nf import compute_decoder_field_loss
            
            # Compute loss for grid points
            grid_loss = compute_decoder_field_loss(
                pred_field * grid_mask.unsqueeze(-1).unsqueeze(-1),
                target_field * grid_mask.unsqueeze(-1).unsqueeze(-1),
                use_cosine_loss=True,
                magnitude_loss_weight=magnitude_loss_weight,
                valid_mask=grid_mask
            )
            
            # Compute loss for neighbor points
            neighbor_loss = compute_decoder_field_loss(
                pred_field * neighbor_mask.unsqueeze(-1).unsqueeze(-1),
                target_field * neighbor_mask.unsqueeze(-1).unsqueeze(-1),
                use_cosine_loss=True,
                magnitude_loss_weight=magnitude_loss_weight,
                valid_mask=neighbor_mask
            )
        else:
            # Use MSE loss
            elementwise_loss = self.criterion(pred_field, target_field)  # [B, n_points, n_atom_types, 3]
            pointwise_loss = elementwise_loss.mean(dim=(-2, -1))  # [B, n_points]
            
            # Apply weights
            weighted_loss = pointwise_loss * weights  # [B, n_points]
            
            # Separate loss for grid and neighbor points
            grid_loss = (weighted_loss * grid_mask).sum() / (grid_mask.sum() + 1e-8)
            neighbor_loss = (weighted_loss * neighbor_mask).sum() / (neighbor_mask.sum() + 1e-8)
        
        # Get loss weights from config (if specified)
        loss_weight_config = self.config.get("loss_weighting", {})
        grid_loss_weight = loss_weight_config.get("grid_loss_weight", 1.0)
        neighbor_loss_weight = loss_weight_config.get("neighbor_loss_weight", 1.0)
        
        # Combine losses
        total_loss = grid_loss_weight * grid_loss + neighbor_loss_weight * neighbor_loss
        
        # Create loss dictionary
        loss_dict = {
            "loss": total_loss,
            "grid_loss": grid_loss,
            "neighbor_loss": neighbor_loss,
            "grid_points_count": float(grid_mask.sum().item()),
            "neighbor_points_count": float(neighbor_mask.sum().item())
        }
        
        return total_loss, loss_dict
    
    def _process_batch(self, batch, codes=None):
        """
        Process a batch and return the predicted and target fields
        
        Args:
            batch: PyTorch Geometric batch object
            codes: Optional precomputed codes [B, n_grid, code_dim]
            
        Returns:
            tuple: (pred_field, target_field) with matching dimensions [B, n_points, n_atom_types, 3]
        """
        # Get predictions from forward pass
        pred_field = self(batch, codes=codes)  # [B, n_points, n_atom_types, 3]
        
        # Get target field from batch
        target_field = batch.target_field
        
        # Get batch size and number of points
        B = len(batch)
        # Use n_points from finetune config if available, otherwise use dset.n_points
        if self.use_denoiser_for_codes and self.n_points is not None:
            n_points = self.n_points
        else:
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
        # Codes will be computed on-the-fly in forward() method
        pred_field, target_field = self._process_batch(batch, codes=None)
        
        # Get point types if available (for separate grid/neighbor loss computation)
        has_point_types = hasattr(batch, 'point_types')
        
        # Calculate loss
        if self.use_denoiser_for_codes and self.use_cosine_loss:
            # Use cosine distance + magnitude loss for denoiser-based training
            if has_point_types:
                # Separate loss computation for grid and neighbor points
                loss, loss_dict = self._compute_separate_loss(
                    pred_field, target_field, batch.point_types,
                    use_cosine_loss=True, magnitude_loss_weight=self.magnitude_loss_weight
                )
            else:
                loss = compute_decoder_field_loss(
                    pred_field,
                    target_field,
                    use_cosine_loss=True,
                    magnitude_loss_weight=self.magnitude_loss_weight
                )
                loss_dict = {"loss": loss}
        elif self.loss_weighting_enabled:
            # Get query points
            query_points = batch.xs
            B = len(batch)
            # Use n_points from finetune config if available, otherwise use dset.n_points
            if self.use_denoiser_for_codes and self.n_points is not None:
                n_points = self.n_points
            else:
                n_points = self.config["dset"]["n_points"]
            if query_points.dim() == 2:
                query_points = query_points.view(B, n_points, 3)
            
            if has_point_types:
                # Separate loss computation for grid and neighbor points
                loss, loss_dict = self._compute_separate_loss(
                    pred_field, target_field, batch.point_types,
                    use_cosine_loss=False, use_weighting=True,
                    batch=batch, query_points=query_points
                )
            else:
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
                loss_dict = {"loss": loss}
        else:
            # Standard MSE loss
            if has_point_types:
                # Separate loss computation for grid and neighbor points
                loss, loss_dict = self._compute_separate_loss(
                    pred_field, target_field, batch.point_types,
                    use_cosine_loss=False, use_weighting=False
                )
            else:
                loss = self.criterion(pred_field, target_field)
                loss_dict = {"loss": loss}
        
        # Log separate losses if available
        if "grid_loss" in loss_dict:
            self.log("train/grid_loss", loss_dict["grid_loss"], batch_size=len(batch), on_step=True, on_epoch=True, prog_bar=False)
            self.log("train/neighbor_loss", loss_dict["neighbor_loss"], batch_size=len(batch), on_step=True, on_epoch=True, prog_bar=False)
            self.log("train/grid_points_count", loss_dict["grid_points_count"], batch_size=len(batch), on_step=True, on_epoch=True, prog_bar=False)
            self.log("train/neighbor_points_count", loss_dict["neighbor_points_count"], batch_size=len(batch), on_step=True, on_epoch=True, prog_bar=False)
        
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
        # Codes will be computed on-the-fly in forward() method
        pred_field, target_field = self._process_batch(batch, codes=None)
        
        # Get point types if available (for separate grid/neighbor loss computation)
        has_point_types = hasattr(batch, 'point_types')
        loss_dict = {}  # Initialize loss_dict
        
        # Calculate loss
        if self.use_denoiser_for_codes and self.use_cosine_loss:
            # Use cosine distance + magnitude loss for denoiser-based training
            if has_point_types:
                # Separate loss computation for grid and neighbor points
                loss, loss_dict = self._compute_separate_loss(
                    pred_field, target_field, batch.point_types,
                    use_cosine_loss=True, magnitude_loss_weight=self.magnitude_loss_weight
                )
            else:
                loss = compute_decoder_field_loss(
                    pred_field,
                    target_field,
                    use_cosine_loss=True,
                    magnitude_loss_weight=self.magnitude_loss_weight
                )
                loss_dict = {"loss": loss}
        elif self.loss_weighting_enabled:
            # Get query points
            query_points = batch.xs
            B = len(batch)
            # Use n_points from finetune config if available, otherwise use dset.n_points
            if self.use_denoiser_for_codes and self.n_points is not None:
                n_points = self.n_points
            else:
                n_points = self.config["dset"]["n_points"]
            if query_points.dim() == 2:
                query_points = query_points.view(B, n_points, 3)
            
            if has_point_types:
                # Separate loss computation for grid and neighbor points
                loss, loss_dict = self._compute_separate_loss(
                    pred_field, target_field, batch.point_types,
                    use_cosine_loss=False, use_weighting=True,
                    batch=batch, query_points=query_points
                )
            else:
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
                loss_dict = {"loss": loss}
        else:
            # Standard MSE loss
            if has_point_types:
                # Separate loss computation for grid and neighbor points
                loss, loss_dict = self._compute_separate_loss(
                    pred_field, target_field, batch.point_types,
                    use_cosine_loss=False, use_weighting=False
                )
            else:
                loss = self.criterion(pred_field, target_field)
                loss_dict = {"loss": loss}
        
        # Log separate losses if available
        if "grid_loss" in loss_dict:
            self.log("val/grid_loss", loss_dict["grid_loss"], batch_size=len(batch), on_step=False, on_epoch=True, prog_bar=False)
            self.log("val/neighbor_loss", loss_dict["neighbor_loss"], batch_size=len(batch), on_step=False, on_epoch=True, prog_bar=False)
            self.log("val/grid_points_count", loss_dict["grid_points_count"], batch_size=len(batch), on_step=False, on_epoch=True, prog_bar=False)
            self.log("val/neighbor_points_count", loss_dict["neighbor_points_count"], batch_size=len(batch), on_step=False, on_epoch=True, prog_bar=False)
        
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
        
        # Keep denoiser in eval mode if used
        if self.use_denoiser_for_codes and self.denoiser is not None:
            self.denoiser.eval()
        
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
        
        # Keep denoiser in eval mode if used
        if self.use_denoiser_for_codes and self.denoiser is not None:
            self.denoiser.eval()
    
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
        
        # Check if warmup is enabled
        warmup_config = self.config.get("warmup", {})
        warmup_enabled = warmup_config.get("enabled", False)
        warmup_steps = warmup_config.get("warmup_steps", 1000)
        
        schedulers = []
        
        # Create warmup scheduler if enabled
        if warmup_enabled:
            # LambdaLR for linear warmup: lr = base_lr * (step / warmup_steps)
            # After warmup_steps, lr = base_lr (multiplier = 1.0)
            def warmup_lambda(step):
                if step < warmup_steps:
                    return float(step + 1) / float(warmup_steps)
                else:
                    return 1.0
            
            warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=warmup_lambda
            )
            
            warmup_scheduler_config = {
                "scheduler": warmup_scheduler,
                "interval": "step",  # Warmup is step-based
                "frequency": 1,
            }
            schedulers.append(warmup_scheduler_config)
            
            print(f"Warmup enabled:")
            print(f"  - warmup_steps: {warmup_steps} (learning rate will linearly increase from 0 to target LR over {warmup_steps} steps)")
        
        # Create learning rate decay scheduler if needed
        if "lr_decay" in self.config and self.config["lr_decay"]:
            # Get scheduler type from config (default: "plateau" for ReduceLROnPlateau)
            scheduler_type = self.config.get("lr_scheduler_type", "plateau")
            
            if scheduler_type == "plateau":
                # Use ReduceLROnPlateau: reduce LR when val_loss stops decreasing
                factor = self.config.get("lr_factor", 0.5)  # 学习率衰减因子
                patience = self.config.get("lr_patience", 10)  # 等待多少个epoch没有改善
                min_lr = self.config.get("lr_min", 1e-6)  # 最小学习率
                mode = self.config.get("lr_mode", "min")  # 'min' for val_loss (default)
                
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode=mode,
                    factor=factor,
                    patience=patience,
                    min_lr=min_lr,
                    verbose=True  # 打印学习率变化信息
                )
                
                print(f"Using ReduceLROnPlateau scheduler:")
                print(f"  - factor: {factor} (LR will be multiplied by this when plateau detected)")
                print(f"  - patience: {patience} (wait {patience} epochs without improvement)")
                print(f"  - min_lr: {min_lr} (minimum learning rate)")
                print(f"  - mode: {mode} (monitoring val_loss)")
                
                # Configure the scheduler for PyTorch Lightning
                scheduler_config = {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                    "monitor": "val_loss"  # Monitor validation loss
                }
                schedulers.append(scheduler_config)
            
            elif scheduler_type == "multistep":
                # Use MultiStepLR: reduce LR at fixed milestones
                milestones = self.config.get("lr_milestones", [500])
                gamma = self.config.get("lr_gamma", 0.1)
                
                print(f"Using MultiStepLR scheduler:")
                print(f"  - milestones: {milestones} (LR will be reduced at these epochs)")
                print(f"  - gamma: {gamma} (LR will be multiplied by this at each milestone)")
                    
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
                schedulers.append(scheduler_config)
            
            else:
                raise ValueError(f"Unknown scheduler type: {scheduler_type}. Choose 'plateau' or 'multistep'")
        
        # Return optimizer with schedulers
        # PyTorch Lightning expects:
        # - Single scheduler: {"optimizer": optimizer, "lr_scheduler": scheduler_config_dict}
        # - Multiple schedulers: {"optimizer": optimizer, "lr_scheduler": [scheduler_config_dict1, scheduler_config_dict2]}
        if len(schedulers) == 0:
            return optimizer
        elif len(schedulers) == 1:
            return {"optimizer": optimizer, "lr_scheduler": schedulers[0]}
        else:
            return {"optimizer": optimizer, "lr_scheduler": schedulers}
    
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
    
    # Resolve config variables and convert to container for modification
    config = OmegaConf.to_container(config, resolve=True)
    
    # Setup GNF Converter for data processing
    data_gnf_converter = create_gnf_converter(config)
    
    # Check if use_denoiser_for_codes mode is enabled
    finetune_config = config.get("finetune_decoder", {})
    finetune_enabled = finetune_config.get("enabled", False)
    use_denoiser_for_codes = finetune_config.get("use_denoiser_for_codes", False)
    
    # On-the-fly mode: use encoder to compute codes in real-time (default for fine-tuning)
    use_on_the_fly_codes = finetune_enabled and not use_denoiser_for_codes
    
    # CRITICAL: Disable data augmentation in fine-tuning mode
    # When encoder is frozen, random rotations would cause codes to change, breaking the correspondence
    # between codes and target fields. We need consistent codes for the same molecule.
    if finetune_enabled:
        original_data_aug = config["dset"].get("data_aug", False)
        config["dset"]["data_aug"] = False
        if use_on_the_fly_codes:
            print(f">> [on-the-fly codes] Disabled data augmentation (rotation) for fine-tuning")
            print(f">>   Encoder is frozen, so codes must be consistent for the same molecule")
        elif use_denoiser_for_codes:
            print(f">> [use_denoiser_for_codes] Disabled data augmentation (rotation) to match denoiser training data")
        print(f">>   Original data_aug value: {original_data_aug}, now set to: False")
    
    if use_denoiser_for_codes:
        # Add sampling configuration to config for create_field_loaders
        # This will be used by dataset_field.py to enable sample_near_atoms_only
        if "joint_finetune" not in config:
            config["joint_finetune"] = {}
        config["joint_finetune"]["enabled"] = True  # Enable joint_finetune mode for sampling
        config["joint_finetune"]["sample_near_atoms_only"] = finetune_config.get("sample_near_atoms_only", True)
        # atom_distance_threshold 现在在 dset 配置中设置，不再在此处设置
        # 如果 finetune_config 中指定了 atom_distance_threshold，则设置到 dset 中
        if "atom_distance_threshold" in finetune_config:
            config["dset"]["atom_distance_threshold"] = finetune_config["atom_distance_threshold"]
        if "n_points" in finetune_config and finetune_config["n_points"] is not None:
            config["joint_finetune"]["n_points"] = finetune_config["n_points"]
        
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
