import sys
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "2"
import omegaconf
import torch
from tqdm import tqdm
import time
import hydra
import math
import random
# Add current dir to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# torch._dynamo config for compile compatibility
import torch._dynamo
torch._dynamo.config.suppress_errors = True

from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_batch

from vecmol.utils.utils_nf import load_neural_field
from vecmol.dataset.dataset_field import create_field_loaders, create_gnf_converter
from vecmol.utils.utils_diffusion import compute_position_weights
from vecmol.models.encoder import create_grid_coords


def _random_rot_matrix(device=None) -> torch.Tensor:
    """Generate random rotation matrix (around x, y, z axes).
    
    Returns:
        torch.Tensor: 3x3 rotation matrix
    """
    theta = random.uniform(0, 2) * math.pi
    rot_x = torch.tensor(
        [
            [1, 0, 0],
            [0, math.cos(theta), -math.sin(theta)],
            [0, math.sin(theta), math.cos(theta)],
        ],
        device=device
    )
    theta = random.uniform(0, 2) * math.pi
    rot_y = torch.tensor(
        [
            [math.cos(theta), 0, -math.sin(theta)],
            [0, 1, 0],
            [math.sin(theta), 0, math.cos(theta)],
        ],
        device=device
    )
    theta = random.uniform(0, 2) * math.pi
    rot_z = torch.tensor(
        [
            [math.cos(theta), -math.sin(theta), 0],
            [math.sin(theta), math.cos(theta), 0],
            [0, 0, 1],
        ],
        device=device
    )
    return rot_z @ rot_y @ rot_x


def apply_rotation_to_batch(batch, device):
    """Apply random rotation to molecule coords in batch.
    
    Args:
        batch: torch_geometric Batch
        device: Device
        
    Returns:
        Augmented batch
    """
    augmented_batch = batch.clone()
    coords = batch.pos
    coords_dense, mask = to_dense_batch(coords, batch.batch, fill_value=0.0)
    batch_size = coords_dense.shape[0]
    
    for b in range(batch_size):
        valid_mask = mask[b]
        mol_coords = coords_dense[b][valid_mask]
        
        if mol_coords.shape[0] > 0:
            center = mol_coords.mean(dim=0, keepdim=True)
            rot_matrix = _random_rot_matrix(device=device)
            mol_coords_centered = mol_coords - center
            mol_coords_rotated = mol_coords_centered @ rot_matrix.T
            mol_coords_rotated = mol_coords_rotated + center
            coords_dense[b][valid_mask] = mol_coords_rotated
    
    # Convert dense back to flat
    augmented_coords = []
    for b in range(batch_size):
        valid_mask = mask[b]
        augmented_coords.append(coords_dense[b][valid_mask])
    augmented_batch.pos = torch.cat(augmented_coords, dim=0)
    
    return augmented_batch


def apply_translation_to_batch(batch, anchor_spacing, device):
    """Apply random translation to molecule coords in batch.
    
    Args:
        batch: torch_geometric Batch
        anchor_spacing: Anchor spacing (Angstrom)
        device: Device
        
    Returns:
        Augmented batch
    """
    augmented_batch = batch.clone()
    coords = batch.pos
    translation_distance = anchor_spacing / 2.0
    from torch_geometric.utils import to_dense_batch
    coords_dense, mask = to_dense_batch(coords, batch.batch, fill_value=0.0)
    batch_size = coords_dense.shape[0]
    
    for b in range(batch_size):
        translation = (torch.rand(3, device=device) * 2 - 1) * translation_distance
        valid_mask = mask[b]
        coords_dense[b][valid_mask] = coords_dense[b][valid_mask] + translation.unsqueeze(0)
    
    # Convert dense back to flat
    augmented_coords = []
    for b in range(batch_size):
        valid_mask = mask[b]
        augmented_coords.append(coords_dense[b][valid_mask])
    augmented_batch.pos = torch.cat(augmented_coords, dim=0)
    
    return augmented_batch


@hydra.main(config_path="configs", config_name="infer_codes", version_base=None)
def main(config):
    # Validate nf_pretrained_path
    nf_pretrained_path = config.get("nf_pretrained_path")
    if not nf_pretrained_path:
        raise ValueError("nf_pretrained_path must be set to Lightning checkpoint path")
    
    if not nf_pretrained_path.endswith('.ckpt') or not os.path.exists(nf_pretrained_path):
        raise ValueError(f"Checkpoint file not found or invalid: {nf_pretrained_path}")
    
    # dirname from nf_pretrained_path directory
    checkpoint_dir = os.path.dirname(nf_pretrained_path)
    
    # Convert config to plain dict
    if isinstance(config, omegaconf.dictconfig.DictConfig):
        config = omegaconf.OmegaConf.to_container(config, resolve=True)
    
    # Preserve yaml split (override checkpoint)
    yaml_split = config.get("split", "train")
    yaml_use_data_augmentation = config.get("use_data_augmentation", False)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Set random seed
    torch.manual_seed(config.get("seed", 1234))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.get("seed", 1234))
    
    # Load Lightning checkpoint (weights_only=False for omegaconf.DictConfig)
    checkpoint = torch.load(nf_pretrained_path, map_location='cpu', weights_only=False)
    
    # Extract config from Lightning checkpoint
    if 'hyper_parameters' in checkpoint:
        config_model = checkpoint['hyper_parameters']
    else:
        config_model = checkpoint.get('config', {})
    
    # Convert config_model to plain dict
    if isinstance(config_model, omegaconf.dictconfig.DictConfig):
        config_model = omegaconf.OmegaConf.to_container(config_model, resolve=True)
    
    for key in config.keys():
        if key in config_model and \
            isinstance(config_model[key], dict) and isinstance(config[key], dict):
            config_model[key].update(config[key])
        else:
            config_model[key] = config[key]
    config = config_model  # update config with checkpoint config
    
    # Prefer yaml split and use_data_augmentation (override checkpoint)
    config["split"] = yaml_split
    config["use_data_augmentation"] = yaml_use_data_augmentation
    
    # Save dir: codes (with aug) or code_no_aug (after merge)
    use_data_augmentation_config = config.get("use_data_augmentation", False)
    split = config.get("split", "train")
    
    if use_data_augmentation_config and split == "train":
        # With augmentation: save to codes dir
        # dirname = os.path.join(checkpoint_dir, "codes", split)
        dirname = os.path.join(checkpoint_dir, "codes_no_shuffle", split)
    else:
        # No augmentation: save to code_no_aug dir
        dirname = os.path.join(checkpoint_dir, "code_no_aug", split)
    
    config["dirname"] = dirname
    enc, _ = load_neural_field(checkpoint, config)

    # Create GNFConverter for data loading
    gnf_converter = create_gnf_converter(config)
    
    # data loader
    loader = create_field_loaders(config, gnf_converter, split=config["split"])
    
    # Disable shuffle to preserve order (codes index matches data index) 
    loader = DataLoader(
        loader.dataset,
        batch_size=min(config["dset"]["batch_size"], len(loader.dataset)),
        num_workers=config["dset"]["num_workers"],
        shuffle=False,  # Preserve order
        pin_memory=True,
        drop_last=True,
    )
    print(f">> DataLoader shuffle disabled to preserve data order")

    # Print config
    print(f">> config: {config}")
    print(f">> seed: {config['seed']}")

    # create output directory
    print(">> saving codes in", config["dirname"])
    os.makedirs(config["dirname"], exist_ok=True)

    # Augmentation: only for train; val/test use original
    use_data_augmentation = use_data_augmentation_config and (split == "train")
    num_augmentations = config.get("num_augmentations", 1)  # Augmentations per molecule (incl. original)
    apply_rotation = config.get("data_augmentation", {}).get("apply_rotation", True)
    apply_translation = config.get("data_augmentation", {}).get("apply_translation", True)
    anchor_spacing = config.get("dset", {}).get("anchor_spacing", 1.5)
    
    # If augmentation disabled, force 1 (original only)
    if not use_data_augmentation_config:
        num_augmentations = 1
    # Non-train split: disable augmentation
    elif split != "train" and use_data_augmentation_config:
        num_augmentations = 1

    # position_weight config (define before file checks)
    position_weight_config = config.get("position_weight", {})
    compute_position_weights_flag = position_weight_config.get("enabled", False)
    radius = position_weight_config.get("radius", 3.0)
    weight_alpha = position_weight_config.get("alpha", 0.5)
    grid_size = config.get("dset", {}).get("grid_size", 9)

    # Regenerate all codes each run; name by num_augmentations (codes_aug{num}_XXX.pt)
    print(f">> Using augmentation-based naming for codes files (codes_aug{num_augmentations}_XXX.pt)")
    print(f">> Will generate all {num_augmentations} codes files (regenerating all files)")
    indices_to_generate = list(range(num_augmentations))
    
    if use_data_augmentation:
        print(">> Data augmentation enabled:")
        print(f"   - num_augmentations: {num_augmentations}")
        print(f"   - apply_rotation: {apply_rotation}")
        print(f"   - apply_translation: {apply_translation}")
        print(f"   - anchor_spacing: {anchor_spacing}")
    
    # start eval
    print(f">> start code inference in {config['split']} split")
    enc.eval()

    # Temp dir for per-batch codes and position_weights
    temp_dir = os.path.join(config["dirname"], "temp_batches")
    os.makedirs(temp_dir, exist_ok=True)
    
    # position_weight config defined above
    if compute_position_weights_flag:
        print(">> Position weight computation enabled:")
        print(f"   - radius: {radius}")
        print(f"   - alpha: {weight_alpha}")
        print(f"   - grid_size: {grid_size}")
        # Create grid coords (once)
        grid_coords = create_grid_coords(1, grid_size, device=device, anchor_spacing=anchor_spacing)
        grid_coords = grid_coords.squeeze(0).cpu()  # [n_grid, 3] on CPU
    else:
        grid_coords = None
        print(">> Position weight computation disabled")

    with torch.no_grad():
        t0 = time.time()
        batch_idx = 0
        for batch in tqdm(loader):
            batch = batch.to(device)
            
            # For each batch: all augmentations and infer codes
            for aug_idx in indices_to_generate:
                if aug_idx == 0:
                    # First: original (no aug)
                    augmented_batch = batch
                else:
                    # Later: apply augmentation
                    augmented_batch = batch.clone()
                    
                    # Apply rotation
                    if apply_rotation:
                        augmented_batch = apply_rotation_to_batch(augmented_batch, device)
                    
                    # Apply translation
                    if apply_translation:
                        augmented_batch = apply_translation_to_batch(augmented_batch, anchor_spacing, device)
                
                # Infer codes
                codes_batch = enc(augmented_batch)
                codes_batch_cpu = codes_batch.detach().cpu()
                
                # Compute position_weights if enabled
                position_weights_batch = None
                if compute_position_weights_flag:
                    # Get atom coords (augmented batch)
                    atom_coords = augmented_batch.pos.cpu()  # [N_total_atoms, 3]
                    batch_idx_atoms = augmented_batch.batch.cpu()  # [N_total_atoms]
                    
                    # Compute position weights
                    position_weights_batch = compute_position_weights(
                        atom_coords=atom_coords,
                        grid_coords=grid_coords,
                        batch_idx=batch_idx_atoms,
                        radius=radius,
                        weight_alpha=weight_alpha,
                        device=torch.device('cpu')  # CPU to save GPU memory
                    )  # [B, n_grid]
                
                # Save to temp file immediately
                temp_file = os.path.join(temp_dir, f"codes_{aug_idx:03d}_batch_{batch_idx:06d}.pt")
                torch.save(codes_batch_cpu, temp_file)
                del codes_batch_cpu  # Free memory
                
                # Save position_weights if computed
                if position_weights_batch is not None:
                    temp_weights_file = os.path.join(temp_dir, f"position_weights_aug{num_augmentations}_{aug_idx:03d}_batch_{batch_idx:06d}.pt")
                    torch.save(position_weights_batch, temp_weights_file)
                    del position_weights_batch  # Free memory
            
            batch_idx += 1
            # Free GPU memory after each batch
            del batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Merge codes from all batches (incremental to avoid OOM); only missing files
        print(f">> merging batch files to final codes files...")
        merge_batch_size = 10  # Batches per merge chunk
        
        for aug_idx in indices_to_generate:
            # All batch files for this aug version
            batch_files = sorted([
                os.path.join(temp_dir, f)
                for f in os.listdir(temp_dir)
                if f.startswith(f"codes_{aug_idx:03d}_batch_") and f.endswith(".pt")
            ])
            
            if not batch_files:
                continue
            
            codes_file_path = os.path.join(config["dirname"], f"codes_aug{num_augmentations}_{aug_idx:03d}.pt")
            
            # Merge in chunks of merge_batch_size
            merged_codes_list = []
            for i in range(0, len(batch_files), merge_batch_size):
                batch_chunk = batch_files[i:i + merge_batch_size]
                
                # Load current chunk batches
                chunk_codes = []
                for batch_file in batch_chunk:
                    codes = torch.load(batch_file, weights_only=False)
                    chunk_codes.append(codes)
                    os.remove(batch_file)  # Remove temp file
                
                # Merge current chunk
                merged_chunk = torch.cat(chunk_codes, dim=0)
                merged_codes_list.append(merged_chunk)
                del chunk_codes, merged_chunk
                
                # If too many chunks, merge some first
                if len(merged_codes_list) >= 5:
                    temp_merged = torch.cat(merged_codes_list, dim=0)
                    merged_codes_list = [temp_merged]
                    del temp_merged
            
            # Final merge and save
            if merged_codes_list:
                final_codes = torch.cat(merged_codes_list, dim=0)
                torch.save(final_codes, codes_file_path)
                print(f"   - saved codes_aug{num_augmentations}_{aug_idx:03d}.pt: shape {final_codes.shape}")
                del final_codes, merged_codes_list
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Merge position_weights if present
            if compute_position_weights_flag:
                weights_batch_files = sorted([
                    os.path.join(temp_dir, f)
                    for f in os.listdir(temp_dir)
                    if f.startswith(f"position_weights_aug{num_augmentations}_{aug_idx:03d}_batch_") and f.endswith(".pt")
                ])
                
                if weights_batch_files:
                    weights_file_path = os.path.join(config["dirname"], f"position_weights_aug{num_augmentations}_{aug_idx:03d}.pt")
                    
                    # Merge position_weights in chunks
                    merged_weights_list = []
                    for i in range(0, len(weights_batch_files), merge_batch_size):
                        weights_chunk = weights_batch_files[i:i + merge_batch_size]
                        
                        chunk_weights = []
                        for weights_file in weights_chunk:
                            weights = torch.load(weights_file, weights_only=False)
                            chunk_weights.append(weights)
                            os.remove(weights_file)  # Remove temp file
                        
                        merged_chunk = torch.cat(chunk_weights, dim=0)
                        merged_weights_list.append(merged_chunk)
                        del chunk_weights, merged_chunk
                        
                        if len(merged_weights_list) >= 5:
                            temp_merged = torch.cat(merged_weights_list, dim=0)
                            merged_weights_list = [temp_merged]
                            del temp_merged
                    
                    # Final merge and save
                    if merged_weights_list:
                        final_weights = torch.cat(merged_weights_list, dim=0)
                        torch.save(final_weights, weights_file_path)
                        print(f"   - saved position_weights_aug{num_augmentations}_{aug_idx:03d}.pt: shape {final_weights.shape}")
                        del final_weights, merged_weights_list
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
        
        # Remove temp dir
        try:
            os.rmdir(temp_dir)
        except OSError:
            pass  # Dir may be non-empty, ignore

        elapsed_time = time.time() - t0
        hours, rem = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(rem, 60)
        print(f">> code inference completed in: {int(hours):0>2}h:{int(minutes):0>2}m:{seconds:05.2f}s")


if __name__ == "__main__":
    main()