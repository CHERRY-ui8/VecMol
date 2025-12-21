import sys
import os
import omegaconf
import torch
from tqdm import tqdm
import time
import hydra
import math
import random
# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 设置torch._dynamo配置以解决编译兼容性问题
import torch._dynamo
torch._dynamo.config.suppress_errors = True

from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_batch

from funcmol.utils.utils_nf import load_neural_field
from funcmol.dataset.dataset_field import create_field_loaders, create_gnf_converter
from funcmol.utils.utils_fm import compute_position_weights
from funcmol.models.encoder import create_grid_coords


def _random_rot_matrix(device=None) -> torch.Tensor:
    """生成随机旋转矩阵（绕x, y, z轴）
    
    Returns:
        torch.Tensor: 旋转矩阵 (3x3)
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
    """对batch中的分子坐标应用随机旋转
    
    Args:
        batch: torch_geometric Batch对象
        device: 设备
        
    Returns:
        增强后的batch
    """
    # 创建新的batch副本
    augmented_batch = batch.clone()
    
    # 获取坐标
    coords = batch.pos  # [total_atoms, 3]
    
    # 计算每个分子的中心
    
    coords_dense, mask = to_dense_batch(coords, batch.batch, fill_value=0.0)
    # coords_dense: [B, max_atoms, 3]
    
    # 对每个分子应用旋转
    batch_size = coords_dense.shape[0]
    
    for b in range(batch_size):
        # 获取当前分子的有效坐标
        valid_mask = mask[b]  # [max_atoms]
        mol_coords = coords_dense[b][valid_mask]  # [n_atoms, 3]
        
        if mol_coords.shape[0] > 0:
            # 计算中心
            center = mol_coords.mean(dim=0, keepdim=True)  # [1, 3]
            
            # 生成旋转矩阵
            rot_matrix = _random_rot_matrix(device=device)
            
            # 应用旋转
            mol_coords_centered = mol_coords - center
            mol_coords_rotated = mol_coords_centered @ rot_matrix.T
            mol_coords_rotated = mol_coords_rotated + center
            
            # 更新coords_dense
            coords_dense[b][valid_mask] = mol_coords_rotated
    
    # 将dense格式转换回flat格式
    augmented_coords = []
    for b in range(batch_size):
        valid_mask = mask[b]
        augmented_coords.append(coords_dense[b][valid_mask])
    augmented_batch.pos = torch.cat(augmented_coords, dim=0)
    
    return augmented_batch


def apply_translation_to_batch(batch, anchor_spacing, device):
    """对batch中的分子坐标应用随机平移
    
    Args:
        batch: torch_geometric Batch对象
        anchor_spacing: 锚点间距（单位：埃）
        device: 设备
        
    Returns:
        增强后的batch
    """
    # 创建新的batch副本
    augmented_batch = batch.clone()
    
    # 获取坐标
    coords = batch.pos  # [total_atoms, 3]
    
    # 计算平移距离：1/2个anchor_spacing
    translation_distance = anchor_spacing / 2.0
    
    # 对每个分子应用不同的平移
    from torch_geometric.utils import to_dense_batch
    coords_dense, mask = to_dense_batch(coords, batch.batch, fill_value=0.0)
    # coords_dense: [B, max_atoms, 3]
    
    batch_size = coords_dense.shape[0]
    
    for b in range(batch_size):
        # 生成随机平移向量（在[-translation_distance, translation_distance]范围内）
        translation = (torch.rand(3, device=device) * 2 - 1) * translation_distance  # [3]
        
        # 应用平移
        valid_mask = mask[b]  # [max_atoms]
        coords_dense[b][valid_mask] = coords_dense[b][valid_mask] + translation.unsqueeze(0)
    
    # 将dense格式转换回flat格式
    augmented_coords = []
    for b in range(batch_size):
        valid_mask = mask[b]
        augmented_coords.append(coords_dense[b][valid_mask])
    augmented_batch.pos = torch.cat(augmented_coords, dim=0)
    
    return augmented_batch


@hydra.main(config_path="configs", config_name="infer_codes", version_base=None)
def main(config):
    # 验证并处理 nf_pretrained_path
    nf_pretrained_path = config.get("nf_pretrained_path")
    if not nf_pretrained_path:
        raise ValueError("必须指定 nf_pretrained_path 参数来指定Lightning checkpoint路径")
    
    if not nf_pretrained_path.endswith('.ckpt') or not os.path.exists(nf_pretrained_path):
        raise ValueError(f"指定的checkpoint文件不存在或格式不正确: {nf_pretrained_path}")
    
    # 自动计算 dirname，基于 nf_pretrained_path 的目录
    checkpoint_dir = os.path.dirname(nf_pretrained_path)
    
    # 将 config 转换为普通字典以避免结构化配置限制
    if isinstance(config, omegaconf.dictconfig.DictConfig):
        config = omegaconf.OmegaConf.to_container(config, resolve=True)
    
    # 保存yaml配置中的split，确保它不会被checkpoint配置覆盖
    yaml_split = config.get("split", "train")
    yaml_use_data_augmentation = config.get("use_data_augmentation", False)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 设置随机种子
    torch.manual_seed(config.get("seed", 1234))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.get("seed", 1234))
    
    # 加载Lightning checkpoint
    # 使用 weights_only=False 以支持包含 omegaconf.DictConfig 的 checkpoint
    checkpoint = torch.load(nf_pretrained_path, map_location='cpu', weights_only=False)
    
    # 从Lightning checkpoint中提取配置
    if 'hyper_parameters' in checkpoint:
        config_model = checkpoint['hyper_parameters']
    else:
        config_model = checkpoint.get('config', {})
    
    # 将 config_model 转换为普通字典以避免结构化配置的限制
    if isinstance(config_model, omegaconf.dictconfig.DictConfig):
        config_model = omegaconf.OmegaConf.to_container(config_model, resolve=True)
    
    for key in config.keys():
        if key in config_model and \
            isinstance(config_model[key], dict) and isinstance(config[key], dict):
            config_model[key].update(config[key])
        else:
            config_model[key] = config[key]
    config = config_model  # update config with checkpoint config
    
    # 确保yaml配置中的split和use_data_augmentation优先（覆盖checkpoint中的配置）
    config["split"] = yaml_split
    config["use_data_augmentation"] = yaml_use_data_augmentation
    
    # 根据是否使用数据增强来决定保存目录（在合并配置后确定）
    use_data_augmentation_config = config.get("use_data_augmentation", False)
    split = config.get("split", "train")
    
    if use_data_augmentation_config and split == "train":
        # 使用数据增强时，保存到 codes 目录
        # dirname = os.path.join(checkpoint_dir, "codes", split)
        dirname = os.path.join(checkpoint_dir, "codes_no_shuffle", split)
    else:
        # 不使用数据增强时，保存到 code_no_aug 目录
        dirname = os.path.join(checkpoint_dir, "code_no_aug", split)
    
    config["dirname"] = dirname
    enc, _ = load_neural_field(checkpoint, config)

    # 创建GNFConverter实例用于数据加载
    gnf_converter = create_gnf_converter(config)
    
    # data loader
    loader = create_field_loaders(config, gnf_converter, split=config["split"])
    
    # 强制禁用shuffle以保持数据顺序（确保codes索引与原始数据索引对应） 
    loader = DataLoader(
        loader.dataset,
        batch_size=min(config["dset"]["batch_size"], len(loader.dataset)),
        num_workers=config["dset"]["num_workers"],
        shuffle=False,  # 强制禁用shuffle，保持顺序
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

    # 获取数据增强配置
    # 数据增强只应该在训练集上使用，验证集和测试集应该使用原始数据
    # 注意：split 和 use_data_augmentation_config 已在上面定义（第174-175行）
    use_data_augmentation = use_data_augmentation_config and (split == "train")
    num_augmentations = config.get("num_augmentations", 1)  # 每个分子生成多少个增强版本（包括原始版本）
    apply_rotation = config.get("data_augmentation", {}).get("apply_rotation", True)
    apply_translation = config.get("data_augmentation", {}).get("apply_translation", True)
    anchor_spacing = config.get("dset", {}).get("anchor_spacing", 1.5)
    
    # 如果禁用数据增强，强制设置为1（只生成原始版本）
    if not use_data_augmentation_config:
        num_augmentations = 1
    # 如果不是训练集，强制禁用数据增强
    elif split != "train" and use_data_augmentation_config:
        num_augmentations = 1

    # 获取position_weight配置（如果启用）- 需要在检查文件存在之前定义
    position_weight_config = config.get("position_weight", {})
    compute_position_weights_flag = position_weight_config.get("enabled", False)
    radius = position_weight_config.get("radius", 3.0)
    weight_alpha = position_weight_config.get("alpha", 0.5)
    grid_size = config.get("dset", {}).get("grid_size", 9)

    # check if codes already exist (检查所有增强版本的文件)
    all_codes_exist = True
    all_weights_exist = True
    for aug_idx in range(num_augmentations):
        codes_file_path = os.path.join(config["dirname"], f"codes_{aug_idx:03d}.pt")
        if not os.path.exists(codes_file_path):
            all_codes_exist = False
            break
        
        # 如果启用了position_weight计算，也检查weights文件
        if compute_position_weights_flag:
            weights_file_path = os.path.join(config["dirname"], f"position_weights_{aug_idx:03d}.pt")
            if not os.path.exists(weights_file_path):
                all_weights_exist = False
    
    if all_codes_exist:
        if compute_position_weights_flag and not all_weights_exist:
            print(f">> codes files exist but position_weights files are missing")
            print(f">> will recompute codes to generate position_weights")
        else:
            print(f">> all codes files already exist in {config['dirname']}")
            if compute_position_weights_flag:
                print(f">> all position_weights files also exist")
            print(">> skipping code inference")
            return
    
    if use_data_augmentation:
        print(">> Data augmentation enabled:")
        print(f"   - num_augmentations: {num_augmentations}")
        print(f"   - apply_rotation: {apply_rotation}")
        print(f"   - apply_translation: {apply_translation}")
        print(f"   - anchor_spacing: {anchor_spacing}")
    
    # start eval
    print(f">> start code inference in {config['split']} split")
    enc.eval()

    # 创建临时目录存储每个batch的codes和position_weights
    temp_dir = os.path.join(config["dirname"], "temp_batches")
    os.makedirs(temp_dir, exist_ok=True)
    
    # position_weight配置已在上面定义，这里直接使用
    if compute_position_weights_flag:
        print(">> Position weight computation enabled:")
        print(f"   - radius: {radius}")
        print(f"   - alpha: {weight_alpha}")
        print(f"   - grid_size: {grid_size}")
        # 创建grid坐标（只需要创建一次）
        grid_coords = create_grid_coords(1, grid_size, device=device, anchor_spacing=anchor_spacing)
        grid_coords = grid_coords.squeeze(0).cpu()  # [n_grid, 3], 移到CPU
    else:
        grid_coords = None
        print(">> Position weight computation disabled")

    with torch.no_grad():
        t0 = time.time()
        batch_idx = 0
        for batch in tqdm(loader):
            batch = batch.to(device)
            
            # 对每个batch，生成多个增强版本并infer codes
            for aug_idx in range(num_augmentations):
                if aug_idx == 0:
                    # 第一个版本：原始版本（不增强）
                    augmented_batch = batch
                else:
                    # 后续版本：应用数据增强
                    augmented_batch = batch.clone()
                    
                    # 应用旋转
                    if apply_rotation:
                        augmented_batch = apply_rotation_to_batch(augmented_batch, device)
                    
                    # 应用平移
                    if apply_translation:
                        augmented_batch = apply_translation_to_batch(augmented_batch, anchor_spacing, device)
                
                # Infer codes
                codes_batch = enc(augmented_batch)
                codes_batch_cpu = codes_batch.detach().cpu()
                
                # 计算position_weights（如果启用）
                position_weights_batch = None
                if compute_position_weights_flag:
                    # 获取原子坐标（使用增强后的batch）
                    atom_coords = augmented_batch.pos.cpu()  # [N_total_atoms, 3]
                    batch_idx_atoms = augmented_batch.batch.cpu()  # [N_total_atoms]
                    
                    # 计算position weights
                    position_weights_batch = compute_position_weights(
                        atom_coords=atom_coords,
                        grid_coords=grid_coords,
                        batch_idx=batch_idx_atoms,
                        radius=radius,
                        weight_alpha=weight_alpha,
                        device=torch.device('cpu')  # 在CPU上计算以节省GPU内存
                    )  # [B, n_grid]
                
                # 立即保存到临时文件
                temp_file = os.path.join(temp_dir, f"codes_{aug_idx:03d}_batch_{batch_idx:06d}.pt")
                torch.save(codes_batch_cpu, temp_file)
                del codes_batch_cpu  # 立即释放内存
                
                # 保存position_weights（如果计算了）
                if position_weights_batch is not None:
                    temp_weights_file = os.path.join(temp_dir, f"position_weights_{aug_idx:03d}_batch_{batch_idx:06d}.pt")
                    torch.save(position_weights_batch, temp_weights_file)
                    del position_weights_batch  # 立即释放内存
            
            batch_idx += 1
            # 每处理完一个batch就释放GPU内存
            del batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # 合并所有batch的codes文件（使用增量合并避免内存溢出）
        print(f">> merging batch files to final codes files...")
        merge_batch_size = 10  # 每次合并的batch数量，避免一次性加载太多
        
        for aug_idx in range(num_augmentations):
            # 获取该增强版本的所有batch文件
            batch_files = sorted([
                os.path.join(temp_dir, f)
                for f in os.listdir(temp_dir)
                if f.startswith(f"codes_{aug_idx:03d}_batch_") and f.endswith(".pt")
            ])
            
            if not batch_files:
                continue
            
            codes_file_path = os.path.join(config["dirname"], f"codes_{aug_idx:03d}.pt")
            
            # 分块合并：每次合并merge_batch_size个batch
            merged_codes_list = []
            for i in range(0, len(batch_files), merge_batch_size):
                batch_chunk = batch_files[i:i + merge_batch_size]
                
                # 加载当前chunk的所有batch
                chunk_codes = []
                for batch_file in batch_chunk:
                    codes = torch.load(batch_file, weights_only=False)
                    chunk_codes.append(codes)
                    os.remove(batch_file)  # 立即删除临时文件
                
                # 合并当前chunk
                merged_chunk = torch.cat(chunk_codes, dim=0)
                merged_codes_list.append(merged_chunk)
                del chunk_codes, merged_chunk
                
                # 如果累积的chunks太多，先合并一部分
                if len(merged_codes_list) >= 5:
                    temp_merged = torch.cat(merged_codes_list, dim=0)
                    merged_codes_list = [temp_merged]
                    del temp_merged
            
            # 最终合并并保存
            if merged_codes_list:
                final_codes = torch.cat(merged_codes_list, dim=0)
                torch.save(final_codes, codes_file_path)
                print(f"   - saved codes_{aug_idx:03d}.pt: shape {final_codes.shape}")
                del final_codes, merged_codes_list
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # 合并position_weights文件（如果存在）
            if compute_position_weights_flag:
                weights_batch_files = sorted([
                    os.path.join(temp_dir, f)
                    for f in os.listdir(temp_dir)
                    if f.startswith(f"position_weights_{aug_idx:03d}_batch_") and f.endswith(".pt")
                ])
                
                if weights_batch_files:
                    weights_file_path = os.path.join(config["dirname"], f"position_weights_{aug_idx:03d}.pt")
                    
                    # 分块合并position_weights
                    merged_weights_list = []
                    for i in range(0, len(weights_batch_files), merge_batch_size):
                        weights_chunk = weights_batch_files[i:i + merge_batch_size]
                        
                        chunk_weights = []
                        for weights_file in weights_chunk:
                            weights = torch.load(weights_file, weights_only=False)
                            chunk_weights.append(weights)
                            os.remove(weights_file)  # 立即删除临时文件
                        
                        merged_chunk = torch.cat(chunk_weights, dim=0)
                        merged_weights_list.append(merged_chunk)
                        del chunk_weights, merged_chunk
                        
                        if len(merged_weights_list) >= 5:
                            temp_merged = torch.cat(merged_weights_list, dim=0)
                            merged_weights_list = [temp_merged]
                            del temp_merged
                    
                    # 最终合并并保存
                    if merged_weights_list:
                        final_weights = torch.cat(merged_weights_list, dim=0)
                        torch.save(final_weights, weights_file_path)
                        print(f"   - saved position_weights_{aug_idx:03d}.pt: shape {final_weights.shape}")
                        del final_weights, merged_weights_list
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
        
        # 删除临时目录
        try:
            os.rmdir(temp_dir)
        except OSError:
            pass  # 目录可能不为空，忽略错误

        elapsed_time = time.time() - t0
        hours, rem = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(rem, 60)
        print(f">> code inference completed in: {int(hours):0>2}h:{int(minutes):0>2}m:{seconds:05.2f}s")


if __name__ == "__main__":
    main()