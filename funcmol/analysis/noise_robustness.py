#!/usr/bin/env python3
"""
Neural Field Codes空间噪声鲁棒性分析脚本

分析neural field在codes空间上对于各向同性高斯噪声的鲁棒性。
从验证集加载codes，添加不同水平的噪声，解码为分子，然后计算各种指标。
"""

import os
import random
import numpy as np
import torch
import pandas as pd
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf

# 设置项目根目录到 Python 路径
import sys
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 禁用RDKit警告
from rdkit import Chem, RDLogger
from rdkit.Geometry.rdGeometry import Point3D
RDLogger.DisableLog('rdApp.*')

# 导入项目模块
from funcmol.dataset.dataset_code import CodeDataset, create_code_loaders
from funcmol.utils.utils_nf import load_neural_field
from funcmol.dataset.dataset_field import create_gnf_converter, create_field_loaders, FieldDataset
from funcmol.utils.utils_base import add_bonds_with_openbabel
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader as PyGDataLoader
from funcmol.evaluate import (
    check_stable_mol,
    check_rdkit_validity,
    compute_valency_distribution,
    compute_atom_type_distribution,
    compute_bond_type_distribution,
    compute_bond_lengths,
    compute_bond_angles,
    compute_total_variation,
    compute_wasserstein1_distance,
    load_test_set_molecules,
    compute_test_set_distributions,
    extract_largest_connected_component
)


def coords_types_to_rdkit_mol(coords: np.ndarray, types: np.ndarray, elements: List[str]) -> Optional[Chem.Mol]:
    """
    将坐标和类型转换为RDKit Mol对象
    
    Args:
        coords: [N, 3] 原子坐标
        types: [N,] 原子类型索引
        elements: 元素符号列表，例如 ["C", "H", "O", "N", "F"]
        
    Returns:
        RDKit Mol对象，如果转换失败返回None
    """
    # 过滤掉无效的原子（types == -1）
    valid_mask = types != -1
    if not valid_mask.any():
        return None
    
    coords = coords[valid_mask]
    types = types[valid_mask]
    
    # 使用OpenBabel添加键并转换为SDF
    sdf_content = add_bonds_with_openbabel(
        coords,
        types,
        elements,
        add_hydrogens=True
    )
    
    if not sdf_content:
        return None
    
    # 从SDF内容创建RDKit分子
    try:
        # 使用MolFromMolBlock从字符串创建分子
        mol = Chem.MolFromMolBlock(sdf_content, sanitize=False)
        if mol is None:
            return None
        
        # 只保留最大连通分支
        if mol.GetNumBonds() > 0:
            mol = extract_largest_connected_component(mol)
        
        return mol
    except Exception as e:
        print(f"警告: 从SDF创建RDKit分子失败: {e}")
        return None


def add_gaussian_noise(codes: torch.Tensor, noise_level: float) -> torch.Tensor:
    """
    对codes添加各向同性高斯噪声
    
    Args:
        codes: [B, grid_size³, code_dim] codes张量
        noise_level: 噪声水平（标准差）
        
    Returns:
        添加噪声后的codes
    """
    if noise_level == 0.0:
        return codes
    
    noise = noise_level * torch.randn_like(codes)
    noisy_codes = codes + noise
    return noisy_codes


@hydra.main(config_path="../configs", config_name="noise_robustness_qm9", version_base=None)
def main(config: DictConfig) -> None:
    """
    主函数：分析neural field codes对噪声的鲁棒性
    
    Args:
        config: Hydra配置对象
    """
    # 设置随机种子
    seed = config.get('seed', 1234)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # 转换配置为字典
    config_dict = OmegaConf.to_container(config, resolve=True)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 1. 加载neural field模型
    print("\n" + "="*80)
    print("加载Neural Field模型")
    print("="*80)
    nf_checkpoint_path = config.get("nf_pretrained_path")
    if not nf_checkpoint_path:
        raise ValueError("配置文件中必须提供 nf_pretrained_path")
    
    print(f"加载checkpoint: {nf_checkpoint_path}")
    checkpoint = torch.load(nf_checkpoint_path, map_location='cpu', weights_only=False)
    nf_config = checkpoint.get("hyper_parameters", {})
    
    # 创建checkpoint字典
    nf_checkpoint = {
        "config": nf_config,
        "enc_state_dict": checkpoint["enc_state_dict"],
        "dec_state_dict": checkpoint["dec_state_dict"]
    }
    
    # 加载encoder和decoder
    encoder, decoder = load_neural_field(nf_checkpoint, None)
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    encoder.eval()
    decoder.eval()
    print("模型加载完成")
    
    # 设置输出目录（提前定义，供后续使用）
    output_dir = Path(config.get("output_dir", "exps/analysis/noise_robustness"))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取数据集类型（提前定义，供后续使用）
    dataset_type = config.get("dataset_type", "qm9")
    
    # 2. 创建GNFConverter
    print("\n" + "="*80)
    print("创建GNFConverter")
    print("="*80)
    converter = create_gnf_converter(config_dict)
    print("GNFConverter创建完成")
    
    # 3. 加载或计算codes
    print("\n" + "="*80)
    compute_codes_on_the_fly = config.get("compute_codes_on_the_fly", False)
    save_computed_codes = config.get("save_computed_codes", False)
    codes_save_dir = config.get("codes_save_dir", None)
    
    if compute_codes_on_the_fly:
        print("现场计算Codes（从test数据集）")
        print("="*80)
        
        # 从test数据集加载分子并计算codes
        num_codes_to_load = config.get("num_codes", 4000)
        
        # 创建FieldDataset来加载test数据集
        field_dataset = FieldDataset(
            gnf_converter=converter,
            dset_name=config_dict.get("dset", {}).get("dset_name", "qm9"),
            data_dir=config_dict.get("dset", {}).get("data_dir", "dataset/data"),
            elements=config_dict.get("dset", {}).get("elements", None),
            split="test",  # 使用test数据集
            n_points=config_dict.get("dset", {}).get("n_points", 4000),
            rotate=False,  # test集不旋转
            resolution=config_dict.get("dset", {}).get("resolution", 0.25),
            grid_dim=config_dict.get("dset", {}).get("grid_dim", 32),
            sample_full_grid=False,
            targeted_sampling_ratio=0,  # test集不使用targeted sampling
            atom_distance_threshold=config_dict.get("dset", {}).get("atom_distance_threshold", 0.5),
        )
        
        total_samples = len(field_dataset)
        if total_samples < num_codes_to_load:
            print(f"警告: test集只有 {total_samples} 个样本，将使用全部")
            num_codes_to_load = total_samples
        
        # 随机选择样本（使用固定种子确保可重复）
        indices = list(range(total_samples))
        random.Random(seed).shuffle(indices)
        selected_indices = indices[:num_codes_to_load]
        
        print(f"从 {total_samples} 个test样本中选择 {num_codes_to_load} 个")
        
        # 创建DataLoader（批次大小为1，因为我们逐个处理）
        selected_dataset = torch.utils.data.Subset(field_dataset, selected_indices)
        field_loader = PyGDataLoader(selected_dataset, batch_size=1, shuffle=False, num_workers=0)
        
        # 使用encoder计算codes
        all_codes = []
        encoder.eval()
        with torch.no_grad():
            for batch_idx, sample in enumerate(tqdm(field_loader, desc="计算codes")):
                # sample已经是torch_geometric.data.Batch对象（DataLoader自动转换）
                # 移动到设备
                sample = sample.to(device)
                
                # 使用encoder计算codes
                codes = encoder(sample)  # [1, grid_size³, code_dim]
                all_codes.append(codes[0].cpu())  # 保存到CPU
        
        all_codes = torch.stack(all_codes).to(device)  # [num_codes, grid_size³, code_dim]
        print(f"成功计算 {len(all_codes)} 个codes，形状: {all_codes.shape}")
        
        # 可选保存codes
        if save_computed_codes:
            if codes_save_dir is None:
                codes_save_dir = output_dir / "computed_codes"
            else:
                codes_save_dir = Path(codes_save_dir)
            
            codes_save_dir.mkdir(parents=True, exist_ok=True)
            
            # 保存为numpy格式（更小）
            codes_np = all_codes.cpu().numpy()
            codes_save_path = codes_save_dir / f"noise_robustness_codes_{dataset_type}.npz"
            np.savez_compressed(codes_save_path, codes=codes_np, indices=selected_indices)
            print(f"Codes已保存到: {codes_save_path} (形状: {codes_np.shape}, 大小: {codes_save_path.stat().st_size / 1024 / 1024:.2f} MB)")
    
    else:
        print("从文件加载Codes")
        print("="*80)
        codes_dir = config.get("codes_dir")
        if not codes_dir:
            raise ValueError("配置文件中必须提供 codes_dir（或设置 compute_codes_on_the_fly=True）")
        
        # 创建CodeDataset
        code_dataset = CodeDataset(
            dset_name=config_dict.get("dset", {}).get("dset_name", "qm9"),
            split="test",
            codes_dir=codes_dir,
            num_augmentations=None
        )
        
        # 加载至少4000个codes
        num_codes_to_load = config.get("num_codes", 4000)
        total_codes = len(code_dataset)
        if total_codes < num_codes_to_load:
            print(f"警告: 验证集只有 {total_codes} 个codes，将使用全部")
            num_codes_to_load = total_codes
        
        # 随机选择codes（使用固定种子确保可重复）
        indices = list(range(total_codes))
        random.Random(seed).shuffle(indices)
        selected_indices = indices[:num_codes_to_load]
        
        print(f"从 {total_codes} 个codes中选择 {num_codes_to_load} 个")
        
        # 加载选中的codes
        all_codes = []
        for idx in tqdm(selected_indices, desc="加载codes"):
            code = code_dataset[idx]  # [grid_size³, code_dim]
            all_codes.append(code)
        
        all_codes = torch.stack(all_codes).to(device)  # [num_codes, grid_size³, code_dim]
        print(f"成功加载 {len(all_codes)} 个codes，形状: {all_codes.shape}")
    
    # 4. 定义噪声水平
    noise_levels = config.get("noise_levels", [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0])
    print(f"\n噪声水平: {noise_levels}")
    
    # 5. 加载测试集分布（用于计算TV和W1距离）
    print("\n" + "="*80)
    print("加载测试集分布")
    print("="*80)
    test_data_dir = config.get("test_data_dir")
    
    test_molecules = []
    test_distributions = None
    if test_data_dir:
        test_molecules = load_test_set_molecules(test_data_dir, limit=None)
        if test_molecules:
            test_distributions = compute_test_set_distributions(test_molecules, ds_type=dataset_type)
            print(f"成功加载 {len(test_molecules)} 个测试集分子")
        else:
            print("警告: 未能加载测试集分子，将跳过分布比较指标")
    else:
        print("警告: 未提供test_data_dir，将跳过分布比较指标")
    
    # 6. 获取元素列表
    elements = config_dict.get("dset", {}).get("elements", ["C", "H", "O", "N", "F"])
    print(f"元素列表: {elements}")
    
    # 7. 对每个噪声水平进行解码和评估
    print("\n" + "="*80)
    print("开始噪声鲁棒性分析")
    print("="*80)
    
    results = []
    
    for noise_level in noise_levels:
        print(f"\n处理噪声水平: {noise_level}")
        
        # 添加噪声
        noisy_codes = add_gaussian_noise(all_codes, noise_level)
        
        # 存储每个分子的指标
        mol_metrics = {
            'stable_mol': [],
            'stable_atom_pct': [],
            'valid': [],
            'valency_distributions': [],
            'atom_type_distributions': [],
            'bond_type_distributions': [],
            'bond_lengths': [],
            'bond_angles': []
        }
        
        # 获取批次大小配置，默认1（不并行）
        batch_size = config.get('batch_size', 1)
        if batch_size > 1:
            print(f"使用批次大小 {batch_size} 进行分子级别并行处理")
        
        # 对每个code进行解码（支持批次处理）
        successful_decodes = 0
        total_codes = len(noisy_codes)
        total_batches = (total_codes + batch_size - 1) // batch_size
        
        for batch_idx in tqdm(range(total_batches), desc=f"解码 (noise={noise_level})", leave=False):
            # 计算当前批次的实际大小和索引
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, total_codes)
            current_batch_size = end_idx - start_idx
            
            # 批次化的codes
            batch_codes = noisy_codes[start_idx:end_idx]  # [current_batch_size, grid_size³, code_dim]
            
            try:
                # 批次解码为分子
                recon_coords, recon_types = converter.gnf2mol(
                    decoder=decoder,
                    codes=batch_codes,
                    sample_id=start_idx  # 使用起始索引作为sample_id
                )
                
                # 处理批次结果
                for i in range(current_batch_size):
                    code_idx = start_idx + i
                    try:
                        # 转换为numpy
                        recon_coords_single = recon_coords[i].cpu().numpy()  # [N, 3]
                        recon_types_single = recon_types[i].cpu().numpy()  # [N]
                        
                        # 过滤掉填充的原子（值为-1）
                        valid_mask = recon_types_single != -1
                        if valid_mask.any():
                            recon_coords_single = recon_coords_single[valid_mask]
                            recon_types_single = recon_types_single[valid_mask]
                        
                        # 转换为RDKit Mol对象
                        mol = coords_types_to_rdkit_mol(recon_coords_single, recon_types_single, elements)
                        
                        if mol is None:
                            continue
                        
                        successful_decodes += 1
                        
                        # 计算基础指标
                        stable_pct, is_stable = check_stable_mol(mol)
                        is_valid = check_rdkit_validity(mol)
                        
                        mol_metrics['stable_mol'].append(1.0 if is_stable else 0.0)
                        mol_metrics['stable_atom_pct'].append(stable_pct)
                        mol_metrics['valid'].append(1.0 if is_valid else 0.0)
                        
                        # 收集分布数据
                        mol_metrics['valency_distributions'].append(compute_valency_distribution(mol))
                        mol_metrics['atom_type_distributions'].append(compute_atom_type_distribution(mol))
                        mol_metrics['bond_type_distributions'].append(compute_bond_type_distribution(mol))
                        mol_metrics['bond_lengths'].extend(compute_bond_lengths(mol))
                        mol_metrics['bond_angles'].extend(compute_bond_angles(mol))
                    
                    except Exception as e:
                        print(f"警告: 处理第 {code_idx} 个code时出错: {e}")
                        continue
            
            except Exception as e:
                print(f"警告: 批次 {batch_idx} 解码时出错: {e}")
                # 继续处理下一个批次
                continue
        
        print(f"成功解码 {successful_decodes}/{len(noisy_codes)} 个codes")
        
        if successful_decodes == 0:
            print(f"警告: 噪声水平 {noise_level} 下没有成功解码的分子")
            # 添加空结果
            result = {
                'noise_level': noise_level,
                'num_decoded': 0,
                'stable_mol': 0.0,
                'stable_atom': 0.0,
                'valid': 0.0,
                'valency_w1': None,
                'atom_tv': None,
                'bond_tv': None,
                'bond_len_w1': None,
                'bond_ang_w1': None
            }
            results.append(result)
            continue
        
        # 计算聚合指标
        stable_mol_rate = np.mean(mol_metrics['stable_mol'])
        stable_atom_pct = np.mean(mol_metrics['stable_atom_pct'])
        valid_rate = np.mean(mol_metrics['valid'])
        
        # 计算分布比较指标
        valency_w1 = None
        atom_tv = None
        bond_tv = None
        bond_len_w1 = None
        bond_ang_w1 = None
        
        if test_distributions:
            # Valency W1
            all_valencies = []
            for val_dist in mol_metrics['valency_distributions']:
                for valency, count in val_dist.items():
                    all_valencies.extend([valency] * count)
            if all_valencies and test_distributions.get('valencies'):
                valency_w1 = compute_wasserstein1_distance(all_valencies, test_distributions['valencies'])
            
            # Atom TV
            all_atom_types = []
            for atom_dist in mol_metrics['atom_type_distributions']:
                for atom_type, count in atom_dist.items():
                    all_atom_types.extend([atom_type] * count)
            if all_atom_types and test_distributions.get('atom_types'):
                gen_atom_counter = Counter(all_atom_types)
                test_atom_counter = Counter(test_distributions['atom_types'])
                atom_tv = compute_total_variation(dict(gen_atom_counter), dict(test_atom_counter))
            
            # Bond TV
            all_bond_types = []
            for bond_dist in mol_metrics['bond_type_distributions']:
                for bond_type, count in bond_dist.items():
                    all_bond_types.extend([bond_type] * count)
            if all_bond_types and test_distributions.get('bond_types'):
                gen_bond_counter = Counter(all_bond_types)
                test_bond_counter = Counter(test_distributions['bond_types'])
                bond_tv = compute_total_variation(dict(gen_bond_counter), dict(test_bond_counter))
            
            # Bond length W1
            if mol_metrics['bond_lengths'] and test_distributions.get('bond_lengths'):
                bond_len_w1 = compute_wasserstein1_distance(
                    mol_metrics['bond_lengths'],
                    test_distributions['bond_lengths']
                )
            
            # Bond angle W1
            if mol_metrics['bond_angles'] and test_distributions.get('bond_angles'):
                bond_ang_w1 = compute_wasserstein1_distance(
                    mol_metrics['bond_angles'],
                    test_distributions['bond_angles']
                )
        
        # 保存结果
        result = {
            'noise_level': noise_level,
            'num_decoded': successful_decodes,
            'stable_mol': stable_mol_rate,
            'stable_atom': stable_atom_pct,
            'valid': valid_rate,
            'valency_w1': valency_w1,
            'atom_tv': atom_tv,
            'bond_tv': bond_tv,
            'bond_len_w1': bond_len_w1,
            'bond_ang_w1': bond_ang_w1
        }
        results.append(result)
        
        print(f"  稳定分子率: {stable_mol_rate:.4f}")
        print(f"  稳定原子百分比: {stable_atom_pct:.4f}")
        print(f"  有效性: {valid_rate:.4f}")
        if valency_w1 is not None:
            print(f"  Valency W1: {valency_w1:.4f}")
        if atom_tv is not None:
            print(f"  Atom TV: {atom_tv:.4f}")
        if bond_tv is not None:
            print(f"  Bond TV: {bond_tv:.4f}")
        if bond_len_w1 is not None:
            print(f"  Bond length W1: {bond_len_w1:.4f}")
        if bond_ang_w1 is not None:
            print(f"  Bond angle W1: {bond_ang_w1:.4f}")
    
    # 8. 保存结果
    print("\n" + "="*80)
    print("保存结果")
    print("="*80)
    
    # 保存CSV
    df = pd.DataFrame(results)
    csv_path = output_dir / f"noise_robustness_{dataset_type}.csv"
    df.to_csv(csv_path, index=False)
    print(f"结果已保存到: {csv_path}")
    
    print("\n" + "="*80)
    print("分析完成！")
    print("="*80)


if __name__ == "__main__":
    main()

