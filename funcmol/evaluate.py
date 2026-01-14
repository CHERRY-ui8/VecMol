#!/usr/bin/env python3
"""
SDF分子评估脚本
从SDF格式文件读取分子，计算化学合理性、多样性和分布一致性指标

- 本脚本默认评估的是补H后的分子（即包含H原子的完整分子）
- 如果post_process_bonds.py中add_hydrogens=True，生成的*_obabel.sdf文件包含H原子
- 本脚本优先加载*_obabel.sdf文件，因此评估的是补H后的分子
- 键长分布、键角分布、原子种类分布等指标都会受到H原子的影响
"""

# 配置参数：实验目录
exp_dir = "/datapool/data2/home/pxg/data/hyc/funcmol-main-neuralfield/exps/funcmol/fm_qm9/20260105/samples/20260105_version_1_last_withbonds_addH"

# 数据集类型：'qm9' 或 'drugs'
dataset_type = 'qm9'  # 设置为 'drugs' 以启用额外的 drugs 指标

# 测试集数据路径（用于分布比较）
test_data_dir = "/datapool/data2/home/pxg/data/hyc/funcmol-main-neuralfield/funcmol/dataset/data/qm9"  # QM9测试集数据目录
# test_data_dir = "/datapool/data2/home/pxg/data/hyc/funcmol-main-neuralfield/funcmol/dataset/data/drugs"  # Drugs测试集数据目录
test_data_limit = None  # 限制测试集样本数量（用于加速计算，设为None则不限制）

import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional
import warnings
import pickle
import torch
import lmdb
from tqdm import tqdm
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Geometry.rdGeometry import Point3D
# 禁用RDKit警告
RDLogger.DisableLog('rdApp.*')
from scipy.stats import wasserstein_distance

# 导入 drugs 指标所需的 RDKit 模块
try:
    from rdkit.Chem import QED, Crippen
    from rdkit.Chem import rdMolDescriptors
    from rdkit.Chem import AllChem
    DRUGS_METRICS_AVAILABLE = True
except ImportError as e:
    print(f"警告: 无法导入某些 RDKit 模块，drugs 指标可能不可用: {e}")
    DRUGS_METRICS_AVAILABLE = False
    # 提供占位符以避免错误
    rdMolDescriptors = None

# SA 分数计算函数
# 注意：标准的 SA 分数需要额外的包（如 rdkit-pypi 的完整版本）
# 这里提供一个基于分子复杂度的简化实现
def compute_sa_score(mol: Chem.Mol) -> Optional[float]:
    """
    计算合成可及性分数（简化版本）
    基于 Ertl & Schuffenhauer 的 SA 分数算法简化实现
    
    Args:
        mol: RDKit分子对象
        
    Returns:
        SA 分数（1-10之间，1表示最容易合成，10表示最难合成），如果计算失败返回None
    """
    if mol is None or not DRUGS_METRICS_AVAILABLE or rdMolDescriptors is None:
        return None
    
    try:
        # 需要先 sanitize 分子
        mol_copy = Chem.Mol(mol)
        try:
            Chem.SanitizeMol(mol_copy)
        except:
            return None
        
        # 基于分子复杂度的简化 SA 分数计算
        num_atoms = mol_copy.GetNumAtoms()
        num_rings = rdMolDescriptors.CalcNumRings(mol_copy)
        num_rotatable_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol_copy)
        num_heteroatoms = sum(1 for atom in mol_copy.GetAtoms() if atom.GetAtomicNum() not in [1, 6])
        
        # 简化的 SA 分数计算（基于复杂度）
        # 更复杂的分子通常更难合成
        # 这个公式是基于 Ertl & Schuffenhauer 的简化版本
        complexity = (num_atoms * 0.1 + num_rings * 2.0 + num_rotatable_bonds * 0.5 + num_heteroatoms * 1.0)
        sa_score = max(1.0, min(10.0, 1.0 + complexity / 10.0))  # 归一化到 1-10
        return sa_score
    except Exception:
        return None

warnings.filterwarnings('ignore')


# QM9原子价规则（原子序数 -> 最大价态）
ATOM_VALENCY = {
    1: 1,   # H
    6: 4,   # C
    7: 3,   # N
    8: 2,   # O
    9: 1,   # F
}

# 原子序数到元素符号的映射
ATOM_NUM_TO_SYMBOL = {
    1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F'
}

# 键类型映射
BOND_TYPE_MAP = {
    1: 'single',
    2: 'double',
    3: 'triple',
    4: 'aromatic',
}


def load_sdf_molecules(sdf_path: Path) -> List[Chem.Mol]:
    """
    从SDF文件加载分子
    优先尝试加载带OpenBabel键的版本（*_obabel.sdf），如果不存在则加载原始文件
    
    注意：
    - 如果post_process_bonds.py中add_hydrogens=True，*_obabel.sdf文件包含H原子
    - 本函数加载的分子可能包含H原子（取决于SDF文件内容）
    - 这会影响键长分布、键角分布、原子种类分布等指标
    
    Args:
        sdf_path: SDF文件路径
        
    Returns:
        分子列表（可能包含H原子）
    """
    molecules = []
    
    # 优先尝试加载带OpenBabel键的版本
    obabel_path = sdf_path.parent / f"{sdf_path.stem}_obabel.sdf"
    actual_path = obabel_path if obabel_path.exists() else sdf_path
    
    if not actual_path.exists():
        print(f"警告: 文件不存在 {actual_path}")
        return molecules
    
    try:
        # 使用SDMolSupplier读取SDF文件（支持多分子）
        supplier = Chem.SDMolSupplier(str(actual_path), sanitize=False)
        for mol in supplier:
            if mol is not None:
                molecules.append(mol)
    except Exception as e:
        print(f"读取SDF文件 {actual_path} 时出错: {e}")
    
    return molecules


def get_atom_valency(atom: Chem.Atom) -> int:
    """
    计算原子的实际价态（考虑键类型）
    
    Args:
        atom: RDKit原子对象
        
    Returns:
        原子的总价态
    """
    valency = 0
    for bond in atom.GetBonds():
        bond_type = bond.GetBondType()
        if bond_type == Chem.BondType.SINGLE:
            valency += 1
        elif bond_type == Chem.BondType.DOUBLE:
            valency += 2
        elif bond_type == Chem.BondType.TRIPLE:
            valency += 3
        elif bond_type == Chem.BondType.AROMATIC:
            valency += 1  # 芳香键通常视为单键
    return valency


def check_stable_atom(atom: Chem.Atom) -> bool:
    """
    检查单个原子是否满足化学价规则
    按照论文定义：atom stable 当且仅当 实际成键数 == 化学上允许的成键数
    
    Args:
        atom: RDKit原子对象
        
    Returns:
        是否稳定
    """
    atomic_num = atom.GetAtomicNum()
    if atomic_num not in ATOM_VALENCY:
        return True  # 未知原子类型，默认稳定
    
    max_valency = ATOM_VALENCY[atomic_num]
    actual_valency = get_atom_valency(atom)
    
    return actual_valency == max_valency # 按照论文定义！！！是=不是<=！


def check_stable_mol(mol: Chem.Mol) -> Tuple[float, bool]:
    """
    检查整个分子是否所有原子都稳定
    
    Args:
        mol: RDKit分子对象
        
    Returns:
        (稳定原子百分比, 是否所有原子都稳定)
    """
    if mol is None or mol.GetNumAtoms() == 0:
        return 0.0, False
    
    stable_count = 0
    total_count = mol.GetNumAtoms()
    
    for atom in mol.GetAtoms():
        if check_stable_atom(atom):
            stable_count += 1
    
    stable_pct = stable_count / total_count if total_count > 0 else 0.0
    is_stable = (stable_count == total_count)
    
    return stable_pct, is_stable


def check_rdkit_validity(mol: Chem.Mol) -> bool:
    """
    检查RDKit分子合法性
    
    Args:
        mol: RDKit分子对象
        
    Returns:
        是否合法
    """
    if mol is None:
        return False
    
    try:
        # 尝试sanitize分子
        mol_copy = Chem.Mol(mol)
        Chem.SanitizeMol(mol_copy)
        return True
    except:
        return False


def get_canonical_smiles(mol: Chem.Mol) -> Optional[str]:
    """
    获取canonical SMILES
    
    Args:
        mol: RDKit分子对象
        
    Returns:
        canonical SMILES字符串，如果失败返回None
    """
    if mol is None:
        return None
    
    try:
        smiles = Chem.MolToSmiles(mol, canonical=True)
        return smiles
    except:
        return None


def extract_largest_connected_component(mol: Chem.Mol) -> Optional[Chem.Mol]:
    """
    从RDKit分子中提取最大连通分支
    
    Args:
        mol: RDKit分子对象
        
    Returns:
        最大连通分支的分子对象，如果失败返回原分子
    """
    if mol is None or mol.GetNumBonds() == 0:
        return mol  # 没有键，返回原分子
    
    try:
        mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
        if len(mol_frags) == 0:
            return mol
        # 选择原子数最多的连通分量
        largest_mol = max(mol_frags, key=lambda m: m.GetNumAtoms())
        return largest_mol
    except Exception as e:
        print(f"警告: 提取最大连通分支失败: {e}")
        return mol


def compute_valency_distribution(mol: Chem.Mol) -> Dict[int, int]:
    """
    统计分子中每个原子的价态分布
    
    Args:
        mol: RDKit分子对象
        
    Returns:
        价态分布字典 {价态: 原子数}
    """
    valency_dist = defaultdict(int)
    
    if mol is None:
        return dict(valency_dist)
    
    for atom in mol.GetAtoms():
        valency = get_atom_valency(atom)
        valency_dist[valency] += 1
    
    return dict(valency_dist)


def compute_atom_type_distribution(mol: Chem.Mol) -> Dict[str, int]:
    """
    统计原子种类分布
    
    Args:
        mol: RDKit分子对象
        
    Returns:
        原子类型分布字典 {元素符号: 原子数}
    """
    atom_dist = defaultdict(int)
    
    if mol is not None:
        for atom in mol.GetAtoms():
            atomic_num = atom.GetAtomicNum()
            symbol = ATOM_NUM_TO_SYMBOL.get(atomic_num, f"Unknown_{atomic_num}")
            atom_dist[symbol] += 1
    
    return dict(atom_dist)


def compute_bond_type_distribution(mol: Chem.Mol) -> Dict[str, int]:
    """
    统计键类型分布
    
    Args:
        mol: RDKit分子对象
        
    Returns:
        键类型分布字典 {键类型: 键数}
    """
    bond_dist = defaultdict(int)
    
    if mol is not None:
        for bond in mol.GetBonds():
            bond_type = bond.GetBondType()
            if bond_type == Chem.BondType.SINGLE:
                bond_dist['single'] += 1
            elif bond_type == Chem.BondType.DOUBLE:
                bond_dist['double'] += 1
            elif bond_type == Chem.BondType.TRIPLE:
                bond_dist['triple'] += 1
            elif bond_type == Chem.BondType.AROMATIC:
                bond_dist['aromatic'] += 1
    
    return dict(bond_dist)


def compute_bond_lengths(mol: Chem.Mol) -> List[float]:
    """
    计算所有键的键长
    
    Args:
        mol: RDKit分子对象
        
    Returns:
        键长列表（单位：Å）
    """
    bond_lengths = []
    
    if mol is None:
        return bond_lengths
    
    conf = mol.GetConformer()
    if conf is None:
        return bond_lengths
    
    for bond in mol.GetBonds():
        begin_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()
        
        begin_pos = conf.GetAtomPosition(begin_idx)
        end_pos = conf.GetAtomPosition(end_idx)
        
        # 计算欧氏距离
        distance = np.sqrt(
            (begin_pos.x - end_pos.x)**2 +
            (begin_pos.y - end_pos.y)**2 +
            (begin_pos.z - end_pos.z)**2
        )
        bond_lengths.append(distance)
    
    return bond_lengths


def compute_bond_angles(mol: Chem.Mol) -> List[float]:
    """
    计算所有键角（三个连续原子形成的角度）
    
    Args:
        mol: RDKit分子对象
        
    Returns:
        键角列表（单位：度）
    """
    bond_angles = []
    
    if mol is None:
        return bond_angles
    
    conf = mol.GetConformer()
    if conf is None:
        return bond_angles
    
    # 对于每个原子，找到它的所有邻居
    for atom in mol.GetAtoms():
        atom_idx = atom.GetIdx()
        neighbors = [n.GetIdx() for n in atom.GetNeighbors()]
        
        # 对于每对邻居，计算键角
        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                neighbor1_idx = neighbors[i]
                neighbor2_idx = neighbors[j]
                
                # 获取三个原子的坐标
                pos_center = conf.GetAtomPosition(atom_idx)
                pos_neighbor1 = conf.GetAtomPosition(neighbor1_idx)
                pos_neighbor2 = conf.GetAtomPosition(neighbor2_idx)
                
                # 计算向量
                vec1 = np.array([
                    pos_neighbor1.x - pos_center.x,
                    pos_neighbor1.y - pos_center.y,
                    pos_neighbor1.z - pos_center.z
                ])
                vec2 = np.array([
                    pos_neighbor2.x - pos_center.x,
                    pos_neighbor2.y - pos_center.y,
                    pos_neighbor2.z - pos_center.z
                ])
                
                # 计算角度
                dot_product = np.dot(vec1, vec2)
                norm1 = np.linalg.norm(vec1)
                norm2 = np.linalg.norm(vec2)
                
                if norm1 > 0 and norm2 > 0:
                    cos_angle = dot_product / (norm1 * norm2)
                    # 限制在[-1, 1]范围内，避免数值误差
                    cos_angle = np.clip(cos_angle, -1.0, 1.0)
                    angle = np.arccos(cos_angle)
                    angle_deg = np.degrees(angle)
                    bond_angles.append(angle_deg)
    
    return bond_angles


def check_single_fragment(mol: Chem.Mol) -> bool:
    """
    检查分子是否只包含一个片段
    
    Args:
        mol: RDKit分子对象
        
    Returns:
        是否为单片段
    """
    if mol is None:
        return False
    
    try:
        mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
        return len(mol_frags) == 1
    except:
        return False


def compute_strain_energy(mol: Chem.Mol) -> Optional[float]:
    """
    计算分子的应变能（生成构象的内能与UFF优化后构象的内能之差）
    
    Args:
        mol: RDKit分子对象（需要包含坐标信息）
        
    Returns:
        应变能（kcal/mol），如果计算失败返回None
    """
    if mol is None or mol.GetNumConformers() == 0:
        return None
    
    if not DRUGS_METRICS_AVAILABLE:
        return None
    
    try:
        # 复制分子以避免修改原始分子
        mol_copy = Chem.Mol(mol)
        mol_copy.RemoveAllConformers()
        
        # 添加原始构象
        original_conf = mol.GetConformer(0)
        new_conf = Chem.Conformer(mol_copy.GetNumAtoms())
        for i in range(mol_copy.GetNumAtoms()):
            pos = original_conf.GetAtomPosition(i)
            new_conf.SetAtomPosition(i, pos)
        mol_copy.AddConformer(new_conf)
        
        # 计算原始构象的能量
        try:
            # 使用 UFF 力场
            ff = AllChem.UFFGetMoleculeForceField(mol_copy, confId=0)
            if ff is None:
                return None
            original_energy = ff.CalcEnergy()
        except:
            return None
        
        # 优化构象
        try:
            # 使用 UFF 优化
            AllChem.UFFOptimizeMolecule(mol_copy, confId=0, maxIters=200)
            
            # 计算优化后的能量
            ff_optimized = AllChem.UFFGetMoleculeForceField(mol_copy, confId=0)
            if ff_optimized is None:
                return None
            optimized_energy = ff_optimized.CalcEnergy()
            
            # 应变能 = 原始能量 - 优化后能量
            strain_energy = original_energy - optimized_energy
            return strain_energy
        except:
            return None
    except Exception as e:
        return None


def compute_ring_sizes(mol: Chem.Mol) -> List[int]:
    """
    计算分子中所有环的大小（环中重原子数）
    
    Args:
        mol: RDKit分子对象
        
    Returns:
        环大小列表
    """
    ring_sizes = []
    
    if mol is None:
        return ring_sizes
    
    try:
        # 获取所有环信息
        ring_info = mol.GetRingInfo()
        atom_rings = ring_info.AtomRings()
        
        for ring in atom_rings:
            # 计算环中重原子数（非氢原子）
            heavy_atoms = sum(1 for atom_idx in ring if mol.GetAtomWithIdx(atom_idx).GetAtomicNum() != 1)
            if heavy_atoms > 0:
                ring_sizes.append(heavy_atoms)
    except:
        pass
    
    return ring_sizes


def compute_atoms_per_mol(mol: Chem.Mol) -> int:
    """
    计算每个分子的原子数（对于多片段分子，只考虑最大片段）
    
    Args:
        mol: RDKit分子对象
        
    Returns:
        最大片段的原子数
    """
    if mol is None:
        return 0
    
    try:
        # 提取最大连通分支
        largest_mol = extract_largest_connected_component(mol)
        if largest_mol is None:
            return 0
        return largest_mol.GetNumAtoms()
    except:
        return 0


def compute_qed(mol: Chem.Mol) -> Optional[float]:
    """
    计算 QED（定量药物相似性）分数
    
    Args:
        mol: RDKit分子对象
        
    Returns:
        QED 分数（0-1之间），如果计算失败返回None
    """
    if mol is None:
        return None
    
    if not DRUGS_METRICS_AVAILABLE:
        return None
    
    try:
        # 需要先 sanitize 分子
        mol_copy = Chem.Mol(mol)
        try:
            Chem.SanitizeMol(mol_copy)
        except:
            return None
        
        qed_score = QED.qed(mol_copy)
        return qed_score
    except:
        return None


def compute_logp(mol: Chem.Mol) -> Optional[float]:
    """
    计算 logP（分配系数的对数值）
    
    Args:
        mol: RDKit分子对象
        
    Returns:
        logP 值，如果计算失败返回None
    """
    if mol is None:
        return None
    
    if not DRUGS_METRICS_AVAILABLE:
        return None
    
    try:
        # 需要先 sanitize 分子
        mol_copy = Chem.Mol(mol)
        try:
            Chem.SanitizeMol(mol_copy)
        except:
            return None
        
        logp = Crippen.MolLogP(mol_copy)
        return logp
    except:
        return None


def compute_wasserstein1_distance(dist1: List[float], dist2: List[float]) -> float:
    """
    计算两个分布的Wasserstein-1距离
    
    Args:
        dist1: 第一个分布的样本值列表
        dist2: 第二个分布的样本值列表
        
    Returns:
        W₁距离
    """
    if len(dist1) == 0 and len(dist2) == 0:
        return 0.0
    if len(dist1) == 0 or len(dist2) == 0:
        return float('inf')
    
    return wasserstein_distance(dist1, dist2)


def compute_total_variation(dist1: Dict, dist2: Dict) -> float:
    """
    计算两个离散分布的Total Variation距离
    
    Args:
        dist1: 第一个分布（字典形式）
        dist2: 第二个分布（字典形式）
        
    Returns:
        TV距离
    """
    # 获取所有可能的键
    all_keys = set(dist1.keys()) | set(dist2.keys())
    
    if len(all_keys) == 0:
        return 0.0
    
    # 归一化分布
    total1 = sum(dist1.values())
    total2 = sum(dist2.values())
    
    if total1 == 0 and total2 == 0:
        return 0.0
    
    # 计算TV距离
    tv = 0.0
    for key in all_keys:
        p1 = dist1.get(key, 0) / total1 if total1 > 0 else 0
        p2 = dist2.get(key, 0) / total2 if total2 > 0 else 0
        tv += abs(p1 - p2)
    
    return 0.5 * tv


def load_test_set_molecules(data_dir: str, limit: Optional[int] = None) -> List[Chem.Mol]:
    """
    从QM9测试集加载分子
    
    Args:
        data_dir: 数据目录路径
        limit: 限制加载的分子数量（用于加速计算）
        
    Returns:
        分子列表
    """
    molecules = []
    data_dir = Path(data_dir)
    
    # 尝试从LMDB加载
    lmdb_path = data_dir / "test_data.lmdb"
    keys_path = data_dir / "test_data_keys.pt"
    
    if lmdb_path.exists() and keys_path.exists():
        print(f"从LMDB加载测试集数据: {lmdb_path}")
        keys = torch.load(keys_path)
        if limit:
            keys = keys[:limit]
        
        db = lmdb.open(str(lmdb_path), readonly=True, lock=False, readahead=False, meminit=False)
        
        for key in tqdm(keys, desc="加载测试集分子"):
            try:
                with db.begin() as txn:
                    if isinstance(key, str):
                        key_bytes = key.encode('utf-8')
                    else:
                        key_bytes = key
                    sample_raw = pickle.loads(txn.get(key_bytes))
                
                # 从sample_raw中提取分子
                if 'mol' in sample_raw:
                    mol = sample_raw['mol']
                    if mol is not None:
                        # 如果有坐标信息，更新分子的坐标
                        if 'coords' in sample_raw:
                            coords = sample_raw['coords']
                            if isinstance(coords, torch.Tensor):
                                coords = coords.numpy()
                            
                            # 更新分子的坐标
                            if mol.GetNumConformers() == 0:
                                conf = Chem.Conformer(mol.GetNumAtoms())
                                for i in range(mol.GetNumAtoms()):
                                    conf.SetAtomPosition(i, Point3D(float(coords[i][0]), float(coords[i][1]), float(coords[i][2])))
                                mol.AddConformer(conf)
                            else:
                                conf = mol.GetConformer(0)
                                for i in range(mol.GetNumAtoms()):
                                    conf.SetAtomPosition(i, Point3D(float(coords[i][0]), float(coords[i][1]), float(coords[i][2])))
                        
                        molecules.append(mol)
            except Exception as e:
                print(f"警告: 加载测试集分子时出错: {e}")
                continue
        
        db.close()
    
    # 如果LMDB不存在，尝试从.pth文件加载
    elif (data_dir / "test_data.pth").exists():
        print(f"从.pth文件加载测试集数据: {data_dir / 'test_data.pth'}")
        data = torch.load(data_dir / "test_data.pth", weights_only=False)
        if limit:
            data = data[:limit]
        
        for sample in tqdm(data, desc="加载测试集分子"):
            try:
                if isinstance(sample, dict) and 'mol' in sample:
                    mol = sample['mol']
                    if mol is not None:
                        # 更新坐标
                        if 'coords' in sample:
                            coords = sample['coords']
                            if isinstance(coords, torch.Tensor):
                                coords = coords.numpy()
                            
                            if mol.GetNumConformers() == 0:
                                conf = Chem.Conformer(mol.GetNumAtoms())
                                for i in range(mol.GetNumAtoms()):
                                    conf.SetAtomPosition(i, Point3D(float(coords[i][0]), float(coords[i][1]), float(coords[i][2])))
                                mol.AddConformer(conf)
                            else:
                                conf = mol.GetConformer(0)
                                for i in range(mol.GetNumAtoms()):
                                    conf.SetAtomPosition(i, Point3D(float(coords[i][0]), float(coords[i][1]), float(coords[i][2])))
                        
                        molecules.append(mol)
            except Exception as e:
                print(f"警告: 加载测试集分子时出错: {e}")
                continue
    
    else:
        print(f"警告: 未找到测试集数据文件（LMDB或.pth）在 {data_dir}")
        return []
    
    print(f"成功加载 {len(molecules)} 个测试集分子")
    return molecules


def compute_test_set_distributions(test_molecules: List[Chem.Mol], ds_type: str = 'qm9') -> Dict:
    """
    计算测试集的分布统计
    
    Args:
        test_molecules: 测试集分子列表
        ds_type: 数据集类型，'qm9' 或 'drugs'
        
    Returns:
        包含所有分布统计的字典
    """
    all_valencies = []
    all_atom_types = []
    all_bond_types = []
    all_bond_lengths = []
    all_bond_angles = []
    
    # Drugs 数据集专用统计
    all_ring_sizes = []
    all_atoms_per_mol = []
    
    for mol in tqdm(test_molecules, desc="计算测试集分布"):
        if mol is None:
            continue
        
        # 价态分布
        val_dist = compute_valency_distribution(mol)
        for valency, count in val_dist.items():
            all_valencies.extend([valency] * count)
        
        # 原子类型分布
        atom_dist = compute_atom_type_distribution(mol)
        for atom_type, count in atom_dist.items():
            all_atom_types.extend([atom_type] * count)
        
        # 键类型分布
        bond_dist = compute_bond_type_distribution(mol)
        for bond_type, count in bond_dist.items():
            all_bond_types.extend([bond_type] * count)
        
        # 键长
        bond_lengths = compute_bond_lengths(mol)
        all_bond_lengths.extend(bond_lengths)
        
        # 键角
        bond_angles = compute_bond_angles(mol)
        all_bond_angles.extend(bond_angles)
        
        # Drugs 数据集专用统计
        if ds_type == 'drugs':
            # Ring sizes
            ring_sizes = compute_ring_sizes(mol)
            all_ring_sizes.extend(ring_sizes)
            
            # Atoms per molecule (largest fragment)
            atoms_per_mol = compute_atoms_per_mol(mol)
            if atoms_per_mol > 0:
                all_atoms_per_mol.append(atoms_per_mol)
    
    result = {
        'valencies': all_valencies,
        'atom_types': all_atom_types,
        'bond_types': all_bond_types,
        'bond_lengths': all_bond_lengths,
        'bond_angles': all_bond_angles,
    }
    
    # 添加 Drugs 数据集专用统计
    if ds_type == 'drugs':
        result['ring_sizes'] = all_ring_sizes
        result['atoms_per_mol'] = all_atoms_per_mol
    
    return result


def evaluate_single_molecule(mol: Chem.Mol, molecule_id: str, save_dir: Optional[Path] = None, ds_type: str = 'qm9') -> Dict:
    """
    评估单个分子
    如果分子有键信息，会提取最大连通分支进行评估
    
    Args:
        mol: RDKit分子对象
        molecule_id: 分子ID
        save_dir: 保存最大连通分支SDF的目录（可选）
        ds_type: 数据集类型，'qm9' 或 'drugs'
        
    Returns:
        评估结果字典
    """
    # 提取最大连通分支（如果分子有键信息）
    original_num_atoms = mol.GetNumAtoms() if mol is not None else 0
    original_num_bonds = mol.GetNumBonds() if mol is not None else 0
    num_components = 1
    
    if mol is not None and mol.GetNumBonds() > 0:
        try:
            mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
            num_components = len(mol_frags)
            if num_components > 1:
                # 选择最大连通分支
                mol = extract_largest_connected_component(mol)
                # 保存最大连通分支
                if save_dir is not None and mol is not None:
                    save_dir.mkdir(parents=True, exist_ok=True)
                    largest_cc_path = save_dir / f"{molecule_id}_largest_cc.sdf"
                    try:
                        sdf_block = Chem.MolToMolBlock(mol)
                        with open(largest_cc_path, 'w', encoding='utf-8') as f:
                            f.write(sdf_block)
                            f.write("\n$$$$\n")
                    except Exception as e:
                        print(f"警告: 保存最大连通分支失败 {largest_cc_path}: {e}")
        except Exception as e:
            print(f"警告: 提取最大连通分支时出错: {e}")
    
    result = {
        'molecule_id': molecule_id,
        'num_atoms': mol.GetNumAtoms() if mol is not None else 0,
        'num_bonds': mol.GetNumBonds() if mol is not None else 0,
        'original_num_atoms': original_num_atoms,
        'original_num_bonds': original_num_bonds,
        'num_components': num_components,
    }
    
    if mol is None:
        result.update({
            'stable_atom_pct': 0.0,
            'is_stable_mol': False,
            'is_valid': False,
            'canonical_smiles': None,
            'is_unique': False,
        })
        return result
    
    # 基础化学合理性检查
    stable_pct, is_stable = check_stable_mol(mol)
    result['stable_atom_pct'] = stable_pct
    result['is_stable_mol'] = is_stable
    
    # RDKit合法性
    result['is_valid'] = check_rdkit_validity(mol)
    
    # Canonical SMILES
    smiles = get_canonical_smiles(mol)
    result['canonical_smiles'] = smiles
    
    # 分布统计
    result['valency_distribution'] = compute_valency_distribution(mol)
    result['atom_type_distribution'] = compute_atom_type_distribution(mol)
    result['bond_type_distribution'] = compute_bond_type_distribution(mol)
    result['bond_lengths'] = compute_bond_lengths(mol)
    result['bond_angles'] = compute_bond_angles(mol)
    
    # Drugs 数据集专用指标
    if ds_type == 'drugs':
        # Single fragment
        result['is_single_fragment'] = check_single_fragment(mol)
        
        # Strain energy
        result['strain_energy'] = compute_strain_energy(mol)
        
        # Ring sizes
        result['ring_sizes'] = compute_ring_sizes(mol)
        
        # Atoms per molecule (largest fragment)
        result['atoms_per_mol'] = compute_atoms_per_mol(mol)
        
        # QED
        result['qed'] = compute_qed(mol)
        
        # SA
        result['sa'] = compute_sa_score(mol)
        
        # logP
        result['logp'] = compute_logp(mol)
    
    return result


def evaluate_molecules(input_dir: Path, output_csv: Path, ds_type: str = 'qm9') -> pd.DataFrame:
    """
    评估目录中的所有SDF分子文件
    
    Args:
        input_dir: 包含SDF文件的目录
        output_csv: 输出CSV文件路径
        ds_type: 数据集类型，'qm9' 或 'drugs'
        
    Returns:
        评估结果DataFrame
    """
    input_dir = Path(input_dir)
    
    # 查找所有SDF文件
    sdf_files = sorted(input_dir.glob("*.sdf"))
    
    if len(sdf_files) == 0:
        print(f"警告: 在 {input_dir} 中未找到SDF文件")
        return pd.DataFrame()
    
    print(f"找到 {len(sdf_files)} 个SDF文件")
    
    all_results = []
    unique_smiles = set()
    
    # 处理每个SDF文件
    for sdf_file in sdf_files:
        molecule_id = sdf_file.stem
        
        # 加载分子
        molecules = load_sdf_molecules(sdf_file)
        
        if len(molecules) == 0:
            print(f"警告: {sdf_file.name} 中未找到有效分子")
            continue
        
        # 处理每个分子（SDF文件可能包含多个分子）
        for idx, mol in enumerate(molecules):
            mol_id = f"{molecule_id}_{idx}" if len(molecules) > 1 else molecule_id
            
            # 评估分子（传入保存目录以保存最大连通分支）
            # 如果input_dir是molecule目录，直接使用；否则使用molecule子目录
            if input_dir.name == 'molecule':
                save_dir = input_dir
            else:
                save_dir = input_dir / 'molecule'
            result = evaluate_single_molecule(mol, mol_id, save_dir=save_dir, ds_type=ds_type)
            
            # 检查唯一性
            smiles = result.get('canonical_smiles')
            if smiles and smiles not in unique_smiles:
                unique_smiles.add(smiles)
                result['is_unique'] = True
            else:
                result['is_unique'] = False
            
            all_results.append(result)
    
    if len(all_results) == 0:
        print("警告: 没有有效的评估结果")
        return pd.DataFrame()
    
    # 转换为DataFrame
    df = pd.DataFrame(all_results)
    
    # 计算总体统计
    print("\n" + "="*80)
    print("评估结果汇总")
    print("="*80)
    
    total_molecules = len(df)
    valid_molecules = df['is_valid'].sum()
    unique_molecules = df['is_unique'].sum()
    stable_molecules = df['is_stable_mol'].sum()
    
    print(f"总分子数: {total_molecules}")
    print(f"有效分子数 (RDKit valid): {valid_molecules} ({100*valid_molecules/total_molecules:.2f}%)")
    print(f"唯一分子数: {unique_molecules} ({100*unique_molecules/total_molecules:.2f}%)")
    print(f"稳定分子数 (所有原子都稳定): {stable_molecules} ({100*stable_molecules/total_molecules:.2f}%)")
    print(f"平均稳定原子百分比: {df['stable_atom_pct'].mean() * 100:.2f}%")
    
    # 计算分布统计
    print("\n" + "="*80)
    print("分布统计")
    print("="*80)
    
    # 价态分布
    all_valencies = []
    for val_dist in df['valency_distribution']:
        for valency, count in val_dist.items():
            all_valencies.extend([valency] * count)
    valency_counter = Counter(all_valencies)
    print(f"价态分布: {dict(sorted(valency_counter.items()))}")
    
    # 原子类型分布
    all_atom_types = []
    for atom_dist in df['atom_type_distribution']:
        for atom_type, count in atom_dist.items():
            all_atom_types.extend([atom_type] * count)
    atom_counter = Counter(all_atom_types)
    print(f"原子类型分布: {dict(sorted(atom_counter.items()))}")
    
    # 键类型分布
    all_bond_types = []
    for bond_dist in df['bond_type_distribution']:
        for bond_type, count in bond_dist.items():
            all_bond_types.extend([bond_type] * count)
    bond_counter = Counter(all_bond_types)
    print(f"键类型分布: {dict(sorted(bond_counter.items()))}")
    
    # 键长统计
    all_bond_lengths = []
    for lengths in df['bond_lengths']:
        all_bond_lengths.extend(lengths)
    if all_bond_lengths:
        print(f"键长统计: 均值={np.mean(all_bond_lengths):.3f}Å, "
              f"中位数={np.median(all_bond_lengths):.3f}Å, "
              f"标准差={np.std(all_bond_lengths):.3f}Å")
    
    # 键角统计
    all_bond_angles = []
    for angles in df['bond_angles']:
        all_bond_angles.extend(angles)
    if all_bond_angles:
        print(f"键角统计: 均值={np.mean(all_bond_angles):.1f}°, "
              f"中位数={np.median(all_bond_angles):.1f}°, "
              f"标准差={np.std(all_bond_angles):.1f}°")
    
    # Drugs 数据集专用指标统计
    if ds_type == 'drugs':
        print("\n" + "="*80)
        print("Drugs 数据集专用指标")
        print("="*80)
        
        # Single fragment
        if 'is_single_fragment' in df.columns:
            single_frag_count = df['is_single_fragment'].sum()
            single_frag_pct = 100 * single_frag_count / total_molecules
            print(f"单片段分子数: {single_frag_count} ({single_frag_pct:.2f}%)")
        
        # Median strain energy
        if 'strain_energy' in df.columns:
            strain_energies = df['strain_energy'].dropna().tolist()
            if strain_energies:
                median_strain_energy = np.median(strain_energies)
                mean_strain_energy = np.mean(strain_energies)
                print(f"中位应变能: {median_strain_energy:.4f} kcal/mol")
                print(f"平均应变能: {mean_strain_energy:.4f} kcal/mol")
        
        # Ring sizes
        if 'ring_sizes' in df.columns:
            all_ring_sizes = []
            for ring_sizes in df['ring_sizes']:
                if isinstance(ring_sizes, list):
                    all_ring_sizes.extend(ring_sizes)
            if all_ring_sizes:
                ring_size_counter = Counter(all_ring_sizes)
                print(f"环大小分布: {dict(sorted(ring_size_counter.items()))}")
        
        # Atoms per molecule
        if 'atoms_per_mol' in df.columns:
            atoms_per_mol = df['atoms_per_mol'].dropna().tolist()
            if atoms_per_mol:
                atoms_per_mol_counter = Counter(atoms_per_mol)
                print(f"每个分子原子数分布（最大片段）: {dict(sorted(atoms_per_mol_counter.items()))}")
        
        # QED
        if 'qed' in df.columns:
            qed_scores = df['qed'].dropna().tolist()
            if qed_scores:
                mean_qed = np.mean(qed_scores)
                median_qed = np.median(qed_scores)
                print(f"QED: 均值={mean_qed:.4f}, 中位数={median_qed:.4f}")
        
        # SA
        if 'sa' in df.columns:
            sa_scores = df['sa'].dropna().tolist()
            if sa_scores:
                mean_sa = np.mean(sa_scores)
                median_sa = np.median(sa_scores)
                print(f"SA: 均值={mean_sa:.4f}, 中位数={median_sa:.4f}")
        
        # logP
        if 'logp' in df.columns:
            logp_scores = df['logp'].dropna().tolist()
            if logp_scores:
                mean_logp = np.mean(logp_scores)
                median_logp = np.median(logp_scores)
                print(f"logP: 均值={mean_logp:.4f}, 中位数={median_logp:.4f}")
    
    # 初始化分布比较变量
    w1_valency = None
    w1_bond_length = None
    w1_bond_angle = None
    tv_atom_types = None
    tv_bond_types = None
    # Drugs 数据集专用分布比较变量
    tv_ring_sizes = None
    tv_atoms_per_mol = None
    
    # 加载测试集并计算分布比较
    print("\n" + "="*80)
    print("分布比较（生成 vs 测试集）")
    print("="*80)
    print("【重要】请确保生成样本和测试集都在同一基础上（都包含H原子或都不包含H原子）")
    print("="*80)
    
    test_data_path = Path(test_data_dir)
    if test_data_path.exists():
        # 加载测试集
        test_molecules = load_test_set_molecules(test_data_dir, limit=test_data_limit)
        
        # 检查测试集是否包含H原子（用于验证一致性）
        # 注意：生成样本默认包含H原子（从*_obabel.sdf加载）
        if len(test_molecules) > 0:
            sample_mol = test_molecules[0]
            if sample_mol is not None:
                has_h_in_test = any(atom.GetAtomicNum() == 1 for atom in sample_mol.GetAtoms())
                print(f"测试集样本检查: 第一个分子{'包含' if has_h_in_test else '不包含'}H原子")
                if not has_h_in_test:
                    print("⚠️  警告: 生成样本包含H原子（从*_obabel.sdf加载），但测试集不包含H原子，分布比较可能不准确！")
        
        if len(test_molecules) > 0:
            # 计算测试集分布
            test_distributions = compute_test_set_distributions(test_molecules, ds_type=ds_type)
            
            # Valency W₁
            if all_valencies and test_distributions['valencies']:
                w1_valency = compute_wasserstein1_distance(all_valencies, test_distributions['valencies'])
                print(f"Valency W₁距离: {w1_valency:.4f}")
            
            # Bond length W₁
            if all_bond_lengths and test_distributions['bond_lengths']:
                w1_bond_length = compute_wasserstein1_distance(all_bond_lengths, test_distributions['bond_lengths'])
                print(f"Bond length W₁距离: {w1_bond_length:.4f}")
            
            # Bond angle W₁
            if all_bond_angles and test_distributions['bond_angles']:
                w1_bond_angle = compute_wasserstein1_distance(all_bond_angles, test_distributions['bond_angles'])
                print(f"Bond angle W₁距离: {w1_bond_angle:.4f}")
            
            # Atoms TV
            if all_atom_types and test_distributions['atom_types']:
                gen_atom_counter = Counter(all_atom_types)
                test_atom_counter = Counter(test_distributions['atom_types'])
                tv_atom_types = compute_total_variation(dict(gen_atom_counter), dict(test_atom_counter))
                print(f"Atom types TV距离: {tv_atom_types:.4f}")
            
            # Bonds TV
            if all_bond_types and test_distributions['bond_types']:
                gen_bond_counter = Counter(all_bond_types)
                test_bond_counter = Counter(test_distributions['bond_types'])
                tv_bond_types = compute_total_variation(dict(gen_bond_counter), dict(test_bond_counter))
                print(f"Bond types TV距离: {tv_bond_types:.4f}")
            
            # Drugs 数据集专用分布比较
            if ds_type == 'drugs':
                # Ring sizes TV
                if 'ring_sizes' in df.columns and 'ring_sizes' in test_distributions:
                    all_gen_ring_sizes = []
                    for ring_sizes in df['ring_sizes']:
                        if isinstance(ring_sizes, list):
                            all_gen_ring_sizes.extend(ring_sizes)
                    if all_gen_ring_sizes and test_distributions['ring_sizes']:
                        gen_ring_counter = Counter(all_gen_ring_sizes)
                        test_ring_counter = Counter(test_distributions['ring_sizes'])
                        tv_ring_sizes = compute_total_variation(dict(gen_ring_counter), dict(test_ring_counter))
                        print(f"Ring sizes TV距离: {tv_ring_sizes:.4f}")
                
                # Atoms per mol TV
                if 'atoms_per_mol' in df.columns and 'atoms_per_mol' in test_distributions:
                    gen_atoms_per_mol = df['atoms_per_mol'].dropna().tolist()
                    if gen_atoms_per_mol and test_distributions['atoms_per_mol']:
                        gen_atoms_counter = Counter(gen_atoms_per_mol)
                        test_atoms_counter = Counter(test_distributions['atoms_per_mol'])
                        tv_atoms_per_mol = compute_total_variation(dict(gen_atoms_counter), dict(test_atoms_counter))
                        print(f"Atoms per mol TV距离: {tv_atoms_per_mol:.4f}")
        else:
            print("警告: 测试集未加载到有效分子")
    else:
        print(f"警告: 测试集数据目录不存在: {test_data_path}")
        print("跳过分布比较")
    
    # 保存详细结果
    # 将字典列转换为JSON字符串以便CSV保存
    df_save = df.copy()
    for col in ['valency_distribution', 'atom_type_distribution', 'bond_type_distribution']:
        if col in df_save.columns:
            df_save[col] = df_save[col].apply(json.dumps)
    for col in ['bond_lengths', 'bond_angles', 'ring_sizes']:
        if col in df_save.columns:
            df_save[col] = df_save[col].apply(lambda x: json.dumps(x) if x else '[]')
    
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_save.to_csv(output_csv, index=False)
    print(f"\n详细结果已保存到: {output_csv}")
    
    # 创建汇总统计CSV
    summary_data = {
        'metric': [
            'total_molecules',
            'valid_molecules',
            'validity_rate',
            'unique_molecules',
            'uniqueness_rate',
            'stable_molecules',
            'stable_mol_rate',
            'avg_stable_atom_pct',
        ],
        'value': [
            total_molecules,
            valid_molecules,
            f"{100*valid_molecules/total_molecules:.2f}%",
            unique_molecules,
            f"{100*unique_molecules/total_molecules:.2f}%",
            stable_molecules,
            f"{100*stable_molecules/total_molecules:.2f}%",
            f"{df['stable_atom_pct'].mean() * 100:.2f}%",
        ]
    }
    
    # 添加分布统计
    if all_valencies:
        valency_dist_normalized = {k: v/len(all_valencies) for k, v in valency_counter.items()}
        summary_data['metric'].append('valency_distribution')
        summary_data['value'].append(json.dumps(valency_dist_normalized))
    
    if all_atom_types:
        atom_dist_normalized = {k: v/len(all_atom_types) for k, v in atom_counter.items()}
        summary_data['metric'].append('atom_type_distribution')
        summary_data['value'].append(json.dumps(atom_dist_normalized))
    
    if all_bond_types:
        bond_dist_normalized = {k: v/len(all_bond_types) for k, v in bond_counter.items()}
        summary_data['metric'].append('bond_type_distribution')
        summary_data['value'].append(json.dumps(bond_dist_normalized))
    
    if all_bond_lengths:
        summary_data['metric'].extend([
            'bond_length_mean',
            'bond_length_median',
            'bond_length_std'
        ])
        summary_data['value'].extend([
            f"{np.mean(all_bond_lengths):.4f}",
            f"{np.median(all_bond_lengths):.4f}",
            f"{np.std(all_bond_lengths):.4f}"
        ])
    
    if all_bond_angles:
        summary_data['metric'].extend([
            'bond_angle_mean',
            'bond_angle_median',
            'bond_angle_std'
        ])
        summary_data['value'].extend([
            f"{np.mean(all_bond_angles):.2f}",
            f"{np.median(all_bond_angles):.2f}",
            f"{np.std(all_bond_angles):.2f}"
        ])
    
    # 添加分布比较指标
    if w1_valency is not None:
        summary_data['metric'].append('valency_w1_distance')
        summary_data['value'].append(f"{w1_valency:.6f}")
    
    if w1_bond_length is not None:
        summary_data['metric'].append('bond_length_w1_distance')
        summary_data['value'].append(f"{w1_bond_length:.6f}")
    
    if w1_bond_angle is not None:
        summary_data['metric'].append('bond_angle_w1_distance')
        summary_data['value'].append(f"{w1_bond_angle:.6f}")
    
    if tv_atom_types is not None:
        summary_data['metric'].append('atom_types_tv_distance')
        summary_data['value'].append(f"{tv_atom_types:.6f}")
    
    if tv_bond_types is not None:
        summary_data['metric'].append('bond_types_tv_distance')
        summary_data['value'].append(f"{tv_bond_types:.6f}")
    
    # 添加 Drugs 数据集专用指标到汇总
    if ds_type == 'drugs':
        # Single fragment
        if 'is_single_fragment' in df.columns:
            single_frag_count = df['is_single_fragment'].sum()
            single_frag_pct = 100 * single_frag_count / total_molecules
            summary_data['metric'].extend(['single_fragment_count', 'single_fragment_rate'])
            summary_data['value'].extend([single_frag_count, f"{single_frag_pct:.2f}%"])
        
        # Median strain energy
        if 'strain_energy' in df.columns:
            strain_energies = df['strain_energy'].dropna().tolist()
            if strain_energies:
                median_strain_energy = np.median(strain_energies)
                mean_strain_energy = np.mean(strain_energies)
                summary_data['metric'].extend(['median_strain_energy', 'mean_strain_energy'])
                summary_data['value'].extend([f"{median_strain_energy:.6f}", f"{mean_strain_energy:.6f}"])
        
        # Ring sizes TV
        if tv_ring_sizes is not None:
            summary_data['metric'].append('ring_sizes_tv_distance')
            summary_data['value'].append(f"{tv_ring_sizes:.6f}")
        
        # Atoms per mol TV
        if tv_atoms_per_mol is not None:
            summary_data['metric'].append('atoms_per_mol_tv_distance')
            summary_data['value'].append(f"{tv_atoms_per_mol:.6f}")
        
        # QED
        if 'qed' in df.columns:
            qed_scores = df['qed'].dropna().tolist()
            if qed_scores:
                mean_qed = np.mean(qed_scores)
                median_qed = np.median(qed_scores)
                summary_data['metric'].extend(['qed_mean', 'qed_median'])
                summary_data['value'].extend([f"{mean_qed:.6f}", f"{median_qed:.6f}"])
        
        # SA
        if 'sa' in df.columns:
            sa_scores = df['sa'].dropna().tolist()
            if sa_scores:
                mean_sa = np.mean(sa_scores)
                median_sa = np.median(sa_scores)
                summary_data['metric'].extend(['sa_mean', 'sa_median'])
                summary_data['value'].extend([f"{mean_sa:.6f}", f"{median_sa:.6f}"])
        
        # logP
        if 'logp' in df.columns:
            logp_scores = df['logp'].dropna().tolist()
            if logp_scores:
                mean_logp = np.mean(logp_scores)
                median_logp = np.median(logp_scores)
                summary_data['metric'].extend(['logp_mean', 'logp_median'])
                summary_data['value'].extend([f"{mean_logp:.6f}", f"{median_logp:.6f}"])
    
    summary_df = pd.DataFrame(summary_data)
    summary_csv = output_csv.parent / f"{output_csv.stem}_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"汇总统计已保存到: {summary_csv}")
    
    return df


def main():
    # 从文件头读取实验目录
    exp_dir_path = Path(exp_dir)
    
    # SDF文件目录
    sdf_dir = exp_dir_path / 'molecule'
    if not sdf_dir.exists():
        print(f"错误: SDF文件目录不存在: {sdf_dir}")
        return
    
    # 输出目录
    output_dir = exp_dir_path / 'evaluate'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 输出CSV文件名
    output_csv = output_dir / 'evaluation_results.csv'
    
    print("="*80)
    print("SDF分子评估")
    print("="*80)
    print(f"实验目录: {exp_dir_path}")
    print(f"SDF文件目录: {sdf_dir}")
    print(f"输出目录: {output_dir}")
    print(f"输出CSV文件: {output_csv}")
    print(f"数据集类型: {dataset_type}")
    print("注意: 默认评估包含H原子的分子（从*_obabel.sdf文件加载）")
    print("="*80)
    
    evaluate_molecules(sdf_dir, output_csv, ds_type=dataset_type)


if __name__ == '__main__':
    main()

