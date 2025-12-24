"""
分子质量评估模块
包含有效性、唯一性、新颖性、稳定性等质量指标评估
"""

import torch
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
from collections import Counter

from rdkit import Chem

from funcmol.analysis.rdkit_functions import Molecule, check_stability
from funcmol.evaluation.utils_evaluation import atom_decoder_dict
from funcmol.evaluation.bond_evaluation import build_xae_molecule


class SimpleSamplingMetrics:
    """简化的分子采样指标评估器，不依赖MiDi"""
    
    def __init__(self, train_smiles=None, dataset_infos=None, test=True):
        self.dataset_infos = dataset_infos
        self.atom_decoder = dataset_infos.atom_decoder if dataset_infos else ['H', 'C', 'N', 'O', 'F']
        self.train_smiles = set(train_smiles) if train_smiles else set()
        self.test = test
        
        # 初始化指标
        self.atom_stable = 0.0
        self.mol_stable = 0.0
        self.validity_metric = 0.0
        self.uniqueness = 0.0
        self.novelty = 0.0
        self.mean_components = 0.0
        self.max_components = 0.0
        
    def compute_validity(self, generated):
        """计算分子有效性"""
        valid = []
        num_components = []
        all_smiles = []
        error_message = Counter()
        
        for i, mol in enumerate(generated):
            rdmol = mol.rdkit_mol
            if rdmol is not None:
                try:
                    mol_frags = Chem.rdmolops.GetMolFrags(rdmol, asMols=True, sanitizeFrags=False)
                    num_components.append(len(mol_frags))
                    if len(mol_frags) > 1:
                        error_message[4] += 1
                    largest_mol = max(mol_frags, default=mol, key=lambda m: m.GetNumAtoms())
                    Chem.SanitizeMol(largest_mol)
                    smiles = Chem.MolToSmiles(largest_mol)
                    valid.append(smiles)
                    all_smiles.append(smiles)
                    error_message[-1] += 1
                except Chem.rdchem.AtomValenceException as e:
                    error_message[1] += 1
                    try:
                        invalid_smiles = Chem.MolToSmiles(rdmol, sanitize=False)
                        all_smiles.append(f"INVALID_AtomValence: {invalid_smiles}")
                    except:
                        all_smiles.append(f"INVALID_AtomValence: 无法生成SMILES")
                except Chem.rdchem.KekulizeException as e:
                    error_message[2] += 1
                    try:
                        invalid_smiles = Chem.MolToSmiles(rdmol, sanitize=False)
                        all_smiles.append(f"INVALID_Kekulize: {invalid_smiles}")
                    except:
                        all_smiles.append(f"INVALID_Kekulize: 无法生成SMILES")
                except (Chem.rdchem.AtomKekulizeException, ValueError) as e:
                    error_message[3] += 1
                    try:
                        invalid_smiles = Chem.MolToSmiles(rdmol, sanitize=False)
                        all_smiles.append(f"INVALID_Other: {invalid_smiles}")
                    except:
                        all_smiles.append(f"INVALID_Other: 无法生成SMILES")
            else:
                all_smiles.append(f"INVALID_NoRDKit: 无法构建RDKit分子")
        
        self.validity_metric = len(valid) / len(generated) if generated else 0.0
        if num_components:
            self.mean_components = sum(num_components) / len(num_components)
            self.max_components = max(num_components)
        else:
            self.mean_components = 0.0
            self.max_components = 0.0
            
        not_connected = 100.0 * error_message[4] / len(generated) if generated else 0.0
        connected_components = 100.0 - not_connected
        return valid, connected_components, all_smiles, error_message
    
    def evaluate(self, generated, local_rank):
        """评估分子质量"""
        # Validity
        valid, connected_components, all_smiles, error_message = self.compute_validity(generated)
        
        validity = self.validity_metric
        uniqueness, novelty = 0, 0
        mean_components = self.mean_components
        max_components = self.max_components
        
        # Uniqueness
        if len(valid) > 0:
            unique = list(set(valid))
            self.uniqueness = len(unique) / len(valid)
            uniqueness = self.uniqueness
            
            if self.train_smiles:
                novel = []
                for smiles in unique:
                    if smiles not in self.train_smiles:
                        novel.append(smiles)
                self.novelty = len(novel) / len(unique)
            novelty = self.novelty
        
        num_molecules = len(generated)
        print(f"Validity over {num_molecules} molecules: {validity * 100:.2f}%")
        print(f"Number of connected components of {num_molecules} molecules: "
              f"mean:{mean_components:.2f} max:{max_components:.2f}")
        print(f"Connected components of {num_molecules} molecules: {connected_components:.2f}")
        print(f"Uniqueness: {uniqueness * 100:.2f}%")
        print(f"Novelty: {novelty * 100:.2f}%")
        
        return all_smiles
    
    def __call__(self, molecules, name, current_epoch, local_rank):
        """调用评估函数"""
        # Atom and molecule stability
        if not self.dataset_infos.remove_h:
            print(f'Analyzing molecule stability on {local_rank}...')
            stable_mols = 0
            stable_atoms = 0
            total_atoms = 0
            
            for i, mol in enumerate(molecules):
                mol_stable, at_stable, num_bonds = check_stability(
                    mol, self.dataset_infos, atom_decoder=self.atom_decoder
                )
                if mol_stable.item() > 0.5:
                    stable_mols += 1
                stable_atoms += at_stable.item()
                total_atoms += num_bonds
            
            self.mol_stable = stable_mols / len(molecules) if molecules else 0.0
            self.atom_stable = stable_atoms / total_atoms if total_atoms > 0 else 0.0
            
            stability_dict = {'mol_stable': self.mol_stable, 'atm_stable': self.atom_stable}
            if local_rank == 0:
                print("Stability metrics:", stability_dict)
        
        # Validity, uniqueness, novelty
        all_generated_smiles = self.evaluate(molecules, local_rank=local_rank)
        
        # Save results
        os.makedirs('graphs', exist_ok=True)
        textfile = open(f'graphs/valid_unique_molecules_e{current_epoch}_GR{local_rank}.txt', "w")
        textfile.writelines([smiles + '\n' for smiles in all_generated_smiles])
        textfile.close()
        
        if self.test:
            filename = f'final_smiles_GR{local_rank}_{0}.txt'
            for i in range(2, 10):
                if os.path.exists(filename):
                    filename = f'final_smiles_GR{local_rank}_{i}.txt'
                else:
                    break
            with open(filename, 'w') as fp:
                for smiles in all_generated_smiles:
                    fp.write("%s\n" % smiles)
                print(f'All smiles saved on rank {local_rank}')


def load_molecules_from_npz(molecule_dir):
    """
    从 .npz 文件加载分子对象
    
    Args:
        molecule_dir: 包含 .npz 文件的目录路径
        
    Returns:
        list: Molecule 对象列表
    """
    molecule_dir = Path(molecule_dir)
    npz_files = sorted(molecule_dir.glob("generated_*.npz"))
    
    print(f"找到 {len(npz_files)} 个 .npz 分子文件")
    
    atom_decoder = atom_decoder_dict['qm9_with_h']
    molecules = []
    
    for npz_file in tqdm(npz_files, desc="加载分子文件"):
        try:
            # 加载 npz 文件
            data = np.load(npz_file)
            coords = data['coords']  # (N, 3) 坐标
            types = data['types']    # (N,) 原子类型
            
            # 转换为torch张量
            positions = torch.tensor(coords, dtype=torch.float32)
            atom_types = torch.tensor(types, dtype=torch.long)
            
            # 过滤掉填充的原子（值为 -1）
            valid_mask = atom_types != -1
            if not valid_mask.any():
                continue
            
            positions = positions[valid_mask]
            atom_types = atom_types[valid_mask]
            
            # 构建键类型矩阵
            dataset_info = {'name': 'qm9'}
            _, _, bond_types = build_xae_molecule(
                positions=positions,
                atom_types=atom_types,
                dataset_info=dataset_info,
                atom_decoder=atom_decoder
            )
            
            # 创建零电荷
            charges = torch.zeros_like(atom_types)
            
            # 创建 Molecule 对象
            molecule = Molecule(
                atom_types=atom_types.long(),
                bond_types=bond_types.long(),
                positions=positions.float(),
                charges=charges,
                atom_decoder=atom_decoder
            )
            
            molecules.append(molecule)
            
        except Exception as e:
            print(f"\n加载文件 {npz_file} 时出错: {e}")
            continue
    
    print(f"成功加载 {len(molecules)} 个分子")
    return molecules


def evaluate_quality(molecules, output_dir=None):
    """
    评估分子质量指标
    
    Args:
        molecules: Molecule 对象列表
        output_dir: 输出目录（可选）
    
    Returns:
        dict: 包含质量评估结果的字典
    """
    if not molecules:
        print("没有有效的分子可以评估！")
        return None
    
    # 创建简化的数据集信息
    class SimpleDatasetInfo:
        def __init__(self):
            self.atom_decoder = atom_decoder_dict['qm9_with_h']
            self.remove_h = False
            # 添加必要的统计信息（简化版本）
            self.statistics = {
                'test': type('obj', (object,), {
                    'num_nodes': {i: 1 for i in range(1, 30)},
                    'atom_types': torch.ones(5),
                    'bond_types': torch.ones(5),
                    'charge_types': torch.ones(5),
                    'valencies': {atom: {0: 1, 1: 1, 2: 1, 3: 1, 4: 1} for atom in atom_decoder_dict['qm9_with_h']},
                    'bond_lengths': {1: {1.5: 1}, 2: {1.3: 1}, 3: {1.2: 1}, 4: {1.4: 1}},
                    'bond_angles': torch.ones(5, 1801)
                })()
            }
    
    dataset_infos = SimpleDatasetInfo()
    
    # 创建采样指标评估器
    sampling_metrics = SimpleSamplingMetrics(
        train_smiles=[],
        dataset_infos=dataset_infos,
        test=True
    )
    
    # 评估分子
    print("\n" + "="*60)
    print(f"开始评估分子质量（共 {len(molecules)} 个分子）...")
    print("="*60)
    
    print("正在计算有效性、唯一性和新颖性...")
    sampling_metrics(
        molecules=molecules,
        name='generated_molecules',
        current_epoch=0,
        local_rank=0
    )
    print("评估完成！")
    
    # 打印结果
    print("\n" + "="*60)
    print("分子质量评估结果")
    print("="*60)
    
    validity = sampling_metrics.validity_metric
    uniqueness = sampling_metrics.uniqueness
    novelty = sampling_metrics.novelty
    mean_components = sampling_metrics.mean_components
    max_components = sampling_metrics.max_components
    mol_stable = sampling_metrics.mol_stable
    atom_stable = sampling_metrics.atom_stable
    
    print(f"\n📊 总体质量指标:")
    print(f"  有效性 (Validity): {validity*100:.2f}%")
    print(f"  唯一性 (Uniqueness): {uniqueness*100:.2f}%")
    print(f"  新颖性 (Novelty): {novelty*100:.2f}%")
    print(f"  平均连通分量数: {mean_components:.2f}")
    print(f"  最大连通分量数: {max_components:.2f}")
    print(f"  分子稳定性: {float(mol_stable)*100:.2f}%")
    print(f"  原子稳定性: {float(atom_stable)*100:.2f}%")
    
    # 统计所有分子
    print(f"\n📊 统计所有分子（共 {len(molecules)} 个）...")
    total_valid = sum(1 for mol in tqdm(molecules, desc="检查有效性", leave=False) if mol.rdkit_mol is not None)
    total_stable = 0
    for mol in tqdm(molecules, desc="检查稳定性"):
        if mol.rdkit_mol is not None:
            try:
                mol_stable, _, _ = check_stability(
                    mol, None, atom_decoder=atom_decoder_dict['qm9_with_h']
                )
                if mol_stable.item() > 0.5:
                    total_stable += 1
            except:
                pass
    
    print(f"\n📊 统计摘要:")
    print(f"  总分子数: {len(molecules)}")
    print(f"  有效分子数: {total_valid}")
    print(f"  稳定分子数: {total_stable}")
    print(f"  有效性: {total_valid/len(molecules)*100:.1f}%")
    print(f"  稳定性: {total_stable/len(molecules)*100:.1f}%")
    
    # 保存结果到文件（如果指定了输出目录）
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results_file = output_dir / "evaluation_results.txt"
        with open(results_file, 'w', encoding='utf-8') as f:
            f.write("分子质量评估结果\n")
            f.write("="*60 + "\n\n")
            f.write(f"总分子数: {len(molecules)}\n")
            f.write(f"有效分子数: {total_valid}\n")
            f.write(f"稳定分子数: {total_stable}\n")
            f.write(f"有效性: {validity*100:.2f}%\n")
            f.write(f"唯一性: {uniqueness*100:.2f}%\n")
            f.write(f"新颖性: {novelty*100:.2f}%\n")
            f.write(f"平均连通分量数: {mean_components:.2f}\n")
            f.write(f"最大连通分量数: {max_components:.2f}\n")
            f.write(f"分子稳定性: {float(mol_stable)*100:.2f}%\n")
            f.write(f"原子稳定性: {float(atom_stable)*100:.2f}%\n")
        
        print(f"\n结果已保存到: {results_file}")
    
    return {
        'validity': validity,
        'uniqueness': uniqueness,
        'novelty': novelty,
        'mean_components': mean_components,
        'max_components': max_components,
        'mol_stable': mol_stable,
        'atom_stable': atom_stable,
        'total_valid': total_valid,
        'total_stable': total_stable,
        'num_molecules': len(molecules)
    }

