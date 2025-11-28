import torch
import numpy as np
import glob
import os

from funcmol.analysis.rdkit_functions import Molecule

# Do not move these imports, the order seems to matter
from rdkit import Chem
import torch
import pytorch_lightning as pl
import torch_geometric

import hydra
import omegaconf

# 简化的导入，不依赖MiDi
from funcmol.analysis.rdkit_functions import Molecule, check_stability
from collections import Counter


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
                    print(f"分子 {i+1}: 有效SMILES = {smiles}")
                except Chem.rdchem.AtomValenceException as e:
                    error_message[1] += 1
                    print(f"分子 {i+1}: AtomValence错误 - {e}")
                    # 即使无效也尝试获取SMILES
                    try:
                        invalid_smiles = Chem.MolToSmiles(rdmol, sanitize=False)
                        all_smiles.append(f"INVALID_AtomValence: {invalid_smiles}")
                        print(f"分子 {i+1}: 无效SMILES = {invalid_smiles}")
                    except:
                        all_smiles.append(f"INVALID_AtomValence: 无法生成SMILES")
                except Chem.rdchem.KekulizeException as e:
                    error_message[2] += 1
                    print(f"分子 {i+1}: Kekulize错误 - {e}")
                    try:
                        invalid_smiles = Chem.MolToSmiles(rdmol, sanitize=False)
                        all_smiles.append(f"INVALID_Kekulize: {invalid_smiles}")
                        print(f"分子 {i+1}: 无效SMILES = {invalid_smiles}")
                    except:
                        all_smiles.append(f"INVALID_Kekulize: 无法生成SMILES")
                except (Chem.rdchem.AtomKekulizeException, ValueError) as e:
                    error_message[3] += 1
                    print(f"分子 {i+1}: 其他错误 - {e}")
                    try:
                        invalid_smiles = Chem.MolToSmiles(rdmol, sanitize=False)
                        all_smiles.append(f"INVALID_Other: {invalid_smiles}")
                        print(f"分子 {i+1}: 无效SMILES = {invalid_smiles}")
                    except:
                        all_smiles.append(f"INVALID_Other: 无法生成SMILES")
            else:
                print(f"分子 {i+1}: RDKit分子对象为None")
                all_smiles.append(f"INVALID_NoRDKit: 无法构建RDKit分子")
        
        print(f"Error messages: AtomValence {error_message[1]}, Kekulize {error_message[2]}, other {error_message[3]}, "
              f" -- No error {error_message[-1]}")
        
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
                mol_stable, at_stable, num_bonds = check_stability(mol, self.dataset_infos, atom_decoder=self.atom_decoder)
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
    
    def compute(self):
        """计算指标值"""
        return self.validity_metric


atom_encoder_dict = {'qm9_with_h': {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4},
                     'qm9_no_h': {'C': 0, 'N': 1, 'O': 2, 'F': 3},
                     'geom_with_h': {'H': 0, 'B': 1, 'C': 2, 'N': 3, 'O': 4, 'F': 5, 'Al': 6, 'Si': 7,'P': 8,
                                     'S': 9, 'Cl': 10, 'As': 11, 'Br': 12, 'I': 13, 'Hg': 14, 'Bi': 15},
                     'geom_no_h': {'B': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4, 'Al': 5, 'Si': 6, 'P': 7, 'S': 8,
                                   'Cl': 9, 'As': 10,'Br': 11, 'I': 12, 'Hg': 13, 'Bi': 14}}

atom_decoder_dict = {'qm9_with_h': ['H', 'C', 'N', 'O', 'F'],
                     'qm9_no_h': ['C', 'N', 'O', 'F'],
                     'geom_with_h': ['H', 'B', 'C', 'N', 'O', 'F', 'Al', 'Si', 'P', 'S', 'Cl', 'As', 'Br',
                                     'I', 'Hg', 'Bi'],
                     'geom_no_h': ['B', 'C', 'N', 'O', 'F', 'Al', 'Si', 'P', 'S', 'Cl', 'As', 'Br', 'I',
                                   'Hg', 'Bi']}

bonds1 = {'H': {'H': 74, 'C': 109, 'N': 101, 'O': 96, 'F': 92,
                'B': 119, 'Si': 148, 'P': 144, 'As': 152, 'S': 134,
                'Cl': 127, 'Br': 141, 'I': 161},
          'C': {'H': 109, 'C': 154, 'N': 147, 'O': 143, 'F': 135,
                'Si': 185, 'P': 184, 'S': 182, 'Cl': 177, 'Br': 194,
                'I': 214},
          'N': {'H': 101, 'C': 147, 'N': 145, 'O': 140, 'F': 136,
                'Cl': 175, 'Br': 214, 'S': 168, 'I': 222, 'P': 177},
          'O': {'H': 96, 'C': 143, 'N': 140, 'O': 148, 'F': 142,
                'Br': 172, 'S': 151, 'P': 163, 'Si': 163, 'Cl': 164,
                'I': 194},
          'F': {'H': 92, 'C': 135, 'N': 136, 'O': 142, 'F': 142,
                'S': 158, 'Si': 160, 'Cl': 166, 'Br': 178, 'P': 156,
                'I': 187},
          'B': {'H':  119, 'Cl': 175},
          'Si': {'Si': 233, 'H': 148, 'C': 185, 'O': 163, 'S': 200,
                 'F': 160, 'Cl': 202, 'Br': 215, 'I': 243 },
          'Cl': {'Cl': 199, 'H': 127, 'C': 177, 'N': 175, 'O': 164,
                 'P': 203, 'S': 207, 'B': 175, 'Si': 202, 'F': 166,
                 'Br': 214},
          'S': {'H': 134, 'C': 182, 'N': 168, 'O': 151, 'S': 204,
                'F': 158, 'Cl': 207, 'Br': 225, 'Si': 200, 'P': 210,
                'I': 234},
          'Br': {'Br': 228, 'H': 141, 'C': 194, 'O': 172, 'N': 214,
                 'Si': 215, 'S': 225, 'F': 178, 'Cl': 214, 'P': 222},
          'P': {'P': 221, 'H': 144, 'C': 184, 'O': 163, 'Cl': 203,
                'S': 210, 'F': 156, 'N': 177, 'Br': 222},
          'I': {'H': 161, 'C': 214, 'Si': 243, 'N': 222, 'O': 194,
                'S': 234, 'F': 187, 'I': 266},
          'As': {'H': 152}
          }

bonds2 = {'C': {'C': 134, 'N': 129, 'O': 120, 'S': 160},
          'N': {'C': 129, 'N': 125, 'O': 121},
          'O': {'C': 120, 'N': 121, 'O': 121, 'P': 150},
          'P': {'O': 150, 'S': 186},
          'S': {'P': 186}}


bonds3 = {'C': {'C': 120, 'N': 116, 'O': 113},
          'N': {'C': 116, 'N': 110},
          'O': {'C': 113}}

margin1, margin2, margin3 = 10, 5, 3

def get_bond_order(atom1, atom2, distance, check_exists=False):
    distance = 100 * distance  # We change the metric

    # Check exists for large molecules where some atom pairs do not have a
    # typical bond length.
    if check_exists:
        if atom1 not in bonds1:
            return 0
        if atom2 not in bonds1[atom1]:
            return 0

    # margin1, margin2 and margin3 have been tuned to maximize the stability of
    # the QM9 true samples.
    if distance < bonds1[atom1][atom2] + margin1:

        # Check if atoms in bonds2 dictionary.
        if atom1 in bonds2 and atom2 in bonds2[atom1]:
            thr_bond2 = bonds2[atom1][atom2] + margin2
            if distance < thr_bond2:
                if atom1 in bonds3 and atom2 in bonds3[atom1]:
                    thr_bond3 = bonds3[atom1][atom2] + margin3
                    if distance < thr_bond3:
                        return 3        # Triple
                return 2            # Double
        return 1                # Single
    return 0                    # No bond


def geom_predictor(p, l, margin1=5, limit_bonds_to_one=False):
    """ p: atom pair (couple of str)
        l: bond length (float)"""
    bond_order = get_bond_order(p[0], p[1], l, check_exists=True)

    # If limit_bonds_to_one is enabled, every bond type will return 1.
    if limit_bonds_to_one:
        return 1 if bond_order > 0 else 0
    else:
        return bond_order


def build_xae_molecule(positions, atom_types, dataset_info, atom_decoder):
    """ Returns a triplet (X, A, E): atom_types, adjacency matrix, edge_types
        args:
        positions: N x 3  (already masked to keep final number nodes)
        atom_types: N
        returns:
        X: N         (int)
        A: N x N     (bool)                  (binary adjacency matrix)
        E: N x N     (int)  (bond type, 0 if no bond) such that A = E.bool()
    """
    n = positions.shape[0]
    X = atom_types
    A = torch.zeros((n, n), dtype=torch.bool)
    E = torch.zeros((n, n), dtype=torch.int)

    pos = positions.unsqueeze(0)
    dists = torch.cdist(pos, pos, p=2).squeeze(0)
    for i in range(n):
        for j in range(i):
            pair = sorted([atom_types[i], atom_types[j]])
            if dataset_info['name'] == 'qm9':
                order = get_bond_order(atom_decoder[pair[0]], atom_decoder[pair[1]], dists[i, j])
            elif dataset_info['name'] == 'geom':
                order = geom_predictor((atom_decoder[pair[0]], atom_decoder[pair[1]]), dists[i, j], limit_bonds_to_one=True)
            # TODO: a batched version of get_bond_order to avoid the for loop
            if order > 0:
                # Warning: the graph should be DIRECTED
                A[i, j] = 1
                E[i, j] = order
                E[j, i] = order
    return X, A, E

def data_atom_xyz(path, name, dataset_info):
    atom_encoder = atom_encoder_dict[name]
    atom_decoder = atom_decoder_dict[name]
    file_list = glob.glob(path + "*.xyz")

    all_data = []
    for file_path in file_list:
        with open(file_path, "r") as f:
            f.readline()
            data = np.loadtxt(f, dtype=[("symbol", "U10"), ("x", float), ("y", float), ("z", float)])
        all_data.append(data)

    all_mols = []
    for data in all_data:
        atom_types_str = data['symbol']
        atom_types = []
        for type in atom_types_str:
            atom_type = atom_encoder[type]
            atom_types.append(atom_type)
        atom_types = torch.tensor(atom_types)
        pos = data[['x', 'y', 'z']]
        positions = pos.tolist()
        positions = torch.tensor(positions)

        X, A, E = build_xae_molecule(positions=positions, atom_types=atom_types, dataset_info=dataset_info, atom_decoder=atom_decoder)
        charges = torch.zeros(X.shape)
        E = E.to(dtype=torch.int64)
        molecule = Molecule(atom_types=X, bond_types=E, positions=positions, charges=charges, atom_decoder=atom_decoder)
        molecule.build_molecule(atom_decoder=atom_decoder)
        all_mols.append(molecule)

    return all_mols


def load_generated_molecules_from_npz(molecule_dir):
    """
    从npz文件加载FuncMol生成的分子
    
    :param molecule_dir: 包含npz文件的目录路径
    :return: Molecule对象列表
    """
    import glob
    from pathlib import Path
    
    atom_decoder = atom_decoder_dict['qm9_with_h']  # QM9包含氢原子
    molecules = []
    molecule_files = sorted(Path(molecule_dir).glob("*.npz"))
    
    print(f"找到 {len(molecule_files)} 个生成分子文件")
    
    for i, mol_file in enumerate(molecule_files):
        try:
            # 加载npz文件
            data = np.load(mol_file)
            coords = data['coords']  # (N, 3) 坐标
            types = data['types']    # (N,) 原子类型
            
            # 转换为torch张量
            positions = torch.tensor(coords, dtype=torch.float32)
            atom_types = torch.tensor(types, dtype=torch.long)
            
            # 创建零电荷（QM9数据集通常没有电荷）
            charges = torch.zeros_like(atom_types)
            
            # 构建键类型矩阵（简化版本，基于距离）
            bond_types = build_simple_bond_matrix(positions, atom_types)
            
            # 创建Molecule对象
            molecule = Molecule(
                atom_types=atom_types,
                bond_types=bond_types,
                positions=positions,
                charges=charges,
                atom_decoder=atom_decoder
            )
            
            molecules.append(molecule)
            
            if i % 10 == 0:
                print(f"已加载 {i+1} 个分子...")
                
        except Exception as e:
            print(f"加载分子文件 {mol_file} 时出错: {e}")
            continue
    
    print(f"成功加载 {len(molecules)} 个生成分子")
    return molecules


def build_simple_bond_matrix(positions, atom_types):
    """
    根据原子间距离构建简化的键类型矩阵
    
    Args:
        positions: 原子位置 (N, 3)
        atom_types: 原子类型 (N,)
    
    Returns:
        torch.Tensor: 键类型矩阵 (N, N)
    """
    n_atoms = len(atom_types)
    bond_types = torch.zeros((n_atoms, n_atoms), dtype=torch.long)
    
    # 计算原子间距离
    distances = torch.cdist(positions, positions)
    
    # 简化的键长阈值
    max_bond_length = 1.8  # 最大键长
    
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            distance = distances[i, j].item()
            
            if distance < max_bond_length:
                # 简化为单键
                bond_types[i, j] = 1
                bond_types[j, i] = 1
    
    return bond_types


def open_babel_preprocess(file, name):
    """
    :param file: str
    :param name: 'qm9_with_h', 'qm9_no_h, 'geom_with_h', 'geom_no_h'
    :return:
    """
    atom_encoder = atom_encoder_dict[name]
    atom_decoder = atom_decoder_dict[name]

    with open(file, "r") as f:
        lines = f.readlines()[3:]

    result = []
    temp = []

    for line in lines:
        line = line.strip()

        if not line or "M" in line or "$" in line or "OpenBabel" in line:
            continue

        vec = line.split()
        if vec != ['end']:
            temp.append(vec)
        else:
            result.append(temp)
            temp = []

    all_mols = []

    for array in result:
        atom_temp = []
        pos_temp = []
        new_pos = []
        col = row = array[0][0]
        for i in range(int(col)):
            element = array[i + 1][3]
            x = atom_encoder.get(element, None)
            if x is None:
                # Handle elements not in the map
                print('Element ' + element + ' is not handled in the current mapping')
            atom_temp.append(x)
            x_pos = array[i + 1][0]
            x_pos = float(x_pos)
            y_pos = array[i + 1][1]
            y_pos = float(y_pos)
            z_pos = array[i + 1][2]
            z_pos = float(z_pos)
            pos_temp.append([x_pos, y_pos, z_pos])
        new_pos.append(pos_temp)

        iteration = array[0][1]
        cols, rows = int(col), int(row)
        matrix = [[0 for x in range(cols)] for y in range(rows)]
        for j in range(int(iteration)):
            d = j + int(col) + 1
            a = int(array[d][0]) - 1
            b = int(array[d][1]) - 1
            c = int(array[d][2])
            matrix[a][b] = c
            matrix[b][a] = c

        X = torch.tensor(atom_temp)
        charges = torch.zeros(X.shape)
        E = torch.tensor(matrix)
        posis = torch.tensor(new_pos[0])
        molecule = Molecule(atom_types=X, bond_types=E, positions=posis, charges=charges, atom_decoder=atom_decoder)
        molecule.build_molecule(atom_decoder=atom_decoder)
        all_mols.append(molecule)

    return all_mols


def evaluate_generated_molecules(molecule_dir):
    """
    评估FuncMol生成的分子质量
    
    :param molecule_dir: 包含npz文件的目录路径
    """
    print("FuncMol生成分子质量评估")
    print("="*50)
    
    # 加载生成的分子
    generated_mols = load_generated_molecules_from_npz(molecule_dir)
    
    if not generated_mols:
        print("没有找到有效的生成分子文件！")
        return
    
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
    sampling_metrics = SimpleSamplingMetrics(train_smiles=[], dataset_infos=dataset_infos, test=True)
    
    # 评估分子
    print("开始评估分子质量...")
    sampling_metrics(molecules=generated_mols, name='generated_molecules', current_epoch=0, local_rank=0)
    
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
    print(f"  分子稳定性: {mol_stable*100:.2f}%")
    print(f"  原子稳定性: {atom_stable*100:.2f}%")
    
    # 分析单个分子
    print(f"\n📈 单个分子分析:")
    valid_molecules = 0
    stable_molecules = 0
    
    for i, mol in enumerate(generated_mols):
        if mol.rdkit_mol is not None:
            valid_molecules += 1
            print(f"  分子 {i+1}: {mol.num_nodes} 个原子, RDKit有效: 是")
            
            # 检查稳定性
            try:
                mol_stable, atom_stable, num_bonds = check_stability(mol, None, atom_decoder=atom_decoder_dict['qm9_with_h'])
                if mol_stable.item() > 0.5:
                    stable_molecules += 1
                    print(f"    稳定性: 稳定 (原子稳定性: {atom_stable.item()/num_bonds*100:.1f}%)")
                else:
                    print(f"    稳定性: 不稳定 (原子稳定性: {atom_stable.item()/num_bonds*100:.1f}%)")
            except Exception as e:
                print(f"    稳定性检查失败: {e}")
        else:
            print(f"  分子 {i+1}: {mol.num_nodes} 个原子, RDKit有效: 否")
    
    print(f"\n📊 统计摘要:")
    print(f"  总分子数: {len(generated_mols)}")
    print(f"  有效分子数: {valid_molecules}")
    print(f"  稳定分子数: {stable_molecules}")
    print(f"  有效性: {valid_molecules/len(generated_mols)*100:.1f}%")
    print(f"  稳定性: {stable_molecules/len(generated_mols)*100:.1f}%")


def open_babel_eval(file: str = None):
    """简化的open_babel评估函数，不依赖MiDi和wandb"""
    print("OpenBabel分子评估（简化版本）")
    print("="*50)
    
    if file is None:
        print("请提供OpenBabel文件路径")
        return
    
    # 简化的数据集信息
    class SimpleDatasetInfo:
        def __init__(self):
            self.atom_decoder = atom_decoder_dict['qm9_with_h']
            self.remove_h = False
    
    dataset_infos = SimpleDatasetInfo()
    
    # 加载OpenBabel分子
    open_babel_mols = open_babel_preprocess(file, 'qm9_with_h')
    
    if not open_babel_mols:
        print("没有找到有效的OpenBabel分子文件！")
        return
    
    # 创建采样指标评估器
    sampling_metrics = SimpleSamplingMetrics(train_smiles=[], dataset_infos=dataset_infos, test=True)
    
    # 评估分子
    sampling_metrics(molecules=open_babel_mols, name='openbabel', current_epoch=-1, local_rank=0)


if __name__ == "__main__":
    # 默认评估FuncMol生成的分子
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--openbabel":
        # 如果指定了--openbabel参数，则评估OpenBabel分子
        open_babel_eval(file=None)
    else:
        # 默认评估FuncMol生成的分子
        molecule_dir = "/home/huayuchen/Neurl-voxel/exps/funcmol/fm_qm9/20250912/molecule"
        evaluate_generated_molecules(molecule_dir)
