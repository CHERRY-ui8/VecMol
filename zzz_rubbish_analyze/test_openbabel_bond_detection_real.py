#!/usr/bin/env python3
"""
测试OpenBabel是否会错误地将距离很远的原子连接起来
使用更接近实际情况的测试（包含更多原子）
"""
import tempfile
import subprocess
from pathlib import Path
from rdkit import Chem
import numpy as np

def create_test_xyz_from_real_case():
    """基于真实案例创建测试XYZ文件"""
    # 从真实的SDF文件中提取原子坐标
    # 包含一个距离很远的Cl原子（原子47）和Br原子（原子48）
    
    xyz_content = """44

C  -0.5094    0.9828   -0.7144
C  -1.1267   -2.8133    1.1791
C  -1.4786    1.0539   -2.6381
C   0.2097    2.0635   -0.0357
C   0.6661   -0.5619    5.1114
C  -1.6427   -1.8977    1.9766
C   3.9174   -0.8844   -3.1178
C   0.3867   -1.8683    3.2104
C  -0.8787   -1.5542    3.0255
C   3.8426   -0.0027   -3.9169
C   2.7753    0.6024   -4.0195
C   2.8677   -1.5082   -2.4412
C   1.0469   -2.8386    2.3717
C   0.1028   -1.7993   -1.4760
C  -0.9653    0.3080   -1.5723
C   1.5923    0.0485   -3.3134
C   0.2842   -3.3623    1.3703
C   1.5720   -0.8419   -2.6356
H  -2.1942    1.6356   -2.4899
H   2.7068    1.4167   -4.6801
H   2.0898   -3.1122    2.5011
H   1.1456    1.9967   -0.2688
H  -0.2264   -0.9067    5.5002
H  -2.6914   -1.4629    1.8149
H   3.0080   -2.3952   -1.7598
H   4.8800   -1.2759   -2.9620
H   0.1903    0.3217    4.8517
H  -1.6966    0.3874   -3.4783
H  -1.3622   -0.9029    3.7277
H   4.7164    0.2907   -4.4691
H  -3.2055   -1.1142   -1.3017
H   1.3752   -0.4257    5.9004
H   0.7134   -4.0492    0.6688
H  -0.0747    2.6129    0.7365
H  -0.9786    1.9464   -2.8425
O   0.4950   -2.7757   -1.1087
O   1.1890   -1.3614    4.2286
O   0.9700    0.6702   -3.5132
O  -3.6676   -3.4850    0.2043
O  -1.8337   -3.8714   -1.2183
N  -2.4855   -1.4659   -1.0050
S  -2.2692   -3.1064   -0.1790
Cl -6.1920    3.1619    0.0464
Br -6.1908    3.1677    0.0419
"""
    return xyz_content

def test_openbabel_bond_detection():
    """测试OpenBabel的键检测"""
    print("=" * 60)
    print("测试OpenBabel是否会错误连接距离很远的原子（真实案例）")
    print("=" * 60)
    
    # 创建临时目录
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        
        # 创建测试XYZ文件
        xyz_file = tmpdir_path / "test_distant_atoms_real.xyz"
        xyz_content = create_test_xyz_from_real_case()
        xyz_file.write_text(xyz_content)
        
        print(f"\n创建的测试XYZ文件包含 {xyz_content.strip().split()[0]} 个原子")
        
        # 计算C和Cl、C和Br的距离
        lines = xyz_content.strip().split('\n')
        atoms = []
        for line in lines[2:]:  # 跳过前两行（原子数和空行）
            parts = line.split()
            if len(parts) >= 4:
                symbol = parts[0]
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                atoms.append((symbol, np.array([x, y, z])))
        
        # 找到Cl和Br原子
        cl_atom = None
        br_atom = None
        c_atoms = []
        for i, (symbol, pos) in enumerate(atoms):
            if symbol == 'Cl':
                cl_atom = (i, pos)
            elif symbol == 'Br':
                br_atom = (i, pos)
            elif symbol == 'C':
                c_atoms.append((i, pos))
        
        print(f"\n找到 {len(c_atoms)} 个C原子，1个Cl原子，1个Br原子")
        
        # 计算最近的C-Cl和C-Br距离
        if cl_atom and c_atoms:
            min_c_cl_dist = min([np.linalg.norm(c_pos - cl_atom[1]) for _, c_pos in c_atoms])
            print(f"\n最近的C-Cl距离: {min_c_cl_dist:.2f} Å")
            print(f"正常C-Cl键长: ~1.77 Å")
            print(f"距离是正常键长的 {min_c_cl_dist/1.77:.1f} 倍")
        
        if br_atom and c_atoms:
            min_c_br_dist = min([np.linalg.norm(c_pos - br_atom[1]) for _, c_pos in c_atoms])
            print(f"\n最近的C-Br距离: {min_c_br_dist:.2f} Å")
            print(f"正常C-Br键长: ~1.94 Å")
            print(f"距离是正常键长的 {min_c_br_dist/1.94:.1f} 倍")
        
        # 使用OpenBabel转换为SDF
        sdf_file = tmpdir_path / "test_distant_atoms_real.sdf"
        print(f"\n使用OpenBabel将XYZ转换为SDF...")
        
        try:
            result = subprocess.run(
                ['obabel', str(xyz_file), '-O', str(sdf_file), '-h'],
                capture_output=True,
                text=True,
                check=True
            )
            print("OpenBabel转换成功")
            if result.stdout:
                print(f"OpenBabel stdout: {result.stdout[:200]}")
            if result.stderr:
                print(f"OpenBabel stderr: {result.stderr[:200]}")
        except subprocess.CalledProcessError as e:
            print(f"OpenBabel转换失败: {e}")
            print(f"stdout: {e.stdout}")
            print(f"stderr: {e.stderr}")
            return
        except FileNotFoundError:
            print("错误: 未找到obabel命令，请确保OpenBabel已安装并在PATH中")
            return
        
        # 读取SDF文件
        if sdf_file.exists():
            sdf_content = sdf_file.read_text()
            print(f"\nSDF文件大小: {len(sdf_content)} 字符")
            
            # 使用RDKit解析SDF
            print("正在解析SDF文件...")
            mol = Chem.MolFromMolBlock(sdf_content, sanitize=False)
            print(f"解析完成，mol对象: {mol is not None}")
            if mol is not None:
                print(f"\n解析后的分子信息：")
                print(f"  原子数: {mol.GetNumAtoms()}")
                print(f"  键数: {mol.GetNumBonds()}")
                
                # 检查是否有C-Cl和C-Br键
                has_c_cl_bond = False
                has_c_br_bond = False
                c_cl_bond_length = None
                c_br_bond_length = None
                
                if mol.GetNumConformers() > 0:
                    conf = mol.GetConformer()
                    for bond in mol.GetBonds():
                        begin_idx = bond.GetBeginAtomIdx()
                        end_idx = bond.GetEndAtomIdx()
                        begin_atom = mol.GetAtomWithIdx(begin_idx)
                        end_atom = mol.GetAtomWithIdx(end_idx)
                        
                        begin_symbol = begin_atom.GetSymbol()
                        end_symbol = end_atom.GetSymbol()
                        
                        # 计算键长
                        begin_pos = conf.GetAtomPosition(begin_idx)
                        end_pos = conf.GetAtomPosition(end_idx)
                        distance = np.sqrt(
                            (begin_pos.x - end_pos.x)**2 +
                            (begin_pos.y - end_pos.y)**2 +
                            (begin_pos.z - end_pos.z)**2
                        )
                        
                        # 检查是否是C-Cl键
                        if (begin_symbol == 'C' and end_symbol == 'Cl') or \
                           (begin_symbol == 'Cl' and end_symbol == 'C'):
                            has_c_cl_bond = True
                            c_cl_bond_length = distance
                            print(f"\n⚠️  发现C-Cl键！")
                            print(f"  原子索引: {begin_idx} ({begin_symbol}) - {end_idx} ({end_symbol})")
                            print(f"  键长: {distance:.2f} Å")
                            print(f"  正常C-Cl键长: ~1.77 Å")
                            print(f"  这个键长是正常值的 {distance/1.77:.1f} 倍")
                        
                        # 检查是否是C-Br键
                        if (begin_symbol == 'C' and end_symbol == 'Br') or \
                           (begin_symbol == 'Br' and end_symbol == 'C'):
                            has_c_br_bond = True
                            c_br_bond_length = distance
                            print(f"\n⚠️  发现C-Br键！")
                            print(f"  原子索引: {begin_idx} ({begin_symbol}) - {end_idx} ({end_symbol})")
                            print(f"  键长: {distance:.2f} Å")
                            print(f"  正常C-Br键长: ~1.94 Å")
                            print(f"  这个键长是正常值的 {distance/1.94:.1f} 倍")
                
                if not has_c_cl_bond and not has_c_br_bond:
                    print(f"\n✓ 未发现C-Cl或C-Br键，OpenBabel没有错误连接")
                elif has_c_cl_bond or has_c_br_bond:
                    print(f"\n❌ 结论: OpenBabel错误地连接了距离很远的原子！")
                
                # 列出所有异常长的键（>3Å）
                print(f"\n所有异常长的键（>3Å）：")
                if mol.GetNumConformers() > 0:
                    conf = mol.GetConformer()
                    long_bonds = []
                    for i, bond in enumerate(mol.GetBonds()):
                        begin_idx = bond.GetBeginAtomIdx()
                        end_idx = bond.GetEndAtomIdx()
                        begin_atom = mol.GetAtomWithIdx(begin_idx)
                        end_atom = mol.GetAtomWithIdx(end_idx)
                        
                        begin_pos = conf.GetAtomPosition(begin_idx)
                        end_pos = conf.GetAtomPosition(end_idx)
                        distance = np.sqrt(
                            (begin_pos.x - end_pos.x)**2 +
                            (begin_pos.y - end_pos.y)**2 +
                            (begin_pos.z - end_pos.z)**2
                        )
                        
                        if distance > 3.0:
                            long_bonds.append((i, begin_atom.GetSymbol(), begin_idx, 
                                             end_atom.GetSymbol(), end_idx, distance))
                    
                    if long_bonds:
                        for i, begin_sym, begin_idx, end_sym, end_idx, dist in long_bonds:
                            print(f"  键 {i+1}: {begin_sym}({begin_idx}) - {end_sym}({end_idx}), "
                                  f"键长: {dist:.2f} Å")
                    else:
                        print("  无")
        else:
            print("错误: SDF文件未生成")

if __name__ == "__main__":
    test_openbabel_bond_detection()
