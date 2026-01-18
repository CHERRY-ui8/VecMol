#!/usr/bin/env python3
"""
测试OpenBabel是否会错误地将距离很远的原子连接起来
"""
import tempfile
import subprocess
from pathlib import Path
from rdkit import Chem
import numpy as np

def create_test_xyz_with_distant_atoms():
    """创建一个包含距离很远的C和Cl原子的测试XYZ文件"""
    # 创建一个合理的分子片段（几个C原子）
    # 然后添加一个距离很远的Cl原子
    
    xyz_content = """9

C  -0.5094    0.9828   -0.7144
C  -1.1267   -2.8133    1.1791
C  -1.4786    1.0539   -2.6381
C   0.2097    2.0635   -0.0357
C  -1.6427   -1.8977    1.9766
C   0.3867   -1.8683    3.2104
C  -0.8787   -1.5542    3.0255
C   0.1028   -1.7993   -1.4760
Cl -6.1920    3.1619    0.0464
"""
    return xyz_content

def test_openbabel_bond_detection():
    """测试OpenBabel的键检测"""
    print("=" * 60)
    print("测试OpenBabel是否会错误连接距离很远的原子")
    print("=" * 60)
    
    # 创建临时目录
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        
        # 创建测试XYZ文件
        xyz_file = tmpdir_path / "test_distant_atoms.xyz"
        xyz_content = create_test_xyz_with_distant_atoms()
        xyz_file.write_text(xyz_content)
        
        print(f"\n创建的测试XYZ文件内容：")
        print(xyz_content)
        
        # 计算C和Cl的距离
        lines = xyz_content.strip().split('\n')
        c_pos = None
        cl_pos = None
        for line in lines[2:]:  # 跳过前两行（原子数和空行）
            parts = line.split()
            if len(parts) >= 4:
                symbol = parts[0]
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                if symbol == 'C' and c_pos is None:
                    c_pos = np.array([x, y, z])
                elif symbol == 'Cl':
                    cl_pos = np.array([x, y, z])
        
        if c_pos is not None and cl_pos is not None:
            distance = np.linalg.norm(c_pos - cl_pos)
            print(f"\nC和Cl之间的距离: {distance:.2f} Å")
            print(f"正常C-Cl键长: ~1.77 Å")
            print(f"距离是正常键长的 {distance/1.77:.1f} 倍")
        
        # 使用OpenBabel转换为SDF
        sdf_file = tmpdir_path / "test_distant_atoms.sdf"
        print(f"\n使用OpenBabel将XYZ转换为SDF...")
        
        try:
            result = subprocess.run(
                ['obabel', str(xyz_file), '-O', str(sdf_file), '-h'],
                capture_output=True,
                text=True,
                check=True
            )
            print("OpenBabel转换成功")
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
            print(f"\n生成的SDF文件内容（前500字符）：")
            print(sdf_content[:500])
            
            # 使用RDKit解析SDF
            mol = Chem.MolFromMolBlock(sdf_content, sanitize=False)
            if mol is not None:
                print(f"\n解析后的分子信息：")
                print(f"  原子数: {mol.GetNumAtoms()}")
                print(f"  键数: {mol.GetNumBonds()}")
                
                # 检查是否有C-Cl键
                has_c_cl_bond = False
                c_cl_bond_length = None
                
                if mol.GetNumConformers() > 0:
                    conf = mol.GetConformer()
                    for bond in mol.GetBonds():
                        begin_idx = bond.GetBeginAtomIdx()
                        end_idx = bond.GetEndAtomIdx()
                        begin_atom = mol.GetAtomWithIdx(begin_idx)
                        end_atom = mol.GetAtomWithIdx(end_idx)
                        
                        begin_symbol = begin_atom.GetSymbol()
                        end_symbol = end_atom.GetSymbol()
                        
                        # 检查是否是C-Cl键
                        if (begin_symbol == 'C' and end_symbol == 'Cl') or \
                           (begin_symbol == 'Cl' and end_symbol == 'C'):
                            has_c_cl_bond = True
                            
                            # 计算键长
                            begin_pos = conf.GetAtomPosition(begin_idx)
                            end_pos = conf.GetAtomPosition(end_idx)
                            distance = np.sqrt(
                                (begin_pos.x - end_pos.x)**2 +
                                (begin_pos.y - end_pos.y)**2 +
                                (begin_pos.z - end_pos.z)**2
                            )
                            c_cl_bond_length = distance
                            
                            print(f"\n⚠️  发现C-Cl键！")
                            print(f"  键长: {distance:.2f} Å")
                            print(f"  正常C-Cl键长: ~1.77 Å")
                            print(f"  这个键长是正常值的 {distance/1.77:.1f} 倍")
                            print(f"  结论: OpenBabel错误地连接了距离很远的C和Cl原子！")
                            break
                
                if not has_c_cl_bond:
                    print(f"\n✓ 未发现C-Cl键，OpenBabel没有错误连接")
                
                # 列出所有键
                print(f"\n所有键的信息：")
                if mol.GetNumConformers() > 0:
                    conf = mol.GetConformer()
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
                        
                        print(f"  键 {i+1}: {begin_atom.GetSymbol()}({begin_idx}) - {end_atom.GetSymbol()}({end_idx}), "
                              f"键长: {distance:.2f} Å, 键型: {bond.GetBondType()}")
        else:
            print("错误: SDF文件未生成")

if __name__ == "__main__":
    test_openbabel_bond_detection()
