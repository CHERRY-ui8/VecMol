"""
键长验证模块：处理键长相关的验证和阈值计算
"""
import numpy as np
from typing import Optional
from funcmol.utils.constants import BOND_LENGTHS_PM, ELEMENTS_HASH_INV, DEFAULT_BOND_LENGTH_THRESHOLD


class BondValidator:
    """键长验证器"""
    
    def __init__(self, bond_length_tolerance: float = 0.4, debug: bool = False):
        """
        初始化键长验证器
        
        Args:
            bond_length_tolerance: 键长合理性检查的容差（单位：Å）
            debug: 是否输出调试信息
        """
        self.bond_length_tolerance = bond_length_tolerance
        self.debug = debug
    
    def get_bond_length_threshold(self, atom1_type: int, atom2_type: int) -> float:
        """
        根据两个原子类型返回合理的键长阈值（单位：Å）
        
        Args:
            atom1_type: 第一个原子类型索引
            atom2_type: 第二个原子类型索引
            
        Returns:
            键长阈值（单位：Å）
        """
        atom1_symbol = ELEMENTS_HASH_INV.get(atom1_type, None)
        atom2_symbol = ELEMENTS_HASH_INV.get(atom2_type, None)
        
        if atom1_symbol is None or atom2_symbol is None:
            return DEFAULT_BOND_LENGTH_THRESHOLD
        
        # 尝试获取键长数据（双向查找）
        bond_length_pm = None
        if atom1_symbol in BOND_LENGTHS_PM and atom2_symbol in BOND_LENGTHS_PM[atom1_symbol]:
            bond_length_pm = BOND_LENGTHS_PM[atom1_symbol][atom2_symbol]
        elif atom2_symbol in BOND_LENGTHS_PM and atom1_symbol in BOND_LENGTHS_PM[atom2_symbol]:
            bond_length_pm = BOND_LENGTHS_PM[atom2_symbol][atom1_symbol]
        
        if bond_length_pm is not None:
            # 转换为Å并加上容差
            return (bond_length_pm / 100.0) + self.bond_length_tolerance
        else:
            return DEFAULT_BOND_LENGTH_THRESHOLD
    
    def check_bond_length_validity(
        self, 
        new_point: np.ndarray, 
        new_atom_type: int,
        reference_points: np.ndarray, 
        reference_types: np.ndarray,
        debug: Optional[bool] = None
    ) -> bool:
        """
        检查新点与所有参考点的键长是否合理
        
        Args:
            new_point: 新点坐标 [3]
            new_atom_type: 新点原子类型
            reference_points: 参考点坐标 [M, 3]
            reference_types: 参考点原子类型 [M]
            debug: 是否输出调试信息（如果为None，使用self.debug）
            
        Returns:
            如果存在至少一个参考点使得键长在合理范围内，返回True；否则返回False
        """
        if debug is None:
            debug = self.debug
            
        if len(reference_points) == 0:
            return True  # 没有参考点，第一轮聚类，直接通过
        
        # 计算新点与所有参考点的距离
        distances = np.sqrt(((reference_points - new_point[None, :]) ** 2).sum(axis=1))
        
        # 找到最近参考点的索引
        min_distance_idx = np.argmin(distances)
        min_distance = distances[min_distance_idx]
        nearest_ref_type = reference_types[min_distance_idx]
        nearest_ref_point = reference_points[min_distance_idx]
        nearest_threshold = self.get_bond_length_threshold(new_atom_type, nearest_ref_type)
        
        # 检查是否至少有一个参考点使得键长合理
        valid_refs = []  # 记录所有满足键长条件的参考点
        for i, (ref_point, ref_type) in enumerate(zip(reference_points, reference_types)):
            distance = distances[i]
            threshold = self.get_bond_length_threshold(new_atom_type, ref_type)
            if distance < threshold:
                valid_refs.append((i, ref_point, ref_type, distance, threshold))
        
        if len(valid_refs) > 0:
            # 如果启用调试，输出所有满足条件的参考点
            if debug:
                new_atom_symbol = ELEMENTS_HASH_INV.get(new_atom_type, f"Type{new_atom_type}")
                print(f"[键长检查通过] 新原子: {new_atom_symbol} {new_point}")
                print(f"  找到 {len(valid_refs)} 个满足键长条件的参考点:")
                for idx, ref_point, ref_type, dist, thresh in valid_refs[:5]:  # 最多显示5个
                    ref_symbol = ELEMENTS_HASH_INV.get(ref_type, f"Type{ref_type}")
                    print(f"    {ref_symbol} {ref_point}: 距离={dist:.4f} Å, 阈值={thresh:.4f} Å")
            return True  # 找到至少一个合理的键长
        
        # 如果没有找到合理的键长，输出调试信息
        if debug:
            new_atom_symbol = ELEMENTS_HASH_INV.get(new_atom_type, f"Type{new_atom_type}")
            nearest_ref_symbol = ELEMENTS_HASH_INV.get(nearest_ref_type, f"Type{nearest_ref_type}")
            print(f"[键长检查失败] 新原子: {new_atom_symbol} {new_point}")
            print(f"  最近参考原子: {nearest_ref_symbol} {nearest_ref_point}")
            print(f"  距离: {min_distance:.4f} Å, 阈值: {nearest_threshold:.4f} Å")
            print(f"  差值: {min_distance - nearest_threshold:.4f} Å (需要增加容差)")
            
            # 输出所有参考点的信息（最多10个最近的，以便找到可能满足条件的参考点）
            sorted_indices = np.argsort(distances)[:10]
            print(f"  前10个最近参考点:")
            for idx in sorted_indices:
                ref_symbol = ELEMENTS_HASH_INV.get(reference_types[idx], f"Type{reference_types[idx]}")
                dist = distances[idx]
                thresh = self.get_bond_length_threshold(new_atom_type, reference_types[idx])
                status = "✓ 满足" if dist < thresh else "✗ 不满足"
                print(f"    {status} {ref_symbol}: 距离={dist:.4f} Å, 阈值={thresh:.4f} Å, 差值={dist-thresh:.4f} Å")
        
        return False  # 没有找到合理的键长

