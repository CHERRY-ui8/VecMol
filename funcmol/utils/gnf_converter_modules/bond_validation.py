"""
键长验证模块：处理键长相关的验证和阈值计算
"""
import numpy as np
from typing import Optional, Tuple
from funcmol.utils.constants import BOND_LENGTHS_PM, ELEMENTS_HASH_INV, DEFAULT_BOND_LENGTH_THRESHOLD


class BondValidator:
    """键长验证器"""
    
    def __init__(self, bond_length_tolerance: float = 0.4, bond_length_lower_tolerance: float = 0.2, debug: bool = False):
        """
        初始化键长验证器
        
        Args:
            bond_length_tolerance: 键长合理性检查的上限容差（单位：Å），在标准键长基础上增加的容差
            bond_length_lower_tolerance: 键长合理性检查的下限容差（单位：Å），从标准键长中减去的容差
            debug: 是否输出调试信息
        """
        self.bond_length_tolerance = bond_length_tolerance
        self.bond_length_lower_tolerance = bond_length_lower_tolerance
        self.debug = debug
    
    def get_bond_length_upper_threshold(self, atom1_type: int, atom2_type: int) -> float:
        """
        根据两个原子类型返回合理的键长上限阈值（单位：Å）
        
        Args:
            atom1_type: 第一个原子类型索引
            atom2_type: 第二个原子类型索引
            
        Returns:
            键长上限阈值（单位：Å）
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
            # 转换为Å并加上上限容差
            return (bond_length_pm / 100.0) + self.bond_length_tolerance
        else:
            return DEFAULT_BOND_LENGTH_THRESHOLD
    
    def get_bond_length_lower_threshold(self, atom1_type: int, atom2_type: int) -> float:
        """
        根据两个原子类型返回合理的键长下限阈值（单位：Å）
        
        Args:
            atom1_type: 第一个原子类型索引
            atom2_type: 第二个原子类型索引
            
        Returns:
            键长下限阈值（单位：Å）
        """
        atom1_symbol = ELEMENTS_HASH_INV.get(atom1_type, None)
        atom2_symbol = ELEMENTS_HASH_INV.get(atom2_type, None)
        
        # if atom1_symbol is None or atom2_symbol is None:
        #     # 如果没有原子类型信息，返回一个默认的下限值（通常键长不会小于0.5Å）
        #     return 0.5
        
        # 尝试获取键长数据（双向查找）
        bond_length_pm = None
        if atom1_symbol in BOND_LENGTHS_PM and atom2_symbol in BOND_LENGTHS_PM[atom1_symbol]:
            bond_length_pm = BOND_LENGTHS_PM[atom1_symbol][atom2_symbol]
        elif atom2_symbol in BOND_LENGTHS_PM and atom1_symbol in BOND_LENGTHS_PM[atom2_symbol]:
            bond_length_pm = BOND_LENGTHS_PM[atom2_symbol][atom1_symbol]
        
        if bond_length_pm is not None:
            # 转换为Å并减去下限容差，确保结果不为负
            lower_threshold = (bond_length_pm / 100.0) - self.bond_length_lower_tolerance
            return max(0.5, lower_threshold)  # 确保下限至少为0.5Å
        else:
            # 如果没有键长数据，返回一个默认的下限值
            return 0.5
    
    def check_bond_length_validity(
        self, 
        new_point: np.ndarray, 
        new_atom_type: int,
        reference_points: np.ndarray, 
        reference_types: np.ndarray,
        debug: Optional[bool] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        检查新点与所有参考点的键长是否合理
        
        检查逻辑：
        1. 下限检查：所有参考点与新点的距离都必须 >= 下限（防止与任何原子太近）
        2. 上限检查：至少有一个参考点与新点的距离 < 上限（确保有连接）
        
        Args:
            new_point: 新点坐标 [3]
            new_atom_type: 新点原子类型
            reference_points: 参考点坐标 [M, 3]
            reference_types: 参考点原子类型 [M]
            debug: 是否输出调试信息（如果为None，使用self.debug）
            
        Returns:
            (is_valid, rejection_reason): 
            - is_valid: 如果所有参考点都满足下限要求，且至少有一个参考点满足上限要求，返回True；否则返回False
            - rejection_reason: 如果失败，返回失败原因（"下限"或"上限"），否则返回None
        """
        if debug is None:
            debug = self.debug
            
        if len(reference_points) == 0:
            return True, None  # 没有参考点，第一轮聚类，直接通过
        
        # 计算新点与所有参考点的距离
        distances = np.sqrt(((reference_points - new_point[None, :]) ** 2).sum(axis=1))
        
        # 检查是否有距离为0或极小的参考点（说明新原子与参考原子位置相同，可能是重复）
        min_distance_idx = np.argmin(distances)
        min_distance = distances[min_distance_idx]
        
        # 如果距离为0或极小（< 0.01 Å），说明新原子就是参考点本身，应该直接拒绝
        if min_distance < 0.01:
            if debug:
                new_atom_symbol = ELEMENTS_HASH_INV.get(new_atom_type, f"Type{new_atom_type}")
                nearest_ref_symbol = ELEMENTS_HASH_INV.get(reference_types[min_distance_idx], f"Type{reference_types[min_distance_idx]}")
                coord_str = f"({new_point[0]:.3f},{new_point[1]:.3f},{new_point[2]:.3f})"
                print(f"  ✗ [重复原子] {new_atom_symbol} {coord_str}: 与参考原子 {nearest_ref_symbol} 位置相同 (距离={min_distance:.6f}Å)")
            return False, "重复"
        
        nearest_ref_type = reference_types[min_distance_idx]
        nearest_ref_point = reference_points[min_distance_idx]
        nearest_threshold = self.get_bond_length_upper_threshold(new_atom_type, nearest_ref_type)
        
        # 键长检查逻辑：
        # 1. 下限检查：所有参考点与新点的距离都必须 >= 下限（防止与任何原子太近）
        # 2. 上限检查：至少有一个参考点与新点的距离 < 上限（确保有连接）
        
        # 首先检查下限：所有参考点都必须满足距离 >= 下限
        all_satisfy_lower = True
        lower_violations = []  # 记录违反下限的参考点
        for i, (ref_point, ref_type) in enumerate(zip(reference_points, reference_types)):
            distance = distances[i]
            lower_threshold = self.get_bond_length_lower_threshold(new_atom_type, ref_type)
            if distance < lower_threshold:
                all_satisfy_lower = False
                lower_violations.append((i, ref_point, ref_type, distance, lower_threshold))
        
        # 如果下限检查失败，直接返回False
        if not all_satisfy_lower:
            if debug:
                new_atom_symbol = ELEMENTS_HASH_INV.get(new_atom_type, f"Type{new_atom_type}")
                print(f"  ✗ [下限失败] {new_atom_symbol}: {len(lower_violations)} 个参考点距离过近")
                for _, ref_point, ref_type, dist, lower_thresh in lower_violations[:3]:  # 最多显示3个
                    ref_symbol = ELEMENTS_HASH_INV.get(ref_type, f"Type{ref_type}")
                    coord_str = f"({ref_point[0]:.3f},{ref_point[1]:.3f},{ref_point[2]:.3f})"
                    print(f"      {ref_symbol} {coord_str}: {dist:.4f}Å < {lower_thresh:.4f}Å (差{lower_thresh-dist:.4f}Å)")
            return False, "下限"
        
        # 下限检查通过，现在检查上限：至少有一个参考点满足距离 < 上限
        valid_refs = []  # 记录所有满足上限条件的参考点
        for i, (ref_point, ref_type) in enumerate(zip(reference_points, reference_types)):
            distance = distances[i]
            upper_threshold = self.get_bond_length_upper_threshold(new_atom_type, ref_type)
            # 检查是否在合理键长范围内（距离 < 上限，且已经通过了下限检查）
            if distance < upper_threshold:
                lower_threshold = self.get_bond_length_lower_threshold(new_atom_type, ref_type)
                valid_refs.append((i, ref_point, ref_type, distance, lower_threshold, upper_threshold))
        
        if len(valid_refs) > 0:
            # 如果启用调试，输出所有满足条件的参考点
            if debug:
                new_atom_symbol = ELEMENTS_HASH_INV.get(new_atom_type, f"Type{new_atom_type}")
                print(f"  ✓ [通过] {new_atom_symbol}: 找到 {len(valid_refs)} 个有效连接")
                for _, ref_point, ref_type, dist, lower_thresh, upper_thresh in valid_refs[:2]:  # 最多显示2个
                    ref_symbol = ELEMENTS_HASH_INV.get(ref_type, f"Type{ref_type}")
                    coord_str = f"({ref_point[0]:.3f},{ref_point[1]:.3f},{ref_point[2]:.3f})"
                    print(f"      {ref_symbol} {coord_str}: {dist:.4f}Å ∈ [{lower_thresh:.4f}, {upper_thresh:.4f}]Å")
            return True, None  # 下限和上限检查都通过
        
        # 下限检查通过，但没有找到满足上限条件的参考点
        if debug:
            new_atom_symbol = ELEMENTS_HASH_INV.get(new_atom_type, f"Type{new_atom_type}")
            nearest_ref_symbol = ELEMENTS_HASH_INV.get(nearest_ref_type, f"Type{nearest_ref_type}")
            coord_str = f"({new_point[0]:.3f},{new_point[1]:.3f},{new_point[2]:.3f})"
            nearest_coord_str = f"({nearest_ref_point[0]:.3f},{nearest_ref_point[1]:.3f},{nearest_ref_point[2]:.3f})"
            print(f"  ✗ [上限失败] {new_atom_symbol} {coord_str}: 所有参考点距离过远")
            print(f"      最近: {nearest_ref_symbol} {nearest_coord_str}, 距离={min_distance:.4f}Å > 上限={nearest_threshold:.4f}Å (差{min_distance - nearest_threshold:.4f}Å)")
        
        return False, "上限"  # 没有找到合理的键长
    
    def check_bond_length_lower_only(
        self, 
        new_point: np.ndarray, 
        new_atom_type: int,
        reference_points: np.ndarray, 
        reference_types: np.ndarray,
        debug: Optional[bool] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        只检查下限：防止新点与参考点太近（用于第一轮聚类，同一类型原子之间的检查）
        
        检查逻辑：
        1. 下限检查：所有参考点与新点的距离都必须 >= 下限（防止与任何原子太近）
        2. 不检查上限（第一轮时，同一类型的原子可能还没有形成连接）
        
        Args:
            new_point: 新点坐标 [3]
            new_atom_type: 新点原子类型
            reference_points: 参考点坐标 [M, 3]
            reference_types: 参考点原子类型 [M]
            debug: 是否输出调试信息（如果为None，使用self.debug）
            
        Returns:
            (is_valid, rejection_reason): 
            - is_valid: 如果所有参考点都满足下限要求，返回True；否则返回False
            - rejection_reason: 如果失败，返回失败原因（"下限"或"重复"），否则返回None
        """
        if debug is None:
            debug = self.debug
            
        if len(reference_points) == 0:
            return True, None  # 没有参考点，直接通过
        
        # 计算新点与所有参考点的距离
        distances = np.sqrt(((reference_points - new_point[None, :]) ** 2).sum(axis=1))
        
        # 检查是否有距离为0或极小的参考点（说明新原子与参考原子位置相同，可能是重复）
        min_distance_idx = np.argmin(distances)
        min_distance = distances[min_distance_idx]
        
        # 如果距离为0或极小（< 0.01 Å），说明新原子就是参考点本身，应该直接拒绝
        if min_distance < 0.01:
            if debug:
                new_atom_symbol = ELEMENTS_HASH_INV.get(new_atom_type, f"Type{new_atom_type}")
                nearest_ref_symbol = ELEMENTS_HASH_INV.get(reference_types[min_distance_idx], f"Type{reference_types[min_distance_idx]}")
                coord_str = f"({new_point[0]:.3f},{new_point[1]:.3f},{new_point[2]:.3f})"
                print(f"  ✗ [重复原子] {new_atom_symbol} {coord_str}: 与参考原子 {nearest_ref_symbol} 位置相同 (距离={min_distance:.6f}Å)")
            return False, "重复"
        
        # 只检查下限：所有参考点都必须满足距离 >= 下限
        all_satisfy_lower = True
        lower_violations = []  # 记录违反下限的参考点
        for i, (ref_point, ref_type) in enumerate(zip(reference_points, reference_types)):
            distance = distances[i]
            lower_threshold = self.get_bond_length_lower_threshold(new_atom_type, ref_type)
            if distance < lower_threshold:
                all_satisfy_lower = False
                lower_violations.append((i, ref_point, ref_type, distance, lower_threshold))
        
        # 如果下限检查失败，返回False
        if not all_satisfy_lower:
            if debug:
                new_atom_symbol = ELEMENTS_HASH_INV.get(new_atom_type, f"Type{new_atom_type}")
                print(f"  ✗ [下限失败] {new_atom_symbol}: {len(lower_violations)} 个参考点距离过近")
                for _, ref_point, ref_type, dist, lower_thresh in lower_violations[:3]:  # 最多显示3个
                    ref_symbol = ELEMENTS_HASH_INV.get(ref_type, f"Type{ref_type}")
                    coord_str = f"({ref_point[0]:.3f},{ref_point[1]:.3f},{ref_point[2]:.3f})"
                    print(f"      {ref_symbol} {coord_str}: {dist:.4f}Å < {lower_thresh:.4f}Å (差{lower_thresh-dist:.4f}Å)")
            return False, "下限"
        
        # 下限检查通过（第一轮不检查上限）
        if debug:
            new_atom_symbol = ELEMENTS_HASH_INV.get(new_atom_type, f"Type{new_atom_type}")
            print(f"  ✓ [通过下限检查] {new_atom_symbol}: 所有参考点距离满足下限要求")
        
        return True, None

