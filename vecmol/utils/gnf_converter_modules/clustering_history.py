"""
历史记录模块：聚类历史的保存和加载
"""
import numpy as np
from pathlib import Path
from typing import List, Optional, Union
from vecmol.utils.gnf_converter_modules.dataclasses import ClusteringHistory
from vecmol.utils.constants import ELEMENTS_HASH_INV


class ClusteringHistorySaver:
    """聚类历史保存器"""
    
    def __init__(self, n_atom_types: int = 5):
        """
        初始化历史记录保存器
        
        Args:
            n_atom_types: 原子类型数量
        """
        self.n_atom_types = n_atom_types
    
    def save_clustering_history(
        self,
        histories: List[ClusteringHistory],
        output_dir: str,
        batch_idx: Union[int, str] = 0,
        elements: Optional[List[str]] = None
    ) -> None:
        """
        保存聚类历史为SDF文件（每轮一个分子）和文本文件（详细记录）
        
        Args:
            histories: 聚类历史列表
            output_dir: 输出目录
            batch_idx: batch索引
            elements: 原子类型符号列表，如 ["C", "H", "O", "N", "F"]
        """
        if elements is None:
            # 默认元素列表
            elements = [ELEMENTS_HASH_INV.get(i, f"Type{i}") for i in range(self.n_atom_types)]
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 合并所有原子类型保存到一个文件
        if len(histories) > 0:
            # 1. 保存为SDF文件（所有类型合并），使用有意义的标识符
            if isinstance(batch_idx, int):
                sdf_path = output_path / f"sample_{batch_idx:04d}_clustering_history.sdf"
                txt_path = output_path / f"sample_{batch_idx:04d}_clustering_history.txt"
            else:
                sdf_path = output_path / f"{batch_idx}_clustering_history.sdf"
                txt_path = output_path / f"{batch_idx}_clustering_history.txt"
            
            self._save_clustering_history_sdf(histories, sdf_path, elements)
            
            # 2. 保存为文本文件（所有类型合并）
            self._save_clustering_history_txt(histories, txt_path, elements)
    
    def _save_clustering_history_sdf(
        self,
        histories: List[ClusteringHistory],
        output_path: Path,
        elements: List[str]
    ) -> None:
        """保存所有原子类型的聚类历史为SDF文件，每轮一个分子（所有类型合并）"""
        try:
            from vecmol.sample_diffusion import xyz_to_sdf
            
            sdf_strings = []
            
            # 找到所有迭代轮数的最大值
            max_iterations = max(len(h.iterations) for h in histories) if histories else 0
            
            # 按迭代轮数组织数据
            for iter_idx in range(max_iterations):
                # 收集这一轮所有类型的新原子
                all_coords_this_iter = []
                all_types_this_iter = []
                type_info = []
                
                for history in histories:
                    if iter_idx < len(history.iterations):
                        record = history.iterations[iter_idx]
                        if len(record.new_atoms_coords) > 0:
                            all_coords_this_iter.append(record.new_atoms_coords)
                            all_types_this_iter.append(record.new_atoms_types)
                            
                            atom_symbol = elements[history.atom_type] if history.atom_type < len(elements) else f"Type{history.atom_type}"
                            type_info.append(f"{atom_symbol}:{record.n_atoms_clustered}")
                
                # 如果有新原子，保存这一轮
                if len(all_coords_this_iter) > 0:
                    combined_coords = np.vstack(all_coords_this_iter)
                    combined_types = np.concatenate(all_types_this_iter)
                    
                    sdf_str = xyz_to_sdf(combined_coords, combined_types, elements)
                    
                    # 修改SDF标题（第一行），包含迭代信息和所有类型信息
                    lines = sdf_str.split('\n')
                    type_info_str = ", ".join(type_info)
                    # 获取第一个记录的eps和min_samples（通常同一轮所有类型使用相同参数）
                    first_record = next((h.iterations[iter_idx] for h in histories if iter_idx < len(h.iterations)), None)
                    if first_record:
                        # SDF格式：第一行是标题行（最多80字符）
                        title = f"Clustering Iter {iter_idx}, eps={first_record.eps:.4f}, min_samples={first_record.min_samples}, atoms={len(combined_coords)} ({type_info_str})"
                        title = title[:80].ljust(80)  # 限制长度并填充到80字符
                        lines[0] = title
                    else:
                        title = f"Clustering Iter {iter_idx}, atoms={len(combined_coords)} ({type_info_str})"
                        title = title[:80].ljust(80)
                        lines[0] = title
                    sdf_strings.append('\n'.join(lines))
            
            # 收集所有迭代轮次的所有原子（最终结果）
            all_final_coords = []
            all_final_types = []
            final_type_info = []
            
            for history in histories:
                # 收集该类型所有迭代轮次的所有原子
                type_coords = []
                type_types = []
                for record in history.iterations:
                    if len(record.new_atoms_coords) > 0:
                        type_coords.append(record.new_atoms_coords)
                        type_types.append(record.new_atoms_types)
                
                if len(type_coords) > 0:
                    # 合并该类型的所有原子
                    combined_type_coords = np.vstack(type_coords)
                    combined_type_types = np.concatenate(type_types)
                    all_final_coords.append(combined_type_coords)
                    all_final_types.append(combined_type_types)
                    
                    atom_symbol = elements[history.atom_type] if history.atom_type < len(elements) else f"Type{history.atom_type}"
                    final_type_info.append(f"{atom_symbol}:{len(combined_type_coords)}")
            
            # 如果有最终结果，创建最终结果的SDF
            if len(all_final_coords) > 0:
                final_combined_coords = np.vstack(all_final_coords)
                final_combined_types = np.concatenate(all_final_types)
                
                final_sdf_str = xyz_to_sdf(final_combined_coords, final_combined_types, elements)
                
                # 修改SDF标题
                lines = final_sdf_str.split('\n')
                final_type_info_str = ", ".join(final_type_info)
                title = f"Final Result, Total atoms={len(final_combined_coords)} ({final_type_info_str})"
                title = title[:80].ljust(80)  # 限制长度并填充到80字符
                lines[0] = title
                sdf_strings.append('\n'.join(lines))
            
            # 写入文件（包含所有迭代轮次和最终结果）
            if sdf_strings:
                with open(output_path, 'w') as f:
                    f.write(''.join(sdf_strings))
        except Exception as e:
            print(f"Warning: Failed to save clustering history SDF: {e}")
    
    def _save_clustering_history_txt(
        self,
        histories: List[ClusteringHistory],
        output_path: Path,
        elements: List[str]
    ) -> None:
        """保存所有原子类型的聚类历史为文本文件，包含详细信息（所有类型合并）"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("自回归聚类过程详细记录（所有原子类型）\n")
                f.write("=" * 80 + "\n\n")
                
                # 总体信息
                total_atoms = sum(h.total_atoms for h in histories)
                max_iterations = max(len(h.iterations) for h in histories) if histories else 0
                f.write(f"总迭代轮数: {max_iterations}\n")
                f.write(f"最终原子总数: {total_atoms}\n")
                f.write("-" * 80 + "\n\n")
                
                # 按迭代轮数组织
                for iter_idx in range(max_iterations):
                    f.write(f"迭代轮数: {iter_idx}\n")
                    
                    # 收集这一轮所有类型的信息
                    type_records = []
                    for history in histories:
                        if iter_idx < len(history.iterations):
                            record = history.iterations[iter_idx]
                            atom_symbol = elements[history.atom_type] if history.atom_type < len(elements) else f"Type{history.atom_type}"
                            type_records.append((atom_symbol, history.atom_type, record))
                    
                    if type_records:
                        # 使用第一个记录的阈值（通常同一轮所有类型使用相同参数）
                        first_record = type_records[0][2]
                        f.write(f"  阈值: eps={first_record.eps:.4f}, min_samples={first_record.min_samples}\n")
                        
                        # 汇总所有类型的信息
                        total_clusters = sum(r[2].n_clusters_found for r in type_records)
                        total_atoms_clustered = sum(r[2].n_atoms_clustered for r in type_records)
                        total_noise = sum(r[2].n_noise_points for r in type_records)
                        
                        f.write(f"  总簇数: {total_clusters}\n")
                        f.write(f"  总聚类原子数: {total_atoms_clustered}\n")
                        f.write(f"  总噪声点数: {total_noise}\n")
                        f.write(f"  键长检查通过: {all(r[2].bond_validation_passed for r in type_records)}\n")
                        f.write("\n")
                        
                        # 按类型详细列出
                        for atom_symbol, atom_type_idx, record in type_records:
                            f.write(f"  类型 {atom_symbol}:\n")
                            f.write(f"    找到簇数: {record.n_clusters_found}\n")
                            f.write(f"    聚类原子数: {record.n_atoms_clustered}\n")
                            f.write(f"    噪声点数: {record.n_noise_points}\n")
                            
                            if len(record.new_atoms_coords) > 0:
                                f.write(f"    新聚类原子坐标:\n")
                                for i, (coord, atom_type) in enumerate(zip(record.new_atoms_coords, record.new_atoms_types)):
                                    atom_sym = elements[atom_type] if atom_type < len(elements) else f"Type{atom_type}"
                                    f.write(f"      {i+1}. {atom_sym}: ({coord[0]:.4f}, {coord[1]:.4f}, {coord[2]:.4f})\n")
                            else:
                                f.write(f"    本轮无新原子被聚类\n")
                            f.write("\n")
                    else:
                        f.write("  本轮无任何类型产生新原子\n")
                    
                    f.write("\n")
                
                f.write("=" * 80 + "\n\n")
        except Exception as e:
            print(f"Warning: Failed to save clustering history text: {e}")

