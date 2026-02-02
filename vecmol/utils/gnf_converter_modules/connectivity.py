"""
连通性分析模块：分子连通性分析和最大连通分支选择
"""
import numpy as np
import torch
from typing import List, Tuple
from vecmol.utils.constants import PADDING_INDEX


class ConnectivityAnalyzer:
    """连通性分析器"""
    
    def __init__(self, bond_validator):
        """
        初始化连通性分析器
        
        Args:
            bond_validator: BondValidator实例，用于获取键长阈值
        """
        self.bond_validator = bond_validator
    
    def build_molecular_graph(self, coords: np.ndarray, atom_types: np.ndarray) -> np.ndarray:
        """
        根据原子坐标和类型构建分子图的邻接矩阵（基于键长阈值）
        
        Args:
            coords: 原子坐标 [N, 3]
            atom_types: 原子类型 [N]
            
        Returns:
            adjacency_matrix: 邻接矩阵 [N, N]，1表示有键，0表示无键
        """
        n_atoms = len(coords)
        if n_atoms == 0:
            return np.zeros((0, 0), dtype=np.bool_)
        
        adjacency_matrix = np.zeros((n_atoms, n_atoms), dtype=np.bool_)
        
        # 计算所有原子对的距离
        distances = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(axis=2))  # [N, N]
        
        # 构建邻接矩阵
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                distance = distances[i, j]
                threshold = self.bond_validator.get_bond_length_upper_threshold(int(atom_types[i]), int(atom_types[j]))
                
                if distance < threshold:
                    adjacency_matrix[i, j] = True
                    adjacency_matrix[j, i] = True
        
        return adjacency_matrix
    
    def find_connected_components(self, adjacency_matrix: np.ndarray) -> List[List[int]]:
        """
        使用DFS找到所有连通分量
        
        Args:
            adjacency_matrix: 邻接矩阵 [N, N]
            
        Returns:
            components: 连通分量列表，每个分量是一个原子索引列表
        """
        n_atoms = adjacency_matrix.shape[0]
        if n_atoms == 0:
            return []
        
        visited = np.zeros(n_atoms, dtype=bool)
        components = []
        
        def dfs(node: int, component: List[int]):
            """深度优先搜索"""
            visited[node] = True
            component.append(node)
            # 找到所有与当前节点相连的节点
            neighbors = np.where(adjacency_matrix[node])[0]
            for neighbor in neighbors:
                if not visited[neighbor]:
                    dfs(neighbor, component)
        
        # 遍历所有未访问的节点
        for i in range(n_atoms):
            if not visited[i]:
                component = []
                dfs(i, component)
                components.append(component)
        
        return components
    
    def select_largest_connected_component(
        self, 
        coords: torch.Tensor, 
        atom_types: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        选择最大连通分支作为最终结果
        
        Args:
            coords: 原子坐标 [N, 3]
            atom_types: 原子类型 [N]
            
        Returns:
            filtered_coords: 过滤后的原子坐标 [M, 3]，M <= N
            filtered_types: 过滤后的原子类型 [M]
        """
        # 过滤掉填充的原子
        valid_mask = (atom_types != PADDING_INDEX) & (atom_types != -1)
        if valid_mask.sum() == 0:
            return torch.empty(0, 3, device=coords.device), torch.empty(0, dtype=torch.long, device=atom_types.device)
        
        valid_coords = coords[valid_mask].cpu().numpy()
        valid_types = atom_types[valid_mask].cpu().numpy()
        
        if len(valid_coords) == 0:
            return torch.empty(0, 3, device=coords.device), torch.empty(0, dtype=torch.long, device=atom_types.device)
        
        # 构建分子图
        adjacency_matrix = self.build_molecular_graph(valid_coords, valid_types)
        
        # 找到所有连通分量
        components = self.find_connected_components(adjacency_matrix)
        
        if len(components) == 0:
            # 如果没有连通分量，返回空结果
            return torch.empty(0, 3, device=coords.device), torch.empty(0, dtype=torch.long, device=atom_types.device)
        
        # 选择最大的连通分量
        largest_component = max(components, key=len)
        
        # 提取最大连通分量的原子
        filtered_coords = valid_coords[largest_component]
        filtered_types = valid_types[largest_component]
        
        # 转换回torch tensor
        filtered_coords_tensor = torch.from_numpy(filtered_coords).to(coords.device)
        filtered_types_tensor = torch.from_numpy(filtered_types).long().to(atom_types.device)
        
        return filtered_coords_tensor, filtered_types_tensor

