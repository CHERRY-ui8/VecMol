import torch
from torch import nn
import torch.nn.functional as F


class ElementPredictor(nn.Module):
    """
    元素存在性预测器
    
    输入: codes [B, grid_size³, code_dim]
    输出: element_existence [B, n_atom_types] (每个元素是否存在，0或1)
    """
    def __init__(self, code_dim: int, grid_size: int, n_atom_types: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.3):
        """
        Args:
            code_dim: 潜在码的维度
            grid_size: 网格大小
            n_atom_types: 原子类型数量
            hidden_dim: 隐藏层维度（小参数量）
            num_layers: 网络层数
            dropout: Dropout率，用于防止过拟合
        """
        super().__init__()
        self.code_dim = code_dim
        self.grid_size = grid_size
        self.n_atom_types = n_atom_types
        self.hidden_dim = hidden_dim
        
        # 首先对codes进行全局池化，得到全局特征
        # 使用平均池化和最大池化的组合
        self.pool = nn.AdaptiveAvgPool1d(1)  # 对grid维度进行池化
        
        # 构建简单的MLP
        layers = []
        input_dim = code_dim * 2  # 平均池化 + 最大池化
        
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Linear(input_dim, hidden_dim))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        
        # 输出层：预测每种元素是否存在
        layers.append(nn.Linear(hidden_dim, n_atom_types))
        
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        """
        Args:
            codes: [B, grid_size³, code_dim]
            
        Returns:
            logits: [B, n_atom_types] (logits，未经过sigmoid，用于BCEWithLogitsLoss)
        """
        B, N, D = codes.shape
        
        # 转置以便进行池化: [B, D, N]
        codes_t = codes.transpose(1, 2)  # [B, code_dim, grid_size³]
        
        # 平均池化
        avg_pooled = self.pool(codes_t).squeeze(-1)  # [B, code_dim]
        
        # 最大池化
        max_pooled = torch.max(codes_t, dim=2)[0]  # [B, code_dim]
        
        # 拼接平均池化和最大池化的结果
        pooled_features = torch.cat([avg_pooled, max_pooled], dim=1)  # [B, code_dim * 2]
        
        # 通过MLP，直接返回logits（不应用sigmoid）
        logits = self.mlp(pooled_features)  # [B, n_atom_types]
        
        return logits


def create_element_predictor(config: dict, n_atom_types: int):
    """
    创建元素存在性预测器
    
    Args:
        config: 配置字典，应包含predictor相关配置
        n_atom_types: 原子类型数量
        
    Returns:
        ElementPredictor实例
    """
    predictor_config = config.get("predictor", {})
    
    predictor = ElementPredictor(
        code_dim=config["decoder"]["code_dim"],
        grid_size=config["dset"]["grid_size"],
        n_atom_types=n_atom_types,
        hidden_dim=predictor_config.get("hidden_dim", 64),
        num_layers=predictor_config.get("num_layers", 2),
        dropout=predictor_config.get("dropout", 0.3)
    )
    
    # 打印参数量
    n_params = sum(p.numel() for p in predictor.parameters() if p.requires_grad)
    print(f">> ElementPredictor has {(n_params/1e3):.02f}K parameters")
    
    return predictor



