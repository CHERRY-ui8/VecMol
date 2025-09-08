import torch
from torch import nn


class MLPResBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_groups: int = 32,
        dropout: float = 0.1,
        bias_free: bool = False,
    ):
        super().__init__()

        # first norm + conv layer
        self.norm1 = nn.GroupNorm(n_groups, in_channels)
        self.act1 = nn.SiLU() if not bias_free else nn.ReLU()
        self.mlp1 = nn.Linear(in_channels, out_channels, bias=not bias_free)

        # second norm + conv layer
        self.norm2 = nn.GroupNorm(n_groups, in_channels)
        self.act2 = nn.SiLU() if not bias_free else nn.ReLU()
        self.mlp2 = nn.Linear(out_channels, out_channels, bias=not bias_free)
        self.mlp2.weight.data.zero_()
        if not bias_free:
            self.mlp2.bias.data.zero_()

        if in_channels != out_channels:
            # self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1)
            self.shortcut = nn.Linear(in_channels, out_channels, bias=not bias_free)
        else:
            self.shortcut = nn.Identity()

        if dropout > 0:
            self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        h = self.norm1(x)
        h = self.act1(h)
        h = self.mlp1(h)

        h = self.norm2(h)
        h = self.act2(h)
        if hasattr(self, "dropout"):
            h = self.dropout(h)
        h = self.mlp2(h)

        return h + self.shortcut(x)


class MLPResCode(nn.Module):
    def __init__(
        self,
        code_dim: int = 1024,
        n_hidden_units: int = 2048,
        num_blocks: int = 4,
        n_groups: int = 32,
        dropout: float = 0.1,
        bias_free: bool = False,
        out_dim: int = None
    ):
        super().__init__()

        self.projection = nn.Linear(code_dim, n_hidden_units, bias=not bias_free)

        # encoder
        enc = []
        for i in range(num_blocks):
            enc.append(MLPResBlock(n_hidden_units, n_hidden_units, n_groups, dropout, bias_free=bias_free))
        self.enc = nn.ModuleList(enc)

        # bottleneck
        self.middle = MLPResBlock(n_hidden_units, n_hidden_units, n_groups, dropout, bias_free=bias_free)

        # decoder
        dec = []
        for i in reversed(range(num_blocks)):
            dec.append(MLPResBlock(n_hidden_units, n_hidden_units, n_groups, dropout, bias_free=bias_free))
        self.dec = nn.ModuleList(dec)

        self.norm = nn.GroupNorm(n_groups, n_hidden_units)
        self.act = nn.SiLU() if not bias_free else nn.ReLU()
        self.final = nn.Linear(n_hidden_units, code_dim if out_dim is None else out_dim, bias=not bias_free)
        self.final.weight.data.zero_()
        if not bias_free:
            self.final.bias.data.zero_()

    def forward(self, x: torch.Tensor):
        # 处理输入维度 - 现在只支持3D输入 [batch_size, n_grid, code_dim]
        if x.dim() == 4:  # [batch_size, 1, n_grid, code_dim]
            x = x.squeeze(1)  # [batch_size, n_grid, code_dim]
        elif x.dim() == 3:  # [batch_size, n_grid, code_dim]
            pass
        else:
            raise ValueError(f"Expected 3D or 4D input, got {x.dim()}D: {x.shape}")
        
        batch_size, n_grid, code_dim = x.shape
        
        # 重塑为2D进行处理
        x_reshaped = x.view(batch_size * n_grid, code_dim)
        
        # 投影
        x_reshaped = self.projection(x_reshaped)

        # encoder
        hidden = [x_reshaped]
        for m in self.enc:
            x_reshaped = m(x_reshaped)
            hidden.append(x_reshaped)

        # bottleneck
        x_reshaped = self.middle(x_reshaped)

        # decoder
        for dec in self.dec:
            hid = hidden.pop()
            x_reshaped = torch.add(x_reshaped, hid)
            x_reshaped = dec(x_reshaped)

        if hasattr(self, "norm"):
            x_reshaped = self.norm(x_reshaped)
        x_reshaped = self.act(x_reshaped)
        x_reshaped = self.final(x_reshaped)
        
        # 重塑回原始维度
        x = x_reshaped.view(batch_size, n_grid, code_dim)

        return x
