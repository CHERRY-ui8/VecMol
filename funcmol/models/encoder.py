import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, knn_graph
import torch.nn.functional as F
from torch_geometric.loader import DataLoader


class CrossGraphEncoder(nn.Module):
    def __init__(self, n_atom_types, grid_size, code_dim, hidden_dim=128, num_layers=4, k_neighbors=8):
        super().__init__()
        self.n_atom_types = n_atom_types
        self.grid_size = grid_size
        self.code_dim = code_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.k_neighbors = k_neighbors

        # learnable latent code for each grid point (G_L)
        self.grid_codes = nn.Parameter(torch.Tensor(grid_size**3, code_dim))
        # self.grid_codes = nn.Buffer(torch.zeros(grid_size**3, code_dim, requires_grad=True)) # 之前这里用buffer，出现报错element 0 of tensors does not require grad and does not have a grad_fn
        nn.init.xavier_uniform_(self.grid_codes) 

        # GNN layers
        self.layers = nn.ModuleList([
            MessagePassingGNN(n_atom_types, code_dim, hidden_dim)
            for _ in range(num_layers)
        ])

    def forward(self, data):
        """
        data: torch_geometric.data.Batch object
              - data.pos: [N_total_atoms, 3], atom coordinates
              - data.x: [N_total_atoms], atom types
              - data.batch: [N_total_atoms], batch index for each atom
        """
        atom_coords = data.pos
        atoms_channel = data.x  # atom types
        atom_batch_idx = data.batch
        
        device = atom_coords.device
        B = data.num_graphs # 一个batch中包含的分子数量
        N_total_atoms = data.num_nodes
        
        n_grid = self.grid_size ** 3

        # 1. 原子类型 one-hot
        # 验证值范围
        if atoms_channel.numel() > 0:
            assert atoms_channel.min() >= 0, f"Negative values in atoms_channel: {atoms_channel.min()}"
            assert atoms_channel.max() < self.n_atom_types, f"atoms_channel max {atoms_channel.max()} >= n_atom_types {self.n_atom_types}"
        
        atom_feat = F.one_hot(atoms_channel.long(), num_classes=self.n_atom_types).float()  # [N_total_atoms, n_atom_types]
        
        # 2. 构造 grid 坐标
        grid_coords_single = create_grid_coords(device, 1, self.grid_size).squeeze(0)  # [n_grid, 3]
        grid_coords_flat = grid_coords_single.repeat(B, 1)  # [B*n_grid, 3]

        # 3. grid latent code
        grid_codes = self.grid_codes.unsqueeze(0).expand(B, -1, -1).reshape(-1, self.code_dim)  # [B*n_grid, code_dim]

        # 4. 拼接所有节点
        # 确保特征维度匹配
        if atom_feat.size(-1) != self.code_dim:
            if not hasattr(self, 'atom_feat_proj'):
                self.atom_feat_proj = nn.Linear(self.n_atom_types, self.code_dim).to(device)
            atom_feat = self.atom_feat_proj(atom_feat)  # [N_total_atoms, code_dim]

        # 创建 grid 的 batch 索引
        grid_batch_idx = torch.arange(B, device=device).repeat_interleave(n_grid)  # [B*n_grid]

        # 拼接所有节点
        node_feats = torch.cat([atom_feat, grid_codes], dim=0)  # [(N_total_atoms + B*n_grid), code_dim]
        node_pos = torch.cat([atom_coords, grid_coords_flat], dim=0)  # [(N_total_atoms + B*n_grid), 3]
        node_batch = torch.cat([atom_batch_idx, grid_batch_idx], dim=0)  # [(N_total_atoms + B*n_grid)]

        # 5. 建边（KNN，原子和grid点都可互连）
        edge_index = knn_graph(
            x=node_pos, k=self.k_neighbors, batch=node_batch, loop=False
        )

        # 6. GNN消息传递
        h = node_feats
        for layer in self.layers:
            h = layer(h, node_pos, edge_index)

        # 7. 只取 grid 部分并重塑为 [B, n_grid, code_dim]
        grid_h = h[N_total_atoms:].reshape(B, n_grid, self.code_dim)
        return grid_h  # [B, n_grid, code_dim]


class MessagePassingGNN(MessagePassing):
    def __init__(self, atom_feat_dim, code_dim, hidden_dim):
        super().__init__(aggr='mean')
        self.code_dim = code_dim
        self.hidden_dim = hidden_dim
        # 修改MLP的输入维度，使其匹配实际输入
        self.mlp = nn.Sequential(
            nn.Linear(2*code_dim + 1, hidden_dim, bias=True),
            nn.ReLU(),
            nn.Linear(hidden_dim, code_dim, bias=True)
        )
        self.layernorm = nn.LayerNorm(code_dim)
        
        # 确保所有参数都设置了requires_grad=True
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, x, pos, edge_index):
        # x: [N, code_dim], pos: [N, 3]
        row, col = edge_index
        rel = pos[row] - pos[col]  # [E, 3]
        dist = torch.norm(rel, dim=-1, keepdim=True)  # [E, 1]
        
        # 确保数据类型正确
        x = x.float()  # 确保x是float类型
        rel = rel.float()  # 确保rel是float类型
        dist = dist.float()  # 确保dist是float类型
        
        msg_input = torch.cat([x[row], x[col], dist], dim=-1)  # [E, 2*code_dim+1]
        
        msg = self.mlp(msg_input)  # [E, code_dim]
        aggr = self.propagate(edge_index, x=x, message=msg)  # [N, code_dim]
        x = x + aggr  # 残差连接
        x = self.layernorm(x)
        return x

def create_grid_coords(device, batch_size, grid_size):
    """Create grid coordinates for a given grid size."""
    grid_1d = torch.linspace(-1, 1, grid_size, device=device)
    mesh = torch.meshgrid(grid_1d, grid_1d, grid_1d, indexing='ij')
    coords = torch.stack(mesh, dim=-1).reshape(-1, 3)  # [n_grid, 3]
    coords = coords.unsqueeze(0).expand(batch_size, -1, -1)  # [B, n_grid, 3]
    return coords


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels,
        level_channels=[32, 64, 128],
        bottleneck_channel=1024,
        smaller=False
    ):
        super(Encoder, self).__init__()
        self.enc_blocks = nn.ModuleList()
        for i in range(len(level_channels)):
            in_ch = in_channels if i == 0 else level_channels[i - 1]
            out_ch = level_channels[i]
            self.enc_blocks.append(
                Conv3DBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    bottleneck=False,
                    smaller=smaller
                )
            )
        self.bottleNeck = Conv3DBlock(
            in_channels=out_ch,
            out_channels=bottleneck_channel,
            bottleneck=True,
            smaller=smaller
        )
        self.fc = nn.Linear(bottleneck_channel, bottleneck_channel)

    def forward(self, voxels):
        # encoder
        out = voxels
        for block in self.enc_blocks:
            out, _ = block(out)
        out, _ = self.bottleNeck(out)

        # pooling
        out = torch.nn.functional.avg_pool3d(out, out.size()[2:])
        out = out.squeeze()
        out = self.fc(out)

        return out


class SingleConv3D(nn.Module):
    def __init__(self, in_channels, out_channels, use_bn=True, non_linearity=True):
        super(SingleConv3D, self).__init__()

        self.use_bn = use_bn
        self.non_linearity = non_linearity

        self.conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 3, 3),
            padding=1,
        )

        if use_bn:
            self.bn = nn.BatchNorm3d(num_features=out_channels)

        if non_linearity:
            self.nl = nn.ReLU()

    def forward(self, input):
        x = self.conv(input)
        if self.use_bn:
            x = self.bn(x)
        if self.non_linearity:
            x = self.nl(x)
        return x


class Conv3DBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        bottleneck=False,
        use_bn=True,
        res_block=True,
        smaller=False,
    ):
        super(Conv3DBlock, self).__init__()
        self.res_block = res_block

        # first conv
        if smaller:
            level_channels_in = [
                in_channels,
                out_channels,
                out_channels // 2,
            ]
            level_channels_out = [
                out_channels,
                out_channels // 2,
                out_channels,
            ]
        else:
            level_channels_in = [
                out_channels,
                out_channels // 2,
                out_channels // 2,
                out_channels // 2,
                out_channels,
            ]
        self.conv_layers = nn.ModuleList()
        for i in range(len(level_channels_in)):
            if smaller:
                self.conv_layers.append(
                    SingleConv3D(
                        in_channels=level_channels_in[i],
                        out_channels=level_channels_out[i],
                        use_bn=use_bn,
                        non_linearity=(i != len(level_channels_out) - 1),
                    )
                )
            else:
                in_ch = in_channels if i == 0 else level_channels_in[i - 1]
                out_ch = level_channels_in[i]
                self.conv_layers.append(
                    SingleConv3D(
                        in_channels=in_ch,
                        out_channels=out_ch,
                        use_bn=use_bn,
                        non_linearity=(i != len(level_channels_in) - 1),
                    )
                )

        # non linearity
        self.nl = nn.ReLU()

        self.bottleneck = bottleneck
        if not bottleneck:
            self.pooling = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2)

    def forward(self, input):
        res = input
        for i, conv in enumerate(self.conv_layers):
            if i == 0:
                x = conv(res)
                res = x.clone()
            else:
                res = conv(res)

        if self.res_block:
            res += x

        res = self.nl(res)

        if not self.bottleneck:
            out = self.pooling(res)
        else:
            out = res

        return out, res
