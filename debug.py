import torch
from funcmol.models.encoder import Encoder
from funcmol.models.decoder import Decoder

# 假设参数
n_atom_types = 5
n_channels = n_atom_types
level_channels = [256, 512, 1024]
bottleneck_channel = 1024  # code_dim
grid_dim = 32
batch_size = 2

# 1. 构造 Encoder
encoder = Encoder(
    in_channels=n_channels,
    level_channels=level_channels,
    bottleneck_channel=bottleneck_channel,
    smaller=False
)

# 2. 构造一个模拟分子体素输入 (batch, channels, D, H, W)
voxels = torch.rand(batch_size, n_channels, grid_dim, grid_dim, grid_dim)

# 3. 得到潜在码 z
z = encoder(voxels)  # shape: [batch_size, code_dim]
print("Encoder output z shape:", z.shape)

# 4. 构造 Decoder
decoder = Decoder(
    n_channels=n_atom_types * 3,  # 输出矢量场
    hidden_dim=128,
    code_dim=bottleneck_channel,
    coord_dim=3,
    n_layers=4,
    input_scale=64,
    grid_dim=grid_dim,
    fabric=type('dummy', (), {'device': 'cpu', 'print': print})()
)

# 5. 构造空间点 (batch, n_points, 3)
n_points = 10
x = torch.rand(batch_size, n_points, 3) * 2 - 1  # [-1, 1] 区间

# 6. Decoder 推理
output = decoder(x, z)
print("Decoder output shape:", output.shape)  # 应为 [batch_size, n_points, n_atom_types, 3]
print("Sample output:", output[0, 0])         # 打印第一个点的所有原子类型的矢量