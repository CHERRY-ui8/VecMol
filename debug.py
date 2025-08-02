import torch
from funcmol.models.encoder import Encoder
from funcmol.models.decoder import Decoder
from funcmol.dataset.dataset_field import FieldDataset
import yaml

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
decoder_config = {
    "grid_size": grid_dim,
    "hidden_dim": 128,         # 你可以根据实际模型调整
    "n_layers": 4,
    "k_neighbors": 8,          # 你可以根据实际模型调整
    "n_channels": n_atom_types,
    "code_dim": bottleneck_channel,
}
decoder = Decoder(decoder_config)

# 5. 构造空间点 (batch, n_points, 3)
n_points = 10
x = torch.rand(batch_size, n_points, 3) * 2 - 1  # [-1, 1] 区间

# 6. Decoder 推理
codes = torch.randn(batch_size, grid_dim**3, bottleneck_channel)
output = decoder(x, codes)
print("Decoder output shape:", output.shape)  # 应为 [batch_size, n_points, n_atom_types, 3]
print("Sample output:", output[0, 0])         # 打印第一个点的所有原子类型的矢量

# 加载配置
with open('funcmol/configs/dset/qm9.yaml', 'r') as f:
    config = yaml.safe_load(f)

# 构造数据集
qm9_dataset = FieldDataset(
    dset_name=config['dset_name'],
    data_dir=config['data_dir'],
    elements=config['elements'],
    split='train',
    n_points=config['n_points'],
    rotate=config['data_aug'],
    grid_spacing=config['grid_spacing'],
    grid_dim=config['grid_dim'],
    radius=config['atomic_radius'],
)

all_min = torch.full((3,), float('inf'))
all_max = torch.full((3,), float('-inf'))

for i in range(100):
    sample = qm9_dataset[i]
    coords = sample['coords']
    mask = coords[:, 0] != 99  # PADDING_INDEX = 99
    coords = coords[mask]
    minv = coords.min(dim=0).values
    maxv = coords.max(dim=0).values
    print(f"Mol {i}: min {minv.tolist()}, max {maxv.tolist()}")
    all_min = torch.min(all_min, minv)
    all_max = torch.max(all_max, maxv)

print(f"\nGlobal min: {all_min.tolist()}, Global max: {all_max.tolist()}")