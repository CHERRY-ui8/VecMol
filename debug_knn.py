import torch
import torch_geometric
from torch_geometric.nn.pool import knn

# 模拟实际的调用链
batch_size = 1
n_points = 3000  # 实际的n_query_points值
grid_size = 8
n_grid = grid_size**3  # 512
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 模拟z_dict[atom_type]的形状
z = torch.randn(n_points, 3, device=device)  # [3000, 3]

# 模拟predicted_field_func中的处理
if z.dim() == 2:  # [n_points, 3]
    z = z.unsqueeze(0)  # [1, n_points, 3]
elif z.dim() == 3:  # [batch, n_points, 3]
    pass
else:
    raise ValueError(f"Unexpected points shape: {z.shape}")

print(f"z after unsqueeze shape: {z.shape}")

# 模拟egnn.py中的处理
query_points = z.reshape(-1, 3).float()  # [B * N, 3] = [3000, 3]
grid_points = torch.randn(n_grid, 3, device=device)  # [512, 3]
grid_coords = grid_points.unsqueeze(0).expand(batch_size, -1, -1).reshape(-1, 3)  # [B * grid_size**3, 3] = [512, 3]

# 创建batch索引
query_batch = torch.arange(batch_size, device=device).repeat_interleave(n_points)  # [3000]
grid_batch = torch.arange(batch_size, device=device).repeat_interleave(n_grid)  # [512]

print(f"query_points shape: {query_points.shape}")
print(f"grid_coords shape: {grid_coords.shape}")
print(f"query_batch shape: {query_batch.shape}")
print(f"grid_batch shape: {grid_batch.shape}")
print(f"query_points.size(0): {query_points.size(0)}")
print(f"query_batch.numel(): {query_batch.numel()}")
print(f"grid_coords.size(0): {grid_coords.size(0)}")
print(f"grid_batch.numel(): {grid_batch.numel()}")

# 检查断言条件
print(f"\n断言检查:")
print(f"y.size(0) == batch_y.numel(): {query_points.size(0)} == {query_batch.numel()} = {query_points.size(0) == query_batch.numel()}")
print(f"x.size(0) == batch_x.numel(): {grid_coords.size(0)} == {grid_batch.numel()} = {grid_coords.size(0) == grid_batch.numel()}")

# 尝试调用knn
try:
    edge_grid_query = knn(
        x=grid_coords,
        y=query_points,
        k=8,
        batch_x=grid_batch,
        batch_y=query_batch
    )
    print("\nknn调用成功")
except Exception as e:
    print(f"\nknn调用失败: {e}")
    import traceback
    traceback.print_exc() 