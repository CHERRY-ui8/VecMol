import torch
import torch_geometric
from torch_geometric.nn.pool import knn

# 模拟实际的参数值
batch_size = 1
n_points = 3000
n_grid = 512
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 创建测试数据
query_points = torch.randn(n_points, 3, device=device)  # [3000, 3]
grid_coords = torch.randn(n_grid, 3, device=device)  # [512, 3]

# 创建batch索引
query_batch = torch.arange(batch_size, device=device).repeat_interleave(n_points)  # [3000]
grid_batch = torch.arange(batch_size, device=device).repeat_interleave(n_grid)  # [512]

print(f"query_points shape: {query_points.shape}")
print(f"grid_coords shape: {grid_coords.shape}")
print(f"query_batch shape: {query_batch.shape}")
print(f"grid_batch shape: {grid_batch.shape}")

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

# 尝试不同的参数组合
print(f"\n尝试不同的参数组合:")

# 测试1: 不使用batch参数
try:
    edge_grid_query = knn(
        x=grid_coords,
        y=query_points,
        k=8
    )
    print("knn调用成功 (无batch参数)")
except Exception as e:
    print(f"knn调用失败 (无batch参数): {e}")

# 测试2: 只使用batch_y
try:
    edge_grid_query = knn(
        x=grid_coords,
        y=query_points,
        k=8,
        batch_y=query_batch
    )
    print("knn调用成功 (只有batch_y)")
except Exception as e:
    print(f"knn调用失败 (只有batch_y): {e}")

# 测试3: 只使用batch_x
try:
    edge_grid_query = knn(
        x=grid_coords,
        y=query_points,
        k=8,
        batch_x=grid_batch
    )
    print("knn调用成功 (只有batch_x)")
except Exception as e:
    print(f"knn调用失败 (只有batch_x): {e}") 