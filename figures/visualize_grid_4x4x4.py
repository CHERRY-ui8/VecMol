import numpy as np
import matplotlib.pyplot as plt

# 创建4x4x4的格点
grid_size = 4
x = np.arange(grid_size)
y = np.arange(grid_size)
z = np.arange(grid_size)

# 创建网格
X, Y, Z = np.meshgrid(x, y, z)

# 展平坐标
x_flat = X.flatten()
y_flat = Y.flatten()
z_flat = Z.flatten()

# 计算每个点到观察点的距离，用于调整点的大小（增强纵深感）
# 使用一个参考点（比如立方体中心）来计算距离
center = (grid_size - 1) / 2
distances = np.sqrt((x_flat - center)**2 + (y_flat - center)**2 + (z_flat - center)**2)
max_dist = np.max(distances)
# 归一化距离
normalized_dist = distances / max_dist if max_dist > 0 else distances

# 根据深度调整点的大小（距离越远，点越小，增强纵深感）
sizes = 300 - normalized_dist * 100

# 绘制立方体的边框线条，增强立体感
def draw_cube_edges(ax, grid_size_val):
    """绘制立方体的边框"""
    # 定义立方体的8个顶点
    vertices = [
        [0, 0, 0], [grid_size_val-1, 0, 0], 
        [grid_size_val-1, grid_size_val-1, 0], [0, grid_size_val-1, 0],
        [0, 0, grid_size_val-1], [grid_size_val-1, 0, grid_size_val-1],
        [grid_size_val-1, grid_size_val-1, grid_size_val-1], [0, grid_size_val-1, grid_size_val-1]
    ]
    
    # 定义12条边
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # 底面
        [4, 5], [5, 6], [6, 7], [7, 4],  # 顶面
        [0, 4], [1, 5], [2, 6], [3, 7]   # 垂直边
    ]
    
    # 绘制每条边（使用虚线）
    for edge in edges:
        points = np.array([vertices[edge[0]], vertices[edge[1]]])
        ax.plot3D(points[:, 0], points[:, 1], points[:, 2], 
                  'k--', linewidth=2, alpha=0.6)

# 绘制所有与坐标轴平行的内部连线
def draw_internal_edges(ax, grid_size_val):
    """绘制所有与坐标轴平行的内部连线"""
    # 沿着X轴方向的连线（固定y和z，连接相邻的x）
    for yi in range(grid_size_val):
        for zi in range(grid_size_val):
            for xi in range(grid_size_val - 1):
                ax.plot3D([xi, xi+1], [yi, yi], [zi, zi], 
                         'k--', linewidth=1, alpha=0.4)
    
    # 沿着Y轴方向的连线（固定x和z，连接相邻的y）
    for xi in range(grid_size_val):
        for zi in range(grid_size_val):
            for yi in range(grid_size_val - 1):
                ax.plot3D([xi, xi], [yi, yi+1], [zi, zi], 
                         'k--', linewidth=1, alpha=0.4)
    
    # 沿着Z轴方向的连线（固定x和y，连接相邻的z）
    for xi in range(grid_size_val):
        for yi in range(grid_size_val):
            for zi in range(grid_size_val - 1):
                ax.plot3D([xi, xi], [yi, yi], [zi, zi+1], 
                         'k--', linewidth=1, alpha=0.4)

def create_grid_plot(use_colors=False, save_path=None):
    """创建并绘制4x4x4网格图
    
    Args:
        use_colors: 如果为True，使用随机颜色；如果为False，使用灰色
        save_path: 保存路径，如果为None则不保存
    """
    # 创建图形，使用更大的尺寸以获得更好的效果
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # 根据参数选择颜色
    if use_colors:
        # 为每个点生成随机颜色（64个点，每个点一个随机RGB颜色）
        np.random.seed(43)  # 设置随机种子以便结果可复现
        num_points = len(x_flat)
        point_colors = np.random.rand(num_points, 3)  # 生成64个随机RGB颜色
    else:
        # 使用灰色
        point_colors = 'gray'
    
    # 绘制所有格点
    ax.scatter(x_flat, y_flat, z_flat, 
               s=sizes, c=point_colors, alpha=0.8, 
               edgecolors='black', linewidths=1.5,
               depthshade=True)
    
    # 绘制立方体边框
    draw_cube_edges(ax, grid_size)
    
    # 绘制内部连线
    draw_internal_edges(ax, grid_size)
    
    # 设置坐标轴范围
    ax.set_xlim(-0.5, grid_size - 0.5)
    ax.set_ylim(-0.5, grid_size - 0.5)
    ax.set_zlim(-0.5, grid_size - 0.5)
    
    # 设置坐标轴宽高比为1:1:1，确保立方体显示为正立方体
    ax.set_box_aspect([1, 1, 1])
    
    # 设置更好的视角，增强立体感
    ax.view_init(elev=25, azim=60)
    
    # 隐藏坐标轴
    ax.set_axis_off()
    
    # 设置背景色为纯白色
    ax.xaxis.pane.fill = True
    ax.yaxis.pane.fill = True
    ax.zaxis.pane.fill = True
    ax.xaxis.pane.set_facecolor('white')
    ax.yaxis.pane.set_facecolor('white')
    ax.zaxis.pane.set_facecolor('white')
    ax.xaxis.pane.set_edgecolor('white')
    ax.yaxis.pane.set_edgecolor('white')
    ax.zaxis.pane.set_edgecolor('white')
    
    # 设置图形背景为白色
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    # 保存图片
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"已保存图片: {save_path}")
    
    plt.close(fig)

# 生成灰色版本的图片
create_grid_plot(use_colors=False, save_path='figures/grid_4x4x4.png')

# 生成彩色版本的图片
create_grid_plot(use_colors=True, save_path='figures/grid_4x4x4_colored.png')

print("所有图片已生成完成！")

