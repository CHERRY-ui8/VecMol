import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

# 计算6个圆圈的范围
circle_radius = 0.25
num_circles = 6
first_circle_x = 0.5
last_circle_x = 0.5 + (num_circles - 1) * 0.6
# 计算圆圈的左右边缘
first_circle_left = first_circle_x - circle_radius
last_circle_right = last_circle_x + circle_radius
# 设置统一的左右留白
padding = 0.15
rect_left = first_circle_left - padding
rect_right = last_circle_right + padding
rect_width = rect_right - rect_left
# 画布
fig, ax = plt.subplots(figsize=(6, 1.5))
ax.set_xlim(rect_left - 0.05, rect_right + 0.05)  # 根据矩形范围调整，留少量边距ax.set_ylim(0, 1)
ax.set_aspect('equal')  # 保持圆形不被拉伸
ax.axis('off')  # 隐藏坐标轴

# 1. 画圆角矩形
rect = patches.FancyBboxPatch(
    (rect_left, 0.2), rect_width, 0.6,  # 位置+大小，根据圆圈范围调整
    boxstyle="round,pad=0.02,rounding_size=0.1",
    edgecolor='black', facecolor='lightgray'
)
ax.add_patch(rect)

# 2. 画圆圈（6个）
for i in range(6):
    x = 0.5 + i * 0.6  # 等距排列，间距更近
    circle = patches.Circle((x, 0.5), circle_radius, 
                           facecolor='lightgreen', 
                           edgecolor='black', 
                           linewidth=1.5)
    ax.add_patch(circle)

# 保存图像到figures目录
output_path = os.path.join(os.path.dirname(__file__), 'codes_visualization.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"图像已保存到: {output_path}")

plt.show()
