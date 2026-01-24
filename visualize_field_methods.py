import numpy as np
import matplotlib.pyplot as plt

# 原子位置
atoms = np.array([-1.0, 1.0])
x = np.linspace(-5, 5, 1000)

# 参数
sigma = 1.0
T = 0.01
sig_sf = 0.1
sig_mag = 0.45
strength = 1.0
scale = 0.1
eps = 1e-8

# 坐标轴范围
xlim = (-5, 5)
ylim = (-1.5, 1.5)

def compute_field(method, x, atoms):
    """计算一维梯度场"""
    # 计算所有原子到所有查询点的距离和方向
    diffs = atoms[:, None] - x[None, :]  # [n_atoms, n_points]
    dists = np.abs(diffs)  # [n_atoms, n_points]
    
    if method == "gaussian":
        w = np.exp(-dists**2 / (2*sigma**2)) / sigma**2
        field = np.sum(diffs * w, axis=0)
        
    elif method == "softmax":
        w = np.exp(-dists / T)
        w = w / w.sum(axis=0, keepdims=True)
        field = np.sum(diffs * w, axis=0)
        # 梯度截断
        mag = np.abs(field)
        field = 2*np.where(mag > 0.3, field * 0.3 / mag, field)
        
    elif method == "sfnorm":
        w = np.exp(-dists / T)
        w = w / w.sum(axis=0, keepdims=True)
        field = np.sum(np.sign(diffs) * w, axis=0)
        
    elif method == "logsumexp":
        g_i = diffs * np.exp(-dists**2 / (2*sigma**2)) / sigma**2
        g_mags = np.abs(g_i)  # [n_atoms, n_points]
        g_dirs = np.sign(g_i)
        lse = np.log(np.sum(np.exp(g_mags), axis=0) + eps)
        field = scale * np.sum(g_dirs, axis=0) * lse
        
    elif method == "inverse-square":
        w = strength / (4*(dists**2 + eps))
        field = np.sum(diffs * w, axis=0)
        
    elif method == "softmax-tanh":
        w_sf = np.exp(-dists / sig_sf)
        w_sf = w_sf / w_sf.sum(axis=0, keepdims=True)
        w_mag = np.tanh(dists / sig_mag)
        field = np.sum(np.sign(diffs) * w_sf * w_mag, axis=0)
        
    elif method == "gaussian magnitude":
        w_sf = np.exp(-dists / sig_sf)
        w_sf = w_sf / w_sf.sum(axis=0, keepdims=True)
        w_mag = np.exp(-dists**2 / (2*sig_mag**2)) * dists
        field = 2*np.sum(np.sign(diffs) * w_sf * w_mag, axis=0)
        
    elif method == "gaussian hole":
        w_sf = np.exp(-dists / sig_sf)
        w_sf = w_sf / w_sf.sum(axis=0, keepdims=True)
        w_mag = np.exp(-min(dists,)**2 / (2*sig_mag**2)) * dists
        field = 2*np.sum(np.sign(diffs) * w_sf * w_mag, axis=0)

    elif method == "distance with clip":
        w_sf = np.exp(-dists / sig_sf)
        w_sf = w_sf / w_sf.sum(axis=0, keepdims=True)
        w_mag = np.clip(dists, 0, 1)
        field = np.sum(np.sign(diffs) * w_sf * w_mag, axis=0)
    
    return field

methods = ["gaussian", "softmax", "inverse-square", "gaussian magnitude", "distance with clip", "softmax-tanh"]

# 2x3子图，展示6种field定义
fig, axes = plt.subplots(3, 2, figsize=(16, 10))
axes = axes.flatten()

for i, method in enumerate(methods):
    field = compute_field(method, x, atoms)
    axes[i].plot(x, field, linewidth=2)
    axes[i].axhline(0, color='gray', linestyle='--', alpha=0.5)
    axes[i].axvline(-1, color='red', linestyle='--', alpha=0.7, label='Atom 1')
    axes[i].axvline(1, color='green', linestyle='--', alpha=0.7, label='Atom 2')
    # 在原子位置添加醒目的实心圆点
    axes[i].scatter([-1, 1], [0, 0], s=200, c=['red', 'green'], marker='o', 
                    edgecolors='black', linewidths=2, zorder=5, alpha=0.9)
    axes[i].set_title(method, fontsize=25)
    axes[i].set_xlabel('')
    axes[i].set_ylabel('')
    axes[i].set_xlim(xlim)
    axes[i].set_ylim(ylim)
    axes[i].grid(True, alpha=0.3)
    axes[i].legend(fontsize=18)

plt.tight_layout()
plt.savefig('field_methods_subplots.pdf', dpi=150, bbox_inches='tight')
print("Saved: field_methods_subplots.pdf")

# 所有方法叠加图
plt.figure(figsize=(10, 6))
colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))

for i, method in enumerate(methods):
    field = compute_field(method, x, atoms)
    plt.plot(x, field, label=method, linewidth=2, color=colors[i])

plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
plt.axvline(-1, color='red', linestyle='--', alpha=0.7, label='Atom 1')
plt.axvline(1, color='green', linestyle='--', alpha=0.7, label='Atom 2')
# 在原子位置添加醒目的实心圆点
plt.scatter([-1, 1], [0, 0], s=200, c=['red', 'green'], marker='o', 
            edgecolors='black', linewidths=2, zorder=5, alpha=0.9)
plt.xlabel('')
plt.ylabel('')
plt.title('Comparison of All Six Methods')
plt.xlim(xlim)
plt.ylim(ylim)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=18)
plt.tight_layout()
plt.savefig('field_methods_overlay.pdf', dpi=150, bbox_inches='tight')
print("Saved: field_methods_overlay.pdf")
plt.show()
