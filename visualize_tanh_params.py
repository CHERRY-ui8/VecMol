import numpy as np
import matplotlib.pyplot as plt

# 原子位置
atoms = np.array([-1.0, 1.0])
x = np.linspace(-5, 5, 1000)

# 参数范围
sig_sf_values = [0.05, 0.1, 0.2, 0.5]  # softmax sharpness
sig_mag_values = [1.0, 2.0, 3.0, 5.0]  # magnitude saturation

def compute_tanh_field(x, atoms, sig_sf, sig_mag):
    """计算Softmax--Tanh field"""
    # 计算所有原子到所有查询点的距离和方向
    diffs = atoms[:, None] - x[None, :]  # [n_atoms, n_points]
    dists = np.abs(diffs)  # [n_atoms, n_points]
    
    # Softmax term
    w_sf = np.exp(-dists / sig_sf)
    w_sf = w_sf / w_sf.sum(axis=0, keepdims=True)
    
    # Tanh magnitude term
    w_mag = np.tanh(dists / sig_mag)
    
    # Combined field
    field = np.sum(np.sign(diffs) * w_sf * w_mag, axis=0)
    
    return field

# 创建图形
fig, axes = plt.subplots(2, 1, figsize=(12, 10))

# 左图：固定 sig_mag，变化 sig_sf
ax1 = axes[0]
colors_sf = plt.cm.viridis(np.linspace(0, 1, len(sig_sf_values)))
sig_mag_fixed = 2.0

for i, sig_sf in enumerate(sig_sf_values):
    field = compute_tanh_field(x, atoms, sig_sf, sig_mag_fixed)
    ax1.plot(x, field, label=f'$\\sigma_{{\\mathrm{{sf}}}}={sig_sf}$', 
             linewidth=2, color=colors_sf[i])

ax1.axhline(0, color='gray', linestyle='--', alpha=0.5)
ax1.axvline(-1, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
ax1.axvline(1, color='green', linestyle='--', alpha=0.7, linewidth=1.5)
ax1.scatter([-1, 1], [0, 0], s=200, c=['red', 'green'], marker='o', 
            edgecolors='black', linewidths=2, zorder=5, alpha=0.9)
ax1.set_xlabel('', fontsize=12)
ax1.set_ylabel('', fontsize=12)
ax1.set_title(f'Varying $\\sigma_{{\\mathrm{{sf}}}}$ (softmax sharpness) with fixed $\\sigma_{{\\mathrm{{mag}}}}={sig_mag_fixed}$', 
              fontsize=18, fontweight='bold')
ax1.set_xlim(-5, 5)
ax1.set_ylim(-1.5, 1.5)
ax1.grid(True, alpha=0.3)
ax1.legend(loc='upper right', fontsize=16)

# 右图：固定 sig_sf，变化 sig_mag
ax2 = axes[1]
colors_mag = plt.cm.plasma(np.linspace(0, 1, len(sig_mag_values)))
sig_sf_fixed = 0.1

for i, sig_mag in enumerate(sig_mag_values):
    field = compute_tanh_field(x, atoms, sig_sf_fixed, sig_mag)
    ax2.plot(x, field, label=f'$\\sigma_{{\\mathrm{{mag}}}}={sig_mag}$', 
             linewidth=2, color=colors_mag[i])

ax2.axhline(0, color='gray', linestyle='--', alpha=0.5)
ax2.axvline(-1, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
ax2.axvline(1, color='green', linestyle='--', alpha=0.7, linewidth=1.5)
ax2.scatter([-1, 1], [0, 0], s=200, c=['red', 'green'], marker='o', 
            edgecolors='black', linewidths=2, zorder=5, alpha=0.9)
ax2.set_xlabel('', fontsize=12)
ax2.set_ylabel('', fontsize=12)
ax2.set_title(f'Varying $\\sigma_{{\\mathrm{{mag}}}}$ (magnitude saturation) with fixed $\\sigma_{{\\mathrm{{sf}}}}={sig_sf_fixed}$', 
              fontsize=18, fontweight='bold')
ax2.set_xlim(-5, 5)
ax2.set_ylim(-1.5, 1.5)
ax2.grid(True, alpha=0.3)
ax2.legend(loc='upper right', fontsize=16)

plt.tight_layout()
plt.savefig('tanh_params_1d.pdf', dpi=300, bbox_inches='tight')
print("Saved: tanh_params_1d.pdf")
plt.show()
