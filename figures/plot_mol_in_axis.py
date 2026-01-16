import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from rdkit import Chem
from rdkit.Chem import AllChem


# =========================
# Molecule loading
# =========================

def load_molecule_from_sdf(sdf_path):
    suppl = Chem.SDMolSupplier(sdf_path, removeHs=False)
    mol = next((m for m in suppl if m is not None), None)
    if mol is None:
        raise ValueError(f"Failed to read molecule from {sdf_path}")

    if not mol.GetNumConformers():
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        AllChem.UFFOptimizeMolecule(mol)

    return mol


def get_coords_and_symbols(mol):
    conf = mol.GetConformer()
    coords = []
    symbols = []
    for i in range(mol.GetNumAtoms()):
        p = conf.GetAtomPosition(i)
        coords.append([p.x, p.y, p.z])
        symbols.append(mol.GetAtomWithIdx(i).GetSymbol())
    return np.asarray(coords), symbols


# =========================
# Spatial bounds
# =========================

def compute_bounds(coords, padding=1.5):
    mins = coords.min(0)
    maxs = coords.max(0)
    center = 0.5 * (mins + maxs)
    half = 0.5 * (maxs - mins) + padding
    r = half.max()
    return (
        (center[0] - r, center[0] + r),
        (center[1] - r, center[1] + r),
        (center[2] - r, center[2] + r),
    )


# =========================
# Background planes + grids
# =========================

def draw_back_planes_with_grid(ax, bounds, spacing=1.0):
    (xmin, xmax), (ymin, ymax), (zmin, zmax) = bounds

    # xOz plane (y = ymin)
    xs = np.arange(xmin, xmax + 1e-6, spacing)
    zs = np.arange(zmin, zmax + 1e-6, spacing)

    for x in xs:
        ax.plot([x, x], [ymin, ymin], [zmin, zmax],
                color="lightgray", linewidth=0.6, alpha=0.5)
    for z in zs:
        ax.plot([xmin, xmax], [ymin, ymin], [z, z],
                color="lightgray", linewidth=0.6, alpha=0.5)

    # yOz plane (x = xmin)
    ys = np.arange(ymin, ymax + 1e-6, spacing)

    for y in ys:
        ax.plot([xmin, xmin], [y, y], [zmin, zmax],
                color="lightgray", linewidth=0.6, alpha=0.5)
    for z in zs:
        ax.plot([xmin, xmin], [ymin, ymax], [z, z],
                color="lightgray", linewidth=0.6, alpha=0.5)


# =========================
# Coordinate axes (long arrows)
# =========================

def draw_axes(ax, bounds, factor=0.35):
    (xmin, xmax), (ymin, ymax), (zmin, zmax) = bounds
    center = np.array([
        0.5 * (xmin + xmax),
        0.5 * (ymin + ymax),
        0.5 * (zmin + zmax),
    ])
    extent = max(xmax - xmin, ymax - ymin, zmax - zmin)
    L = factor * extent

    ax.quiver(*center, L, 0, 0, color="red",
              arrow_length_ratio=0.08, linewidth=2.5)
    ax.quiver(*center, 0, L, 0, color="green",
              arrow_length_ratio=0.08, linewidth=2.5)
    ax.quiver(*center, 0, 0, L, color="blue",
              arrow_length_ratio=0.08, linewidth=2.5)

    ax.text(center[0] + L * 1.1, center[1], center[2], "X", color="red", fontsize=12)
    ax.text(center[0], center[1] + L * 1.1, center[2], "Y", color="green", fontsize=12)
    ax.text(center[0], center[1], center[2] + L * 1.1, "Z", color="blue", fontsize=12)


# =========================
# Molecule drawing (PyMOL-like)
# =========================

def element_color(sym):
    return {
        "H": (0.95, 0.95, 0.95),
        "C": (0.25, 0.25, 0.25),
        "N": (0.1, 0.2, 0.8),
        "O": (0.85, 0.1, 0.1),
        "S": (0.9, 0.75, 0.1),
        "P": (1.0, 0.5, 0.1),
        "F": (0.2, 0.8, 0.2),
        "Cl": (0.2, 0.8, 0.2),
    }.get(sym, (0.6, 0.6, 0.6))


def covalent_radius(sym):
    return {
        "H": 0.25,
        "C": 0.40,
        "N": 0.40,
        "O": 0.40,
        "S": 0.55,
        "P": 0.55,
        "F": 0.40,
        "Cl": 0.55,
    }.get(sym, 0.45)


def draw_sphere(ax, center, r, color):
    u = np.linspace(0, 2 * np.pi, 60)
    v = np.linspace(0, np.pi, 60)
    x = r * np.outer(np.cos(u), np.sin(v)) + center[0]
    y = r * np.outer(np.sin(u), np.sin(v)) + center[1]
    z = r * np.outer(np.ones_like(u), np.cos(v)) + center[2]
    ax.plot_surface(x, y, z, color=color, linewidth=0, alpha=1.0, shade=True)


def draw_molecule(ax, mol, coords, symbols):
    # Bonds first
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        p, q = coords[i], coords[j]
        ax.plot([p[0], q[0]], [p[1], q[1]], [p[2], q[2]],
                color=(0.2, 0.2, 0.2), linewidth=3.0, alpha=0.9)

    # Atoms (true spheres)
    for pos, sym in zip(coords, symbols):
        draw_sphere(ax, pos, covalent_radius(sym), element_color(sym))


# =========================
# Main figure
# =========================

def create_figure(sdf, output):
    mol = load_molecule_from_sdf(sdf)
    coords, symbols = get_coords_and_symbols(mol)
    bounds = compute_bounds(coords)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    draw_back_planes_with_grid(ax, bounds, spacing=1.0)
    draw_axes(ax, bounds)
    draw_molecule(ax, mol, coords, symbols)

    (xmin, xmax), (ymin, ymax), (zmin, zmax) = bounds
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(zmin, zmax)
    ax.set_box_aspect([1, 1, 1])

    ax.view_init(elev=22, azim=55)

    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {output}")


# =========================
# CLI
# =========================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sdf", required=True)
    parser.add_argument("--output", default="figure1_molecule_space.png")
    args = parser.parse_args()

    create_figure(args.sdf, args.output)
