import argparse
from typing import Tuple, List

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
except ImportError as e:
    raise ImportError(
        "RDKit is required for this script. Please install RDKit first, "
        "e.g. via conda: `conda install -c rdkit rdkit`."
    ) from e


def load_molecule_from_sdf(sdf_path: str):
    """Load first molecule from an SDF file (with 3D coordinates)."""
    suppl = Chem.SDMolSupplier(sdf_path, removeHs=False)
    mol = next((m for m in suppl if m is not None), None)
    if mol is None:
        raise ValueError(f"Failed to read molecule from SDF: {sdf_path}")

    # Ensure we have 3D coordinates
    if not mol.GetNumConformers():
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        AllChem.UFFOptimizeMolecule(mol)

    return mol


def get_molecule_coordinates(mol) -> Tuple[np.ndarray, List[str]]:
    """Return atom coordinates (N, 3) and element symbols."""
    conf = mol.GetConformer()
    num_atoms = mol.GetNumAtoms()
    coords = np.zeros((num_atoms, 3), dtype=float)
    symbols: List[str] = []

    for i in range(num_atoms):
        pos = conf.GetAtomPosition(i)
        coords[i] = [pos.x, pos.y, pos.z]
        symbols.append(mol.GetAtomWithIdx(i).GetSymbol())

    return coords, symbols


def compute_bounds(coords: np.ndarray, padding: float = 1.5) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
    """Compute axis-aligned bounding box with padding."""
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    center = 0.5 * (mins + maxs)
    half_extent = 0.5 * (maxs - mins) + padding

    # Use isotropic extent to get a nice cube
    max_half = float(half_extent.max())
    mins_box = center - max_half
    maxs_box = center + max_half

    return (mins_box[0], maxs_box[0]), (mins_box[1], maxs_box[1]), (mins_box[2], maxs_box[2])


def draw_bounding_box(ax, bounds, face_color=(0.8, 0.8, 1.0), edge_color="navy", alpha=0.10, linewidth=1.5):
    """Draw a semi-transparent 3D bounding box."""
    (xmin, xmax), (ymin, ymax), (zmin, zmax) = bounds

    # 8 vertices of the box
    v = np.array(
        [
            [xmin, ymin, zmin],
            [xmax, ymin, zmin],
            [xmax, ymax, zmin],
            [xmin, ymax, zmin],
            [xmin, ymin, zmax],
            [xmax, ymin, zmax],
            [xmax, ymax, zmax],
            [xmin, ymax, zmax],
        ]
    )

    # 6 faces defined by 4 vertices each
    faces = [
        [v[0], v[1], v[2], v[3]],  # bottom
        [v[4], v[5], v[6], v[7]],  # top
        [v[0], v[1], v[5], v[4]],  # front
        [v[2], v[3], v[7], v[6]],  # back
        [v[1], v[2], v[6], v[5]],  # right
        [v[3], v[0], v[4], v[7]],  # left
    ]

    box = Poly3DCollection(
        faces,
        facecolors=face_color,
        edgecolors=edge_color,
        linewidths=linewidth,
        alpha=alpha,
    )
    ax.add_collection3d(box)

    # Also draw slightly stronger edges for better depth perception
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]
    for i, j in edges:
        xs, ys, zs = zip(v[i], v[j])
        ax.plot(xs, ys, zs, color=edge_color, linewidth=1.0, alpha=0.4)


def draw_sparse_voxel_grid(ax, bounds, spacing: float = 1.0, color="lightgray", alpha=0.4, linewidth=0.4):
    """Draw a sparse voxel-like grid inside the bounding box."""
    (xmin, xmax), (ymin, ymax), (zmin, zmax) = bounds

    xs = np.arange(xmin, xmax + 1e-6, spacing)
    ys = np.arange(ymin, ymax + 1e-6, spacing)
    zs = np.arange(zmin, zmax + 1e-6, spacing)

    # Lines parallel to X
    for y in ys[::2]:
        for z in zs[::2]:
            ax.plot([xs[0], xs[-1]], [y, y], [z, z], color=color, alpha=alpha, linewidth=linewidth)

    # Lines parallel to Y
    for x in xs[::2]:
        for z in zs[::2]:
            ax.plot([x, x], [ys[0], ys[-1]], [z, z], color=color, alpha=alpha, linewidth=linewidth)

    # Lines parallel to Z
    for x in xs[::2]:
        for y in ys[::2]:
            ax.plot([x, x], [y, y], [zmin, zmax], color=color, alpha=alpha, linewidth=linewidth)


def draw_axes_arrows(ax, bounds, arrow_length_factor: float = 0.15):
    """Draw coordinate axes as arrows centered at the box center."""
    (xmin, xmax), (ymin, ymax), (zmin, zmax) = bounds
    center = np.array(
        [
            0.5 * (xmin + xmax),
            0.5 * (ymin + ymax),
            0.5 * (zmin + zmax),
        ]
    )
    extent = max(xmax - xmin, ymax - ymin, zmax - zmin)
    L = arrow_length_factor * extent

    # X axis (red)
    ax.quiver(
        center[0],
        center[1],
        center[2],
        L,
        0,
        0,
        color="red",
        arrow_length_ratio=0.2,
        linewidth=2.0,
    )
    # Y axis (green)
    ax.quiver(
        center[0],
        center[1],
        center[2],
        0,
        L,
        0,
        color="green",
        arrow_length_ratio=0.2,
        linewidth=2.0,
    )
    # Z axis (blue)
    ax.quiver(
        center[0],
        center[1],
        center[2],
        0,
        0,
        L,
        color="blue",
        arrow_length_ratio=0.2,
        linewidth=2.0,
    )

    ax.text(center[0] + L * 1.1, center[1], center[2], "X", color="red")
    ax.text(center[0], center[1] + L * 1.1, center[2], "Y", color="green")
    ax.text(center[0], center[1], center[2] + L * 1.1, "Z", color="blue")


def get_element_color(symbol: str):
    """Simple element-to-color mapping."""
    colors = {
        "H": (1.0, 1.0, 1.0),  # white
        "C": (0.2, 0.2, 0.2),  # dark gray
        "N": (0.0, 0.0, 1.0),  # blue
        "O": (1.0, 0.0, 0.0),  # red
        "F": (0.0, 0.8, 0.0),  # green
        "Cl": (0.0, 0.8, 0.0),
        "S": (1.0, 0.8, 0.0),  # yellow
        "P": (1.0, 0.5, 0.0),  # orange
    }
    return colors.get(symbol, (0.6, 0.6, 0.6))


def get_element_size(symbol: str, base: float = 250.0):
    """Rough size mapping for atoms (for scatter size)."""
    factors = {
        "H": 0.45,
        "C": 0.8,
        "N": 0.8,
        "O": 0.8,
        "F": 0.8,
        "Cl": 1.0,
        "S": 1.0,
        "P": 1.0,
    }
    return base * factors.get(symbol, 0.8)


def draw_molecule(
    ax,
    mol,
    coords: np.ndarray,
    symbols: List[str],
    draw_bonds: bool = True,
):
    """Draw atoms (and optionally bonds) inside the current 3D axes."""
    # Draw bonds first so atoms are on top
    if draw_bonds:
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            p1 = coords[i]
            p2 = coords[j]
            ax.plot(
                [p1[0], p2[0]],
                [p1[1], p2[1]],
                [p1[2], p2[2]],
                color="black",
                linewidth=1.5,
                alpha=0.9,
            )

    # Draw atoms
    xs, ys, zs = coords[:, 0], coords[:, 1], coords[:, 2]
    colors = [get_element_color(sym) for sym in symbols]
    sizes = [get_element_size(sym) for sym in symbols]

    ax.scatter(
        xs,
        ys,
        zs,
        s=sizes,
        c=colors,
        edgecolors="black",
        linewidths=0.8,
        alpha=0.95,
        depthshade=True,
    )


def set_equal_aspect(ax, bounds):
    """Set equal aspect ratio and limits based on bounds."""
    (xmin, xmax), (ymin, ymax), (zmin, zmax) = bounds
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(zmin, zmax)
    ax.set_box_aspect([xmax - xmin, ymax - ymin, zmax - zmin])


def create_molecule_grid_figure(
    sdf_path: str,
    save_path: str,
    draw_bonds: bool = True,
    grid_spacing: float = 1.0,
    padding: float = 1.5,
    dpi: int = 300,
):
    """Create the 3D visualization: bounding box + sparse grid + molecule."""
    mol = load_molecule_from_sdf(sdf_path)
    coords, symbols = get_molecule_coordinates(mol)
    bounds = compute_bounds(coords, padding=padding)

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection="3d")

    # Background similar to visualize_grid_4x4x4.py but keep axes visible
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Semi-transparent bounding box
    draw_bounding_box(ax, bounds)

    # Sparse voxel-like grid
    if grid_spacing > 0:
        draw_sparse_voxel_grid(ax, bounds, spacing=grid_spacing)

    # Coordinate axes arrows
    draw_axes_arrows(ax, bounds)

    # Molecule
    draw_molecule(ax, mol, coords, symbols, draw_bonds=draw_bonds)

    # Equal aspect and camera/view angle matching visualize_grid_4x4x4.py
    set_equal_aspect(ax, bounds)
    ax.view_init(elev=25, azim=60)

    # Tidy axes: light ticks, but no grid
    ax.grid(False)
    ax.set_xlabel("X", labelpad=10)
    ax.set_ylabel("Y", labelpad=10)
    ax.set_zlabel("Z", labelpad=10)

    # Save
    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    print(f"Saved figure to: {save_path}")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Visualize a molecule from SDF in a 3D coordinate system with "
            "a semi-transparent bounding box and sparse voxel grid."
        )
    )
    parser.add_argument(
        "--sdf",
        type=str,
        required=True,
        help="Path to the input SDF file (e.g. genmol_0006.sdf).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="figures/molecule_in_grid.png",
        help="Path to save the output image.",
    )
    parser.add_argument(
        "--no-bonds",
        action="store_true",
        help="If set, only atoms are drawn (no bonds).",
    )
    parser.add_argument(
        "--grid-spacing",
        type=float,
        default=1.0,
        help="Spacing of the sparse voxel grid (in the same units as coordinates).",
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=1.5,
        help="Extra padding around the molecule for the bounding box.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for saving the figure.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    create_molecule_grid_figure(
        sdf_path=args.sdf,
        save_path=args.output,
        draw_bonds=not args.no_bonds,
        grid_spacing=args.grid_spacing,
        padding=args.padding,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()


