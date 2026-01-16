import argparse
from typing import Tuple, List

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  # needed for 3D projection


def _setup_3d_axis(
    ax: plt.Axes,
    coords_list: List[np.ndarray],
    margin: float = 0.5,
) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
    """模仿 funcmol/utils/gnf_visualizer.py 中的 _setup_3d_axis 样式设置 3D 坐标轴。

    Args:
        ax: 3D 绘图轴
        coords_list: 坐标数组列表，用于确定轴范围
        margin: 坐标轴边距

    Returns:
        三个轴的范围 (x_min, x_max), (y_min, y_max), (z_min, z_max)
    """
    ax.set_xlabel("X (Å)")
    ax.set_ylabel("Y (Å)")
    ax.set_zlabel("Z (Å)")
    ax.grid(True, alpha=0.3)

    all_coords = np.vstack(coords_list)
    x_min, x_max = all_coords[:, 0].min() - margin, all_coords[:, 0].max() + margin
    y_min, y_max = all_coords[:, 1].min() - margin, all_coords[:, 1].max() + margin
    z_min, z_max = all_coords[:, 2].min() - margin, all_coords[:, 2].max() + margin

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)
    ax.set_box_aspect([1, 1, 1])

    return (x_min, x_max), (y_min, y_max), (z_min, z_max)


def create_empty_3d_axis_figure(
    center: Tuple[float, float, float],
    extent: float,
    save_path: str,
    elev: float = 30.0,
    azim: float = 60.0,
    dpi: int = 300,
) -> None:
    """创建一个仅包含 3D 空间坐标轴的图像，风格模仿 gnf_visualizer.py。

    坐标范围为 [center_i - extent, center_i + extent]，i ∈ {x, y, z}。
    """
    cx, cy, cz = center
    x = np.array([cx - extent, cx + extent])
    y = np.array([cy - extent, cy + extent])
    z = np.array([cz - extent, cz + extent])
    # 构造 8 个角点，仅用于确定范围
    X, Y, Z = np.meshgrid(x, y, z)
    coords = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=-1)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    # 坐标轴样式和范围
    _setup_3d_axis(ax, [coords], margin=0.0)

    # 视角与 gnf_visualizer 中的大部分可视化一致
    ax.view_init(elev=elev, azim=azim)

    # 仅保留坐标轴和网格，不画任何点
    ax.set_title("Empty 3D Coordinate Axis")

    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    print(f"Saved empty 3D axis figure to: {save_path}")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create an empty 3D coordinate axis figure, "
            "mimicking the style used in funcmol/utils/gnf_visualizer.py."
        )
    )
    parser.add_argument(
        "--center",
        type=float,
        nargs=3,
        default=[0.0, 0.0, 0.0],
        metavar=("CX", "CY", "CZ"),
        help="Center of the coordinate box (default: 0 0 0).",
    )
    parser.add_argument(
        "--extent",
        type=float,
        default=5.0,
        help="Half-length of the box along each axis (default: 5.0).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="figures/empty_3d_axis.png",
        help="Path to save the output image (default: figures/empty_3d_axis.png).",
    )
    parser.add_argument(
        "--elev",
        type=float,
        default=30.0,
        help="Elevation angle in the z plane for the view (default: 30).",
    )
    parser.add_argument(
        "--azim",
        type=float,
        default=60.0,
        help="Azimuth angle in the x,y plane for the view (default: 60).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for saving the figure (default: 300).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    create_empty_3d_axis_figure(
        center=tuple(args.center),
        extent=args.extent,
        save_path=args.output,
        elev=args.elev,
        azim=args.azim,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()


