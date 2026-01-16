import numpy as np
import matplotlib.pyplot as plt


def draw_3plane_grid(
    ax,
    xlim=(-5, 5),
    ylim=(-5, 5),
    zlim=(-5, 5),
    spacing=1.0,
    color="gray",
    linewidth=0.6,
    alpha=0.5,
):
    xmin, xmax = xlim
    ymin, ymax = ylim
    zmin, zmax = zlim

    xs = np.arange(xmin, xmax + 1e-6, spacing)
    ys = np.arange(ymin, ymax + 1e-6, spacing)
    zs = np.arange(zmin, zmax + 1e-6, spacing)

    # =========================
    # xOy plane (z = zmin)
    # =========================
    for x in xs:
        ax.plot(
            [x, x], [ymin, ymax], [zmin, zmin],
            color=color, linewidth=linewidth, alpha=alpha
        )
    for y in ys:
        ax.plot(
            [xmin, xmax], [y, y], [zmin, zmin],
            color=color, linewidth=linewidth, alpha=alpha
        )

    # =========================
    # xOz plane (y = ymin)
    # =========================
    for x in xs:
        ax.plot(
            [x, x], [ymin, ymin], [zmin, zmax],
            color=color, linewidth=linewidth, alpha=alpha
        )
    for z in zs:
        ax.plot(
            [xmin, xmax], [ymin, ymin], [z, z],
            color=color, linewidth=linewidth, alpha=alpha
        )

    # =========================
    # yOz plane (x = xmin)
    # =========================
    for y in ys:
        ax.plot(
            [xmin, xmin], [y, y], [zmin, zmax],
            color=color, linewidth=linewidth, alpha=alpha
        )
    for z in zs:
        ax.plot(
            [xmin, xmin], [ymin, ymax], [z, z],
            color=color, linewidth=linewidth, alpha=alpha
        )


def create_background_figure(
    output="background_3plane_grid.png",
    extent=5,
    spacing=1.0,
):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    xlim = (-extent, extent)
    ylim = (-extent, extent)
    zlim = (-extent, extent)

    draw_3plane_grid(
        ax,
        xlim=xlim,
        ylim=ylim,
        zlim=zlim,
        spacing=spacing,
    )

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)
    ax.set_box_aspect([1, 1, 1])

    # 这个视角是“论文友好”的经典视角
    ax.view_init(elev=22, azim=55)

    # 彻底关闭坐标轴
    ax.set_axis_off()

    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved background grid to {output}")


if __name__ == "__main__":
    create_background_figure(
        output="figure_background_3plane_grid.png",
        extent=5,
        spacing=1.0,
    )
