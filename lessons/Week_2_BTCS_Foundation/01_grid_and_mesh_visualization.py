"""
01_grid_and_mesh_visualization.py  —  Week 2: BTCS

Educational goal
----------------
Shows the BTCS computational mesh and stencil dependency pattern.

BTCS means:
- Backward-Time: use the NEXT time level j+1 for the space derivative.
- Centered-Space: use left, center, and right at level j+1 (all unknown).

The key coupling at each time step:

    U_{i-1}^{j+1},  U_i^{j+1},  U_{i+1}^{j+1}  are ALL unknown
    and linked by:  -r U_{i-1}^{j+1} + (1+2r) U_i^{j+1} - r U_{i+1}^{j+1} = U_i^j

This means the unknowns cannot be found one at a time; they must be solved
simultaneously as a tridiagonal linear system.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs" / "mesh_images"


def node_label(space_index, time_index):
    return rf"$U_{{{space_index}}}^{{{time_index}}}$"


def draw_btcs_dependency_mesh(show_labels=True):
    """
    Draw a space-time grid highlighting the BTCS implicit stencil.

    Known nodes (current level j) are blue.
    Unknown nodes coupled together at level j+1 are red.
    Arrows flow from the known RHS value up into all three unknowns.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    space_points = range(5)
    time_levels = range(4)
    center_i = 2
    current_j = 1
    future_j = current_j + 1

    fig, ax = plt.subplots(figsize=(10, 7))

    for x in space_points:
        ax.plot([x, x], [min(time_levels), max(time_levels)], color="0.82", linewidth=1)
    for t in time_levels:
        ax.plot([min(space_points), max(space_points)], [t, t], color="0.82", linewidth=1)

    for t in time_levels:
        for x in space_points:
            ax.scatter(x, t, s=120, color="white", edgecolor="0.35", zorder=3)
            if show_labels:
                ax.text(x, t - 0.16, node_label(x, t), ha="center", va="top", fontsize=10)

    known_node = (center_i, current_j)
    unknown_nodes = [
        (center_i - 1, future_j),
        (center_i,     future_j),
        (center_i + 1, future_j),
    ]

    ax.scatter(*known_node, s=280, color="#2f6fdb", edgecolor="black", zorder=5)

    for x, t in unknown_nodes:
        ax.scatter(x, t, s=300, color="#d62828", edgecolor="black", zorder=5)
        ax.annotate(
            "", xy=(x, t - 0.08), xytext=(center_i, current_j + 0.08),
            arrowprops=dict(arrowstyle="->", color="#d62828", lw=2),
        )

    ax.text(center_i, current_j + 0.25,
            "KNOWN (RHS)", color="#2f6fdb", fontsize=10, ha="center", fontweight="bold")
    ax.text(center_i, future_j + 0.25,
            "UNKNOWN — solved as system", color="#d62828", fontsize=10, ha="center", fontweight="bold")

    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(-0.5, 3.8)
    ax.set_xlabel("Space index i", fontsize=12)
    ax.set_ylabel("Time level j", fontsize=12)
    ax.set_title("BTCS Implicit Stencil\n"
                 r"$-r\,U_{i-1}^{j+1}+(1+2r)\,U_i^{j+1}-r\,U_{i+1}^{j+1}=U_i^j$",
                 fontsize=13)
    ax.set_xticks(list(space_points))
    ax.set_yticks(list(time_levels))
    ax.set_xticklabels([f"i={i}" for i in space_points])
    ax.set_yticklabels([f"j={t}" for t in time_levels])
    ax.grid(False)

    known_patch   = mpatches.Patch(color="#2f6fdb", label="Known (RHS) at level j")
    unknown_patch = mpatches.Patch(color="#d62828", label="Unknown at level j+1 (system)")
    ax.legend(handles=[known_patch, unknown_patch], loc="upper left", fontsize=10)

    fig.tight_layout()
    output_path = OUTPUT_DIR / "btcs_dependency_mesh.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"BTCS mesh visualisation saved: {output_path}")
    return output_path


if __name__ == "__main__":
    draw_btcs_dependency_mesh()
