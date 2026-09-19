"""
01_grid_and_mesh_visualization.py

Educational goal
----------------
This script introduces the computational mesh used by the FTCS method for the
1D heat equation.

A mesh is the collection of points where we approximate the solution. For a
space point x_i and a time level t_j, the approximate temperature is written as
U_i^j.

FTCS means:
- Forward-Time: use the current time level j to predict the next level j+1.
- Centered-Space: use the left, center, and right spatial neighbors at time j.

The key dependency is:

    U_i^(j+1) depends on U_(i-1)^j, U_i^j, and U_(i+1)^j.

In this picture, known values are blue and the future value being predicted is
red. The arrows show exactly where the numerical method gets its information.
"""

from pathlib import Path

import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs" / "mesh_images"


def node_label(space_index, time_index):
    """Return a readable mathematical label for a grid node."""
    return rf"$U_{{{space_index}}}^{{{time_index}}}$"


def draw_ftcs_dependency_mesh(show_labels=True):
    """
    Draw a small space-time grid and highlight the FTCS dependency pattern.

    The x-axis is space, and the y-axis is time. Moving upward means moving
    forward in time. The red node is unknown before the FTCS update is applied.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    space_points = range(5)
    time_levels = range(4)
    center_i = 2
    current_j = 1
    future_j = current_j + 1

    fig, ax = plt.subplots(figsize=(10, 7))

    # Draw the mesh lines. These lines represent the discrete places where the
    # continuous heat equation is sampled.
    for x in space_points:
        ax.plot([x, x], [min(time_levels), max(time_levels)], color="0.82", linewidth=1)
    for t in time_levels:
        ax.plot([min(space_points), max(space_points)], [t, t], color="0.82", linewidth=1)

    # Draw all nodes first as neutral points.
    for t in time_levels:
        for x in space_points:
            ax.scatter(x, t, s=120, color="white", edgecolor="0.35", zorder=3)
            if show_labels:
                ax.text(x, t - 0.16, node_label(x, t), ha="center", va="top", fontsize=10)

    known_nodes = [(center_i - 1, current_j), (center_i, current_j), (center_i + 1, current_j)]
    unknown_node = (center_i, future_j)

    for x, t in known_nodes:
        ax.scatter(x, t, s=260, color="#2f6fdb", edgecolor="black", zorder=5, label="Known values")

    ax.scatter(
        unknown_node[0],
        unknown_node[1],
        s=320,
        color="#d62828",
        edgecolor="black",
        zorder=6,
        label="Unknown future value",
    )

    # Arrows show the formula dependencies:
    # U_i^(j+1) = r U_(i-1)^j + (1 - 2r) U_i^j + r U_(i+1)^j.
    for x, t in known_nodes:
        ax.annotate(
            "",
            xy=unknown_node,
            xytext=(x, t),
            arrowprops=dict(arrowstyle="->", linewidth=2.2, color="#333333"),
            zorder=4,
        )

    ax.annotate("left neighbor", xy=(center_i - 1, current_j), xytext=(0.15, 0.55),
                arrowprops=dict(arrowstyle="->", color="#2f6fdb"), fontsize=11)
    ax.annotate("center", xy=(center_i, current_j), xytext=(1.75, 0.25),
                arrowprops=dict(arrowstyle="->", color="#2f6fdb"), fontsize=11)
    ax.annotate("right neighbor", xy=(center_i + 1, current_j), xytext=(3.15, 0.55),
                arrowprops=dict(arrowstyle="->", color="#2f6fdb"), fontsize=11)
    ax.annotate(
        "future value being predicted",
        xy=unknown_node,
        xytext=(2.35, 2.75),
        arrowprops=dict(arrowstyle="->", color="#d62828"),
        fontsize=11,
        color="#8a1f1f",
    )

    ax.set_title("FTCS Dependency Structure", fontsize=16, weight="bold")
    ax.set_xlabel("Space index i", fontsize=12)
    ax.set_ylabel("Time level j", fontsize=12)
    ax.set_xticks(list(space_points))
    ax.set_yticks(list(time_levels))
    ax.set_xlim(-0.45, 4.45)
    ax.set_ylim(-0.35, 3.25)
    ax.grid(False)

    # Remove duplicate legend entries caused by repeated scatter calls.
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), loc="upper left")

    output_path = OUTPUT_DIR / "ftcs_dependency_structure.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.show()
    return output_path


def main():
    """Create and save the FTCS dependency mesh visualization."""
    saved_path = draw_ftcs_dependency_mesh(show_labels=True)
    print(f"Saved mesh visualization to: {saved_path}")


if __name__ == "__main__":
    main()
