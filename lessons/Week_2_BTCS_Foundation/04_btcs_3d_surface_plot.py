"""
04_btcs_3d_surface_plot.py  —  Week 2: BTCS

Generates a 3D surface plot of the BTCS heat equation solution.
Space on the x-axis, time on the y-axis, temperature as the surface height.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import solve_banded


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs" / "graphs"


def btcs_simulate(x, time_values, r, initial_values, left_bc=0.0, right_bc=0.0):
    N = len(x)
    M_int = N - 2
    solution = np.zeros((len(time_values), N))
    solution[0] = initial_values
    solution[:, 0]  = left_bc
    solution[:, -1] = right_bc
    if M_int <= 0:
        return solution
    ab = np.zeros((3, M_int))
    ab[0, 1:]  = -r
    ab[1, :]   = 1.0 + 2.0 * r
    ab[2, :-1] = -r
    for n in range(len(time_values) - 1):
        rhs = solution[n, 1:-1].copy()
        rhs[0]  += r * left_bc
        rhs[-1] += r * right_bc
        solution[n + 1, 1:-1] = solve_banded((1, 1), ab, rhs)
        solution[n + 1, 0]  = left_bc
        solution[n + 1, -1] = right_bc
    return solution


def run_3d_surface_plot():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    x = np.linspace(0, 1, 21)
    h = x[1] - x[0]
    alpha = 1.0
    r = 1.0
    k = r * h**2 / alpha
    T = 0.2
    M = int(round(T / k))
    time_values = np.linspace(0, T, M + 1)
    initial = np.sin(np.pi * x)

    solution = btcs_simulate(x, time_values, r, initial)

    X, T_grid = np.meshgrid(x, time_values)

    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(X, T_grid, solution, cmap="plasma", edgecolor="none", alpha=0.85)
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label="Temperature")
    ax.set_xlabel("Position x")
    ax.set_ylabel("Time t")
    ax.set_zlabel("Temperature")
    ax.set_title(f"BTCS 3D Heat Surface\nsin(πx) IC, r={r} (unconditionally stable)")
    ax.view_init(elev=28, azim=-55)

    fig.tight_layout()
    output_path = OUTPUT_DIR / "btcs_3d_surface.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"BTCS 3D surface saved: {output_path}")
    return output_path


if __name__ == "__main__":
    run_3d_surface_plot()
