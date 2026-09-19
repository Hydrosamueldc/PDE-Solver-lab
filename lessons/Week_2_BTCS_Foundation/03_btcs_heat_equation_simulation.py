"""
03_btcs_heat_equation_simulation.py  —  Week 2: BTCS

Standard BTCS heat equation simulation with a Gaussian initial condition.
Saves static profile plots and heatmap to outputs/graphs/.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import solve_banded


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs" / "graphs"


def btcs_simulate(x, time_values, r, initial_values, left_bc=0.0, right_bc=0.0):
    """Run BTCS simulation and return the solution array (time × space)."""
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


def run_btcs_simulation():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Gaussian initial condition
    x = np.linspace(0, 1, 21)
    h = x[1] - x[0]
    alpha = 1.0
    r = 1.0
    k = r * h**2 / alpha
    T = 0.2
    M = int(round(T / k))
    time_values = np.linspace(0, T, M + 1)
    initial = np.exp(-50 * (x - 0.5)**2)

    solution = btcs_simulate(x, time_values, r, initial)

    # Profile plot
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#1f2937")
    ax.set_facecolor("#111827")
    palette = ["#e63946", "#f4a261", "#2a9d8f", "#457b9d", "#a8dadc", "#ffffff"]
    selected = np.linspace(0, len(time_values) - 1, 6, dtype=int)
    for idx, j in enumerate(selected):
        ax.plot(x, solution[j], color=palette[idx % len(palette)], linewidth=2,
                label=f"t={time_values[j]:.4f}", marker="o", markersize=3)
    ax.set_title(f"BTCS Heat Diffusion (Gaussian IC, r={r})", color="white")
    ax.set_xlabel("Position x", color="white")
    ax.set_ylabel("Temperature", color="white")
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#374151")
    ax.grid(True, alpha=0.3, color="#4b5563")
    ax.legend(facecolor="#1f2937", edgecolor="#374151", labelcolor="white")
    fig.tight_layout()
    path1 = OUTPUT_DIR / "btcs_simulation_profiles.png"
    fig.savefig(path1, dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)

    # Heatmap
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#1f2937")
    ax.set_facecolor("#111827")
    im = ax.imshow(solution, aspect="auto", origin="lower", cmap="inferno",
                   extent=[x[0], x[-1], time_values[0], time_values[-1]])
    plt.colorbar(im, ax=ax, label="Temperature").ax.yaxis.label.set_color("white")
    ax.set_title(f"BTCS Space-Time Heatmap (r={r}, unconditionally stable)", color="white")
    ax.set_xlabel("Position x", color="white")
    ax.set_ylabel("Time t", color="white")
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#374151")
    fig.tight_layout()
    path2 = OUTPUT_DIR / "btcs_simulation_heatmap.png"
    fig.savefig(path2, dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)

    print(f"BTCS simulation complete. Profiles: {path1}  Heatmap: {path2}")
    return solution


if __name__ == "__main__":
    run_btcs_simulation()
