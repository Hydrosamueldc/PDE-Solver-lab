"""
03_ftcs_heat_equation_simulation.py

Educational goal
----------------
This script creates a visual simulation of 1D heat diffusion.

The heat equation

    u_t = alpha u_xx

models the way temperature smooths out over time. If the center is hot and the
ends are cold, heat flows from the center toward the colder parts. The heat
profile smooths over time due to diffusion.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs" / "heatmaps"


def create_hot_center_initial_condition(x):
    """Create an initial temperature profile with a hot center region."""
    temperature = np.zeros_like(x)
    temperature[(x >= 0.4) & (x <= 0.6)] = 100.0
    return temperature


def ftcs_heat_simulation(number_of_points=81, number_of_steps=160, alpha=1.0, r=0.4):
    """
    Simulate the 1D heat equation using FTCS.

    We choose k from r = alpha*k/h^2 so the update has the desired stability
    behavior. For a stable FTCS heat simulation, r should be less than or equal
    to 0.5.
    """
    x = np.linspace(0.0, 1.0, number_of_points)
    h = x[1] - x[0]
    k = r * h**2 / alpha

    solution = np.zeros((number_of_steps + 1, number_of_points))
    solution[0] = create_hot_center_initial_condition(x)

    # Boundary conditions: the ends are held at zero temperature.
    solution[:, 0] = 0.0
    solution[:, -1] = 0.0

    for n in range(number_of_steps):
        for i in range(1, number_of_points - 1):
            solution[n + 1, i] = (
                r * solution[n, i - 1]
                + (1 - 2 * r) * solution[n, i]
                + r * solution[n, i + 1]
            )

    return x, solution, k


def plot_selected_time_profiles(x, solution, k):
    """Plot several time levels so students can see the smoothing process."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    selected_steps = [0, 10, 30, 60, 100, 160]
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, len(selected_steps)))

    fig, ax = plt.subplots(figsize=(10, 6))

    for step, color in zip(selected_steps, colors):
        ax.plot(
            x,
            solution[step],
            color=color,
            linewidth=2,
            label=f"step {step}, t = {step * k:.4f}",
        )

    ax.set_title("1D Heat Diffusion using FTCS", fontsize=16, weight="bold")
    ax.set_xlabel("Position x")
    ax.set_ylabel("Temperature")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.text(
        0.02,
        0.95,
        "The hot center spreads out and cools as heat diffuses.",
        transform=ax.transAxes,
        va="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="0.75"),
    )

    output_path = OUTPUT_DIR / "ftcs_heat_diffusion_profiles.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.show()
    return output_path


def main():
    """Run and save the 1D FTCS heat diffusion line-plot visualization."""
    x, solution, k = ftcs_heat_simulation()
    saved_path = plot_selected_time_profiles(x, solution, k)
    print(f"Saved heat diffusion plot to: {saved_path}")


if __name__ == "__main__":
    main()
