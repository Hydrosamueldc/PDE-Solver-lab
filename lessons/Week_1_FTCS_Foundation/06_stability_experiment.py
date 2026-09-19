"""
06_stability_experiment.py

Educational goal
----------------
This script demonstrates why the FTCS method for the heat equation has a
stability restriction.

For the 1D heat equation, FTCS is stable only when:

    r = alpha*k/h^2 <= 0.5

This is a CFL-type restriction: the time step k must be small enough compared
with the space step h. If r is too large, the numerical method sends too much
information forward in one time step, creating artificial oscillations that can
grow instead of physical diffusion.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs" / "graphs"


def create_hot_center_initial_condition(x):
    """Create a hot center region for the stable and unstable experiments."""
    temperature = np.zeros_like(x)
    temperature[(x >= 0.45) & (x <= 0.55)] = 100.0
    return temperature


def run_ftcs_case(r, number_of_points=81, number_of_steps=80, alpha=1.0):
    """Run one FTCS simulation for a chosen r value."""
    x = np.linspace(0.0, 1.0, number_of_points)
    h = x[1] - x[0]
    k = r * h**2 / alpha

    solution = np.zeros((number_of_steps + 1, number_of_points))
    solution[0] = create_hot_center_initial_condition(x)
    solution[:, 0] = 0.0
    solution[:, -1] = 0.0

    for n in range(number_of_steps):
        solution[n + 1, 1:-1] = (
            r * solution[n, :-2]
            + (1 - 2 * r) * solution[n, 1:-1]
            + r * solution[n, 2:]
        )

    return x, solution, k


def plot_stability_comparison():
    """Plot stable and unstable FTCS behavior side by side."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    stable_r = 0.25
    unstable_r = 0.75
    selected_steps = [0, 10, 20, 40, 80]

    x_stable, stable_solution, stable_k = run_ftcs_case(stable_r)
    x_unstable, unstable_solution, unstable_k = run_ftcs_case(unstable_r)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True)
    colors = plt.cm.coolwarm(np.linspace(0.1, 0.9, len(selected_steps)))

    for step, color in zip(selected_steps, colors):
        axes[0].plot(
            x_stable,
            stable_solution[step],
            color=color,
            linewidth=2,
            label=f"step {step}, t = {step * stable_k:.4f}",
        )
        axes[1].plot(
            x_unstable,
            unstable_solution[step],
            color=color,
            linewidth=2,
            label=f"step {step}, t = {step * unstable_k:.4f}",
        )

    axes[0].set_title("Stable FTCS: r = 0.25")
    axes[0].set_xlabel("Position x")
    axes[0].set_ylabel("Temperature")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)
    axes[0].annotate(
        "Smooth physical diffusion",
        xy=(0.5, stable_solution[40].max()),
        xytext=(0.15, 70),
        arrowprops=dict(arrowstyle="->"),
    )

    axes[1].set_title("Unstable FTCS: r = 0.75")
    axes[1].set_xlabel("Position x")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)
    axes[1].annotate(
        "FTCS becomes unstable when r > 0.5",
        xy=(0.5, unstable_solution[20].max()),
        xytext=(0.06, 120),
        arrowprops=dict(arrowstyle="->"),
        color="#9b2226",
    )

    axes[1].set_ylim(
        min(-120, np.nanmin(unstable_solution[selected_steps])),
        max(160, np.nanmax(unstable_solution[selected_steps])),
    )

    fig.suptitle("Stable vs Unstable FTCS Heat Equation Behavior", fontsize=16, weight="bold")
    fig.text(
        0.5,
        0.01,
        "CFL-type stability restriction: r = alpha*k/h^2 must satisfy r <= 0.5 for the 1D heat equation.",
        ha="center",
        fontsize=11,
    )

    output_path = OUTPUT_DIR / "ftcs_stability_comparison.png"
    fig.tight_layout(rect=[0, 0.04, 1, 0.94])
    fig.savefig(output_path, dpi=200)
    plt.show()
    return output_path


def main():
    """Run the stability experiment and save the comparison plot."""
    saved_path = plot_stability_comparison()
    print(f"Saved stability comparison plot to: {saved_path}")


if __name__ == "__main__":
    main()
