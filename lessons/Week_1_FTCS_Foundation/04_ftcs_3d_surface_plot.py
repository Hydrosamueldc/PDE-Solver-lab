"""
04_ftcs_3d_surface_plot.py

Educational goal
----------------
This script visualizes the heat equation solution as a space-time surface.

The x-axis is position, the y-axis is time, and the z-axis is temperature. This
surface shows how the temperature changes simultaneously across space and time.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs" / "graphs"


def create_hot_center_initial_condition(x):
    """Create a block of heat in the center of the domain."""
    temperature = np.zeros_like(x)
    temperature[(x >= 0.4) & (x <= 0.6)] = 100.0
    return temperature


def simulate_heat_equation(number_of_points=81, number_of_steps=140, alpha=1.0, r=0.4):
    """Return x values, time values, and the FTCS solution array."""
    x = np.linspace(0.0, 1.0, number_of_points)
    h = x[1] - x[0]
    k = r * h**2 / alpha
    time_values = np.arange(number_of_steps + 1) * k

    temperature = np.zeros((number_of_steps + 1, number_of_points))
    temperature[0] = create_hot_center_initial_condition(x)
    temperature[:, 0] = 0.0
    temperature[:, -1] = 0.0

    for n in range(number_of_steps):
        temperature[n + 1, 1:-1] = (
            r * temperature[n, :-2]
            + (1 - 2 * r) * temperature[n, 1:-1]
            + r * temperature[n, 2:]
        )

    return x, time_values, temperature


def plot_space_time_surface(x, time_values, temperature):
    """Create and save a 3D surface plot of temperature over space and time."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # meshgrid turns 1D coordinate arrays into 2D coordinate surfaces so that
    # every temperature value has a matching (space, time) location.
    space_grid, time_grid = np.meshgrid(x, time_values)

    fig = plt.figure(figsize=(11, 7))
    ax = fig.add_subplot(111, projection="3d")

    surface = ax.plot_surface(
        space_grid,
        time_grid,
        temperature,
        cmap="plasma",
        linewidth=0,
        antialiased=True,
        alpha=0.95,
    )

    ax.set_title("Space-Time Evolution of Heat Distribution", fontsize=15, weight="bold")
    ax.set_xlabel("Space x")
    ax.set_ylabel("Time t")
    ax.set_zlabel("Temperature")
    ax.view_init(elev=28, azim=-130)

    colorbar = fig.colorbar(surface, ax=ax, shrink=0.65, pad=0.1)
    colorbar.set_label("Temperature")

    output_path = OUTPUT_DIR / "ftcs_space_time_surface.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.show()
    return output_path


def main():
    """Run the FTCS simulation and save the 3D surface plot."""
    x, time_values, temperature = simulate_heat_equation()
    saved_path = plot_space_time_surface(x, time_values, temperature)
    print(f"Saved 3D surface plot to: {saved_path}")


if __name__ == "__main__":
    main()
