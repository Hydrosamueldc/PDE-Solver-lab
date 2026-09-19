"""
05_ftcs_animation.py

Educational goal
----------------
This script animates heat diffusion in one space dimension.

Animation is useful because the heat equation is a time-evolution problem. Each
frame represents one saved time level. As the frames advance, students can see
the initially hot center smooth out and spread toward cooler regions.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs" / "animations"


def create_hot_center_initial_condition(x):
    """Create the initial hot center used by the animation."""
    temperature = np.zeros_like(x)
    temperature[(x >= 0.4) & (x <= 0.6)] = 100.0
    return temperature


def simulate_heat_equation(number_of_points=101, number_of_steps=180, alpha=1.0, r=0.4):
    """Compute all animation frames using the FTCS update rule."""
    x = np.linspace(0.0, 1.0, number_of_points)
    h = x[1] - x[0]
    k = r * h**2 / alpha

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

    return x, temperature, k


def create_heat_animation(x, temperature, k, fps=15):
    """
    Create and save a GIF animation of the FTCS heat equation solution.

    The frame index is the time-step number. The moving title helps connect the
    visual frame to the numerical time level.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    line, = ax.plot([], [], color="#d62828", linewidth=3)

    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(-5, 110)
    ax.set_xlabel("Position x")
    ax.set_ylabel("Temperature")
    ax.grid(True, alpha=0.3)

    explanation = ax.text(
        0.02,
        0.92,
        "Heat spreads from hot points toward cooler neighboring points.",
        transform=ax.transAxes,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="0.75"),
    )

    def initialize():
        """Set up the first empty frame."""
        line.set_data([], [])
        ax.set_title("FTCS Heat Diffusion Animation")
        return line, explanation

    def update(frame_index):
        """Update the curve for one animation frame."""
        line.set_data(x, temperature[frame_index])
        ax.set_title(
            f"FTCS Heat Diffusion Animation | step {frame_index}, t = {frame_index * k:.4f}"
        )
        return line, explanation

    frame_step = 2
    frames = range(0, len(temperature), frame_step)
    animation = FuncAnimation(
        fig,
        update,
        frames=frames,
        init_func=initialize,
        blit=True,
        interval=1000 / fps,
    )

    output_path = OUTPUT_DIR / "ftcs_heat_diffusion_animation.gif"
    animation.save(output_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return output_path


def main(fps=15):
    """Run the simulation and save the heat diffusion animation."""
    x, temperature, k = simulate_heat_equation()
    saved_path = create_heat_animation(x, temperature, k, fps=fps)
    print(f"Saved animation to: {saved_path}")


if __name__ == "__main__":
    main()
