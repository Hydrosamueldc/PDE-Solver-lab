"""
05_btcs_animation.py  —  Week 2: BTCS

Animated GIF of the BTCS heat diffusion solution.
Frame-by-frame advance through time levels.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.animation as animation
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


def run_btcs_animation():
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

    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor("#1f2937")
    ax.set_facecolor("#111827")
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(-0.1, 1.1)
    ax.set_xlabel("Position x", color="white")
    ax.set_ylabel("Temperature", color="white")
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#374151")

    line, = ax.plot([], [], "o-", color="#e63946", linewidth=2, markersize=5)
    title = ax.set_title("", color="white")

    def init():
        line.set_data([], [])
        return line,

    def update(frame):
        line.set_data(x, solution[frame])
        title.set_text(f"BTCS j={frame}, t={time_values[frame]:.5f}, r={r}")
        return line, title

    frames = list(range(len(time_values)))
    ani = animation.FuncAnimation(
        fig, update, frames=frames, init_func=init,
        interval=150, blit=True,
    )

    output_path = OUTPUT_DIR / "btcs_animation.gif"
    try:
        ani.save(str(output_path), writer="pillow", fps=8)
        print(f"BTCS animation saved: {output_path}")
    except Exception as exc:
        print(f"Animation save skipped ({exc}) — Pillow may not be installed.")
    plt.close(fig)
    return output_path


if __name__ == "__main__":
    run_btcs_animation()
