"""
06_stability_experiment.py  —  Week 2: BTCS

BTCS Unconditional Stability Demonstration
===========================================

Runs the SAME heat equation problem at many different r values:
  - r = 0.5 (FTCS limit — both methods work)
  - r = 1    (BTCS fine, FTCS just stable)
  - r = 2    (BTCS fine, FTCS unstable — would blow up)
  - r = 5    (BTCS fine, FTCS completely unusable)
  - r = 20   (BTCS fine but low temporal accuracy)

All BTCS solutions stay physically meaningful. The exact solution is shown
as a dashed reference so you can see how accuracy degrades at large r.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import solve_banded


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs" / "stability"


def btcs_solve(x, r, T, initial, left_bc=0.0, right_bc=0.0, alpha=1.0):
    """Run BTCS from t=0 to t=T and return the final-time solution."""
    h = x[1] - x[0]
    k = r * h**2 / alpha
    M = max(1, int(round(T / k)))
    time_values = np.linspace(0, T, M + 1)

    N = len(x)
    M_int = N - 2
    solution = np.zeros((len(time_values), N))
    solution[0] = initial
    solution[:, 0]  = left_bc
    solution[:, -1] = right_bc

    if M_int <= 0:
        return solution, time_values

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

    return solution, time_values


def run_stability_experiment():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    x = np.linspace(0, 1, 21)
    T = 0.1
    alpha = 1.0
    initial = np.sin(np.pi * x)
    exact_final = np.exp(-alpha * np.pi**2 * T) * np.sin(np.pi * x)

    r_values = [0.5, 1.0, 2.0, 5.0, 20.0]
    colors   = ["#22c55e", "#84cc16", "#f59e0b", "#ef4444", "#7c3aed"]

    fig, axes = plt.subplots(1, len(r_values), figsize=(18, 4), sharey=True)
    fig.patch.set_facecolor("#1f2937")
    fig.suptitle(
        "BTCS Unconditional Stability — All r Values Remain Stable\n"
        "(dashed = exact solution)",
        color="white", fontsize=13,
    )

    for ax, r, color in zip(axes, r_values, colors):
        ax.set_facecolor("#111827")
        solution, _ = btcs_solve(x, r, T, initial, alpha=alpha)
        ax.plot(x, initial, color="#374151", linewidth=1.5, linestyle=":", label="IC")
        ax.plot(x, solution[-1], color=color, linewidth=2.5, label=f"r={r}")
        ax.plot(x, exact_final, color="white", linewidth=1.5, linestyle="--", alpha=0.7, label="Exact")
        ax.set_title(f"r = {r}", color=color, fontweight="bold")
        ax.set_xlabel("x", color="white")
        ax.tick_params(colors="white")
        ax.spines[:].set_color("#374151")
        ax.legend(fontsize=7, facecolor="#1f2937", edgecolor="#374151", labelcolor="white")

    axes[0].set_ylabel("Temperature", color="white")
    fig.tight_layout()
    output_path = OUTPUT_DIR / "btcs_stability_experiment.png"
    fig.savefig(output_path, dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"BTCS stability experiment saved: {output_path}")

    # Print max errors
    print(f"\n{'r':>6}  {'Steps':>8}  {'Max error':>14}")
    print("-" * 32)
    for r in r_values:
        sol, tvs = btcs_solve(x, r, T, initial, alpha=alpha)
        err = float(np.max(np.abs(sol[-1] - exact_final)))
        h = x[1] - x[0]
        k = r * h**2 / alpha
        M = max(1, int(round(T / k)))
        print(f"{r:>6.1f}  {M:>8d}  {err:>14.6e}")

    return output_path


if __name__ == "__main__":
    run_stability_experiment()
