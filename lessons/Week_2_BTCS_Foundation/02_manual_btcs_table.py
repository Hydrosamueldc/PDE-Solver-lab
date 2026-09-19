"""
02_manual_btcs_table.py  —  Week 2: BTCS

Educational goal
----------------
Simulates a hand calculation of the BTCS implicit method for the 1D heat equation.

We use a tiny temperature row:

    [0, 40, 80, 40, 0]

Boundary points are fixed at zero. At each time step, we must solve the
tridiagonal system:

    A U^{j+1} = U^j

where A has diagonal (1+2r) and off-diagonals (-r).

Key contrast with FTCS: you cannot compute each future value one at a time.
All interior unknowns at the next level are coupled and must be solved together.
"""

from pathlib import Path

import numpy as np
from scipy.linalg import solve_banded


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs" / "graphs"


def build_banded_matrix(interior_count, r):
    """
    Build the BTCS tridiagonal system in scipy banded format.

    scipy.linalg.solve_banded expects:
        ab[0, 1:]  = upper diagonal (elements shifted right by 1)
        ab[1, :]   = main diagonal
        ab[2, :-1] = lower diagonal (elements shifted left by 1)
    """
    ab = np.zeros((3, interior_count))
    ab[0, 1:]  = -r
    ab[1, :]   = 1.0 + 2.0 * r
    ab[2, :-1] = -r
    return ab


def compute_next_btcs_step(current_values, r):
    """
    Compute one BTCS time step by solving the tridiagonal system.

    The interior unknowns at level j+1 satisfy:
        -r U_{i-1}^{j+1} + (1+2r) U_i^{j+1} - r U_{i+1}^{j+1} = U_i^j

    This is assembled into A U_int^{j+1} = RHS and solved with the
    Thomas algorithm (via scipy.linalg.solve_banded).
    """
    next_values = current_values.copy()
    interior = current_values[1:-1].copy()
    interior_count = len(interior)

    if interior_count == 0:
        return next_values

    rhs = interior.copy()
    rhs[0]  += r * float(current_values[0])
    rhs[-1] += r * float(current_values[-1])

    ab = build_banded_matrix(interior_count, r)
    interior_new = solve_banded((1, 1), ab, rhs)

    print(f"\n  RHS (U_int^j + boundary terms): {rhs}")
    print(f"  Solved U_int^{{j+1}}:             {np.round(interior_new, 4)}")

    next_values[1:-1] = interior_new
    return next_values


def format_table(solution_history):
    """Create a formatted text table showing all time levels."""
    number_of_points = solution_history.shape[1]
    header = "time level | " + " | ".join(f"  U_{i}" for i in range(number_of_points))
    divider = "-" * len(header)
    rows = [header, divider]
    for time_level, values in enumerate(solution_history):
        row = f"j = {time_level:<5} | " + " | ".join(f"{v:7.4f}" for v in values)
        rows.append(row)
    return "\n".join(rows)


def run_manual_btcs_demo():
    """
    Step through BTCS by hand for a small example and print the result table.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    initial = np.array([0.0, 40.0, 80.0, 40.0, 0.0])
    r = 0.5
    number_of_steps = 4

    print("=" * 60)
    print("BTCS Manual Calculation Demo")
    print(f"r = {r}, initial temperatures = {initial}")
    print("=" * 60)
    print(
        "\nBTCS equation:  -r U_{i-1}^{j+1} + (1+2r) U_i^{j+1} - r U_{i+1}^{j+1} = U_i^j"
    )
    print(
        f"\nWith r={r}: diagonals are ({1+2*r}), off-diagonals are ({-r})"
    )

    history = np.zeros((number_of_steps + 1, len(initial)))
    history[0] = initial

    for step in range(number_of_steps):
        print(f"\n{'─'*50}")
        print(f"Step {step} → {step+1}: solving tridiagonal system")
        print(f"  Current: {np.round(history[step], 4)}")
        history[step + 1] = compute_next_btcs_step(history[step], r)
        print(f"  Result:  {np.round(history[step + 1], 4)}")

    print(f"\n{'═'*60}")
    print("FULL SOLUTION TABLE")
    print(f"{'═'*60}")
    print(format_table(history))

    output_path = OUTPUT_DIR / "btcs_manual_table.txt"
    output_path.write_text(format_table(history), encoding="utf-8")
    print(f"\nTable written to: {output_path}")
    return history


if __name__ == "__main__":
    run_manual_btcs_demo()
