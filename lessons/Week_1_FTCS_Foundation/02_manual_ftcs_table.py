"""
02_manual_ftcs_table.py

Educational goal
----------------
This script simulates a hand calculation of the FTCS method.

We use a tiny temperature row:

    [0, 40, 80, 40, 0]

The boundary points are fixed at zero temperature. The hot center point diffuses
heat toward the colder neighboring points. The FTCS update is:

    U_i^(j+1) = r U_(i-1)^j + (1 - 2r) U_i^j + r U_(i+1)^j

The important idea is that each future value depends on neighboring previous
values. This is why FTCS is called an explicit method.
"""

from pathlib import Path

import numpy as np


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs" / "graphs"


def compute_next_ftcs_step(current_values, r):
    """
    Compute one FTCS time step for a 1D heat equation.

    Parameters
    ----------
    current_values:
        Temperatures at the current time level.
    r:
        The mesh ratio alpha*k/h^2. For FTCS heat diffusion, r controls how
        much influence the neighboring points have.

    Returns
    -------
    numpy.ndarray
        Temperatures at the next time level.
    """
    next_values = current_values.copy()

    # Boundary values stay fixed. Only interior values are updated.
    for i in range(1, len(current_values) - 1):
        left_neighbor = current_values[i - 1]
        center_value = current_values[i]
        right_neighbor = current_values[i + 1]

        print(
            f"Computing U_{i} at next time level: "
            f"{r}({left_neighbor:.2f}) + (1 - 2*{r})({center_value:.2f}) "
            f"+ {r}({right_neighbor:.2f})"
        )

        next_values[i] = (
            r * left_neighbor
            + (1 - 2 * r) * center_value
            + r * right_neighbor
        )

    return next_values


def format_table(solution_history):
    """Create a formatted text table showing time levels and node values."""
    number_of_points = solution_history.shape[1]
    header = "time level | " + " | ".join(f"U_{i}" for i in range(number_of_points))
    divider = "-" * len(header)
    rows = [header, divider]

    for time_level, values in enumerate(solution_history):
        row = f"j = {time_level:<5} | " + " | ".join(f"{value:7.2f}" for value in values)
        rows.append(row)

    return "\n".join(rows)


def run_manual_example(r=0.25, number_of_steps=3):
    """
    Run three beginner-friendly FTCS updates and save the numerical table.

    Heat is diffusing from hotter regions to colder regions, so the 80 degree
    center value should decrease while the nearby cooler points receive heat.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    initial_values = np.array([0, 40, 80, 40, 0], dtype=float)
    solution_history = [initial_values]

    print("Manual FTCS heat equation example")
    print("Boundary conditions: U_0^j = 0 and U_4^j = 0")
    print(f"Using r = {r}")
    print("Heat is diffusing from hotter regions to colder regions.\n")

    current_values = initial_values
    for step in range(1, number_of_steps + 1):
        print(f"--- Computing time level j = {step} ---")
        current_values = compute_next_ftcs_step(current_values, r)
        solution_history.append(current_values)
        print()

    solution_history = np.array(solution_history)
    table_text = format_table(solution_history)

    print("Final table:")
    print(table_text)

    output_path = OUTPUT_DIR / "manual_ftcs_table.txt"
    output_path.write_text(table_text + "\n", encoding="utf-8")
    print(f"\nSaved numerical table to: {output_path}")
    return solution_history, output_path


def main():
    """Run the manual FTCS table example."""
    run_manual_example()


if __name__ == "__main__":
    main()
