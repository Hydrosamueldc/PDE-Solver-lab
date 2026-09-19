"""Finite-difference solvers for the one-dimensional heat equation."""

import math

import numpy as np
from numpy.typing import NDArray

from .conditions import build_initial_condition
from .models import HeatEquationProblem, MethodName, SolutionResult


def solve_heat_equation(
    problem: HeatEquationProblem, method: MethodName
) -> SolutionResult:
    """Solve a 1D heat problem with FTCS, BTCS, or Crank-Nicolson."""
    problem.validate()
    if method not in {"ftcs", "btcs", "crank_nicolson"}:
        raise ValueError("method must be ftcs, btcs, or crank_nicolson.")

    x = np.linspace(problem.x_start, problem.x_end, problem.space_points)
    space_step = float(x[1] - x[0])
    requested_time_step = (
        problem.time_step
        if problem.time_step is not None
        else float(problem.r) * space_step**2 / problem.alpha
    )
    steps = math.ceil(problem.final_time / requested_time_step)
    time_step = problem.final_time / steps
    r = problem.alpha * time_step / space_step**2
    time = np.linspace(0.0, problem.final_time, steps + 1)

    values = np.zeros((steps + 1, problem.space_points), dtype=float)
    values[0] = build_initial_condition(problem.initial_condition, x)
    values[:, 0] = problem.left_boundary
    values[:, -1] = problem.right_boundary

    if method == "ftcs":
        _solve_ftcs(values, r)
    elif method == "btcs":
        _solve_btcs(values, r, problem.left_boundary, problem.right_boundary)
    else:
        _solve_crank_nicolson(values, r, problem.left_boundary, problem.right_boundary)

    warnings: list[str] = []
    if method == "ftcs" and r > 0.5:
        warnings.append("FTCS is unstable for this mesh because r is greater than 0.5.")
    if not math.isclose(time_step, requested_time_step, rel_tol=1e-12, abs_tol=1e-15):
        warnings.append("The time step was adjusted slightly so the final time is reached exactly.")

    return SolutionResult(
        method=method,
        x=x,
        time=time,
        values=values,
        space_step=space_step,
        time_step=time_step,
        r=r,
        warnings=tuple(warnings),
    )


def _solve_ftcs(values: NDArray[np.float64], r: float) -> None:
    for step in range(values.shape[0] - 1):
        previous = values[step]
        values[step + 1, 1:-1] = (
            r * previous[:-2] + (1.0 - 2.0 * r) * previous[1:-1] + r * previous[2:]
        )


def _solve_btcs(
    values: NDArray[np.float64], r: float, left_boundary: float, right_boundary: float
) -> None:
    interior_count = values.shape[1] - 2
    diagonal = np.full(interior_count, 1.0 + 2.0 * r)
    lower = np.full(interior_count - 1, -r)
    upper = np.full(interior_count - 1, -r)

    for step in range(values.shape[0] - 1):
        rhs = values[step, 1:-1].copy()
        rhs[0] += r * left_boundary
        rhs[-1] += r * right_boundary
        values[step + 1, 1:-1] = _solve_tridiagonal(lower, diagonal, upper, rhs)


def _solve_crank_nicolson(
    values: NDArray[np.float64], r: float, left_boundary: float, right_boundary: float
) -> None:
    interior_count = values.shape[1] - 2
    diagonal = np.full(interior_count, 1.0 + r)
    lower = np.full(interior_count - 1, -r / 2.0)
    upper = np.full(interior_count - 1, -r / 2.0)

    for step in range(values.shape[0] - 1):
        previous = values[step]
        rhs = (
            (r / 2.0) * previous[:-2]
            + (1.0 - r) * previous[1:-1]
            + (r / 2.0) * previous[2:]
        )
        rhs[0] += (r / 2.0) * left_boundary
        rhs[-1] += (r / 2.0) * right_boundary
        values[step + 1, 1:-1] = _solve_tridiagonal(lower, diagonal, upper, rhs)


def _solve_tridiagonal(
    lower: NDArray[np.float64],
    diagonal: NDArray[np.float64],
    upper: NDArray[np.float64],
    rhs: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Solve a tridiagonal linear system with the Thomas algorithm."""
    diagonal = diagonal.copy()
    rhs = rhs.copy()

    for index in range(1, len(diagonal)):
        multiplier = lower[index - 1] / diagonal[index - 1]
        diagonal[index] -= multiplier * upper[index - 1]
        rhs[index] -= multiplier * rhs[index - 1]

    solution = np.empty_like(rhs)
    solution[-1] = rhs[-1] / diagonal[-1]
    for index in range(len(diagonal) - 2, -1, -1):
        solution[index] = (rhs[index] - upper[index] * solution[index + 1]) / diagonal[index]
    return solution
