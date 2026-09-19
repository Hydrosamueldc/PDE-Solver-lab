"""Reusable solvers for introductory one-dimensional PDE problems."""

from .analysis import exact_sine_solution, max_absolute_error
from .models import HeatEquationProblem, SolutionResult
from .solvers import solve_heat_equation

__all__ = [
    "HeatEquationProblem",
    "SolutionResult",
    "exact_sine_solution",
    "max_absolute_error",
    "solve_heat_equation",
]
