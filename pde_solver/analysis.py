"""Analysis helpers shared by all heat-equation methods."""

import numpy as np
from numpy.typing import NDArray

from .models import SolutionResult


def exact_sine_solution(result: SolutionResult, alpha: float) -> NDArray[np.float64]:
    """Return the exact solution for sin(pi*x) with zero boundaries on [0, 1]."""
    return np.exp(-alpha * np.pi**2 * result.time[:, None]) * np.sin(np.pi * result.x)


def max_absolute_error(result: SolutionResult, exact: NDArray[np.float64]) -> float:
    """Return the largest pointwise difference from an exact solution grid."""
    if result.values.shape != exact.shape:
        raise ValueError("exact must have the same shape as result.values.")
    return float(np.max(np.abs(result.values - exact)))
