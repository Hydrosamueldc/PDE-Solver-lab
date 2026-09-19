"""Safe initial-condition construction for the heat-equation solver."""

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from .models import InitialCondition


def build_initial_condition(
    condition: InitialCondition, x: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Build a profile from a preset, array, or callable without ``eval``."""
    if isinstance(condition, str):
        if condition == "sin_pi":
            values = np.sin(np.pi * x)
        elif condition == "gaussian":
            center = (x[0] + x[-1]) / 2.0
            width = (x[-1] - x[0]) / 10.0
            values = np.exp(-((x - center) / width) ** 2)
        elif condition == "hot_center":
            values = np.zeros_like(x)
            center = (x[0] + x[-1]) / 2.0
            half_width = (x[-1] - x[0]) / 10.0
            values[np.abs(x - center) <= half_width] = 100.0
        elif condition == "zero":
            values = np.zeros_like(x)
        else:
            raise ValueError(
                "Unknown initial condition. Use sin_pi, gaussian, hot_center, zero, "
                "a NumPy array, or a callable."
            )
    elif isinstance(condition, np.ndarray):
        values = condition.astype(float, copy=True)
    elif isinstance(condition, Callable):
        values = np.asarray(condition(x), dtype=float)
    else:
        raise TypeError("initial_condition must be a preset name, NumPy array, or callable.")

    if values.shape != x.shape:
        raise ValueError("Initial-condition values must have one value per space point.")
    return values
