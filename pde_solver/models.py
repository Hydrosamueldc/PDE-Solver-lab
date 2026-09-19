"""Problem definitions and result objects for the 1D heat equation."""

from dataclasses import dataclass, field
from typing import Callable, Literal

import numpy as np
from numpy.typing import NDArray


InitialCondition = str | NDArray[np.float64] | Callable[[NDArray[np.float64]], NDArray[np.float64]]
MethodName = Literal["ftcs", "btcs", "crank_nicolson"]


@dataclass(frozen=True)
class HeatEquationProblem:
    """Parameters for ``u_t = alpha * u_xx`` with fixed end temperatures.

    Provide exactly one of ``time_step`` and ``r``. The final time is reached
    exactly by making a small adjustment to the last uniform time step when
    necessary; the effective value is available in ``SolutionResult``.
    """

    alpha: float
    x_start: float
    x_end: float
    final_time: float
    space_points: int
    left_boundary: float = 0.0
    right_boundary: float = 0.0
    initial_condition: InitialCondition = "sin_pi"
    time_step: float | None = None
    r: float | None = None

    def validate(self) -> None:
        if self.alpha <= 0:
            raise ValueError("alpha must be positive.")
        if self.x_end <= self.x_start:
            raise ValueError("x_end must be greater than x_start.")
        if self.final_time <= 0:
            raise ValueError("final_time must be positive.")
        if self.space_points < 3:
            raise ValueError("space_points must be at least 3.")
        if (self.time_step is None) == (self.r is None):
            raise ValueError("Provide exactly one of time_step or r.")
        if self.time_step is not None and self.time_step <= 0:
            raise ValueError("time_step must be positive.")
        if self.r is not None and self.r <= 0:
            raise ValueError("r must be positive.")


@dataclass(frozen=True)
class SolutionResult:
    """A solved heat-equation grid and metadata shared by every method."""

    method: MethodName
    x: NDArray[np.float64]
    time: NDArray[np.float64]
    values: NDArray[np.float64]
    space_step: float
    time_step: float
    r: float
    warnings: tuple[str, ...] = field(default_factory=tuple)
