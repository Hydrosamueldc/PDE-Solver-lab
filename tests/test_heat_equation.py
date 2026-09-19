"""Focused checks for the shared heat-equation solver."""

import unittest

import numpy as np

from pde_solver import HeatEquationProblem, exact_sine_solution, max_absolute_error, solve_heat_equation


class HeatEquationSolverTests(unittest.TestCase):
    def setUp(self) -> None:
        self.problem = HeatEquationProblem(
            alpha=1.0,
            x_start=0.0,
            x_end=1.0,
            final_time=0.05,
            space_points=41,
            r=0.25,
            initial_condition="sin_pi",
        )

    def test_all_methods_preserve_fixed_boundaries(self) -> None:
        for method in ("ftcs", "btcs", "crank_nicolson"):
            result = solve_heat_equation(self.problem, method)
            np.testing.assert_allclose(result.values[:, 0], 0.0)
            np.testing.assert_allclose(result.values[:, -1], 0.0)
            self.assertAlmostEqual(result.time[-1], self.problem.final_time)

    def test_implicit_methods_match_sine_solution(self) -> None:
        for method in ("btcs", "crank_nicolson"):
            result = solve_heat_equation(self.problem, method)
            error = max_absolute_error(result, exact_sine_solution(result, self.problem.alpha))
            self.assertLess(error, 0.01)

    def test_ftcs_reports_an_unstable_mesh(self) -> None:
        unstable_problem = HeatEquationProblem(
            alpha=1.0,
            x_start=0.0,
            x_end=1.0,
            final_time=0.015,
            space_points=11,
            r=0.75,
        )
        result = solve_heat_equation(unstable_problem, "ftcs")
        self.assertTrue(any("unstable" in warning.lower() for warning in result.warnings))

    def test_crank_nicolson_error_decreases_on_a_refined_grid(self) -> None:
        coarse_problem = HeatEquationProblem(
            alpha=1.0, x_start=0.0, x_end=1.0, final_time=0.05,
            space_points=21, r=0.25, initial_condition="sin_pi",
        )
        fine_problem = HeatEquationProblem(
            alpha=1.0, x_start=0.0, x_end=1.0, final_time=0.05,
            space_points=41, r=0.25, initial_condition="sin_pi",
        )
        coarse_result = solve_heat_equation(coarse_problem, "crank_nicolson")
        fine_result = solve_heat_equation(fine_problem, "crank_nicolson")
        coarse_error = max_absolute_error(coarse_result, exact_sine_solution(coarse_result, 1.0))
        fine_error = max_absolute_error(fine_result, exact_sine_solution(fine_result, 1.0))
        self.assertLess(fine_error, coarse_error)

    def test_time_step_input_derives_the_mesh_ratio(self) -> None:
        problem = HeatEquationProblem(
            alpha=2.0,
            x_start=0.0,
            x_end=1.0,
            final_time=0.02,
            space_points=11,
            time_step=0.002,
        )
        result = solve_heat_equation(problem, "crank_nicolson")
        self.assertAlmostEqual(result.space_step, 0.1)
        self.assertAlmostEqual(result.time_step, 0.002)
        self.assertAlmostEqual(result.r, 0.4)


if __name__ == "__main__":
    unittest.main()
