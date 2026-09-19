"""Compare all currently supported heat-equation methods on one problem."""

from pde_solver import HeatEquationProblem, exact_sine_solution, max_absolute_error, solve_heat_equation


def main() -> None:
    problem = HeatEquationProblem(
        alpha=1.0,
        x_start=0.0,
        x_end=1.0,
        final_time=0.1,
        space_points=51,
        r=0.25,
        initial_condition="sin_pi",
    )

    print("method            r        max absolute error")
    print("----------------------------------------------")
    for method in ("ftcs", "btcs", "crank_nicolson"):
        result = solve_heat_equation(problem, method)
        exact = exact_sine_solution(result, problem.alpha)
        error = max_absolute_error(result, exact)
        print(f"{method:16} {result.r:0.5f}  {error:.6e}")


if __name__ == "__main__":
    main()
