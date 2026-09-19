# Numerical Methods for PDE

An educational numerical-PDE project growing into a parameter-driven solver for
the one-dimensional heat equation.

> Status: Active development. This is an early, working foundation rather than
> a finished general-purpose PDE package. The current implementation focuses on
> the 1D heat equation while the architecture is being expanded method by method
> and equation by equation.

## Roadmap

- Completed: shared 1D heat-equation engine with FTCS, BTCS, and
  Crank-Nicolson solvers.
- Completed: interactive solve lab, educational method steps, validation tables,
  visualizations, and method comparison.
- Next: improve the landing-page navigation and comparison experience, then add
  convergence studies and broader PDE models.
- Future: support additional discretization families, including finite volume
  and finite element methods, plus higher-dimensional equations.

## Project Layout

```text
Numerical Methods For PDE/
|-- pde_solver/       Reusable numerical engine
|-- lessons/          Guided FTCS, BTCS, and future Crank-Nicolson modules
|-- examples/         Small runnable uses of the solver API
|-- tests/            Automated solver checks
|-- docs/             Architecture and product notes
|-- notebooks/        Exploratory notebooks
|-- images/           Documentation images
`-- requirements.txt  Python dependencies
```

The `pde_solver` package is the product foundation. It currently solves the
1D heat equation with FTCS, BTCS, and Crank-Nicolson methods. The material in
`lessons/` remains focused on explaining the numerical ideas visually, with
Week 3 using the shared Crank-Nicolson solver directly.

## Quick Start

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m examples.compare_methods
.\.venv\Scripts\python.exe apps\pde_solver_home.py
.\.venv\Scripts\python.exe apps\crank_nicolson_lab.py
```

## Solver API

```python
from pde_solver import HeatEquationProblem, solve_heat_equation

problem = HeatEquationProblem(
    alpha=1.0,
    x_start=0.0,
    x_end=1.0,
    final_time=0.1,
    space_points=51,
    r=0.25,
    initial_condition="sin_pi",
)

result = solve_heat_equation(problem, method="crank_nicolson")
```

`result` contains the space grid, time grid, solution matrix, effective mesh
steps, stability ratio, and warnings. Built-in initial profiles are `sin_pi`,
`gaussian`, `hot_center`, and `zero`; a NumPy array or callable can also be
supplied.

See `docs/architecture.md` for the intended path from this solver core to a
user-facing PDE application.
