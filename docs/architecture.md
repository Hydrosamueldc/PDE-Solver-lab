# Architecture

## Purpose

The project has two complementary parts:

- `pde_solver/` is reusable, tested numerical code.
- `lessons/` contains guided teaching material and method-specific visuals.

Keeping these separate lets the future application use the same trusted solver
without inheriting lesson-specific scripts and dashboards.

## Current Scope

The shared engine models the one-dimensional heat equation:

```text
u_t = alpha * u_xx
```

It supports fixed boundary values and the FTCS, BTCS, and Crank-Nicolson
finite-difference schemes.

## Growth Path

1. Migrate the numerical calculations in the existing lessons to `pde_solver`.
2. Add a Week 3 Crank-Nicolson lesson that calls the shared engine.
3. Add common result visualizations and CSV export outside lesson runners.
4. Build a parameter form around `HeatEquationProblem`.
5. Add more equation families only after the 1D heat-equation workflow is
   well-tested.

## Boundaries

- Solver modules must not open browsers, write files, or depend on UI code.
- Visualization and app code consume `SolutionResult` objects from the solver.
- User-entered profiles should use presets or a safe expression parser, never
  unrestricted Python evaluation.
