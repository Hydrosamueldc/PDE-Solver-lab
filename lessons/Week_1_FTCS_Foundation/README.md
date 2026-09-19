# Week 1: FTCS Heat Equation Learning Module

This folder is a beginner-friendly computational laboratory for the 1D heat
equation using the FTCS finite difference method.

The purpose is not only to compute numbers. The purpose is to help students see
the grid, understand the update formula, perform manual calculations, simulate
heat diffusion, and compare stable and unstable numerical behavior.

## Mathematical Background

The heat equation is

```text
u_t = alpha u_xx
```

It describes diffusion: heat moves from hotter regions toward colder regions.

Using a forward difference in time and a centered difference in space gives

```text
(U_i^(j+1) - U_i^j) / k
= alpha (U_(i+1)^j - 2U_i^j + U_(i-1)^j) / h^2
```

Define

```text
r = alpha*k/h^2
```

Then the FTCS update becomes

```text
U_i^(j+1) = r U_(i-1)^j + (1 - 2r) U_i^j + r U_(i+1)^j
```

For the 1D heat equation, the usual stability condition is:

```text
r <= 0.5
```

## Folder Structure

```text
Week_1_FTCS_Foundation/
├── 01_grid_and_mesh_visualization.py
├── 02_manual_ftcs_table.py
├── 03_ftcs_heat_equation_simulation.py
├── 04_ftcs_3d_surface_plot.py
├── 05_ftcs_animation.py
├── 06_stability_experiment.py
├── notes/
│   ├── ftcs_formula.md
│   ├── grid_notation.md
│   └── interpretation.md
└── outputs/
    ├── mesh_images/
    ├── heatmaps/
    ├── animations/
    └── graphs/
```

## Required Libraries

Install the beginner scientific Python stack:

```bash
pip install numpy matplotlib pillow plotly
```

`pillow` is used by Matplotlib to save the animation as a GIF. `plotly` is used
to build the interactive dashboard with hover, click inspection, a time slider,
and a rotatable 3D surface.

If you are using the project virtual environment created by `uv`, run:

```powershell
uv venv .venv
uv pip install -r requirements.txt
```

## How to Run

From this folder, run:

```bash
python 00_run_question_lab.py
python 01_grid_and_mesh_visualization.py
python 02_manual_ftcs_table.py
python 03_ftcs_heat_equation_simulation.py
python 04_ftcs_3d_surface_plot.py
python 05_ftcs_animation.py
python 06_stability_experiment.py
```

Each script saves its output automatically under `outputs/`.

## Main Practice Workflow

For regular practice, you only need two files:

- `question_config.py`: enter the question parameters here, then run this file.
- `00_run_question_lab.py`: the engine called automatically by the config file.

Run from the project root:

```powershell
.\.venv\Scripts\python.exe Week_1_FTCS_Foundation\question_config.py
```

The runner creates files inside:

```text
outputs/question_results/
```

Generated files include:

- `question_report.md`
- `mesh_grid_points.csv`
- `ftcs_solution_table.txt`
- `ftcs_solution_table.csv`
- `simulation_data_long.csv`
- `solution_profiles.png`
- `solution_heatmap.png`
- `output_dashboard.html`
- `interactive_output_dashboard.html`

When `run_supporting_engines` is `True` in `question_config.py`, the question
runner also generates the mesh dependency image, 3D surface, animation, and
stability comparison from the other lesson scripts.

Use `interactive_output_dashboard.html` as the main result panel. It lets you
rotate the 3D surface, hover over values, click points to inspect coordinates
and temperatures, and move through the solution with a time-step player.

## Learning Objectives

After working through this module, students should be able to:

- Explain what a computational mesh is.
- Interpret the notation `U_i^j`.
- Describe the FTCS dependency pattern.
- Compute a few FTCS updates by hand.
- Simulate 1D heat diffusion with NumPy.
- Read 2D and 3D heat diffusion visualizations.
- Explain why FTCS becomes unstable when `r > 0.5`.

## Screenshots

Add generated images here after running the scripts:

- `outputs/mesh_images/ftcs_dependency_structure.png`
- `outputs/heatmaps/ftcs_heat_diffusion_profiles.png`
- `outputs/graphs/ftcs_space_time_surface.png`
- `outputs/graphs/ftcs_stability_comparison.png`
- `outputs/animations/ftcs_heat_diffusion_animation.gif`

## Educational Philosophy

This project is designed to help students understand Numerical PDEs visually and
computationally. It should feel like an interactive numerical PDE teaching
laboratory: clear, visual, mathematical, and intuitive.
