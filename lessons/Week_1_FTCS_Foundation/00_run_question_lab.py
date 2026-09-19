"""
00_run_question_lab.py

This is the main runner for your FTCS practice questions.

Workflow
--------
1. Open question_config.py.
2. Enter the question parameters.
3. Run this file (or run question_config.py directly).
4. Check outputs/question_results/.

This file turns one heat-equation question into:
- a mesh grid file
- a numerical solution table
- simulation data
- plots and an iron-rod animation
- error analysis graph (when an exact solution is supplied)
- a Markdown report
- an interactive HTML dashboard

Mathematical model
------------------
The 1D heat equation is:

    u_t = alpha u_xx

The FTCS scheme is:

    U_i^(j+1) = r U_(i-1)^j + (1 - 2r) U_i^j + r U_(i+1)^j

where:

    r = alpha*k/h^2

For the 1D heat equation, FTCS is stable when:

    r <= 0.5
"""

from pathlib import Path
import json
import os
import subprocess
import sys
import webbrowser

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs
from plotly.utils import PlotlyJSONEncoder

from question_config import QUESTION


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs" / "question_results"


# ---------------------------------------------------------------------------
# Validation and mesh
# ---------------------------------------------------------------------------

def validate_question(question):
    """Check that the question contains enough information for FTCS."""
    required_keys = [
        "title",
        "alpha",
        "x_start",
        "x_end",
        "final_time",
        "number_of_space_points",
        "left_boundary",
        "right_boundary",
        "initial_condition_type",
    ]

    missing_keys = [key for key in required_keys if key not in question]
    if missing_keys:
        raise ValueError(f"Missing required question fields: {missing_keys}")

    if question["number_of_space_points"] < 3:
        raise ValueError("number_of_space_points must be at least 3.")

    if question["x_end"] <= question["x_start"]:
        raise ValueError("x_end must be greater than x_start.")

    if question["final_time"] <= 0:
        raise ValueError("final_time must be positive.")

    has_r = question.get("r") is not None
    has_time_step = question.get("time_step") is not None

    if has_r == has_time_step:
        raise ValueError('Provide exactly one of "r" or "time_step".')


def build_mesh(question):
    """
    Build the space and time mesh.

    h is the space step. k is the time step. The mesh ratio r controls FTCS
    stability and neighbor influence.
    """
    alpha = float(question["alpha"])
    x = np.linspace(
        float(question["x_start"]),
        float(question["x_end"]),
        int(question["number_of_space_points"]),
    )
    h = x[1] - x[0]

    if question.get("r") is not None:
        r = float(question["r"])
        k = r * h**2 / alpha
    else:
        k = float(question["time_step"])
        r = alpha * k / h**2

    number_of_time_steps = int(np.ceil(float(question["final_time"]) / k))
    time_values = np.arange(number_of_time_steps + 1) * k

    return x, time_values, h, k, r


# ---------------------------------------------------------------------------
# Initial condition
# ---------------------------------------------------------------------------

def initial_condition(question, x):
    """
    Create the starting temperature profile U_i^0 from the question config.

    Supported types:
      "hot_center"   — flat hot zone defined by u_left, u_right, u_value
      "sin_pi"       — U_i^0 = sin(pi * x_i)
      "custom_list"  — explicit list of N+1 values
      "function"     — NumPy expression string in initial_condition_expr
    """
    initial_type = question["initial_condition_type"]

    if initial_type == "hot_center":
        values = np.zeros_like(x)
        # u_left / u_right / u_value define U_i^0 = u_value for x_L <= x_i <= x_R
        # Also accept old key names for backward compatibility.
        left = float(question.get("u_left", question.get("hot_center_left", 0.4)))
        right = float(question.get("u_right", question.get("hot_center_right", 0.6)))
        hot_temperature = float(question.get("u_value", question.get("hot_center_temperature", 100.0)))
        values[(x >= left) & (x <= right)] = hot_temperature
        return values

    if initial_type == "sin_pi":
        # U_i^0 = sin(pi * x_i)
        return np.sin(np.pi * x)

    if initial_type == "custom_list":
        values = np.array(question["custom_initial_values"], dtype=float)
        if len(values) != len(x):
            raise ValueError(
                "custom_initial_values length must match number_of_space_points."
            )
        return values

    if initial_type == "function":
        # Evaluate a NumPy expression.  Available names: x, np, pi, sin, cos, exp.
        expr = question["initial_condition_expr"]
        env = {
            "np": np, "x": x,
            "pi": np.pi, "sin": np.sin, "cos": np.cos, "exp": np.exp,
        }
        return np.asarray(eval(expr, env), dtype=float)  # noqa: S307

    raise ValueError(
        'initial_condition_type must be "hot_center", "sin_pi", "custom_list", or "function".'
    )


# ---------------------------------------------------------------------------
# FTCS solver
# ---------------------------------------------------------------------------

def solve_with_ftcs(question, x, time_values, r):
    """
    Solve the heat equation with FTCS.

    Boundary values are fixed at every time level. Interior values are
    updated from the previous time level using:

        U_i^{j+1} = r U_{i-1}^j + (1-2r) U_i^j + r U_{i+1}^j
    """
    solution = np.zeros((len(time_values), len(x)))
    solution[0] = initial_condition(question, x)

    solution[:, 0] = float(question["left_boundary"])
    solution[:, -1] = float(question["right_boundary"])

    for n in range(len(time_values) - 1):
        solution[n + 1, 1:-1] = (
            r * solution[n, :-2]
            + (1 - 2 * r) * solution[n, 1:-1]
            + r * solution[n, 2:]
        )
        # Reapply boundary conditions to make the physical assumptions explicit.
        solution[n + 1, 0] = float(question["left_boundary"])
        solution[n + 1, -1] = float(question["right_boundary"])

    return solution


# ---------------------------------------------------------------------------
# Exact solution
# ---------------------------------------------------------------------------

def exact_solution(question, x, time_values):
    """
    Return the exact solution array (shape: time x space) when one is available.

    Supported exact_solution_type values:
      "sin_pi_zero_boundary"
          U_exact(x,t) = e^{-alpha pi^2 t} sin(pi x)
          Valid only when alpha=1, domain [0,1], zero boundaries, IC=sin(pi x).

      "function"
          Evaluate exact_solution_expr for each time level.
          Available names in the expression: x (array), t (scalar), alpha (scalar),
          np, pi, sin, cos, exp.
    """
    exact_type = question.get("exact_solution_type")
    if exact_type is None:
        return None

    alpha = float(question["alpha"])

    if exact_type == "sin_pi_zero_boundary":
        result = np.zeros((len(time_values), len(x)))
        for n, t in enumerate(time_values):
            result[n] = np.exp(-alpha * np.pi**2 * t) * np.sin(np.pi * x)
        return result

    if exact_type == "function":
        expr = question.get("exact_solution_expr")
        if expr is None:
            raise ValueError(
                'exact_solution_type "function" requires "exact_solution_expr".'
            )
        env_base = {
            "np": np, "alpha": alpha,
            "pi": np.pi, "sin": np.sin, "cos": np.cos, "exp": np.exp,
        }
        result = np.zeros((len(time_values), len(x)))
        for n, t in enumerate(time_values):
            env = {**env_base, "x": x, "t": t}
            result[n] = np.asarray(eval(expr, env), dtype=float)  # noqa: S307
        return result

    raise ValueError(
        f'Unknown exact_solution_type: "{exact_type}". '
        'Use "sin_pi_zero_boundary" or "function".'
    )


# ---------------------------------------------------------------------------
# Error analysis
# ---------------------------------------------------------------------------

def compute_error_analysis(solution, exact):
    """
    Return per-step and global error metrics when an exact solution is available.

    Returns None if exact is None.
    """
    if exact is None:
        return None

    abs_error = np.abs(solution - exact)
    return {
        "max_error_per_step": np.max(abs_error, axis=1),
        "l2_error_per_step": np.sqrt(np.mean(abs_error**2, axis=1)),
        "global_max_error": float(np.max(abs_error)),
        "global_l2_error": float(np.sqrt(np.mean(abs_error**2))),
    }


# ---------------------------------------------------------------------------
# File output helpers
# ---------------------------------------------------------------------------

def save_mesh_grid(x, time_values):
    """Save every space-time grid point to a CSV file."""
    output_path = OUTPUT_DIR / "mesh_grid_points.csv"

    with output_path.open("w", encoding="utf-8") as file:
        file.write("time_index,space_index,t,x\n")
        for j, time in enumerate(time_values):
            for i, position in enumerate(x):
                file.write(f"{j},{i},{time:.10f},{position:.10f}\n")

    return output_path


def save_solution_table(x, time_values, solution):
    """Save the numerical solution table in TXT and CSV formats."""
    txt_path = OUTPUT_DIR / "ftcs_solution_table.txt"
    csv_path = OUTPUT_DIR / "ftcs_solution_table.csv"

    with txt_path.open("w", encoding="utf-8") as file:
        header = "time_index | time | " + " | ".join(f"U_{i}" for i in range(len(x)))
        file.write(header + "\n")
        file.write("-" * len(header) + "\n")

        for j, time in enumerate(time_values):
            row = f"{j:10d} | {time:8.5f} | "
            row += " | ".join(f"{value:10.5f}" for value in solution[j])
            file.write(row + "\n")

    with csv_path.open("w", encoding="utf-8") as file:
        file.write("time_index,time," + ",".join(f"U_{i}" for i in range(len(x))) + "\n")
        for j, time in enumerate(time_values):
            values = ",".join(f"{value:.10f}" for value in solution[j])
            file.write(f"{j},{time:.10f},{values}\n")

    return txt_path, csv_path


def save_simulation_data(x, time_values, solution):
    """Save long-form simulation data for spreadsheet analysis."""
    output_path = OUTPUT_DIR / "simulation_data_long.csv"

    with output_path.open("w", encoding="utf-8") as file:
        file.write("time_index,space_index,t,x,temperature\n")
        for j, time in enumerate(time_values):
            for i, position in enumerate(x):
                file.write(
                    f"{j},{i},{time:.10f},{position:.10f},{solution[j, i]:.10f}\n"
                )

    return output_path


# ---------------------------------------------------------------------------
# Static matplotlib plots
# ---------------------------------------------------------------------------

def plot_solution_profiles(x, time_values, solution, exact=None):
    """Save a line plot showing how the heat profile changes over time."""
    output_path = OUTPUT_DIR / "solution_profiles.png"

    selected_indices = np.linspace(0, len(time_values) - 1, min(6, len(time_values)))
    selected_indices = sorted(set(int(index) for index in selected_indices))
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, len(selected_indices)))

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#1f2937")
    ax.set_facecolor("#111827")

    for index, color in zip(selected_indices, colors):
        ax.plot(x, solution[index], color=color, linewidth=2,
                label=f"j={index}, t={time_values[index]:.4f}")
        if exact is not None:
            ax.plot(x, exact[index], color=color, linewidth=1.5,
                    linestyle="--", alpha=0.7)

    if exact is not None:
        ax.plot([], [], color="white", linewidth=2, label="FTCS (solid)")
        ax.plot([], [], color="white", linewidth=1.5, linestyle="--", label="Exact (dashed)")

    ax.set_title("FTCS Solution Profiles", color="white")
    ax.set_xlabel("Position x", color="white")
    ax.set_ylabel("Temperature", color="white")
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#374151")
    ax.grid(True, alpha=0.3, color="#4b5563")
    legend = ax.legend(facecolor="#1f2937", edgecolor="#374151", labelcolor="white")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)
    return output_path


def plot_heatmap(x, time_values, solution):
    """Save a heatmap of temperature over the whole space-time mesh."""
    output_path = OUTPUT_DIR / "solution_heatmap.png"

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#1f2937")
    ax.set_facecolor("#111827")

    heatmap = ax.imshow(
        solution,
        aspect="auto",
        origin="lower",
        extent=[x.min(), x.max(), time_values.min(), time_values.max()],
        cmap="inferno",
    )

    ax.set_title("FTCS Space-Time Heatmap", color="white")
    ax.set_xlabel("Position x", color="white")
    ax.set_ylabel("Time t", color="white")
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#374151")

    colorbar = fig.colorbar(heatmap, ax=ax)
    colorbar.set_label("Temperature", color="white")
    colorbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(colorbar.ax.yaxis.get_ticklabels(), color="white")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)
    return output_path


def plot_error_analysis(time_values, error_analysis):
    """Save a plot of max and L2 error vs time when an exact solution is available."""
    if error_analysis is None:
        return None

    output_path = OUTPUT_DIR / "error_analysis.png"

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#1f2937")
    ax.set_facecolor("#111827")

    ax.semilogy(time_values, error_analysis["max_error_per_step"],
                color="#e63946", linewidth=2, label="Max absolute error")
    ax.semilogy(time_values, error_analysis["l2_error_per_step"],
                color="#457b9d", linewidth=2, linestyle="--", label="L2 (RMS) error")

    ax.set_title("Error Analysis: FTCS vs Exact Solution", color="white")
    ax.set_xlabel("Time t", color="white")
    ax.set_ylabel("Error (log scale)", color="white")
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#374151")
    ax.grid(True, alpha=0.3, color="#4b5563", which="both")
    ax.legend(facecolor="#1f2937", edgecolor="#374151", labelcolor="white")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def build_report(question, x, time_values, h, k, r, solution, exact, error_analysis, output_files):
    """Write a readable report that summarizes the question and results."""
    report_path = OUTPUT_DIR / "question_report.md"
    stability_message = (
        "Stable for FTCS heat equation (r ≤ 0.5)"
        if r <= 0.5
        else "Unstable for FTCS heat equation (r > 0.5)"
    )

    if error_analysis is None:
        error_section = "No exact solution check requested."
    else:
        error_section = (
            f"- Global maximum absolute error: `{error_analysis['global_max_error']:.8e}`\n"
            f"- Global L2 (RMS) error:         `{error_analysis['global_l2_error']:.8e}`\n"
        )

    final_values = ", ".join(f"{value:.5f}" for value in solution[-1])

    report = f"""# FTCS Question Report

## Question

**Title:** {question["title"]}

**Reference:** {question.get("reference", "N/A")}

**PDE:** `{question.get("pde", "u_t = alpha u_xx")}`

## Parameters

| Symbol | Value |
|--------|-------|
| α (alpha) | `{question["alpha"]}` |
| x₀ (x_start) | `{question["x_start"]}` |
| x_N (x_end) | `{question["x_end"]}` |
| T (final_time) | `{question["final_time"]}` |
| N+1 (space points) | `{question["number_of_space_points"]}` |
| h (space step) | `{h:.10f}` |
| k (time step) | `{k:.10f}` |
| r = αk/h² | `{r:.10f}` |
| U₀ʲ (left boundary) | `{question["left_boundary"]}` |
| U_Nʲ (right boundary) | `{question["right_boundary"]}` |
| Initial condition type | `{question["initial_condition_type"]}` |

## Stability Check

FTCS for the 1D heat equation requires:

```
r ≤ 0.5
```

**Result:** {stability_message}

## Error Analysis

{error_section}

For ordinary diffusion with stable r, the heat profile should become smoother
over time. Boundary values should remain fixed at every time level.

## Final Time-Level Values (U_i^J)

```
{final_values}
```

## Generated Output Files

"""

    for label, path in output_files.items():
        report += f"- {label}: `{path.name}`\n"

    report += """
## How To Read This Result

- `mesh_grid_points.csv` — every (x_i, t_j) point in the mesh.
- `ftcs_solution_table.txt` — the main numerical solution table U_i^j.
- `simulation_data_long.csv` — long-form data for spreadsheets or extra plotting.
- `solution_profiles.png` — selected time levels overlaid.
- `solution_heatmap.png` — full space-time simulation as a colour map.
- `error_analysis.png` — max and L2 error vs time (when exact solution is provided).
"""

    report_path.write_text(report, encoding="utf-8")
    return report_path


# ---------------------------------------------------------------------------
# Supporting engines
# ---------------------------------------------------------------------------

def run_supporting_engines(question):
    """
    Run the supporting teaching scripts after the custom question is solved.

    These scripts generate the mesh picture, standard heat simulations,
    animation, 3D surface, and stability comparison. The custom question result
    remains in outputs/question_results/.
    """
    if not question.get("run_supporting_engines", True):
        return {}

    engine_files = [
        ("mesh dependency engine", "01_grid_and_mesh_visualization.py"),
        ("manual FTCS table engine", "02_manual_ftcs_table.py"),
        ("standard heat simulation engine", "03_ftcs_heat_equation_simulation.py"),
        ("3D surface engine", "04_ftcs_3d_surface_plot.py"),
        ("stability experiment engine", "06_stability_experiment.py"),
    ]

    if question.get("generate_animation", True):
        engine_files.insert(3, ("animation engine", "05_ftcs_animation.py"))

    environment = os.environ.copy()
    environment["MPLBACKEND"] = "Agg"

    results = {}
    for label, filename in engine_files:
        completed = subprocess.run(
            [sys.executable, str(BASE_DIR / filename)],
            cwd=BASE_DIR,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        results[label] = {
            "file": filename,
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }

    return results


# ---------------------------------------------------------------------------
# Static HTML dashboard
# ---------------------------------------------------------------------------

def build_dashboard(output_files, support_results):
    """Create a simple HTML dashboard for viewing generated results together."""
    dashboard_path = OUTPUT_DIR / "output_dashboard.html"

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>FTCS Question Output Dashboard</title>
  <style>
    body {{
      font-family: Arial, sans-serif;
      margin: 24px;
      line-height: 1.5;
      color: #222;
      background: #f7f7f4;
    }}
    h1, h2 {{ color: #1d3557; }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 18px;
    }}
    .card {{
      background: white;
      border: 1px solid #ddd;
      border-radius: 8px;
      padding: 16px;
    }}
    img {{
      max-width: 100%;
      border: 1px solid #ddd;
      background: white;
    }}
    table {{
      border-collapse: collapse;
      width: 100%;
      background: white;
    }}
    th, td {{
      border: 1px solid #ddd;
      padding: 8px;
      vertical-align: top;
    }}
    pre {{
      white-space: pre-wrap;
      margin: 0;
      font-size: 12px;
    }}
    a {{ color: #1d4ed8; }}
  </style>
</head>
<body>
  <h1>FTCS Question Output Dashboard</h1>
  <p>This dashboard is generated by <code>00_run_question_lab.py</code>.</p>

  <h2>Main Question Results</h2>
  <div class="grid">
    <div class="card">
      <h3>Report and Tables</h3>
      <p><a href="question_report.md">Open question_report.md</a></p>
      <p><a href="ftcs_solution_table.txt">Open ftcs_solution_table.txt</a></p>
      <p><a href="ftcs_solution_table.csv">Open ftcs_solution_table.csv</a></p>
      <p><a href="mesh_grid_points.csv">Open mesh_grid_points.csv</a></p>
      <p><a href="simulation_data_long.csv">Open simulation_data_long.csv</a></p>
    </div>
    <div class="card">
      <h3>Solution Profiles</h3>
      <img src="solution_profiles.png" alt="FTCS solution profiles">
    </div>
    <div class="card">
      <h3>Solution Heatmap</h3>
      <img src="solution_heatmap.png" alt="FTCS solution heatmap">
    </div>
    <div class="card">
      <h3>Error Analysis</h3>
      <img src="error_analysis.png" alt="Error analysis plot">
    </div>
  </div>

  <h2>Supporting Engine Outputs</h2>
  <div class="grid">
    <div class="card">
      <h3>Mesh Dependency</h3>
      <img src="../mesh_images/ftcs_dependency_structure.png" alt="FTCS dependency mesh">
    </div>
    <div class="card">
      <h3>Standard Heat Profiles</h3>
      <img src="../heatmaps/ftcs_heat_diffusion_profiles.png" alt="Standard FTCS profiles">
    </div>
    <div class="card">
      <h3>3D Surface</h3>
      <img src="../graphs/ftcs_space_time_surface.png" alt="FTCS 3D surface">
    </div>
    <div class="card">
      <h3>Manual Table Engine</h3>
      <p><a href="../graphs/manual_ftcs_table.txt">Open manual_ftcs_table.txt</a></p>
    </div>
    <div class="card">
      <h3>Stability Comparison</h3>
      <img src="../graphs/ftcs_stability_comparison.png" alt="FTCS stability comparison">
    </div>
    <div class="card">
      <h3>Animation</h3>
      <p><a href="../animations/ftcs_heat_diffusion_animation.gif">Open animation GIF</a></p>
    </div>
  </div>

</body>
</html>
"""

    dashboard_path.write_text(html, encoding="utf-8")
    return dashboard_path


# ---------------------------------------------------------------------------
# Interactive Plotly dashboard
# ---------------------------------------------------------------------------

def _build_rod_animation_figure(x, time_values, solution):
    """
    Build a 3D animated cylinder rod showing heat diffusion along its length.

    The rod is a proper 3D cylinder surface (like MATLAB surf).  Temperature is
    encoded as surface colour using a physical blackbody scale: dark/cold → red
    → orange → white/hot.  Each animation frame advances one time step.
    """
    # Cylinder geometry: rod runs along x-axis; y and z form the circular cross-section.
    n_theta = 32
    theta = np.linspace(0, 2 * np.pi, n_theta)
    R = 0.12   # rod radius (visual, proportional to unit length)

    # X_cyl[i_theta, i_x] = x[i_x]  (replicated across all theta)
    X_cyl = np.tile(x, (n_theta, 1))
    Y_cyl = R * np.outer(np.cos(theta), np.ones(len(x)))
    Z_cyl = R * np.outer(np.sin(theta), np.ones(len(x)))

    z_min = float(np.min(solution))
    z_max = float(np.max(solution))
    if z_max == z_min:
        z_max = z_min + 1.0

    rod_colorscale = [
        [0.00, "rgb(10,10,30)"],
        [0.15, "rgb(80,0,0)"],
        [0.35, "rgb(180,30,0)"],
        [0.55, "rgb(220,100,0)"],
        [0.75, "rgb(255,180,0)"],
        [0.90, "rgb(255,240,120)"],
        [1.00, "rgb(255,255,240)"],
    ]

    def make_surface(n):
        # Tile temperature across all theta angles so the colour wraps the cylinder.
        C = np.tile(solution[n], (n_theta, 1))
        return go.Surface(
            x=X_cyl.tolist(),
            y=Y_cyl.tolist(),
            z=Z_cyl.tolist(),
            surfacecolor=C.tolist(),
            colorscale=rod_colorscale,
            cmin=z_min,
            cmax=z_max,
            colorbar=dict(title="Temperature", thickness=18, len=0.75),
            showscale=True,
            lighting=dict(ambient=0.45, diffuse=0.85, specular=0.4, roughness=0.4),
            lightposition=dict(x=200, y=300, z=400),
            hovertemplate=(
                "x = %{x:.4f}<br>"
                "Temperature = %{surfacecolor:.4f}<extra></extra>"
            ),
        )

    max_frames = min(len(time_values), 60)
    frame_indices = np.linspace(0, len(time_values) - 1, max_frames, dtype=int)

    frames = []
    slider_steps = []
    for n in frame_indices:
        name = str(n)
        frames.append(go.Frame(
            data=[make_surface(n)],
            name=name,
            layout=go.Layout(
                title=dict(text=f"3D Iron Rod — j={n}, t={time_values[n]:.5f}")
            ),
        ))
        slider_steps.append(dict(
            args=[[name], dict(frame=dict(duration=0, redraw=True), mode="immediate")],
            label=f"t={time_values[n]:.3f}",
            method="animate",
        ))

    fig = go.Figure(
        data=[make_surface(0)],
        frames=frames,
        layout=go.Layout(
            title="3D Iron Rod Heat Diffusion — j=0",
            scene=dict(
                xaxis=dict(title="Position x", showgrid=True, gridcolor="#444"),
                yaxis=dict(title="", showticklabels=False, showgrid=False, zeroline=False),
                zaxis=dict(title="", showticklabels=False, showgrid=False, zeroline=False),
                camera=dict(eye=dict(x=1.6, y=1.8, z=0.7)),
                aspectmode="manual",
                aspectratio=dict(x=3.5, y=1, z=1),
                bgcolor="rgb(15,15,25)",
            ),
            height=460,
            margin=dict(l=0, r=0, t=60, b=90),
            updatemenus=[dict(
                type="buttons",
                showactive=False,
                y=1.12,
                x=0.5,
                xanchor="center",
                buttons=[
                    dict(label="▶ Play", method="animate",
                         args=[None, dict(frame=dict(duration=800, redraw=True), fromcurrent=True)]),
                    dict(label="⏸ Pause", method="animate",
                         args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")]),
                ],
            )],
            sliders=[dict(
                active=0,
                steps=slider_steps,
                y=0,
                len=1.0,
                pad=dict(t=40),
            )],
        ),
    )
    return fig


def _build_error_figure(time_values, error_analysis):
    """Build a Plotly error-vs-time figure.  Returns None if no exact solution."""
    if error_analysis is None:
        return None

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=time_values.tolist(),
        y=error_analysis["max_error_per_step"].tolist(),
        mode="lines+markers",
        name="Max absolute error",
        line=dict(color="#e63946", width=2),
        marker=dict(size=4),
        hovertemplate="t=%{x:.5f}<br>max error=%{y:.3e}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=time_values.tolist(),
        y=error_analysis["l2_error_per_step"].tolist(),
        mode="lines",
        name="L2 (RMS) error",
        line=dict(color="#457b9d", width=2, dash="dash"),
        hovertemplate="t=%{x:.5f}<br>L2 error=%{y:.3e}<extra></extra>",
    ))
    fig.update_layout(
        title="Error Analysis: FTCS vs Exact Solution",
        xaxis_title="Time t",
        yaxis_title="Error",
        yaxis_type="log",
        hovermode="x unified",
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0)"),
    )
    return fig


def build_ftcs_matrix(number_of_points, r, left_boundary, right_boundary):
    """
    Build the matrix form of the FTCS method for the interior unknowns.

    For zero Dirichlet boundaries, the interior vector satisfies

        U_int^(n+1) = A U_int^n

    If the boundary values are nonzero but fixed, the mathematically complete
    form is

        U_int^(n+1) = A U_int^n + b

    where b contains the left and right boundary contributions.
    """
    interior_count = number_of_points - 2
    matrix = np.zeros((interior_count, interior_count), dtype=float)

    for row in range(interior_count):
        matrix[row, row] = 1 - 2 * r
        if row > 0:
            matrix[row, row - 1] = r
        if row < interior_count - 1:
            matrix[row, row + 1] = r

    boundary_vector = np.zeros(interior_count, dtype=float)
    if interior_count > 0:
        boundary_vector[0] += r * left_boundary
        boundary_vector[-1] += r * right_boundary

    return matrix, boundary_vector


def latex_number(value):
    """Format a number so matrix entries stay readable in MathJax."""
    value = float(value)
    if abs(value) < 1e-12:
        return "0"
    return f"{value:.6g}"


def latex_column_vector(values, max_entries=10):
    """
    Convert a 1D array into a LaTeX column vector.

    Long vectors are shown with leading and trailing values so the page remains
    readable for larger question-bank examples. The complete numerical table is
    still printed later in the dashboard.
    """
    values = np.asarray(values, dtype=float).ravel()
    if len(values) == 0:
        return r"\begin{bmatrix}\end{bmatrix}"

    if len(values) > max_entries:
        shown = list(values[:5]) + [None] + list(values[-3:])
    else:
        shown = list(values)

    rows = [r"\vdots" if value is None else latex_number(value) for value in shown]
    return r"\begin{bmatrix}" + r"\\".join(rows) + r"\end{bmatrix}"


def latex_matrix(matrix, r, max_size=8):
    """
    Convert the FTCS matrix to LaTeX.

    Small matrices are shown numerically. Large matrices are shown in standard
    tridiagonal textbook form because a 25-by-25 matrix would hide the idea.
    """
    matrix = np.asarray(matrix, dtype=float)
    rows_count, cols_count = matrix.shape
    if rows_count == 0:
        return r"\begin{bmatrix}\end{bmatrix}"

    if rows_count <= max_size and cols_count <= max_size:
        rows = []
        for row in matrix:
            rows.append(" & ".join(latex_number(value) for value in row))
        return r"\begin{bmatrix}" + r"\\".join(rows) + r"\end{bmatrix}"

    center = latex_number(1 - 2 * r)
    side = latex_number(r)
    return (
        r"\left[\begin{array}{cccccc}"
        f"{center} & {side} & 0 & \\cdots & 0 & 0\\\\"
        f"{side} & {center} & {side} & \\ddots &  & 0\\\\"
        f"0 & {side} & {center} & \\ddots & \\ddots & \\vdots\\\\"
        r"\vdots & \ddots & \ddots & \ddots & "
        f"{side} & 0\\\\"
        f"0 &  & \\ddots & {side} & {center} & {side}\\\\"
        f"0 & 0 & \\cdots & 0 & {side} & {center}"
        r"\end{array}\right]"
        f"_{{{rows_count}\\times {cols_count}}}"
    )


def build_interactive_dashboard(question, x, time_values, h, k, r, solution, exact, error_analysis, support_results):
    """
    Create an interactive Plotly dashboard.

    Features:
    - Time evolution player with slider and Play/Pause
    - Solution profiles, heatmap, rotatable 3D surface, mesh grid
    - Iron rod animation (blackbody glow)
    - Error analysis graph (when exact solution is provided)
    - Dark mode toggle (also re-themes all Plotly charts)
    - Click-to-inspect point details
    """
    dashboard_path = OUTPUT_DIR / "interactive_output_dashboard.html"

    selected_indices = np.linspace(0, len(time_values) - 1, min(6, len(time_values)))
    selected_indices = sorted(set(int(index) for index in selected_indices))

    # --- Profile figure ---
    profile_fig = go.Figure()
    for index in selected_indices:
        profile_fig.add_trace(go.Scatter(
            x=x.tolist(), y=solution[index].tolist(),
            mode="lines+markers",
            name=f"j={index}, t={time_values[index]:.5f}",
            customdata=np.column_stack([
                np.full(len(x), index),
                np.full(len(x), time_values[index]),
                np.arange(len(x)),
            ]).tolist(),
            hovertemplate=(
                "i=%{customdata[2]}<br>j=%{customdata[0]}<br>"
                "x=%{x:.5f}<br>t=%{customdata[1]:.5f}<br>"
                "U=%{y:.5f}<extra></extra>"
            ),
        ))
    if exact is not None:
        for index in selected_indices:
            profile_fig.add_trace(go.Scatter(
                x=x.tolist(), y=exact[index].tolist(),
                mode="lines",
                name=f"Exact j={index}",
                line=dict(dash="dash"),
                showlegend=True,
                hovertemplate="x=%{x:.5f}<br>U_exact=%{y:.5f}<extra></extra>",
            ))
    profile_fig.update_layout(
        title="FTCS Solution Profiles",
        xaxis_title="Position x", yaxis_title="Temperature", hovermode="closest",
    )

    # --- Heatmap figure ---
    heatmap_fig = go.Figure(data=go.Heatmap(
        x=x.tolist(), y=time_values.tolist(), z=solution.tolist(),
        colorscale="Inferno", colorbar=dict(title="Temperature"),
        hovertemplate="x=%{x:.5f}<br>t=%{y:.5f}<br>U=%{z:.5f}<extra></extra>",
    ))
    heatmap_fig.update_layout(
        title="Interactive Space-Time Heatmap",
        xaxis_title="Position x", yaxis_title="Time t",
    )

    # --- 3D surface figure ---
    surface_fig = go.Figure(data=[
        go.Surface(
            x=x.tolist(), y=time_values.tolist(), z=solution.tolist(),
            colorscale="Plasma", colorbar=dict(title="Temperature"),
            opacity=0.78,
            hovertemplate="x=%{x:.5f}<br>t=%{y:.5f}<br>U=%{z:.5f}<extra></extra>",
            name="Full space-time surface",
        ),
        go.Scatter3d(
            x=x.tolist(),
            y=[float(time_values[0])] * len(x),
            z=solution[0].tolist(),
            mode="lines+markers",
            line=dict(color="#e63946", width=7),
            marker=dict(size=4, color="#e63946"),
            name="Current time slice",
        ),
    ])
    surface_fig.update_layout(
        title="Rotatable 3D Space-Time Surface",
        scene=dict(xaxis_title="Space x", yaxis_title="Time t", zaxis_title="Temperature"),
        margin=dict(l=0, r=0, t=50, b=0),
    )

    # --- Mesh figure ---
    max_time_rows = min(8, len(time_values))
    max_space_points = min(12, len(x))
    sampled_j = np.linspace(0, len(time_values) - 1, max_time_rows, dtype=int)
    sampled_i = np.linspace(0, len(x) - 1, max_space_points, dtype=int)
    mesh_x, mesh_t, mesh_i_list, mesh_j_list, mesh_labels = [], [], [], [], []
    for j in sampled_j:
        for i in sampled_i:
            mesh_x.append(float(x[i]))
            mesh_t.append(float(time_values[j]))
            mesh_i_list.append(int(i))
            mesh_j_list.append(int(j))
            mesh_labels.append(f"U_{i}^{j}")
    mesh_fig = go.Figure(data=go.Scatter(
        x=mesh_x, y=mesh_t,
        mode="markers+text",
        text=mesh_labels, textposition="top center",
        marker=dict(size=10, color=mesh_j_list, colorscale="Viridis",
                    colorbar=dict(title="Time level j")),
        customdata=list(zip(mesh_i_list, mesh_j_list)),
        hovertemplate=(
            "node=%{text}<br>i=%{customdata[0]}<br>j=%{customdata[1]}<br>"
            "x=%{x:.5f}<br>t=%{y:.5f}<extra></extra>"
        ),
    ))
    mesh_fig.update_layout(
        title="Interactive Mesh Grid Points",
        xaxis_title="Space x", yaxis_title="Time t",
    )

    # --- Iron rod animation ---
    rod_fig = _build_rod_animation_figure(x, time_values, solution)

    # --- Error figure ---
    error_fig = _build_error_figure(time_values, error_analysis)

    # --- Matrix generalisation: U_int^(n+1) = A U_int^n (+ b) ---
    ftcs_matrix, boundary_vector = build_ftcs_matrix(
        len(x), r, float(question["left_boundary"]), float(question["right_boundary"])
    )
    boundary_is_zero = bool(np.allclose(boundary_vector, 0.0))
    matrix_formula = (
        r"U_{\mathrm{int}}^{\,n+1}=A\,U_{\mathrm{int}}^{\,n}"
        if boundary_is_zero
        else r"U_{\mathrm{int}}^{\,n+1}=A\,U_{\mathrm{int}}^{\,n}+b"
    )
    matrix_latex = latex_matrix(ftcs_matrix, r)
    boundary_latex = latex_column_vector(boundary_vector)
    center_coefficient = 1 - 2 * r
    center_sign = "+" if center_coefficient >= 0 else "-"
    numeric_scalar_scheme = (
        rf"U_i^{{n+1}}={latex_number(r)}U_{{i-1}}^n"
        rf"{center_sign}{latex_number(abs(center_coefficient))}U_i^n"
        rf"+{latex_number(r)}U_{{i+1}}^n"
    )
    general_matrix_latex = (
        r"\begin{bmatrix}"
        r"1-2r & r & 0 & \cdots & 0\\"
        r"r & 1-2r & r & \ddots & \vdots\\"
        r"0 & r & 1-2r & \ddots & 0\\"
        r"\vdots & \ddots & \ddots & \ddots & r\\"
        r"0 & \cdots & 0 & r & 1-2r"
        r"\end{bmatrix}_{(N-1)\times(N-1)}"
    )
    boundary_matrix_note = (
        " Because the boundary values are zero, there is no extra boundary vector."
        if boundary_is_zero
        else " Because the boundary values are nonzero, the boundary vector \\(b\\) adds their fixed contribution at the two ends."
    )

    matrix_examples_html = ""
    transitions_to_show = min(2, max(0, len(time_values) - 1))
    for n in range(transitions_to_show):
        current_vector = solution[n, 1:-1]
        next_vector = solution[n + 1, 1:-1]
        matrix_product = ftcs_matrix @ current_vector + boundary_vector
        boundary_term_latex = "" if boundary_is_zero else f"+ {boundary_latex}"
        matrix_examples_html += f"""
          <div class="matrix-step">
            <h3>Time level \\(n={n}\\) to \\(n+1={n+1}\\)</h3>
            <p>
              Start with the interior values only. The boundary values are fixed, so
              they are not placed inside the unknown vector.
            </p>
            <div class="formula">
              \\[
                U_{{\\mathrm{{int}}}}^{{{n}}}
                =
                {latex_column_vector(current_vector)}
              \\]
              \\[
                A U_{{\\mathrm{{int}}}}^{{{n}}} {boundary_term_latex}
                =
                {latex_column_vector(matrix_product)}
              \\]
              \\[
                U_{{\\mathrm{{int}}}}^{{{n+1}}}
                =
                {latex_column_vector(next_vector)}
              \\]
            </div>
            <p>
              Each row of \\(A\\) gives one new interior value. The first row gives
              \\(U_1^{{{n+1}}}\\), the second row gives \\(U_2^{{{n+1}}}\\), and the
              same pattern continues across the rod.
            </p>
          </div>
        """

    # --- Complete table rows ---
    complete_header = "<tr><th>time level</th><th>physical time</th>"
    for i in range(len(x)):
        complete_header += f"<th>\\(U_{{{i}}}^{{j}}\\)<br><small>x={x[i]:.5f}</small></th>"
    complete_header += "</tr>"
    complete_table_rows = ""
    for j in range(len(time_values)):
        complete_table_rows += f"<tr><td>\\(j={j}\\)</td><td>\\(t={time_values[j]:.6g}\\)</td>"
        for i in range(len(x)):
            complete_table_rows += f"<td>{solution[j, i]:.6f}</td>"
        complete_table_rows += "</tr>\n"

    # --- Descriptive messages ---
    initial_peak = float(np.max(solution[0]))
    final_peak = float(np.max(solution[-1]))
    physical_message = (
        "The maximum temperature decreased — this is expected from heat diffusion."
        if final_peak <= initial_peak
        else "The maximum temperature increased. For a stable heat problem this is a warning sign."
    )
    stability_message = (
        "Stable — r ≤ 0.5, so the FTCS method should smooth the heat profile."
        if r <= 0.5
        else "Unstable — r > 0.5, so oscillations or exploding values may appear."
    )

    if error_analysis is None:
        exact_message = "No exact solution check requested."
    else:
        exact_message = (
            f"Max absolute error: {error_analysis['global_max_error']:.6e} &nbsp;|&nbsp; "
            f"L2 error: {error_analysis['global_l2_error']:.6e}"
        )

    # --- Educational metadata for new dashboard panels ---
    _ic_map = {
        "hot_center": (
            f"\\(U_i^0 = {question.get('u_value','?')}\\)"
            f" for \\({question.get('u_left','?')} \\le x_i \\le {question.get('u_right','?')}\\)"
        ),
        "sin_pi":     "\\(U_i^0 = \\sin(\\pi x_i)\\)",
        "custom_list": "custom list of \\(N+1\\) values",
        "function":   (
            question.get("initial_condition_display")
            or f"\\(U_i^0 =\\) <code>{question.get('initial_condition_expr','custom function')}</code>"
        ),
    }
    ic_desc_html = _ic_map.get(
        question.get("initial_condition_type", ""), question.get("initial_condition_type", "")
    )
    stability_color = "#22c55e" if r <= 0.5 else "#ef4444"
    stability_label = "STABLE" if r <= 0.5 else "UNSTABLE"
    num_interior_pts = int(len(x) - 2)
    num_time_steps_total = int(len(time_values) - 1)
    total_unknowns = num_interior_pts * num_time_steps_total
    _exact_map = {
        "sin_pi_zero_boundary": "\\(u(x,t)=e^{-\\alpha\\pi^2t}\\sin(\\pi x)\\)",
        "function": f"\\(u(x,t)=\\) <code>{question.get('exact_solution_expr','custom exact solution')}</code>",
    }
    exact_type_label = _exact_map.get(question.get("exact_solution_type"), "None")
    if r <= 0.5:
        stability_verdict_html = (
            f'<p style="color:#22c55e;font-weight:700">&#10003; r = {r:.4f} &le; 0.5 &mdash; '
            'FTCS is <strong>stable</strong>. The heat profile smooths out over time.</p>'
        )
    else:
        stability_verdict_html = (
            f'<p style="color:#ef4444;font-weight:700">&#10007; r = {r:.4f} &gt; 0.5 &mdash; '
            'FTCS is <strong>unstable</strong>. Oscillations may grow without bound.</p>'
        )
    _err_rows = ""
    if error_analysis:
        _err_rows = (
            f'<tr><td>Global max error \\(\\|e\\|_\\infty\\)</td>'
            f'<td><strong>{error_analysis["global_max_error"]:.6e}</strong></td></tr>'
            f'<tr><td>Global L2 error \\(\\|e\\|_2\\)</td>'
            f'<td><strong>{error_analysis["global_l2_error"]:.6e}</strong></td></tr>'
        )
    _reference = question.get("reference", "")
    _ref_html = (
        f'<p><strong>Reference:</strong> <em>{_reference}</em></p>'
        if _reference else ""
    )
    # question_text / graph_notes — plain string values; LaTeX {braces} are safe
    # because values substituted via {variable} in f-strings are never re-parsed.
    _q_text_html = question.get("question_text", "")
    _gn_raw      = question.get("graph_notes", {})
    _g_profiles  = _gn_raw.get("profiles", "")
    _g_rod       = _gn_raw.get("rod",      "")
    _g_surface   = _gn_raw.get("surface",  "")
    # Pre-build the note <div>s so only a simple {variable} appears in the template.
    _g_rod_div      = f'<div class="graph-note">{_g_rod}</div>'      if _g_rod      else ""
    _g_profiles_div = f'<div class="graph-note">{_g_profiles}</div>' if _g_profiles else ""
    _g_surface_div  = f'<div class="graph-note">{_g_surface}</div>'  if _g_surface  else ""
    exact_profile_note = (
        " Dashed lines show the exact solution, so matching them closely means the numerical method is accurate."
        if exact is not None else ""
    )

    # --- Serialise figures ---
    figures = {
        "profilePlot": profile_fig.to_plotly_json(),
        "surfacePlot": surface_fig.to_plotly_json(),
        "meshPlot": mesh_fig.to_plotly_json(),
        "rodAnimation": rod_fig.to_plotly_json(),
    }
    has_error_panel = error_fig is not None
    if has_error_panel:
        figures["errorPlot"] = error_fig.to_plotly_json()

    figures_json = json.dumps(figures, cls=PlotlyJSONEncoder)
    x_json = json.dumps(x.tolist(), cls=PlotlyJSONEncoder)
    time_json = json.dumps(time_values.tolist(), cls=PlotlyJSONEncoder)
    solution_json = json.dumps(solution.tolist(), cls=PlotlyJSONEncoder)

    if has_error_panel:
        error_panel_html = f"""
      <div class="panel lesson">
        <div class="section-label">Section 12 &mdash; Accuracy Check</div>
        <h2>Error Analysis: Numerical Answer vs Exact Answer</h2>
        <p>
          Error analysis compares the numerical FTCS value with the exact value
          supplied by the problem. At each grid point we compute
          \\[
            e_i^j = |U_i^j - u(x_i,t_j)|.
          \\]
          Here \\(U_i^j\\) is the value produced by the code, while
          \\(u(x_i,t_j)\\) is the exact mathematical answer supplied by the question.
        </p>
        <div class="table-scroll">
          <table>
            <tr><th>Curve</th><th>Meaning</th><th>How to read it</th></tr>
            <tr>
              <td><strong>Max absolute error</strong></td>
              <td>The largest mistake anywhere on the rod at one time level.</td>
              <td>If this is small, even the worst grid point is accurate.</td>
            </tr>
            <tr>
              <td><strong>L2 / RMS error</strong></td>
              <td>The average-size mistake across the whole rod.</td>
              <td>If this is small, the whole temperature profile is accurate overall.</td>
            </tr>
          </table>
        </div>
        <p>
          The vertical axis uses a logarithmic scale. That makes tiny errors easier
          to see, but it also means equal vertical gaps represent multiplicative
          changes, not ordinary equal-size changes.
        </p>
        <p>
          Global max error:
          <strong>{error_analysis['global_max_error']:.6e}</strong> &nbsp;|&nbsp;
          Global L2/RMS error:
          <strong>{error_analysis['global_l2_error']:.6e}</strong>
        </p>
        <div id="errorPlot" class="plot"></div>
      </div>"""
    else:
        error_panel_html = """
      <div class="panel lesson">
        <div class="section-label">Section 12 &mdash; Accuracy Check</div>
        <h2>Error Analysis</h2>
        <p>
          This question does not provide an exact solution, so the dashboard cannot
          compute a true error curve. That is normal: many real PDE problems have no
          simple closed-form answer.
        </p>
        <p>
          When an exact solution is available, the lab compares each numerical value
          \\(U_i^j\\) with the exact value \\(u(x_i,t_j)\\), then plots how the error
          changes over time.
        </p>
      </div>"""

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Interactive FTCS Output Dashboard</title>
  <style>
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Arial, sans-serif;
      color: #1f2933;
      background: #f4f6f8;
      overflow-x: hidden;
    }}
    body.dark {{
      color: #e5e7eb;
      background: #111827;
    }}
    header {{
      padding: 18px 24px;
      background: #12355b;
      color: white;
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: center;
      flex-wrap: wrap;
    }}
    .header-left {{ flex: 1; }}
    .header-right {{ display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }}
    .author-line {{
      font-size: 0.78em;
      opacity: 0.70;
      font-style: normal;
      letter-spacing: 0.04em;
      margin-top: 3px;
      font-variant: small-caps;
    }}
    .theme-toggle {{
      border: 1px solid rgba(255,255,255,0.55);
      background: rgba(255,255,255,0.12);
      color: white;
      padding: 9px 13px;
      border-radius: 6px;
      cursor: pointer;
      font-weight: 700;
    }}
    main {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(300px, 360px);
      gap: 16px;
      padding: 16px;
      max-width: 100vw;
      align-items: start;
    }}
    .panel {{
      background: white;
      border: 1px solid #d9dee5;
      border-radius: 8px;
      padding: 14px;
      margin-bottom: 16px;
      min-width: 0;
      overflow: hidden;
    }}
    body.dark .panel {{
      background: #1f2937;
      border-color: #374151;
    }}
    .lesson {{
      border-left: 4px solid #2f6fdb;
      background: #f8fbff;
    }}
    body.dark .lesson {{
      background: #182235;
    }}
    .formula {{
      background: #eef2f6;
      padding: 10px;
      border-radius: 6px;
      overflow-x: auto;
      line-height: 1.9;
    }}
    body.dark .formula,
    body.dark code,
    body.dark pre {{
      background: #111827;
    }}
    .math {{
      font-family: "Cambria Math", Cambria, "Times New Roman", serif;
      font-size: 1.08em;
    }}
    .formula-stack {{
      display: grid;
      gap: 10px;
      min-width: 420px;
      line-height: 1.8;
    }}
    .computed-value {{
      min-width: 110px;
      text-align: center;
      font-size: 1.18em;
    }}
    .computed-result {{
      min-width: 120px;
      text-align: center;
      font-weight: bold;
    }}
    .control-row {{
      display: grid;
      grid-template-columns: auto auto 1fr auto;
      gap: 12px;
      align-items: center;
    }}
    .control-button {{
      border: 1px solid #b8c2cc;
      background: #ffffff;
      color: #12355b;
      padding: 9px 13px;
      border-radius: 6px;
      cursor: pointer;
      font-weight: 700;
      min-width: 72px;
    }}
    body.dark .control-button {{
      background: #111827;
      color: #e5e7eb;
      border-color: #4b5563;
    }}
    input[type="range"] {{ width: 100%; }}
    .plot {{ width: 100%; height: 520px; min-width: 0; }}
    .plot-rod {{ width: 100%; height: 500px; min-width: 0; }}
    .side {{
      position: sticky;
      top: 16px;
      align-self: start;
      max-height: calc(100vh - 32px);
      overflow-y: auto;
      padding-right: 4px;
      position: relative;
      min-width: 200px;
    }}
    .side::-webkit-scrollbar {{ width: 8px; }}
    .side::-webkit-scrollbar-thumb {{ background: #cbd5e1; border-radius: 999px; }}
    body.dark .side::-webkit-scrollbar-thumb {{ background: #475569; }}
    .side .panel {{ overflow: hidden; }}
    #sideResizeHandle {{
      position: absolute;
      left: 0;
      top: 0;
      width: 7px;
      height: 100%;
      cursor: col-resize;
      z-index: 20;
      border-left: 3px solid transparent;
      transition: border-color .15s;
      box-sizing: border-box;
    }}
    #sideResizeHandle:hover,
    #sideResizeHandle.dragging {{
      border-left-color: #2f6fdb;
    }}
    body.dark #sideResizeHandle:hover,
    body.dark #sideResizeHandle.dragging {{
      border-left-color: #60a5fa;
    }}
    code, pre {{
      background: #eef2f6;
      padding: 2px 4px;
      border-radius: 4px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
      table-layout: auto;
    }}
    td, th {{
      border: 1px solid #d9dee5;
      padding: 12px 14px;
      text-align: left;
      overflow-wrap: anywhere;
      vertical-align: middle;
    }}
    th {{
      background: #eef4fb;
      font-weight: 700;
    }}
    body.dark th {{ background: #273449; }}
    body.dark td, body.dark th {{ border-color: #374151; }}
    a {{ color: #0f5cc0; overflow-wrap: anywhere; }}
    pre {{ white-space: pre-wrap; overflow-wrap: anywhere; max-width: 100%; }}
    .table-scroll {{ width: 100%; overflow-x: auto; }}
    .pill {{ display:inline-block; padding:2px 9px; border-radius:99px; font-size:.78em; font-weight:700; color:white; }}
    .section-label {{ font-size:.68em; text-transform:uppercase; letter-spacing:.1em; color:#64748b; font-weight:700; border-bottom:1px solid #e5e7eb; padding-bottom:4px; margin-bottom:10px; }}
    body.dark .section-label {{ color:#9ca3af; border-color:#374151; }}
    .pde-badge {{ display:grid; grid-template-columns:1fr 1fr; gap:8px; margin:10px 0; }}
    .pde-badge-item {{ background:#f8fafc; border:1px solid #e2e8f0; border-radius:6px; padding:10px; text-align:center; }}
    body.dark .pde-badge-item {{ background:#1e293b; border-color:#334155; }}
    .pde-badge-item strong {{ display:block; font-size:.7em; text-transform:uppercase; letter-spacing:.05em; color:#64748b; margin-bottom:4px; }}
    .pde-badge-item span {{ font-size:1.1em; font-weight:700; }}
    .stencil-wrap {{ display:flex; flex-direction:column; align-items:center; gap:10px; padding:16px 0; }}
    .stencil-row {{ display:flex; gap:20px; justify-content:center; align-items:center; }}
    .stencil-node {{ width:100px; height:62px; border-radius:8px; display:flex; flex-direction:column; align-items:center; justify-content:center; gap:1px; border:2px solid transparent; }}
    .stencil-node.known {{ background:#1d4ed8; color:white; border-color:#93c5fd; }}
    .stencil-node.computed {{ background:#dc2626; color:white; border-color:#fca5a5; }}
    .stencil-node.boundary {{ background:#6b7280; color:white; }}
    .stencil-arrows {{ font-size:1.6em; color:#64748b; text-align:center; letter-spacing:10px; }}
    .step-box {{ background:#f0f9ff; border-left:3px solid #0ea5e9; padding:10px 14px; border-radius:0 6px 6px 0; margin:7px 0; font-size:.9em; }}
    body.dark .step-box {{ background:#0c1a2e; }}
    .step-label {{ font-weight:700; font-size:.7em; text-transform:uppercase; letter-spacing:.07em; color:#0369a1; margin-bottom:4px; }}
    body.dark .step-label {{ color:#38bdf8; }}
    .phys-note {{ font-style:italic; color:#0369a1; font-size:.87em; padding:6px 0 0; border-top:1px dashed #bae6fd; margin-top:8px; }}
    body.dark .phys-note {{ color:#7dd3fc; border-color:#1e3a5f; }}
    .question-box {{
      background:#f5f0ff; border:2px solid #7c3aed; border-radius:8px;
      padding:18px 22px; margin:10px 0; line-height:1.9; font-size:.97em;
    }}
    body.dark .question-box {{ background:#1a1530; border-color:#7c3aed; }}
    .question-box p:first-child {{ margin-top:0; }}
    .question-box p:last-child  {{ margin-bottom:0; }}
    .graph-note {{
      background:#fffbeb; border-left:3px solid #f59e0b;
      padding:9px 14px; border-radius:0 6px 6px 0;
      margin:10px 0; font-size:.9em; line-height:1.8; color:#44403c;
    }}
    body.dark .graph-note {{ background:#1c1a0e; border-color:#d97706; color:#d6d3d1; }}
    .lecture-note {{
      background:#f8fafc;
      border:1px solid #e2e8f0;
      border-left:4px solid #0ea5e9;
      border-radius:7px;
      padding:12px 14px;
      margin:12px 0;
      line-height:1.85;
    }}
    body.dark .lecture-note {{ background:#111827; border-color:#334155; }}
    .matrix-step {{
      border:1px solid #dbeafe;
      background:#ffffff;
      border-radius:8px;
      padding:12px 14px;
      margin:12px 0;
      overflow-x:auto;
    }}
    body.dark .matrix-step {{ background:#172033; border-color:#334155; }}
    .complete-table {{
      min-width: 900px;
      font-family: "Cambria Math", Cambria, "Times New Roman", serif;
      font-size: 12.5px;
    }}
    .complete-table td, .complete-table th {{
      text-align:center;
      white-space:nowrap;
    }}
    @media (max-width: 1000px) {{
      main {{ grid-template-columns: 1fr; }}
      .side {{ position: static; max-height: none; overflow: visible; padding-right: 0; }}
    }}
  </style>
  <script>
    window.MathJax = {{
      tex: {{
        inlineMath: [["\\\\(", "\\\\)"], ["$", "$"]],
        displayMath: [["\\\\[", "\\\\]"]]
      }},
      svg: {{fontCache: "global"}}
    }};
  </script>
  <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>
  <script>{get_plotlyjs()}</script>
</head>
<body>
  <header>
    <div class="header-left">
      <h1 style="margin:0 0 2px">Interactive FTCS Heat Equation Lab</h1>
      <div class="author-line">Developed by &nbsp;Adegboyega Samuel</div>
    </div>
    <div class="header-right">
      <label for="speedSelect" style="color:white;font-size:.85em;opacity:.82;font-weight:600;white-space:nowrap">Speed:</label>
      <select id="speedSelect" class="theme-toggle" style="font-weight:normal;cursor:pointer;padding:9px 10px" title="Playback speed for all animations">
        <option value="2500">Slow</option>
        <option value="1200" selected>Medium</option>
        <option value="500">Fast</option>
      </select>
      <button id="collapsePanel" class="theme-toggle" type="button" title="Collapse / expand side panel">&#8614; Panel</button>
      <button id="themeToggle" class="theme-toggle" type="button">Dark mode</button>
    </div>
  </header>
  <main>
    <section>
      <div class="panel lesson" id="secQuestion">
        <div class="section-label">Textbook Question</div>
        <h2 style="margin-top:6px">Problem</h2>
        <div class="question-box">
          {_q_text_html}
        </div>
        {_ref_html}
      </div>

      <div class="panel lesson" id="secProblem">
        <div class="section-label">Section 1 — Problem Statement</div>
        <h2 style="margin-top:6px">{question["title"]}</h2>
        <p><strong>Governing PDE:</strong>
          \\( u_t = \\alpha\\,u_{{xx}} \\) &nbsp;on&nbsp;
          \\( [{float(question["x_start"])},\\,{float(question["x_end"])}]
          \\times [0,\\,{float(question["final_time"])}] \\)
        </p>
        <div class="pde-badge">
          <div class="pde-badge-item"><strong>Diffusivity &alpha;</strong><span>{float(question["alpha"])}</span></div>
          <div class="pde-badge-item"><strong>Domain [a, b]</strong><span>[{float(question["x_start"])}, {float(question["x_end"])}]</span></div>
          <div class="pde-badge-item"><strong>Final Time T</strong><span>{float(question["final_time"])}</span></div>
          <div class="pde-badge-item"><strong>Grid Points N+1</strong><span>{int(question["number_of_space_points"])}</span></div>
        </div>
        <p>
          <strong>Boundary conditions:</strong>
          \\( U(x_0,t) = {float(question["left_boundary"])} \\) &nbsp;and&nbsp;
          \\( U(x_N,t) = {float(question["right_boundary"])} \\) for all \\( t \\ge 0 \\)
        </p>
        <p><strong>Initial condition (t = 0):</strong> {ic_desc_html}</p>
        <p><strong>Exact solution available:</strong> {exact_type_label}</p>
      </div>

      <div class="panel lesson" id="secStability">
        <div class="section-label">Section 8 — Stability Analysis</div>
        <h2 style="margin-top:6px">FTCS Stability Analysis</h2>
        <p>
          FTCS is an explicit method, so each new row of values is computed
          directly from the previous row. For the heat equation, the time step
          must be small enough compared with the space step. The stability number is
          \\[
            r = \\frac{{\\alpha k}}{{h^2}}.
          \\]
          For the standard 1D heat equation, the stable region is
          \\[
            0 \\le r \\le \\frac{{1}}{{2}}.
          \\]
        </p>
        <div class="pde-badge">
          <div class="pde-badge-item"><strong>Your r</strong><span>{r:.6g}</span></div>
          <div class="pde-badge-item"><strong>Limit</strong><span>0.5</span></div>
          <div class="pde-badge-item"><strong>Status</strong>
            <span style="color:{stability_color}">{stability_label}</span></div>
          <div class="pde-badge-item"><strong>r / 0.5</strong><span>{r/0.5:.3f}&times;</span></div>
        </div>
        <p>
          When \\(r\\le 0.5\\), the update behaves like a weighted average of nearby
          values, so heat spreads smoothly. When \\(r>0.5\\), the centre coefficient
          \\(1-2r\\) becomes negative. Then the computation can oscillate or grow
          even though the physical heat problem should be smoothing out.
        </p>
        {stability_verdict_html}
      </div>

      <div class="panel lesson" id="secRod">
        <div class="section-label">Section 4 — Rod Visualization</div>
        <h2>Iron Rod Heat Diffusion</h2>
        <p>
          A one-dimensional heat-transfer problem can be viewed as heat moving
          along a thin rod. The rod model is useful because the temperature changes
          mainly in one direction, while the thickness of the material is treated
          as small compared with its length.
        </p>
        <p>
          The bar below represents a cross-section of an iron rod. Colour encodes
          temperature using a blackbody scale: dark/black = cold, red = warm,
          orange = hot, white/yellow = very hot.
          Press <strong>&#9654; Play</strong> to watch heat spread. Drag the slider to any time level.
        </p>
        <div class="lecture-note">
          The colour shows the temperature at each position on the rod. Over time,
          heat moves from hotter regions toward cooler neighbouring regions. In a
          stable computation, the colours should gradually spread out and become
          smoother.
        </div>
        {_g_rod_div}
        <div id="rodAnimation" class="plot-rod"></div>
      </div>
      <div class="panel lesson" id="secManual">
        <div class="section-label">Section 7 — Matrix Generalisation</div>
        <h2>Matrix Method: \\(U^{{n+1}} = A U^n\\)</h2>
        <p>
          The FTCS formula first computes one interior grid point from three values
          at the previous time level:
          \\[
            U_i^{{n+1}}
            =
            rU_{{i-1}}^n
            +
            (1-2r)U_i^n
            +
            rU_{{i+1}}^n.
          \\]
          For this question, after substituting the value of \\(r\\), the update
          becomes
          \\[
            {numeric_scalar_scheme}.
          \\]
          To update all interior grid points together, collect the unknowns into
          one column vector:
          \\[
            U_{{\\mathrm{{int}}}}^n =
            \\begin{{bmatrix}}
              U_1^n & U_2^n & \\cdots & U_{{N-1}}^n
            \\end{{bmatrix}}^T.
          \\]
          One matrix multiplication then updates the whole rod interior from time
          level \\(n\\) to time level \\(n+1\\).
        </p>
        <div class="formula">
          \\[
            {matrix_formula}
          \\]
          \\[
            A = {matrix_latex}
          \\]
        </div>
        <div class="lecture-note">
          The matrix \\(A\\) is tridiagonal because the heat equation only looks at
          three neighbouring values: left, centre, and right. The coefficient
          \\(r\\) sits beside the left and right neighbours. The coefficient
          \\(1-2r\\) sits on the centre value.
          {boundary_matrix_note}
        </div>
        {matrix_examples_html}
        <p>{stability_message}</p>
        <p>{physical_message}</p>
      </div>
      <div class="panel lesson" id="secTable">
        <div class="section-label">Section 8 — Complete Numerical Table</div>
        <h2>Complete FTCS Table</h2>
        <p>
          This table is the computed solution. Each row is one time level
          \\(j\\), and each column is one spatial grid point \\(i\\). Reading across
          a row shows the temperature profile along the rod at one moment. Reading
          down a column shows how one fixed point in the rod changes over time.
        </p>
        <div class="table-scroll">
          <table class="complete-table">
            {complete_header}
            {complete_table_rows}
          </table>
        </div>
      </div>
      <div class="panel lesson" id="secPlayer">
        <div class="section-label">Section 9 — Temperature Evolution Simulation</div>
        <h2>Time Evolution Player</h2>
        <p>
          Move the slider or press Play. The 2D graph and the red 3D time slice
          update together. This panel shows the meaning of
          \\(U_i^j\\): for one selected \\(j\\), the graph draws all values
          \\(U_0^j, U_1^j, \\ldots, U_N^j\\) along the rod.
        </p>
        <div class="control-row">
          <button id="playButton" class="control-button" type="button">▶ Play</button>
          <button id="pauseButton" class="control-button" type="button">⏸ Pause</button>
          <input id="timeSlider" type="range" min="0" max="{len(time_values) - 1}" step="1" value="0">
          <strong id="timeReadout">j = 0</strong>
        </div>
        <div id="controlledTimePlot" class="plot"></div>
        <p id="timeExplanation">At j = 0, this is the initial temperature profile before diffusion starts.</p>
      </div>
      <div class="panel" id="secProfiles">
        <div class="section-label">Section 9b — Solution Profiles</div>
        <h2>Solution Profiles</h2>
        <p>
          This plot places several time levels on the same axes. Earlier curves
          show the starting shape, and later curves show how that shape changes.
          In a stable heat-diffusion problem, sharp peaks reduce and the profile
          becomes smoother.
          {exact_profile_note}
        </p>
        {_g_profiles_div}
        <div id="profilePlot" class="plot"></div>
      </div>
      <div class="panel" id="secSurface">
        <div class="section-label">Section 10 — 3D Surface Evolution</div>
        <h2>Rotatable 3D Surface</h2>
        <p>
          Drag to rotate. The horizontal direction is space, the depth direction is
          time, and the height is temperature. This surface is the whole
          solution table drawn as a graph. The red curve tracks the current
          time level from the player above.
        </p>
        {_g_surface_div}
        <div id="surfacePlot" class="plot"></div>
      </div>
      <div class="panel" id="secMesh">
        <div class="section-label">Section 5 — Mesh/Grid Visualization</div>
        <h2>Mesh Grid Points</h2>
        <p>
          Each dot is one numerical location \\(U_i^j\\).
          <strong>Click any interior node</strong> to see the FTCS stencil
          applied at that point in the inspector below.
        </p>
        <div class="lecture-note">
          The horizontal axis lists positions on the rod. The vertical axis lists
          time levels. The colour legend tells you which time level a point belongs
          to. Cooler colours are earlier sampled time levels; warmer colours are
          later sampled time levels. A boundary node is fixed by the problem, while
          an interior node is computed from nearby nodes one row below.
        </div>
        <div id="meshPlot" class="plot"></div>
      </div>

      <div class="panel lesson" id="secStencil">
        <div class="section-label">Section 6 — FTCS Stencil Visualization &amp; Calculation Inspector</div>
        <h2 style="margin-top:6px">FTCS Step-by-Step Calculation</h2>
        <p>
          The FTCS stencil uses
          <span class="pill" style="background:#1d4ed8">3 known nodes</span>
          at time level \\(j\\!-\\!1\\) to predict
          <span class="pill" style="background:#dc2626">1 unknown node</span>
          at level \\(j\\):
          \\[
            U_i^j = r\\,U_{{i-1}}^{{j-1}} + (1-2r)\\,U_i^{{j-1}} + r\\,U_{{i+1}}^{{j-1}}
          \\]
          Click any <strong>interior</strong> mesh node above (or any point on the time-player)
          to see the full calculation with substituted values.
        </p>
        <div id="stencilContent">
          <div style="text-align:center;color:#94a3b8;padding:48px 0;font-style:italic;font-size:1.05em">
            &#8593; Click an interior mesh node above to inspect the FTCS calculation here.
          </div>
        </div>
      </div>

      {error_panel_html}

      <div class="panel" id="secSummary">
        <div class="section-label">Section 11 — Numerical Summary</div>
        <h2 style="margin-top:6px">Numerical Summary</h2>
        <div class="table-scroll">
          <table>
            <tr><th>Quantity</th><th>Symbol</th><th>Value</th></tr>
            <tr><td>Space step</td><td>\\(h\\)</td><td><strong>{h:.8g}</strong></td></tr>
            <tr><td>Time step</td><td>\\(k\\)</td><td><strong>{k:.8g}</strong></td></tr>
            <tr><td>Stability ratio</td><td>\\(r = \\alpha k / h^2\\)</td>
                <td><strong>{r:.8g}</strong> &nbsp;
                  <span class="pill" style="background:{stability_color}">{stability_label}</span>
                </td></tr>
            <tr><td>Interior spatial points</td><td>\\(N-1\\)</td><td><strong>{num_interior_pts}</strong></td></tr>
            <tr><td>Time steps computed</td><td>\\(M\\)</td><td><strong>{num_time_steps_total}</strong></td></tr>
            <tr><td>Total interior unknowns</td><td>\\((N-1)\\cdot M\\)</td>
                <td><strong>{total_unknowns:,}</strong></td></tr>
            {_err_rows}
          </table>
        </div>
      </div>
    </section>
    <aside class="side">
      <div id="sideResizeHandle" title="Drag to resize · Double-click to reset"></div>
      <div class="panel lesson">
        <h2>What This PDE Models</h2>
        <p>
          The <strong>heat equation</strong> \\(u_t = \\alpha\\,u_{{xx}}\\) says:
          <em>the rate at which temperature changes at a point equals \\(\\alpha\\) times
          how sharply the temperature profile curves at that point.</em>
        </p>
        <ul style="margin:8px 0;padding-left:18px;line-height:2">
          <li><strong>Hot peak</strong> &rarr; profile curves downward &rarr; temperature <em>falls</em> (heat flows away).</li>
          <li><strong>Cold valley</strong> &rarr; profile curves upward &rarr; temperature <em>rises</em> (heat flows in).</li>
        </ul>
        <p>Result: every sharp feature smooths out; the rod eventually reaches a uniform temperature.</p>
        <p>{exact_message}</p>
      </div>
      <div class="panel lesson">
        <h2>Notation Link</h2>
        <p>
          The continuous solution is written as \\(u(x,t)\\). On a grid, the
          position and time are sampled as
          \\[
            x_i = ih, \\qquad t_j = jk,
          \\]
          where \\(h=\\Delta x\\) is the space step and \\(k=\\Delta t\\) is the
          time step.
        </p>
        <div class="formula">
          \\[
            u(x_i,t_j) \\approx u(ih,jk) \\approx U_i^j.
          \\]
        </div>
        <p>
          The symbol \\(U_i^j\\) means one value at one grid point. The symbol
          \\(U^n\\), or more clearly \\(U_{{\\mathrm{{int}}}}^n\\), means a column
          vector containing many interior values at the same time level:
          \\[
            U_{{\\mathrm{{int}}}}^n =
            \\begin{{bmatrix}}
              U_1^n & U_2^n & \\cdots & U_{{N-1}}^n
            \\end{{bmatrix}}^T.
          \\]
        </p>
      </div>
      <div class="panel lesson">
        <h2>General FTCS Matrix</h2>
        <p>
          For \\(N+1\\) spatial grid points, the unknown interior vector has
          \\(N-1\\) entries:
          \\[
            U_{{\\mathrm{{int}}}}^n =
            \\begin{{bmatrix}}
              U_1^n & U_2^n & \\cdots & U_{{N-1}}^n
            \\end{{bmatrix}}^T.
          \\]
        </p>
        <div class="formula">
          \\[
            A = {general_matrix_latex}
          \\]
        </div>
        <p>
          The diagonal term keeps part of the current point. The two neighbouring
          diagonals bring in heat from the left and right points.
        </p>
      </div>
      <div class="panel lesson">
        <h2>Key Terms</h2>
        <dl style="margin:0;line-height:1.9;font-size:.92em">
          <dt style="font-weight:700">\\(\\alpha\\) — Thermal diffusivity</dt>
          <dd style="margin:0 0 8px 14px">
            How fast heat spreads through the material.
            High \\(\\alpha\\) (e.g. copper) &rarr; fast spreading.
            Low \\(\\alpha\\) (e.g. concrete) &rarr; slow spreading.
            <br><strong>This problem:</strong> \\(\\alpha = {float(question["alpha"]):.6g}\\)
          </dd>
          <dt style="font-weight:700">\\(h\\) — Space step</dt>
          <dd style="margin:0 0 8px 14px">
            Gap between neighbouring grid points in space.
            Smaller \\(h\\) &rarr; finer grid &rarr; more accurate result.
            <br><strong>This problem:</strong> \\(h = {h:.6g}\\)
          </dd>
          <dt style="font-weight:700">\\(k\\) — Time step</dt>
          <dd style="margin:0 0 8px 14px">
            How far forward in time each FTCS step moves.
            Determined from \\(r\\) and \\(h\\).
            <br><strong>This problem:</strong> \\(k = {k:.6g}\\)
          </dd>
          <dt style="font-weight:700">\\(r = \\alpha k / h^2\\) — Stability ratio</dt>
          <dd style="margin:0 0 8px 14px">
            The critical number for FTCS.
            Must be \\(\\le 0.5\\) — otherwise the method amplifies errors instead of damping them,
            producing completely wrong values.
            <br><strong>This problem:</strong>
            \\(r = {r:.6g}\\)&nbsp;
            <span class="pill" style="background:{stability_color}">{stability_label}</span>
          </dd>
          <dt style="font-weight:700">Boundary condition</dt>
          <dd style="margin:0 0 8px 14px">
            The fixed temperature at both ends of the rod, held constant for all time.
            Here: left end \\(={float(question["left_boundary"]):.4g}\\),
            right end \\(={float(question["right_boundary"]):.4g}\\).
          </dd>
          <dt style="font-weight:700">Initial condition</dt>
          <dd style="margin:0 0 8px 14px">
            The temperature distribution at \\(t=0\\), before diffusion begins.
          </dd>
          <dt style="font-weight:700">FTCS</dt>
          <dd style="margin:0 0 0 14px">
            Forward-Time Centered-Space — the numerical recipe used to compute each new
            time level from the previous one:
            \\[U_i^{{j+1}} = r\\,U_{{i-1}}^j + (1-2r)\\,U_i^j + r\\,U_{{i+1}}^j.\\]
          </dd>
        </dl>
      </div>
      <div class="panel lesson" id="inspectorPanel">
        <div class="section-label">Selected Node — Inspector</div>
        <h2 style="margin-top:6px">Point Details</h2>
        <div id="inspector" style="font-size:.88em;line-height:1.85;color:#475569">
          <p style="color:#94a3b8;font-style:italic">
            Click any point on any chart to inspect it. For interior mesh nodes,
            the full FTCS calculation appears in the Stencil panel.
          </p>
        </div>
      </div>
    </aside>
  </main>
  <script>
    const figures      = {figures_json};
    const xValues      = {x_json};
    const timeValues   = {time_json};
    const solutionValues = {solution_json};
    const rValue       = {r};
    const hasErrorPlot = {"true" if has_error_panel else "false"};
    const config = {{responsive: true, displaylogo: false}};

    let currentTimeIndex = 0;
    let playTimer = null;

    // Initialise every figure; register animation frames where present.
    // Using async so newPlot Promises resolve before the slider is set up.
    let timePlotInitialized = false;

    async function initDashboard() {{
      for (const [id, fig] of Object.entries(figures)) {{
        await Plotly.newPlot(id, fig.data, fig.layout, config);
        if (fig.frames && fig.frames.length) {{
          await Plotly.addFrames(id, fig.frames);
        }}
      }}
      // Draw the initial time-slice only after all plots exist.
      await drawControlledTimePlot(0);
      // Bind Plotly click events now that every plot div is initialised.
      bindPlotlyClicks();
      if (window.MathJax && window.MathJax.typesetPromise) {{
        window.MathJax.typesetPromise();
      }}
    }}

    initDashboard();

    // ---- Dark mode ----
    const LIGHT_BG   = "#ffffff";
    const LIGHT_PAPER = "#ffffff";
    const LIGHT_FONT  = "#1f2933";
    const LIGHT_GRID  = "#e5e7eb";

    const DARK_BG    = "#111827";
    const DARK_PAPER = "#1f2937";
    const DARK_FONT  = "#e5e7eb";
    const DARK_GRID  = "#374151";

    function allPlotIds() {{
      const ids = ["profilePlot","surfacePlot","meshPlot",
                   "rodAnimation","controlledTimePlot"];
      if (hasErrorPlot) ids.push("errorPlot");
      return ids;
    }}

    function applyThemeToPlots(dark) {{
      const bg    = dark ? DARK_BG    : LIGHT_BG;
      const paper = dark ? DARK_PAPER : LIGHT_PAPER;
      const font  = dark ? DARK_FONT  : LIGHT_FONT;
      const grid  = dark ? DARK_GRID  : LIGHT_GRID;
      const update = {{
        paper_bgcolor: paper,
        plot_bgcolor: bg,
        font: {{color: font}},
        "xaxis.gridcolor": grid,
        "yaxis.gridcolor": grid,
        "xaxis.linecolor": grid,
        "yaxis.linecolor": grid,
      }};
      for (const id of allPlotIds()) {{
        const el = document.getElementById(id);
        if (el && el.data) Plotly.relayout(id, update);
      }}
      // 3D scene colours
      if (document.getElementById("surfacePlot")?.data) {{
        Plotly.relayout("surfacePlot", {{
          paper_bgcolor: paper,
          "scene.bgcolor": bg,
          "scene.xaxis.gridcolor": grid,
          "scene.yaxis.gridcolor": grid,
          "scene.zaxis.gridcolor": grid,
        }});
      }}
    }}

    document.getElementById("themeToggle").addEventListener("click", () => {{
      const isDark = document.body.classList.toggle("dark");
      document.getElementById("themeToggle").textContent = isDark ? "Light mode" : "Dark mode";
      applyThemeToPlots(isDark);
    }});

    // ---- Global speed control — updates all animations ----
    document.getElementById("speedSelect").addEventListener("change", (e) => {{
      const newDuration = Number(e.target.value);
      // Update the Plotly rod animation play-button frame duration
      const rodDiv = document.getElementById("rodAnimation");
      if (rodDiv && rodDiv.layout && rodDiv.layout.updatemenus && rodDiv.layout.updatemenus.length) {{
        Plotly.relayout("rodAnimation", {{
          "updatemenus[0].buttons[0].args[1].frame.duration": newDuration
        }}).catch(() => {{}});
      }}
      // If the profile slider is currently playing, restart it with the new speed
      if (playTimer !== null) {{
        clearInterval(playTimer);
        playTimer = setInterval(() => {{
          const next = currentTimeIndex >= solutionValues.length - 1 ? 0 : currentTimeIndex + 1;
          setTimeIndex(next);
        }}, newDuration);
      }}
    }});

    // ---- Side panel drag-to-resize ----
    (function () {{
      const handle = document.getElementById("sideResizeHandle");
      const sideEl = document.querySelector(".side");
      const mainEl = document.querySelector("main");
      let dragging = false, startX = 0, startW = 0;

      handle.addEventListener("mousedown", (e) => {{
        e.preventDefault();
        dragging = true;
        startX = e.clientX;
        startW = sideEl.getBoundingClientRect().width;
        handle.classList.add("dragging");
        document.body.style.cursor = "col-resize";
        document.body.style.userSelect = "none";
      }});

      document.addEventListener("mousemove", (e) => {{
        if (!dragging) return;
        const dx = startX - e.clientX;           // drag left = wider panel
        const newW = Math.max(200, Math.min(700, startW + dx));
        mainEl.style.gridTemplateColumns = `minmax(0,1fr) ${{newW}}px`;
      }});

      document.addEventListener("mouseup", () => {{
        if (!dragging) return;
        dragging = false;
        handle.classList.remove("dragging");
        document.body.style.cursor = "";
        document.body.style.userSelect = "";
      }});

      handle.addEventListener("dblclick", () => {{
        mainEl.style.gridTemplateColumns = "";    // snap back to CSS default
      }});
    }})();

    // ---- Side panel collapse / expand ----
    (function () {{
      const btn    = document.getElementById("collapsePanel");
      const sideEl = document.querySelector(".side");
      const mainEl = document.querySelector("main");
      let collapsed = false;
      let savedCols = null;

      btn.addEventListener("click", () => {{
        if (collapsed) {{
          // Expand: restore saved width or CSS default
          mainEl.style.gridTemplateColumns = savedCols || "";
          sideEl.style.display  = "";
          collapsed = false;
          btn.innerHTML = "&#8614;&nbsp;Panel";
          btn.title = "Collapse side panel";
        }} else {{
          // Collapse: remember current width and hide side
          savedCols = mainEl.style.gridTemplateColumns || null;
          mainEl.style.gridTemplateColumns = "minmax(0,1fr) 0px";
          sideEl.style.display  = "none";
          collapsed = true;
          btn.innerHTML = "&#8612;&nbsp;Panel";
          btn.title = "Expand side panel";
        }}
      }});
    }})();

    // ---- Time slice player ----
    function timeSliceLayout(j) {{
      return {{
        title: `Single Time Level: j=${{j}}, t=${{timeValues[j].toFixed(6)}}`,
        xaxis: {{title: "Position x"}},
        yaxis: {{
          title: "Temperature",
          range: [{float(np.min(solution)) - 5}, {float(np.max(solution)) + 5}]
        }},
        hovermode: "closest"
      }};
    }}

    function updateSurfaceTimeSlice(j) {{
      Plotly.restyle("surfacePlot", {{
        y: [xValues.map(() => timeValues[j])],
        z: [solutionValues[j]]
      }}, [1]);
    }}

    function setTimeIndex(j) {{
      currentTimeIndex = Math.max(0, Math.min(solutionValues.length - 1, Number(j)));
      document.getElementById("timeSlider").value = currentTimeIndex;
      drawControlledTimePlot(currentTimeIndex);
      updateSurfaceTimeSlice(currentTimeIndex);
    }}

    async function drawControlledTimePlot(j) {{
      const isDark = document.body.classList.contains("dark");
      const baseLayout = timeSliceLayout(j);
      const layout = Object.assign({{}}, baseLayout, {{
        paper_bgcolor: isDark ? DARK_PAPER : LIGHT_PAPER,
        plot_bgcolor:  isDark ? DARK_BG    : LIGHT_BG,
        font: {{color: isDark ? DARK_FONT : LIGHT_FONT}},
      }});
      const trace = {{
        x: xValues,
        y: solutionValues[j],
        mode: "lines+markers",
        line: {{width: 3, color: "#d62828"}},
        marker: {{size: 8}},
        customdata: xValues.map((_, i) => [i, j, timeValues[j]]),
        hovertemplate:
          "i=%{{customdata[0]}}<br>j=%{{customdata[1]}}<br>" +
          "x=%{{x:.5f}}<br>t=%{{customdata[2]:.5f}}<br>" +
          "U=%{{y:.5f}}<extra></extra>"
      }};
      if (!timePlotInitialized) {{
        await Plotly.newPlot("controlledTimePlot", [trace], layout, config);
        timePlotInitialized = true;
      }} else {{
        Plotly.react("controlledTimePlot", [trace], layout, config);
      }}
      document.getElementById("timeReadout").textContent = `j = ${{j}}, t = ${{timeValues[j].toFixed(6)}}`;
      document.getElementById("timeExplanation").textContent = explainTimeLevel(j);
    }}

    function explainTimeLevel(j) {{
      const vals = solutionValues[j];
      const maxVal = Math.max(...vals);
      const minVal = Math.min(...vals);
      if (j === 0) return "At j = 0, this is the initial temperature profile before diffusion starts.";
      if (rValue > 0.5) return `At j=${{j}}: r > 0.5, so oscillations may be numerical instability. min=${{minVal.toFixed(4)}}, max=${{maxVal.toFixed(4)}}.`;
      return `At j=${{j}}: heat has flowed from hot to cool neighbours. Profile should gradually smooth. min=${{minVal.toFixed(4)}}, max=${{maxVal.toFixed(4)}}.`;
    }}

    function formulaForPoint(j, i) {{
      if (j === 0) return "Initial condition — FTCS has not been applied yet.";
      if (i === 0 || i === xValues.length - 1) return "Boundary point — value is fixed by the boundary condition.";
      const left   = solutionValues[j - 1][i - 1];
      const center = solutionValues[j - 1][i];
      const right  = solutionValues[j - 1][i + 1];
      const result = solutionValues[j][i];
      return (
        `FTCS step for U_${{i}}^${{j}}:\\n` +
        `U_${{i}}^${{j}} = r·U_${{i-1}}^${{j-1}} + (1-2r)·U_${{i}}^${{j-1}} + r·U_${{i+1}}^${{j-1}}\\n` +
        `       = ${{rValue.toFixed(4)}}×${{left.toFixed(4)}} + (1-2×${{rValue.toFixed(4)}})×${{center.toFixed(4)}} + ${{rValue.toFixed(4)}}×${{right.toFixed(4)}}\\n` +
        `       = ${{result.toFixed(6)}}`
      );
    }}

    // ---- DOM event listeners (safe to bind before plots are ready) ----
    document.getElementById("timeSlider").addEventListener("input", (e) => {{
      setTimeIndex(Number(e.target.value));
    }});

    document.getElementById("playButton").addEventListener("click", () => {{
      if (playTimer !== null) return;
      const delay = Number(document.getElementById("speedSelect").value);
      playTimer = setInterval(() => {{
        const next = currentTimeIndex >= solutionValues.length - 1 ? 0 : currentTimeIndex + 1;
        setTimeIndex(next);
      }}, delay);
    }});

    document.getElementById("pauseButton").addEventListener("click", () => {{
      clearInterval(playTimer);
      playTimer = null;
    }});

    // ---- Inspector helper ----
    // profilePlot  customdata: [j, t_value, i]
    // meshPlot     customdata: [i, j]
    // controlledTimePlot customdata: [i, j, t]
    function showPoint(source, pt) {{
      const inspEl = document.getElementById("inspector");
      let rows = [];
      let nodeI = null, nodeJ = null;

      if (source === "controlledTimePlot" && pt.customdata) {{
        nodeI = pt.customdata[0]; nodeJ = pt.customdata[1];
        const t = pt.customdata[2];
        rows.push(`<strong>U<sub>${{nodeI}}</sub><sup>${{nodeJ}}</sup></strong> = ${{Number(pt.y).toFixed(6)}}`);
        rows.push(`Space index &nbsp;<strong>i = ${{nodeI}}</strong> &ensp; x = ${{Number(pt.x).toFixed(5)}}`);
        rows.push(`Time index &nbsp;<strong>j = ${{nodeJ}}</strong> &ensp; t = ${{Number(t).toFixed(5)}}`);
      }} else if (source === "profilePlot" && pt.customdata) {{
        nodeJ = pt.customdata[0]; const t = pt.customdata[1]; nodeI = pt.customdata[2];
        rows.push(`<strong>U<sub>${{nodeI}}</sub><sup>${{nodeJ}}</sup></strong> = ${{Number(pt.y).toFixed(6)}}`);
        rows.push(`Space index &nbsp;<strong>i = ${{nodeI}}</strong> &ensp; x = ${{Number(pt.x).toFixed(5)}}`);
        rows.push(`Time index &nbsp;<strong>j = ${{nodeJ}}</strong> &ensp; t = ${{Number(t).toFixed(5)}}`);
      }} else if (source === "meshPlot" && pt.customdata) {{
        nodeI = pt.customdata[0]; nodeJ = pt.customdata[1];
        rows.push(`<strong>Node U<sub>${{nodeI}}</sub><sup>${{nodeJ}}</sup></strong>`);
        rows.push(`Space index &nbsp;<strong>i = ${{nodeI}}</strong> &ensp; x = ${{Number(pt.x).toFixed(5)}}`);
        rows.push(`Time index &nbsp;<strong>j = ${{nodeJ}}</strong> &ensp; t = ${{Number(pt.y).toFixed(5)}}`);
        rows.push(`Value = ${{solutionValues[nodeJ][nodeI].toFixed(6)}}`);
      }} else if (source === "surfacePlot") {{
        rows.push(`x = ${{Number(pt.x).toFixed(5)}}`);
        rows.push(`t = ${{Number(pt.y).toFixed(5)}}`);
        rows.push(`U = ${{Number(pt.z).toFixed(6)}}`);
      }} else {{
        if (pt.x !== undefined) rows.push(`x = ${{pt.x}}`);
        if (pt.y !== undefined) rows.push(`y = ${{pt.y}}`);
        if (pt.z !== undefined) rows.push(`z = ${{pt.z}}`);
      }}

      inspEl.innerHTML = rows.map(r => `<div style="padding:2px 0;border-bottom:1px solid rgba(148,163,184,.18)">${{r}}</div>`).join("");

      if (nodeI !== null && nodeJ !== null) {{
        updateStencilPanel(nodeI, nodeJ);
        const sp = document.getElementById("secStencil");
        if (sp) sp.scrollIntoView({{behavior:"smooth", block:"nearest"}});
      }}
    }}

    // ---- FTCS stencil panel ----
    function updateStencilPanel(i, j) {{
      const panel = document.getElementById("stencilContent");
      if (!panel) return;
      const n = xValues.length;

      if (j === 0) {{
        panel.innerHTML =
          `<div class="step-box"><div class="step-label">Initial Condition — j = 0</div>
           U<sub>${{i}}</sub><sup>0</sup> = <strong>${{solutionValues[0][i].toFixed(6)}}</strong><br>
           This value comes from the initial condition. FTCS has not yet been applied.</div>`;
        return;
      }}
      if (i === 0 || i >= n - 1) {{
        panel.innerHTML =
          `<div class="step-box"><div class="step-label">Boundary Point</div>
           U<sub>${{i}}</sub><sup>${{j}}</sup> = <strong>${{solutionValues[j][i].toFixed(6)}}</strong><br>
           This is a fixed boundary value — it is not computed by FTCS.</div>`;
        return;
      }}

      const uL  = solutionValues[j-1][i-1];
      const uC  = solutionValues[j-1][i];
      const uR  = solutionValues[j-1][i+1];
      const uNew = solutionValues[j][i];
      const r   = rValue, w2 = 1 - 2*r;
      const t1  = r*uL, t2 = w2*uC, t3 = r*uR;
      const computed = t1 + t2 + t3;
      const diff = uNew - uC;
      const physDir = diff >  1e-9 ? "increased &mdash; net heat gained from neighbours"
                    : diff < -1e-9 ? "decreased &mdash; net heat lost to neighbours"
                    :                "unchanged &mdash; balanced heat flow";

      panel.innerHTML = `
        <div class="stencil-wrap">
          <div style="font-size:.78em;color:#64748b;font-weight:600;letter-spacing:.05em;text-transform:uppercase">
            Known &mdash; time level j&minus;1 = ${{j-1}} &nbsp;(t = ${{timeValues[j-1].toFixed(5)}})
          </div>
          <div class="stencil-row">
            <div class="stencil-node known">
              <span style="font-size:.68em;opacity:.85">U<sub>${{i-1}}</sub><sup>${{j-1}}</sup></span>
              <strong style="font-size:.95em">${{uL.toFixed(5)}}</strong>
              <span style="font-size:.62em;opacity:.75">left neighbour</span>
            </div>
            <div class="stencil-node known">
              <span style="font-size:.68em;opacity:.85">U<sub>${{i}}</sub><sup>${{j-1}}</sup></span>
              <strong style="font-size:.95em">${{uC.toFixed(5)}}</strong>
              <span style="font-size:.62em;opacity:.75">centre</span>
            </div>
            <div class="stencil-node known">
              <span style="font-size:.68em;opacity:.85">U<sub>${{i+1}}</sub><sup>${{j-1}}</sup></span>
              <strong style="font-size:.95em">${{uR.toFixed(5)}}</strong>
              <span style="font-size:.62em;opacity:.75">right neighbour</span>
            </div>
          </div>
          <div class="stencil-arrows">&#8600;&ensp;&#8595;&ensp;&#8601;</div>
          <div class="stencil-row">
            <div class="stencil-node computed">
              <span style="font-size:.68em;opacity:.85">U<sub>${{i}}</sub><sup>${{j}}</sup></span>
              <strong style="font-size:.95em">${{uNew.toFixed(5)}}</strong>
              <span style="font-size:.62em;opacity:.75">PREDICTED</span>
            </div>
          </div>
          <div style="font-size:.78em;color:#64748b;font-weight:600;letter-spacing:.05em;text-transform:uppercase">
            Computed &mdash; time level j = ${{j}} &nbsp;(t = ${{timeValues[j].toFixed(5)}})
          </div>
        </div>

        <div class="step-box">
          <div class="step-label">Step 1 &mdash; Symbolic FTCS Formula</div>
          U<sub>${{i}}</sub><sup>${{j}}</sup>
          &nbsp;=&nbsp; r&thinsp;U<sub>${{i-1}}</sub><sup>${{j-1}}</sup>
          &nbsp;+&nbsp; (1&minus;2r)&thinsp;U<sub>${{i}}</sub><sup>${{j-1}}</sup>
          &nbsp;+&nbsp; r&thinsp;U<sub>${{i+1}}</sub><sup>${{j-1}}</sup>
        </div>
        <div class="step-box">
          <div class="step-label">Step 2 &mdash; Substitute r = ${{r.toFixed(4)}}, 1&minus;2r = ${{w2.toFixed(4)}}</div>
          U<sub>${{i}}</sub><sup>${{j}}</sup>
          &nbsp;=&nbsp; ${{r.toFixed(4)}}&thinsp;U<sub>${{i-1}}</sub><sup>${{j-1}}</sup>
          &nbsp;+&nbsp; ${{w2.toFixed(4)}}&thinsp;U<sub>${{i}}</sub><sup>${{j-1}}</sup>
          &nbsp;+&nbsp; ${{r.toFixed(4)}}&thinsp;U<sub>${{i+1}}</sub><sup>${{j-1}}</sup>
        </div>
        <div class="step-box">
          <div class="step-label">Step 3 &mdash; Substitute Numerical Values</div>
          = ${{r.toFixed(4)}} &times; (${{uL.toFixed(5)}})
          &nbsp;+&nbsp; ${{w2.toFixed(4)}} &times; (${{uC.toFixed(5)}})
          &nbsp;+&nbsp; ${{r.toFixed(4)}} &times; (${{uR.toFixed(5)}})
        </div>
        <div class="step-box">
          <div class="step-label">Step 4 &mdash; Evaluate Each Term</div>
          = ${{t1.toFixed(6)}}
          &nbsp;+&nbsp; ${{t2.toFixed(6)}}
          &nbsp;+&nbsp; ${{t3.toFixed(6)}}
        </div>
        <div class="step-box" style="border-left-color:#22c55e;background:#f0fdf4">
          <div class="step-label" style="color:#15803d">&#10003; Result</div>
          U<sub>${{i}}</sub><sup>${{j}}</sup> = <strong>${{computed.toFixed(6)}}</strong>
        </div>
        <p class="phys-note">
          &#9728; <strong>Physical meaning:</strong> Temperature at x&nbsp;=&nbsp;${{xValues[i].toFixed(4)}}
          has ${{physDir}} from t&nbsp;=&nbsp;${{timeValues[j-1].toFixed(5)}}
          to t&nbsp;=&nbsp;${{timeValues[j].toFixed(5)}}.
        </p>`;
    }}

    // Plotly .on() handlers require the plot to exist — bound inside initDashboard.
    function bindPlotlyClicks() {{
      ["controlledTimePlot","profilePlot","surfacePlot","meshPlot"].forEach((id) => {{
        const el = document.getElementById(id);
        if (el) el.on("plotly_click", (event) => {{
          if (event.points && event.points.length) showPoint(id, event.points[0]);
        }});
      }});
    }}
  </script>
</body>
</html>
"""

    dashboard_path.write_text(html, encoding="utf-8")
    return dashboard_path


# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------

def run_question_lab():
    """Run the complete FTCS practice-question workflow."""
    validate_question(QUESTION)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    x, time_values, h, k, r = build_mesh(QUESTION)
    solution = solve_with_ftcs(QUESTION, x, time_values, r)
    exact = exact_solution(QUESTION, x, time_values)
    error_analysis = compute_error_analysis(solution, exact)

    mesh_path = save_mesh_grid(x, time_values)
    table_txt_path, table_csv_path = save_solution_table(x, time_values, solution)
    simulation_data_path = save_simulation_data(x, time_values, solution)
    profiles_path = plot_solution_profiles(x, time_values, solution, exact)
    heatmap_path = plot_heatmap(x, time_values, solution)
    error_path = plot_error_analysis(time_values, error_analysis)

    output_files = {
        "mesh grid points": mesh_path,
        "solution table txt": table_txt_path,
        "solution table csv": table_csv_path,
        "simulation data": simulation_data_path,
        "solution profiles plot": profiles_path,
        "solution heatmap": heatmap_path,
    }
    if error_path is not None:
        output_files["error analysis plot"] = error_path

    report_path = build_report(
        QUESTION, x, time_values, h, k, r, solution, exact, error_analysis, output_files,
    )
    support_results = run_supporting_engines(QUESTION)
    build_dashboard(output_files, support_results)
    interactive_dashboard_path = build_interactive_dashboard(
        QUESTION, x, time_values, h, k, r, solution, exact, error_analysis, support_results,
    )

    print("FTCS question lab completed.")
    print(f"Report:               {report_path}")
    print(f"Interactive dashboard: {interactive_dashboard_path}")
    print(f"Stability ratio r = {r:.6f}")
    if r <= 0.5:
        print("Stability check: PASS (r <= 0.5)")
    else:
        print("Stability check: WARNING — r > 0.5, FTCS is unstable.")
    if error_analysis is not None:
        print(f"Max absolute error: {error_analysis['global_max_error']:.6e}")
        print(f"L2 error:           {error_analysis['global_l2_error']:.6e}")

    if QUESTION.get("open_dashboard_after_run", False):
        try:
            os.startfile(interactive_dashboard_path)
        except OSError as error:
            opened = webbrowser.open(interactive_dashboard_path.resolve().as_uri())
            if not opened:
                print(f"Could not open dashboard automatically: {error}")
                print("Open this file manually:")
                print(interactive_dashboard_path)


if __name__ == "__main__":
    run_question_lab()
