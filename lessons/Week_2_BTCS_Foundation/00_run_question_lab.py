"""
00_run_question_lab.py  —  Week 2: BTCS (Backward-Time Centered-Space)

Workflow
--------
1. Open question_config.py.
2. Set ACTIVE_QUESTION to the question number you want (1-11).
3. Run this file (or run question_config.py directly — it calls this file).
4. Check outputs/question_results/.

Mathematical model
------------------
The 1D heat equation:

    u_t = alpha * u_xx

The BTCS (implicit) scheme discretises the time derivative BACKWARD:

    -r U_{i-1}^{j+1} + (1+2r) U_i^{j+1} - r U_{i+1}^{j+1} = U_i^j

At each time step this gives a TRIDIAGONAL LINEAR SYSTEM to solve.
In matrix form:

    A U_int^{j+1} = U_int^j + b

where A has diagonal 1+2r and off-diagonals -r, and b contains boundary
contributions.

Stability: BTCS is UNCONDITIONALLY STABLE — no restriction on r = alpha*k/h².
Accuracy:  O(k) in time, O(h²) in space (same as FTCS).
"""

from pathlib import Path
import json
import math
import os
import subprocess
import sys
import webbrowser

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import solve_banded
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
    required_keys = [
        "title", "alpha", "x_start", "x_end", "final_time",
        "number_of_space_points", "left_boundary", "right_boundary",
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
    number_of_time_steps = max(1, int(round(float(question["final_time"]) / k)))
    time_values = np.linspace(0.0, float(question["final_time"]), number_of_time_steps + 1)
    return x, time_values, h, k, r


# ---------------------------------------------------------------------------
# Initial condition
# ---------------------------------------------------------------------------

def initial_condition(question, x):
    ic_type = question.get("initial_condition_type", "")
    if ic_type == "hot_center":
        u_val   = float(question.get("u_value", 100))
        u_left  = float(question.get("u_left",  0.4))
        u_right = float(question.get("u_right", 0.6))
        values = np.zeros_like(x)
        values[(x >= u_left) & (x <= u_right)] = u_val
        return values
    if ic_type == "sin_pi":
        return np.sin(np.pi * x)
    if ic_type == "custom_list":
        return np.asarray(question["initial_condition_values"], dtype=float)
    if ic_type == "function":
        return np.asarray(eval(question["initial_condition_expr"],
                               {"x": x, "np": np, "math": math}), dtype=float)
    raise ValueError(f"Unknown initial_condition_type: '{ic_type}'")


# ---------------------------------------------------------------------------
# BTCS solver — tridiagonal system at each time step
# ---------------------------------------------------------------------------

def solve_with_btcs(question, x, time_values, r):
    """
    Solve the heat equation with BTCS.

    At each time step solve: A U_int^{n+1} = U_int^n + b
    A: diagonal (1+2r), off-diagonals (-r). Unconditionally stable for all r > 0.
    """
    N = len(x)
    M_int = N - 2
    solution = np.zeros((len(time_values), N))
    solution[0] = initial_condition(question, x)
    solution[:, 0]  = float(question["left_boundary"])
    solution[:, -1] = float(question["right_boundary"])
    if M_int <= 0:
        return solution

    ab = np.zeros((3, M_int))
    ab[0, 1:]  = -r
    ab[1, :]   = 1.0 + 2.0 * r
    ab[2, :-1] = -r

    left_bc  = float(question["left_boundary"])
    right_bc = float(question["right_boundary"])

    for n in range(len(time_values) - 1):
        rhs = solution[n, 1:-1].copy()
        rhs[0]  += r * left_bc
        rhs[-1] += r * right_bc
        solution[n + 1, 1:-1] = solve_banded((1, 1), ab, rhs)
        solution[n + 1, 0]  = left_bc
        solution[n + 1, -1] = right_bc
    return solution


# ---------------------------------------------------------------------------
# Exact solution
# ---------------------------------------------------------------------------

def exact_solution(question, x, time_values):
    exact_type = question.get("exact_solution_type")
    if exact_type is None:
        return None
    alpha = float(question["alpha"])
    exact = np.zeros((len(time_values), len(x)))
    if exact_type == "sin_pi_zero_boundary":
        for j, t in enumerate(time_values):
            exact[j] = np.exp(-alpha * np.pi**2 * t) * np.sin(np.pi * x)
        return exact
    if exact_type == "function":
        expr = question.get("exact_solution_expr", "")
        for j, t in enumerate(time_values):
            exact[j] = eval(expr, {"x": x, "t": t, "np": np, "math": math, "alpha": alpha})
        return exact
    return None


# ---------------------------------------------------------------------------
# Error analysis
# ---------------------------------------------------------------------------

def compute_error_analysis(solution, exact):
    if exact is None:
        return None
    diff = np.abs(solution - exact)
    max_error_per_step = diff.max(axis=1)
    l2_error_per_step  = np.sqrt((diff**2).mean(axis=1))
    return {
        "max_error_per_step": max_error_per_step,
        "l2_error_per_step":  l2_error_per_step,
        "global_max_error":   float(max_error_per_step.max()),
        "global_l2_error":    float(l2_error_per_step.max()),
    }


# ---------------------------------------------------------------------------
# Output file writers
# ---------------------------------------------------------------------------

def save_mesh_grid(x, time_values):
    path = OUTPUT_DIR / "mesh_grid_points.csv"
    rows = ["x,t"]
    for t in time_values:
        for xi in x:
            rows.append(f"{xi},{t}")
    path.write_text("\n".join(rows), encoding="utf-8")
    return path


def save_solution_table(x, time_values, solution):
    txt_path = OUTPUT_DIR / "btcs_solution_table.txt"
    csv_path = OUTPUT_DIR / "btcs_solution_table.csv"
    header     = f"{'j':>6}  {'t':>12}  " + "  ".join(f"U_{i}^j" for i in range(len(x)))
    csv_header = "j,t," + ",".join(f"U_{i}" for i in range(len(x)))
    rows, csv_rows = [header], [csv_header]
    for j, (t, row) in enumerate(zip(time_values, solution)):
        vals = "  ".join(f"{v:12.6f}" for v in row)
        rows.append(f"{j:>6}  {t:>12.6f}  {vals}")
        csv_rows.append(f"{j},{t}," + ",".join(f"{v:.8f}" for v in row))
    txt_path.write_text("\n".join(rows), encoding="utf-8")
    csv_path.write_text("\n".join(csv_rows), encoding="utf-8")
    return txt_path, csv_path


def save_simulation_data(x, time_values, solution):
    path = OUTPUT_DIR / "simulation_data_long.csv"
    rows = ["j,t,i,x,U"]
    for j, t in enumerate(time_values):
        for i, xi in enumerate(x):
            rows.append(f"{j},{t},{i},{xi},{solution[j, i]}")
    path.write_text("\n".join(rows), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Static matplotlib plots
# ---------------------------------------------------------------------------

def plot_solution_profiles(x, time_values, solution, exact=None):
    output_path = OUTPUT_DIR / "solution_profiles.png"
    selected = np.linspace(0, len(time_values) - 1, min(6, len(time_values)), dtype=int)
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#1f2937")
    ax.set_facecolor("#111827")
    palette = ["#e63946", "#f4a261", "#2a9d8f", "#457b9d", "#a8dadc", "#ffffff"]
    for idx, j in enumerate(selected):
        color = palette[idx % len(palette)]
        ax.plot(x, solution[j], color=color, linewidth=2,
                label=f"j={j}, t={time_values[j]:.5f}", marker="o", markersize=4)
        if exact is not None:
            ax.plot(x, exact[j], color=color, linestyle="--", linewidth=1.2, alpha=0.7)
    ax.set_title("BTCS Solution Profiles", color="white")
    ax.set_xlabel("Position x", color="white")
    ax.set_ylabel("Temperature", color="white")
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#374151")
    ax.grid(True, alpha=0.3, color="#4b5563")
    ax.legend(facecolor="#1f2937", edgecolor="#374151", labelcolor="white")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)
    return output_path


def plot_heatmap(x, time_values, solution):
    output_path = OUTPUT_DIR / "solution_heatmap.png"
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#1f2937")
    ax.set_facecolor("#111827")
    im = ax.imshow(solution, aspect="auto", origin="lower", cmap="inferno",
                   extent=[x[0], x[-1], time_values[0], time_values[-1]])
    plt.colorbar(im, ax=ax, label="Temperature").ax.yaxis.label.set_color("white")
    ax.set_title("BTCS Space-Time Heatmap", color="white")
    ax.set_xlabel("Position x", color="white")
    ax.set_ylabel("Time t", color="white")
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#374151")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)
    return output_path


def plot_error_analysis(time_values, error_analysis):
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
    ax.set_title("Error Analysis: BTCS vs Exact Solution", color="white")
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
    report_path = OUTPUT_DIR / "question_report.md"
    if error_analysis is None:
        error_section = "No exact solution check requested."
    else:
        error_section = (
            f"- Global maximum absolute error: `{error_analysis['global_max_error']:.8e}`\n"
            f"- Global L2 (RMS) error:         `{error_analysis['global_l2_error']:.8e}`\n"
        )
    final_values = ", ".join(f"{v:.5f}" for v in solution[-1])
    report = f"""# BTCS Question Report

## Question
**Title:** {question["title"]}
**Reference:** {question.get("reference", "N/A")}
**PDE:** `{question.get("pde", "u_t = alpha u_xx")}`

## Parameters
| Symbol | Value |
|--------|-------|
| α | `{question["alpha"]}` |
| h | `{h:.10f}` |
| k | `{k:.10f}` |
| r = αk/h² | `{r:.10f}` |
| Stability | UNCONDITIONALLY STABLE (BTCS) |

## Final Values
```
{final_values}
```

## Error Analysis
{error_section}
"""
    for label, path in output_files.items():
        report += f"- {label}: `{path.name}`\n"
    report_path.write_text(report, encoding="utf-8")
    return report_path


# ---------------------------------------------------------------------------
# Supporting engines
# ---------------------------------------------------------------------------

def run_supporting_engines(question):
    if not question.get("run_supporting_engines", True):
        return {}
    engine_files = [
        ("mesh dependency engine",   "01_grid_and_mesh_visualization.py"),
        ("manual BTCS table engine", "02_manual_btcs_table.py"),
        ("standard heat simulation", "03_btcs_heat_equation_simulation.py"),
        ("3D surface engine",        "04_btcs_3d_surface_plot.py"),
        ("stability experiment",     "06_stability_experiment.py"),
    ]
    if question.get("generate_animation", True):
        engine_files.insert(3, ("animation engine", "05_btcs_animation.py"))
    environment = os.environ.copy()
    environment["MPLBACKEND"] = "Agg"
    results = {}
    for label, filename in engine_files:
        script = BASE_DIR / filename
        if not script.exists():
            results[label] = f"skipped — {filename} not found"
            continue
        try:
            proc = subprocess.run(
                [sys.executable, str(script)],
                capture_output=True, text=True, timeout=120, env=environment,
            )
            results[label] = "ok" if proc.returncode == 0 else f"error: {proc.stderr[:200]}"
        except subprocess.TimeoutExpired:
            results[label] = "timeout"
        except Exception as exc:
            results[label] = f"exception: {exc}"
    return results


def build_dashboard(output_files, support_results):
    dashboard_path = OUTPUT_DIR / "static_dashboard.html"
    file_rows   = "".join(f"<tr><td>{l}</td><td><code>{p.name}</code></td></tr>"
                          for l, p in output_files.items())
    engine_rows = "".join(f"<tr><td>{l}</td><td>{s}</td></tr>"
                          for l, s in support_results.items()) or \
                  "<tr><td colspan='2'>No supporting engines ran.</td></tr>"
    html = (f"<!doctype html><html><head><meta charset='utf-8'><title>BTCS Dashboard</title></head>"
            f"<body><h1>BTCS Outputs</h1><table border='1'>{file_rows}</table>"
            f"<h2>Engines</h2><table border='1'>{engine_rows}</table></body></html>")
    dashboard_path.write_text(html, encoding="utf-8")
    return dashboard_path


# ---------------------------------------------------------------------------
# Plotly helper figures
# ---------------------------------------------------------------------------

def _build_rod_animation_figure(x, time_values, solution):
    n_theta = 32
    theta   = np.linspace(0, 2 * np.pi, n_theta)
    R       = 0.12
    X_cyl   = np.tile(x, (n_theta, 1))
    Y_cyl   = R * np.outer(np.cos(theta), np.ones(len(x)))
    Z_cyl   = R * np.outer(np.sin(theta), np.ones(len(x)))
    z_min   = float(np.min(solution))
    z_max   = float(np.max(solution))
    if z_max == z_min:
        z_max = z_min + 1.0

    rod_colorscale = [
        [0.00, "rgb(10,10,30)"],   [0.15, "rgb(80,0,0)"],
        [0.35, "rgb(180,30,0)"],   [0.55, "rgb(220,100,0)"],
        [0.75, "rgb(255,180,0)"],  [0.90, "rgb(255,240,120)"],
        [1.00, "rgb(255,255,240)"],
    ]

    def make_surface(n):
        C = np.tile(solution[n], (n_theta, 1))
        return go.Surface(
            x=X_cyl.tolist(), y=Y_cyl.tolist(), z=Z_cyl.tolist(),
            surfacecolor=C.tolist(), colorscale=rod_colorscale,
            cmin=z_min, cmax=z_max,
            colorbar=dict(title="Temperature", thickness=18, len=0.75),
            showscale=True,
            lighting=dict(ambient=0.45, diffuse=0.85, specular=0.4, roughness=0.4),
            lightposition=dict(x=200, y=300, z=400),
            hovertemplate="x = %{x:.4f}<br>Temperature = %{surfacecolor:.4f}<extra></extra>",
        )

    max_frames   = min(len(time_values), 60)
    frame_indices = np.linspace(0, len(time_values) - 1, max_frames, dtype=int)
    frames, slider_steps = [], []
    for n in frame_indices:
        name = str(n)
        frames.append(go.Frame(
            data=[make_surface(n)], name=name,
            layout=go.Layout(title=dict(text=f"3D Iron Rod — j={n}, t={time_values[n]:.5f}")),
        ))
        slider_steps.append(dict(
            args=[[name], dict(frame=dict(duration=0, redraw=True), mode="immediate")],
            label=f"t={time_values[n]:.3f}", method="animate",
        ))

    fig = go.Figure(
        data=[make_surface(0)], frames=frames,
        layout=go.Layout(
            title="3D Iron Rod Heat Diffusion (BTCS) — j=0",
            scene=dict(
                xaxis=dict(title="Position x", showgrid=True, gridcolor="#444"),
                yaxis=dict(title="", showticklabels=False, showgrid=False, zeroline=False),
                zaxis=dict(title="", showticklabels=False, showgrid=False, zeroline=False),
                camera=dict(eye=dict(x=1.6, y=1.8, z=0.7)),
                aspectmode="manual", aspectratio=dict(x=3.5, y=1, z=1),
                bgcolor="rgb(15,15,25)",
            ),
            height=460, margin=dict(l=0, r=0, t=60, b=90),
            updatemenus=[dict(
                type="buttons", showactive=False, y=1.12, x=0.5, xanchor="center",
                buttons=[
                    dict(label="▶ Play", method="animate",
                         args=[None, dict(frame=dict(duration=800, redraw=True), fromcurrent=True)]),
                    dict(label="⏸ Pause", method="animate",
                         args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")]),
                ],
            )],
            sliders=[dict(active=0, steps=slider_steps, y=0, len=1.0, pad=dict(t=40))],
        ),
    )
    return fig


def _build_error_figure(time_values, error_analysis):
    if error_analysis is None:
        return None
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=time_values.tolist(), y=error_analysis["max_error_per_step"].tolist(),
        mode="lines+markers", name="Max absolute error",
        line=dict(color="#e63946", width=2), marker=dict(size=4),
        hovertemplate="t=%{x:.5f}<br>max error=%{y:.3e}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=time_values.tolist(), y=error_analysis["l2_error_per_step"].tolist(),
        mode="lines", name="L2 (RMS) error",
        line=dict(color="#457b9d", width=2, dash="dash"),
        hovertemplate="t=%{x:.5f}<br>L2 error=%{y:.3e}<extra></extra>",
    ))
    fig.update_layout(
        title="Error Analysis: BTCS vs Exact Solution",
        xaxis_title="Time t", yaxis_title="Error",
        yaxis_type="log", hovermode="x unified",
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0)"),
    )
    return fig


# ---------------------------------------------------------------------------
# BTCS matrix helpers
# ---------------------------------------------------------------------------

def build_btcs_matrix(number_of_points, r, left_boundary, right_boundary):
    """BTCS implicit system: A U_int^{n+1} = U_int^n + b."""
    interior_count = number_of_points - 2
    matrix = np.zeros((interior_count, interior_count), dtype=float)
    for row in range(interior_count):
        matrix[row, row] = 1 + 2 * r
        if row > 0:
            matrix[row, row - 1] = -r
        if row < interior_count - 1:
            matrix[row, row + 1] = -r
    boundary_vector = np.zeros(interior_count, dtype=float)
    if interior_count > 0:
        boundary_vector[0]  += r * left_boundary
        boundary_vector[-1] += r * right_boundary
    return matrix, boundary_vector


def latex_number(value):
    value = float(value)
    if abs(value) < 1e-12:
        return "0"
    return f"{value:.6g}"


def latex_column_vector(values, max_entries=10):
    values = np.asarray(values, dtype=float).ravel()
    if len(values) == 0:
        return r"\begin{bmatrix}\end{bmatrix}"
    if len(values) > max_entries:
        shown = list(values[:5]) + [None] + list(values[-3:])
    else:
        shown = list(values)
    rows = [r"\vdots" if v is None else latex_number(v) for v in shown]
    return r"\begin{bmatrix}" + r"\\".join(rows) + r"\end{bmatrix}"


def latex_btcs_matrix(matrix, r, max_size=8):
    matrix = np.asarray(matrix, dtype=float)
    rows_count, cols_count = matrix.shape
    if rows_count == 0:
        return r"\begin{bmatrix}\end{bmatrix}"
    if rows_count <= max_size and cols_count <= max_size:
        rows = []
        for row in matrix:
            rows.append(" & ".join(latex_number(v) for v in row))
        return r"\begin{bmatrix}" + r"\\".join(rows) + r"\end{bmatrix}"
    center = latex_number(1 + 2 * r)
    side   = latex_number(-r)
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


# ---------------------------------------------------------------------------
# Interactive dashboard  (mirrors Week 1 structure exactly)
# ---------------------------------------------------------------------------

def build_interactive_dashboard(question, x, time_values, h, k, r,
                                solution, exact, error_analysis, support_results):
    dashboard_path = OUTPUT_DIR / "interactive_output_dashboard.html"

    selected_indices = np.linspace(0, len(time_values) - 1, min(6, len(time_values)))
    selected_indices = sorted(set(int(i) for i in selected_indices))

    # ---- Plotly figures ----
    profile_fig = go.Figure()
    for index in selected_indices:
        profile_fig.add_trace(go.Scatter(
            x=x.tolist(), y=solution[index].tolist(),
            mode="lines+markers", name=f"j={index}, t={time_values[index]:.5f}",
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
                mode="lines", name=f"Exact j={index}",
                line=dict(dash="dash"), showlegend=True,
                hovertemplate="x=%{x:.5f}<br>U_exact=%{y:.5f}<extra></extra>",
            ))
    profile_fig.update_layout(
        title="BTCS Solution Profiles",
        xaxis_title="Position x", yaxis_title="Temperature", hovermode="closest",
    )

    surface_fig = go.Figure(data=[
        go.Surface(
            x=x.tolist(), y=time_values.tolist(), z=solution.tolist(),
            colorscale="Plasma", colorbar=dict(title="Temperature"),
            opacity=0.78,
            hovertemplate="x=%{x:.5f}<br>t=%{y:.5f}<br>U=%{z:.5f}<extra></extra>",
            name="Full space-time surface",
        ),
        go.Scatter3d(
            x=x.tolist(), y=[float(time_values[0])] * len(x), z=solution[0].tolist(),
            mode="lines+markers",
            line=dict(color="#e63946", width=7), marker=dict(size=4, color="#e63946"),
            name="Current time slice",
        ),
    ])
    surface_fig.update_layout(
        title="Rotatable 3D Space-Time Surface (BTCS)",
        scene=dict(xaxis_title="Space x", yaxis_title="Time t", zaxis_title="Temperature"),
        margin=dict(l=0, r=0, t=50, b=0),
    )

    max_time_rows    = min(8, len(time_values))
    max_space_points = min(12, len(x))
    sampled_j = np.linspace(0, len(time_values) - 1, max_time_rows, dtype=int)
    sampled_i = np.linspace(0, len(x) - 1, max_space_points, dtype=int)
    mesh_x, mesh_t, mesh_i_list, mesh_j_list, mesh_labels = [], [], [], [], []
    for j in sampled_j:
        for i in sampled_i:
            mesh_x.append(float(x[i]));       mesh_t.append(float(time_values[j]))
            mesh_i_list.append(int(i));        mesh_j_list.append(int(j))
            mesh_labels.append(f"U_{i}^{j}")
    mesh_fig = go.Figure(data=go.Scatter(
        x=mesh_x, y=mesh_t, mode="markers+text",
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
        title="Interactive Mesh Grid Points (BTCS)",
        xaxis_title="Space x", yaxis_title="Time t",
    )

    rod_fig   = _build_rod_animation_figure(x, time_values, solution)
    error_fig = _build_error_figure(time_values, error_analysis)

    # ---- BTCS matrix ----
    btcs_matrix, boundary_vector = build_btcs_matrix(
        len(x), r, float(question["left_boundary"]), float(question["right_boundary"])
    )
    boundary_is_zero = bool(np.allclose(boundary_vector, 0.0))
    matrix_formula   = (
        r"A\,U_{\mathrm{int}}^{\,n+1}=U_{\mathrm{int}}^{\,n}"
        if boundary_is_zero
        else r"A\,U_{\mathrm{int}}^{\,n+1}=U_{\mathrm{int}}^{\,n}+b"
    )
    matrix_latex  = latex_btcs_matrix(btcs_matrix, r)
    boundary_latex = latex_column_vector(boundary_vector)
    general_matrix_latex = (
        r"\begin{bmatrix}"
        r"1+2r & -r & 0 & \cdots & 0\\"
        r"-r & 1+2r & -r & \ddots & \vdots\\"
        r"0 & -r & 1+2r & \ddots & 0\\"
        r"\vdots & \ddots & \ddots & \ddots & -r\\"
        r"0 & \cdots & 0 & -r & 1+2r"
        r"\end{bmatrix}_{(N-1)\times(N-1)}"
    )
    boundary_matrix_note = (
        " Because the boundary values are zero, there is no extra boundary vector."
        if boundary_is_zero
        else " Because the boundary values are nonzero, the boundary vector \\(b\\) adds their contributions at both ends."
    )

    # Numeric scalar scheme (BTCS)
    numeric_scalar_scheme = (
        rf"{latex_number(-r)}\,U_{{i-1}}^{{n+1}}"
        rf"+{latex_number(1+2*r)}\,U_i^{{n+1}}"
        rf"+{latex_number(-r)}\,U_{{i+1}}^{{n+1}}"
        rf"=U_i^n"
    )

    # Matrix step examples — full side-by-side multiplication
    def _col_vec(rows, max_show=7):
        rows = list(rows)
        if len(rows) <= max_show:
            return r"\begin{bmatrix}" + r"\\".join(rows) + r"\end{bmatrix}"
        return (r"\begin{bmatrix}" + r"\\".join(rows[:3])
                + r"\\ \vdots \\" + r"\\".join(rows[-2:]) + r"\end{bmatrix}")

    matrix_examples_html = ""
    transitions_to_show  = min(2, max(0, len(time_values) - 1))
    M_int = max(0, len(x) - 2)
    for n in range(transitions_to_show):
        current_interior = solution[n, 1:-1]
        next_interior    = solution[n + 1, 1:-1]
        rhs_vec          = current_interior + boundary_vector

        # symbolic unknown labels U_1^{n+1} … U_{M}^{n+1}
        unk_rows = [f"U_{{{i+1}}}^{{{n+1}}}" for i in range(M_int)]
        cur_rows = [latex_number(v) for v in current_interior]
        bnd_rows = [latex_number(v) for v in boundary_vector]
        rhs_rows = [latex_number(v) for v in rhs_vec]
        sol_rows = [latex_number(v) for v in next_interior]

        boundary_block = (
            "" if boundary_is_zero
            else f"\\;+\\;{_col_vec(bnd_rows)}"
        )

        matrix_examples_html += f"""
          <div class="matrix-step">
            <h3>
              Time level \\(n={n}\\) &#8594; \\(n+1={n+1}\\)
              <small style="font-weight:400;color:#64748b;margin-left:8px">
                \\(t={time_values[n]:.5g}\\) &#8594; \\(t={time_values[n+1]:.5g}\\)
              </small>
            </h3>

            <p><strong>Step 1</strong> — Write \\(A\\,U_{{\\mathrm{{int}}}}^{{{n+1}}} = U_{{\\mathrm{{int}}}}^{{{n}}} + b\\) in full:</p>
            <div class="formula" style="overflow-x:auto">
              \\[
                {latex_btcs_matrix(btcs_matrix, r)}
                \\;
                {_col_vec(unk_rows)}
                \\;=\\;
                {_col_vec(cur_rows)}
                {boundary_block}
              \\]
            </div>

            <p><strong>Step 2</strong> — Evaluate the known right-hand side
              \\(\\mathrm{{RHS}} = U_{{\\mathrm{{int}}}}^{{{n}}} + b\\):</p>
            <div class="formula" style="overflow-x:auto">
              \\[
                \\mathrm{{RHS}} = {_col_vec(rhs_rows)}
              \\]
            </div>

            <p><strong>Step 3</strong> — Solve \\(A\\,U_{{\\mathrm{{int}}}}^{{{n+1}}} = \\mathrm{{RHS}}\\)
              via the Thomas (tridiagonal) algorithm — \\(O(N)\\) work:</p>
            <div class="formula" style="overflow-x:auto">
              \\[
                U_{{\\mathrm{{int}}}}^{{{n+1}}}
                = A^{{-1}}\\,\\mathrm{{RHS}}
                = {_col_vec(sol_rows)}
              \\]
            </div>
          </div>
        """

    # Complete table
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

    # Descriptive messages
    initial_peak   = float(np.max(solution[0]))
    final_peak     = float(np.max(solution[-1]))
    physical_message = (
        "The maximum temperature decreased — expected from heat diffusion."
        if final_peak <= initial_peak
        else "The maximum temperature increased — check boundary conditions."
    )
    stability_color   = "#22c55e"
    stability_label   = "STABLE"
    stability_message = "Unconditionally stable — r can be any positive value."

    if error_analysis is None:
        exact_message = "No exact solution check requested."
    else:
        exact_message = (
            f"Max absolute error: {error_analysis['global_max_error']:.6e} &nbsp;|&nbsp; "
            f"L2 error: {error_analysis['global_l2_error']:.6e}"
        )

    _ic_map = {
        "hot_center": (
            f"\\(U_i^0 = {question.get('u_value','?')}\\)"
            f" for \\({question.get('u_left','?')} \\le x_i \\le {question.get('u_right','?')}\\)"
        ),
        "sin_pi":      "\\(U_i^0 = \\sin(\\pi x_i)\\)",
        "custom_list": "custom list of \\(N+1\\) values",
        "function": (
            question.get("initial_condition_display")
            or f"\\(U_i^0 =\\) <code>{question.get('initial_condition_expr','custom function')}</code>"
        ),
    }
    ic_desc_html = _ic_map.get(question.get("initial_condition_type", ""),
                               question.get("initial_condition_type", ""))

    stability_verdict_html = (
        f'<p style="color:#22c55e;font-weight:700">&#10003; r = {r:.4f} &mdash; '
        'BTCS is <strong>unconditionally stable</strong>. '
        'Any r &gt; 0 is valid. Larger r means a larger time step and less temporal '
        'accuracy, but <em>never</em> instability.</p>'
    )

    num_interior_pts     = int(len(x) - 2)
    num_time_steps_total = int(len(time_values) - 1)
    total_unknowns       = num_interior_pts * num_time_steps_total

    _exact_map = {
        "sin_pi_zero_boundary": "\\(u(x,t)=e^{-\\alpha\\pi^2t}\\sin(\\pi x)\\)",
        "function": f"\\(u(x,t)=\\) <code>{question.get('exact_solution_expr','custom')}</code>",
    }
    exact_type_label = _exact_map.get(question.get("exact_solution_type"), "None")

    _err_rows = ""
    if error_analysis:
        _err_rows = (
            f'<tr><td>Global max error \\(\\|e\\|_\\infty\\)</td>'
            f'<td><strong>{error_analysis["global_max_error"]:.6e}</strong></td></tr>'
            f'<tr><td>Global L2 error \\(\\|e\\|_2\\)</td>'
            f'<td><strong>{error_analysis["global_l2_error"]:.6e}</strong></td></tr>'
        )

    _reference  = question.get("reference", "")
    _ref_html   = f'<p><strong>Reference:</strong> <em>{_reference}</em></p>' if _reference else ""
    _q_text_html = question.get("question_text", "")
    _gn_raw      = question.get("graph_notes", {})
    _g_profiles  = _gn_raw.get("profiles", "")
    _g_rod       = _gn_raw.get("rod", "")
    _g_surface   = _gn_raw.get("surface", "")
    _g_rod_div      = f'<div class="graph-note">{_g_rod}</div>'      if _g_rod      else ""
    _g_profiles_div = f'<div class="graph-note">{_g_profiles}</div>' if _g_profiles else ""
    _g_surface_div  = f'<div class="graph-note">{_g_surface}</div>'  if _g_surface  else ""
    exact_profile_note = (
        " Dashed lines show the exact solution — matching them closely means the method is accurate."
        if exact is not None else ""
    )

    # Serialise figures
    figures = {
        "profilePlot":  profile_fig.to_plotly_json(),
        "surfacePlot":  surface_fig.to_plotly_json(),
        "meshPlot":     mesh_fig.to_plotly_json(),
        "rodAnimation": rod_fig.to_plotly_json(),
    }
    has_error_panel = error_fig is not None
    if has_error_panel:
        figures["errorPlot"] = error_fig.to_plotly_json()

    figures_json  = json.dumps(figures, cls=PlotlyJSONEncoder)
    x_json        = json.dumps(x.tolist(), cls=PlotlyJSONEncoder)
    time_json     = json.dumps(time_values.tolist(), cls=PlotlyJSONEncoder)
    solution_json = json.dumps(solution.tolist(), cls=PlotlyJSONEncoder)

    # Error panel
    if has_error_panel:
        error_panel_html = f"""
      <div class="panel lesson" id="secError">
        <div class="section-label">Section 12 &mdash; Accuracy Check</div>
        <h2>Error Analysis: Numerical vs Exact</h2>
        <p>
          At each grid point: \\(e_i^j = |U_i^j - u(x_i,t_j)|\\).
          BTCS at large \\(r\\) is stable but less accurate — temporal truncation
          error grows as \\(O(k) = O(r\\,h^2/\\alpha)\\).
        </p>
        <p>
          Global max error: <strong>{error_analysis['global_max_error']:.6e}</strong>
          &nbsp;|&nbsp;
          Global L2/RMS error: <strong>{error_analysis['global_l2_error']:.6e}</strong>
        </p>
        <div id="errorPlot" class="plot"></div>
      </div>"""
    else:
        error_panel_html = """
      <div class="panel lesson" id="secError">
        <div class="section-label">Section 12 &mdash; Accuracy Check</div>
        <h2>Error Analysis</h2>
        <p>
          No exact solution provided for this question.
          When an exact solution is available, the lab compares each \\(U_i^j\\)
          with \\(u(x_i,t_j)\\) and plots how BTCS temporal error grows with \\(r\\).
        </p>
      </div>"""

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Interactive BTCS Output Dashboard</title>
  <style>
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Arial, sans-serif;
      color: #1f2933;
      background: #f4f6f8;
      overflow-x: hidden;
    }}
    body.dark {{ color: #e5e7eb; background: #111827; }}
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
      font-size: 0.78em; opacity: 0.70; font-style: normal;
      letter-spacing: 0.04em; margin-top: 3px; font-variant: small-caps;
    }}
    .theme-toggle {{
      border: 1px solid rgba(255,255,255,0.55);
      background: rgba(255,255,255,0.12);
      color: white; padding: 9px 13px; border-radius: 6px;
      cursor: pointer; font-weight: 700;
    }}
    main {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(300px, 360px);
      gap: 0; padding: 16px 0 16px 16px; max-width: 100vw; align-items: stretch;
    }}
    .panel {{
      background: white; border: 1px solid #d9dee5; border-radius: 8px;
      padding: 14px; margin-bottom: 16px; min-width: 0; overflow: hidden;
    }}
    body.dark .panel {{ background: #1f2937; border-color: #374151; }}
    .lesson {{
      border-left: 4px solid #2f6fdb;
      background: #f8fbff;
    }}
    body.dark .lesson {{ background: #182235; }}
    .formula {{
      background: #eef2f6; padding: 10px; border-radius: 6px;
      overflow-x: auto; line-height: 1.9;
    }}
    body.dark .formula, body.dark code, body.dark pre {{ background: #111827; }}
    .pde-badge {{
      display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin: 10px 0;
    }}
    .pde-badge-item {{
      background: #f8fafc; border: 1px solid #e2e8f0;
      border-radius: 6px; padding: 10px; text-align: center;
    }}
    body.dark .pde-badge-item {{ background: #1e293b; border-color: #334155; }}
    .pde-badge-item strong {{
      display: block; font-size: .7em; text-transform: uppercase;
      letter-spacing: .05em; color: #64748b; margin-bottom: 4px;
    }}
    .pde-badge-item span {{ font-size: 1.1em; font-weight: 700; }}
    .question-box {{
      background: #f5f0ff; border: 2px solid #7c3aed; border-radius: 8px;
      padding: 18px 22px; margin: 10px 0; line-height: 1.9; font-size: .97em;
    }}
    body.dark .question-box {{ background: #1a1530; border-color: #7c3aed; }}
    .question-box p:first-child {{ margin-top: 0; }}
    .question-box p:last-child  {{ margin-bottom: 0; }}
    .graph-note {{
      background: #fffbeb; border-left: 3px solid #f59e0b;
      padding: 9px 14px; border-radius: 0 6px 6px 0;
      margin: 10px 0; font-size: .9em; line-height: 1.8; color: #44403c;
    }}
    body.dark .graph-note {{ background: #1c1a0e; border-color: #d97706; color: #d6d3d1; }}
    .lecture-note {{
      background: #f8fafc; border: 1px solid #e2e8f0;
      border-left: 4px solid #0ea5e9; border-radius: 7px;
      padding: 12px 14px; margin: 12px 0; line-height: 1.85;
    }}
    body.dark .lecture-note {{ background: #111827; border-color: #334155; }}
    .control-row {{
      display: grid; grid-template-columns: auto auto 1fr auto;
      gap: 12px; align-items: center;
    }}
    .control-button {{
      border: 1px solid #b8c2cc; background: #ffffff; color: #12355b;
      padding: 9px 13px; border-radius: 6px; cursor: pointer;
      font-weight: 700; min-width: 72px;
    }}
    body.dark .control-button {{
      background: #111827; color: #e5e7eb; border-color: #4b5563;
    }}
    input[type="range"] {{ width: 100%; }}
    .plot {{ width: 100%; height: 520px; min-width: 0; }}
    .plot-rod {{ width: 100%; height: 500px; min-width: 0; }}
    .side {{
      position: relative;
      padding: 16px 16px 16px 8px;
      min-width: 200px;
      background: #f0f4fa;
      border-left: 1px solid #d9dee5;
      align-self: stretch;
    }}
    body.dark .side {{
      background: #161d2d;
      border-left-color: #334151;
    }}
    #sideResizeHandle {{
      position: absolute; left: 0; top: 0; width: 7px; height: 100%;
      cursor: col-resize; z-index: 20; border-left: 3px solid transparent;
      transition: border-color .15s; box-sizing: border-box;
    }}
    #sideResizeHandle:hover, #sideResizeHandle.dragging {{ border-left-color: #2f6fdb; }}
    body.dark #sideResizeHandle:hover,
    body.dark #sideResizeHandle.dragging {{ border-left-color: #60a5fa; }}
    #mainResizeHandle {{
      position: absolute; right: -4px; top: 0; width: 7px; height: 100%;
      cursor: col-resize; z-index: 21; border-right: 3px solid transparent;
      transition: border-color .15s; box-sizing: border-box;
    }}
    #mainResizeHandle:hover, #mainResizeHandle.dragging {{ border-right-color: #2f6fdb; }}
    body.dark #mainResizeHandle:hover,
    body.dark #mainResizeHandle.dragging {{ border-right-color: #60a5fa; }}
    main > section {{ position: relative; padding-right: 16px; min-width: 0; }}
    body.fullscreen-left main {{ grid-template-columns: 1fr !important; }}
    body.fullscreen-left .side {{ display: none !important; }}
    code, pre {{
      background: #eef2f6; padding: 2px 4px; border-radius: 4px;
    }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    td {{ border: 1px solid #d1d5db; padding: 12px 14px; text-align: left; overflow-wrap: anywhere; }}
    th {{ background: #eef4fb; border: 1px solid #d1d5db; padding: 12px 14px; text-align: center; font-weight: 700; }}
    body.dark th {{ background: #273449; }}
    body.dark td {{ border-color: #374151; }}
    .table-scroll {{ overflow-x: auto; }}
    .section-label {{
      font-size: .68em; text-transform: uppercase; letter-spacing: .1em;
      color: #64748b; font-weight: 700; border-bottom: 1px solid #e5e7eb;
      padding-bottom: 4px; margin-bottom: 10px;
    }}
    body.dark .section-label {{ color: #9ca3af; border-color: #374151; }}
    .pill {{
      display: inline-block; color: white; border-radius: 999px;
      padding: 1px 8px; font-size: .75em; font-weight: 700;
    }}
    .stencil-wrap {{
      display: flex; flex-direction: column; align-items: center;
      gap: 10px; padding: 16px 0;
    }}
    .stencil-row {{ display: flex; gap: 20px; justify-content: center; align-items: center; }}
    .stencil-node {{
      width: 100px; height: 62px; border-radius: 8px;
      display: flex; flex-direction: column; align-items: center;
      justify-content: center; gap: 1px; border: 2px solid transparent;
    }}
    .stencil-node.known    {{ background: #1d4ed8; color: white; border-color: #93c5fd; }}
    .stencil-node.unknown  {{ background: #dc2626; color: white; border-color: #fca5a5; }}
    .stencil-node.computed {{ background: #15803d; color: white; border-color: #86efac; }}
    .stencil-node.boundary {{ background: #6b7280; color: white; }}
    .stencil-arrows {{ font-size: 1.6em; color: #64748b; text-align: center; letter-spacing: 10px; }}
    .step-box {{
      background: #f0f9ff; border-left: 3px solid #0ea5e9;
      padding: 10px 14px; border-radius: 0 6px 6px 0; margin: 7px 0; font-size: .9em;
    }}
    body.dark .step-box {{ background: #0c1a2e; }}
    .step-label {{
      font-weight: 700; font-size: .7em; text-transform: uppercase;
      letter-spacing: .07em; color: #0369a1; margin-bottom: 4px;
    }}
    body.dark .step-label {{ color: #38bdf8; }}
    .phys-note {{
      font-style: italic; color: #0369a1; font-size: .87em;
      padding: 6px 0 0; border-top: 1px dashed #bae6fd; margin-top: 8px;
    }}
    body.dark .phys-note {{ color: #7dd3fc; border-color: #1e3a5f; }}
    .matrix-step {{
      border: 1px solid #dbeafe; background: #ffffff;
      border-radius: 8px; padding: 12px 14px; margin: 12px 0; overflow-x: auto;
    }}
    body.dark .matrix-step {{ background: #0c1a2e; border-color: #1e3a5f; }}
    .complete-table {{
      min-width: 900px;
      font-family: "Cambria Math", Cambria, "Times New Roman", serif;
      font-size: 12.5px;
    }}
    .complete-table td, .complete-table th {{ text-align: center; white-space: nowrap; }}
    .source-ref {{
      background: #eef4fb; border-left: 3px solid #2f6fdb;
      padding: 8px 12px; border-radius: 0 6px 6px 0;
      font-size: .9em; line-height: 1.7; margin: 8px 0;
    }}
    body.dark .source-ref {{
      background: #1e293b; border-left-color: #3b82f6; color: #e5e7eb;
    }}
    .muted-note {{
      font-size: .88em; color: #64748b;
    }}
    body.dark .muted-note {{
      color: #9ca3af;
    }}
    .dev-avatar {{
      width: 48px; height: 48px; border-radius: 50%;
      background: #12355b; color: white;
      display: flex; align-items: center; justify-content: center;
      font-size: 1.3em; font-weight: 700; flex-shrink: 0;
    }}
    body.dark .dev-avatar {{ background: #2563eb; }}
    .motivation-list {{
      font-size: .9em; line-height: 2; padding-left: 16px; margin: 6px 0;
    }}
    body.dark .motivation-list {{ color: #d1d5db; }}
    .motivation-quote {{
      font-size: .88em; color: #64748b; font-style: italic;
      border-top: 1px dashed #e2e8f0; padding-top: 8px; margin-top: 8px;
    }}
    body.dark .motivation-quote {{
      color: #9ca3af; border-top-color: #374151;
    }}
    @media (max-width: 1000px) {{
      main {{ grid-template-columns: 1fr; padding: 16px; gap: 16px; }}
      .side {{ border-left: none; border-top: 1px solid #d9dee5; padding: 16px; }}
      body.dark .side {{ border-top-color: #334151; }}
    }}
  </style>
  <script>
    window.MathJax = {{
      tex: {{
        inlineMath: [["\\\\(", "\\\\)"], ["$", "$"]],
        displayMath: [["\\\\[", "\\\\]"]]
      }},
      chtml: {{scale: 0.92, displayOverflow: "overflow"}}
    }};
  </script>
  <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>
  <script>{get_plotlyjs()}</script>
</head>
<body>
  <header>
    <div class="header-left">
      <h1 style="margin:0 0 2px">BTCS Interactive Lab</h1>
      <div class="author-line">Implicit Method &nbsp;&middot;&nbsp; Unconditionally Stable &nbsp;&middot;&nbsp; Developed by Samuel Adegboyega</div>
    </div>
    <div class="header-right">
      <label for="speedSelect" style="color:white;font-size:.85em;opacity:.82;font-weight:600;white-space:nowrap">Speed:</label>
      <select id="speedSelect" class="theme-toggle" style="font-weight:normal;cursor:pointer;padding:9px 10px" title="Playback speed for all animations">
        <option value="2500">Slow</option>
        <option value="1200" selected>Medium</option>
        <option value="500">Fast</option>
      </select>
      <button id="fullscreenBtn" class="theme-toggle" type="button" title="Expand left panel to full width">&#x26F6; Full</button>
      <button id="collapsePanel" class="theme-toggle" type="button" title="Collapse / expand side panel">&#8614; Panel</button>
      <button id="themeToggle" class="theme-toggle" type="button">Dark mode</button>
    </div>
  </header>

  <main>
    <section>

      <!-- Textbook question box -->
      <div class="panel lesson" id="secQuestion">
        <div class="section-label">Textbook Question</div>
        <h2 style="margin-top:6px">Problem</h2>
        <div class="question-box">
          {_q_text_html}
        </div>
      </div>

      <!-- Problem statement with pde-badge -->
      <div class="panel lesson" id="secProblem">
        <div class="section-label">Section 1 &mdash; Problem Parameters</div>
        <h2 style="margin-top:6px">Numerical Parameters</h2>
        <div class="pde-badge">
          <div class="pde-badge-item"><strong>Diffusivity &alpha;</strong><span>{float(question["alpha"])}</span></div>
          <div class="pde-badge-item"><strong>Domain [a,&nbsp;b]</strong><span>[{float(question["x_start"])},&nbsp;{float(question["x_end"])}]</span></div>
          <div class="pde-badge-item"><strong>Final Time T</strong><span>{float(question["final_time"])}</span></div>
          <div class="pde-badge-item"><strong>Grid Points N+1</strong><span>{int(question["number_of_space_points"])}</span></div>
        </div>
        <p>
          <strong>Boundary conditions:</strong>
          \\( U(x_0,t) = {float(question["left_boundary"])} \\) &nbsp;and&nbsp;
          \\( U(x_N,t) = {float(question["right_boundary"])} \\) for all \\( t \\ge 0 \\)
        </p>
        <p><strong>Initial condition (t=0):</strong> {ic_desc_html}</p>
        <p><strong>Exact solution available:</strong> {exact_type_label}</p>
      </div>

      <!-- Stability analysis -->
      <div class="panel lesson" id="secStability">
        <div class="section-label">Section 8 &mdash; Stability Analysis</div>
        <h2 style="margin-top:6px">BTCS Stability Analysis</h2>
        <p>
          BTCS is an <strong>implicit</strong> method. At each time step the unknown
          future values appear simultaneously in a linear system. Because of this,
          BTCS is <em>unconditionally stable</em> &mdash; the mesh ratio \\(r\\) can be
          any positive value:
          \\[
            r = \\frac{{\\alpha k}}{{h^2}}, \\qquad r > 0 \\text{{ (no upper limit)}}.
          \\]
          Compare this with FTCS (Week 1), which required \\(r \\le 0.5\\).
        </p>
        <div class="pde-badge">
          <div class="pde-badge-item"><strong>Your r</strong><span>{r:.6g}</span></div>
          <div class="pde-badge-item"><strong>FTCS Limit</strong><span>0.5</span></div>
          <div class="pde-badge-item"><strong>Status</strong>
            <span style="color:#22c55e">STABLE</span></div>
          <div class="pde-badge-item"><strong>r / 0.5</strong><span>{r/0.5:.3f}&times;</span></div>
        </div>
        <p>
          Large \\(r\\) means a large time step \\(k\\). The solution remains bounded but
          loses temporal accuracy — the error grows as \\(O(k) = O(r\\,h^2/\\alpha)\\).
          Choose \\(r \\le 1\\) for good accuracy; use larger \\(r\\) when speed matters
          more than precision.
        </p>
        {stability_verdict_html}
      </div>

      <!-- Rod animation -->
      <div class="panel lesson" id="secRod">
        <div class="section-label">Section 4 &mdash; Rod Visualization</div>
        <h2>Iron Rod Heat Diffusion (BTCS)</h2>
        <p>
          A one-dimensional heat problem can be viewed as heat moving along a thin rod.
          Colour encodes temperature on a blackbody scale: dark/black = cold,
          red = warm, orange = hot, white/yellow = very hot.
        </p>
        <p>
          Press <strong>&#9654; Play</strong> to watch heat spread step by step.
          Drag the slider to any time level. Each frame is one BTCS solve.
        </p>
        <div class="lecture-note">
          BTCS allows larger time steps than FTCS — each frame may represent a bigger
          physical time jump, but the temperature always remains physically realistic
          (no blow-up or oscillations).
        </div>
        {_g_rod_div}
        <div id="rodAnimation" class="plot-rod"></div>
      </div>

      <!-- Matrix method section -->
      <div class="panel lesson" id="secManual">
        <div class="section-label">Section 7 &mdash; Matrix Generalisation</div>
        <h2>Tridiagonal System: \\(A\\,U^{{n+1}} = U^n + b\\)</h2>
        <p>
          The BTCS formula couples three <em>unknown</em> future values at level \\(n+1\\):
          \\[
            -r\\,U_{{i-1}}^{{n+1}}
            +
            (1+2r)\\,U_i^{{n+1}}
            -
            r\\,U_{{i+1}}^{{n+1}}
            = U_i^n.
          \\]
          For this question, substituting \\(r = {r:.6g}\\), the equation becomes:
          \\[
            {numeric_scalar_scheme}.
          \\]
          Collecting all interior equations into one vector gives the linear system:
          \\[
            U_{{\\mathrm{{int}}}}^{{n+1}} =
            \\begin{{bmatrix}}
              U_1^{{n+1}} & U_2^{{n+1}} & \\cdots & U_{{N-1}}^{{n+1}}
            \\end{{bmatrix}}^T.
          \\]
          One tridiagonal solve then updates the entire rod interior from level \\(n\\)
          to level \\(n+1\\).
        </p>
        <div class="formula">
          \\[ {matrix_formula} \\]
          \\[ A = {matrix_latex} \\]
          {"" if boundary_is_zero else f"\\[ b = {boundary_latex} \\]"}
        </div>
        <div class="lecture-note">
          The matrix \\(A\\) is tridiagonal with diagonal \\(1+2r\\) and
          off-diagonals \\(-r\\). It is <em>strictly diagonally dominant</em>
          for all \\(r > 0\\) (diagonal exceeds sum of off-diagonals), guaranteeing
          a unique solution at every step &mdash; the mathematical source of BTCS
          unconditional stability.
          {boundary_matrix_note}
        </div>
        {matrix_examples_html}
        <p>{stability_message}</p>
        <p>{physical_message}</p>
      </div>

      <!-- Complete solution table -->
      <div class="panel lesson" id="secTable">
        <div class="section-label">Section 8 &mdash; Complete Numerical Table</div>
        <h2>Complete BTCS Solution Table</h2>
        <p>
          Each row is one time level \\(j\\). Each column is one spatial grid point \\(i\\).
          Reading across shows the temperature profile at one moment;
          reading down shows how one fixed point evolves over time.
        </p>
        <div class="table-scroll">
          <table class="complete-table">
            {complete_header}
            {complete_table_rows}
          </table>
        </div>
      </div>

      <!-- Time player -->
      <div class="panel lesson" id="secPlayer">
        <div class="section-label">Section 9 &mdash; Temperature Evolution Simulation</div>
        <h2>Time Evolution Player</h2>
        <p>
          Move the slider or press Play. The 2D graph and the red 3D time slice
          update together. This shows \\(U_i^j\\): for one selected \\(j\\), all values
          \\(U_0^j,\\,U_1^j,\\,\\ldots,\\,U_N^j\\) are drawn along the rod.
        </p>
        <div class="control-row">
          <button id="playButton" class="control-button" type="button">&#9654; Play</button>
          <button id="pauseButton" class="control-button" type="button">&#9646;&#9646; Pause</button>
          <input id="timeSlider" type="range" min="0" max="{len(time_values)-1}" step="1" value="0">
          <strong id="timeReadout">j = 0</strong>
        </div>
        <div id="controlledTimePlot" class="plot"></div>
        <p id="timeExplanation">At j = 0, this is the initial temperature profile before diffusion starts.</p>
      </div>

      <!-- Solution profiles -->
      <div class="panel" id="secProfiles">
        <div class="section-label">Section 9b &mdash; Solution Profiles</div>
        <h2>Solution Profiles</h2>
        <p>
          Several time levels overlaid on the same axes. Earlier curves show the
          starting shape; later curves show how diffusion smooths it out.
          {exact_profile_note}
        </p>
        {_g_profiles_div}
        <div id="profilePlot" class="plot"></div>
      </div>

      <!-- 3D surface -->
      <div class="panel" id="secSurface">
        <div class="section-label">Section 10 &mdash; 3D Surface Evolution</div>
        <h2>Rotatable 3D Surface</h2>
        <p>
          Drag to rotate. Horizontal axis = space, depth = time, height = temperature.
          The red curve tracks the current time level from the player above.
        </p>
        {_g_surface_div}
        <div id="surfacePlot" class="plot"></div>
      </div>

      <!-- Mesh grid -->
      <div class="panel" id="secMesh">
        <div class="section-label">Section 5 &mdash; Mesh / Grid Visualization</div>
        <h2>Mesh Grid Points</h2>
        <p>
          Each dot is one location \\(U_i^j\\).
          <strong>Click any interior node</strong> to verify the BTCS equation
          at that point in the stencil inspector below.
        </p>
        <div class="lecture-note">
          In BTCS all interior unknowns at level \\(j+1\\) are solved simultaneously,
          unlike FTCS where each is computed from three known past values independently.
        </div>
        <div id="meshPlot" class="plot"></div>
      </div>

      <!-- BTCS stencil inspector -->
      <div class="panel lesson" id="secStencil">
        <div class="section-label">Section 6 &mdash; BTCS Stencil &amp; Calculation Inspector</div>
        <h2 style="margin-top:6px">BTCS Step-by-Step Calculation</h2>
        <p>
          The BTCS stencil uses
          <span class="pill" style="background:#1d4ed8">1 known node</span>
          at time level \\(j\\) as the right-hand side, and
          <span class="pill" style="background:#dc2626">3 unknown nodes</span>
          at level \\(j+1\\) appear in the same equation:
          \\[
            -r\\,U_{{i-1}}^{{j+1}} + (1+2r)\\,U_i^{{j+1}} - r\\,U_{{i+1}}^{{j+1}} = U_i^j
          \\]
          After BTCS is solved, substituting computed values verifies the equation.
          Click any <strong>interior</strong> mesh node above.
        </p>
        <div id="stencilContent">
          <div style="text-align:center;color:#94a3b8;padding:48px 0;font-style:italic;font-size:1.05em">
            &#8593; Click an interior mesh node above to verify the BTCS equation here.
          </div>
        </div>
      </div>

      {error_panel_html}

      <!-- Numerical summary -->
      <div class="panel" id="secSummary">
        <div class="section-label">Section 11 &mdash; Numerical Summary</div>
        <h2 style="margin-top:6px">Numerical Summary</h2>
        <div class="table-scroll">
          <table>
            <tr><th>Quantity</th><th>Symbol</th><th>Value</th></tr>
            <tr><td>Space step</td><td>\\(h\\)</td><td><strong>{h:.8g}</strong></td></tr>
            <tr><td>Time step</td><td>\\(k\\)</td><td><strong>{k:.8g}</strong></td></tr>
            <tr><td>Mesh ratio</td><td>\\(r=\\alpha k/h^2\\)</td>
                <td><strong>{r:.8g}</strong> &nbsp;
                  <span class="pill" style="background:#22c55e">STABLE</span>
                </td></tr>
            <tr><td>Interior spatial points</td><td>\\(N-1\\)</td><td><strong>{num_interior_pts}</strong></td></tr>
            <tr><td>Time steps computed</td><td>\\(M\\)</td><td><strong>{num_time_steps_total}</strong></td></tr>
            <tr><td>Total interior unknowns</td><td>\\((N-1)\\cdot M\\)</td>
                <td><strong>{total_unknowns:,}</strong></td></tr>
            <tr><td>Exact solution</td><td>&mdash;</td><td>{exact_type_label}</td></tr>
            {_err_rows}
          </table>
        </div>
      </div>

      <div id="mainResizeHandle" title="Drag to resize left panel &middot; Double-click to reset"></div>
    </section>

    <aside class="side">
      <div id="sideResizeHandle" title="Drag to resize &middot; Double-click to reset"></div>

      <!-- About This Lab -->
      <div class="panel lesson">
        <h2>About This Lab</h2>
        <p>
          All 11 questions in this lab are adapted from
          <strong>Chapter&nbsp;11</strong> of:
        </p>
        <div class="source-ref">
          <em>Numerical Methods for Scientific and Engineering Computation</em><br>
          <strong>Jain, Iyengar &amp; Jain</strong> — 6th Edition<br>
          Sections 11.9&ndash;11.12 (BTCS scheme)
        </div>
        <p class="muted-note">
          Problems appear as-is or lightly adapted to vary \\(r\\), domain, or
          initial condition so each question highlights a different aspect of
          the implicit BTCS method.
        </p>
      </div>

      <!-- Developer -->
      <div class="panel lesson">
        <h2>Developer</h2>
        <div style="display:flex;align-items:center;gap:12px;margin-bottom:10px">
          <div class="dev-avatar">SA</div>
          <div>
            <strong style="font-size:1.05em">Samuel Adegboyega</strong><br>
            <span class="muted-note">University of Lagos (UNILAG), Nigeria</span>
          </div>
        </div>
        <p style="font-size:.9em;line-height:1.8">
          Samuel is a student of <strong>Numerical Methods for Partial Differential
          Equations</strong> at UNILAG. His coursework covers finite-difference schemes
          (FTCS, BTCS, Crank&ndash;Nicolson), stability analysis, and their physical
          interpretations in heat conduction and fluid dynamics.
        </p>
      </div>

      <!-- Motivation -->
      <div class="panel lesson">
        <h2>Why This Tool?</h2>
        <p style="font-size:.9em;line-height:1.85">
          Textbook tables of numbers are hard to interpret. This lab was built
          to make the BTCS implicit scheme <em>tangible</em>:
        </p>
        <ul class="motivation-list">
          <li>The <strong>3D rod animation</strong> turns numbers into heat flow you can watch.</li>
          <li>The <strong>stencil inspector</strong> lets you click any node and verify the BTCS equation residual to machine precision.</li>
          <li>The <strong>matrix step section</strong> shows the full \\(A\\,U^{{n+1}}=U^n+b\\) solve at each time level, side by side.</li>
          <li>The <strong>stability badge</strong> compares your \\(r\\) against the FTCS limit &mdash; confirming BTCS needs no restriction.</li>
        </ul>
        <p class="motivation-quote">
          &ldquo;The best way to understand a numerical method is to watch it
          run on real numbers, one step at a time.&rdquo;
        </p>
      </div>

      <div class="panel lesson">
        <h2>What This PDE Models</h2>
        <p>
          The <strong>heat equation</strong> \\(u_t = \\alpha\\,u_{{xx}}\\) says:
          the rate of temperature change at a point equals \\(\\alpha\\) times
          how sharply the profile curves there.
        </p>
        <ul style="margin:8px 0;padding-left:18px;line-height:2">
          <li><strong>Hot peak</strong> &rarr; curves downward &rarr; temperature <em>falls</em>.</li>
          <li><strong>Cold valley</strong> &rarr; curves upward &rarr; temperature <em>rises</em>.</li>
        </ul>
        <p>Result: every sharp feature smooths out; the rod approaches a steady-state profile.</p>
        <p>{exact_message}</p>
      </div>

      <div class="panel lesson">
        <h2>Notation Link</h2>
        <p>
          The continuous solution \\(u(x,t)\\) is sampled on a grid:
          \\[x_i = ih,\\qquad t_j = jk.\\]
        </p>
        <div class="formula">
          \\[u(x_i,t_j)\\approx U_i^j.\\]
        </div>
        <p>
          The interior unknowns at level \\(n+1\\) form a column vector:
          \\[U_{{\\mathrm{{int}}}}^{{n+1}}=
          \\begin{{bmatrix}}U_1^{{n+1}}&\\cdots&U_{{N-1}}^{{n+1}}\\end{{bmatrix}}^T.\\]
        </p>
      </div>

      <div class="panel lesson">
        <h2>General BTCS Matrix \\(A\\)</h2>
        <p>
          For \\(N-1\\) interior unknowns, the system matrix is:
        </p>
        <div class="formula">
          \\[A = {general_matrix_latex}\\]
        </div>
        <p>
          Diagonal \\(1+2r\\) exceeds \\(2r\\) (sum of off-diagonals) for all \\(r>0\\),
          so \\(A\\) is always non-singular &mdash; BTCS is unconditionally stable.
        </p>
      </div>

      <div class="panel lesson">
        <h2>Key Terms</h2>
        <dl style="margin:0;line-height:1.9;font-size:.92em">
          <dt style="font-weight:700">\\(\\alpha\\) &mdash; Thermal diffusivity</dt>
          <dd style="margin:0 0 8px 14px">
            How fast heat spreads. High \\(\\alpha\\) (copper) &rarr; fast;
            low \\(\\alpha\\) (concrete) &rarr; slow.
            <br><strong>This problem:</strong> \\(\\alpha = {float(question["alpha"]):.6g}\\)
          </dd>
          <dt style="font-weight:700">\\(h\\) &mdash; Space step</dt>
          <dd style="margin:0 0 8px 14px">
            Gap between neighbouring grid points.
            <br><strong>This problem:</strong> \\(h = {h:.6g}\\)
          </dd>
          <dt style="font-weight:700">\\(k\\) &mdash; Time step</dt>
          <dd style="margin:0 0 8px 14px">
            How far BTCS advances per tridiagonal solve.
            <br><strong>This problem:</strong> \\(k = {k:.6g}\\)
          </dd>
          <dt style="font-weight:700">\\(r = \\alpha k / h^2\\) &mdash; Mesh ratio</dt>
          <dd style="margin:0 0 8px 14px">
            Controls accuracy. Any \\(r > 0\\) is stable for BTCS.
            <br><strong>This problem:</strong>
            \\(r = {r:.6g}\\)&nbsp;
            <span class="pill" style="background:#22c55e">STABLE</span>
          </dd>
          <dt style="font-weight:700">Boundary conditions</dt>
          <dd style="margin:0 0 8px 14px">
            Fixed temperatures at both ends.
            Left \\(= {float(question["left_boundary"]):.4g}\\),
            right \\(= {float(question["right_boundary"]):.4g}\\).
          </dd>
          <dt style="font-weight:700">BTCS</dt>
          <dd style="margin:0 0 0 14px">
            Backward-Time Centered-Space — the implicit equation:
            \\[-r\\,U_{{i-1}}^{{j+1}}+(1+2r)\\,U_i^{{j+1}}-r\\,U_{{i+1}}^{{j+1}}=U_i^j.\\]
            Solved as a tridiagonal system at each time level.
          </dd>
        </dl>
      </div>

      <div class="panel lesson" id="inspectorPanel">
        <div class="section-label">Selected Node &mdash; Inspector</div>
        <h2 style="margin-top:6px">Point Details</h2>
        <div id="inspector" style="font-size:.88em;line-height:1.85;color:#475569">
          <p style="color:#94a3b8;font-style:italic">
            Click any point on any chart to inspect it. Interior mesh nodes
            show the BTCS equation verification in the Stencil panel.
          </p>
        </div>
      </div>
    </aside>
  </main>

  <script>
    const figures        = {figures_json};
    const xValues        = {x_json};
    const timeValues     = {time_json};
    const solutionValues = {solution_json};
    const rValue         = {r};
    const hasErrorPlot   = {"true" if has_error_panel else "false"};
    const config = {{responsive: true, displaylogo: false}};

    let currentTimeIndex = 0;
    let playTimer = null;
    let timePlotInitialized = false;

    async function initDashboard() {{
      for (const [id, fig] of Object.entries(figures)) {{
        await Plotly.newPlot(id, fig.data, fig.layout, config);
        if (fig.frames && fig.frames.length) {{
          await Plotly.addFrames(id, fig.frames);
        }}
      }}
      await drawControlledTimePlot(0);
      bindPlotlyClicks();
      if (window.MathJax && window.MathJax.typesetPromise) {{
        window.MathJax.typesetPromise();
      }}
    }}

    initDashboard();

    // ---- Dark mode ----
    const LIGHT_BG = "#ffffff", LIGHT_PAPER = "#ffffff";
    const LIGHT_FONT = "#1f2933", LIGHT_GRID = "#e5e7eb";
    const DARK_BG = "#111827", DARK_PAPER = "#1f2937";
    const DARK_FONT = "#e5e7eb", DARK_GRID = "#374151";

    function allPlotIds() {{
      const ids = ["profilePlot","surfacePlot","meshPlot","rodAnimation","controlledTimePlot"];
      if (hasErrorPlot) ids.push("errorPlot");
      return ids;
    }}

    function applyThemeToPlots(dark) {{
      const bg = dark ? DARK_BG : LIGHT_BG, paper = dark ? DARK_PAPER : LIGHT_PAPER;
      const font = dark ? DARK_FONT : LIGHT_FONT, grid = dark ? DARK_GRID : LIGHT_GRID;
      const upd = {{ paper_bgcolor: paper, plot_bgcolor: bg, font: {{color: font}},
        "xaxis.gridcolor": grid, "yaxis.gridcolor": grid,
        "xaxis.linecolor": grid, "yaxis.linecolor": grid }};
      for (const id of allPlotIds()) {{
        const el = document.getElementById(id);
        if (el && el.data) Plotly.relayout(id, upd);
      }}
      if (document.getElementById("surfacePlot")?.data) {{
        Plotly.relayout("surfacePlot", {{
          paper_bgcolor: paper, "scene.bgcolor": bg,
          "scene.xaxis.gridcolor": grid, "scene.yaxis.gridcolor": grid, "scene.zaxis.gridcolor": grid,
        }});
      }}
    }}

    document.getElementById("themeToggle").addEventListener("click", () => {{
      const isDark = document.body.classList.toggle("dark");
      document.getElementById("themeToggle").textContent = isDark ? "Light mode" : "Dark mode";
      applyThemeToPlots(isDark);
    }});

    // ---- Global speed control ----
    document.getElementById("speedSelect").addEventListener("change", (e) => {{
      const newDuration = Number(e.target.value);
      const rodDiv = document.getElementById("rodAnimation");
      if (rodDiv && rodDiv.layout && rodDiv.layout.updatemenus && rodDiv.layout.updatemenus.length) {{
        Plotly.relayout("rodAnimation", {{
          "updatemenus[0].buttons[0].args[1].frame.duration": newDuration
        }}).catch(() => {{}});
      }}
      if (playTimer !== null) {{
        clearInterval(playTimer);
        playTimer = setInterval(() => {{
          const next = currentTimeIndex >= solutionValues.length - 1 ? 0 : currentTimeIndex + 1;
          setTimeIndex(next);
        }}, newDuration);
      }}
    }});

    // ---- Shared resize logic (side panel handle + main section handle) ----
    function makeResizeHandle(handleId, getW, setW, minW, maxW, resetCols) {{
      const handle = document.getElementById(handleId);
      const mainEl = document.querySelector("main");
      if (!handle) return;
      let dragging = false, startX = 0, startW = 0;
      handle.addEventListener("mousedown", (e) => {{
        e.preventDefault(); dragging = true; startX = e.clientX; startW = getW();
        handle.classList.add("dragging");
        document.body.style.cursor = "col-resize"; document.body.style.userSelect = "none";
      }});
      document.addEventListener("mousemove", (e) => {{
        if (!dragging) return;
        const newW = Math.max(minW, Math.min(maxW, setW(startW, e.clientX - startX)));
        mainEl.style.gridTemplateColumns = `minmax(0,1fr) ${{newW}}px`;
      }});
      document.addEventListener("mouseup", () => {{
        if (!dragging) return;
        dragging = false; handle.classList.remove("dragging");
        document.body.style.cursor = ""; document.body.style.userSelect = "";
      }});
      handle.addEventListener("dblclick", () => {{ mainEl.style.gridTemplateColumns = resetCols || ""; }});
    }}

    const sideEl = document.querySelector(".side");
    const mainEl = document.querySelector("main");

    // Right panel handle — drag left to widen side panel
    makeResizeHandle(
      "sideResizeHandle",
      () => sideEl.getBoundingClientRect().width,
      (startW, dx) => startW - dx,
      200, 700, ""
    );

    // Left section handle — drag right to widen main section (= shrink side panel)
    makeResizeHandle(
      "mainResizeHandle",
      () => sideEl.getBoundingClientRect().width,
      (startW, dx) => startW - dx,
      200, 700, ""
    );

    // ---- Side panel collapse / expand ----
    (function () {{
      const btn = document.getElementById("collapsePanel");
      let collapsed = false, savedCols = null;
      btn.addEventListener("click", () => {{
        if (collapsed) {{
          mainEl.style.gridTemplateColumns = savedCols || "";
          sideEl.style.display = ""; collapsed = false;
          btn.innerHTML = "&#8614;&nbsp;Panel"; btn.title = "Collapse side panel";
        }} else {{
          savedCols = mainEl.style.gridTemplateColumns || null;
          mainEl.style.gridTemplateColumns = "minmax(0,1fr) 0px";
          sideEl.style.display = "none"; collapsed = true;
          btn.innerHTML = "&#8612;&nbsp;Panel"; btn.title = "Expand side panel";
        }}
      }});
    }})();

    // ---- Fullscreen — left panel takes 100 % ----
    (function () {{
      const btn = document.getElementById("fullscreenBtn");
      let full = false;
      btn.addEventListener("click", () => {{
        full = !full;
        document.body.classList.toggle("fullscreen-left", full);
        btn.textContent = full ? "&#x26F6; Exit Full" : "&#x26F6; Full";
        btn.innerHTML   = full ? "&#x26F6; Exit Full" : "&#x26F6; Full";
        btn.title = full ? "Exit fullscreen" : "Expand left panel to full width";
      }});
    }})();

    // ---- Time player ----
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
      const layout = Object.assign({{}}, timeSliceLayout(j), {{
        paper_bgcolor: isDark ? DARK_PAPER : LIGHT_PAPER,
        plot_bgcolor:  isDark ? DARK_BG    : LIGHT_BG,
        font: {{color: isDark ? DARK_FONT : LIGHT_FONT}},
      }});
      const trace = {{
        x: xValues, y: solutionValues[j],
        mode: "lines+markers",
        line: {{width: 3, color: "#2f6fdb"}},
        marker: {{size: 8}},
        customdata: xValues.map((_, i) => [i, j, timeValues[j]]),
        hovertemplate:
          "i=%{{customdata[0]}}<br>j=%{{customdata[1]}}<br>" +
          "x=%{{x:.5f}}<br>t=%{{customdata[2]:.5f}}<br>U=%{{y:.5f}}<extra></extra>"
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
      const maxVal = Math.max(...vals), minVal = Math.min(...vals);
      if (j === 0) return "At j = 0, this is the initial temperature profile before diffusion starts.";
      return `At j=${{j}}: BTCS solved one tridiagonal system (r=${{rValue.toFixed(4)}}). ` +
             `min=${{minVal.toFixed(4)}}, max=${{maxVal.toFixed(4)}}.`;
    }}

    // ---- DOM event listeners ----
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
      clearInterval(playTimer); playTimer = null;
    }});

    // ---- Inspector ----
    function showPoint(source, pt) {{
      const inspEl = document.getElementById("inspector");
      let rows = [], nodeI = null, nodeJ = null;
      if (source === "controlledTimePlot" && pt.customdata) {{
        nodeI = pt.customdata[0]; nodeJ = pt.customdata[1];
        rows.push(`<strong>U<sub>${{nodeI}}</sub><sup>${{nodeJ}}</sup></strong> = ${{Number(pt.y).toFixed(6)}}`);
        rows.push(`Space index &nbsp;<strong>i = ${{nodeI}}</strong> &ensp; x = ${{Number(pt.x).toFixed(5)}}`);
        rows.push(`Time index &nbsp;<strong>j = ${{nodeJ}}</strong> &ensp; t = ${{Number(pt.customdata[2]).toFixed(5)}}`);
      }} else if (source === "profilePlot" && pt.customdata) {{
        nodeJ = pt.customdata[0]; nodeI = pt.customdata[2];
        rows.push(`<strong>U<sub>${{nodeI}}</sub><sup>${{nodeJ}}</sup></strong> = ${{Number(pt.y).toFixed(6)}}`);
        rows.push(`Space index &nbsp;<strong>i = ${{nodeI}}</strong> &ensp; x = ${{Number(pt.x).toFixed(5)}}`);
        rows.push(`Time index &nbsp;<strong>j = ${{nodeJ}}</strong> &ensp; t = ${{Number(pt.customdata[1]).toFixed(5)}}`);
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
      }}
      inspEl.innerHTML = rows.map(r => `<div style="padding:2px 0;border-bottom:1px solid rgba(148,163,184,.18)">${{r}}</div>`).join("");
      if (nodeI !== null && nodeJ !== null) {{
        updateStencilPanel(nodeI, nodeJ);
        const sp = document.getElementById("secStencil");
        if (sp) sp.scrollIntoView({{behavior:"smooth",block:"nearest"}});
      }}
    }}

    // ---- BTCS stencil verification ----
    function updateStencilPanel(i, j) {{
      const panel = document.getElementById("stencilContent");
      if (!panel) return;
      const n = xValues.length;
      if (j === 0) {{
        panel.innerHTML = `<div class="step-box"><div class="step-label">Initial Condition — j = 0</div>
          U<sub>${{i}}</sub><sup>0</sup> = <strong>${{solutionValues[0][i].toFixed(6)}}</strong><br>
          This value comes from the initial condition. BTCS has not yet been applied.</div>`;
        return;
      }}
      if (i === 0 || i >= n - 1) {{
        panel.innerHTML = `<div class="step-box"><div class="step-label">Boundary Point</div>
          U<sub>${{i}}</sub><sup>${{j}}</sup> = <strong>${{solutionValues[j][i].toFixed(6)}}</strong><br>
          This is a fixed boundary value — not part of the BTCS linear system.</div>`;
        return;
      }}
      const uL = solutionValues[j][i-1], uC = solutionValues[j][i], uR = solutionValues[j][i+1];
      const uPrev = solutionValues[j-1][i];
      const r = rValue;
      const lhs = -r*uL + (1+2*r)*uC - r*uR;
      const resid = Math.abs(lhs - uPrev);
      const diff = uC - uPrev;
      const physDir = diff > 1e-9 ? "increased &mdash; net heat gained"
                    : diff < -1e-9 ? "decreased &mdash; net heat lost"
                    : "unchanged &mdash; balanced heat flow";
      panel.innerHTML = `
        <div class="stencil-wrap">
          <div style="font-size:.78em;color:#64748b;font-weight:600;letter-spacing:.05em;text-transform:uppercase">
            Known RHS &mdash; time level j&minus;1 = ${{j-1}} &nbsp;(t = ${{timeValues[j-1].toFixed(5)}})
          </div>
          <div class="stencil-row">
            <div class="stencil-node known">
              <span style="font-size:.68em;opacity:.85">U<sub>${{i}}</sub><sup>${{j-1}}</sup></span>
              <strong>${{uPrev.toFixed(5)}}</strong>
              <span style="font-size:.62em;opacity:.75">RHS (known)</span>
            </div>
          </div>
          <div class="stencil-arrows">&#8595; tridiagonal solve &#8595;</div>
          <div style="font-size:.78em;color:#64748b;font-weight:600;letter-spacing:.05em;text-transform:uppercase">
            Solved unknowns &mdash; time level j = ${{j}} &nbsp;(t = ${{timeValues[j].toFixed(5)}})
          </div>
          <div class="stencil-row">
            <div class="stencil-node unknown">
              <span style="font-size:.68em;opacity:.85">U<sub>${{i-1}}</sub><sup>${{j}}</sup></span>
              <strong>${{uL.toFixed(5)}}</strong>
              <span style="font-size:.62em;opacity:.75">left (solved)</span>
            </div>
            <div class="stencil-node computed">
              <span style="font-size:.68em;opacity:.85">U<sub>${{i}}</sub><sup>${{j}}</sup></span>
              <strong>${{uC.toFixed(5)}}</strong>
              <span style="font-size:.62em;opacity:.75">centre (solved)</span>
            </div>
            <div class="stencil-node unknown">
              <span style="font-size:.68em;opacity:.85">U<sub>${{i+1}}</sub><sup>${{j}}</sup></span>
              <strong>${{uR.toFixed(5)}}</strong>
              <span style="font-size:.62em;opacity:.75">right (solved)</span>
            </div>
          </div>
        </div>
        <div class="step-box">
          <div class="step-label">Step 1 &mdash; BTCS Symbolic Equation</div>
          &minus;r&thinsp;U<sub>${{i-1}}</sub><sup>${{j}}</sup>
          + (1+2r)&thinsp;U<sub>${{i}}</sub><sup>${{j}}</sup>
          &minus; r&thinsp;U<sub>${{i+1}}</sub><sup>${{j}}</sup>
          = U<sub>${{i}}</sub><sup>${{j-1}}</sup>
        </div>
        <div class="step-box">
          <div class="step-label">Step 2 &mdash; Substitute r = ${{r.toFixed(4)}}, 1+2r = ${{(1+2*r).toFixed(4)}}</div>
          &minus;${{r.toFixed(4)}}&thinsp;U<sub>${{i-1}}</sub><sup>${{j}}</sup>
          + ${{(1+2*r).toFixed(4)}}&thinsp;U<sub>${{i}}</sub><sup>${{j}}</sup>
          &minus; ${{r.toFixed(4)}}&thinsp;U<sub>${{i+1}}</sub><sup>${{j}}</sup>
          = U<sub>${{i}}</sub><sup>${{j-1}}</sup>
        </div>
        <div class="step-box">
          <div class="step-label">Step 3 &mdash; Substitute computed values</div>
          &minus;${{r.toFixed(4)}}&times;(${{uL.toFixed(5)}})
          + ${{(1+2*r).toFixed(4)}}&times;(${{uC.toFixed(5)}})
          &minus; ${{r.toFixed(4)}}&times;(${{uR.toFixed(5)}})
        </div>
        <div class="step-box">
          <div class="step-label">Step 4 &mdash; Evaluate left-hand side</div>
          LHS = ${{lhs.toFixed(8)}} &nbsp; RHS = ${{uPrev.toFixed(8)}}
        </div>
        <div class="step-box" style="border-left-color:#22c55e;background:#f0fdf4">
          <div class="step-label" style="color:#0369a1">&#10003; Residual</div>
          |LHS &minus; RHS| = <strong>${{resid.toExponential(3)}}</strong>
          &nbsp;<em>(near machine zero &asymp; 10&minus;14 confirms correct solve)</em>
        </div>
        <p class="phys-note">
          &#9728; <strong>Physical meaning:</strong> Temperature at x&nbsp;=&nbsp;${{xValues[i].toFixed(4)}}
          has ${{physDir}} from t&nbsp;=&nbsp;${{timeValues[j-1].toFixed(5)}}
          to t&nbsp;=&nbsp;${{timeValues[j].toFixed(5)}}.
        </p>`;
    }}

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
    """Run the complete BTCS practice-question workflow."""
    validate_question(QUESTION)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    x, time_values, h, k, r = build_mesh(QUESTION)
    solution       = solve_with_btcs(QUESTION, x, time_values, r)
    exact          = exact_solution(QUESTION, x, time_values)
    error_analysis = compute_error_analysis(solution, exact)

    mesh_path                      = save_mesh_grid(x, time_values)
    table_txt_path, table_csv_path = save_solution_table(x, time_values, solution)
    simulation_data_path           = save_simulation_data(x, time_values, solution)
    profiles_path                  = plot_solution_profiles(x, time_values, solution, exact)
    heatmap_path                   = plot_heatmap(x, time_values, solution)
    error_path                     = plot_error_analysis(time_values, error_analysis)

    output_files = {
        "mesh grid points":   mesh_path,
        "solution table txt": table_txt_path,
        "solution table csv": table_csv_path,
        "simulation data":    simulation_data_path,
        "solution profiles":  profiles_path,
        "solution heatmap":   heatmap_path,
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

    print("BTCS question lab completed.")
    print(f"Report:                {report_path}")
    print(f"Interactive dashboard: {interactive_dashboard_path}")
    print(f"Mesh ratio r = {r:.6f}  (BTCS — unconditionally stable)")
    print(f"Time step k  = {k:.8f}")
    print(f"Space step h = {h:.8f}")
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
