"""Compare FTCS, BTCS, and Crank-Nicolson on one 1D heat-equation problem."""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


LESSON_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = LESSON_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pde_solver import HeatEquationProblem, exact_sine_solution, max_absolute_error, solve_heat_equation
from question_config import QUESTION


OUTPUT_DIR = LESSON_DIR / "outputs" / "comparison_results"
METHODS = ("ftcs", "btcs", "crank_nicolson")
METHOD_LABELS = {"ftcs": "FTCS", "btcs": "BTCS", "crank_nicolson": "Crank-Nicolson"}
METHOD_COLORS = {"ftcs": "#d9603d", "btcs": "#235789", "crank_nicolson": "#2a7a62"}


def build_problem(question: dict[str, object]) -> HeatEquationProblem:
    return HeatEquationProblem(
        alpha=float(question["alpha"]), x_start=float(question["x_start"]), x_end=float(question["x_end"]),
        final_time=float(question["final_time"]), space_points=int(question["space_points"]),
        r=float(question["r"]), left_boundary=float(question["left_boundary"]),
        right_boundary=float(question["right_boundary"]), initial_condition=question["initial_condition"],
    )


def solve_all(problem: HeatEquationProblem) -> tuple[dict[str, object], np.ndarray | None, dict[str, float]]:
    results: dict[str, object] = {}
    runtimes: dict[str, float] = {}
    for method in METHODS:
        started = perf_counter()
        results[method] = solve_heat_equation(problem, method)
        runtimes[method] = perf_counter() - started
    exact = None
    if problem.initial_condition == "sin_pi" and problem.x_start == 0.0 and problem.x_end == 1.0 and problem.left_boundary == 0.0 and problem.right_boundary == 0.0:
        exact = exact_sine_solution(results["ftcs"], problem.alpha)
    return results, exact, runtimes


def save_summary_csv(results: dict[str, object], exact: np.ndarray | None, runtimes: dict[str, float]) -> Path:
    path = OUTPUT_DIR / "method_summary.csv"
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["method", "r", "stable", "maximum_absolute_error", "runtime_seconds", "warnings"])
        for method, result in results.items():
            error = max_absolute_error(result, exact) if exact is not None else None
            stable = not any("unstable" in warning.lower() for warning in result.warnings)
            writer.writerow([METHOD_LABELS[method], result.r, stable, error, runtimes[method], " | ".join(result.warnings)])
    return path


def save_plots(results: dict[str, object], exact: np.ndarray | None) -> tuple[Path, Path]:
    profile_path = OUTPUT_DIR / "final_time_comparison.png"
    error_path = OUTPUT_DIR / "error_history.png"
    figure, axis = plt.subplots(figsize=(10, 5))
    for method, result in results.items():
        axis.plot(result.x, result.values[-1], label=METHOD_LABELS[method], color=METHOD_COLORS[method], linewidth=2)
    if exact is not None:
        axis.plot(results["ftcs"].x, exact[-1], "--", color="#172033", label="Exact", linewidth=2)
    axis.set(title="Final-time solution comparison", xlabel="x", ylabel="u(x, T)")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(profile_path, dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(10, 5))
    if exact is not None:
        for method, result in results.items():
            error_history = np.max(np.abs(result.values - exact), axis=1)
            axis.semilogy(result.time, error_history, label=METHOD_LABELS[method], color=METHOD_COLORS[method], linewidth=2)
    axis.set(title="Maximum error over time", xlabel="time", ylabel="maximum absolute error")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(error_path, dpi=160)
    plt.close(figure)
    return profile_path, error_path


def save_dashboard(results: dict[str, object], exact: np.ndarray | None, runtimes: dict[str, float]) -> Path:
    path = OUTPUT_DIR / "interactive_comparison.html"
    figure = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=("Final-time profiles", "Maximum error over time", "Runtime", "Method guidance"),
        specs=[[{}, {}], [{}, {"type": "domain"}]],
    )
    for method, result in results.items():
        figure.add_trace(go.Scatter(x=result.x, y=result.values[-1], mode="lines", name=METHOD_LABELS[method], line={"color": METHOD_COLORS[method], "width": 3}), row=1, col=1)
        if exact is not None:
            error_history = np.max(np.abs(result.values - exact), axis=1)
            figure.add_trace(go.Scatter(x=result.time, y=error_history, mode="lines", name=f"{METHOD_LABELS[method]} error", legendgroup=method, showlegend=False, line={"color": METHOD_COLORS[method], "width": 2}), row=1, col=2)
    if exact is not None:
        figure.add_trace(go.Scatter(x=results["ftcs"].x, y=exact[-1], mode="lines", name="Exact", line={"color": "#172033", "dash": "dash"}), row=1, col=1)
    figure.add_trace(go.Bar(x=[METHOD_LABELS[method] for method in METHODS], y=[runtimes[method] for method in METHODS], marker_color=[METHOD_COLORS[method] for method in METHODS], name="Runtime"), row=2, col=1)
    figure.add_trace(go.Pie(labels=["FTCS", "BTCS", "Crank-Nicolson"], values=[1, 1, 1], text=["Fast, conditional", "Stable, first-order time", "Stable, second-order time"], textinfo="text", marker={"colors": [METHOD_COLORS[method] for method in METHODS]}), row=2, col=2)
    figure.update_xaxes(title_text="x", row=1, col=1)
    figure.update_yaxes(title_text="u(x, T)", row=1, col=1)
    figure.update_xaxes(title_text="time", row=1, col=2)
    figure.update_yaxes(title_text="maximum absolute error", type="log", row=1, col=2)
    figure.update_yaxes(title_text="seconds", row=2, col=1)
    figure.update_layout(title="Method comparison: accuracy, stability, and cost", height=820, showlegend=True)

    surface = go.Figure()
    for index, method in enumerate(METHODS):
        result = results[method]
        surface.add_trace(go.Surface(x=result.x, y=result.time, z=result.values, name=METHOD_LABELS[method], visible=index == 0, colorscale="Turbo", hovertemplate="x=%{x:.5f}<br>t=%{y:.6f}<br>u=%{z:.6f}<extra></extra>"))
    surface.update_layout(
        title="3D solution surface: choose a method",
        scene={"xaxis_title": "x", "yaxis_title": "time", "zaxis_title": "u"},
        height=620,
        updatemenus=[{
            "type": "buttons", "direction": "right", "x": 0.0, "y": 1.12,
            "buttons": [
                {"label": METHOD_LABELS[method], "method": "update", "args": [{"visible": [candidate == method for candidate in METHODS]}, {"title": f"3D solution surface: {METHOD_LABELS[method]}"}]}
                for method in METHODS
            ],
        }],
    )
    guide = """
    <section style="max-width:1100px;margin:24px auto;font-family:Arial,sans-serif;color:#172033">
      <h1>FTCS vs BTCS vs Crank-Nicolson</h1>
      <p>Every method uses the same PDE, mesh, boundary conditions, and initial condition. Read the top-left plot against the dashed exact curve, then use error history to see accuracy through the full simulation.</p>
      <ul><li><b>FTCS:</b> fast, but only safe when r <= 0.5.</li><li><b>BTCS:</b> stable for any r, with first-order time accuracy.</li><li><b>Crank-Nicolson:</b> stable with second-order time accuracy; inspect its 3D surface with the method buttons.</li></ul>
    </section>
    """
    page = "<html><head><title>Method Comparison</title></head><body>" + guide
    page += figure.to_html(full_html=False, include_plotlyjs=True)
    page += surface.to_html(full_html=False, include_plotlyjs=False)
    page += "</body></html>"
    path.write_text(page, encoding="utf-8")
    return path


def save_report(results: dict[str, object], exact: np.ndarray | None, runtimes: dict[str, float], files: dict[str, Path]) -> Path:
    path = OUTPUT_DIR / "comparison_report.md"
    lines = ["# Method Comparison", "", "## Shared Model", "", "`u_t = alpha * u_xx` with the same mesh, boundaries, and initial condition for every method.", "", "## Results", "", "| Method | r | Maximum absolute error | Runtime (seconds) |", "| --- | ---: | ---: | ---: |"]
    for method, result in results.items():
        error = max_absolute_error(result, exact) if exact is not None else None
        error_text = f"{error:.6e}" if error is not None else "No exact solution"
        lines.append(f"| {METHOD_LABELS[method]} | {result.r:.6f} | {error_text} | {runtimes[method]:.6e} |")
    lines.extend(["", "## Generated Files", ""])
    lines.extend(f"- {label}: `{file.name}`" for label, file in files.items())
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_comparison_lab() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results, exact, runtimes = solve_all(build_problem(QUESTION))
    summary_path = save_summary_csv(results, exact, runtimes)
    profiles_path, errors_path = save_plots(results, exact)
    dashboard_path = save_dashboard(results, exact, runtimes)
    files = {"summary table": summary_path, "final profiles": profiles_path, "error history": errors_path, "interactive dashboard": dashboard_path}
    report_path = save_report(results, exact, runtimes, files)
    print("Method comparison completed.")
    print(f"Report: {report_path}")
    print(f"Dashboard: {dashboard_path}")


if __name__ == "__main__":
    run_comparison_lab()
