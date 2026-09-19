"""Run a Crank-Nicolson heat-equation lesson using the shared solver core."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go


LESSON_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = LESSON_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pde_solver import HeatEquationProblem, exact_sine_solution, max_absolute_error, solve_heat_equation
from question_config import QUESTION


OUTPUT_DIR = LESSON_DIR / "outputs" / "question_results"


def build_problem(question: dict[str, object]) -> HeatEquationProblem:
    """Translate the lesson configuration into the shared solver input model."""
    return HeatEquationProblem(
        alpha=float(question["alpha"]),
        x_start=float(question["x_start"]),
        x_end=float(question["x_end"]),
        final_time=float(question["final_time"]),
        space_points=int(question["space_points"]),
        left_boundary=float(question["left_boundary"]),
        right_boundary=float(question["right_boundary"]),
        initial_condition=question["initial_condition"],
        r=float(question["r"]) if question.get("r") is not None else None,
        time_step=float(question["time_step"]) if question.get("time_step") is not None else None,
    )


def save_solution_csv(x: np.ndarray, time: np.ndarray, values: np.ndarray) -> Path:
    path = OUTPUT_DIR / "crank_nicolson_solution.csv"
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["time_index", "time", "space_index", "x", "temperature"])
        for time_index, current_time in enumerate(time):
            for space_index, current_x in enumerate(x):
                writer.writerow([time_index, current_time, space_index, current_x, values[time_index, space_index]])
    return path


def save_static_plots(
    x: np.ndarray, time: np.ndarray, values: np.ndarray, exact: np.ndarray | None
) -> tuple[Path, Path]:
    profiles_path = OUTPUT_DIR / "solution_profiles.png"
    heatmap_path = OUTPUT_DIR / "solution_heatmap.png"

    figure, axis = plt.subplots(figsize=(10, 5))
    selected = np.unique(np.linspace(0, len(time) - 1, min(6, len(time)), dtype=int))
    for index in selected:
        axis.plot(x, values[index], label=f"t = {time[index]:.4f}")
        if exact is not None:
            axis.plot(x, exact[index], "--", color="black", alpha=0.35)
    axis.set(title="Crank-Nicolson solution profiles", xlabel="x", ylabel="u(x, t)")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(profiles_path, dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(10, 5))
    image = axis.pcolormesh(x, time, values, shading="auto", cmap="inferno")
    figure.colorbar(image, ax=axis, label="temperature")
    axis.set(title="Crank-Nicolson heat diffusion", xlabel="x", ylabel="time")
    figure.tight_layout()
    figure.savefig(heatmap_path, dpi=160)
    plt.close(figure)
    return profiles_path, heatmap_path


def save_dashboard(x: np.ndarray, time: np.ndarray, values: np.ndarray, exact: np.ndarray | None) -> Path:
    path = OUTPUT_DIR / "interactive_dashboard.html"
    figure = go.Figure()
    figure.add_trace(
        go.Surface(x=x, y=time, z=values, colorscale="Turbo", colorbar_title="temperature")
    )
    if exact is not None:
        figure.add_trace(
            go.Surface(
                x=x,
                y=time,
                z=exact,
                colorscale="Greys",
                opacity=0.35,
                showscale=False,
                name="Exact solution",
            )
        )
    figure.update_layout(
        title="Crank-Nicolson: space-time temperature surface",
        scene={"xaxis_title": "x", "yaxis_title": "time", "zaxis_title": "temperature"},
        margin={"l": 0, "r": 0, "b": 0, "t": 45},
    )
    figure.write_html(path, include_plotlyjs=True)
    return path


def save_report(question: dict[str, object], result, error: float | None, files: dict[str, Path]) -> Path:
    path = OUTPUT_DIR / "question_report.md"
    lines = [
        f"# {question['title']}",
        "",
        "## Method",
        "",
        "Crank-Nicolson for the 1D heat equation `u_t = alpha * u_xx`.",
        "",
        "## Mesh",
        "",
        f"- Space points: {len(result.x)}",
        f"- Time levels: {len(result.time)}",
        f"- Space step h: {result.space_step:.8f}",
        f"- Time step k: {result.time_step:.8f}",
        f"- Mesh ratio r: {result.r:.8f}",
        "",
        "## Accuracy",
        "",
        "- Crank-Nicolson is unconditionally stable for this linear heat equation.",
    ]
    if error is not None:
        lines.append(f"- Maximum absolute error: {error:.6e}")
    if result.warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in result.warnings)
    lines.extend(["", "## Generated Files", ""])
    lines.extend(f"- {label}: `{file.name}`" for label, file in files.items())
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_question_lab() -> None:
    """Generate numerical, visual, and report outputs for the selected question."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    problem = build_problem(QUESTION)
    result = solve_heat_equation(problem, "crank_nicolson")

    exact = None
    error = None
    if QUESTION.get("exact_solution") == "sin_pi_zero_boundary":
        exact = exact_sine_solution(result, problem.alpha)
        error = max_absolute_error(result, exact)

    solution_path = save_solution_csv(result.x, result.time, result.values)
    profiles_path, heatmap_path = save_static_plots(result.x, result.time, result.values, exact)
    dashboard_path = save_dashboard(result.x, result.time, result.values, exact)
    files = {
        "solution table": solution_path,
        "solution profiles": profiles_path,
        "solution heatmap": heatmap_path,
        "interactive dashboard": dashboard_path,
    }
    report_path = save_report(QUESTION, result, error, files)

    print("Crank-Nicolson question lab completed.")
    print(f"Report: {report_path}")
    print(f"Dashboard: {dashboard_path}")
    print(f"Mesh ratio r = {result.r:.6f}")
    if error is not None:
        print(f"Maximum absolute error: {error:.6e}")


if __name__ == "__main__":
    run_question_lab()
