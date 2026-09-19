"""A local browser interface for exploring Crank-Nicolson heat-equation solves."""

from __future__ import annotations

import html
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import plotly.graph_objects as go


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pde_solver import HeatEquationProblem, exact_sine_solution, max_absolute_error, solve_heat_equation


DEFAULTS = {
    "method": "crank_nicolson",
    "mesh_input": "points_and_r",
    "alpha": "1.0",
    "x_start": "0.0",
    "x_end": "1.0",
    "final_time": "0.1",
    "space_points": "51",
    "space_step": "0.02",
    "time_step": "0.0002",
    "r": "0.5",
    "left_boundary": "0.0",
    "right_boundary": "0.0",
    "initial_condition": "sin_pi",
}
ALLOWED_CONDITIONS = {"sin_pi", "gaussian", "hot_center", "zero"}
MESH_INPUT_LABELS = {
    "points_and_r": "Grid points and mesh ratio r",
    "steps": "Delta x and delta t",
}
METHOD_LABELS = {
    "ftcs": "FTCS",
    "btcs": "BTCS",
    "crank_nicolson": "Crank-Nicolson",
}


def _value(query: dict[str, list[str]], name: str) -> str:
    return query.get(name, [DEFAULTS[name]])[0]


def _problem_from_query(query: dict[str, list[str]]) -> tuple[HeatEquationProblem, dict[str, str]]:
    values = {name: _value(query, name) for name in DEFAULTS}
    if values["initial_condition"] not in ALLOWED_CONDITIONS:
        raise ValueError("Choose one of the available initial-condition presets.")
    if values["method"] not in METHOD_LABELS:
        raise ValueError("Choose FTCS, BTCS, or Crank-Nicolson.")
    if values["mesh_input"] not in MESH_INPUT_LABELS:
        raise ValueError("Choose a mesh-input mode.")

    x_start = float(values["x_start"])
    x_end = float(values["x_end"])
    if values["mesh_input"] == "steps":
        space_step = float(values["space_step"])
        if space_step <= 0:
            raise ValueError("Spatial step h must be positive.")
        intervals = (x_end - x_start) / space_step
        rounded_intervals = round(intervals)
        if rounded_intervals < 2 or not np.isclose(intervals, rounded_intervals, rtol=1e-9, atol=1e-12):
            raise ValueError("The domain length must be an exact multiple of spatial step h, with at least two intervals.")
        space_points = rounded_intervals + 1
        time_step = float(values["time_step"])
        r = None
    else:
        space_points = int(values["space_points"])
        time_step = None
        r = float(values["r"])

    problem = HeatEquationProblem(
        alpha=float(values["alpha"]),
        x_start=x_start,
        x_end=x_end,
        final_time=float(values["final_time"]),
        space_points=space_points,
        time_step=time_step,
        r=r,
        left_boundary=float(values["left_boundary"]),
        right_boundary=float(values["right_boundary"]),
        initial_condition=values["initial_condition"],
    )
    problem.validate()
    if problem.space_points > 201:
        raise ValueError("Use 201 space points or fewer in the interactive lab.")
    return problem, values


def _figure_html(result, exact: np.ndarray | None) -> tuple[str, str]:
    selected = np.unique(np.linspace(0, len(result.time) - 1, min(6, len(result.time)), dtype=int))
    profile = go.Figure()
    for index in selected:
        profile.add_trace(
            go.Scatter(
                x=result.x,
                y=result.values[index],
                mode="lines",
                name=f"{METHOD_LABELS[result.method]}, t={result.time[index]:.4f}",
                hovertemplate="x=%{x:.5f}<br>u=%{y:.6f}<extra></extra>",
            )
        )
    if exact is not None:
        profile.add_trace(
            go.Scatter(
                x=result.x,
                y=exact[-1],
                mode="lines",
                line={"color": "#172033", "dash": "dash"},
                name=f"Exact, t={result.time[-1]:.4f}",
                hovertemplate="x=%{x:.5f}<br>exact u=%{y:.6f}<extra></extra>",
            )
        )
    profile.add_hline(y=0.0, line={"color": "#8795a1", "width": 1, "dash": "dot"})
    profile.update_layout(
        title="Temperature profiles: solid = numerical, dashed = exact final profile", xaxis_title="space x", yaxis_title="u(x, t)",
        margin={"l": 40, "r": 20, "t": 45, "b": 40}, height=360,
    )

    surface = go.Figure(go.Surface(
        x=result.x, y=result.time, z=result.values, colorscale="Turbo",
        hovertemplate="x=%{x:.5f}<br>t=%{y:.6f}<br>u=%{z:.6f}<extra></extra>",
        name="Numerical solution",
    ))
    surface.update_layout(
        title="Interactive numerical space-time surface", scene={"xaxis_title": "x", "yaxis_title": "time", "zaxis_title": "u"},
        margin={"l": 0, "r": 0, "t": 45, "b": 0}, height=520,
    )
    return (
        profile.to_html(full_html=False, include_plotlyjs="cdn"),
        surface.to_html(full_html=False, include_plotlyjs=False),
    )


def _steps_html(result, problem: HeatEquationProblem) -> str:
    count = min(7, len(result.x))
    first_row = "".join(
        f"<tr><td>{index}</td><td>{result.x[index]:.4f}</td><td>{result.values[1, index]:.6f}</td></tr>"
        for index in range(count)
    )
    formulas = {
        "ftcs": r"\\[ U_i^{n+1} = rU_{i-1}^n + (1-2r)U_i^n + rU_{i+1}^n \\]",
        "btcs": r"\\[ -rU_{i-1}^{n+1} + (1+2r)U_i^{n+1} - rU_{i+1}^{n+1} = U_i^n \\]",
        "crank_nicolson": r"\\[ -\\frac{r}{2}U_{i-1}^{n+1} + (1+r)U_i^{n+1} - \\frac{r}{2}U_{i+1}^{n+1} = \\frac{r}{2}U_{i-1}^n + (1-r)U_i^n + \\frac{r}{2}U_{i+1}^n \\]",
    }
    method_notes = {
        "ftcs": "FTCS is explicit: the next value is calculated from the previous time level. It requires r <= 0.5 for stability.",
        "btcs": "BTCS is implicit: a tridiagonal linear system is solved at each time level. It is unconditionally stable.",
        "crank_nicolson": "Crank-Nicolson averages the explicit and implicit spatial terms. It solves a tridiagonal system and is unconditionally stable.",
    }
    return f"""
    <section class="band" id="steps">
      <div class="section-heading"><span>02</span><div><h2>{METHOD_LABELS[result.method]} steps</h2><p>{method_notes[result.method]}</p></div></div>
      <div class="steps-grid">
        <article class="step"><b>1. Build the mesh</b><p>h = {result.space_step:.6f}, k = {result.time_step:.6f}, r = alpha k / h^2 = {result.r:.6f}.</p></article>
        <article class="step"><b>2. Form the equation</b><p>{formulas[result.method]}</p></article>
        <article class="step"><b>3. Apply boundaries</b><p>\\(u({problem.x_start:g},t) = {problem.left_boundary:g}\\) and \\(u({problem.x_end:g},t) = {problem.right_boundary:g}\\). Their contributions move into the right-hand side.</p></article>
        <article class="step"><b>4. Advance in time</b><p>{"The Thomas tridiagonal algorithm computes the interior temperatures at every time level." if result.method != "ftcs" else "Use the explicit update formula to compute every interior temperature at the next time level."}</p></article>
      </div>
      <div class="table-wrap"><table><thead><tr><th>i</th><th>x</th><th>U(i, 1)</th></tr></thead><tbody>{first_row}</tbody></table></div>
    </section>"""


def _solution_table_html(result) -> str:
    """Render every computed time level with horizontal scrolling for wide grids."""
    headings = "".join(f"<th>U({x:.4g})</th>" for x in result.x)
    rows = "".join(
        "<tr>"
        f"<td>{time_index}</td><td>{current_time:.6f}</td>"
        + "".join(f"<td>{value:.6f}</td>" for value in result.values[time_index])
        + "</tr>"
        for time_index, current_time in enumerate(result.time)
    )
    return f"""
    <section class="band" id="values">
      <div class="section-heading"><span>03</span><div><h2>Computed values at every time level</h2><p>Each row is one complete numerical solution \\(U_i^n\\) across the spatial mesh.</p></div></div>
      <div class="table-wrap solution-table"><table><thead><tr><th>n</th><th>t</th>{headings}</tr></thead><tbody>{rows}</tbody></table></div>
    </section>"""


def _model_html(problem: HeatEquationProblem) -> str:
    return f"""
    <div class="model">
      <h3>Model used for this question</h3>
      <div class="equation">\\[ \\frac{{\\partial u}}{{\\partial t}} = \\alpha \\frac{{\\partial^2 u}}{{\\partial x^2}}, \\qquad {problem.x_start:g} \\le x \\le {problem.x_end:g}, \\quad 0 \\le t \\le {problem.final_time:g} \\]</div>
      <p>This lab uses the diffusion-coefficient form. If a textbook writes \\(u_{{xx}} = c u_t\\), enter \\(\\alpha = 1/c\\). If it writes \\(u_t = c u_{{xx}}\\), enter \\(\\alpha = c\\).</p>
      <div class="symbol-grid"><div><b>\\(u(x,t)\\)</b><span>temperature or concentration</span></div><div><b>\\(\\alpha\\)</b><span>diffusivity</span></div><div><b>\\(h\\)</b><span>space step</span></div><div><b>\\(k\\)</b><span>time step</span></div><div><b>\\(r=\\alpha k/h^2\\)</b><span>mesh ratio</span></div><div><b>\\(U_i^n\\)</b><span>value at node \\(x_i\\), time \\(t_n\\)</span></div></div>
    </div>"""


def _page(query: dict[str, list[str]]) -> str:
    error_message = ""
    try:
        problem, values = _problem_from_query(query)
        result = solve_heat_equation(problem, values["method"])
        if len(result.time) > 1001:
            raise ValueError("These parameters create more than 1001 time levels. Increase r or reduce final time to keep the complete table readable.")
        exact = None
        error = None
        if problem.initial_condition == "sin_pi" and problem.x_start == 0.0 and problem.x_end == 1.0 and problem.left_boundary == 0.0 and problem.right_boundary == 0.0:
            exact = exact_sine_solution(result, problem.alpha)
            error = max_absolute_error(result, exact)
        profile, surface = _figure_html(result, exact)
        exact_note = (
            f'<p class="exact-note">The dashed profile is the analytical reference solution: \\(u(x,t)=e^{{-\\alpha\\pi^2t}}\\sin(\\pi x)\\). It is available only because this problem uses \\(u(x,0)=\\sin(\\pi x)\\) with zero boundaries on \\([0,1]\\).</p>'
            if exact is not None else ""
        )
        summary = f"""
          <div class="metrics"><div><small>Method</small><strong>{METHOD_LABELS[result.method]}</strong></div><div><small>Mesh ratio</small><strong>{result.r:.5f}</strong></div><div><small>Time levels</small><strong>{len(result.time)}</strong></div><div><small>Max error</small><strong>{error:.3e}</strong></div></div>
          <p class="question">Solve \\(u_t = {problem.alpha:g}u_{{xx}}\\) on \\({problem.x_start:g} \\le x \\le {problem.x_end:g}\\), from \\(t = 0\\) to \\(t = {problem.final_time:g}\\), using {METHOD_LABELS[result.method]}.</p>
        """ if error is not None else f"""
          <div class="metrics"><div><small>Method</small><strong>{METHOD_LABELS[result.method]}</strong></div><div><small>Mesh ratio</small><strong>{result.r:.5f}</strong></div><div><small>Time levels</small><strong>{len(result.time)}</strong></div><div><small>Stability</small><strong>{"r <= 0.5 required" if result.method == "ftcs" else "Unconditional"}</strong></div></div>
          <p class="question">Solve \\(u_t = {problem.alpha:g}u_{{xx}}\\) on \\({problem.x_start:g} \\le x \\le {problem.x_end:g}\\), from \\(t = 0\\) to \\(t = {problem.final_time:g}\\), using {METHOD_LABELS[result.method]}.</p>
        """
        content = f"""
          <section class="band result"><div class="section-heading"><span>01</span><div><h2>Your problem and solution</h2><p>Numerical result from the parameters above.</p></div></div>{summary}{exact_note}{_model_html(problem)}</section>
          {_steps_html(result, problem)}
          {_solution_table_html(result)}
          <section class="band" id="graphs"><div class="section-heading"><span>04</span><div><h2>Explore the solution</h2><p>Compare profile curves or rotate the space-time surface.</p></div></div><div class="plots"><div>{profile}</div><div>{surface}</div></div></section>
        """
    except (TypeError, ValueError) as exc:
        values = DEFAULTS
        content = f'<section class="band error"><h2>Check the parameters</h2><p>{html.escape(str(exc))}</p></section>'
        error_message = "Fix the highlighted values and solve again."

    common_inputs = "".join(
        f'<label>{label}<input name="{name}" value="{html.escape(values[name])}" type="number" step="any" required></label>'
        for name, label in (("alpha", "Diffusivity alpha"), ("x_start", "Domain start"), ("x_end", "Domain end"), ("final_time", "Final time"), ("left_boundary", "Left boundary"), ("right_boundary", "Right boundary"))
    )
    grid_inputs = "".join(
        f'<label>{label}<input name="{name}" value="{html.escape(values[name])}" type="number" step="{step}" required></label>'
        for name, label, step in (("space_points", "Number of spatial points", "1"), ("r", "Mesh ratio r", "any"))
    )
    step_inputs = "".join(
        f'<label>{label}<input name="{name}" value="{html.escape(values[name])}" type="number" step="any" required></label>'
        for name, label in (("space_step", "Delta x",), ("time_step", "Delta t",))
    )
    inputs = f"""<div class="form-section"><div class="form-section-title">Physical model</div>{common_inputs}</div>
      <div class="form-section mesh-section"><div class="form-section-title">Computational mesh</div>
      <div id="pointsAndRFields">{grid_inputs}</div>
      <div id="stepFields">{step_inputs}</div></div>
      <style>
        .form-section {{border-top:1px solid #d9e0e7;margin-top:16px;padding-top:12px}}
        .form-section-title {{color:#235789;font-size:12px;font-weight:700;letter-spacing:.04em;text-transform:uppercase}}
        .mesh-section {{background:#f8fafc;margin-left:-8px;margin-right:-8px;padding:12px 8px 2px}}
        .exact-note {{margin-top:14px;padding:10px 12px;background:#edf6f2;border-left:3px solid #2a7a62;color:#28493e;font-size:13px}}
        @media (min-width:801px) {{
          form {{max-height:calc(100vh - 32px);overflow-y:auto;overscroll-behavior:contain;scrollbar-color:#93a4b7 #edf1f5}}
        }}
      </style>
      <script>
        const meshSelect = document.querySelector('[name="mesh_input"]');
        const pointsAndRFields = document.getElementById('pointsAndRFields');
        const stepFields = document.getElementById('stepFields');
        function updateMeshFields() {{
          const useSteps = meshSelect.value === 'steps';
          pointsAndRFields.hidden = useSteps;
          stepFields.hidden = !useSteps;
          pointsAndRFields.querySelectorAll('input').forEach((input) => input.disabled = useSteps);
          stepFields.querySelectorAll('input').forEach((input) => input.disabled = !useSteps);
        }}
        meshSelect.addEventListener('change', updateMeshFields);
        updateMeshFields();
      </script>"""
    options = "".join(
        f'<option value="{name}" {"selected" if values["initial_condition"] == name else ""}>{label}</option>'
        for name, label in (("sin_pi", "Sine wave"), ("gaussian", "Gaussian pulse"), ("hot_center", "Hot centre"), ("zero", "Zero"))
    )
    method_options = "".join(
        f'<option value="{name}" {"selected" if values["method"] == name else ""}>{label}</option>'
        for name, label in METHOD_LABELS.items()
    )
    mesh_options = "".join(
        f'<option value="{name}" {"selected" if values["mesh_input"] == name else ""}>{label}</option>'
        for name, label in MESH_INPUT_LABELS.items()
    )
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Crank-Nicolson PDE Lab</title><style>
    *{{box-sizing:border-box}} body{{margin:0;background:#f5f7fa;color:#172033;font:15px/1.55 Arial,sans-serif}} header{{background:#143642;color:white;padding:30px max(22px,calc((100% - 1200px)/2));border-bottom:4px solid #d9603d}} h1,h2,p{{margin:0}} h1{{font-size:28px;letter-spacing:0}} header p{{color:#d7e5e6;margin-top:5px}} main{{max-width:1200px;margin:auto;padding:28px 22px 56px}} .workspace{{display:grid;grid-template-columns:290px 1fr;gap:24px;align-items:start}} form{{background:white;border:1px solid #d9e0e7;border-radius:6px;padding:18px;position:sticky;top:16px}} form h2{{font-size:18px;margin-bottom:4px}} form p{{font-size:13px;color:#526175;margin-bottom:14px}} label{{display:block;font-size:12px;font-weight:700;color:#3d4b5d;margin-top:10px}} input,select{{width:100%;margin-top:3px;border:1px solid #aeb9c7;border-radius:4px;padding:8px;background:#fff;color:#172033;font:inherit}} button{{width:100%;margin-top:18px;background:#d9603d;color:#fff;border:0;border-radius:4px;padding:10px;font-weight:700;cursor:pointer}} .band{{background:white;border-top:3px solid #235789;padding:23px;margin-bottom:20px}} .result{{border-top-color:#d9603d}} .error{{border-top-color:#c43d3d}} .section-heading{{display:flex;gap:12px;align-items:start;margin-bottom:18px}} .section-heading span{{display:grid;place-items:center;width:30px;height:30px;background:#e5edf4;color:#235789;font-weight:700}} h2{{font-size:20px}} .section-heading p{{font-size:13px;color:#66758a}} .metrics{{display:grid;grid-template-columns:repeat(4,1fr);gap:1px;background:#d9e0e7;margin-bottom:18px}} .metrics div{{background:#f8fafc;padding:13px}} small{{display:block;color:#66758a}} strong{{font-size:17px}} .question{{padding:13px;background:#fff8f3;border-left:3px solid #d9603d}} code{{font-family:Consolas,monospace}} .steps-grid{{display:grid;grid-template-columns:repeat(2,1fr);gap:10px}} .step{{border:1px solid #d9e0e7;padding:13px;min-height:118px}} .step p{{font-size:13px;color:#435268;margin-top:6px}} .table-wrap{{overflow:auto;margin-top:16px}} table{{border-collapse:collapse;width:100%;font-size:13px}} th,td{{padding:7px 10px;text-align:left;border-bottom:1px solid #d9e0e7}} th{{background:#eef3f7}} .plots{{display:grid;grid-template-columns:1fr 1fr;gap:16px}} .surface{{margin-top:16px}} @media(max-width:800px){{.workspace{{grid-template-columns:1fr}} form{{position:static}} .metrics,.steps-grid,.plots{{grid-template-columns:1fr}} header{{padding:22px}} main{{padding:18px}}}}
    </style><script>window.MathJax={{tex:{{inlineMath:[['\\\\(','\\\\)']],displayMath:[['\\\\[','\\\\]']]}}}};</script><script async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script></head><body><header><h1>1D Heat Equation PDE Lab</h1><p>Set the problem, choose a numerical method, inspect the computation, then explore the result.</p></header><main><div class="workspace"><form method="get"><h2>Set your problem</h2><p>{error_message or "Choose values and solve the 1D heat equation."}</p><label>Solver method<select name="method">{method_options}</select></label><label>Mesh information supplied by the question<select name="mesh_input">{mesh_options}</select></label>{inputs}<label>Initial condition<select name="initial_condition">{options}</select></label><button type="submit">Solve PDE</button></form><div>{content}</div></div></main></body></html>"""


class LabHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        query = parse_qs(urlparse(self.path).query)
        page = _page(query).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(page)))
        self.end_headers()
        self.wfile.write(page)

    def log_message(self, format: str, *args: object) -> None:
        return


def main() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 8501), LabHandler)
    print("Crank-Nicolson PDE Lab: http://127.0.0.1:8501")
    server.serve_forever()


if __name__ == "__main__":
    main()
