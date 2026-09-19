"""Unified single-problem workspace for the 1D heat-equation solver."""

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


METHOD_LABELS = {"ftcs": "FTCS", "btcs": "BTCS", "crank_nicolson": "Crank-Nicolson"}
DEFAULTS = {
    "method": "crank_nicolson", "alpha": "1.0", "x_start": "0.0", "x_end": "1.0",
    "final_time": "0.1", "space_points": "31", "r": "0.25", "left_boundary": "0.0",
    "right_boundary": "0.0", "initial_condition": "sin_pi",
}


def _value(query: dict[str, list[str]], name: str) -> str:
    return query.get(name, [DEFAULTS[name]])[0]


def _build_problem(query: dict[str, list[str]]) -> tuple[HeatEquationProblem, dict[str, str]]:
    values = {name: _value(query, name) for name in DEFAULTS}
    if values["method"] not in METHOD_LABELS:
        raise ValueError("Choose FTCS, BTCS, or Crank-Nicolson.")
    problem = HeatEquationProblem(
        alpha=float(values["alpha"]), x_start=float(values["x_start"]), x_end=float(values["x_end"]),
        final_time=float(values["final_time"]), space_points=int(values["space_points"]), r=float(values["r"]),
        left_boundary=float(values["left_boundary"]), right_boundary=float(values["right_boundary"]),
        initial_condition=values["initial_condition"],
    )
    problem.validate()
    if problem.space_points > 121:
        raise ValueError("Use 121 spatial points or fewer in this workspace.")
    return problem, values


def _plots(result, exact: np.ndarray | None) -> tuple[str, str]:
    selected = np.unique(np.linspace(0, len(result.time) - 1, min(6, len(result.time)), dtype=int))
    profile = go.Figure()
    for index in selected:
        profile.add_trace(go.Scatter(x=result.x, y=result.values[index], mode="lines", name=f"t = {result.time[index]:.5f}", hovertemplate="x=%{x:.5f}<br>u=%{y:.6f}<extra></extra>"))
    if exact is not None:
        profile.add_trace(go.Scatter(x=result.x, y=exact[-1], mode="lines", name="Exact at final time", line={"color": "#172433", "dash": "dash", "width": 2}))
    profile.update_layout(title="Temperature profiles", xaxis_title="space x", yaxis_title="u(x, t)", height=440, margin={"l": 60, "r": 28, "t": 58, "b": 58}, legend={"orientation": "h", "y": -0.24})

    surface = go.Figure(go.Surface(x=result.x, y=result.time, z=result.values, colorscale="Viridis", hovertemplate="x=%{x:.5f}<br>t=%{y:.6f}<br>u=%{z:.6f}<extra></extra>"))
    surface.update_layout(title="3D numerical solution", height=720, margin={"l": 72, "r": 72, "t": 65, "b": 72}, scene={"xaxis": {"title": "space x", "automargin": True}, "yaxis": {"title": "time t", "automargin": True}, "zaxis": {"title": "solution u(x, t)", "automargin": True}, "aspectmode": "manual", "aspectratio": {"x": 1.45, "y": 1.15, "z": 0.8}})
    return profile.to_html(full_html=False, include_plotlyjs="cdn"), surface.to_html(full_html=False, include_plotlyjs=False)


def _table(result) -> str:
    headings = "".join(f"<th>U({x:.3g})</th>" for x in result.x)
    rows = "".join("<tr>" + f"<td>{index}</td><td>{time:.6f}</td>" + "".join(f"<td>{item:.6f}</td>" for item in result.values[index]) + "</tr>" for index, time in enumerate(result.time))
    return f"<section class='table-section'><div class='section-title'><span>03</span><div><h2>Computed values</h2><p>Every row is one complete time level across the spatial grid.</p></div></div><div class='table-wrap'><table><thead><tr><th>n</th><th>t</th>{headings}</tr></thead><tbody>{rows}</tbody></table></div></section>"


def _field(name: str, label: str, values: dict[str, str], step: str = "any") -> str:
    return f"<label>{label}<input name='{name}' value='{html.escape(values[name])}' type='number' step='{step}' required></label>"


def _page(query: dict[str, list[str]]) -> str:
    message = ""
    try:
        problem, values = _build_problem(query)
        result = solve_heat_equation(problem, values["method"])
        exact = None
        if problem.initial_condition == "sin_pi" and problem.x_start == 0.0 and problem.x_end == 1.0 and problem.left_boundary == 0.0 and problem.right_boundary == 0.0:
            exact = exact_sine_solution(result, problem.alpha)
        profile, surface = _plots(result, exact)
        error = max_absolute_error(result, exact) if exact is not None else None
        stability = "Unstable mesh" if any("unstable" in warning.lower() for warning in result.warnings) else ("r <= 0.5" if result.method == "ftcs" else "Unconditional")
        formula = {
            "ftcs": r"\\[ U_i^{n+1}=rU_{i-1}^n+(1-2r)U_i^n+rU_{i+1}^n \\]",
            "btcs": r"\\[ -rU_{i-1}^{n+1}+(1+2r)U_i^{n+1}-rU_{i+1}^{n+1}=U_i^n \\]",
            "crank_nicolson": r"\\[ -\\frac{r}{2}U_{i-1}^{n+1}+(1+r)U_i^{n+1}-\\frac{r}{2}U_{i+1}^{n+1}=\\frac{r}{2}U_{i-1}^{n}+(1-r)U_i^n+\\frac{r}{2}U_{i+1}^{n} \\]",
        }[result.method]
        error_text = f"{error:.2e}" if error is not None else "No reference"
        content = f"""
        <section class='result-intro'><div><p class='eyebrow'>Computed solution</p><h2>{METHOD_LABELS[result.method]} on the 1D heat equation</h2><p class='subcopy'>\\(u_t = {problem.alpha:g}u_{{xx}}\\), with the selected domain, boundaries, and initial condition.</p></div><div class='formula'>{formula}</div></section>
        <section class='metrics'><article><span>Mesh ratio</span><strong>{result.r:.5f}</strong></article><article><span>Time levels</span><strong>{len(result.time)}</strong></article><article><span>Stability</span><strong>{stability}</strong></article><article><span>Max error</span><strong>{error_text}</strong></article></section>
        <section class='chart'>{profile}</section>
        {_table(result)}
        <section class='surface'>{surface}</section>"""
    except (ValueError, TypeError) as exc:
        values = DEFAULTS
        content = f"<section class='error'><h2>Check the question details</h2><p>{html.escape(str(exc))}</p></section>"
        message = "Correct the values and solve again."

    fields = "".join([
        _field("alpha", "Diffusivity alpha", values), _field("x_start", "Domain start", values),
        _field("x_end", "Domain end", values), _field("final_time", "Final time", values),
        _field("space_points", "Spatial points", values, "1"), _field("r", "Mesh ratio r", values),
        _field("left_boundary", "Left boundary", values), _field("right_boundary", "Right boundary", values),
    ])
    methods = "".join(f"<option value='{name}' {'selected' if values['method'] == name else ''}>{label}</option>" for name, label in METHOD_LABELS.items())
    conditions = "".join(f"<option value='{name}' {'selected' if values['initial_condition'] == name else ''}>{label}</option>" for name, label in (("sin_pi", "Sine wave"), ("gaussian", "Gaussian pulse"), ("hot_center", "Hot centre"), ("zero", "Zero")))
    return f"""<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>Solve PDE | PDE Solver</title><style>
    *{{box-sizing:border-box}}body{{margin:0;background:#f4f7f8;color:#16232d;font:15px/1.5 Inter,Arial,sans-serif}}a{{color:inherit}}.topbar{{height:64px;background:#102d3a;color:#fff;display:flex;align-items:center;justify-content:space-between;padding:0 max(22px,calc((100% - 1320px)/2))}}.brand{{font-size:18px;font-weight:700;text-decoration:none}}.nav{{display:flex;gap:18px}}.nav a{{font-size:14px;color:#d2e1e4;text-decoration:none}}.nav a.active{{color:#fff;border-bottom:2px solid #f18158;padding-bottom:4px}}main{{max-width:1320px;margin:auto;padding:30px 22px 64px}}h1,h2{{margin:0}}h1{{font-size:30px}}.heading p{{margin:7px 0 24px;color:#5c6d76}}.workspace{{display:grid;grid-template-columns:300px minmax(0,1fr);gap:24px;align-items:start}}form{{background:#fff;border:1px solid #d7e1e5;border-radius:6px;padding:18px;position:sticky;top:16px;max-height:calc(100vh - 32px);overflow:auto}}form h2{{font-size:18px}}form>p{{margin:4px 0 14px;color:#61727b;font-size:13px}}.form-group{{border-top:1px solid #e2e8ea;margin-top:15px;padding-top:13px}}.group-title,.eyebrow{{font-size:11px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:#2a7280}}label{{display:block;color:#42545e;font-size:12px;font-weight:600;margin-top:10px}}input,select{{font:inherit;color:#16232d;width:100%;margin-top:4px;padding:8px;border:1px solid #b9c8ce;border-radius:4px;background:#fff}}button{{margin-top:18px;width:100%;padding:10px;border:0;border-radius:4px;background:#df633f;color:#fff;font:600 14px inherit;cursor:pointer}}.result-intro{{display:flex;justify-content:space-between;gap:22px;background:#fff;border-top:4px solid #df633f;padding:24px}}.result-intro h2{{font-size:23px;margin:4px 0 5px}}.subcopy{{color:#596b75;margin:0}}.formula{{max-width:470px;align-self:center;overflow:auto}}.metrics{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:16px 0}}.metrics article{{background:#fff;border:1px solid #dce5e8;padding:14px}}.metrics span{{display:block;color:#60717a;font-size:12px}}.metrics strong{{display:block;margin-top:4px;font-size:17px;font-variant-numeric:tabular-nums}}.chart,.surface,.table-section{{background:#fff;border:1px solid #dce5e8;margin-top:16px;overflow:hidden}}.table-section{{padding:22px}}.section-title{{display:flex;gap:12px;align-items:start;margin-bottom:14px}}.section-title>span{{display:grid;place-items:center;width:30px;height:30px;background:#e7f0f4;color:#2a7280;font-weight:700}}.section-title h2{{font-size:19px}}.section-title p{{color:#61727b;margin:3px 0 0;font-size:13px}}.table-wrap{{overflow:auto;max-height:560px;border:1px solid #dce5e8}}table{{border-collapse:collapse;width:max-content;min-width:100%;font-size:13px}}th,td{{padding:7px 10px;text-align:right;border-bottom:1px solid #e2e9eb;white-space:nowrap}}th{{position:sticky;top:0;background:#edf3f5;z-index:1}}.surface{{padding:6px 16px 16px}}.error{{background:#fff;border-top:4px solid #bf4141;padding:24px}}@media(max-width:920px){{.workspace{{grid-template-columns:1fr}}form{{position:static;max-height:none}}.result-intro{{display:block}}.formula{{max-width:none;margin-top:16px}}.metrics{{grid-template-columns:repeat(2,1fr)}}}}@media(max-width:560px){{main{{padding:22px 14px 44px}}.topbar{{padding:0 16px}}.nav{{gap:10px}}.nav a{{font-size:12px}}.metrics{{grid-template-columns:1fr}}}}
    </style><script>window.MathJax={{tex:{{inlineMath:[['\\\\(','\\\\)']],displayMath:[['\\\\[','\\\\]']]}}}};</script><script async src='https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js'></script></head><body><header class='topbar'><a class='brand' href='http://127.0.0.1:8500'>PDE Solver</a><nav class='nav'><a href='http://127.0.0.1:8500'>Home</a><a class='active' href='http://127.0.0.1:8501'>Solve</a><a href='http://127.0.0.1:8502'>Compare</a></nav></header><main><section class='heading'><h1>Solve one PDE problem</h1><p>Choose a numerical method, set the equation inputs, then inspect the result, the scheme, every time level, and the full solution surface.</p></section><div class='workspace'><form method='get'><h2>Problem setup</h2><p>{message or 'The current equation family is the 1D heat equation.'}</p><div class='form-group'><div class='group-title'>Numerical method</div><label>Solver method<select name='method'>{methods}</select></label></div><div class='form-group'><div class='group-title'>Model and mesh</div>{fields}<label>Initial condition<select name='initial_condition'>{conditions}</select></label></div><button type='submit'>Solve PDE</button></form><div>{content}</div></div></main></body></html>"""


class SolverHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        page = _page(parse_qs(urlparse(self.path).query)).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(page)))
        self.end_headers()
        self.wfile.write(page)

    def log_message(self, format: str, *args: object) -> None:
        return


if __name__ == "__main__":
    server = ThreadingHTTPServer(("127.0.0.1", 8501), SolverHandler)
    print("PDE Solver workspace: http://127.0.0.1:8501")
    server.serve_forever()
