"""Interactive comparison workspace for the shared 1D heat-equation solver."""

from __future__ import annotations

import html
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from time import perf_counter
from urllib.parse import parse_qs, urlparse

import numpy as np
import plotly.graph_objects as go


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pde_solver import HeatEquationProblem, exact_sine_solution, max_absolute_error, solve_heat_equation


METHODS = ("ftcs", "btcs", "crank_nicolson")
METHOD_LABELS = {"ftcs": "FTCS", "btcs": "BTCS", "crank_nicolson": "Crank-Nicolson"}
METHOD_COLORS = {"ftcs": "#dc6242", "btcs": "#2878b9", "crank_nicolson": "#159570"}
DEFAULTS = {
    "alpha": "1.0", "x_start": "0.0", "x_end": "1.0", "final_time": "0.1",
    "space_points": "31", "r": "0.25", "left_boundary": "0.0", "right_boundary": "0.0",
    "initial_condition": "sin_pi",
}
PRESETS = {
    "ftcs_example_11_10": {
        "label": "FTCS lesson: Example 11.10 — sine-wave cooling",
        "values": {**DEFAULTS, "space_points": "6", "r": "0.5", "final_time": "0.12"},
    },
    "btcs_stability": {
        "label": "BTCS lesson: Q2 — sine wave with r = 2",
        "values": {**DEFAULTS, "space_points": "6", "r": "2.0", "final_time": "0.5"},
    },
    "cn_verification": {
        "label": "Crank-Nicolson lesson: Q1 — sine-wave verification",
        "values": {**DEFAULTS, "space_points": "51", "r": "0.5", "final_time": "0.1"},
    },
}


def _value(query: dict[str, list[str]], name: str, defaults: dict[str, str]) -> str:
    return query.get(name, [defaults[name]])[0]


def _problem_from_query(query: dict[str, list[str]]) -> tuple[HeatEquationProblem, dict[str, str]]:
    preset_name = query.get("preset", [""])[0]
    preset_values = PRESETS.get(preset_name, {}).get("values", DEFAULTS)
    values = {name: _value(query, name, preset_values) for name in DEFAULTS}
    problem = HeatEquationProblem(
        alpha=float(values["alpha"]), x_start=float(values["x_start"]), x_end=float(values["x_end"]),
        final_time=float(values["final_time"]), space_points=int(values["space_points"]), r=float(values["r"]),
        left_boundary=float(values["left_boundary"]), right_boundary=float(values["right_boundary"]),
        initial_condition=values["initial_condition"],
    )
    problem.validate()
    if problem.space_points > 121:
        raise ValueError("Use 121 spatial points or fewer in the comparison workspace.")
    return problem, values


def _solve_all(problem: HeatEquationProblem):
    results = {}
    runtimes = {}
    for method in METHODS:
        started = perf_counter()
        results[method] = solve_heat_equation(problem, method)
        runtimes[method] = perf_counter() - started
    exact = None
    if (
        problem.initial_condition == "sin_pi" and problem.x_start == 0.0 and problem.x_end == 1.0
        and problem.left_boundary == 0.0 and problem.right_boundary == 0.0
    ):
        exact = exact_sine_solution(results["ftcs"], problem.alpha)
    return results, exact, runtimes


def _figures(results, exact, runtimes) -> tuple[str, str, str]:
    profile = go.Figure()
    error = go.Figure()
    for method in METHODS:
        result = results[method]
        profile.add_trace(go.Scatter(
            x=result.x, y=result.values[-1], mode="lines", name=METHOD_LABELS[method],
            line={"color": METHOD_COLORS[method], "width": 3},
            hovertemplate="x=%{x:.5f}<br>u=%{y:.6f}<extra></extra>",
        ))
        if exact is not None:
            history = np.max(np.abs(result.values - exact), axis=1)
            error.add_trace(go.Scatter(
                x=result.time, y=history, mode="lines", name=METHOD_LABELS[method],
                line={"color": METHOD_COLORS[method], "width": 3},
                hovertemplate="t=%{x:.6f}<br>max error=%{y:.3e}<extra></extra>",
            ))
    if exact is not None:
        profile.add_trace(go.Scatter(
            x=results["ftcs"].x, y=exact[-1], mode="lines", name="Exact reference",
            line={"color": "#182433", "width": 2, "dash": "dash"},
        ))
    profile.update_layout(
        title="Final-time profiles", xaxis_title="space x", yaxis_title="u(x, T)",
        height=420, margin={"l": 56, "r": 26, "t": 55, "b": 55}, legend={"orientation": "h", "y": -0.22},
    )
    error.update_layout(
        title="Maximum error over time", xaxis_title="time", yaxis_title="maximum absolute error",
        yaxis_type="log", height=420, margin={"l": 64, "r": 26, "t": 55, "b": 55}, legend={"orientation": "h", "y": -0.22},
    )
    if exact is None:
        error.add_annotation(text="No exact reference is available for this initial/boundary condition.", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)

    surface = go.Figure()
    for index, method in enumerate(METHODS):
        result = results[method]
        surface.add_trace(go.Surface(
            x=result.x, y=result.time, z=result.values, name=METHOD_LABELS[method], visible=index == 0,
            colorscale="Viridis", colorbar={"title": "u", "len": 0.7},
            hovertemplate="x=%{x:.5f}<br>t=%{y:.6f}<br>u=%{z:.6f}<extra></extra>",
        ))
    surface.update_layout(
        title="3D solution surface", height=760, margin={"l": 70, "r": 70, "t": 115, "b": 70},
        scene={
            "xaxis": {"title": {"text": "space x"}, "automargin": True},
            "yaxis": {"title": {"text": "time t"}, "automargin": True},
            "zaxis": {"title": {"text": "solution u(x, t)"}, "automargin": True},
            "aspectmode": "manual", "aspectratio": {"x": 1.45, "y": 1.2, "z": 0.8},
        },
        updatemenus=[{
            "type": "buttons", "direction": "right", "x": 0.0, "y": 1.08,
            "buttons": [
                {"label": METHOD_LABELS[method], "method": "update", "args": [{"visible": [candidate == method for candidate in METHODS]}, {"title": f"3D solution surface: {METHOD_LABELS[method]}"}]}
                for method in METHODS
            ],
        }],
    )
    return (
        profile.to_html(full_html=False, include_plotlyjs="cdn"),
        error.to_html(full_html=False, include_plotlyjs=False),
        surface.to_html(full_html=False, include_plotlyjs=False),
    )


def _metric_cards(results, exact, runtimes) -> str:
    cards = []
    for method in METHODS:
        result = results[method]
        error = max_absolute_error(result, exact) if exact is not None else None
        status = "Unstable mesh" if any("unstable" in item.lower() for item in result.warnings) else "Stable"
        error_text = f"{error:.2e}" if error is not None else "No reference"
        cards.append(f"<article class='metric'><div class='metric-top'><span class='dot {method}'></span><b>{METHOD_LABELS[method]}</b><span class='status'>{status}</span></div><div class='metric-row'><span>Max error</span><strong>{error_text}</strong></div><div class='metric-row'><span>Runtime</span><strong>{runtimes[method] * 1000:.2f} ms</strong></div></article>")
    return "".join(cards)


def _input(name: str, label: str, values: dict[str, str], step: str = "any") -> str:
    return f"<label>{label}<input name='{name}' value='{html.escape(values[name])}' type='number' step='{step}' required></label>"


def _page(query: dict[str, list[str]]) -> str:
    error_message = ""
    try:
        problem, values = _problem_from_query(query)
        results, exact, runtimes = _solve_all(problem)
        profile, error, surface = _figures(results, exact, runtimes)
        exact_note = "The dashed line is the exact sine-wave solution, used only for this compatible verification problem." if exact is not None else "No analytical reference is available for this problem, so compare stability and profiles rather than error."
        content = f"""
        <section class='summary'><div><p class='eyebrow'>Comparison result</p><h2>One PDE. Three numerical choices.</h2><p class='summary-copy'>All methods use the same equation, mesh, initial condition, and boundaries. The differences below come from the method, not changed inputs.</p></div><p class='exact-note'>{exact_note}</p></section>
        <section class='metrics'>{_metric_cards(results, exact, runtimes)}</section>
        <section class='chart-grid'><div class='chart'>{profile}</div><div class='chart'>{error}</div></section>
        <section class='reading'><h3>How to read this</h3><p><b>FTCS</b> is fast but needs \\(r \\le 0.5\\). <b>BTCS</b> stays stable with larger time steps but is first-order in time. <b>Crank-Nicolson</b> is stable and second-order in time; use the full-width 3D viewer to inspect how each method evolves.</p></section>
        <section class='surface-wrap'>{surface}</section>"""
    except (ValueError, TypeError) as exc:
        values = DEFAULTS
        content = f"<section class='error'><h2>Check the question details</h2><p>{html.escape(str(exc))}</p></section>"
        error_message = "Correct the highlighted values and run the comparison again."

    fieldset = "".join([
        _input("alpha", "Diffusivity alpha", values), _input("x_start", "Domain start", values),
        _input("x_end", "Domain end", values), _input("final_time", "Final time", values),
        _input("space_points", "Spatial points", values, "1"), _input("r", "Mesh ratio r", values),
        _input("left_boundary", "Left boundary", values), _input("right_boundary", "Right boundary", values),
    ])
    preset_options = "<option value=''>Start with a new question</option>" + "".join(
        f"<option value='{name}' {'selected' if query.get('preset', [''])[0] == name else ''}>{item['label']}</option>"
        for name, item in PRESETS.items()
    )
    condition_options = "".join(
        f"<option value='{name}' {'selected' if values['initial_condition'] == name else ''}>{label}</option>"
        for name, label in (("sin_pi", "Sine wave"), ("gaussian", "Gaussian pulse"), ("hot_center", "Hot centre"), ("zero", "Zero"))
    )
    preset_data = json.dumps({key: item["values"] for key, item in PRESETS.items()})
    return f"""<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>Compare Methods | PDE Solver</title><style>
    *{{box-sizing:border-box}}body{{margin:0;background:#f4f7f8;color:#16232d;font:15px/1.5 Inter,Arial,sans-serif}}a{{color:inherit}}.topbar{{height:64px;background:#102d3a;color:#fff;display:flex;align-items:center;justify-content:space-between;padding:0 max(22px,calc((100% - 1320px)/2))}}.brand{{font-weight:700;font-size:18px;text-decoration:none}}.nav{{display:flex;gap:18px;font-size:14px}}.nav a{{color:#d2e1e4;text-decoration:none}}.nav a.active{{color:#fff;border-bottom:2px solid #f18158;padding-bottom:4px}}main{{max-width:1320px;margin:auto;padding:30px 22px 64px}}.heading{{display:flex;justify-content:space-between;gap:22px;align-items:end;margin-bottom:24px}}h1{{font-size:30px;line-height:1.1;margin:0}}h2{{font-size:22px;line-height:1.2;margin:0}}h3{{font-size:16px;margin:0}}.heading p{{margin:7px 0 0;color:#5c6d76;max-width:720px}}.workspace{{display:grid;grid-template-columns:300px minmax(0,1fr);gap:24px;align-items:start}}form{{background:#fff;border:1px solid #d7e1e5;border-radius:6px;padding:18px;position:sticky;top:16px;max-height:calc(100vh - 32px);overflow:auto}}form h2{{font-size:18px}}form>p{{margin:4px 0 15px;color:#61727b;font-size:13px}}.form-group{{border-top:1px solid #e2e8ea;margin-top:15px;padding-top:13px}}.group-title{{font-size:11px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:#2a7280;margin-bottom:5px}}label{{display:block;color:#42545e;font-size:12px;font-weight:600;margin-top:10px}}input,select{{font:inherit;color:#16232d;width:100%;margin-top:4px;padding:8px;border:1px solid #b9c8ce;border-radius:4px;background:#fff}}button{{margin-top:18px;width:100%;padding:10px;border:0;border-radius:4px;background:#df633f;color:#fff;font:600 14px inherit;cursor:pointer}}.summary{{display:flex;justify-content:space-between;gap:24px;background:#fff;border-top:4px solid #df633f;padding:24px;margin-bottom:16px}}.eyebrow{{margin:0 0 4px;color:#2a7280;text-transform:uppercase;letter-spacing:.08em;font-size:11px;font-weight:700}}.summary-copy{{color:#53656f;max-width:620px;margin:8px 0 0}}.exact-note{{align-self:center;max-width:300px;border-left:3px solid #159570;background:#edf7f4;color:#31554a;padding:10px 12px;font-size:13px;margin:0}}.metrics{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:16px}}.metric{{background:#fff;border:1px solid #dce5e8;padding:16px}}.metric-top{{display:flex;align-items:center;gap:8px;border-bottom:1px solid #e5ebed;padding-bottom:10px;margin-bottom:8px}}.dot{{width:9px;height:9px;border-radius:50%;display:inline-block}}.dot.ftcs{{background:#dc6242}}.dot.btcs{{background:#2878b9}}.dot.crank_nicolson{{background:#159570}}.status{{margin-left:auto;color:#5b6c76;font-size:12px}}.metric-row{{display:flex;justify-content:space-between;margin-top:5px;color:#5b6c76;font-size:13px}}.metric-row strong{{color:#20303a;font-variant-numeric:tabular-nums}}.chart-grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}}.chart,.surface-wrap{{background:#fff;border:1px solid #dce5e8;overflow:hidden}}.reading{{border-left:3px solid #2878b9;padding:14px 16px;margin:16px 0;background:#edf5fa}}.reading p{{margin:6px 0 0;color:#3e5662}}.surface-wrap{{padding:4px 12px 12px}}.error{{background:#fff;border-top:4px solid #bf4141;padding:24px}}@media(max-width:920px){{.workspace{{grid-template-columns:1fr}}form{{position:static;max-height:none}}.metrics,.chart-grid{{grid-template-columns:1fr}}.summary{{display:block}}.exact-note{{max-width:none;margin-top:16px}}}}@media(max-width:560px){{.topbar{{padding:0 16px}}.nav{{gap:10px;font-size:12px}}main{{padding:22px 14px 44px}}.heading{{display:block}}h1{{font-size:26px}}}}
    </style><script>window.MathJax={{tex:{{inlineMath:[['\\\\(','\\\\)']]}}}};</script><script async src='https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js'></script></head><body><header class='topbar'><a class='brand' href='http://127.0.0.1:8500'>PDE Solver</a><nav class='nav'><a href='http://127.0.0.1:8500'>Home</a><a href='http://127.0.0.1:8501'>Solve</a><a class='active' href='http://127.0.0.1:8502'>Compare</a></nav></header><main><section class='heading'><div><h1>Compare numerical methods</h1><p>Enter a new heat-equation question or begin with an existing FTCS, BTCS, or Crank-Nicolson lesson problem. Every comparison uses exactly the same inputs across methods.</p></div></section><div class='workspace'><form method='get'><h2>Comparison question</h2><p>{error_message or 'Load a lesson setup, or enter a new question below.'}</p><div class='form-group'><div class='group-title'>Start point</div><label>Load a lesson problem<select name='preset' id='preset'>{preset_options}</select></label></div><div class='form-group'><div class='group-title'>Model and domain</div>{fieldset}<label>Initial condition<select name='initial_condition'>{condition_options}</select></label></div><button type='submit'>Run all three methods</button></form><div>{content}</div></div></main><script>const presets={preset_data};document.getElementById('preset').addEventListener('change',(event)=>{{const selected=event.target.value;if(!selected)return;const data=presets[selected];if(!data)return;Object.entries(data).forEach(([name,value])=>{{const input=document.querySelector(`[name="${{name}}"]`);if(input)input.value=value;}});}});</script></body></html>"""


class ComparisonHandler(BaseHTTPRequestHandler):
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
    server = ThreadingHTTPServer(("127.0.0.1", 8502), ComparisonHandler)
    print("PDE Solver comparison lab: http://127.0.0.1:8502")
    server.serve_forever()
