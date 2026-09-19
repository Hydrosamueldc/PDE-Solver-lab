"""Local landing page for the 1D heat-equation PDE solver."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def home_page() -> bytes:
    return b"""<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>PDE Solver</title><style>
*{box-sizing:border-box}body{margin:0;background:#f5f7fa;color:#172033;font:16px/1.5 Arial,sans-serif}header{background:#143642;color:#fff;border-bottom:4px solid #d9603d;padding:34px max(22px,calc((100% - 1040px)/2))}h1,h2,p{margin:0}h1{font-size:32px}header p{color:#d7e5e6;margin-top:6px}main{max-width:1040px;margin:auto;padding:34px 22px}h2{font-size:22px;margin-bottom:6px}.lead{color:#526175;margin-bottom:22px}.routes{display:grid;grid-template-columns:repeat(2,1fr);gap:18px}.route{background:#fff;border-top:3px solid #235789;padding:22px;min-height:238px}.route.compare{border-top-color:#2a7a62}.route p{color:#526175;margin:10px 0 18px}.route ul{padding-left:18px;color:#435268;font-size:14px}.button{display:inline-block;background:#d9603d;color:#fff;text-decoration:none;padding:10px 14px;border-radius:4px;font-weight:700}.compare .button{background:#2a7a62}.note{margin-top:28px;border-left:3px solid #d9603d;padding:12px;background:#fff8f3;color:#435268}@media(max-width:700px){.routes{grid-template-columns:1fr}header{padding:26px 22px}}</style></head><body><header><h1>PDE Solver</h1><p>1D heat equation numerical laboratory</p></header><main><h2>Choose a workspace</h2><p class='lead'>Solve one configured problem, or compare every available numerical method on the same model.</p><section class='routes'><article class='route'><h2>Solve a PDE</h2><p>Choose FTCS, BTCS, or Crank-Nicolson. Enter the physical conditions and mesh data, then inspect each time-level value and its solution graphs.</p><ul><li>Method-specific finite-difference steps</li><li>Delta x / delta t input option</li><li>Full computed-value table</li></ul><a class='button' href='http://127.0.0.1:8501'>Open solver lab</a></article><article class='route compare'><h2>Compare methods</h2><p>Run FTCS, BTCS, and Crank-Nicolson on one shared problem to see accuracy, stability, runtime, and their 3D solution surfaces.</p><ul><li>Final profile versus exact solution</li><li>Error history over time</li><li>Method-selectable 3D surface</li></ul><a class='button' href='http://127.0.0.1:8001/interactive_comparison.html'>Open comparison lab</a></article></section><p class='note'>The comparison workspace uses one shared problem so differences come from the numerical method, not from changed parameters.</p></main></body></html>"""


class HomeHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        page = home_page()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(page)))
        self.end_headers()
        self.wfile.write(page)

    def log_message(self, format: str, *args: object) -> None:
        return


if __name__ == "__main__":
    server = ThreadingHTTPServer(("127.0.0.1", 8500), HomeHandler)
    print("PDE Solver home: http://127.0.0.1:8500")
    server.serve_forever()
