"""Local landing page for the PDE Solver workspaces."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def home_page() -> bytes:
    return b"""<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>PDE Solver</title><style>
*{box-sizing:border-box}body{margin:0;background:#f4f7f8;color:#16232d;font:16px/1.55 Inter,Arial,sans-serif}a{color:inherit}.topbar{height:64px;background:#102d3a;color:#fff;display:flex;align-items:center;justify-content:space-between;padding:0 max(22px,calc((100% - 1160px)/2))}.brand{font-size:18px;font-weight:700;text-decoration:none}.nav{display:flex;gap:18px}.nav a{font-size:14px;color:#d1e0e3;text-decoration:none}.nav a.active{color:#fff;border-bottom:2px solid #f18158;padding-bottom:4px}main{max-width:1160px;margin:auto;padding:58px 22px 72px}.intro{max-width:760px;border-left:5px solid #df633f;padding-left:20px}.eyebrow{margin:0;color:#2a7280;font-size:12px;font-weight:700;letter-spacing:.08em;text-transform:uppercase}.intro h1{font-size:42px;line-height:1.05;letter-spacing:0;margin:8px 0 13px}.intro p:last-child{color:#55666f;margin:0;font-size:18px}.routes{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px;margin-top:42px}.route{display:flex;flex-direction:column;align-items:flex-start;background:#fff;border:1px solid #dce5e8;border-top:4px solid #2878b9;padding:25px;min-height:270px}.route.compare{border-top-color:#159570}.route h2{font-size:22px;margin:0}.route p{color:#52646e;margin:9px 0 14px}.route ul{padding-left:20px;color:#40535e;margin:0 0 22px;font-size:14px}.route li+li{margin-top:4px}.button{display:inline-block;margin-top:auto;border-radius:4px;background:#df633f;color:#fff;padding:10px 14px;text-decoration:none;font-weight:700;font-size:14px}.compare .button{background:#159570}.footnote{margin:34px 0 0;color:#60717a;font-size:14px}@media(max-width:720px){main{padding:42px 18px}.intro h1{font-size:32px}.routes{grid-template-columns:1fr}.topbar{padding:0 18px}.nav{gap:12px}.nav a{font-size:13px}}</style></head><body><header class='topbar'><a class='brand' href='http://127.0.0.1:8500'>PDE Solver</a><nav class='nav'><a class='active' href='http://127.0.0.1:8500'>Home</a><a href='http://127.0.0.1:8501'>Solve</a><a href='http://127.0.0.1:8502'>Compare</a></nav></header><main><section class='intro'><p class='eyebrow'>Numerical methods laboratory</p><h1>Build intuition. Check results. Choose a solver.</h1><p>Start with a single one-dimensional heat-equation question, or compare every available numerical method on exactly the same problem.</p></section><section class='routes'><article class='route'><h2>Solve a PDE</h2><p>Work through one configured heat-equation problem from inputs to computed values, method steps, and plots.</p><ul><li>Select FTCS, BTCS, or Crank-Nicolson</li><li>Use either delta x / delta t or grid-point input</li><li>Inspect the full time-level solution table</li></ul><a class='button' href='http://127.0.0.1:8501'>Open solver workspace</a></article><article class='route compare'><h2>Compare methods</h2><p>Enter a fresh question or load a lesson problem, then evaluate FTCS, BTCS, and Crank-Nicolson side by side.</p><ul><li>Same model and mesh across every method</li><li>Error history, runtime, and final profile</li><li>Full-width 3D solution viewer for each method</li></ul><a class='button' href='http://127.0.0.1:8502'>Open comparison workspace</a></article></section><p class='footnote'>Current scope: the one-dimensional heat equation. The solver is designed to grow toward more equations and numerical families over time.</p></main></body></html>"""


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
