"""
question_config.py  —  Week 2: BTCS (Backward-Time Centered-Space)

Select the question you want to solve by setting ACTIVE_QUESTION at the
bottom of this file, then run 00_run_question_lab.py.

All problems are taken from Chapter 11 of:
  Jain, Iyengar & Jain — Numerical Methods for Scientific and Engineering
  Computation (6th ed.), Section 11.9 – 11.12.

Key difference from Week 1 (FTCS):
  BTCS is UNCONDITIONALLY STABLE — there is no upper limit on r = αk/h².
  Large time steps give physically correct (but less accurate) results.
  This week shows WHY the implicit approach unlocks larger time steps.
"""

# ---------------------------------------------------------------------------
# Helper — builds the question_text HTML string for a problem
# ---------------------------------------------------------------------------

def _q(title, _ref, pde_str, domain, ic_html, bc_html, extra_html=""):
    # _ref kept as parameter for call-site compatibility; displayed in the right panel
    return (
        f"<p><strong>{title}</strong></p>"
        f"<p><strong>PDE:</strong> \\({pde_str}\\)</p>"
        f"<p><strong>Domain:</strong> {domain}</p>"
        f"<p><strong>Initial condition:</strong> {ic_html}</p>"
        f"<p><strong>Boundary conditions:</strong> {bc_html}</p>"
        + (f"<p>{extra_html}</p>" if extra_html else "")
    )


QUESTION_BANK = {

    # ------------------------------------------------------------------
    # Q1  Textbook Example 11.9 (BTCS version)
    #     Non-uniform boundary conditions; classic steady-state test.
    #     FTCS would need r ≤ 0.5 (k ≤ 0.5). With BTCS we use r = 1 (k = 1).
    # ------------------------------------------------------------------
    1: {
        "title": "Q1 — Heat conduction with non-zero boundary temperatures (BTCS, r=1)",
        "reference": "Jain–Iyengar–Jain, Example 11.9 (adapted to BTCS)",
        "pde": "u_t = u_xx",

        "alpha":                  1,
        "x_start":                0,
        "x_end":                  5,
        "final_time":             5,
        "number_of_space_points": 6,   # h = 1

        "r":                      1,    # k = r·h²/α = 1 — BTCS handles this fine

        "left_boundary":          0,
        "right_boundary":         100,

        "initial_condition_type": "function",
        "initial_condition_expr": "20 * np.ones_like(x)",
        "initial_condition_display": "\\(U_i^0 = 20\\) (uniform 20°C interior)",

        "exact_solution_type":    None,

        "run_supporting_engines":     True,
        "generate_animation":         True,
        "open_dashboard_after_run":   True,

        "question_text": _q(
            "Q1 — Non-uniform Boundary Conditions",
            "Jain–Iyengar–Jain, Example 11.9 (BTCS adaptation)",
            r"u_t = u_{xx},\quad 0 \le x \le 5,\quad 0 < t",
            "\\(x \\in [0,5]\\), \\(t > 0\\)",
            "\\(u(x,0)=20\\) (uniform temperature)",
            "\\(u(0,t)=0\\), \\(u(5,t)=100\\)",
            "Use BTCS with \\(h=1\\), \\(r=1\\) (\\(k=1\\)). "
            "Compare with Crank-Nicolson answer from textbook: "
            "\\(u_1\\approx10.05\\), \\(u_2\\approx20.2\\), \\(u_3\\approx30.75\\), \\(u_4\\approx62.69\\) at \\(t=1\\)."
        ),
        "graph_notes": {
            "profiles": (
                "Observe how the profile evolves from uniform 20°C toward the "
                "steady-state linear profile \\(u(x)=20x\\). "
                "The left end is fixed at 0°C; the right end at 100°C."
            ),
            "rod":     "The rod starts uniformly warm, then cools from the left and warms from the right.",
            "surface": "Diagonal colour bands show heat flowing from the hot right end toward the cold left end.",
        },
    },

    # ------------------------------------------------------------------
    # Q2  Schmidt (FTCS) problem solved with BTCS at r = 2
    #     Identical setup to Week-1 Q1 but r doubled — FTCS would explode.
    # ------------------------------------------------------------------
    2: {
        "title": "Q2 — Sine IC, BTCS with r=2 (FTCS would be unstable)",
        "reference": "Jain–Iyengar–Jain, Example 11.10 (BTCS with r=2)",
        "pde": "u_t = u_xx",

        "alpha":                  1,
        "x_start":                0,
        "x_end":                  1,
        "final_time":             0.5,
        "number_of_space_points": 6,    # h = 0.2

        "r":                      2,    # k = 2·0.04 = 0.08 — unstable for FTCS

        "left_boundary":          0,
        "right_boundary":         0,

        "initial_condition_type": "sin_pi",

        "exact_solution_type":    "sin_pi_zero_boundary",

        "run_supporting_engines":     True,
        "generate_animation":         True,
        "open_dashboard_after_run":   True,

        "question_text": _q(
            "Q2 — BTCS Stability Demonstration (r=2)",
            "Jain–Iyengar–Jain, Example 11.10 (modified for BTCS)",
            r"u_t = u_{xx},\quad 0 \le x \le 1",
            "\\(x \\in [0,1]\\)",
            "\\(u(x,0)=\\sin(\\pi x)\\)",
            "\\(u(0,t)=u(1,t)=0\\)",
            "\\(r=2\\) — four times the FTCS stability limit. "
            "BTCS still produces a smooth, decaying solution. "
            "The exact solution is \\(u(x,t)=e^{-\\pi^2 t}\\sin(\\pi x)\\). "
            "Notice that the BTCS error is larger than at \\(r=0.5\\) — bigger time steps "
            "reduce temporal accuracy, but there is no blow-up."
        ),
        "graph_notes": {
            "profiles": (
                "Dashed lines are the exact solution. BTCS at r=2 is less accurate "
                "than FTCS at r=0.5 but remains stable. The error grows with r."
            ),
            "rod":     "The rod cools smoothly — no oscillations despite large r.",
            "surface": "Clean smooth surface confirms unconditional stability of BTCS.",
        },
    },

    # ------------------------------------------------------------------
    # Q3  Textbook Example 11.11 — non-trivial IC, BTCS at r = 1
    # ------------------------------------------------------------------
    3: {
        "title": "Q3 — Non-uniform IC: u_t=4u_xx, parabolic initial profile (BTCS, r=1)",
        "reference": "Jain–Iyengar–Jain, Example 11.11 (BTCS adaptation)",
        "pde": "u_t = 4 u_xx",

        "alpha":                  4,
        "x_start":                0,
        "x_end":                  8,
        "final_time":             1,
        "number_of_space_points": 9,    # h = 1

        "r":                      1,    # k = r·h²/α = 1/4 = 0.25  (FTCS needed r≤0.5)

        "left_boundary":          0,
        "right_boundary":         0,

        "initial_condition_type": "function",
        "initial_condition_expr": "4*x - 0.5*x**2",
        "initial_condition_display": "\\(U_i^0 = 4x - \\tfrac{1}{2}x^2\\)",

        "exact_solution_type":    None,

        "run_supporting_engines":     True,
        "generate_animation":         True,
        "open_dashboard_after_run":   True,

        "question_text": _q(
            "Q3 — Parabolic Initial Temperature Profile",
            "Jain–Iyengar–Jain, Example 11.11 (BTCS with r=1)",
            r"u_t = 4u_{xx},\quad 0 \le x \le 8",
            "\\(x \\in [0,8]\\), h=1",
            "\\(u(x,0) = 4x - \\tfrac{1}{2}x^2\\)",
            "\\(u(0,t)=0\\), \\(u(8,t)=0\\)",
            "BTCS uses \\(k=0.25\\) (\\(r=1\\)). "
            "FTCS on this problem requires \\(k \\le 0.125\\) (\\(r \\le 0.5\\)). "
            "BTCS reaches the same physical time with half the number of time steps."
        ),
        "graph_notes": {
            "profiles": "The parabolic hump flattens and decays symmetrically toward zero at both ends.",
            "rod":      "Colour flows show the peak cooling as heat diffuses rapidly (α=4).",
            "surface":  "The high-α value causes fast decay — the surface drops steeply with time.",
        },
    },

    # ------------------------------------------------------------------
    # Q4  Exercise 11.4 Q1 — u_xx = 2u_t
    # ------------------------------------------------------------------
    4: {
        "title": "Q4 — Exercise 11.4 Q1: u_xx = 2u_t, quadratic IC (BTCS, r=1)",
        "reference": "Jain–Iyengar–Jain, Exercise 11.4, Q1",
        "pde": "u_t = 0.5 u_xx",

        "alpha":                  0.5,
        "x_start":                0,
        "x_end":                  4,
        "final_time":             6,
        "number_of_space_points": 5,    # h = 1

        "r":                      1,    # k = r·h²/α = 1·1/0.5 = 2

        "left_boundary":          0,
        "right_boundary":         0,

        "initial_condition_type": "function",
        "initial_condition_expr": "x * (4 - x)",
        "initial_condition_display": "\\(U_i^0 = x(4-x)\\)",

        "exact_solution_type":    None,

        "run_supporting_engines":     True,
        "generate_animation":         True,
        "open_dashboard_after_run":   True,

        "question_text": _q(
            "Q4 — Exercise 11.4 Q1 (BTCS)",
            "Jain–Iyengar–Jain, Exercise 11.4, Problem 1",
            r"u_{xx} = 2u_t \;\Leftrightarrow\; u_t = \tfrac{1}{2}u_{xx}",
            "\\(x \\in [0,4]\\), h=1",
            "\\(u(x,0)=x(4-x)\\)",
            "\\(u(0,t)=u(4,t)=0\\)",
            "BTCS with \\(r=1\\) gives \\(k=2\\) (time step of 2 units). "
            "FTCS with \\(r=0.25\\) needed \\(k=0.5\\). "
            "BTCS covers the same physical time in far fewer linear-system solves. "
            "Note: larger \\(k\\) means less temporal resolution but no instability."
        ),
        "graph_notes": {
            "profiles": "The quadratic arch gradually flattens toward zero as heat dissipates through the boundaries.",
            "rod":      "The central hotspot cools steadily. Slow diffusion (α=0.5) keeps the rod warm longer.",
            "surface":  "Gentle slope due to small α=0.5 — heat diffuses at half the standard rate.",
        },
    },

    # ------------------------------------------------------------------
    # Q5  Exercise 11.4 Q2 — u_t = u_xx, x(1-x) IC
    # ------------------------------------------------------------------
    5: {
        "title": "Q5 — Exercise 11.4 Q2: u_t=u_xx, x(1-x) initial condition (BTCS, r=1)",
        "reference": "Jain–Iyengar–Jain, Exercise 11.4, Q2",
        "pde": "u_t = u_xx",

        "alpha":                  1,
        "x_start":                0,
        "x_end":                  1,
        "final_time":             0.1,
        "number_of_space_points": 11,   # h = 0.1

        "r":                      1,    # k = 1·0.01 = 0.01

        "left_boundary":          0,
        "right_boundary":         0,

        "initial_condition_type": "function",
        "initial_condition_expr": "x * (1 - x)",
        "initial_condition_display": "\\(U_i^0 = x(1-x)\\)",

        "exact_solution_type":    None,

        "run_supporting_engines":     True,
        "generate_animation":         True,
        "open_dashboard_after_run":   True,

        "question_text": _q(
            "Q5 — Exercise 11.4 Q2 (BTCS)",
            "Jain–Iyengar–Jain, Exercise 11.4, Problem 2",
            r"u_t = u_{xx},\quad 0 \le x \le 1",
            "\\(x \\in [0,1]\\), h=0.1",
            "\\(u(x,0)=x(1-x)\\)",
            "\\(u(0,t)=u(1,t)=0\\)",
            "Tabulate for \\(t=k,\\,2k,\\,3k\\) using BTCS with \\(r=1\\) (\\(k=0.01\\)). "
            "With FTCS at \\(r=0.25\\), \\(k=0.0025\\) — four times smaller. "
            "BTCS computes 10 steps versus 40 steps for FTCS to reach the same final time."
        ),
        "graph_notes": {
            "profiles": "The symmetric parabolic arch decays uniformly toward the boundaries.",
            "rod":      "Heat disperses evenly from the hump toward both cold ends.",
            "surface":  "Smooth symmetric decay surface — confirming BTCS unconditional stability.",
        },
    },

    # ------------------------------------------------------------------
    # Q6  Exercise 11.4 Q3 — u_t = u_xx, x²(25-x²) IC
    # ------------------------------------------------------------------
    6: {
        "title": "Q6 — Exercise 11.4 Q3: u_t=u_xx, x²(25-x²) initial profile (BTCS, r=1)",
        "reference": "Jain–Iyengar–Jain, Exercise 11.4, Q3",
        "pde": "u_t = u_xx",

        "alpha":                  1,
        "x_start":                0,
        "x_end":                  5,
        "final_time":             5,
        "number_of_space_points": 6,    # h = 1

        "r":                      1,    # k = 1 (vs r=0.5 → k=0.5 for FTCS)

        "left_boundary":          0,
        "right_boundary":         0,

        "initial_condition_type": "function",
        "initial_condition_expr": "x**2 * (25 - x**2)",
        "initial_condition_display": "\\(U_i^0 = x^2(25 - x^2)\\)",

        "exact_solution_type":    None,

        "run_supporting_engines":     True,
        "generate_animation":         True,
        "open_dashboard_after_run":   True,

        "question_text": _q(
            "Q6 — Exercise 11.4 Q3 (BTCS)",
            "Jain–Iyengar–Jain, Exercise 11.4, Problem 3",
            r"u_t = u_{xx},\quad 0 \le x \le 5",
            "\\(x \\in [0,5]\\), h=1",
            "\\(u(x,0)=x^2(25-x^2)\\)",
            "\\(u(0,t)=u(5,t)=0\\)",
            "Solve with BTCS (\\(r=1\\), \\(k=1\\)). "
            "The textbook uses the explicit method (\\(r=0.5\\), \\(k=0.5\\)). "
            "BTCS doubles the time step with no stability penalty."
        ),
        "graph_notes": {
            "profiles": "Large initial values near x=3 to x=4 decay and spread toward the zero boundaries.",
            "rod":      "The asymmetric temperature peak (higher near the right) gradually levels off.",
            "surface":  "Non-symmetric initial profile relaxes to zero — note how the peak migrates slightly leftward.",
        },
    },

    # ------------------------------------------------------------------
    # Q7  Exercise 11.4 Q5 — triangular IC (Bendre-Schmidt / BTCS)
    # ------------------------------------------------------------------
    7: {
        "title": "Q7 — Exercise 11.4 Q5: triangular initial condition (BTCS, r=0.5)",
        "reference": "Jain–Iyengar–Jain, Exercise 11.4, Q5",
        "pde": "u_t = u_xx",

        "alpha":                  1,
        "x_start":                0,
        "x_end":                  1,
        "final_time":             0.25,
        "number_of_space_points": 5,    # h = 0.25

        "r":                      0.5,  # k = 0.5·0.0625 = 0.03125

        "left_boundary":          0,
        "right_boundary":         0,

        "initial_condition_type": "function",
        "initial_condition_expr": "np.where(x <= 0.5, 2*x, 2*(1-x))",
        "initial_condition_display": (
            "\\(U_i^0 = \\begin{cases}2x & 0 \\le x \\le 0.5\\\\"
            "2(1-x) & 0.5 < x \\le 1\\end{cases}\\)"
        ),

        "exact_solution_type":    None,

        "run_supporting_engines":     True,
        "generate_animation":         True,
        "open_dashboard_after_run":   True,

        "question_text": _q(
            "Q7 — Exercise 11.4 Q5 (Triangular IC, BTCS)",
            "Jain–Iyengar–Jain, Exercise 11.4, Problem 5",
            r"u_t = u_{xx},\quad 0 \le x \le 1",
            "\\(x \\in [0,1]\\), h=0.25",
            ("\\(u(x,0)=\\begin{cases}2x &amp; x\\le 0.5\\\\2(1-x) &amp; x>0.5\\end{cases}\\)"),
            "\\(u(0,t)=u(1,t)=0\\)",
            "The sharp triangular peak smooths rapidly. "
            "BTCS handles the discontinuity in the IC derivative without instability. "
            "Compare with Week-1 FTCS solution to see how both methods capture the smoothing."
        ),
        "graph_notes": {
            "profiles": "The sharp triangular peak rounds off rapidly — BTCS diffuses it slightly more than FTCS at the same r.",
            "rod":      "A hot spike at the rod centre cools outward — watch the symmetric spread.",
            "surface":  "The triangular ridge quickly becomes a smooth Gaussian-like hill.",
        },
    },

    # ------------------------------------------------------------------
    # Q8  Exact-solution verification — large r
    # ------------------------------------------------------------------
    8: {
        "title": "Q8 — Exact-solution check with large time step (BTCS, r=4)",
        "reference": "Custom verification problem",
        "pde": "u_t = u_xx",

        "alpha":                  1,
        "x_start":                0,
        "x_end":                  1,
        "final_time":             0.5,
        "number_of_space_points": 11,   # h = 0.1

        "r":                      4,    # k = 4·0.01 = 0.04 — 8× FTCS limit

        "left_boundary":          0,
        "right_boundary":         0,

        "initial_condition_type": "sin_pi",

        "exact_solution_type":    "sin_pi_zero_boundary",

        "run_supporting_engines":     True,
        "generate_animation":         True,
        "open_dashboard_after_run":   True,

        "question_text": _q(
            "Q8 — BTCS Accuracy at r=4 (Exact Solution Available)",
            "Custom verification (sine IC with known exact solution)",
            r"u_t = u_{xx},\quad 0 \le x \le 1",
            "\\(x \\in [0,1]\\)",
            "\\(u(x,0)=\\sin(\\pi x)\\)",
            "\\(u(0,t)=u(1,t)=0\\)",
            "Exact solution: \\(u(x,t)=e^{-\\pi^2 t}\\sin(\\pi x)\\). "
            "BTCS with \\(r=4\\) (\\(k=0.04\\)) — 8× the FTCS stability limit. "
            "Use the error panel to see how temporal accuracy degrades with large \\(r\\), "
            "while stability is maintained. Compare errors with Q2 (r=2) and Q5 (r=1)."
        ),
        "graph_notes": {
            "profiles": (
                "Dashed = exact. Larger r means greater temporal truncation error. "
                "The numerical solution decays slightly faster than the exact — characteristic of BTCS."
            ),
            "rod":     "BTCS overdamps at large r — the rod cools slightly faster than reality.",
            "surface": "Compare with Q2 (r=2) surface: larger r gives more smoothing per time step.",
        },
    },

    # ------------------------------------------------------------------
    # Q9  User-defined — hot-centre pulse
    # ------------------------------------------------------------------
    9: {
        "title": "Q9 — Hot centre pulse (user exploration slot)",
        "reference": "Custom problem",
        "pde": "u_t = u_xx",

        "alpha":                  1,
        "x_start":                0,
        "x_end":                  1,
        "final_time":             0.3,
        "number_of_space_points": 11,

        "r":                      1,

        "left_boundary":          0,
        "right_boundary":         0,

        "initial_condition_type": "hot_center",
        "u_value":                100,
        "u_left":                 0.4,
        "u_right":                0.6,

        "exact_solution_type":    None,

        "run_supporting_engines":     True,
        "generate_animation":         True,
        "open_dashboard_after_run":   True,

        "question_text": (
            "<p><strong>Q9 — Hot Centre Pulse (Exploration)</strong></p>"
            "<p>A short central segment of the rod is heated to 100°C. All other regions start at 0°C.</p>"
            "<p>Observe how the heat pulse spreads outward and eventually cools to zero at both ends.</p>"
            "<p>Try changing <code>u_left</code>, <code>u_right</code>, and <code>r</code> in "
            "<code>question_config.py</code> to explore different scenarios.</p>"
        ),
        "graph_notes": {
            "profiles": "The rectangular temperature pulse spreads outward in both directions, becoming Gaussian-like.",
            "rod":      "The bright central band spreads and dims — classical heat diffusion from a localised source.",
            "surface":  "A ridge in the centre spreading outward in a V-shape characteristic of diffusion.",
        },
    },

    # ------------------------------------------------------------------
    # Q10  BTCS stability demo — r = 5  (FTCS would be catastrophically unstable)
    # ------------------------------------------------------------------
    10: {
        "title": "Q10 — BTCS stability demo: r=5 (FTCS would completely explode)",
        "reference": "BTCS unconditional stability demonstration",
        "pde": "u_t = u_xx",

        "alpha":                  1,
        "x_start":                0,
        "x_end":                  1,
        "final_time":             1.0,
        "number_of_space_points": 6,    # h = 0.2

        "r":                      5,    # k = 5·0.04 = 0.2 — 10× FTCS limit

        "left_boundary":          0,
        "right_boundary":         0,

        "initial_condition_type": "sin_pi",

        "exact_solution_type":    "sin_pi_zero_boundary",

        "run_supporting_engines":     True,
        "generate_animation":         True,
        "open_dashboard_after_run":   True,

        "question_text": (
            "<p><strong>Q10 — Unconditional Stability at r=5</strong></p>"
            "<p>FTCS with \\(r=5\\) would immediately produce oscillations that grow "
            "without bound, making the solution completely useless.</p>"
            "<p>BTCS with \\(r=5\\) produces a physically correct, smoothly decaying solution "
            "— though with more temporal error than at \\(r=0.5\\).</p>"
            "<p>\\(r=5\\) means \\(k=0.20\\): each time step jumps 0.20 time units. "
            "Compare the exact solution error here against Q2 (r=2) and Q8 (r=4).</p>"
            "<p><strong>Key insight:</strong> The implicit system \\(A\\,U^{n+1}=U^n\\) "
            "is always solvable. The matrix \\(A\\) is diagonally dominant for all \\(r&gt;0\\), "
            "so no r-value can make the system singular or numerically unstable.</p>"
        ),
        "graph_notes": {
            "profiles": (
                "Compare with Week-1 Q10 (r=0.6, FTCS) which shows oscillations. "
                "Here BTCS at r=5 gives a monotonically decaying profile — no oscillations."
            ),
            "rod":     "Smooth colour progression from hot to cool — unconditional stability in action.",
            "surface": "Surface is smooth despite extreme r. Temporal resolution is coarse but physically correct.",
        },
    },

    # ------------------------------------------------------------------
    # Q11  Very large r — accuracy vs efficiency tradeoff
    # ------------------------------------------------------------------
    11: {
        "title": "Q11 — Extreme step size: r=20 (accuracy vs efficiency tradeoff)",
        "reference": "BTCS large time-step accuracy study",
        "pde": "u_t = u_xx",

        "alpha":                  1,
        "x_start":                0,
        "x_end":                  1,
        "final_time":             2.0,
        "number_of_space_points": 6,    # h = 0.2

        "r":                      20,   # k = 20·0.04 = 0.8 — 40× FTCS limit

        "left_boundary":          0,
        "right_boundary":         0,

        "initial_condition_type": "sin_pi",

        "exact_solution_type":    "sin_pi_zero_boundary",

        "run_supporting_engines":     True,
        "generate_animation":         True,
        "open_dashboard_after_run":   True,

        "question_text": (
            "<p><strong>Q11 — Accuracy–Efficiency Tradeoff at r=20</strong></p>"
            "<p>\\(r=20\\) means \\(k=0.8\\): the solver covers 2 seconds of physical time "
            "in just 3 time steps. FTCS would need 200 steps for the same final time at \\(r=0.5\\).</p>"
            "<p>Inspect the error panel to see how much temporal accuracy is lost. "
            "This illustrates the fundamental tradeoff: BTCS is unconditionally stable, "
            "but accuracy still requires reasonably small \\(k\\).</p>"
            "<p><strong>Rule of thumb:</strong> use \\(r \\le 1\\) for good accuracy, "
            "\\(r \\le 5\\) when efficiency matters more than precision, "
            "and \\(r \\gg 1\\) only for qualitative trends.</p>"
        ),
        "graph_notes": {
            "profiles": (
                "Exact solution shown dashed. At r=20 the BTCS severely overdamps — "
                "the numerical solution decays much faster than reality."
            ),
            "rod":     "At r=20 the rod appears to cool almost instantly — exaggerated overdamping.",
            "surface": "Only a handful of time levels visible — showing both the coarseness and the stability.",
        },
    },
}


# ---------------------------------------------------------------------------
# Active question
# ---------------------------------------------------------------------------

ACTIVE_QUESTION = 3   # Change this to run a different question (1–11)
QUESTION = QUESTION_BANK[ACTIVE_QUESTION]


def main():
    import subprocess
    import sys
    from pathlib import Path

    module_dir  = Path(__file__).resolve().parent
    runner_path = module_dir / "00_run_question_lab.py"
    completed   = subprocess.run([sys.executable, str(runner_path)], cwd=module_dir)
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()