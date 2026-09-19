"""
question_config.py

To run a question, change the number on this one line:

    ACTIVE_QUESTION = 1          <-- set to any number from 1 to 11

Then run:

    python question_config.py

Questions 1-7  are taken directly from the textbook (Chapter 11).
Question 8     is a custom verification case with a known exact solution.
Question 9     is your own blank slot.
Questions 10-11 are intentionally UNSTABLE (r > 0.5) — demonstration only.
"""


QUESTION_BANK = {

    # -------------------------------------------------------------------
    # 1  Example 11.10
    # -------------------------------------------------------------------
    1: {
        "title": "Example 11.10",
        "reference": (
            "Numerical Methods in Engineering and Science — "
            "Jain, Iyengar & Jain, Ch. 11, Example 11.10"
        ),
        "pde": "u_t = u_xx",
        "alpha": 1.0,
        "x_start": 0.0,
        "x_end":   1.0,
        "final_time": 0.12,
        "number_of_space_points": 6,
        "r": 0.5,
        "time_step": None,
        "left_boundary":  0.0,
        "right_boundary": 0.0,
        "initial_condition_type": "sin_pi",
        "exact_solution_type": "sin_pi_zero_boundary",
        "run_supporting_engines":   True,
        "generate_animation":       True,
        "open_dashboard_after_run": True,

        "question_text": (
            "<p><strong>Example 11.10.</strong>&ensp;"
            "Solve the one-dimensional heat equation"
            "\\[u_t = u_{xx}, \\quad 0 < x < 1,\\quad t > 0\\]"
            "subject to the boundary conditions \\(u(0,t)=0\\) and \\(u(1,t)=0\\), "
            "with initial condition"
            "\\[u(x,0) = \\sin(\\pi x).\\]"
            "Apply the Schmidt explicit (FTCS) method with \\(h = 0.2\\) "
            "(six grid points) and stability ratio "
            "\\(r = \\dfrac{\\alpha\\,k}{h^2} = 0.5\\). "
            "Advance to \\(t = 0.12\\) and compare with the exact solution"
            "\\[U(x,t) = e^{-\\pi^2 t}\\sin(\\pi x).\\]</p>"
        ),
        "graph_notes": {
            "profiles": (
                "Each curve shows the temperature across the rod at a different moment in time. "
                "The tallest curve is the starting condition \\(\\sin(\\pi x)\\) — a smooth arch peaking at 1. "
                "Every later curve is the same arch shape but shorter, showing the rod cooling evenly. "
                "The dashed lines show the exact mathematical answer; they nearly overlap the solid FTCS lines, confirming the method is accurate here."
            ),
            "rod": (
                "The rod starts with a smooth warm glow at its centre — brightest at \\(x = 0.5\\) "
                "and completely cold at both ends, matching the \\(\\sin(\\pi x)\\) shape. "
                "Press Play and watch the glow fade evenly. "
                "The hottest spot stays at the centre throughout; it just dims uniformly over time."
            ),
            "surface": (
                "This shows the full space-time history as a 3D shape. "
                "The tall arch at the front edge is the starting temperature; the surface drops away as time moves to the right. "
                "Every cross-section from front to back is the same arch, just smaller — confirming the rod cools without changing its overall shape. "
                "Drag to rotate."
            ),
        },
    },

    # -------------------------------------------------------------------
    # 2  Example 11.12(a)
    # -------------------------------------------------------------------
    2: {
        "title": "Example 11.12(a)",
        "reference": (
            "Numerical Methods in Engineering and Science — "
            "Jain, Iyengar & Jain, Ch. 11, Example 11.12(a)"
        ),
        "pde": "u_t = u_xx",
        "alpha": 1.0,
        "x_start": 0.0,
        "x_end":   1.0,
        "final_time": 0.10,
        "number_of_space_points": 4,
        "r": 0.25,
        "time_step": None,
        "left_boundary":  0.0,
        "right_boundary": 0.0,
        "initial_condition_type": "sin_pi",
        "exact_solution_type": "sin_pi_zero_boundary",
        "run_supporting_engines":   True,
        "generate_animation":       True,
        "open_dashboard_after_run": True,

        "question_text": (
            "<p><strong>Example 11.12(a).</strong>&ensp;"
            "Solve"
            "\\[u_t = u_{xx}, \\quad 0 < x < 1,\\quad t > 0\\]"
            "with \\(u(0,t)=0\\), \\(u(1,t)=0\\) and initial condition "
            "\\(u(x,0)=\\sin(\\pi x)\\). "
            "Use \\(h=\\tfrac{1}{3}\\) (four grid points) and \\(r=0.25\\), "
            "giving \\(k \\approx 0.02\\overline{7}\\). "
            "Compare with the exact solution \\(U(x,t)=e^{-\\pi^2 t}\\sin(\\pi x)\\).</p>"
            "<p>Compare with Question 1 (same IC but \\(h=0.2\\), \\(r=0.5\\)): "
            "a coarser mesh with a smaller \\(r\\) — more time steps, fewer space points.</p>"
        ),
        "graph_notes": {
            "profiles": (
                "Same starting condition \\(\\sin(\\pi x)\\) as Question 1, "
                "but the calculation is done at only 4 grid points (just 2 interior values per time step). "
                "The curves are less detailed than Question 1 because of the coarser mesh. "
                "The exact solution (dashed) and FTCS result (solid) should still be close — the method works, just less precisely."
            ),
            "rod": (
                "With only 4 grid points, you can see the temperature at just a few locations. "
                "The cooling behaviour is still correct — the rod dims from centre outward — "
                "but the very coarse grid means you miss the fine detail of the temperature profile."
            ),
            "surface": (
                "The coarse mesh makes a blocky-looking surface with only 4 columns. "
                "Despite this, the surface drops correctly over time, showing the cooling process. "
                "Compare with Question 1's smoother surface to see the direct effect of mesh refinement."
            ),
        },
    },

    # -------------------------------------------------------------------
    # 3  Example 11.11
    # -------------------------------------------------------------------
    3: {
        "title": "Example 11.11",
        "reference": (
            "Numerical Methods in Engineering and Science — "
            "Jain, Iyengar & Jain, Ch. 11, Example 11.11"
        ),
        "pde": "u_t = 4 u_xx",
        "alpha": 4.0,
        "x_start": 0.0,
        "x_end":   8.0,
        "final_time": 0.625,
        "number_of_space_points": 9,
        "r": 0.5,
        "time_step": None,
        "left_boundary":  0.0,
        "right_boundary": 0.0,
        "initial_condition_type": "function",
        "initial_condition_expr": "4*x - 0.5*x**2",
        "initial_condition_display": "\\(U_i^0 = 4x - \\tfrac{x^2}{2}\\)",
        "exact_solution_type": None,
        "run_supporting_engines":   True,
        "generate_animation":       True,
        "open_dashboard_after_run": True,

        "question_text": (
            "<p><strong>Example 11.11.</strong>&ensp;"
            "Solve the heat equation with diffusivity \\(\\alpha = 4\\)"
            "\\[u_t = 4\\,u_{xx}, \\quad 0 < x < 8,\\quad t > 0\\]"
            "with \\(u(0,t)=0\\), \\(u(8,t)=0\\) and initial condition"
            "\\[u(x,0) = 4x - \\tfrac{x^2}{2}.\\]"
            "Use \\(h=1\\) (nine grid points) and \\(r=0.5\\), "
            "giving \\(k=0.125\\). Advance to \\(t=0.625\\) (five time steps).</p>"
            "<p>Note: \\(\\alpha=4\\) means heat spreads four times faster than the "
            "standard case \\(\\alpha=1\\). "
            "The initial condition \\(4x - x^2/2\\) is a downward-curving shape "
            "that peaks near \\(x=4\\).</p>"
        ),
        "graph_notes": {
            "profiles": (
                "The starting temperature \\(4x - x^2/2\\) is highest near \\(x=4\\). "
                "With \\(\\alpha=4\\), heat spreads four times faster than the standard case — "
                "notice how the profiles flatten dramatically in just 5 time steps. "
                "The rod loses its shape quickly because high diffusivity drives rapid heat flow toward the cold ends."
            ),
            "rod": (
                "The rod starts hottest toward its right end (near \\(x=4\\)). "
                "Because \\(\\alpha=4\\) (high diffusivity), heat floods toward the cold boundaries very quickly. "
                "Compare the animation speed of cooling here with Question 4 (\\(\\alpha=0.5\\)) — "
                "this rod cools eight times faster."
            ),
            "surface": (
                "The surface starts as a sloping arch (higher on the right) and falls steeply along the time axis. "
                "The steep drop reflects how fast \\(\\alpha=4\\) drives the temperatures toward zero. "
                "A material with high diffusivity (like copper) looks like this — rapid decay toward equilibrium."
            ),
        },
    },

    # -------------------------------------------------------------------
    # 4  Exercise 11.4, Question 1
    # -------------------------------------------------------------------
    4: {
        "title": "Exercise 11.4 Q1",
        "reference": (
            "Numerical Methods in Engineering and Science — "
            "Jain, Iyengar & Jain, Ch. 11, Exercise 11.4, Q1"
        ),
        "pde": "u_t = (1/2) u_xx",
        "alpha": 0.5,
        "x_start": 0.0,
        "x_end":   4.0,
        "final_time": 5.0,
        "number_of_space_points": 5,
        "r": 0.25,
        "time_step": None,
        "left_boundary":  0.0,
        "right_boundary": 0.0,
        "initial_condition_type": "function",
        "initial_condition_expr": "x * (4 - x)",
        "initial_condition_display": "\\(U_i^0 = x(4-x)\\)",
        "exact_solution_type": None,
        "run_supporting_engines":   True,
        "generate_animation":       True,
        "open_dashboard_after_run": True,

        "question_text": (
            "<p><strong>Exercise 11.4, Q1.</strong>&ensp;"
            "The textbook writes the PDE as \\(u_{xx} = 2\\,u_t\\), "
            "which rearranges to"
            "\\[u_t = \\tfrac{1}{2}\\,u_{xx}, \\quad 0 < x < 4,\\quad t > 0.\\]"
            "Boundary conditions: \\(u(0,t)=0\\), \\(u(4,t)=0\\). "
            "Initial condition:"
            "\\[u(x,0) = x(4-x).\\]"
            "Use \\(h=1\\) (five grid points) and \\(r=0.25\\), "
            "giving \\(k=0.5\\). Advance to \\(t=5\\) (ten time steps).</p>"
            "<p>\\(\\alpha=0.5\\) — half the standard diffusivity. "
            "Heat spreads at half the rate compared to \\(\\alpha=1\\).</p>"
        ),
        "graph_notes": {
            "profiles": (
                "The starting temperature \\(x(4-x)\\) is a symmetric arch peaking at 4 at the centre \\((x=2)\\). "
                "With \\(\\alpha=0.5\\) (slow diffusion), the arch shrinks gradually — "
                "compare with Question 3 (\\(\\alpha=4\\)) to see how much slower the cooling is here. "
                "Each step uses \\(k=0.5\\), so the x-axis in time covers a large range."
            ),
            "rod": (
                "The rod starts with a gentle warm arch at its centre. "
                "With \\(\\alpha=0.5\\), heat spreads at half the normal rate — the rod keeps its central warmth for much longer. "
                "This is like a low-conductivity material (concrete or wood) compared to a high-conductivity one (copper)."
            ),
            "surface": (
                "The surface decays slowly because \\(\\alpha=0.5\\). "
                "The height stays elevated much further along the time axis than in Question 3. "
                "This directly shows how diffusivity controls the cooling rate — "
                "halving \\(\\alpha\\) means roughly halving the decay rate."
            ),
        },
    },

    # -------------------------------------------------------------------
    # 5  Exercise 11.4, Question 2
    # -------------------------------------------------------------------
    5: {
        "title": "Exercise 11.4 Q2",
        "reference": (
            "Numerical Methods in Engineering and Science — "
            "Jain, Iyengar & Jain, Ch. 11, Exercise 11.4, Q2"
        ),
        "pde": "u_t = u_xx",
        "alpha": 1.0,
        "x_start": 0.0,
        "x_end":   1.0,
        "final_time": 0.05,
        "number_of_space_points": 11,
        "r": 0.25,
        "time_step": None,
        "left_boundary":  0.0,
        "right_boundary": 0.0,
        "initial_condition_type": "function",
        "initial_condition_expr": "x * (1 - x)",
        "initial_condition_display": "\\(U_i^0 = x(1-x)\\)",
        "exact_solution_type": None,
        "run_supporting_engines":   True,
        "generate_animation":       True,
        "open_dashboard_after_run": True,

        "question_text": (
            "<p><strong>Exercise 11.4, Q2.</strong>&ensp;"
            "Solve"
            "\\[u_t = u_{xx}, \\quad 0 < x < 1,\\quad t > 0\\]"
            "with \\(u(0,t)=0\\), \\(u(1,t)=0\\) and initial condition"
            "\\[u(x,0) = x(1-x).\\]"
            "Use \\(h=0.1\\) (eleven grid points) and \\(r=0.25\\), "
            "giving \\(k=0.0025\\). Advance to \\(t=0.05\\) (twenty time steps).</p>"
        ),
        "graph_notes": {
            "profiles": (
                "The starting temperature \\(x(1-x)\\) is a smooth arch peaking at 0.25 at \\(x=0.5\\). "
                "Watch how the shape of the curves changes slightly in the early steps — "
                "the initial parabola contains hidden components that smooth out quickly — "
                "then the curves become a pure smooth arch fading steadily."
            ),
            "rod": (
                "The rod starts with a gentle symmetric warmth (max 0.25 at centre). "
                "The fine mesh (11 grid points, \\(h=0.1\\)) gives a smooth, detailed animation. "
                "The temperature profile becomes a clean arch shape very quickly and then just fades away."
            ),
            "surface": (
                "The surface drops quickly at first (the shape changes rapidly in early steps) "
                "then more gradually as only the slowly-decaying component remains. "
                "The two-speed behaviour — fast then slow — is a signature of an initial condition "
                "that contains multiple frequency components."
            ),
        },
    },

    # -------------------------------------------------------------------
    # 6  Exercise 11.4, Question 3
    # -------------------------------------------------------------------
    6: {
        "title": "Exercise 11.4 Q3",
        "reference": (
            "Numerical Methods in Engineering and Science — "
            "Jain, Iyengar & Jain, Ch. 11, Exercise 11.4, Q3"
        ),
        "pde": "u_t = u_xx",
        "alpha": 1.0,
        "x_start": 0.0,
        "x_end":   5.0,
        "final_time": 3.0,
        "number_of_space_points": 6,
        "r": 0.5,
        "time_step": None,
        "left_boundary":  0.0,
        "right_boundary": 0.0,
        "initial_condition_type": "function",
        "initial_condition_expr": "x**2 * (25 - x**2)",
        "initial_condition_display": "\\(U_i^0 = x^2(25-x^2)\\)",
        "exact_solution_type": None,
        "run_supporting_engines":   True,
        "generate_animation":       True,
        "open_dashboard_after_run": True,

        "question_text": (
            "<p><strong>Exercise 11.4, Q3.</strong>&ensp;"
            "Solve"
            "\\[u_t = u_{xx}, \\quad 0 < x < 5,\\quad t > 0\\]"
            "with \\(u(0,t)=u(5,t)=0\\) and initial condition"
            "\\[u(x,0) = x^2(25 - x^2).\\]"
            "Use \\(h=1\\) (six grid points) and \\(r=0.5\\), giving \\(k=0.5\\). "
            "Advance to \\(t=3\\) (six time steps).</p>"
            "<p>The starting temperature peaks near \\(x \\approx 3.5\\) "
            "with a maximum value of about 156.</p>"
        ),
        "graph_notes": {
            "profiles": (
                "The starting temperature \\(x^2(25-x^2)\\) reaches about 156 near \\(x=3.5\\). "
                "These are very high values — the curves fall dramatically with each step. "
                "With \\(k=0.5\\) (a large time step), a lot of heat is lost in each jump. "
                "By the last curve, most of the heat has been drained into the cold boundaries."
            ),
            "rod": (
                "The rod starts extremely hot on its right half (near \\(x=3.5\\)). "
                "The very high starting temperatures (over 150) create a huge contrast with "
                "the cold ends, which drives rapid cooling. "
                "In just six steps of \\(k=0.5\\) each, the rod is nearly at zero."
            ),
            "surface": (
                "The surface is much taller than other examples (peak around 156) and falls sharply. "
                "Each step with \\(k=0.5\\) removes a large fraction of the remaining heat. "
                "The dramatic drop from the first to the last surface ring shows how quickly "
                "the heat escapes into the zero-temperature boundaries."
            ),
        },
    },

    # -------------------------------------------------------------------
    # 7  Exercise 11.4, Question 5  —  triangular IC
    # -------------------------------------------------------------------
    7: {
        "title": "Exercise 11.4 Q5",
        "reference": (
            "Numerical Methods in Engineering and Science — "
            "Jain, Iyengar & Jain, Ch. 11, Exercise 11.4, Q5"
        ),
        "pde": "u_t = u_xx",
        "alpha": 1.0,
        "x_start": 0.0,
        "x_end":   1.0,
        "final_time": 0.5,
        "number_of_space_points": 5,
        "r": 0.5,
        "time_step": None,
        "left_boundary":  0.0,
        "right_boundary": 0.0,
        "initial_condition_type": "function",
        "initial_condition_expr": "np.where(x <= 0.5, 2*x, 2*(1 - x))",
        "initial_condition_display": (
            "\\[U_i^0 = \\begin{cases} 2x & x \\le \\tfrac{1}{2} \\\\"
            " 2(1-x) & x > \\tfrac{1}{2} \\end{cases}\\]"
        ),
        "exact_solution_type": None,
        "run_supporting_engines":   True,
        "generate_animation":       True,
        "open_dashboard_after_run": True,

        "question_text": (
            "<p><strong>Exercise 11.4, Q5.</strong>&ensp;"
            "Solve"
            "\\[u_t = u_{xx}, \\quad 0 < x < 1,\\quad t > 0\\]"
            "with zero boundary conditions and the triangular initial condition"
            "\\[u(x,0) = \\begin{cases} 2x & 0 \\le x \\le \\tfrac{1}{2} \\\\"
            "2(1-x) & \\tfrac{1}{2} < x \\le 1. \\end{cases}\\]"
            "Use \\(h=0.25\\) (five grid points) and \\(r=0.5\\), "
            "giving \\(k=0.03125\\). Advance to \\(t=0.5\\).</p>"
            "<p>The triangular IC has a sharp point at \\(x=0.5\\). "
            "Watch what the heat equation does to it immediately.</p>"
        ),
        "graph_notes": {
            "profiles": (
                "The starting temperature is a triangle — it rises to 1 at \\(x=0.5\\), then falls. "
                "Look at how quickly the sharp point disappears: within just a few time steps "
                "the kink rounds into a smooth curve. "
                "The heat equation always removes sharp corners immediately — "
                "this smoothing property is one of its most important characteristics."
            ),
            "rod": (
                "The rod starts with a sharp hot spike at its centre. "
                "Press Play and watch the spike instantly become a smooth dome. "
                "Sharpness cannot survive in a heat diffusion problem — "
                "any sharp feature is immediately spread outward."
            ),
            "surface": (
                "The sharp ridge at the left edge of the surface (\\(t=0\\)) immediately becomes a rounded arch. "
                "The steep slope of the surface right at \\(t=0\\) shows how fast the sharp initial peak is smoothed out. "
                "After the first few steps, the surface looks just like the smooth cases."
            ),
        },
    },

    # -------------------------------------------------------------------
    # 8  Custom verification case
    # -------------------------------------------------------------------
    8: {
        "title": "Custom — u_t = 0.5 u_xx with exact solution",
        "reference": "Custom verification case (not from textbook)",
        "pde": "u_t = 0.5 u_xx",
        "alpha": 0.5,
        "x_start": 0.0,
        "x_end":   1.0,
        "final_time": 0.2,
        "number_of_space_points": 31,
        "r": 0.25,
        "time_step": None,
        "left_boundary":  0.0,
        "right_boundary": 0.0,
        "initial_condition_type": "function",
        "initial_condition_expr": "np.sin(np.pi * x)",
        "initial_condition_display": "\\(U_i^0 = \\sin(\\pi x)\\)",
        "exact_solution_type": "function",
        "exact_solution_expr": "np.exp(-alpha * np.pi**2 * t) * np.sin(np.pi * x)",
        "run_supporting_engines":   True,
        "generate_animation":       True,
        "open_dashboard_after_run": True,

        "question_text": (
            "<p><strong>Custom Case.</strong>&ensp;"
            "Solve"
            "\\[u_t = \\tfrac{1}{2}\\,u_{xx}, \\quad 0 < x < 1,\\quad t > 0\\]"
            "with zero boundary conditions and \\(u(x,0)=\\sin(\\pi x)\\). "
            "Use a fine mesh: \\(h=\\tfrac{1}{30}\\approx 0.033\\) (31 points), "
            "\\(r=0.25\\). Advance to \\(t=0.2\\).</p>"
            "<p>Exact solution: \\[U(x,t)=e^{-\\pi^2 t/2}\\sin(\\pi x).\\]"
            "This case is designed to check accuracy: with a fine mesh the "
            "solid FTCS curves and dashed exact curves should nearly overlap.</p>"
        ),
        "graph_notes": {
            "profiles": (
                "Same \\(\\sin(\\pi x)\\) shape as Questions 1 and 2, but \\(\\alpha=0.5\\) means the arch shrinks at half the speed. "
                "The fine mesh (31 points) makes both the solid FTCS lines and the dashed exact lines very smooth. "
                "The two should nearly overlap — any visible gap is the numerical error of the method."
            ),
            "rod": (
                "The familiar \\(\\sin(\\pi x)\\) shape, but cooling at half the rate of Question 1. "
                "The fine 31-point mesh makes the animation very smooth and detailed. "
                "Watch how slowly the glow fades compared to Question 1 — that is the effect of \\(\\alpha=0.5\\)."
            ),
            "surface": (
                "The arch-surface decays slowly (half speed vs Question 1). "
                "At the end time \\(t=0.2\\), about 37% of the original heat remains. "
                "The fine mesh makes the surface smooth, and the gentle slope in the time direction "
                "confirms the slow decay rate."
            ),
        },
    },

    # -------------------------------------------------------------------
    # 9  User slot
    # -------------------------------------------------------------------
    9: {
        "title": "My Question",
        "reference": "User-defined question",
        "pde": "u_t = alpha u_xx",
        "alpha": 1.0,
        "x_start": 0.0,
        "x_end":   1.0,
        "final_time": 0.05,
        "number_of_space_points": 41,
        "r": 0.25,
        "time_step": None,
        "left_boundary":  0.0,
        "right_boundary": 0.0,
        "initial_condition_type": "hot_center",
        "u_left":  0.4,
        "u_right": 0.6,
        "u_value": 100.0,
        "exact_solution_type": None,
        "run_supporting_engines":   True,
        "generate_animation":       True,
        "open_dashboard_after_run": True,

        "question_text": (
            "<p><strong>User Question.</strong>&ensp;"
            "Edit this slot in <code>question_config.py</code> to enter your own problem.</p>"
            "<p>Current setup: \\(u_t = u_{xx}\\) on \\([0,1]\\) with zero boundaries. "
            "Initial temperature: 100 on the centre strip \\(0.4 \\le x \\le 0.6\\), zero elsewhere. "
            "\\(h=0.025\\), \\(r=0.25\\).</p>"
        ),
        "graph_notes": {
            "profiles": (
                "The starting temperature is a rectangular pulse — 100 in the centre and 0 everywhere else. "
                "Watch the sharp edges immediately round off and the pulse spread outward as heat diffuses toward the cold boundaries. "
                "The flat-top shape cannot survive even one time step."
            ),
            "rod": (
                "The rod starts with a bright white-hot block at its centre. "
                "Press Play and watch the heat spread outward — the bright region widens and dims as "
                "energy flows into the surrounding cold sections."
            ),
            "surface": (
                "The flat-topped block at \\(t=0\\) immediately rounds into a dome shape. "
                "The surface shows the pulse spreading and flattening over time. "
                "The peak stays at \\(x=0.5\\) throughout because the starting condition is symmetric."
            ),
        },
    },

    # -------------------------------------------------------------------
    # 10  UNSTABLE DEMONSTRATION — r = 0.6 (slightly above limit)
    # -------------------------------------------------------------------
    10: {
        "title": "Instability Demo — r = 0.6",
        "reference": "Instability demonstration (not from textbook)",
        "pde": "u_t = u_xx",
        "alpha": 1.0,
        "x_start": 0.0,
        "x_end":   1.0,
        "final_time": 0.30,
        "number_of_space_points": 11,
        "r": 0.6,
        "time_step": None,
        "left_boundary":  0.0,
        "right_boundary": 0.0,
        "initial_condition_type": "hot_center",
        "u_left":  0.4,
        "u_right": 0.6,
        "u_value": 100.0,
        "exact_solution_type": None,
        "run_supporting_engines":   False,
        "generate_animation":       True,
        "open_dashboard_after_run": True,

        "question_text": (
            "<p><strong>Instability Demonstration — \\(r = 0.6\\).</strong>&ensp;"
            "This is the same physical problem as Question 9 "
            "(hot-centre pulse, \\(u_t = u_{xx}\\) on \\([0,1]\\)) "
            "but with \\(r = 0.6\\) — just above the stability limit of 0.5.</p>"
            "<p>FTCS requires \\(r \\le 0.5\\). When \\(r > 0.5\\), "
            "small numerical errors are amplified at each time step instead of being damped. "
            "Over many steps these errors grow large enough to corrupt the solution.</p>"
            "<p>Compare the graphs here with Question 9 (\\(r=0.25\\)) "
            "to see what instability looks like.</p>"
        ),
        "graph_notes": {
            "profiles": (
                "Look carefully: instead of the curves smoothly shrinking toward zero, "
                "they develop wiggles, dip below zero, or spike above the initial value. "
                "These are impossible in real heat flow — they are pure numerical error growing out of control. "
                "This is what instability looks like: the method gives wrong, non-physical answers."
            ),
            "rod": (
                "Watch for temperatures going below zero (black) or above the initial maximum (white or beyond). "
                "Neither is physically possible with this boundary setup. "
                "If you see the rod flicker between extremes or produce impossible values, "
                "that is the FTCS instability in action."
            ),
            "surface": (
                "Instead of a smooth, uniformly shrinking surface, look for peaks and valleys "
                "that grow over time rather than shrink. "
                "A stable run always produces a surface that falls monotonically — "
                "any surface that rises or oscillates confirms instability."
            ),
        },
    },

    # -------------------------------------------------------------------
    # 11  UNSTABLE DEMONSTRATION — r = 2.0 (explosive)
    # -------------------------------------------------------------------
    11: {
        "title": "Instability Demo — r = 2.0 (explosive)",
        "reference": "Instability demonstration (not from textbook)",
        "pde": "u_t = u_xx",
        "alpha": 1.0,
        "x_start": 0.0,
        "x_end":   1.0,
        "final_time": 0.10,
        "number_of_space_points": 11,
        "r": 2.0,
        "time_step": None,
        "left_boundary":  0.0,
        "right_boundary": 0.0,
        "initial_condition_type": "hot_center",
        "u_left":  0.4,
        "u_right": 0.6,
        "u_value": 100.0,
        "exact_solution_type": None,
        "run_supporting_engines":   False,
        "generate_animation":       True,
        "open_dashboard_after_run": True,

        "question_text": (
            "<p><strong>Instability Demonstration — \\(r = 2.0\\).</strong>&ensp;"
            "The same physical problem again, but now \\(r = 2.0\\) — "
            "four times the stability limit.</p>"
            "<p>With \\(r = 2.0\\), the FTCS update formula gives a negative weight "
            "\\((1-2r) = -3\\) to the centre node. "
            "This means each step subtracts three times the current value and adds "
            "twice each neighbour — the solution explodes immediately.</p>"
            "<p>After just 2-3 time steps the values become wildly large and completely wrong. "
            "All the computed temperatures are numerical garbage.</p>"
        ),
        "graph_notes": {
            "profiles": (
                "The values grow wildly from the very first step — expect temperatures "
                "far above 100 or deeply negative, alternating between grid points. "
                "This is not delayed instability; it is immediate explosion. "
                "The correct physical answer (gentle spreading of the pulse) is nowhere to be seen."
            ),
            "rod": (
                "The rod will show extreme temperatures within the first few steps — "
                "values that shoot above the initial 100 or go deeply negative. "
                "This is the clearest possible demonstration that \\(r > 0.5\\) makes FTCS produce "
                "completely wrong answers."
            ),
            "surface": (
                "The surface will have enormous, alternating peaks and valleys — "
                "the opposite of the smooth, gently falling surface you see in stable cases. "
                "There is no physical meaning to any of these values. "
                "The only lesson from this surface is: always check your stability ratio before trusting FTCS results."
            ),
        },
    },
}


# -----------------------------------------------------------------------
# ACTIVE QUESTION — change this number (1-11) to select a question
# -----------------------------------------------------------------------
ACTIVE_QUESTION = 3

QUESTION = QUESTION_BANK[ACTIVE_QUESTION]


def main():
    import subprocess
    import sys
    from pathlib import Path

    module_dir = Path(__file__).resolve().parent
    runner_path = module_dir / "00_run_question_lab.py"

    completed = subprocess.run([sys.executable, str(runner_path)], cwd=module_dir)
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
