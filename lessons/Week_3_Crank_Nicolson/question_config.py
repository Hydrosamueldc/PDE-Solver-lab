"""Choose a Crank-Nicolson practice problem, then run 00_run_question_lab.py."""

QUESTION_BANK = {
    1: {
        "title": "Sine-wave cooling with Crank-Nicolson",
        "reference": "Verification problem with an exact solution",
        "alpha": 1.0,
        "x_start": 0.0,
        "x_end": 1.0,
        "final_time": 0.1,
        "space_points": 51,
        "r": 0.5,
        "left_boundary": 0.0,
        "right_boundary": 0.0,
        "initial_condition": "sin_pi",
        "exact_solution": "sin_pi_zero_boundary",
    },
    2: {
        "title": "Large time step: stable but less accurate",
        "reference": "Crank-Nicolson stability and accuracy demonstration",
        "alpha": 1.0,
        "x_start": 0.0,
        "x_end": 1.0,
        "final_time": 0.1,
        "space_points": 21,
        "r": 2.0,
        "left_boundary": 0.0,
        "right_boundary": 0.0,
        "initial_condition": "sin_pi",
        "exact_solution": "sin_pi_zero_boundary",
    },
    3: {
        "title": "Gaussian heat pulse",
        "reference": "Qualitative diffusion experiment",
        "alpha": 0.5,
        "x_start": 0.0,
        "x_end": 1.0,
        "final_time": 0.3,
        "space_points": 81,
        "r": 1.0,
        "left_boundary": 0.0,
        "right_boundary": 0.0,
        "initial_condition": "gaussian",
        "exact_solution": None,
    },
}

ACTIVE_QUESTION = 1
QUESTION = QUESTION_BANK[ACTIVE_QUESTION]
