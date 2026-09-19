"""Configure one shared heat-equation problem for method comparison."""

QUESTION = {
    "title": "Method comparison: sine-wave cooling",
    "alpha": 1.0,
    "x_start": 0.0,
    "x_end": 1.0,
    "final_time": 0.1,
    "space_points": 51,
    "r": 0.25,
    "left_boundary": 0.0,
    "right_boundary": 0.0,
    "initial_condition": "sin_pi",
    "exact_solution": "sin_pi_zero_boundary",
}
