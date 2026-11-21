"""
hwchecker/subjects/physics.py
Physics SubjectProfile. deterministic_checker points at
checkers/unit_check.py (Pint) -- unit-consistency and final-numeric-
answer verification runs at $0 before any LLM call, wired in by
subjects/registry.py to avoid a circular import with checkers/.
"""

from .base import DEFAULT_SYSTEM_PROMPT_TEMPLATE, RubricCriterion, SubjectProfile

PHYSICS_PROFILE = SubjectProfile(
    subject_id="physics",
    grading_paradigm="objective",
    topic_weights={
        "kinematics": 0.15, "dynamics": 0.15, "energy_work": 0.15, "electromagnetism": 0.20,
        "thermodynamics": 0.15, "waves_optics": 0.10, "modern_physics": 0.10,
    },
    rubric_criteria=[
        RubricCriterion("final_answer", 50, "The final numeric answer, with correct units, matches the correct result"),
        RubricCriterion("method", 35, "The physical reasoning and equations used constitute a valid approach"),
        RubricCriterion("units_sig_figs", 15, "Units are tracked correctly and precision/significant figures are reasonable"),
    ],
    error_taxonomy=[
        "Calculation Error", "Unit Error", "Formula Error", "Sign Error",
        "Conceptual Error", "Missing Step", "Incorrect Assumption",
    ],
    system_prompt_template=DEFAULT_SYSTEM_PROMPT_TEMPLATE,
    deterministic_checker=None,
)
