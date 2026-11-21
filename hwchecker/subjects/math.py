"""
hwchecker/subjects/math.py
Math SubjectProfile. deterministic_checker points at
checkers/sympy_check.py -- final-answer verification for objective
numeric/symbolic problems runs at $0 before any LLM call.
"""

from .base import DEFAULT_SYSTEM_PROMPT_TEMPLATE, RubricCriterion, SubjectProfile

MATH_PROFILE = SubjectProfile(
    subject_id="math",
    grading_paradigm="objective",
    topic_weights={
        "arithmetic": 0.05, "algebra": 0.25, "geometry": 0.15, "trigonometry": 0.10,
        "calculus": 0.20, "linear_algebra": 0.10, "probability_statistics": 0.10, "discrete_math": 0.05,
    },
    rubric_criteria=[
        RubricCriterion("final_answer", 60, "The final answer matches the correct result"),
        RubricCriterion("method", 30, "The steps shown constitute a mathematically valid method"),
        RubricCriterion("rigor", 10, "Notation, justification, and completeness of the work shown"),
    ],
    error_taxonomy=[
        "Arithmetic Error", "Algebra Error", "Sign Error", "Conceptual Error",
        "Missing Step", "Notation Error", "Incorrect Assumption",
    ],
    system_prompt_template=DEFAULT_SYSTEM_PROMPT_TEMPLATE,
    deterministic_checker=None,  # wired in by subjects/registry.py to avoid a circular import with checkers/
)
