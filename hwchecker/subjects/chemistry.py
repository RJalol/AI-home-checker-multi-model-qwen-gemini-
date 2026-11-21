"""
hwchecker/subjects/chemistry.py
Chemistry SubjectProfile. deterministic_checker points at
checkers/equation_balance.py -- balanced-equation verification runs at
$0 before any LLM call, wired in by subjects/registry.py.
"""

from .base import DEFAULT_SYSTEM_PROMPT_TEMPLATE, RubricCriterion, SubjectProfile

CHEMISTRY_PROFILE = SubjectProfile(
    subject_id="chemistry",
    grading_paradigm="objective",
    topic_weights={
        "stoichiometry": 0.20, "equilibrium": 0.15, "thermochemistry": 0.10, "kinetics": 0.10,
        "acid_base": 0.15, "electrochemistry": 0.10, "organic_chemistry": 0.15, "atomic_structure": 0.05,
    },
    rubric_criteria=[
        RubricCriterion("final_answer", 55, "The final answer (value, formula, or balanced equation) is correct"),
        RubricCriterion("method", 30, "The chemical reasoning/calculation steps constitute a valid approach"),
        RubricCriterion("notation", 15, "Chemical formulas, equations, and units are correctly notated"),
    ],
    error_taxonomy=[
        "Calculation Error", "Stoichiometry Error", "Formula Error", "Nomenclature Error",
        "Unbalanced Equation", "Unit Error", "Conceptual Error", "Missing Step",
    ],
    system_prompt_template=DEFAULT_SYSTEM_PROMPT_TEMPLATE,
    deterministic_checker=None,
)
