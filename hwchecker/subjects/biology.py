"""
hwchecker/subjects/biology.py
Biology SubjectProfile. deterministic_checker points at
checkers/fuzzy_match.py -- short factual-answer matching against an
accepted-answer list runs at $0 before any LLM call, wired in by
subjects/registry.py.
"""

from .base import DEFAULT_SYSTEM_PROMPT_TEMPLATE, RubricCriterion, SubjectProfile

BIOLOGY_PROFILE = SubjectProfile(
    subject_id="biology",
    grading_paradigm="mixed",  # short factual answers are objective; explanations are rubric-graded
    topic_weights={
        "cell_biology": 0.15, "genetics": 0.20, "evolution": 0.10, "ecology": 0.10,
        "physiology": 0.20, "molecular_biology": 0.15, "microbiology": 0.10,
    },
    rubric_criteria=[
        RubricCriterion("accuracy", 60, "The stated facts/terminology/mechanism are scientifically accurate"),
        RubricCriterion("completeness", 25, "The explanation covers the relevant steps/factors, not just a fragment"),
        RubricCriterion("terminology", 15, "Correct, precise biological terminology is used"),
    ],
    error_taxonomy=[
        "Factual Error", "Terminology Error", "Conceptual Error",
        "Classification Error", "Incomplete Explanation",
    ],
    system_prompt_template=DEFAULT_SYSTEM_PROMPT_TEMPLATE,
    deterministic_checker=None,
)
