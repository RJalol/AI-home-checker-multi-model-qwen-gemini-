"""
hwchecker/subjects/history.py
History SubjectProfile. No deterministic_checker -- historical analysis
is inherently open-ended (argument quality, evidence use), not reducible
to a symbolic/exact-match check the way math/physics/chemistry are.
MCQ-format history questions still benefit from checkers/mcq_autograder.py,
applied generically across subjects rather than being wired in here.
"""

from .base import DEFAULT_SYSTEM_PROMPT_TEMPLATE, RubricCriterion, SubjectProfile

HISTORY_PROFILE = SubjectProfile(
    subject_id="history",
    grading_paradigm="rubric",
    topic_weights={
        "ancient_history": 0.10, "medieval_history": 0.10, "early_modern": 0.15,
        "industrial_revolution": 0.15, "20th_century": 0.30, "contemporary": 0.20,
    },
    rubric_criteria=[
        RubricCriterion("factual_accuracy", 40, "Dates, events, people, and facts stated are accurate"),
        RubricCriterion("argument_quality", 35, "Claims are supported by evidence and reasoning, not just assertion"),
        RubricCriterion("context_causation", 25, "Cause/effect and broader context are correctly understood"),
    ],
    error_taxonomy=[
        "Factual Error", "Chronology Error", "Weak Argument",
        "Missing Evidence", "Missing Context", "Source Misinterpretation",
    ],
    system_prompt_template=DEFAULT_SYSTEM_PROMPT_TEMPLATE,
    deterministic_checker=None,
)
