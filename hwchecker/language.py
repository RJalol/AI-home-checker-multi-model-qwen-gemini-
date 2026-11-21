"""
hwchecker/language.py
Which language the AI writes its feedback in -- one setting, used by both the
tiered hwchecker pipeline (subjects/base.py) and the original single-call engine
(prompts.py).

Before this existed no prompt said anything about the reply language, so the model
answered in whatever language it happened to pick (usually the language of the
instructions, i.e. English, or of the student's work). Now it is explicit:

    FEEDBACK_LANGUAGE="Uzbek (Latin script)"   # default
    FEEDBACK_LANGUAGE="Russian"                # any language name the model understands
    FEEDBACK_LANGUAGE="auto"                   # same language as the student's work

Only the human-readable text (feedback, analysis, mistakes, suggestions, summary lists)
is translated. JSON keys, classification labels and error-category names stay in English
so the CRM, the database and the analytics keep seeing stable values.

Stdlib only: this module is imported from both code bases and must stay dependency-free.
"""

import os

DEFAULT_FEEDBACK_LANGUAGE = "Uzbek (Latin script)"
_AUTO_VALUES = {"auto", "student", "same", "match"}


def get_feedback_language() -> str:
    """The configured feedback language (read on every call, so a changed env var is picked up)."""
    value = os.environ.get("FEEDBACK_LANGUAGE", "").strip()
    return value or DEFAULT_FEEDBACK_LANGUAGE


def language_instruction(language: str = None) -> str:
    """The '# RESPONSE LANGUAGE' block appended to every grading system prompt."""
    language = (language or get_feedback_language()).strip()

    keep_english = (
        "Keep every JSON key, and every fixed label (classification values, grade names, "
        "error-category / error_type names), exactly as specified in this prompt, in English. "
        "Quote the student's own words, formulas and code exactly as written -- never translate them."
    )

    if language.lower() in _AUTO_VALUES:
        return (
            "# RESPONSE LANGUAGE\n"
            "Write every human-readable text value (feedback, analysis, mistakes, suggestions, summary "
            "lists, error locations) in the language the student's work is written in. If the teacher's "
            "additional instructions are in another language, use that one for the explanations. "
            "If you cannot tell, use English.\n" + keep_english
        )

    return (
        "# RESPONSE LANGUAGE\n"
        f"Write every human-readable text value (feedback, analysis, mistakes, suggestions, summary "
        f"lists, error locations) in {language}, whatever language the student's work or these "
        f"instructions are in.\n" + keep_english + "\n"
        f"For a language-learning task (e.g. English, Russian or Turkish homework) the corrections and "
        f"example sentences stay in the language being studied; only the explanations around them are "
        f"written in {language}."
    )
