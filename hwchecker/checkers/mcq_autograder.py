"""
hwchecker/checkers/mcq_autograder.py
Deterministic (no LLM, $0) exact-match grading for multiple-choice /
single-correct-answer questions. Generic across every subject and
language skill -- a test question is a test question whether it's math,
history, or a reading-comprehension check. Called directly by
router/grading_router.py before any subject-specific deterministic
checker, whenever the homework metadata marks the question as MCQ.
"""

import re
from typing import Optional

from hwchecker.providers.base import GradingResult

# "Option B", "Answer: B", "Javob: B", "Ответ B", ... -> the prefix word is dropped
_PREFIX_RE = re.compile(
    r"^(?:the\s+)?(?:correct\s+)?(?:option|answer|ans|choice|variant|javob|variant|ответ|вариант|cevap|seçenek)\s*[:\-]?\s*",
    re.IGNORECASE,
)
# A choice token at the start: "B", "B)", "(B)", "b.", "B: Paris", "B) Paris", "2)"
_CHOICE_RE = re.compile(r"^[\(\[]?([A-Za-z0-9])(?:[\)\]\.:\-]|$)")
_CHOICE_TOKEN_RE = re.compile(r"^[A-Z0-9]$")


def _normalize_choice(text: str) -> str:
    """'B)', '(b)', 'Option B', 'Answer: b.' and 'B) Paris' all normalise to 'B'; other text
    is just upper-cased and stripped of surrounding punctuation."""
    cleaned = _PREFIX_RE.sub("", str(text).strip()).strip()
    match = _CHOICE_RE.match(cleaned)
    if match:
        return match.group(1).upper()
    return cleaned.strip(" .:;()[]{}").upper()


def check(homework: dict) -> Optional[GradingResult]:
    """
    homework: {"student_answer": "B", "correct_choice": "B",
               "question_type": "mcq", ...}
    Only activates when homework explicitly declares question_type ==
    "mcq" (or "multiple_choice") -- this checker does not try to guess
    whether a question is MCQ from its text.
    """
    question_type = (homework.get("question_type") or "").lower()
    if question_type not in ("mcq", "multiple_choice"):
        return None

    student_choice = homework.get("student_answer")
    correct_choice = homework.get("correct_choice")
    if not student_choice or not correct_choice:
        return None

    student_token = _normalize_choice(str(student_choice))
    correct_token = _normalize_choice(str(correct_choice))
    is_correct = student_token == correct_token
    if not is_correct and not (
        _CHOICE_TOKEN_RE.match(student_token) and _CHOICE_TOKEN_RE.match(correct_token)
    ):
        # The student wrote something that is not a clean choice letter ("I think it is B",
        # a full sentence): do not auto-fail it with confidence 1.0 -- let the AI tiers read it.
        return None

    return GradingResult(
        is_correct=is_correct,
        classification="CORRECT" if is_correct else "INCORRECT",
        score=100.0 if is_correct else 0.0,
        error_type=None if is_correct else "Incorrect Choice",
        error_location=None,
        feedback=(
            f"Correct -- '{student_choice}' is the right answer (no AI call needed)."
            if is_correct
            else f"Incorrect. The correct answer was '{correct_choice}', not '{student_choice}' (no AI call needed)."
        ),
        sub_scores={},
        confidence=1.0,
        raw_tokens={"input": 0, "output": 0},
        latency_ms=0,
        provider_name="mcq_autograder",
    )
