"""
review_validation.py
Validation of a teacher's `corrected_result` (POST /submissions/{id}/review, action="edit").

The corrected result becomes the submission's FINAL result and is pushed to the CRM, so it must
have the same shape the AI result has (see hwchecker/pipeline/crm_adapter.py): numeric
`overall_score` in 0..100 and a `questions` list. Anything else is rejected with a precise message
instead of being stored as somebody's grade. Unknown extra keys are allowed (forward compatible).
"""

import json

MAX_QUESTIONS = 200
MAX_JSON_BYTES = 256 * 1024
_TEXT_FIELDS = ("question", "student_answer", "expected_solution", "analysis", "correct_answer", "difficulty")
_LIST_FIELDS = ("mistakes", "error_categories", "suggestions")


class ReviewResultError(ValueError):
    """The corrected_result is not an acceptable grading result."""


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value == value  # NaN != NaN


def _check_score(value, where: str) -> None:
    if not _is_number(value) or not (0 <= value <= 100):
        raise ReviewResultError(f"{where} must be a number between 0 and 100, got {value!r}.")


def validate_review_result(result) -> dict:
    """Returns `result` unchanged if valid, otherwise raises ReviewResultError."""
    if not isinstance(result, dict):
        raise ReviewResultError("corrected_result must be a JSON object.")
    try:
        size = len(json.dumps(result, ensure_ascii=False).encode("utf-8"))
    except (TypeError, ValueError):
        raise ReviewResultError("corrected_result must be plain JSON.")
    if size > MAX_JSON_BYTES:
        raise ReviewResultError(f"corrected_result is too large ({size} bytes; the limit is {MAX_JSON_BYTES}).")

    if "overall_score" not in result:
        raise ReviewResultError("corrected_result.overall_score is required.")
    _check_score(result["overall_score"], "corrected_result.overall_score")

    questions = result.get("questions")
    if not isinstance(questions, list):
        raise ReviewResultError("corrected_result.questions must be a list.")
    if len(questions) > MAX_QUESTIONS:
        raise ReviewResultError(f"corrected_result.questions has more than {MAX_QUESTIONS} items.")
    for index, question in enumerate(questions, start=1):
        where = f"corrected_result.questions[{index}]"
        if not isinstance(question, dict):
            raise ReviewResultError(f"{where} must be an object.")
        if "score" in question:
            _check_score(question["score"], f"{where}.score")
        for field in _TEXT_FIELDS:
            if field in question and not isinstance(question[field], str):
                raise ReviewResultError(f"{where}.{field} must be a string.")
        for field in _LIST_FIELDS:
            if field in question and not (
                isinstance(question[field], list) and all(isinstance(item, str) for item in question[field])
            ):
                raise ReviewResultError(f"{where}.{field} must be a list of strings.")

    if "grade" in result and not isinstance(result["grade"], str):
        raise ReviewResultError("corrected_result.grade must be a string.")
    if "summary" in result and not isinstance(result["summary"], dict):
        raise ReviewResultError("corrected_result.summary must be an object.")
    return result
