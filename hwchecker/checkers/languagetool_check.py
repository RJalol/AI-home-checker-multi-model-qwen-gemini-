"""
hwchecker/checkers/languagetool_check.py
Deterministic (no LLM, $0 beyond LanguageTool's own free-tier limits)
grammar/spelling checking for English and Russian (with partial Turkish
support; NOT meaningfully supported for Uzbek -- see
subjects/languages.py). Uses LanguageTool's public REST API over HTTP
(no local Java server required, which keeps this deployable anywhere).

This is conservative about what it fully resolves, same philosophy as
the other checkers: grammar correctness alone doesn't tell you whether
an essay achieved its task, so this only auto-resolves the narrow case
of a dedicated grammar-correction exercise (homework declares
question_type="grammar_correction" with an expected corrected answer).
For general Writing/Reading/Listening grading, use detect_errors()
below to enrich the LLM's context instead -- exposed separately so
router/grading_router.py or a provider can call it without needing a
full GradingResult resolution.
"""

import difflib
from typing import List, Optional

from hwchecker.providers.base import GradingResult

_LANGUAGE_CODES = {
    "english": "en-US",
    "russian": "ru-RU",
    "turkish": "tr",  # partial LanguageTool support
}

LANGUAGETOOL_API_URL = "https://api.languagetool.org/v2/check"


def detect_errors(text: str, language_id: str, timeout: float = 8.0):
    """
    Returns a list of {"message": str, "context": str, "rule_id": str}
    for each grammar/spelling issue LanguageTool finds, or None if the
    language isn't supported or the request fails (network issue, rate
    limit, etc.) -- callers should treat None as "no signal available",
    not as "no errors found".
    """
    language_code = _LANGUAGE_CODES.get(language_id)
    if not language_code or not text.strip():
        return None

    try:
        import requests

        response = requests.post(
            LANGUAGETOOL_API_URL,
            data={"text": text, "language": language_code},
            timeout=timeout,
        )
        response.raise_for_status()
        data = response.json()
    except Exception:
        return None

    errors = []
    for match in data.get("matches", []):
        errors.append({
            "message": match.get("message", ""),
            "context": (match.get("context") or {}).get("text", ""),
            "rule_id": (match.get("rule") or {}).get("id", ""),
        })
    return errors


def check(homework: dict) -> Optional[GradingResult]:
    """
    homework: {"student_answer": str, "language_id": "english",
               "question_type": "grammar_correction",
               "expected_corrected_text": str, ...}
    Only auto-resolves a dedicated grammar-correction exercise where the
    student's answer text-matches the expected corrected version AND
    LanguageTool confirms zero remaining errors. Everything else
    (general Writing/Reading/Listening grading) returns None -- use
    detect_errors() directly to enrich the LLM's rubric context instead.
    """
    if (homework.get("question_type") or "").lower() != "grammar_correction":
        return None

    language_id = homework.get("language_id", "")
    student = (homework.get("student_answer") or "").strip()
    expected = (homework.get("expected_corrected_text") or "").strip()
    if not student or not expected or language_id not in _LANGUAGE_CODES:
        return None

    similarity = difflib.SequenceMatcher(None, student.lower(), expected.lower()).ratio()
    if similarity < 0.97:
        return None  # meaningfully different from the expected correction -- defer to the LLM

    errors = detect_errors(student, language_id)
    if errors is None:
        return None  # LanguageTool unreachable -- defer, don't guess
    if len(errors) > 0:
        return None  # still has flagged issues -- defer to the LLM for a real judgment

    return GradingResult(
        is_correct=True,
        classification="CORRECT",
        score=100.0,
        error_type=None,
        error_location=None,
        feedback="Matches the expected correction with no remaining grammar/spelling issues detected (no AI call needed).",
        sub_scores={"grammar": 100},
        confidence=0.95,
        raw_tokens={"input": 0, "output": 0},
        latency_ms=0,
        provider_name="languagetool_check",
    )
