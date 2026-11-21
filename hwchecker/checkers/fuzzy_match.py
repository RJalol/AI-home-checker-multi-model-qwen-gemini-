"""
hwchecker/checkers/fuzzy_match.py
Deterministic (no LLM, $0) short factual-answer matching for biology
(subjects/biology.py) -- compares a student's short answer against a
list of accepted answers/synonyms using string similarity, not exact
match (so "mitochondria" vs "mitochondrion" vs minor typos still match).

Only ever resolves the CORRECT case with high similarity confidence;
anything below the threshold, or any answer requiring explanation (not
just a short factual term), falls through to the LLM tiers.
"""

import difflib
import re
from typing import List, Optional

from hwchecker.providers.base import GradingResult

DEFAULT_SIMILARITY_THRESHOLD = 0.85   # kept for API compatibility; see _is_acceptable_typo
MAX_ANSWER_WORDS = 6  # longer answers likely need explanation-quality judgment, not just term-matching
MIN_LENGTH_FOR_TYPO_TOLERANCE = 8   # in shorter words ONE edit can make a different word (mitosis/meiosis, 5/15)

_NUMBER_RE = re.compile(r"[-+]?\d+(?:[.,]\d+)?")


def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def _similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


def _numbers(text: str) -> list:
    """Every number WITH its sign and decimal part, e.g. '-5' and '1.5' -- punctuation stripping
    in _normalize would turn them into '5' and '15'."""
    return [n.replace(",", ".") for n in _NUMBER_RE.findall(text)]


def _edit_distance_at_most_one(a: str, b: str) -> bool:
    if a == b:
        return True
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        mismatches = [i for i in range(len(a)) if a[i] != b[i]]
        if len(mismatches) == 1:
            return True
        # one adjacent transposition ("teh" -> "the")
        return len(mismatches) == 2 and mismatches[1] == mismatches[0] + 1 \
            and a[mismatches[0]] == b[mismatches[1]] and a[mismatches[1]] == b[mismatches[0]]
    shorter, longer = (a, b) if len(a) < len(b) else (b, a)
    return any(longer[:i] + longer[i + 1:] == shorter for i in range(len(longer)))


def _is_acceptable_typo(student_norm: str, accepted_norm: str) -> bool:
    """A near miss is accepted only when it is plausibly a typo of that very word: one edit away,
    a word long enough that one edit cannot yield another real term, and nothing else changed."""
    return len(accepted_norm) >= MIN_LENGTH_FOR_TYPO_TOLERANCE and _edit_distance_at_most_one(student_norm, accepted_norm)


def check(
    homework: dict,
    accepted_answers=None,
    threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> Optional[GradingResult]:
    """
    homework: {"student_answer": str, "accepted_answers": [str, ...], ...}
    (accepted_answers can also be passed directly as an argument, e.g.
    from a homework-specific answer key -- either source works.)

    Resolves CORRECT only for an exact match after normalisation (case, spacing, punctuation) or a
    single-typo match of a long word. Numbers must agree exactly including sign and decimals, and
    similar-looking different terms (mitosis/meiosis) are never matched: those go to the AI tiers.
    """
    student = (homework.get("student_answer") or "").strip()
    answers = accepted_answers or homework.get("accepted_answers") or []

    if not student or not answers:
        return None
    if len(student.split()) > MAX_ANSWER_WORDS:
        return None  # likely needs explanation-quality judgment, not term-matching

    normalized_student = _normalize(student)
    student_numbers = _numbers(student)
    best = None   # (exact, similarity)
    for accepted in answers:
        accepted = str(accepted)
        if _numbers(accepted) != student_numbers:
            continue
        normalized_accepted = _normalize(accepted)
        if not normalized_accepted:
            continue
        if normalized_student == normalized_accepted:
            best = (True, 1.0)
            break
        if _is_acceptable_typo(normalized_student, normalized_accepted):
            similarity = _similarity(normalized_student, normalized_accepted)
            if best is None or similarity > best[1]:
                best = (False, similarity)

    if best is not None:
        exact, similarity = best
        feedback = (
            f"'{student}' matches an accepted answer (no AI call needed)."
            if exact else
            f"'{student}' matches an accepted answer apart from a small typo (similarity {similarity:.0%}, no AI call needed)."
        )
        return GradingResult(
            is_correct=True,
            classification="CORRECT",
            score=100.0,
            error_type=None,
            error_location=None,
            feedback=feedback,
            sub_scores={},
            confidence=1.0 if exact else 0.9,
            raw_tokens={"input": 0, "output": 0},
            latency_ms=0,
            provider_name="fuzzy_match",
        )

    return None  # not a confident match -- defer to the LLM tier
