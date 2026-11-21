"""
hwchecker/router/difficulty.py
Determines a homework item's difficulty tier ("easy" | "medium" | "hard")
so grading_router.py knows which provider tier to use. Per spec section
2/6: Python/metadata first ($0); only escalate to an LLM (Qwen3.7 Flash,
the "router" tier in providers/registry.py) when metadata alone can't
decide -- expected to be roughly 10-20% of cases (for the CRM upload path,
which carries no metadata, it is every submission -- the call is cheap).
"""

from pathlib import Path
from typing import Optional

from hwchecker.infra.deadline import JobDeadline
from hwchecker.providers.base import extract_json_object

VALID_TIERS = {"easy", "medium", "hard"}

# Step-count heuristic, used when a subject's rubric declares how many
# reasoning steps a correct solution needs (homework.get("rubric_step_count")).
_STEP_COUNT_THRESHOLDS = {"easy": (0, 2), "medium": (3, 5), "hard": (6, float("inf"))}


def _classify_from_step_count(step_count: int) -> Optional[str]:
    for tier, (lo, hi) in _STEP_COUNT_THRESHOLDS.items():
        if lo <= step_count <= hi:
            return tier
    return None


def classify_from_metadata(homework: dict, subject_profile=None) -> Optional[str]:
    """
    Tries to determine difficulty from metadata alone, $0, no LLM call.
    Returns "easy" | "medium" | "hard", or None if metadata is
    insufficient (caller should then escalate to classify_with_llm()).

    Checked in order of reliability:
      1. Explicit homework.get("difficulty_hint") -- e.g. from the
         source curriculum/textbook metadata, if the CRM provides it.
      2. homework.get("rubric_step_count") -- number of reasoning steps
         a correct solution needs.

    NOTE: SubjectProfile.topic_weights is deliberately NOT used as a
    difficulty signal here, even though `subject_profile` is accepted
    for future use -- topic_weights represents how much curriculum
    coverage a topic gets, not how hard it is (e.g. "arithmetic" has a
    low weight in MATH_PROFILE because it's a small slice of a broader
    curriculum, not because it's hard -- using weight as a difficulty
    proxy actively misclassifies cases like this). A confidently wrong
    heuristic is worse than falling through to the LLM classifier.
    """
    explicit = (homework.get("difficulty_hint") or "").strip().lower()
    if explicit in VALID_TIERS:
        return explicit

    step_count = homework.get("rubric_step_count")
    if isinstance(step_count, int):
        tier = _classify_from_step_count(step_count)
        if tier:
            return tier

    return None


_LLM_DIFFICULTY_RULES = """- easy: a single concept, minimal steps, a typical intro-level question.
- medium: multiple steps or concepts combined, but a standard/well-practiced approach applies.
- hard: requires a non-obvious approach, proof, optimization, or synthesis of multiple advanced concepts."""

_LLM_DIFFICULTY_PROMPT = """Classify the difficulty of this homework question as exactly one word: easy, medium, or hard.

""" + _LLM_DIFFICULTY_RULES + """

Subject: {subject_id}
Question: {question_text}

Respond with ONLY the JSON object: {{"difficulty": "easy" | "medium" | "hard"}}"""

# Used when the CRM sends only the student's file (a photo / scan / document) and no question text.
_LLM_DIFFICULTY_FILE_PROMPT = """Look at the attached homework and classify its overall difficulty as exactly one word: easy, medium, or hard. If it holds several questions, judge by the most demanding one the student attempted.

""" + _LLM_DIFFICULTY_RULES + """

Subject: {subject_id}

Respond with ONLY the JSON object: {{"difficulty": "easy" | "medium" | "hard"}}"""


def classify_with_llm(homework: dict, subject_id: str) -> str:
    """
    Escalates to the "router" tier provider (Qwen3.7 Flash by default,
    see providers/registry.py) for a cheap classification call. Only
    reached when classify_from_metadata() returns None.

    What it looks at: the question text when the caller supplied one,
    otherwise the student's file itself (attached to the request). It used
    to classify an empty string in the second case -- which is every upload
    arriving through the CRM -- so the "difficulty" was a coin flip on no
    information. With neither text nor file there is nothing to judge, so no
    call is made.

    Never raises: a missing API key, an unreachable endpoint or an unusable
    reply all fall back to "medium" (see the end of this function).
    """
    from hwchecker.providers.registry import get_provider

    question_text = (homework.get("question_text") or homework.get("student_answer") or "").strip()
    file_path = (homework.get("content") or {}).get("file_path")
    if file_path and not Path(file_path).is_file():
        file_path = None

    if not question_text and not file_path:
        return "medium"

    try:
        provider = get_provider("router")   # inside the try: a missing QWEN_API_KEY must not abort grading
        complete_json = getattr(provider, "complete_json", None)
        if complete_json is None:
            return "medium"   # this provider has no plain-completion call (e.g. Gemini/Claude as router)

        if question_text:
            prompt = _LLM_DIFFICULTY_PROMPT.format(subject_id=subject_id, question_text=question_text[:2000])
            attachment = None
        else:
            prompt = _LLM_DIFFICULTY_FILE_PROMPT.format(subject_id=subject_id)
            attachment = file_path

        raw = complete_json(prompt, file_path=attachment)
        data = extract_json_object(raw, prefer_keys=("difficulty",))
        tier = str(data.get("difficulty", "")).strip().lower()
        if tier in VALID_TIERS:
            return tier
    except JobDeadline:
        raise   # the job's time budget is spent: do not carry on with a default and keep grading
    except Exception:
        pass

    # If the LLM call fails or returns something unusable, default to
    # "medium" -- the safest middle ground (neither the cheapest/weakest
    # tier nor the most expensive one) rather than raising and blocking
    # the whole grading pipeline over a difficulty-classification hiccup.
    return "medium"


def detect_difficulty(homework: dict, subject_profile=None) -> str:
    """
    The single entry point grading_router.py calls: metadata-first,
    LLM-fallback only when needed.
    """
    subject_id = homework.get("subject_id") or (subject_profile.subject_id if subject_profile else "")
    tier = classify_from_metadata(homework, subject_profile)
    if tier:
        return tier
    return classify_with_llm(homework, subject_id)
