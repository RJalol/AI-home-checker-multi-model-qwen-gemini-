"""
hwchecker/providers/base.py
The AIProvider abstraction every concrete provider (GeminiProvider,
QwenProvider, GLMProvider, ClaudeProvider, OpenAIProvider,
SelfHostedProvider) implements. Swapping which provider handles a tier
is meant to be a one-line change in providers/registry.py -- nothing in
router/, infra/, or pipeline/ should ever import a concrete provider
module directly.

Also holds the helpers every provider shares for turning raw model text
into a GradingResult, so a quirk of one model's output (a ```json fence,
a confidence of "85" instead of 0.85, ...) is handled in exactly one
place instead of being re-solved -- or forgotten -- per provider.
"""

import json
import math
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterable, Optional

import os as _os

# Per-request timeout (seconds) for every provider SDK call; see config.PROVIDER_TIMEOUT_SECONDS.
PROVIDER_TIMEOUT_SECONDS = float(_os.environ.get("PROVIDER_TIMEOUT_SECONDS", "120"))

VALID_CLASSIFICATIONS = frozenset({"CORRECT", "INCORRECT", "PARTIALLY_CORRECT", "INCORRECT_WITH_MISCONCEPTION"})

def output_token_limit(default: int) -> int:
    """
    The max_output_tokens a provider sends. HWCHECKER_MAX_OUTPUT_TOKENS (when set to a number)
    replaces `default` -- a cost guard against runaway replies. It never goes below 1024:
    Gemini counts its hidden thinking tokens against this limit and a JSON reply cut off
    half-way is unusable (it would be retried, i.e. paid for twice).
    """
    raw = _os.environ.get("HWCHECKER_MAX_OUTPUT_TOKENS", "").strip()
    if raw.isdigit():
        return max(1024, int(raw))
    return default


_THINKING_OFF_WORDS = frozenset({"off", "none", "false", "disabled", "disable", "no", "0"})
_THINKING_DEFAULT_WORDS = frozenset({"", "default", "auto", "provider"})


def normalize_thinking(value) -> Optional[str]:
    """
    One spelling for the per-tier "how much hidden reasoning" setting.

    Returns None for "leave the provider's own default alone", "off" for "no thinking at all",
    otherwise the lower-cased level or token budget as a string (e.g. "low", "medium", "high",
    "max", "1024"). Each provider maps these onto its own API knob (Gemini `thinking_level`,
    Qwen `enable_thinking`/`thinking_budget`, GLM `thinking`/`reasoning_effort`, Claude
    `output_config.effort`); a level that provider does not know is ignored, never an error.
    """
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in _THINKING_DEFAULT_WORDS:
        return None
    if text in _THINKING_OFF_WORDS:
        return "off"
    return text


@dataclass
class GradingResult:
    """
    Structured output of one grading call. This is the ONE shape every
    provider must return, regardless of which underlying API produced it
    -- everything downstream (router, verification, JSON schema
    validation, storage) depends on this being consistent.
    """

    is_correct: bool
    classification: str  # "CORRECT" | "INCORRECT" | "PARTIALLY_CORRECT" | "INCORRECT_WITH_MISCONCEPTION"
    score: float          # 0-100
    error_type: Optional[str] = None
    error_location: Optional[str] = None
    feedback: str = ""
    sub_scores: dict = field(default_factory=dict)   # e.g. {"content": 80, "grammar": 70} for languages
    confidence: float = 1.0                            # 0.0-1.0; drives secondary verification
    raw_tokens: dict = field(default_factory=dict)      # {"input": int, "output": int}
    latency_ms: int = 0
    provider_name: str = ""
    # Optional per-question breakdown (see normalise_questions() for the item shape) and
    # teacher-facing summary. Empty when the model gave none, or for deterministic results.
    questions: list = field(default_factory=list)
    summary: dict = field(default_factory=dict)
    # Set only when a second-opinion pass ran: both verdicts and whether they disagreed, so the
    # teacher-facing result can flag the cases where two models did not agree.
    verification: dict = field(default_factory=dict)


class AIProviderError(Exception):
    """Raised by a provider when a grading call fails after its own retries."""

    #: False = retrying the same call cannot help (and it says nothing about the provider's
    #: health), e.g. an unsupported input type. The router skips retries and the circuit
    #: breaker for these.
    retryable: bool = True


class ProviderUnavailableError(AIProviderError):
    """The provider cannot be used at all in this deployment (API key missing, SDK not installed)."""

    retryable = False


class UnsupportedInputError(AIProviderError):
    """This provider/model cannot accept the given input (e.g. an image for a text-only model)."""

    retryable = False


class AIProvider(ABC):
    """
    Every concrete provider wraps one underlying AI API (or, for
    SelfHostedProvider, your own server) behind this same interface.
    """

    #: Set by subclasses -- used in logs/GradingResult.provider_name and
    #: for cost-tracking; should be short and stable (e.g. "gemini-3-flash").
    name: str = "unknown"

    #: What this provider can read. The router uses these to avoid sending a photo
    #: to a text-only model or a recording to a model with no audio input.
    supports_vision: bool = True
    supports_audio: bool = False

    @abstractmethod
    def grade_homework(self, content: dict, rubric: dict) -> GradingResult:
        """
        content: {"text": str | None, "images": [bytes|str, ...] | None,
                   "audio": bytes | None, "question": str, "student_answer": str, ...}
                  -- exact keys are provider-agnostic; each provider maps
                  what it needs out of this dict to its own API call shape.
        rubric: the dict returned by SubjectProfile.build_rubric() (see
                subjects/base.py) -- criteria, error taxonomy, grading_paradigm.
        Returns a GradingResult. Raises AIProviderError if grading
        ultimately fails (network error, malformed response after
        retries, etc.) -- callers (grading_router) decide what to do
        next (retry a different provider, fail the job, ...).
        """
        raise NotImplementedError

    @abstractmethod
    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Returns an estimated USD cost for a call with these token
        counts, using this provider's published per-token pricing.
        Used for cost logging/monitoring, not for billing decisions."""
        raise NotImplementedError

    def health_check(self) -> bool:
        """
        Cheap reachability check (NOT a full grading call) used by the
        circuit breaker / monitoring to decide if this provider is
        currently usable. Default implementation always returns True;
        override with a real lightweight check (e.g. a models.list()
        call) where the underlying SDK makes that easy and cheap.
        """
        return True

    def _timed_call(self, fn, *args, **kwargs):
        """Helper: runs fn(*args, **kwargs), returns (result, latency_ms)."""
        start = time.monotonic()
        result = fn(*args, **kwargs)
        latency_ms = int((time.monotonic() - start) * 1000)
        return result, latency_ms


# ---------------------------------------------------------------------------
# Turning raw model text into a GradingResult
# ---------------------------------------------------------------------------
_THINK_RE = re.compile(r"<think(?:ing)?>.*?</think(?:ing)?>", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"```[A-Za-z0-9_-]*[ \t]*\r?\n?(.*?)```", re.DOTALL)


def extract_json_object(raw, prefer_keys: Iterable[str] = ()) -> dict:
    """
    Pulls the JSON object out of a model reply. Models asked for "ONLY JSON" still
    sometimes wrap it in a ```json fence, prepend a sentence, or (reasoning models)
    leave a <think> block in front -- a plain json.loads() on any of those fails and,
    in the router, silently turned a good grade into a skipped verification.

    `prefer_keys`: when the reply holds more than one JSON object, return the first one
    containing any of these keys. Raises ValueError when no JSON object is found.
    """
    if isinstance(raw, dict):
        return raw
    if raw is None:
        raise ValueError("the model returned no text")
    text = _THINK_RE.sub("", str(raw)).strip()
    if not text:
        raise ValueError("the model returned an empty reply")
    prefer = tuple(prefer_keys)

    candidates = [text]
    candidates.extend(m.strip() for m in _FENCE_RE.findall(text))
    first_dict = None
    for cand in candidates:
        try:
            obj = json.loads(cand)
        except ValueError:
            continue
        if isinstance(obj, dict):
            if not prefer or any(k in obj for k in prefer):
                return obj
            first_dict = first_dict or obj

    # Last resort: a JSON object embedded in surrounding prose.
    decoder = json.JSONDecoder()
    start = text.find("{")
    while start != -1:
        try:
            obj, _ = decoder.raw_decode(text, start)
        except ValueError:
            obj = None
        if isinstance(obj, dict):
            if not prefer or any(k in obj for k in prefer):
                return obj
            first_dict = first_dict or obj
        start = text.find("{", start + 1)

    if first_dict is not None:
        return first_dict
    raise ValueError("no JSON object found in the model's reply")


def _to_float(value, name: str) -> float:
    if isinstance(value, bool) or value is None:
        raise ValueError(f"'{name}' is missing or not a number: {value!r}")
    if isinstance(value, str):
        value = value.strip().rstrip("%").strip()
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"'{name}' is not a number: {value!r}") from None
    if math.isnan(number) or math.isinf(number):
        raise ValueError(f"'{name}' is not a finite number: {value!r}")
    return number


def _str_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, (str, int, float)):
        value = [value]
    if not isinstance(value, (list, tuple)):
        return []
    return [str(v).strip() for v in value if v is not None and str(v).strip()]


def normalise_questions(raw, default_score: float = 0.0) -> list:
    """
    Cleans the optional per-question list a model returns. Unusable entries are dropped
    and missing/odd fields get safe defaults, so downstream code (report generation, the
    CRM adapter) never has to defend itself against a half-filled item.

    Item shape: question_number, question, student_answer, correct_answer, score (0-100),
    mistakes [str], error_categories [str], analysis, suggestions [str]
    (+ expected_solution / difficulty when the model supplied them).
    """
    if not isinstance(raw, (list, tuple)):
        return []
    questions = []
    for index, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            continue
        try:
            score = min(max(_to_float(item.get("score"), "score"), 0.0), 100.0)
        except ValueError:
            score = float(default_score)
        try:
            number = int(item.get("question_number", index))
        except (TypeError, ValueError):
            number = index
        q = {
            "question_number": number,
            "question": str(item.get("question") or ""),
            "student_answer": str(item.get("student_answer") or ""),
            "correct_answer": str(item.get("correct_answer") or ""),
            "score": score,
            "mistakes": _str_list(item.get("mistakes")),
            "error_categories": _str_list(item.get("error_categories") or item.get("error_types")),
            "analysis": str(item.get("analysis") or item.get("feedback") or ""),
            "suggestions": _str_list(item.get("suggestions")),
        }
        for optional in ("expected_solution", "difficulty"):
            if item.get(optional):
                q[optional] = str(item[optional])
        questions.append(q)
    return questions


def _normalise_summary(raw) -> dict:
    if not isinstance(raw, dict):
        return {}
    keys = ("strengths", "weaknesses", "topics_to_review", "recommendations")
    summary = {key: _str_list(raw.get(key)) for key in keys}
    return summary if any(summary.values()) else {}


def count_or_zero(value) -> int:
    """A token count from a provider usage object, or 0 when it is absent / not a plain number."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    return max(int(value), 0)


def build_grading_result(
    data: dict,
    *,
    default_confidence: float,
    input_tokens: int,
    output_tokens: int,
    latency_ms: int,
    provider_name: str,
    cached_input_tokens: int = 0,
) -> GradingResult:
    """
    Validates + normalises the parsed model JSON into a GradingResult. Raises
    AIProviderError (retryable) when the reply is unusable. A missing or unknown
    classification used to be silently turned into "INCORRECT" -- which would hand a
    student a failing grade for a malformed reply -- so it is now an error instead,
    and the router retries / falls back.
    """
    try:
        classification = str(data.get("classification", "")).strip().upper().replace(" ", "_").replace("-", "_")
        if classification not in VALID_CLASSIFICATIONS:
            raise ValueError(f"'classification' must be one of {sorted(VALID_CLASSIFICATIONS)}, got {data.get('classification')!r}")
        score = min(max(_to_float(data.get("score"), "score"), 0.0), 100.0)

        confidence = default_confidence
        if data.get("confidence") is not None:
            confidence = _to_float(data.get("confidence"), "confidence")
            if 1.0 < confidence <= 100.0:   # "85" meaning 85% -- clamping would turn it into 100% sure
                confidence /= 100.0
            confidence = min(max(confidence, 0.0), 1.0)
    except ValueError as exc:
        raise AIProviderError(f"{provider_name} returned an unusable grading reply: {exc}") from exc

    sub_scores = data.get("sub_scores")
    feedback = data.get("feedback", "")
    if isinstance(feedback, (list, tuple)):
        feedback = "\n".join(str(f) for f in feedback)

    def _opt_str(value):
        return None if value in (None, "", "null", "None") else str(value)

    return GradingResult(
        is_correct=classification == "CORRECT",
        classification=classification,
        score=score,
        error_type=_opt_str(data.get("error_type")),
        error_location=_opt_str(data.get("error_location")),
        feedback=str(feedback or ""),
        sub_scores=sub_scores if isinstance(sub_scores, dict) else {},
        confidence=confidence,
        raw_tokens={
            "input": int(input_tokens or 0), "output": int(output_tokens or 0),
            # Input tokens the provider served from its prompt cache (billed at a discount);
            # a SUBSET of "input". Lets us measure whether the shared assignment prefix is cached.
            "cached": count_or_zero(cached_input_tokens),
        },
        latency_ms=latency_ms,
        provider_name=provider_name,
        questions=normalise_questions(data.get("questions"), default_score=score),
        summary=_normalise_summary(data.get("summary")),
    )
