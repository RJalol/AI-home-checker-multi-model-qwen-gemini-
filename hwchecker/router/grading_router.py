"""
hwchecker/router/grading_router.py
GradingRouter.grade(homework) -- the core orchestration, matching spec
section 7's pseudocode:

    def grade(self, homework):
        profile = SUBJECT_REGISTRY[homework.subject_id]
        if profile.deterministic_checker:
            result = profile.deterministic_checker(homework)
            if result is not None:
                return result                          # $0, LLM chaqirilmadi
        difficulty = self._detect_difficulty(homework, profile.topic_weights)
        provider = self._pick_provider(difficulty, profile.subject_id)
        result = provider.grade_homework(homework.content, profile.build_rubric(homework))
        if self._needs_verification(result):           # past-ishonch/chegara-ball (~20%)
            result = self._verify(result, homework, profile)
        return result

Also implements the MCQ autograder as an even-earlier $0 short-circuit
(checkers/mcq_autograder.py runs before any subject-specific
deterministic_checker, since it applies across every subject uniformly).

On top of that pseudocode, the router now also:
  * routes by what the provider can READ -- a photo / scanned PDF is never sent to a
    text-only model (GLM), audio only to an audio-capable one; the next capable tier
    is used instead;
  * routes by what is CONFIGURED -- a tier whose API key is missing is skipped in favour of
    the next usable provider, and only if none is usable does it raise one clear
    GradingRouterError listing what is missing;
  * retries a failed call with backoff (infra/retry.py), skips a provider whose circuit is
    open (infra/circuit_breaker.py), can pace calls per provider (infra/rate_limiter.py),
    logs tokens / cost / latency of every call (infra/monitoring.py) and serves an identical
    resubmission from a result cache (infra/cache.py). All of that degrades quietly to
    in-process state when Redis is not available.
"""

import contextvars
import os
import time
from pathlib import Path
from typing import Optional

from hwchecker.checkers import mcq_autograder
from hwchecker.infra import budget
from hwchecker.infra import cache as result_cache
from hwchecker.infra import circuit_breaker, rate_limiter
from hwchecker.infra.monitoring import log_grading_event
from hwchecker.infra.retry import RetryExhaustedError, call_with_retry
from hwchecker.pipeline.ocr import needs_vision_ocr
from hwchecker.pipeline.schema import grading_result_to_schema
from hwchecker.providers.base import AIProviderError, GradingResult
from hwchecker.providers.registry import allow_same_vendor_verification, config_fingerprint, get_provider
from hwchecker.subjects.registry import get_subject

from .difficulty import detect_difficulty


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    try:
        return float(value)
    except ValueError:
        return default


# Results below this confidence get a second opinion from the "verification" tier provider
# (Claude Haiku 5.5 by default). Every second opinion is an extra paid call, so this and the
# margin below are the main dial on how many submissions are graded twice: lower = fewer
# re-checks = cheaper. Tune from your own logs (infra/monitoring.py) -- the share of results
# carrying a "verification" block is the measured re-check rate.
VERIFICATION_CONFIDENCE_THRESHOLD = _env_float("HWCHECKER_VERIFY_CONFIDENCE_THRESHOLD", 0.70)
# Scores this close to a classification-band boundary also trigger
# verification, even if reported confidence was high -- a self-reported
# 0.9 confidence next to a 69.5/100 score (right at the "partially
# correct" boundary) is exactly the kind of case a second opinion helps most.
BORDERLINE_SCORE_MARGIN = _env_float("HWCHECKER_VERIFY_SCORE_MARGIN", 2.0)
_BAND_BOUNDARIES = (40, 70, 90)  # matches the 4-classification scoring bands used across this project

# A result from one of these tiers is never re-checked: they already are the strongest model
# available, so a (cheaper) verifier overriding them would make the answer worse, not better.
_NO_VERIFY_TIERS = frozenset({"hard_vision", "verification", "verification_gemini", "verification_qwen"})

# Second opinions are tried in this order; the first one that is configured, can read the input
# and comes from a DIFFERENT vendor than the grader wins. With only Qwen and Gemini keys that
# makes the two check each other (Qwen's grades by Gemini, Gemini's by Qwen).
_VERIFICATION_TIERS = ("verification", "verification_gemini", "verification_qwen")

_DIFFICULTY_TO_TIER = {"easy": "easy", "medium": "medium", "hard": "hard"}

# When the tier a submission was routed to is unusable for it (API key missing, cannot read
# photos), these tiers are tried instead, in this order. Hard questions go to the strongest
# vision model first; everything else to the cheaper all-rounder.
_SUBSTITUTE_TIERS = {
    "hard": ("hard_vision", "medium", "medium_alt", "verification", "fallback"),
    "default": ("medium", "medium_alt", "fallback", "verification"),
}
# Recordings go to the audio tier first: the cheap "medium" model is not strong enough to
# assess speech, and the text/vision tiers cannot hear at all.
_AUDIO_TIERS = ("audio",)
# After a call has failed for good, these tiers are tried (in this order) before giving up.
_FALLBACK_TIERS = ("fallback", "medium", "medium_alt", "verification")
MAX_FALLBACK_PROVIDERS = 2

MAX_ATTEMPTS = 3                  # per provider (the first call + 2 retries)
RETRY_BACKOFF_SECONDS = (2, 4)
FALLBACK_MAX_ATTEMPTS = 2
RATE_LIMIT_MAX_WAIT_SECONDS = 15  # how long to wait for a free slot before going ahead anyway

_AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".ogg"}


# RQ's per-job deadline exception derives from Exception: it must NOT be treated as "this provider failed"
from hwchecker.infra.deadline import JobDeadline as _JobDeadline  # noqa: E402


# Every provider call of the homework being graded right now (one list per grade() call, per thread/task).
_CALL_LEDGER: contextvars.ContextVar = contextvars.ContextVar("hwchecker_call_ledger", default=None)


def _usage_summary(calls: Optional[list]) -> dict:
    """What one homework used: every provider call (grader, verifier, retries that returned a result)
    and the totals. The cost is the provider's list-price estimate from input and output tokens; it does not
    apply a cache discount, so with prompt caching the real bill is lower."""
    calls = list(calls or [])
    return {
        "calls": calls,
        "call_count": len(calls),
        "input_tokens": sum(c["input_tokens"] for c in calls),
        "output_tokens": sum(c["output_tokens"] for c in calls),
        "cached_tokens": sum(c["cached_tokens"] for c in calls),
        "estimated_cost_usd": round(sum(c["estimated_cost_usd"] for c in calls), 6),
        "latency_ms": sum(c["latency_ms"] for c in calls),
    }


def _vendor_of(provider) -> Optional[str]:
    """The vendor behind a provider: its `vendor` attribute if it has one, else the name of the module
    that defines its class (providers/qwen.py -> "qwen", providers/gemini.py -> "gemini", ...).
    Providers did not set `vendor`, so before this the "never the grader's own vendor" rule did nothing."""
    if provider is None:
        return None
    declared = getattr(provider, "vendor", None)
    if isinstance(declared, str):
        return declared
    module = type(provider).__module__
    return module.rsplit(".", 1)[-1] if module.startswith("hwchecker.providers.") else None


class GradingRouterError(Exception):
    pass


class _NonRetryable(Exception):
    """Wraps an AIProviderError that retrying cannot fix, so call_with_retry lets it straight through."""

    def __init__(self, original: Exception):
        super().__init__(str(original))
        self.original = original


def _input_kind(content: dict) -> str:
    """What a provider must be able to read to grade this submission: "text" | "vision" | "audio"."""
    file_path = content.get("file_path")
    if not file_path:
        return "text"
    if Path(file_path).suffix.lower() in _AUDIO_EXTENSIONS:
        return "audio"
    return "vision" if needs_vision_ocr(file_path) else "text"


def _can_read(provider, kind: str) -> bool:
    if kind == "vision":
        return bool(getattr(provider, "supports_vision", True))
    if kind == "audio":
        return bool(getattr(provider, "supports_audio", False))
    return True


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def _rate_limit_per_minute() -> int:
    try:
        return max(0, int(os.environ.get("HWCHECKER_RATE_LIMIT_PER_MINUTE", "0")))
    except ValueError:
        return 0


class GradingRouter:
    def __init__(
        self,
        enable_verification: bool = True,
        enable_cache: Optional[bool] = None,
        max_attempts: int = MAX_ATTEMPTS,
        backoff_seconds: tuple = RETRY_BACKOFF_SECONDS,
    ):
        self.enable_verification = enable_verification
        # None -> HWCHECKER_ENABLE_CACHE (default on). An explicit True/False wins.
        self.enable_cache = _env_flag("HWCHECKER_ENABLE_CACHE", True) if enable_cache is None else enable_cache
        self.max_attempts = max_attempts
        self.backoff_seconds = backoff_seconds

    # -- Step 1: deterministic checkers (spec section 6, step 2) --------
    def _try_deterministic(self, homework: dict, subject_profile) -> Optional[GradingResult]:
        mcq_result = mcq_autograder.check(homework)
        if mcq_result is not None:
            return mcq_result

        if subject_profile.deterministic_checker:
            try:
                result = subject_profile.deterministic_checker(homework)
                if result is not None:
                    return result
            except _JobDeadline:
                raise
            except Exception:
                pass  # a broken deterministic checker should never block grading -- fall through to the LLM tiers
        return None

    # -- Step 2: difficulty -> provider tier -----------------------------
    def _detect_difficulty(self, homework: dict, subject_profile) -> str:
        return detect_difficulty(homework, subject_profile)

    def _load_tier(self, tier: str, kind: str):
        """(provider, None) when the tier is configured and can read `kind`, else (None, reason)."""
        try:
            provider = get_provider(tier)
        except AIProviderError as exc:
            return None, f"{tier}: {exc}"
        if not _can_read(provider, kind):
            return None, f"{tier} ({provider.name}) cannot read {kind} input"
        return provider, None

    def _pick_provider(self, difficulty: str, subject_id: str, kind: str = "text"):
        tier = _DIFFICULTY_TO_TIER.get(difficulty, "medium")
        substitutes = _SUBSTITUTE_TIERS["hard" if tier == "hard" else "default"]
        first = _AUDIO_TIERS if kind == "audio" else ()
        problems = []
        order = first + (tier,) + tuple(t for t in substitutes if t != tier)
        for candidate in dict.fromkeys(order):   # de-duplicated, order kept
            provider, why = self._load_tier(candidate, kind)
            if provider is not None:
                return provider
            problems.append(why)
        raise GradingRouterError(
            "No AI provider can grade this submission. " + "; ".join(problems)
            + ". Set the API key of at least one provider (see .env.example)."
        )

    # -- calling one provider: retry + circuit breaker + rate limit + monitoring ----
    def _respect_rate_limit(self, provider_name: str) -> None:
        limit = _rate_limit_per_minute()
        if limit <= 0:
            return
        try:
            for _ in range(RATE_LIMIT_MAX_WAIT_SECONDS):
                if rate_limiter.check_and_increment(provider_name, limit, 60):
                    return
                time.sleep(1)
        except _JobDeadline:
            raise   # this loop can wait for a while: the job deadline must get through
        except Exception:
            pass  # pacing is best-effort; it must never block grading

    def _log_call(self, provider, result: Optional[GradingResult], started: float, exc: Optional[Exception]) -> None:
        try:
            latency_ms = int((time.monotonic() - started) * 1000)
            if result is not None:
                tokens = result.raw_tokens or {}
                tokens_in, tokens_out = tokens.get("input", 0), tokens.get("output", 0)
                cost = provider.estimate_cost(tokens_in, tokens_out)
                ledger = _CALL_LEDGER.get()
                if ledger is not None:
                    ledger.append({
                        "provider": _vendor_of(provider), "model": provider.name,
                        "input_tokens": tokens_in, "output_tokens": tokens_out,
                        "cached_tokens": tokens.get("cached", 0), "estimated_cost_usd": round(cost, 6),
                        "latency_ms": result.latency_ms or latency_ms,
                    })
                log_grading_event(provider.name, tokens_in, tokens_out, cost, result.latency_ms or latency_ms, True)
                budget.record_spend(cost)
            else:
                log_grading_event(provider.name, 0, 0, 0.0, latency_ms, False, type(exc).__name__ if exc else None)
        except Exception:
            pass  # monitoring must never break grading

    def _attempt(self, provider, content: dict, rubric: dict) -> GradingResult:
        """One call to one provider. Raises AIProviderError (worth retrying), _NonRetryable, or CircuitOpenError."""
        if circuit_breaker.is_open(provider.name):
            raise circuit_breaker.CircuitOpenError(
                f"Circuit breaker is open for provider '{provider.name}' -- too many recent failures."
            )
        self._respect_rate_limit(provider.name)
        started = time.monotonic()
        try:
            result = provider.grade_homework(content, rubric)
        except AIProviderError as exc:
            self._log_call(provider, None, started, exc)
            if not getattr(exc, "retryable", True):
                raise _NonRetryable(exc) from exc   # the request's fault, not the provider's health
            circuit_breaker.record_failure(provider.name)
            raise
        except _JobDeadline:
            raise   # the worker's time budget is spent: stop now, do not start another provider call
        except Exception as exc:   # a bug or I/O problem we did not foresee: do not retry, let another provider try
            self._log_call(provider, None, started, exc)
            raise _NonRetryable(AIProviderError(f"{provider.name} failed unexpectedly: {exc!r}")) from exc
        circuit_breaker.record_success(provider.name)
        self._log_call(provider, result, started, None)
        return result

    def _call_provider(self, provider, content: dict, rubric: dict, max_attempts: Optional[int] = None) -> GradingResult:
        """Runs `provider` with retries. Raises AIProviderError if it never produces a usable result."""
        try:
            return call_with_retry(
                self._attempt, provider, content, rubric,
                max_attempts=max_attempts or self.max_attempts,
                backoff_seconds=self.backoff_seconds,
                retry_on=(AIProviderError,),
            )
        except RetryExhaustedError as exc:
            raise exc.last_error
        except _NonRetryable as exc:
            raise exc.original
        except circuit_breaker.CircuitOpenError as exc:
            raise AIProviderError(str(exc)) from exc

    # -- Step 3: needs a second opinion? ---------------------------------
    def _needs_verification(self, result: GradingResult, graded_by=None) -> bool:
        if not self.enable_verification:
            return False
        if getattr(graded_by, "tier", None) in _NO_VERIFY_TIERS:
            return False   # already graded by the strongest model: nobody cheaper can overrule it
        if result.confidence < VERIFICATION_CONFIDENCE_THRESHOLD:
            return True
        for boundary in _BAND_BOUNDARIES:
            if abs(result.score - boundary) <= BORDERLINE_SCORE_MARGIN:
                return True
        return False

    def _pick_verifier(self, graded_by, kind: str):
        """
        The provider that gives the second opinion, or None when there is no suitable one.
        It must be configured, able to read this kind of input, and not the grader's own model or
        vendor: a model re-checking a sibling from the same vendor tends to repeat its mistakes.
        """
        grader_name = getattr(graded_by, "name", None)
        grader_vendor = _vendor_of(graded_by)
        for tier in _VERIFICATION_TIERS:
            try:
                candidate = get_provider(tier)
            except AIProviderError:
                continue   # no key / SDK for this tier: try the next vendor
            if grader_name is not None and candidate.name == grader_name:
                continue   # asking the same model again is not a second opinion
            if (grader_vendor is not None and _vendor_of(candidate) == grader_vendor
                    and not allow_same_vendor_verification()):
                continue
            if not _can_read(candidate, kind):
                continue
            return candidate
        return None

    def _verify(self, result: GradingResult, homework: dict, subject_profile, graded_by=None, kind: str = "text") -> GradingResult:
        """
        Re-grades with the provider _pick_verifier() chooses (Claude Haiku 5.5
        when available and not the grader's vendor, else Gemini/Qwen). If the second opinion's classification/score
        disagrees meaningfully with the first, the verification result
        wins (it's an independent second look, specifically invoked because
        the first result wasn't confident) -- but we keep the average of
        the two scores when they're both plausible and close, since
        neither result should be assumed infallible.

        Returns the very same `result` object when no second opinion could be obtained
        (verifier not configured, cannot read this input, is the model that already graded
        it, or failed) -- callers use that identity to tell "verified" from "could not verify".
        """
        verifier = self._pick_verifier(graded_by, kind)
        if verifier is None:
            return result   # nobody independent can check this -- keep the first result un-verified rather than failing the grade

        content = dict(homework.get("content") or {})
        content["system_prompt"] = (
            content.get("system_prompt", "")
            + f"\n\nAnother grading pass scored this {result.score}/100 "
            f"({result.classification}) with confidence {result.confidence:.2f}. "
            f"Independently re-grade it yourself -- do not simply defer to that prior result."
        )
        rubric = subject_profile.build_rubric(homework)
        try:
            verified = self._call_provider(verifier, content, rubric, max_attempts=FALLBACK_MAX_ATTEMPTS)
        except AIProviderError:
            # Verification failing shouldn't discard a valid first result --
            # just return it as-is, un-verified, rather than failing the whole grade.
            return result

        agreed = verified.classification == result.classification and abs(verified.score - result.score) <= 5
        first_summary = {
            "provider": result.provider_name, "classification": result.classification,
            "score": result.score, "confidence": result.confidence,
        }
        second_summary = {
            "provider": verified.provider_name, "classification": verified.classification,
            "score": verified.score, "confidence": verified.confidence,
        }
        verification = {
            "outcome": "agreed" if agreed else "verifier_overrode",
            "classification_changed": verified.classification != result.classification,
            "score_difference": round(abs(verified.score - result.score), 1),
            "first_pass": first_summary,
            "second_pass": second_summary,
        }

        if agreed:
            merged = GradingResult(
                is_correct=verified.is_correct,
                classification=verified.classification,
                score=round((verified.score + result.score) / 2, 1),
                error_type=verified.error_type or result.error_type,
                error_location=verified.error_location or result.error_location,
                feedback=verified.feedback or result.feedback,
                sub_scores=verified.sub_scores or result.sub_scores,
                confidence=max(verified.confidence, result.confidence),
                raw_tokens={
                    "input": result.raw_tokens.get("input", 0) + verified.raw_tokens.get("input", 0),
                    "output": result.raw_tokens.get("output", 0) + verified.raw_tokens.get("output", 0),
                    "cached": result.raw_tokens.get("cached", 0) + verified.raw_tokens.get("cached", 0),
                },
                latency_ms=result.latency_ms + verified.latency_ms,
                provider_name=f"{result.provider_name}+{verified.provider_name}(verified)",
                questions=verified.questions or result.questions,
                summary=verified.summary or result.summary,
                verification=verification,
            )
            return merged

        # The two passes disagree: the verifier's whole verdict (including its own per-question
        # breakdown) replaces the first one -- mixing the two would contradict itself.
        verified.raw_tokens = {
            "input": result.raw_tokens.get("input", 0) + verified.raw_tokens.get("input", 0),
            "output": result.raw_tokens.get("output", 0) + verified.raw_tokens.get("output", 0),
            "cached": result.raw_tokens.get("cached", 0) + verified.raw_tokens.get("cached", 0),
        }
        verified.latency_ms = result.latency_ms + verified.latency_ms
        verified.provider_name = f"{verified.provider_name}(verified, overrode {result.provider_name})"
        verified.verification = verification
        return verified

    # -- Step 4: fallback on provider failure ----------------------------
    def _fallback_providers(self, kind: str, tried_names: set):
        """Yields the other usable providers to try after a failure, at most MAX_FALLBACK_PROVIDERS."""
        yielded = 0
        for tier in _FALLBACK_TIERS:
            if yielded >= MAX_FALLBACK_PROVIDERS:
                return
            provider, _ = self._load_tier(tier, kind)
            if provider is None or provider.name in tried_names:
                continue
            tried_names.add(provider.name)
            yielded += 1
            yield provider

    def _grade_with_fallback(self, provider, content: dict, rubric: dict, kind: str = "text"):
        """Returns (GradingResult, the provider that produced it)."""
        try:
            return self._call_provider(provider, content, rubric), provider
        except AIProviderError as primary_error:
            failures = [f"{provider.name}: {primary_error}"]

        for alternate in self._fallback_providers(kind, {provider.name}):
            try:
                result = self._call_provider(alternate, content, rubric, max_attempts=FALLBACK_MAX_ATTEMPTS)
            except AIProviderError as exc:
                failures.append(f"{alternate.name}: {exc}")
                continue
            result.provider_name = f"{result.provider_name}(fallback, {provider.name} failed)"
            return result, alternate

        raise GradingRouterError(
            f"The primary provider ({provider.name}) and every fallback provider failed. " + " | ".join(failures)
        )

    # -- result cache -------------------------------------------------------
    def _cache_key(self, homework: dict, content: dict, subject_id: str) -> Optional[str]:
        if not self.enable_cache:
            return None
        file_path = content.get("file_path")
        try:
            if not file_path or not Path(file_path).is_file():
                return None
            # Everything that changes the verdict goes into the key: the file, the subject/skill, both
            # prompts (they carry the rubric, the teacher's extra instructions and the answer language).
            context = "\n".join([
                str(content.get("user_prompt", "")),
                str(content.get("system_prompt", "")),
                f"verification={self.enable_verification}",
                # a different model/thinking set-up must not be answered from an older set-up's result
                f"config={config_fingerprint()}",
                f"thresholds={VERIFICATION_CONFIDENCE_THRESHOLD}/{BORDERLINE_SCORE_MARGIN}",
            ])
            return result_cache.compute_submission_hash(
                file_path, f"{subject_id}:{homework.get('skill') or ''}", context
            )
        except Exception:
            return None

    def _cache_get(self, key: Optional[str]) -> Optional[dict]:
        if not key:
            return None
        try:
            hit = result_cache.get(key)
        except Exception:
            return None
        if not hit:
            return None
        payload = dict(hit)
        payload["tokens"] = {"input": 0, "output": 0}   # nothing was spent on this one
        payload["latency_ms"] = 0
        payload["cached"] = True
        return payload

    def _cache_put(self, key: Optional[str], payload: dict) -> None:
        if not key:
            return
        try:
            result_cache.set(key, payload)
        except Exception:
            pass

    # -- Public entry point -----------------------------------------------
    def grade(self, homework: dict) -> dict:
        """Grades one homework. The result carries a "usage" block: tokens, estimated cost and latency of
        every provider call this homework needed (zero calls for a cache hit or a deterministic check)."""
        ledger: list = []
        token = _CALL_LEDGER.set(ledger)
        try:
            payload = self._grade(homework)
        finally:
            _CALL_LEDGER.reset(token)
        payload = dict(payload)
        payload["usage"] = _usage_summary(ledger)
        return payload

    def _grade(self, homework: dict) -> dict:
        """
        homework: {
          "subject_id": str, "content": {"file_path": Path, "user_prompt": str, "system_prompt": str, ...},
          "student_answer": str, "gold_answer": str (optional, for deterministic checkers),
          "question_type": str (optional, "mcq"/"grammar_correction"/...),
          "difficulty_hint": str (optional), "rubric_step_count": int (optional),
          ...
        }
        Returns a dict matching pipeline/schema.py's REQUIRED_FIELDS --
        the same shape regardless of which path (deterministic/easy/
        medium/hard/verified) actually produced it. It also carries "questions" and "summary"
        when the model produced a per-question breakdown, and "cached": True when an identical
        earlier submission was served from the cache.
        """
        subject_id = homework.get("subject_id")
        if not subject_id:
            raise GradingRouterError("homework['subject_id'] is required.")
        subject_profile = get_subject(subject_id)

        # Step 1: deterministic checkers -- $0, no LLM call if they resolve it.
        deterministic_result = self._try_deterministic(homework, subject_profile)
        if deterministic_result is not None:
            return grading_result_to_schema(deterministic_result)

        content = dict(homework.get("content") or {})
        content.setdefault("system_prompt", subject_profile.build_system_prompt(homework.get("skill")))
        rubric = subject_profile.build_rubric(homework)
        kind = _input_kind(content)
        prepared = {**homework, "content": content}

        # An identical resubmission costs nothing.
        cache_key = self._cache_key(homework, content, subject_id)
        # homework["bypass_cache"]=True (a teacher-requested regrade) skips the lookup, but the
        # fresh result still replaces the cached one below.
        cached = None if homework.get("bypass_cache") else self._cache_get(cache_key)
        if cached is not None:
            return cached

        # Step 2: difficulty -> tier -> a provider that is configured and can read this input.
        difficulty = self._detect_difficulty(prepared, subject_profile)
        provider = self._pick_provider(difficulty, subject_id, kind)
        result, graded_by = self._grade_with_fallback(provider, content, rubric, kind)

        # Step 3: low-confidence/borderline -> second opinion.
        needs_second_opinion = self._needs_verification(result, graded_by)
        second_opinion_obtained = True
        if needs_second_opinion:
            checked = self._verify(result, prepared, subject_profile, graded_by=graded_by, kind=kind)
            second_opinion_obtained = checked is not result   # _verify returns the same object when it could not verify
            result = checked

        payload = grading_result_to_schema(result)

        # Do not cache a result that was supposed to be double-checked but wasn't (verifier down):
        # an identical resubmission later should get the chance to be verified.
        if second_opinion_obtained:
            self._cache_put(cache_key, payload)
        return payload
