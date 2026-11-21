"""
hwchecker/pipeline/schema.py
The structured JSON schema every grading call must conform to, per spec
section 8. Used both to validate raw LLM output before it's trusted,
and to serialize a GradingResult (providers/base.py) into the final
response shape.

Where the validation actually happens: providers/base.py's
build_grading_result() rejects an unusable model reply (AIProviderError),
router/grading_router.py retries it with backoff (infra/retry.py), skips a
provider whose circuit is open (infra/circuit_breaker.py) and moves on to
the next provider. validate() below is the final gate on the payload that
leaves the router.

Beyond the required fields, a payload may carry the optional "questions"
(per-question breakdown) and "summary" (strengths/weaknesses/...) keys.
"""

from hwchecker.providers.base import VALID_CLASSIFICATIONS  # noqa: F401  (re-exported: benchmarks/runner.py imports it from here)

REQUIRED_FIELDS = {
    "classification", "score", "max_score", "error_type", "error_location",
    "feedback", "sub_scores", "confidence", "provider", "tokens", "latency_ms",
}


class SchemaValidationError(Exception):
    pass


def validate(payload: dict) -> None:
    """Raises SchemaValidationError with a specific, actionable message
    if `payload` doesn't conform to the required output schema. Returns
    None (no exception) if it's valid."""
    if not isinstance(payload, dict):
        raise SchemaValidationError(f"Expected a JSON object, got {type(payload).__name__}.")

    missing = REQUIRED_FIELDS - payload.keys()
    if missing:
        raise SchemaValidationError(f"Missing required fields: {sorted(missing)}")

    if payload["classification"] not in VALID_CLASSIFICATIONS:
        raise SchemaValidationError(
            f"classification must be one of {sorted(VALID_CLASSIFICATIONS)}, got {payload['classification']!r}"
        )

    score = payload["score"]
    if not isinstance(score, (int, float)) or not (0 <= score <= payload.get("max_score", 100)):
        raise SchemaValidationError(f"score must be a number between 0 and max_score, got {score!r}")

    confidence = payload["confidence"]
    if not isinstance(confidence, (int, float)) or not (0.0 <= confidence <= 1.0):
        raise SchemaValidationError(f"confidence must be a number between 0.0 and 1.0, got {confidence!r}")

    if not isinstance(payload["tokens"], dict) or "input" not in payload["tokens"] or "output" not in payload["tokens"]:
        raise SchemaValidationError('tokens must be {"input": int, "output": int}')

    if not isinstance(payload["sub_scores"], dict):
        raise SchemaValidationError("sub_scores must be an object (dict), even if empty.")

    if "questions" in payload:
        questions = payload["questions"]
        if not isinstance(questions, list) or not all(isinstance(q, dict) for q in questions):
            raise SchemaValidationError("questions must be a list of objects.")
    if "summary" in payload and not isinstance(payload["summary"], dict):
        raise SchemaValidationError("summary must be an object (dict).")


def grading_result_to_schema(result, max_score: float = 100.0) -> dict:
    """
    Converts a providers.base.GradingResult into the exact JSON shape
    from spec section 8. This is what gets JSON-schema-validated,
    stored, and returned to callers. The optional "questions" / "summary"
    keys are only included when the grader produced them.
    """
    payload = {
        "classification": result.classification,
        "score": result.score,
        "max_score": max_score,
        "error_type": result.error_type,
        "error_location": result.error_location,
        "feedback": result.feedback,
        "sub_scores": result.sub_scores or {},
        "confidence": result.confidence,
        "provider": result.provider_name,
        "tokens": {
            "input": result.raw_tokens.get("input", 0), "output": result.raw_tokens.get("output", 0),
            "cached": result.raw_tokens.get("cached", 0),
        },
        "latency_ms": result.latency_ms,
    }
    questions = getattr(result, "questions", None)
    if questions:
        payload["questions"] = questions
    summary = getattr(result, "summary", None)
    if summary:
        payload["summary"] = summary
    verification = getattr(result, "verification", None)
    if verification:
        payload["verification"] = verification
    validate(payload)
    return payload
