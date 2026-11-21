"""
token_usage_logger.py
Logs how many input/output/cached tokens each graded submission actually
used, so you can see real token spend broken down by subject, model,
backend, day, etc. -- separate from training_logger.py, which stores the
grading content itself rather than cost/usage numbers.

Each record is one line of JSON appended to config.TOKEN_USAGE_LOG:
    {
      "timestamp": "...", "backend": "gemini", "model_name": "...",
      "subject_key": "...", "language_key": "...", "skill_key": "...",
      "input_tokens": 1234, "output_tokens": 567,
      "cached_tokens": 1100, "total_tokens": 1801,
    }

"cached_tokens" (from Gemini's context caching) is a subset of
input_tokens, not additional -- it tells you how much of the input was
billed at the cheaper cached rate vs. the normal rate.
"""

import json
from datetime import datetime, timezone
from typing import Optional

from config import TOKEN_USAGE_LOG, TRACK_TOKEN_USAGE


def log_usage(
    input_tokens: int = 0,
    output_tokens: int = 0,
    cached_tokens: int = 0,
    total_tokens: Optional[int] = None,
    backend: str = "",
    model_name: str = "",
    subject_key: str = "",
    custom_subject_name: str = "",
    language_key: str = "",
    skill_key: str = "",
) -> None:
    """
    Appends one usage record. Never raises -- a logging failure should
    never break homework grading for whoever's waiting on a result.
    """
    if not TRACK_TOKEN_USAGE:
        return
    try:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "backend": backend,
            "model_name": model_name,
            "subject_key": subject_key,
            "custom_subject_name": custom_subject_name,
            "language_key": language_key,
            "skill_key": skill_key,
            "input_tokens": input_tokens or 0,
            "output_tokens": output_tokens or 0,
            "cached_tokens": cached_tokens or 0,
            "total_tokens": total_tokens if total_tokens is not None else (input_tokens or 0) + (output_tokens or 0),
        }
        with open(TOKEN_USAGE_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass


def get_usage_summary() -> dict:
    """
    Aggregates every logged record into totals, plus breakdowns by
    subject/skill and by model. Used by the Streamlit sidebar and the
    GET /usage API endpoint.
    """
    totals = {"input_tokens": 0, "output_tokens": 0, "cached_tokens": 0, "total_tokens": 0, "requests": 0}
    by_subject: dict = {}
    by_model: dict = {}

    if not TOKEN_USAGE_LOG.exists():
        return {"totals": totals, "by_subject": by_subject, "by_model": by_model}

    with open(TOKEN_USAGE_LOG, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue

            totals["input_tokens"] += r.get("input_tokens", 0)
            totals["output_tokens"] += r.get("output_tokens", 0)
            totals["cached_tokens"] += r.get("cached_tokens", 0)
            totals["total_tokens"] += r.get("total_tokens", 0)
            totals["requests"] += 1

            subject_label = r.get("subject_key") or f"{r.get('language_key','')}-{r.get('skill_key','')}"
            bucket = by_subject.setdefault(subject_label, {"input_tokens": 0, "output_tokens": 0, "cached_tokens": 0, "requests": 0})
            bucket["input_tokens"] += r.get("input_tokens", 0)
            bucket["output_tokens"] += r.get("output_tokens", 0)
            bucket["cached_tokens"] += r.get("cached_tokens", 0)
            bucket["requests"] += 1

            model_label = r.get("model_name") or "unknown"
            mbucket = by_model.setdefault(model_label, {"input_tokens": 0, "output_tokens": 0, "cached_tokens": 0, "requests": 0})
            mbucket["input_tokens"] += r.get("input_tokens", 0)
            mbucket["output_tokens"] += r.get("output_tokens", 0)
            mbucket["cached_tokens"] += r.get("cached_tokens", 0)
            mbucket["requests"] += 1

    return {"totals": totals, "by_subject": by_subject, "by_model": by_model}
