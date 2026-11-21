"""
batch_processor.py
Bulk homework grading via Gemini's Batch API: a flat 50% discount on
input+output tokens, in exchange for asynchronous turnaround (typically
minutes in practice, guaranteed within ~24h). This is the right tool for
"grade all of today's submissions at once" workflows -- e.g. a CRM that
exports a batch of homework overnight -- NOT for a teacher waiting on
one result in the Streamlit UI (use homework_checker.get_checker() +
the real-time job queue in jobs.py for that).

Combines automatically with explicit context caching (cache_registry.py):
each item in the batch reuses the shared cache for its subject/skill
when one exists, so caching's ~90%-off input savings and batch's 50%-off
both apply together on the same request.

Typical usage:

    items = [
        BatchSubmission(item_id="student_1", file_path="uploads/hw1.pdf", subject_key="math"),
        BatchSubmission(item_id="student_2", file_path="uploads/hw2.pdf", subject_key="math"),
        ...
    ]
    batch_name = submit_batch(items, api_key="...")
    # ... later (minutes to hours) ...
    status = poll_batch(batch_name, api_key="...")
    if status["state"] == "succeeded":
        for item_id, result in status["results"].items():
            ...  # same JSON schema as HomeworkChecker.analyze()
"""

import time
from pathlib import Path
from typing import Dict, List, Optional

import cache_registry
import token_usage_logger
from config import (
    BATCH_MAX_WAIT_SECONDS,
    BATCH_POLL_INTERVAL_SECONDS,
    CACHE_TTL_SECONDS,
    DEFAULT_MODEL,
    ENABLE_CONTEXT_CACHING,
    GEMINI_API_KEY,
    GENERATION_CONFIG,
)
from homework_checker import _safe_cache_display_name
from prompts import DEFAULT_SUBJECT_KEY, build_system_prompt, build_user_prompt
from utils import extract_text_from_docx, guess_mime_type, parse_and_validate_json


class BatchProcessorError(Exception):
    pass


class BatchSubmission:
    """One homework file to grade as part of a batch."""

    def __init__(
        self,
        item_id: str,
        file_path,
        subject_key: str = DEFAULT_SUBJECT_KEY,
        custom_subject_name: str = "",
        language_key: str = "",
        skill_key: str = "",
        extra_instructions: str = "",
        model_name: Optional[str] = None,
    ):
        self.item_id = item_id  # your own id (e.g. student name/uuid) to match results back
        self.file_path = Path(file_path)
        self.subject_key = subject_key
        self.custom_subject_name = custom_subject_name
        self.language_key = language_key
        self.skill_key = skill_key
        self.extra_instructions = extra_instructions
        self.model_name = model_name or DEFAULT_MODEL


def _resolve_cache_or_system_instruction(client, types, item: BatchSubmission, system_prompt: str) -> dict:
    """Same cache-or-fallback logic as HomeworkChecker, reused here so
    batch requests benefit from the same shared cache."""
    if not ENABLE_CONTEXT_CACHING:
        return {"system_instruction": system_prompt}

    cache_key = "|".join([
        item.model_name, item.subject_key, item.custom_subject_name.lower().strip(),
        item.language_key, item.skill_key,
    ])
    status, cache_name = cache_registry.lookup(cache_key)

    if status == "cached":
        return {"cached_content": cache_name}
    if status == "uncacheable":
        return {"system_instruction": system_prompt}

    display_name = _safe_cache_display_name(
        item.model_name, item.subject_key or "lang", item.language_key, item.skill_key
    )
    try:
        cache = client.caches.create(
            model=item.model_name,
            config=types.CreateCachedContentConfig(
                system_instruction=system_prompt,
                ttl=f"{CACHE_TTL_SECONDS}s",
                display_name=display_name,
            ),
        )
        cache_registry.set_cache(cache_key, cache.name, CACHE_TTL_SECONDS)
        return {"cached_content": cache.name}
    except Exception:
        cache_registry.mark_uncacheable(cache_key)
        return {"system_instruction": system_prompt}


def _build_inlined_request(client, types, item: BatchSubmission):
    system_prompt = build_system_prompt(
        subject_key=item.subject_key,
        custom_subject_name=item.custom_subject_name,
        language_key=item.language_key,
        skill_key=item.skill_key,
    )
    user_prompt = build_user_prompt(item.extra_instructions)
    ext = item.file_path.suffix.lower()

    if ext == ".docx":
        contents = [
            types.Part.from_text(text=user_prompt),
            types.Part.from_text(
                text="The homework was submitted as a Word document. "
                "Here is the extracted text:\n\n" + extract_text_from_docx(item.file_path)
            ),
        ]
    else:
        mime_type = guess_mime_type(item.file_path)
        file_bytes = item.file_path.read_bytes()
        contents = [
            types.Part.from_text(text=user_prompt),
            types.Part.from_bytes(data=file_bytes, mime_type=mime_type),
        ]

    cache_kwargs = _resolve_cache_or_system_instruction(client, types, item, system_prompt)
    config = types.GenerateContentConfig(
        temperature=GENERATION_CONFIG.get("temperature", 0.1),
        top_p=GENERATION_CONFIG.get("top_p", 0.9),
        max_output_tokens=GENERATION_CONFIG.get("max_output_tokens", 8192),
        response_mime_type=GENERATION_CONFIG.get("response_mime_type", "application/json"),
        **cache_kwargs,
    )

    return types.InlinedRequest(
        model=item.model_name,
        contents=contents,
        config=config,
        metadata={
            "item_id": item.item_id,
            "subject_key": item.subject_key,
            "custom_subject_name": item.custom_subject_name,
            "language_key": item.language_key,
            "skill_key": item.skill_key,
            "model_name": item.model_name,
        },
    )


def submit_batch(items: List[BatchSubmission], api_key: Optional[str] = None) -> str:
    """
    Submits all items as ONE Gemini Batch job and returns immediately
    with the batch job's name (id) -- does not wait for grading.
    All items must use the same model_name (Gemini's batch API groups
    a batch under one model).
    """
    if not items:
        raise BatchProcessorError("No items to submit.")

    model_names = {item.model_name for item in items}
    if len(model_names) > 1:
        raise BatchProcessorError(
            f"All items in one batch must share the same model_name; got {model_names}. "
            "Submit separate batches per model instead."
        )

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key or GEMINI_API_KEY)

    inlined_requests = [_build_inlined_request(client, types, item) for item in items]

    batch_job = client.batches.create(
        model=items[0].model_name,
        src=inlined_requests,
        config=types.CreateBatchJobConfig(display_name=f"hwcheck-batch-{int(time.time())}"),
    )
    return batch_job.name


def poll_batch(batch_name: str, api_key: Optional[str] = None) -> dict:
    """
    Checks a batch job's status. Returns:
      {"state": "pending" | "running" | "succeeded" | "failed" | "cancelled"}
    plus, when state == "succeeded":
      {"results": {item_id: {...grading JSON...} or {"error": "..."} }}
    """
    from google import genai

    client = genai.Client(api_key=api_key or GEMINI_API_KEY)
    job = client.batches.get(name=batch_name)

    state_name = str(job.state).replace("JobState.JOB_STATE_", "").lower()
    response = {"state": state_name, "batch_name": batch_name}

    if state_name != "succeeded":
        if job.error:
            response["error"] = str(job.error)
        return response

    results = {}
    inlined = getattr(job.dest, "inlined_responses", None) or []
    for item in inlined:
        meta = item.metadata or {}
        item_id = meta.get("item_id", "unknown")
        if item.error:
            results[item_id] = {"error": str(item.error)}
            continue
        try:
            raw_text = item.response.text
            results[item_id] = parse_and_validate_json(raw_text)

            usage = getattr(item.response, "usage_metadata", None)
            if usage is not None:
                token_usage_logger.log_usage(
                    input_tokens=getattr(usage, "prompt_token_count", 0) or 0,
                    output_tokens=getattr(usage, "candidates_token_count", 0) or 0,
                    cached_tokens=getattr(usage, "cached_content_token_count", 0) or 0,
                    total_tokens=getattr(usage, "total_token_count", 0) or 0,
                    backend="gemini-batch",  # tagged separately since batch is billed at 50% off
                    model_name=meta.get("model_name", ""),
                    subject_key=meta.get("subject_key", ""),
                    custom_subject_name=meta.get("custom_subject_name", ""),
                    language_key=meta.get("language_key", ""),
                    skill_key=meta.get("skill_key", ""),
                )
        except Exception as exc:
            results[item_id] = {"error": f"Failed to parse response: {exc}"}

    response["results"] = results
    return response


def wait_for_batch(
    batch_name: str,
    api_key: Optional[str] = None,
    poll_interval: int = BATCH_POLL_INTERVAL_SECONDS,
    max_wait: int = BATCH_MAX_WAIT_SECONDS,
) -> dict:
    """
    Blocks, polling periodically, until the batch finishes or max_wait is
    hit. Useful for a script/cron-style workflow; for a web server, prefer
    submit_batch() + your own periodic poll_batch() calls (e.g. a
    scheduled task) so you don't hold an HTTP request open for a long time.
    """
    start = time.monotonic()
    while True:
        status = poll_batch(batch_name, api_key=api_key)
        if status["state"] in ("succeeded", "failed", "cancelled"):
            return status
        if time.monotonic() - start > max_wait:
            raise BatchProcessorError(
                f"Batch {batch_name} did not finish within {max_wait}s (last state: {status['state']})"
            )
        time.sleep(poll_interval)
