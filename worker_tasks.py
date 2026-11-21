"""
worker_tasks.py
The actual grading work, executed by RQ worker processes -- NOT by the
API process itself. This is what lets api.py's POST /jobs return
instantly: it just hands this function's name + arguments to Redis, and
a separate `rq worker` process (started with `rq worker homework_grading`)
picks it up and runs it whenever it's free.

This must be a plain, importable module-level function (not a closure or
a method) because RQ pickles a reference to it (module path + function
name) to store in Redis -- the worker process re-imports it to run the job.
"""

import json
from pathlib import Path

import db
from config import COLLECT_TRAINING_DATA, MODEL_BACKEND, WEBHOOK_TIMEOUT_SECONDS
from homework_checker import get_checker
import training_logger
import webhook_security

# Subjects/languages the new hwchecker/ pipeline (providers/ + subjects/
# + checkers/ + router/, per the multi-model-grading-architecture spec)
# actually covers. Everything else (essay, computer_science, general,
# french, german, Speaking) still goes through the original single-call
# get_checker() system below -- nothing that already worked stops working,
# the new tiered/verified/deterministic-checked pipeline is used wherever
# it applies, and the old one quietly covers the rest.
_HWCHECKER_SUBJECTS = {"math", "physics", "chemistry", "biology", "history"}
_HWCHECKER_LANGUAGES = {"english", "russian", "uzbek", "turkish"}


def _resolve_hwchecker_subject(submission: dict):
    """Returns (subject_id, skill) if this submission should use the new
    GradingRouter pipeline, or (None, None) if it should use the
    original get_checker() system instead."""
    language_key = submission.get("language_key") or ""
    skill_key = submission.get("skill_key") or ""
    subject_key = submission.get("subject_key") or ""

    if language_key:
        if language_key in _HWCHECKER_LANGUAGES and skill_key != "speaking":
            return language_key, skill_key
        return None, None  # french/german/Speaking -- not in the new registry yet

    if subject_key in _HWCHECKER_SUBJECTS:
        return subject_key, None

    return None, None  # essay/computer_science/general -- use the original system


def grade_homework_task(
    saved_path_str: str,
    student_name: str = "Student",
    api_key: str = "",
    model_name: str = "",
    subject_key: str = "math",
    custom_subject_name: str = "",
    language_key: str = "",
    skill_key: str = "",
    extra_instructions: str = "",
) -> dict:
    """
    Grades one homework submission. Runs inside a worker process, so any
    exception raised here is captured by RQ as the job's failure info
    (visible via GET /jobs/{id}) rather than crashing the API.

    Returns the same JSON schema POST /analyze always returned -- RQ
    stores this dict as the job's `.result`.
    """
    saved_path = Path(saved_path_str)

    checker = get_checker(
        api_key=api_key or None,
        model_name=model_name or None,
        subject_key=subject_key,
        custom_subject_name=custom_subject_name,
        language_key=language_key,
        skill_key=skill_key,
    )
    result = checker.analyze(saved_path, extra_instructions=extra_instructions)

    if COLLECT_TRAINING_DATA:
        training_logger.log_raw_example(
            input_file_path=saved_path,
            system_prompt=getattr(checker, "system_prompt", ""),
            user_prompt=getattr(checker, "last_user_prompt", ""),
            model_output=result,
            subject_key=subject_key,
            custom_subject_name=custom_subject_name,
            language_key=language_key,
            skill_key=skill_key,
            model_name=getattr(checker, "model_name", model_name or ""),
            backend=MODEL_BACKEND,
        )

    return {"student_name": student_name, **result}


def grade_submission_task(submission_id: str) -> None:
    """
    Grades a CRM-review-workflow Submission row (db.py): reads its
    params from Postgres, grades it, and writes the AI result back --
    moving status from "pending_ai" to "pending_review" so it shows up
    in a teacher's review queue.

    Dispatches to the new tiered hwchecker/ GradingRouter (difficulty-
    tier provider routing, deterministic $0 checkers, secondary
    verification, cross-provider fallback) for the subjects/languages it
    covers, and to the original single-call get_checker() system for
    everything else -- see _resolve_hwchecker_subject() above. Either
    way, training_logger gets the same kind of entry for future
    fine-tuning.
    """
    submission = db.get_submission(submission_id, include_internal=True)
    if submission is None:
        return
    if submission["status"] != "pending_ai":
        # A duplicate or late job (client retry, reconciler racing a slow worker, a regrade that
        # already finished): the submission was graded, reviewed or approved meanwhile.
        # Do not pay for another grading and never touch its stored result.
        return

    hwchecker_subject, hwchecker_skill = _resolve_hwchecker_subject(submission)
    bypass_cache = _consume_regrade_flag(submission_id)

    try:
        if hwchecker_subject:
            result, usage, system_prompt, user_prompt, model_name = _grade_via_hwchecker_router(
                submission, hwchecker_subject, hwchecker_skill, bypass_cache=bypass_cache
            )
            backend_label = f"hwchecker:{result.get('provider', 'unknown')}"
        else:
            checker = get_checker(
                subject_key=submission["subject_key"],
                custom_subject_name=submission.get("custom_subject_name") or "",
                language_key=submission.get("language_key") or "",
                skill_key=submission.get("skill_key") or "",
                model_name=submission.get("model_name") or None,
                backend=submission.get("backend") or None,
            )
            result = checker.analyze(
                Path(submission["input_file_path"]),
                extra_instructions=submission.get("extra_instructions") or "",
            )
            usage = getattr(checker, "last_usage", {}) or {}
            system_prompt = getattr(checker, "system_prompt", "")
            user_prompt = getattr(checker, "last_user_prompt", "")
            model_name = getattr(checker, "model_name", submission.get("model_name") or "")
            backend_label = submission.get("backend") or MODEL_BACKEND

        db.mark_submission_graded(
            submission_id,
            ai_result=result,
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            cached_tokens=usage.get("cached_tokens", 0),
        )

        if COLLECT_TRAINING_DATA:
            training_logger.log_raw_example(
                input_file_path=Path(submission["input_file_path"]),
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model_output=result,
                subject_key=submission["subject_key"],
                custom_subject_name=submission.get("custom_subject_name") or "",
                language_key=submission.get("language_key") or "",
                skill_key=submission.get("skill_key") or "",
                model_name=model_name,
                backend=backend_label,
            )
    except Exception as exc:
        db.mark_submission_failed(submission_id, str(exc))
        raise  # still let RQ record this as a failed job for visibility


def regrade_flag_key(submission_id: str) -> str:
    return f"hw:regrade:{submission_id}"


def _consume_regrade_flag(submission_id: str) -> bool:
    """True (once) if a teacher asked for a regrade: the result cache must be bypassed so
    the file is really graded again instead of the same cached verdict coming back."""
    try:
        import jobs
        return bool(jobs.get_redis_connection().delete(regrade_flag_key(submission_id)))
    except Exception:
        return False


def _grade_via_hwchecker_router(submission: dict, subject_id: str, skill: str, bypass_cache: bool = False):
    """
    Runs one submission through the new hwchecker/ GradingRouter and
    adapts its output into the shape db.py stores (see
    hwchecker/pipeline/crm_adapter.py).
    Returns (adapted_result, usage, system_prompt, user_prompt, provider_label); the system
    prompt is the exact one the router sent to the model, so the training log records it.
    """
    from prompts import build_user_prompt
    from hwchecker.pipeline.crm_adapter import adapt_router_result
    from hwchecker.router.grading_router import GradingRouter
    from hwchecker.subjects.registry import get_subject

    router = GradingRouter()
    assignment = submission.get("assignment") or {}
    user_prompt = build_user_prompt(
        submission.get("extra_instructions") or "",
        task_text=assignment.get("task_text") or "",
        answer_key=assignment.get("answer_key") or "",
    )
    homework = {
        "subject_id": subject_id,
        "skill": skill,
        "content": {
            "file_path": Path(submission["input_file_path"]),
            "user_prompt": user_prompt,
        },
    }
    if bypass_cache:
        homework["bypass_cache"] = True
    schema_result = router.grade(homework)

    adapted = adapt_router_result(schema_result)
    usage = {
        "input_tokens": schema_result["tokens"]["input"],
        "output_tokens": schema_result["tokens"]["output"],
        "cached_tokens": schema_result["tokens"].get("cached", 0),
    }
    system_prompt = get_subject(subject_id).build_system_prompt(skill)
    return adapted, usage, system_prompt, user_prompt, schema_result["provider"]


def _score_of(result) -> "float | int | None":
    """overall_score of a result dict, or None when absent/not a number (never raises)."""
    value = result.get("overall_score") if isinstance(result, dict) else None
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def build_webhook_payload(submission: dict) -> dict:
    """
    The body POSTed to the CRM for an approved submission. The original fields (submission_id,
    external_submission_id, status, result, approved_at) are unchanged; the extra top-level
    fields are what a gradebook needs without digging into `result`:

      final_score        the score the teacher approved (the edited one when they changed it)
      ai_score           what the AI originally scored (kept for audit, even after an edit)
      edited_by_teacher  True when the teacher changed the AI's result before approving
      reviewer_note      the teacher's note on an edit (may be null)
      reviewed_at        when the teacher reviewed it
    """
    final_result = submission.get("final_result")
    return {
        "submission_id": submission["id"],
        "external_submission_id": submission.get("external_submission_id"),
        "status": submission["status"],
        "result": final_result,
        "final_score": _score_of(final_result),
        "ai_score": _score_of(submission.get("ai_result")),
        "edited_by_teacher": bool(submission.get("reviewed_result")),
        "reviewer_note": submission.get("reviewer_note"),
        "reviewed_at": submission.get("reviewed_at"),
        "approved_at": submission.get("approved_at"),
    }


def deliver_webhook_task(submission_id: str) -> None:
    """
    POSTs an approved submission's final result to its organization's
    registered webhook_url. Scheduled with RQ's Retry (see jobs.py) so
    transient failures (CRM endpoint briefly down, network blip) retry
    with backoff automatically without blocking a worker on a sleep.

    If there's no webhook_url configured, or delivery ultimately fails
    after all retries, GET /submissions?status=approved&since=... remains
    available as a polling fallback -- nothing is lost, just not pushed.
    """
    import requests

    submission = db.get_submission(submission_id)
    if submission is None:
        return
    if submission["status"] != "approved":
        return  # never push a result that is not final

    org = db.get_organization(submission["organization_id"])
    if not org or not org.get("webhook_url"):
        return  # no webhook configured for this org -- polling-only, that's fine

    payload = build_webhook_payload(submission)
    body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json", **webhook_security.signature_headers(org["id"], body)}

    try:
        # Re-validated at every delivery: the stored URL may predate the rules, or its DNS may
        # have changed since registration. Redirects are never followed (a 3xx could bounce the
        # signed request to an internal address) and count as a failed delivery.
        url = webhook_security.validate_webhook_url(org["webhook_url"])
        response = requests.post(
            url, data=body, headers=headers, timeout=WEBHOOK_TIMEOUT_SECONDS, allow_redirects=False
        )
    except Exception as exc:
        db.record_webhook_attempt(submission_id, success=False, error=str(exc))
        raise  # re-raise so RQ's Retry schedules another attempt

    success = 200 <= response.status_code < 300
    db.record_webhook_attempt(submission_id, success=success, status_code=response.status_code)  # exactly one record per attempt
    if not success:
        raise RuntimeError(f"Webhook endpoint returned HTTP {response.status_code}")  # RQ's Retry schedules another attempt
