"""
api.py
REST API for Homework Checker AI, built with FastAPI.

This runs ALONGSIDE the Streamlit app (app.py) -- it does not replace it.
Use this when another website, mobile app, or backend service needs to
submit a homework file (or a Speaking audio recording) programmatically
and get back the same JSON grading result the Streamlit UI shows, plus
endpoints to turn that result into HTML/PDF/Excel reports.

Run with:
    uvicorn api:app --reload --port 8000

Interactive API docs (Swagger UI) are then available at:
    http://localhost:8000/docs

Example (curl):
    curl -X POST http://localhost:8000/analyze \\
      -F "file=@homework.pdf" \\
      -F "subject_key=math" \\
      -F "api_key=YOUR_GEMINI_API_KEY"

Example (language skill, Speaking):
    curl -X POST http://localhost:8000/analyze \\
      -F "file=@recording.mp3" \\
      -F "language_key=english" \\
      -F "skill_key=speaking" \\
      -F "api_key=YOUR_GEMINI_API_KEY"
"""

from datetime import datetime
import hmac
import logging
from pathlib import Path
from typing import List, Optional
import uuid

from fastapi import Body, FastAPI, File, Form, Header, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response

from batch_processor import BatchProcessorError, BatchSubmission, poll_batch, submit_batch
import config
import hardening
from config import DEFAULT_MODEL, INTERNAL_API_KEY, MODEL_BACKEND, COLLECT_TRAINING_DATA
from homework_checker import get_checker, HomeworkCheckerError
from prompts import (
    DEFAULT_LANGUAGE_KEY,
    DEFAULT_SKILL_KEY,
    DEFAULT_SUBJECT_KEY,
    LANGUAGES,
    SKILL_TEMPLATES,
    SUBJECTS,
    build_system_prompt,
    build_user_prompt,
)
from report_generator import (
    generate_excel_gradebook,
    generate_html_report,
    generate_pdf_report,
)
from utils import (
    FileTooLargeError,
    UnsupportedFileError,
    save_uploaded_file,
    validate_upload,
)
import training_logger
import token_usage_logger
import jobs
import db
import reconcile
import retention
import utils
import webhook_security
from review_validation import ReviewResultError, validate_review_result

logger = logging.getLogger("hwchecker.api")

app = FastAPI(
    title="Homework Checker AI API",
    description=(
        "Upload a student's homework (PDF, image, Word doc, or an audio "
        "recording for Speaking tasks) and receive AI-graded, "
        "question-by-question feedback as JSON, plus optional HTML/PDF/"
        "Excel reports generated from that result."
    ),
    version="1.0.0",
)

if config.IS_PRODUCTION:
    _problems = config.production_config_problems()
    if _problems:
        raise RuntimeError(
            "Refusing to start with APP_ENV=production and an unsafe configuration:\n  - " + "\n  - ".join(_problems)
        )

# CORS is OFF unless CORS_ALLOW_ORIGINS lists the browser origins that may call this API
# (server-to-server callers -- a CRM backend, the Streamlit app -- do not need it).
if config.CORS_ALLOW_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.CORS_ALLOW_ORIGINS,
        allow_methods=["*"],
        allow_headers=["*"],
    )

if not INTERNAL_API_KEY:
    logger.warning(
        "INTERNAL_API_KEY is not set: protected endpoints answer 503 %s",
        "(ALLOW_UNAUTHENTICATED_API=true is set: they are OPEN to anyone who can reach this port)"
        if config.ALLOW_UNAUTHENTICATED_API else "until it is configured.",
    )


def _save_upload_or_400(file_bytes: bytes, filename: str):
    """save_uploaded_file(), but an unreadable photo / missing HEIC support answers HTTP 400
    with the reason instead of crashing the request with a 500."""
    try:
        return save_uploaded_file(file_bytes, filename)
    except UnsupportedFileError as exc:
        raise HTTPException(status_code=400, detail=f"{filename}: {exc}")


def _read_upload_or_413(upload) -> bytes:
    """The upload's bytes, read with a hard size cap (HTTP 413 as soon as it is exceeded)."""
    try:
        return hardening.read_upload_capped(upload)
    except hardening.RequestTooLarge as exc:
        raise HTTPException(status_code=413, detail=str(exc))


def _check_fields_or_422(**fields) -> None:
    try:
        hardening.check_field_lengths(**fields)
    except hardening.FieldTooLong as exc:
        raise HTTPException(status_code=422, detail=str(exc))


def _check_org_rate_or_429(organization_id: str) -> None:
    try:
        hardening.check_org_rate(organization_id)
    except hardening.RateLimitExceeded as exc:
        raise HTTPException(
            status_code=429,
            detail=f"Too many submissions ({exc.scope} limit of {exc.limit}). Retry after {exc.retry_after} seconds.",
            headers={"Retry-After": str(exc.retry_after)},
        )


def _check_internal_key(x_internal_api_key: Optional[str]) -> None:
    """
    Gate for the admin / paid endpoints. FAILS CLOSED:
      * INTERNAL_API_KEY configured -> the request must send a matching X-Internal-Api-Key header
        (compared in constant time);
      * INTERNAL_API_KEY empty      -> 503, unless ALLOW_UNAUTHENTICATED_API=true was set on purpose
        (local development), in which case the API is open.
    Otherwise anyone who could reach the port could create tenants and spend your LLM quota.
    """
    if not INTERNAL_API_KEY:
        if config.ALLOW_UNAUTHENTICATED_API:
            return
        raise HTTPException(
            status_code=503,
            detail="Server is not configured: set INTERNAL_API_KEY (or ALLOW_UNAUTHENTICATED_API=true for local development only).",
        )
    supplied = (x_internal_api_key or "").encode("utf-8")
    if not hmac.compare_digest(supplied, INTERNAL_API_KEY.encode("utf-8")):
        raise HTTPException(
            status_code=401,
            detail="Missing or invalid X-Internal-Api-Key header.",
        )


def _authenticate_organization(x_organization_api_key: Optional[str]) -> dict:
    """
    Resolves the calling CRM's organization (tenant) from its
    X-Organization-Api-Key header. Every submissions/* endpoint requires
    this -- it's what keeps one school's homework data invisible to
    another's, since every query in db.py is scoped to organization_id.
    This is separate from X-Internal-Api-Key: that one gates the whole
    API from anonymous traffic; this one identifies WHICH tenant is
    calling and enforces data isolation between tenants.
    """
    if not x_organization_api_key:
        raise HTTPException(status_code=401, detail="Missing X-Organization-Api-Key header.")
    org = db.get_organization_by_api_key(x_organization_api_key)
    if not org:
        raise HTTPException(status_code=401, detail="Invalid X-Organization-Api-Key.")
    return org


def _strip_student_name(payload: dict) -> dict:
    """/analyze responses embed student_name alongside the grading JSON;
    strip it back out before feeding the dict to the report generators,
    which expect only the grading schema."""
    return {k: v for k, v in payload.items() if k != "student_name"}


# ---------------------------------------------------------------------------
# Discovery endpoints -- so a calling app can build its own subject/
# language/skill picker without hard-coding the list on its own side.
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    """Liveness only: the process is up. Use /readyz to know whether it can actually do work."""
    return {"status": "ok"}


@app.get("/readyz")
def readiness():
    """Readiness: can this instance reach PostgreSQL and Redis? 503 if either is down."""
    problems = {}
    try:
        from sqlalchemy import text
        with db.session_scope() as session:
            session.execute(text("SELECT 1"))
    except Exception as exc:
        problems["postgres"] = type(exc).__name__
    try:
        jobs.get_redis_connection().ping()
    except Exception as exc:
        problems["redis"] = type(exc).__name__
    if problems:
        raise HTTPException(status_code=503, detail={"status": "not_ready", "problems": problems})
    return {"status": "ready"}


@app.get("/backend")
def backend_info(x_internal_api_key: Optional[str] = Header(None)):
    """Reports which model backend this server is currently configured to use."""
    _check_internal_key(x_internal_api_key)
    stats = training_logger.count_examples()
    return {
        "model_backend": MODEL_BACKEND,
        "training_data_collection_enabled": config.COLLECT_TRAINING_DATA,
        "training_examples_logged": stats,
    }


@app.get("/usage")
def usage_stats(x_internal_api_key: Optional[str] = Header(None)):
    """
    Token usage totals and breakdowns (by subject/skill, by model) across
    every graded submission so far -- input/output/cached token counts,
    not a dollar estimate (pricing varies by model and changes over time;
    see your Gemini billing dashboard for actual cost).
    """
    _check_internal_key(x_internal_api_key)
    return token_usage_logger.get_usage_summary()


@app.get("/subjects")
def list_subjects():
    """Standard subjects usable as `subject_key` in POST /analyze."""
    return {key: profile.label for key, profile in SUBJECTS.items()}


@app.get("/languages")
def list_languages():
    """Languages usable as `language_key` in POST /analyze."""
    return LANGUAGES


@app.get("/skills")
def list_skills():
    """Language skills usable as `skill_key` in POST /analyze."""
    return {key: tmpl["label"] for key, tmpl in SKILL_TEMPLATES.items()}


# ---------------------------------------------------------------------------
# Core grading endpoint
# ---------------------------------------------------------------------------
@app.post("/analyze")
def analyze_homework(
    file: UploadFile = File(..., description="Homework file: PDF, PNG/JPG/WEBP, DOCX, or audio (mp3/wav/m4a/ogg) for Speaking."),
    student_name: str = Form("Student"),
    model_name: Optional[str] = Form(None, description="Model name override. Defaults to config.DEFAULT_MODEL (Gemini) or the configured local model, depending on MODEL_BACKEND."),
    subject_key: str = Form(DEFAULT_SUBJECT_KEY, description="Standard subject key, e.g. 'math', 'physics', 'general'. Ignored if language_key+skill_key are set."),
    custom_subject_name: str = Form("", description="Required only when subject_key='general'."),
    language_key: str = Form("", description="e.g. 'english', 'russian', 'turkish'. Set together with skill_key to grade a language skill instead of a standard subject."),
    skill_key: str = Form("", description="e.g. 'writing', 'reading', 'listening', 'speaking'."),
    extra_instructions: str = Form("", description="Optional extra context, e.g. 'CEFR level: B1' or 'Be strict about rigor.'"),
    api_key: Optional[str] = Form(None, description="Gemini API key. Ignored entirely when MODEL_BACKEND=local. Optional if GEMINI_API_KEY is set as a server env var."),
    x_internal_api_key: Optional[str] = Header(None),
):
    """
    Grades one homework submission and returns the same JSON schema the
    Streamlit app displays: overall_score, grade, questions[], summary.
    Backend (Gemini vs your own server) is controlled by MODEL_BACKEND.

    This call is SYNCHRONOUS -- it blocks until grading finishes (often
    10-30+ seconds). Fine for testing or low-volume use. If many
    submissions can arrive at once (e.g. from a CRM), use POST /jobs
    instead: it returns instantly and you poll GET /jobs/{job_id} for
    the result, so nothing times out and nothing blocks under load.
    """
    _check_internal_key(x_internal_api_key)

    file_bytes = _read_upload_or_413(file)
    try:
        validate_upload(file.filename, len(file_bytes))
    except (UnsupportedFileError, FileTooLargeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    if subject_key == "general" and not language_key and not custom_subject_name.strip():
        raise HTTPException(
            status_code=400,
            detail="custom_subject_name is required when subject_key='general'.",
        )

    saved_path = _save_upload_or_400(file_bytes, file.filename)

    try:
        checker = get_checker(
            api_key=api_key or None,
            model_name=model_name,
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
    except HomeworkCheckerError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {exc}")

    return {"student_name": student_name, **result}


# ---------------------------------------------------------------------------
# Async job queue -- recommended for CRM/production use. POST /jobs returns
# instantly (milliseconds); grading happens in a separate `rq worker`
# process. Poll GET /jobs/{job_id} until status is "finished" or "failed".
# Requires Redis running and at least one worker: see README.md.
# ---------------------------------------------------------------------------
@app.post("/jobs")
def submit_job(
    file: UploadFile = File(..., description="Homework file: PDF, PNG/JPG/WEBP, DOCX, or audio (mp3/wav/m4a/ogg) for Speaking."),
    student_name: str = Form("Student"),
    model_name: Optional[str] = Form(None),
    subject_key: str = Form(DEFAULT_SUBJECT_KEY),
    custom_subject_name: str = Form(""),
    language_key: str = Form(""),
    skill_key: str = Form(""),
    extra_instructions: str = Form(""),
    api_key: Optional[str] = Form(None),
    x_internal_api_key: Optional[str] = Header(None),
):
    """
    Enqueues one homework submission for grading and returns a job_id
    immediately -- it does NOT wait for grading to finish. Poll
    GET /jobs/{job_id} to retrieve the result once status is "finished".
    """
    _check_internal_key(x_internal_api_key)

    if api_key:
        # A job's arguments are stored in Redis (readable by anything with Redis access) until the
        # job expires, so a caller-supplied provider key must not travel through the queue.
        raise HTTPException(
            status_code=400,
            detail="api_key is not accepted on /jobs: configure GEMINI_API_KEY (and the other provider keys) on the server. "
                   "Use /analyze if you need to send your own key per request.",
        )

    file_bytes = _read_upload_or_413(file)
    try:
        validate_upload(file.filename, len(file_bytes))
    except (UnsupportedFileError, FileTooLargeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    if subject_key == "general" and not language_key and not custom_subject_name.strip():
        raise HTTPException(
            status_code=400,
            detail="custom_subject_name is required when subject_key='general'.",
        )

    saved_path = _save_upload_or_400(file_bytes, file.filename)

    try:
        job_id = jobs.enqueue_grading_job(
            saved_path_str=str(saved_path),
            student_name=student_name,
            api_key="",
            model_name=model_name or "",
            subject_key=subject_key,
            custom_subject_name=custom_subject_name,
            language_key=language_key,
            skill_key=skill_key,
            extra_instructions=extra_instructions,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Could not reach the job queue (is Redis running?): {exc}",
        )

    return {"job_id": job_id, "status": "queued"}


@app.get("/jobs/{job_id}")
def check_job(job_id: str, x_internal_api_key: Optional[str] = Header(None)):
    """
    Returns the current status of a job submitted via POST /jobs:
    "queued" | "started" | "finished" | "failed" | "not_found".
    When "finished", the response includes "result" (the grading JSON,
    same shape as /analyze). When "failed", it includes a short "error".
    """
    _check_internal_key(x_internal_api_key)
    return jobs.get_job_status(job_id)


# ---------------------------------------------------------------------------
# Report generation endpoints -- feed back a previous /analyze response
# (as-is, including "student_name") to get a downloadable file.
# ---------------------------------------------------------------------------
_REPORT_RESPONSE_HEADERS = {
    # The report is a static document: no script, no network, no framing -- even if some markup
    # slipped past sanitising, the browser will not run it.
    "Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'; img-src data:; base-uri 'none'; form-action 'none'; frame-ancestors 'none'",
    "X-Content-Type-Options": "nosniff",
}
_MAX_REPORT_QUESTIONS = 500


def _report_student_name(payload: dict) -> str:
    name = payload.get("student_name", "Student")
    if not isinstance(name, str) or len(name) > 200:
        raise HTTPException(status_code=422, detail="student_name must be a string of at most 200 characters.")
    return name or "Student"


def _validated_report_result(payload) -> dict:
    """Checks the shape the report generators rely on (so bad input is a 422, not a 500) and
    returns the grading dict without the embedded student_name."""
    if not isinstance(payload, dict):
        raise HTTPException(status_code=422, detail="Body must be the JSON object returned by POST /analyze.")
    result = _strip_student_name(payload)
    overall = result.get("overall_score")
    if not isinstance(overall, (int, float)) or isinstance(overall, bool) or not 0 <= overall <= 100:
        raise HTTPException(status_code=422, detail="overall_score must be a number between 0 and 100.")
    questions = result.get("questions", [])
    if not isinstance(questions, list) or len(questions) > _MAX_REPORT_QUESTIONS or not all(isinstance(q, dict) for q in questions):
        raise HTTPException(status_code=422, detail="questions must be a list of objects.")
    normalised = []
    for q in questions:
        q = dict(q)
        score = q.get("score", overall)
        if not isinstance(score, (int, float)) or isinstance(score, bool) or not 0 <= score <= 100:
            raise HTTPException(status_code=422, detail="every question score must be a number between 0 and 100.")
        q["score"] = score
        normalised.append(q)
    result["questions"] = normalised
    return result


@app.post("/reports/html")
def report_html(
    payload: dict = Body(..., description="The full JSON body returned by POST /analyze."),
    x_internal_api_key: Optional[str] = Header(None),
):
    _check_internal_key(x_internal_api_key)
    student_name = _report_student_name(payload)
    result = _validated_report_result(payload)
    path = generate_html_report(result, student_name=student_name)
    return FileResponse(path, filename=path.name, media_type="text/html", headers=_REPORT_RESPONSE_HEADERS)


@app.post("/reports/pdf")
def report_pdf(
    payload: dict = Body(..., description="The full JSON body returned by POST /analyze."),
    x_internal_api_key: Optional[str] = Header(None),
):
    _check_internal_key(x_internal_api_key)
    student_name = _report_student_name(payload)
    result = _validated_report_result(payload)
    path = generate_pdf_report(result, student_name=student_name)
    return FileResponse(path, filename=path.name, media_type="application/pdf", headers=_REPORT_RESPONSE_HEADERS)


@app.post("/reports/gradebook")
def report_gradebook(
    records: list = Body(..., description="A list of /analyze response bodies (each including student_name) to combine into one gradebook."),
    x_internal_api_key: Optional[str] = Header(None),
):
    _check_internal_key(x_internal_api_key)
    if not records:
        raise HTTPException(status_code=400, detail="records must contain at least one item.")
    tuples = [(_report_student_name(r) if isinstance(r, dict) else "Student", _validated_report_result(r)) for r in records]
    path = generate_excel_gradebook(tuples)
    return FileResponse(
        path,
        filename=path.name,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


# ---------------------------------------------------------------------------
# Batch grading -- 50% cheaper than /analyze or /jobs, in exchange for
# asynchronous turnaround (typically minutes, guaranteed within ~24h).
# Use this for bulk/overnight grading runs (e.g. a CRM exporting a whole
# day's submissions at once), not for a teacher waiting on one result.
# All files in one batch must use the same subject/language+skill/model,
# since Gemini's Batch API groups a batch under one model; submit
# separate batches if you need to mix subjects in one run.
# ---------------------------------------------------------------------------
@app.post("/jobs/batch")
def submit_batch_job(
    files: List[UploadFile] = File(..., description="Multiple homework files to grade together as one batch."),
    student_names: str = Form("", description="Optional comma-separated student names, same order as files. Defaults to each file's filename."),
    model_name: str = Form(DEFAULT_MODEL),
    subject_key: str = Form(DEFAULT_SUBJECT_KEY),
    custom_subject_name: str = Form(""),
    language_key: str = Form(""),
    skill_key: str = Form(""),
    extra_instructions: str = Form(""),
    api_key: Optional[str] = Form(None),
    x_internal_api_key: Optional[str] = Header(None),
):
    """
    Submits multiple homework files as ONE Gemini Batch job and returns
    immediately with a batch_name to poll. Roughly half the cost of the
    same files graded individually via /analyze or /jobs, but results
    are not ready right away -- poll GET /jobs/batch/{batch_name}.
    """
    _check_internal_key(x_internal_api_key)

    if not files:
        raise HTTPException(status_code=400, detail="At least one file is required.")

    if subject_key == "general" and not language_key and not custom_subject_name.strip():
        raise HTTPException(
            status_code=400,
            detail="custom_subject_name is required when subject_key='general'.",
        )

    name_list = [n.strip() for n in student_names.split(",")] if student_names.strip() else []

    items = []
    id_to_label = {}
    for i, f in enumerate(files):
        file_bytes = _read_upload_or_413(f)
        try:
            validate_upload(f.filename, len(file_bytes))
        except (UnsupportedFileError, FileTooLargeError) as exc:
            raise HTTPException(status_code=400, detail=f"{f.filename}: {exc}")

        saved_path = _save_upload_or_400(file_bytes, f.filename)
        item_id = uuid.uuid4().hex
        label = name_list[i] if i < len(name_list) and name_list[i] else f.filename
        id_to_label[item_id] = label

        items.append(BatchSubmission(
            item_id=item_id,
            file_path=saved_path,
            subject_key=subject_key,
            custom_subject_name=custom_subject_name,
            language_key=language_key,
            skill_key=skill_key,
            extra_instructions=extra_instructions,
            model_name=model_name,
        ))

    try:
        batch_name = submit_batch(items, api_key=api_key or None)
    except BatchProcessorError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Failed to submit batch: {exc}")

    return {
        "batch_name": batch_name,
        "item_count": len(items),
        "items": id_to_label,  # item_id -> student name/filename, so you can map results back
    }


@app.get("/jobs/batch/{batch_name:path}")
def check_batch_job(batch_name: str, x_internal_api_key: Optional[str] = Header(None)):
    """
    Returns the current status of a batch submitted via POST /jobs/batch:
    "pending" | "running" | "succeeded" | "failed" | "cancelled".
    When "succeeded", includes "results": {item_id: {...grading JSON...}}
    -- match item_id back to your students using the "items" mapping
    POST /jobs/batch returned at submission time.
    """
    _check_internal_key(x_internal_api_key)
    try:
        return poll_batch(batch_name)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Failed to check batch status: {exc}")


# ---------------------------------------------------------------------------
# CRM review workflow (Phase 3): multi-tenant "human-in-the-loop" grading.
#
#   1. CRM calls POST /submissions with a homework file -> we grade it with
#      AI and store the result as status="pending_review".
#   2. The CRM's teacher-facing UI calls GET /submissions?status=pending_review
#      &teacher_external_id=... to show that teacher their review queue.
#   3. The teacher approves as-is, or edits the result, via
#      POST /submissions/{id}/review -> status becomes "approved".
#   4. We push the approved result to the organization's webhook_url
#      (retrying with backoff on failure); GET /submissions?status=approved
#      &since=... remains available as a polling fallback regardless.
#
# Every endpoint here is scoped to one organization (tenant), resolved
# from the X-Organization-Api-Key header -- one school's data is never
# visible to another's. Organizations themselves are created via
# POST /organizations, which is an admin operation gated by
# X-Internal-Api-Key (not by an organization key, since the org doesn't
# exist yet).
# ---------------------------------------------------------------------------
@app.post("/organizations")
def create_organization_endpoint(
    name: str = Body(..., embed=True),
    webhook_url: Optional[str] = Body(None, embed=True),
    x_internal_api_key: Optional[str] = Header(None),
):
    """
    Admin operation: registers a new tenant (school/training center) and
    returns its id and API key. THE API KEY IS SHOWN ONLY ONCE -- store
    it immediately; only its hash is kept server-side, so it cannot be
    recovered later (create a new organization instead if it's lost).

    `webhook_url` must be https and point to a public address (see webhook_security.py).
    The response also carries `webhook_secret`: the key used to verify the X-Webhook-Signature
    header of every webhook (it can be read again via GET /organizations/me/webhook-secret).
    """
    _check_internal_key(x_internal_api_key)
    name = (name or "").strip()
    if not name or len(name) > 255:
        raise HTTPException(status_code=422, detail="name must be 1-255 characters.")
    if webhook_url:
        try:
            webhook_url = webhook_security.validate_webhook_url(webhook_url)
        except webhook_security.WebhookUrlError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
    else:
        webhook_url = None
    org_id, api_key = db.create_organization(name=name, webhook_url=webhook_url)
    return {
        "organization_id": org_id,
        "api_key": api_key,
        "webhook_url": webhook_url,
        "webhook_secret": webhook_security.derive_org_secret(org_id),
    }


@app.get("/organizations/me/webhook-secret")
def get_webhook_secret(x_organization_api_key: Optional[str] = Header(None)):
    """The calling organization's webhook signing secret (null if the server has no signing
    secret configured -- then webhooks are sent unsigned)."""
    org = _authenticate_organization(x_organization_api_key)
    return {"organization_id": org["id"], "webhook_secret": webhook_security.derive_org_secret(org["id"])}


@app.post("/admin/reconcile")
def admin_reconcile(x_internal_api_key: Optional[str] = Header(None)):
    """Runs one pass of reconcile.py (re-queue or fail submissions stuck in pending_ai). Meant to
    be called every minute or two from cron / a scheduler; see reconcile.py."""
    _check_internal_key(x_internal_api_key)
    try:
        return reconcile.reconcile_once()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Reconciliation failed: {type(exc).__name__}")


@app.post("/admin/retention")
def admin_retention(x_internal_api_key: Optional[str] = Header(None)):
    """Runs one pass of retention.py (deletes old approved files and orphan uploads). Call it daily."""
    _check_internal_key(x_internal_api_key)
    try:
        return retention.purge_once()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Retention pass failed: {type(exc).__name__}")


@app.get("/metrics")
def metrics(x_internal_api_key: Optional[str] = Header(None)):
    """Prometheus text metrics (queue lengths, submissions by status, oldest waiting job, AI spend)."""
    _check_internal_key(x_internal_api_key)
    import metrics as metrics_module
    return Response(content=metrics_module.render(), media_type="text/plain; version=0.0.4; charset=utf-8")


MAX_ASSIGNMENT_TEXT_CHARS = 20000   # task text / answer key; keeps the shared prompt prefix bounded


@app.post("/submissions")
def submit_for_review(
    file: UploadFile = File(..., description="Homework file: PDF, PNG/JPG/WEBP, DOCX, or audio for Speaking."),
    teacher_external_id: Optional[str] = Form(None, description="Your CRM's id for the teacher who will review this."),
    teacher_name: Optional[str] = Form(None),
    student_external_id: Optional[str] = Form(None, description="Your CRM's id for the student."),
    student_name: Optional[str] = Form(None),
    external_assignment_id: Optional[str] = Form(None, description="Your CRM's id for the homework (assignment) this file answers. Send the same id for every student of one homework: the task text and answer key are stored once and the shared prompt prefix is cached by the AI providers, which lowers the cost per student."),
    task_text: str = Form("", description="The homework task text (optional; stored on the assignment, shared by all its students)."),
    answer_key: str = Form("", description="The teacher's reference answers (optional; stored on the assignment, never returned by the API)."),
    external_submission_id: Optional[str] = Form(None, description="Your CRM's own id for this submission -- resubmitting the same id returns the existing submission instead of duplicating it."),
    model_name: Optional[str] = Form(None),
    subject_key: str = Form(DEFAULT_SUBJECT_KEY),
    custom_subject_name: str = Form(""),
    language_key: str = Form(""),
    skill_key: str = Form(""),
    extra_instructions: str = Form(""),
    x_organization_api_key: Optional[str] = Header(None),
):
    """
    Submits one homework file into the review workflow: AI grades it in
    the background, then it waits at status="pending_review" for a
    teacher to approve or edit via POST /submissions/{id}/review. Returns
    immediately with the new submission's id and status="pending_ai".
    """
    org = _authenticate_organization(x_organization_api_key)

    _check_fields_or_422(
        teacher_external_id=teacher_external_id, teacher_name=teacher_name,
        student_external_id=student_external_id, student_name=student_name,
        external_assignment_id=external_assignment_id, external_submission_id=external_submission_id,
        model_name=model_name, subject_key=subject_key, custom_subject_name=custom_subject_name,
        language_key=language_key, skill_key=skill_key, extra_instructions=extra_instructions,
    )
    _check_org_rate_or_429(org["id"])

    file_bytes = _read_upload_or_413(file)
    try:
        validate_upload(file.filename, len(file_bytes))
    except (UnsupportedFileError, FileTooLargeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    if subject_key == "general" and not language_key and not custom_subject_name.strip():
        raise HTTPException(
            status_code=400,
            detail="custom_subject_name is required when subject_key='general'.",
        )

    if len(task_text) > MAX_ASSIGNMENT_TEXT_CHARS or len(answer_key) > MAX_ASSIGNMENT_TEXT_CHARS:
        raise HTTPException(status_code=400, detail=f"task_text and answer_key are limited to {MAX_ASSIGNMENT_TEXT_CHARS} characters each.")
    if (task_text or answer_key) and not external_assignment_id:
        raise HTTPException(status_code=400, detail="external_assignment_id is required when task_text or answer_key is sent.")

    # A client retry of an existing submission must not write a second copy of the file to disk
    # (nothing would ever refer to it): save the file only for a submission that is new.
    already_there = bool(external_submission_id) and db.submission_exists(org["id"], external_submission_id)
    saved_path = "" if already_there else _save_upload_or_400(file_bytes, file.filename)

    assignment_id = (
        db.get_or_create_assignment(org["id"], external_assignment_id, task_text or None, answer_key or None)
        if external_assignment_id else None
    )

    submission_id = db.create_submission(
        organization_id=org["id"],
        assignment_id=assignment_id,
        teacher_external_id=teacher_external_id,
        teacher_name=teacher_name,
        student_external_id=student_external_id,
        student_name=student_name,
        external_submission_id=external_submission_id,
        subject_key=subject_key,
        custom_subject_name=custom_subject_name,
        language_key=language_key,
        skill_key=skill_key,
        model_name=model_name or "",
        backend=MODEL_BACKEND,
        extra_instructions=extra_instructions,
        input_file_path=str(saved_path) if saved_path else "",
    )

    existing = db.get_submission(submission_id, organization_id=org["id"])
    if existing and existing["status"] == "pending_ai":
        # Idempotent: enqueue_submission_grading returns the live job instead of creating a
        # second (paid) one when this is a client retry of a submission already queued/running --
        # and queues a fresh job when the previous one was lost (that is how a client retry
        # recovers a submission whose first enqueue failed).
        try:
            jobs.enqueue_submission_grading(submission_id)
        except Exception as exc:
            # The row exists but the queue is unreachable. Tell the client to retry (same
            # external_submission_id is safe); reconcile.py also picks the row up once Redis is back.
            logger.error("could not enqueue grading for submission %s: %s", submission_id, type(exc).__name__)
            raise HTTPException(
                status_code=503,
                detail={
                    "message": "The grading queue is unavailable; the submission was saved. Retry the same request later.",
                    "submission_id": submission_id,
                },
            )

    return db.get_submission(submission_id, organization_id=org["id"])


@app.get("/submissions/{submission_id}")
def get_submission_endpoint(submission_id: str, x_organization_api_key: Optional[str] = Header(None)):
    """Fetches one submission's full state (AI result, teacher review, status)."""
    org = _authenticate_organization(x_organization_api_key)
    submission = db.get_submission(submission_id, organization_id=org["id"])
    if not submission:
        raise HTTPException(status_code=404, detail="Submission not found.")
    return submission


@app.get("/submissions")
def list_submissions_endpoint(
    status: Optional[str] = None,
    teacher_external_id: Optional[str] = None,
    student_external_id: Optional[str] = None,
    since: Optional[str] = None,
    limit: int = Query(50, ge=0),
    offset: int = Query(0, ge=0),
    x_organization_api_key: Optional[str] = Header(None),
):
    """
    Lists submissions for the calling organization. Two main uses:
      - A teacher's review queue: ?status=pending_review&teacher_external_id=...
      - A CRM's polling fallback for missed webhooks: ?status=approved&since=2026-01-01T00:00:00Z
    `since` filters by approved_at when status=approved, otherwise by created_at.
    """
    org = _authenticate_organization(x_organization_api_key)

    since_dt = None
    if since:
        try:
            since_dt = datetime.fromisoformat(since.replace("Z", "+00:00"))
        except ValueError:
            raise HTTPException(status_code=400, detail="`since` must be an ISO 8601 timestamp, e.g. 2026-01-01T00:00:00Z")

    results = db.list_submissions(
        organization_id=org["id"],
        status=status,
        teacher_external_id=teacher_external_id,
        student_external_id=student_external_id,
        since=since_dt,
        limit=min(limit, 200),
        offset=offset,
    )
    return {"count": len(results), "submissions": results}


@app.post("/submissions/{submission_id}/review")
def review_submission(
    submission_id: str,
    action: str = Body(..., description="'approve' (accept the AI result as-is) or 'edit' (save corrected_result and approve)."),
    corrected_result: Optional[dict] = Body(None, description="Required when action='edit': the corrected grading JSON, same schema as the AI's result (overall_score 0-100 and a questions list)."),
    reviewer_note: Optional[str] = Body(None),
    reviewer_external_id: Optional[str] = Body(None, description="Your CRM's id of the teacher making this decision (kept in the audit trail)."),
    x_organization_api_key: Optional[str] = Header(None),
):
    """
    A teacher's decision on a pending_review submission. Either way, the
    submission becomes status="approved" -- that's the signal the result
    is final and safe to show in the student's gradebook. This also
    delivers a webhook (if the organization has one configured) and, only when
    COLLECT_TRAINING_DATA is enabled, logs the approved result as a verified "gold" example.
    """
    org = _authenticate_organization(x_organization_api_key)
    _check_fields_or_422(reviewer_note=reviewer_note, reviewer_external_id=reviewer_external_id)

    if action not in ("approve", "edit"):
        raise HTTPException(status_code=400, detail="action must be 'approve' or 'edit'.")
    if action == "edit":
        if not corrected_result:
            raise HTTPException(status_code=400, detail="corrected_result is required when action='edit'.")
        try:
            validate_review_result(corrected_result)
        except ReviewResultError as exc:
            raise HTTPException(status_code=422, detail=str(exc))

    existing = db.get_submission(submission_id, organization_id=org["id"], include_internal=True)
    if not existing:
        raise HTTPException(status_code=404, detail="Submission not found.")
    if existing["status"] not in ("pending_review",):
        raise HTTPException(
            status_code=409,
            detail=f"Submission is in status '{existing['status']}', not 'pending_review' -- nothing to review yet or already reviewed.",
        )

    try:
        updated = db.approve_submission(
            submission_id,
            organization_id=org["id"],
            corrected_result=corrected_result if action == "edit" else None,
            reviewer_note=reviewer_note,
            reviewer_external_id=reviewer_external_id,
        )
    except db.InvalidTransition as exc:
        # Lost a race against another approval (the check above is only a fast path).
        raise HTTPException(
            status_code=409,
            detail=f"Submission is in status '{exc.current_status}', not 'pending_review' -- already reviewed.",
        )
    if updated is None:
        raise HTTPException(status_code=404, detail="Submission not found.")

    # Feed the fine-tuning dataset with this teacher-verified example -- only if collection is
    # explicitly enabled (training_logger enforces the same flag).
    if config.COLLECT_TRAINING_DATA and existing.get("input_file_path"):
        try:
            system_prompt = build_system_prompt(
                subject_key=updated["subject_key"],
                custom_subject_name=updated.get("custom_subject_name") or "",
                language_key=updated.get("language_key") or "",
                skill_key=updated.get("skill_key") or "",
            )
            training_logger.log_gold_from_submission(
                input_file_path=existing.get("input_file_path", ""),
                system_prompt=system_prompt,
                user_prompt=build_user_prompt(""),
                final_result=updated["final_result"],
                was_corrected=(action == "edit"),
                subject_key=updated["subject_key"],
                custom_subject_name=updated.get("custom_subject_name") or "",
                language_key=updated.get("language_key") or "",
                skill_key=updated.get("skill_key") or "",
                model_name=updated.get("model_name") or "",
                backend=updated.get("backend") or "",
            )
        except Exception:
            logger.exception("gold-example logging failed for submission %s", submission_id)  # never block the review

    # Push to the CRM's webhook (if configured); polling remains available regardless. Only the
    # request that actually approved the submission reaches this point, so exactly one webhook.
    try:
        jobs.enqueue_webhook_delivery(submission_id)
    except Exception:
        logger.exception("could not enqueue webhook for submission %s", submission_id)  # best-effort; polling still works

    return updated


@app.delete("/submissions/{submission_id}")
def erase_submission(
    submission_id: str,
    reviewer_external_id: Optional[str] = Query(None, description="Your CRM's id of whoever asked for the erasure (audit trail)."),
    x_organization_api_key: Optional[str] = Header(None),
):
    """
    Erases a submission and the student's file (a deletion request from a school or a parent). The
    row, the stored file and the webhook log are removed; an anonymous audit event stays. Refused
    (409) while the AI is still grading it. The grade in YOUR gradebook is not touched.
    """
    org = _authenticate_organization(x_organization_api_key)
    _check_fields_or_422(reviewer_external_id=reviewer_external_id)
    try:
        path = db.delete_submission(submission_id, organization_id=org["id"], actor=reviewer_external_id)
    except db.InvalidTransition:
        raise HTTPException(status_code=409, detail="The submission is still being graded; erase it once grading finished.")
    if path is None:
        raise HTTPException(status_code=404, detail="Submission not found.")
    if path:
        try:
            p = Path(path)
            if retention._inside(p, Path(utils.UPLOAD_DIR)):
                p.unlink(missing_ok=True)
        except OSError:
            logger.exception("could not delete the file of erased submission %s", submission_id)
    return {"erased": True, "submission_id": submission_id}


@app.post("/submissions/{submission_id}/regrade")
def regrade_submission(submission_id: str, x_organization_api_key: Optional[str] = Header(None)):
    """
    Grades a submission again: for one whose AI grading failed (provider outage, ...) or whose
    AI result the teacher does not trust. Approved submissions are final and cannot be regraded.
    The cached verdict for an identical file is bypassed, so this really calls the models again.
    """
    org = _authenticate_organization(x_organization_api_key)
    try:
        updated = db.reset_for_regrade(submission_id, organization_id=org["id"])
    except db.RegradeLimitReached as exc:
        raise HTTPException(
            status_code=409,
            detail=f"This submission was already regraded {exc.limit} times (the limit); ask an administrator.",
        )
    except db.InvalidTransition as exc:
        raise HTTPException(
            status_code=409,
            detail=f"Submission is in status '{exc.current_status}'; only 'grading_failed' or 'pending_review' can be regraded.",
        )
    if updated is None:
        raise HTTPException(status_code=404, detail="Submission not found.")
    try:
        from worker_tasks import regrade_flag_key
        conn = jobs.get_redis_connection()
        conn.set(regrade_flag_key(submission_id), "1", ex=24 * 3600)
        jobs.enqueue_submission_grading(submission_id, force=True)
    except Exception as exc:
        logger.error("could not enqueue regrade for %s: %s", submission_id, type(exc).__name__)
        raise HTTPException(
            status_code=503,
            detail={"message": "The grading queue is unavailable; retry the regrade request later.", "submission_id": submission_id},
        )
    return updated
