"""
jobs.py
Thin wrapper around Redis + RQ for the async job queue behind
POST /jobs and GET /jobs/{job_id} in api.py.

Why a queue instead of grading inside the HTTP request (like /analyze
does): when many students submit at once -- e.g. from a CRM -- handling
each one synchronously means the caller's HTTP connection sits open for
however long Gemini takes (often 10-30+ seconds), which risks CRM
timeouts, and many requests arriving together can hit Gemini's own rate
limits. With a queue, POST /jobs returns in milliseconds (it just records
the job and returns an id); one or more separate `rq worker` processes
pull jobs off the queue and grade them one at a time (or several in
parallel if you run several workers), retrying/backing off as needed
without ever blocking the API itself.
"""

import redis
from rq import Queue, Retry
from rq.job import Job

from config import (
    JOB_QUEUE_NAME,
    JOB_RESULT_TTL_SECONDS,
    JOB_TIMEOUT_SECONDS,
    REDIS_URL,
    WEBHOOK_MAX_RETRIES,
)

_redis_conn = None
_queue = None
_webhook_queue = None

WEBHOOK_QUEUE_NAME = "webhook_delivery"
# Backoff schedule between webhook retry attempts (seconds): quick at
# first in case it was a momentary blip, then progressively longer.
WEBHOOK_RETRY_INTERVALS = [10, 30, 120, 600, 1800]


def get_redis_connection():
    global _redis_conn
    if _redis_conn is None:
        _redis_conn = redis.from_url(REDIS_URL)
    return _redis_conn


def get_queue() -> Queue:
    global _queue
    if _queue is None:
        _queue = Queue(JOB_QUEUE_NAME, connection=get_redis_connection())
    return _queue


def get_webhook_queue() -> Queue:
    global _webhook_queue
    if _webhook_queue is None:
        _webhook_queue = Queue(WEBHOOK_QUEUE_NAME, connection=get_redis_connection())
    return _webhook_queue


def enqueue_grading_job(**task_kwargs) -> str:
    """
    Enqueues one grading job and returns its job_id immediately (does not
    wait for grading to finish). task_kwargs are passed straight through
    to worker_tasks.grade_homework_task.
    """
    from worker_tasks import grade_homework_task

    job = get_queue().enqueue(
        grade_homework_task,
        kwargs=task_kwargs,
        job_timeout=JOB_TIMEOUT_SECONDS,
        result_ttl=JOB_RESULT_TTL_SECONDS,
        failure_ttl=JOB_RESULT_TTL_SECONDS,
    )
    return job.id


GRADING_JOB_ID_PREFIX = "grade-"
_IN_FLIGHT_STATUSES = {"queued", "started", "scheduled", "deferred"}


def grading_job_id(submission_id: str) -> str:
    """One deterministic RQ job id per submission, so a duplicate request can see the job
    that is already queued/running instead of paying for a second grading."""
    return f"{GRADING_JOB_ID_PREFIX}{submission_id}"


def grading_job_in_flight(submission_id: str) -> bool:
    """True if this submission's grading job is currently queued, running or scheduled."""
    try:
        job = Job.fetch(grading_job_id(submission_id), connection=get_redis_connection())
    except Exception:  # NoSuchJobError (never queued, expired, or Redis was flushed)
        return False
    status = job.get_status(refresh=True)
    # RQ returns a JobStatus (str, Enum); str() of it is "JobStatus.QUEUED" on current Python
    # versions, so compare the enum's plain value, never str(status).
    return getattr(status, "value", status) in _IN_FLIGHT_STATUSES


def enqueue_submission_grading(submission_id: str, force: bool = False) -> str:
    """
    Enqueues AI grading for a CRM-review-workflow submission (db.py).
    Unlike enqueue_grading_job, this doesn't return a result directly --
    the worker writes the AI result straight into the submissions table
    (status: pending_ai -> pending_review), which the teacher's review
    queue (GET /submissions) picks up from there.

    Idempotent: if a grading job for this submission is already queued or running, no
    second (paid) job is created and the existing job id is returned. force=True only skips
    the short duplicate-request lock (used by the reconciler and by regrade).
    """
    from worker_tasks import grade_submission_task

    conn = get_redis_connection()
    job_id = grading_job_id(submission_id)
    lock_key = f"hw:enqueue-lock:{submission_id}"

    if grading_job_in_flight(submission_id):
        return job_id  # never create a second job next to a live one, even with force
    # Short lock so two simultaneous requests cannot both pass the check above.
    got_lock = conn.set(lock_key, "1", nx=True, ex=30)
    if not got_lock and not force:
        return job_id
    try:
        try:  # a finished/failed job with this id may still be stored -- replace it
            Job.fetch(job_id, connection=conn).delete()  # not in flight (checked above)
        except Exception:
            pass
        job = get_queue().enqueue(
            grade_submission_task,
            args=(submission_id,),
            job_id=job_id,
            job_timeout=JOB_TIMEOUT_SECONDS,
            result_ttl=60,  # we don't use the RQ result for this, just its pass/fail
            failure_ttl=JOB_RESULT_TTL_SECONDS,
        )
        return job.id
    except Exception:
        conn.delete(lock_key)  # enqueue failed: let the next attempt (client retry / reconciler) proceed
        raise


def enqueue_webhook_delivery(submission_id: str) -> str:
    """
    Enqueues delivery of an approved submission's result to its
    organization's webhook_url, on a separate queue from grading so a
    burst of webhook deliveries never delays homework grading (or vice
    versa). Uses RQ's Retry so transient failures (CRM endpoint briefly
    down) retry with backoff automatically -- no blocking sleeps tying up
    a worker.
    """
    from worker_tasks import deliver_webhook_task

    job = get_webhook_queue().enqueue(
        deliver_webhook_task,
        args=(submission_id,),
        job_timeout=60,
        retry=Retry(max=WEBHOOK_MAX_RETRIES, interval=WEBHOOK_RETRY_INTERVALS[:WEBHOOK_MAX_RETRIES] or [10]),
    )
    return job.id


def get_job_status(job_id: str) -> dict:
    """
    Returns {"job_id", "status", ...}. status is one of:
    "queued", "started", "finished", "failed", "not_found".
    "finished" responses include "result" (the grading JSON);
    "failed" responses include a short "error" message.
    """
    conn = get_redis_connection()
    try:
        job = Job.fetch(job_id, connection=conn)
    except Exception:
        return {"job_id": job_id, "status": "not_found"}

    status = job.get_status(refresh=True)
    status = getattr(status, "value", status)  # plain "queued"/"started"/... string
    response = {"job_id": job_id, "status": status}

    if status == "finished":
        response["result"] = job.result
    elif status == "failed":
        exc_text = job.exc_info or ""
        last_line = exc_text.strip().splitlines()[-1] if exc_text.strip() else "Unknown error"
        response["error"] = last_line

    return response
