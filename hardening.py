"""
hardening.py
Input and abuse protections for the REST API, kept apart from api.py so they can be unit-tested:

  * read_upload_capped(): reads an upload in chunks and stops as soon as it exceeds the size limit
    (api.py used to read the whole body into memory first and compare the size afterwards).
  * check_field_lengths(): refuses over-long text fields up front (the database columns are only
    255 / 64 / 32 characters wide; a longer value used to end in HTTP 500).
  * check_org_rate(): per-organization fixed-window counters in Redis, a safety net against a leaked
    key or a CRM retry loop creating unlimited paid grading.
"""

import logging
import time
from typing import Optional

import config

logger = logging.getLogger("hwchecker.hardening")

_CHUNK = 1024 * 1024


class RequestTooLarge(Exception):
    pass


class FieldTooLong(ValueError):
    pass


class RateLimitExceeded(Exception):
    def __init__(self, scope: str, limit: int, retry_after: int):
        super().__init__(f"{scope} limit of {limit} submissions exceeded")
        self.scope = scope
        self.limit = limit
        self.retry_after = max(int(retry_after), 1)


def read_upload_capped(upload_file, max_bytes: Optional[int] = None) -> bytes:
    """All bytes of a starlette UploadFile (or any file object), raising RequestTooLarge as soon as
    more than `max_bytes` (default: the configured limit) have been read."""
    limit = max_bytes if max_bytes is not None else config.MAX_FILE_SIZE_MB * 1024 * 1024
    stream = getattr(upload_file, "file", upload_file)
    parts, total = [], 0
    while True:
        chunk = stream.read(_CHUNK)
        if not chunk:
            break
        total += len(chunk)
        if total > limit:
            raise RequestTooLarge(f"The file is larger than the limit of {config.MAX_FILE_SIZE_MB} MB.")
        parts.append(chunk)
    return b"".join(parts)


# field -> maximum characters. Matches the database columns (db.py) and config limits.
FIELD_LIMITS = {
    "teacher_external_id": 255, "teacher_name": 255, "student_external_id": 255, "student_name": 255,
    "external_assignment_id": 255, "external_submission_id": 255, "reviewer_external_id": 255,
    "model_name": 128, "subject_key": 64, "custom_subject_name": 255, "language_key": 32, "skill_key": 32,
}


def check_field_lengths(**fields) -> None:
    """Raises FieldTooLong naming the first field that is over its limit. None values are skipped."""
    limits = dict(FIELD_LIMITS)
    limits["extra_instructions"] = config.MAX_EXTRA_INSTRUCTIONS_CHARS
    limits["reviewer_note"] = config.MAX_REVIEWER_NOTE_CHARS
    for name, value in fields.items():
        if value is None:
            continue
        limit = limits.get(name)
        if limit is not None and len(str(value)) > limit:
            raise FieldTooLong(f"{name} is too long ({len(str(value))} characters; the limit is {limit}).")


def _window_hit(conn, key: str, ttl: int) -> int:
    count = conn.incr(key)
    if count == 1:
        conn.expire(key, ttl)
    return int(count)


def check_org_rate(organization_id: str, redis_connection=None, now: Optional[float] = None) -> None:
    """Counts one submission for this organization and raises RateLimitExceeded if the per-minute or
    per-day limit is passed. Fails OPEN (logs, allows) if Redis cannot be reached: grading needs the
    same Redis, so a Redis outage is reported by the enqueue step anyway."""
    per_minute, per_day = config.ORG_MAX_SUBMISSIONS_PER_MINUTE, config.ORG_MAX_SUBMISSIONS_PER_DAY
    if not per_minute and not per_day:
        return
    now = time.time() if now is None else now
    try:
        if redis_connection is None:
            import jobs
            redis_connection = jobs.get_redis_connection()
        if per_minute:
            window = int(now // 60)
            if _window_hit(redis_connection, f"hw:rate:min:{organization_id}:{window}", 120) > per_minute:
                raise RateLimitExceeded("per-minute", per_minute, 60 - (now % 60))
        if per_day:
            window = int(now // 86400)
            if _window_hit(redis_connection, f"hw:rate:day:{organization_id}:{window}", 2 * 86400) > per_day:
                raise RateLimitExceeded("per-day", per_day, 86400 - (now % 86400))
    except RateLimitExceeded:
        raise
    except Exception as exc:
        logger.warning("organization rate limit not checked (%s)", type(exc).__name__)
