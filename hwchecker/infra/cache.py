"""
hwchecker/infra/cache.py
Submission-hash cache per spec section 9: if the exact same homework
(same file bytes + same subject/rubric) is submitted twice, don't pay
for grading it twice. Redis-backed (reusing REDIS_URL) with a
configurable TTL.

Distinct from cache_registry.py (project root) and Gemini's own
explicit context caching -- those cache the shared SYSTEM PROMPT across
many different submissions; this caches the RESULT of one specific
submission so an identical resubmission is free.
"""

import hashlib
import json
import time
from pathlib import Path
from typing import Optional

from ._redis import get_redis

DEFAULT_TTL_SECONDS = 60 * 60 * 24 * 7  # 1 week

_memory_cache = {}  # fallback: {hash: (expires_at, result_json)}


def _get_redis():
    # Shared + memoised, see _redis.py (kept under this name so callers/tests that patch it still work).
    return get_redis()


def compute_submission_hash(file_path, subject_id: str, extra_instructions: str = "") -> str:
    """
    Hashes the file's actual bytes + subject_id + extra_instructions --
    NOT the filename (so re-uploading the same content under a different
    name still hits the cache), and IS sensitive to subject/instructions
    (so grading the same file under a different subject correctly misses).
    """
    hasher = hashlib.sha256()
    hasher.update(Path(file_path).read_bytes())
    hasher.update(subject_id.encode("utf-8"))
    hasher.update(extra_instructions.encode("utf-8"))
    return hasher.hexdigest()


def get(submission_hash: str) -> Optional[dict]:
    """Returns the cached grading result dict, or None on a cache miss."""
    r = _get_redis()
    if r is not None:
        try:
            raw = r.get(f"submission_cache:{submission_hash}")
            return json.loads(raw) if raw else None
        except Exception:
            pass

    entry = _memory_cache.get(submission_hash)
    if entry and entry[0] > time.time():
        return entry[1]
    return None


def set(submission_hash: str, result: dict, ttl_seconds: int = DEFAULT_TTL_SECONDS) -> None:
    r = _get_redis()
    if r is not None:
        try:
            r.set(f"submission_cache:{submission_hash}", json.dumps(result), ex=ttl_seconds)
            return
        except Exception:
            pass
    _memory_cache[submission_hash] = (time.time() + ttl_seconds, result)
