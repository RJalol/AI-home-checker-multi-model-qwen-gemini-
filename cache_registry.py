"""
cache_registry.py
Tracks which Gemini explicit context caches currently exist for which
system prompts, so multiple processes (RQ workers, the API process, a
Streamlit session) reuse the SAME server-side cache instead of each
creating its own duplicate.

Why this matters at scale: system prompts (per subject / language+skill)
are identical across every request for that combination. Explicit
caching lets Gemini charge ~10% of the normal input rate for the cached
portion instead of full price, every time it's reused -- but only if
requests actually point at the same cache. With N worker processes each
blindly creating their own cache, you'd pay for N caches instead of
sharing 1. Backing this registry with Redis (which the project already
uses for the job queue) means every process looks up the same shared
state.

Falls back to a simple in-process dict if Redis isn't reachable -- still
correct, just not shared across processes (fine for a single Streamlit
session or light local use).
"""

import json
import time
from typing import Optional, Tuple

from config import REDIS_URL

_memory_registry = {}
_redis_client = None
_redis_checked = False

_UNCACHEABLE_MARKER = "__uncacheable__"


def _get_redis():
    """Lazily connects to Redis once per process; remembers if it failed
    so we don't retry a slow connection attempt on every single call."""
    global _redis_client, _redis_checked
    if _redis_checked:
        return _redis_client
    _redis_checked = True
    try:
        import redis

        client = redis.from_url(REDIS_URL, socket_connect_timeout=1)
        client.ping()
        _redis_client = client
    except Exception:
        _redis_client = None
    return _redis_client


def _redis_key(key: str) -> str:
    return f"gemini_cache:{key}"


def lookup(key: str) -> Tuple[str, Optional[str]]:
    """
    Returns one of:
      ("cached", cache_name)  -- a valid cache exists, use it
      ("uncacheable", None)   -- this prompt is known to be too small /
                                  otherwise unable to be cached; don't
                                  retry creating one every request
      ("unknown", None)       -- no info yet; try creating a cache
    """
    r = _get_redis()
    if r is not None:
        try:
            raw = r.get(_redis_key(key))
        except Exception:
            raw = None
        if raw:
            try:
                obj = json.loads(raw)
            except Exception:
                return "unknown", None
            if obj.get("cache_name") == _UNCACHEABLE_MARKER:
                return "uncacheable", None
            if obj.get("expires_at", 0) > time.time():
                return "cached", obj.get("cache_name")
        return "unknown", None

    entry = _memory_registry.get(key)
    if entry is None:
        return "unknown", None
    cache_name, expires_at = entry
    if expires_at <= time.time():
        return "unknown", None
    if cache_name == _UNCACHEABLE_MARKER:
        return "uncacheable", None
    return "cached", cache_name


def set_cache(key: str, cache_name: str, ttl_seconds: int) -> None:
    """Records a newly created cache, valid until (slightly before) its TTL."""
    # Refresh a little early so we never hand out a cache_name that's
    # about to expire mid-request.
    expires_at = time.time() + max(ttl_seconds - 30, 5)
    r = _get_redis()
    if r is not None:
        try:
            r.set(
                _redis_key(key),
                json.dumps({"cache_name": cache_name, "expires_at": expires_at}),
                ex=ttl_seconds,
            )
            return
        except Exception:
            pass
    _memory_registry[key] = (cache_name, expires_at)


def mark_uncacheable(key: str, retry_after_seconds: int = 3600) -> None:
    """
    Remembers that creating a cache for `key` failed (most commonly: the
    system prompt is under Gemini's minimum cacheable token count). We
    still retry periodically (retry_after_seconds) in case the prompt
    changes and grows past the threshold later.
    """
    set_cache(key, _UNCACHEABLE_MARKER, retry_after_seconds)
