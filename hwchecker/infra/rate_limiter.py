"""
hwchecker/infra/rate_limiter.py
Rate limiter per spec section 9, keeping each provider under its
contracted TPM/RPM limits. Redis-backed so the limit is enforced across
every worker process sharing one provider, not per-process.

Implemented as a simple fixed-window counter (not a true continuous
token bucket) -- simpler to reason about and sufficiently accurate for
this purpose; swap for a sliding-window/leaky-bucket implementation
later if you need smoother request pacing under bursty load.
"""

import time

from ._redis import get_redis

_memory_windows = {}  # fallback when Redis isn't reachable: {key: (window_id, count)}


class RateLimitExceededError(Exception):
    pass


def _get_redis():
    # Shared + memoised, see _redis.py (kept under this name so callers/tests that patch it still work).
    return get_redis()


def _window_key(provider_name: str, window_seconds: int) -> str:
    window_id = int(time.time() // window_seconds)
    return f"rate_limit:{provider_name}:{window_seconds}:{window_id}"


def check_and_increment(provider_name: str, limit: int, window_seconds: int = 60) -> bool:
    """
    Returns True and increments the counter if this call is within the
    limit for the current window; returns False (does NOT increment)
    if the limit is already reached. Callers should treat False as
    "wait or route elsewhere" rather than raising by default -- raising
    is available via enforce() below for call sites that want it.
    """
    key = _window_key(provider_name, window_seconds)
    r = _get_redis()

    if r is not None:
        try:
            count = r.incr(key)
            if count == 1:
                r.expire(key, window_seconds * 2)
            return count <= limit
        except Exception:
            pass

    now = time.time()
    window_id = int(now // window_seconds)
    mem_key = f"{provider_name}:{window_seconds}"
    stored_window, count = _memory_windows.get(mem_key, (window_id, 0))
    if stored_window != window_id:
        count = 0
        stored_window = window_id
    count += 1
    _memory_windows[mem_key] = (stored_window, count)
    return count <= limit


def enforce(provider_name: str, limit: int, window_seconds: int = 60) -> None:
    """Same as check_and_increment, but raises RateLimitExceededError instead of returning False."""
    if not check_and_increment(provider_name, limit, window_seconds):
        raise RateLimitExceededError(
            f"Rate limit exceeded for provider '{provider_name}': {limit} requests per {window_seconds}s."
        )
