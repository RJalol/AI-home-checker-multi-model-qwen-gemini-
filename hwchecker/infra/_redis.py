"""
hwchecker/infra/_redis.py
One shared, memoised Redis connection lookup for cache.py, circuit_breaker.py
and rate_limiter.py. Each of them used to open a fresh connection and PING on
every single call -- fine on a laptop, but once they sit on the grading hot path
a Redis host that is down or unreachable would add up to a 1-second connect
timeout to every one of those calls. The outcome (client or None) is remembered
for a short while instead, and each module keeps its in-memory fallback.
"""

import time

_CLIENT_TTL_SECONDS = 30      # re-check a working connection this often
_UNAVAILABLE_TTL_SECONDS = 15  # re-try an unreachable Redis this often

_state = {"client": None, "checked_at": None}


def get_redis():
    """Returns a connected redis client, or None if Redis isn't installed/reachable."""
    now = time.monotonic()
    checked_at = _state["checked_at"]
    if checked_at is not None:
        ttl = _CLIENT_TTL_SECONDS if _state["client"] is not None else _UNAVAILABLE_TTL_SECONDS
        if now - checked_at < ttl:
            return _state["client"]

    client = None
    try:
        import redis
        from config import REDIS_URL

        client = redis.from_url(REDIS_URL, socket_connect_timeout=1)
        client.ping()
    except Exception:
        client = None

    _state["client"] = client
    _state["checked_at"] = now
    return client


def reset() -> None:
    """Forget the remembered connection state (tests)."""
    _state["client"] = None
    _state["checked_at"] = None
