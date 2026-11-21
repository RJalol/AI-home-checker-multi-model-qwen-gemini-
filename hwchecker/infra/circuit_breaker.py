"""
hwchecker/infra/circuit_breaker.py
Per-provider circuit breaker per spec section 9: after N consecutive
failures, a provider is marked "open" (skip calling it, go straight to
fallback) for a cooldown period, then "half-open" (try it once; close
again on success, re-open on failure). Redis-backed (reusing this
project's existing REDIS_URL) so the breaker state is shared across
every worker process, not just the one that happened to see the failures.
"""

import time

from ._redis import get_redis

FAILURE_THRESHOLD = 5           # consecutive failures before opening
COOLDOWN_SECONDS = 60            # how long to stay open before trying again
FAILURE_WINDOW_SECONDS = 300      # failures older than this don't count toward the threshold

_memory_state = {}  # fallback when Redis isn't reachable: {provider_name: {"failures": [...], "opened_at": float|None}}


class CircuitOpenError(Exception):
    pass


def _get_redis():
    # Shared + memoised, see _redis.py (kept under this name so callers/tests that patch it still work).
    return get_redis()


def _key(provider_name: str, suffix: str) -> str:
    return f"circuit_breaker:{provider_name}:{suffix}"


def is_open(provider_name: str) -> bool:
    """True if this provider's circuit is currently open (should be skipped)."""
    r = _get_redis()
    now = time.time()

    if r is not None:
        try:
            opened_at = r.get(_key(provider_name, "opened_at"))
            if opened_at is None:
                return False
            opened_at = float(opened_at)
            if now - opened_at >= COOLDOWN_SECONDS:
                r.delete(_key(provider_name, "opened_at"))  # cooldown elapsed -- half-open: allow one attempt
                return False
            return True
        except Exception:
            pass

    state = _memory_state.get(provider_name)
    if not state or state.get("opened_at") is None:
        return False
    if now - state["opened_at"] >= COOLDOWN_SECONDS:
        state["opened_at"] = None
        return False
    return True


def record_success(provider_name: str) -> None:
    """Resets the failure count -- called after any successful provider call."""
    r = _get_redis()
    if r is not None:
        try:
            r.delete(_key(provider_name, "failures"))
            r.delete(_key(provider_name, "opened_at"))
            return
        except Exception:
            pass
    _memory_state[provider_name] = {"failures": [], "opened_at": None}


def record_failure(provider_name: str) -> None:
    """Records a failure; opens the circuit if the threshold is crossed
    within the failure window."""
    now = time.time()
    r = _get_redis()

    if r is not None:
        try:
            key = _key(provider_name, "failures")
            r.rpush(key, now)
            r.expire(key, FAILURE_WINDOW_SECONDS)
            raw = r.lrange(key, 0, -1)
            recent = [float(t) for t in raw if now - float(t) <= FAILURE_WINDOW_SECONDS]
            if len(recent) >= FAILURE_THRESHOLD:
                r.set(_key(provider_name, "opened_at"), now, ex=COOLDOWN_SECONDS * 2)
            return
        except Exception:
            pass

    state = _memory_state.setdefault(provider_name, {"failures": [], "opened_at": None})
    state["failures"] = [t for t in state["failures"] if now - t <= FAILURE_WINDOW_SECONDS] + [now]
    if len(state["failures"]) >= FAILURE_THRESHOLD:
        state["opened_at"] = now


def call_with_breaker(provider_name: str, fn, *args, **kwargs):
    """
    Runs fn(*args, **kwargs) through the circuit breaker for
    `provider_name`: raises CircuitOpenError immediately (without
    calling fn) if the circuit is open; otherwise calls fn, recording
    success/failure. Re-raises whatever fn raised, after recording it.
    """
    if is_open(provider_name):
        raise CircuitOpenError(f"Circuit breaker is open for provider '{provider_name}' -- too many recent failures.")
    try:
        result = fn(*args, **kwargs)
        record_success(provider_name)
        return result
    except Exception:
        record_failure(provider_name)
        raise
