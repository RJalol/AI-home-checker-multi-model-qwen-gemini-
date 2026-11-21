"""
hwchecker/infra/budget.py
Spend watch-dog: adds up the estimated cost of every provider call per UTC day and per UTC
month and raises an ALERT (a warning log line and, if sentry_sdk is installed, a Sentry
message) when the total reaches 80% and 100% of a budget you set.

It never blocks, slows down or degrades grading -- quality does not drop because a budget is
reached. It is a smoke detector for runaway cost (a retry loop, a spam upload, a model
accidentally pointed at an expensive tier), not a quota. Set the budget to the level you
would call abnormal (for example 150% of the monthly cost you expect).

    HWCHECKER_DAILY_BUDGET_USD=40       alert at $32 and $40 spent today (UTC)
    HWCHECKER_MONTHLY_BUDGET_USD=900    alert at $720 and $900 spent this month (UTC)

Both unset/0 = off (the default). Totals live in Redis when it is reachable (shared by every
worker) and in process memory otherwise. Costs are the providers' estimate_cost() figures --
list prices, not your invoice. Alerts are sent once per period and level.
"""

import logging
import os
import threading
import time
from typing import Dict, Optional

from ._redis import get_redis

logger = logging.getLogger("hwchecker.budget")

ALERT_LEVELS = (0.8, 1.0)
_DAY_KEY_TTL_SECONDS = 3 * 24 * 3600
_MONTH_KEY_TTL_SECONDS = 40 * 24 * 3600

_lock = threading.Lock()
_memory_totals: Dict[str, float] = {}
_alerted = set()   # (period key, level) already announced by this process


def _get_redis():
    # Same indirection as infra/cache.py: tests (and callers) can patch this name.
    return get_redis()


def _budget(name: str) -> float:
    raw = os.environ.get(name, "").strip()
    try:
        value = float(raw) if raw else 0.0
    except ValueError:
        return 0.0
    return value if value > 0 else 0.0


def daily_budget_usd() -> float:
    return _budget("HWCHECKER_DAILY_BUDGET_USD")


def monthly_budget_usd() -> float:
    return _budget("HWCHECKER_MONTHLY_BUDGET_USD")


def _period_keys(now: Optional[float] = None):
    stamp = time.gmtime(now if now is not None else time.time())
    return (
        ("day", f"hwchecker:spend:day:{time.strftime('%Y%m%d', stamp)}", _DAY_KEY_TTL_SECONDS),
        ("month", f"hwchecker:spend:month:{time.strftime('%Y%m', stamp)}", _MONTH_KEY_TTL_SECONDS),
    )


def _add(key: str, amount: float, ttl: int) -> float:
    r = _get_redis()
    if r is not None:
        try:
            total = float(r.incrbyfloat(key, amount))
            r.expire(key, ttl)
            return total
        except Exception:
            pass   # fall through to the in-process total
    with _lock:
        _memory_totals[key] = _memory_totals.get(key, 0.0) + amount
        return _memory_totals[key]


def _first_to_announce(key: str, level: float, ttl: int) -> bool:
    """True exactly once per (period, level) across all workers when Redis is up, else per process."""
    token = (key, level)
    with _lock:
        if token in _alerted:
            return False
        _alerted.add(token)
    r = _get_redis()
    if r is not None:
        try:
            return bool(r.set(f"{key}:alert:{level}", "1", nx=True, ex=ttl))
        except Exception:
            pass
    return True


def _alert(period: str, spent: float, budget: float, level: float) -> None:
    message = (
        f"hwchecker budget alert: {spent:.2f} USD spent this {period} (UTC) = "
        f"{spent / budget:.0%} of the {budget:.2f} USD budget (alert level {level:.0%}). "
        f"Grading is NOT limited; check the cost log for a retry loop, spam or a mis-pointed tier."
    )
    logger.warning(message)
    try:
        import sentry_sdk

        sentry_sdk.capture_message(message, level="warning")
    except Exception:
        pass


def record_spend(cost_usd: float, now: Optional[float] = None) -> None:
    """Adds one call's estimated cost to today's and this month's totals and raises any due alert.
    Does nothing when no budget is configured. Never raises."""
    try:
        if not cost_usd or cost_usd <= 0:
            return
        budgets = {"day": daily_budget_usd(), "month": monthly_budget_usd()}
        if not any(budgets.values()):
            return
        for period, key, ttl in _period_keys(now):
            budget = budgets[period]
            if not budget:
                continue
            spent = _add(key, float(cost_usd), ttl)
            # the highest level reached is announced (a single call can jump over 80%)
            reached = [level for level in ALERT_LEVELS if spent >= budget * level]
            for level in reached:
                if _first_to_announce(key, level, ttl):
                    _alert(period, spent, budget, level)
    except Exception:
        pass   # monitoring must never break the grading pipeline


def current_spend(now: Optional[float] = None) -> Dict[str, float]:
    """{"day": usd, "month": usd} recorded so far (0 when no budget is configured or nothing spent)."""
    out = {}
    for period, key, _ in _period_keys(now):
        value = 0.0
        r = _get_redis()
        if r is not None:
            try:
                raw = r.get(key)
                value = float(raw) if raw else 0.0
            except Exception:
                value = _memory_totals.get(key, 0.0)
        else:
            value = _memory_totals.get(key, 0.0)
        out[period] = round(value, 6)
    return out


def reset() -> None:
    """Forget in-process totals and announced alerts (tests)."""
    with _lock:
        _memory_totals.clear()
        _alerted.clear()
