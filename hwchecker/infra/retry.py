"""
hwchecker/infra/retry.py
Exponential backoff retry, per spec section 9: max 3-5 attempts,
backoff 2/4/8/16s. Used to wrap provider calls and schema validation --
a malformed-JSON response gets re-prompted through this before being
treated as a real failure (see pipeline/schema.py's SchemaValidationError
and providers/base.py's AIProviderError as the two exception types this
is meant to retry on).
"""

import time
from functools import wraps
from typing import Callable, Tuple, Type

DEFAULT_BACKOFF_SECONDS = (2, 4, 8, 16)
DEFAULT_MAX_ATTEMPTS = 5  # len(DEFAULT_BACKOFF_SECONDS) + 1 (the first attempt isn't a "retry")


class RetryExhaustedError(Exception):
    def __init__(self, attempts: int, last_error: Exception):
        self.attempts = attempts
        self.last_error = last_error
        super().__init__(f"Gave up after {attempts} attempts. Last error: {last_error}")


def call_with_retry(
    fn: Callable,
    *args,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    backoff_seconds: Tuple[int, ...] = DEFAULT_BACKOFF_SECONDS,
    retry_on: Tuple[Type[Exception], ...] = (Exception,),
    on_retry: Callable = None,
    **kwargs,
):
    """
    Calls fn(*args, **kwargs), retrying on any exception in `retry_on`
    up to max_attempts times total, sleeping backoff_seconds[attempt]
    between attempts (clamped to the last value if there are more
    attempts than backoff values). Raises RetryExhaustedError if every
    attempt fails; returns the successful result otherwise.

    `on_retry(attempt, exc)` is called before each sleep, if given --
    useful for logging/monitoring (see infra/monitoring.py) without this
    module needing to know about it directly.
    """
    last_error = None
    for attempt in range(max_attempts):
        try:
            return fn(*args, **kwargs)
        except retry_on as exc:
            last_error = exc
            if attempt == max_attempts - 1:
                break
            if on_retry:
                try:
                    on_retry(attempt + 1, exc)
                except Exception:
                    pass
            delay = backoff_seconds[min(attempt, len(backoff_seconds) - 1)]
            time.sleep(delay)

    raise RetryExhaustedError(max_attempts, last_error)


def with_retry(
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    backoff_seconds: Tuple[int, ...] = DEFAULT_BACKOFF_SECONDS,
    retry_on: Tuple[Type[Exception], ...] = (Exception,),
):
    """Decorator form of call_with_retry, for wrapping a function definition directly."""

    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            return call_with_retry(
                fn, *args, max_attempts=max_attempts, backoff_seconds=backoff_seconds, retry_on=retry_on, **kwargs
            )

        return wrapper

    return decorator
