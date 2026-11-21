"""
hwchecker/infra/deadline.py
The exception the job queue raises when a grading job has used up its time budget.

RQ's JobTimeoutException derives from Exception, so every `except Exception:` around a model call
would swallow it -- the provider would be reported as "failed", the router would retry and then try
the next provider, and the job kept running far past JOB_TIMEOUT_SECONDS (the alarm fires only once).
Wherever code catches Exception around work that can take long (model calls, retries, rate-limit
waits), put this first so the deadline always reaches the worker:

    try:
        ...
    except JobDeadline:
        raise
    except Exception as exc:
        ...

Without RQ installed (library / CLI use) there is no job deadline; the class is then an unused
BaseException subclass so the `except` clause above still works.
"""

try:
    from rq.timeouts import BaseTimeoutException as JobDeadline
except Exception:  # pragma: no cover - RQ not installed
    class JobDeadline(BaseException):
        """Placeholder: no job queue, so nothing ever raises it."""

__all__ = ["JobDeadline"]
