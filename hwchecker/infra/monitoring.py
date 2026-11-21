"""
hwchecker/infra/monitoring.py
Per spec section 9: every provider call logs token/cost/latency/error-
type, feeding a real-time dashboard + alerts (Prometheus/Grafana +
Sentry). Structured logging (stdlib `logging`) always works with zero
extra dependencies; Prometheus counters/histograms and Sentry error
capture are used IF those packages happen to be installed and
configured, but their absence never breaks grading -- monitoring must
never be able to take down the actual grading pipeline.
"""

import logging
import time

logger = logging.getLogger("hwchecker.monitoring")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False  # don't also flood the root logger (see project's homework_checker.py for why this matters)

_prometheus_metrics = None


def _get_prometheus_metrics():
    """Lazily creates Prometheus metrics if prometheus_client is installed. Returns None otherwise."""
    global _prometheus_metrics
    if _prometheus_metrics is not None:
        return _prometheus_metrics or None
    try:
        from prometheus_client import Counter, Histogram

        _prometheus_metrics = {
            "requests_total": Counter(
                "hwchecker_provider_requests_total", "Total provider calls", ["provider", "status"]
            ),
            "latency_seconds": Histogram(
                "hwchecker_provider_latency_seconds", "Provider call latency", ["provider"]
            ),
            "cost_usd_total": Counter("hwchecker_cost_usd_total", "Estimated USD cost", ["provider"]),
        }
    except ImportError:
        _prometheus_metrics = False  # sentinel: "checked, not available"
    return _prometheus_metrics or None


def log_grading_event(
    provider_name: str,
    input_tokens: int,
    output_tokens: int,
    estimated_cost_usd: float,
    latency_ms: int,
    success: bool,
    error_type: str = None,
) -> None:
    """Called after every provider call (success or failure) -- the single instrumentation point everything else hooks into."""
    status = "success" if success else "failure"
    message = (
        f"provider={provider_name} status={status} input_tokens={input_tokens} "
        f"output_tokens={output_tokens} cost_usd={estimated_cost_usd:.6f} latency_ms={latency_ms}"
    )
    if error_type:
        message += f" error_type={error_type}"
    logger.info(message)

    metrics = _get_prometheus_metrics()
    if metrics:
        try:
            metrics["requests_total"].labels(provider=provider_name, status=status).inc()
            metrics["latency_seconds"].labels(provider=provider_name).observe(latency_ms / 1000.0)
            metrics["cost_usd_total"].labels(provider=provider_name).inc(estimated_cost_usd)
        except Exception:
            pass  # monitoring must never break the actual grading pipeline

    if not success:
        try:
            import sentry_sdk

            sentry_sdk.capture_message(
                f"hwchecker provider failure: {provider_name} ({error_type})", level="warning"
            )
        except ImportError:
            pass
        except Exception:
            pass


class timed_event:
    """
    Context manager convenience wrapper:
        with timed_event("gemini-3.5-flash") as ev:
            result = provider.grade_homework(...)
            ev.tokens(result.raw_tokens["input"], result.raw_tokens["output"])
            ev.cost(provider.estimate_cost(...))
    Logs automatically on exit (success or exception).
    """

    def __init__(self, provider_name: str):
        self.provider_name = provider_name
        self._input_tokens = 0
        self._output_tokens = 0
        self._cost = 0.0
        self._start = None

    def tokens(self, input_tokens: int, output_tokens: int):
        self._input_tokens, self._output_tokens = input_tokens, output_tokens

    def cost(self, cost_usd: float):
        self._cost = cost_usd

    def __enter__(self):
        self._start = time.monotonic()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        latency_ms = int((time.monotonic() - self._start) * 1000)
        log_grading_event(
            provider_name=self.provider_name,
            input_tokens=self._input_tokens,
            output_tokens=self._output_tokens,
            estimated_cost_usd=self._cost,
            latency_ms=latency_ms,
            success=exc_type is None,
            error_type=exc_type.__name__ if exc_type else None,
        )
        return False  # never swallow exceptions
