"""
metrics.py
Prometheus text-format metrics for GET /metrics (protected by X-Internal-Api-Key). No extra
dependency: the exposition format is plain text. Point Prometheus (or any scraper) at it and use
deploy/alerts.yml as a starting set of alert rules.

    hw_submissions{status="pending_ai"}      number of submissions in each status
    hw_oldest_pending_ai_seconds             age of the oldest submission still waiting for the AI
    hw_queue_length{queue="homework_grading"} jobs waiting in each queue
    hw_queue_failed{queue="..."}             jobs in the failed registry of each queue
    hw_ai_spend_usd{period="day"|"month"}    estimated AI spend (list prices, UTC periods)
    hw_schema_version                        highest applied schema migration
"""

import logging

import config

logger = logging.getLogger("hwchecker.metrics")


def _line(name: str, value, labels: dict = None) -> str:
    if labels:
        rendered = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
        return f"{name}{{{rendered}}} {value}"
    return f"{name} {value}"


def _safe(collector, lines: list, label: str) -> None:
    """Runs one collector; a failing source (Redis down, ...) is reported as hw_scrape_error, never a 500."""
    try:
        collector(lines)
    except Exception as exc:
        logger.warning("metrics source %s failed: %s", label, type(exc).__name__)
        lines.append(_line("hw_scrape_error", 1, {"source": label}))


def _collect_db(lines: list) -> None:
    import db
    lines.append("# TYPE hw_submissions gauge")
    for status, count in sorted(db.count_by_status().items()):
        lines.append(_line("hw_submissions", count, {"status": status}))
    lines.append("# TYPE hw_oldest_pending_ai_seconds gauge")
    lines.append(_line("hw_oldest_pending_ai_seconds", round(db.oldest_pending_ai_age_seconds(), 1)))
    import migrations
    lines.append("# TYPE hw_schema_version gauge")
    lines.append(_line("hw_schema_version", migrations.current_version(db.get_engine())))


def _collect_queues(lines: list) -> None:
    import jobs
    from rq import Queue
    from rq.registry import FailedJobRegistry
    conn = jobs.get_redis_connection()
    lines.append("# TYPE hw_queue_length gauge")
    lines.append("# TYPE hw_queue_failed gauge")
    for name in dict.fromkeys([config.JOB_QUEUE_NAME, "homework_grading", "webhook_delivery"]):
        queue = Queue(name, connection=conn)
        lines.append(_line("hw_queue_length", queue.count, {"queue": name}))
        lines.append(_line("hw_queue_failed", len(FailedJobRegistry(queue=queue)), {"queue": name}))


def _collect_spend(lines: list) -> None:
    from hwchecker.infra import budget
    spend = budget.current_spend()
    lines.append("# TYPE hw_ai_spend_usd gauge")
    lines.append(_line("hw_ai_spend_usd", round(spend.get("day", 0.0), 6), {"period": "day"}))
    lines.append(_line("hw_ai_spend_usd", round(spend.get("month", 0.0), 6), {"period": "month"}))


def render() -> str:
    lines = []
    _safe(_collect_db, lines, "database")
    _safe(_collect_queues, lines, "redis")
    _safe(_collect_spend, lines, "budget")
    return "\n".join(lines) + "\n"
