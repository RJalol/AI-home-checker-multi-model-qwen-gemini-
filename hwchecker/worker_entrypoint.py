"""
hwchecker/worker_entrypoint.py
The RQ task function actually executed by `rq worker homework_grading`
for homework routed through the new tiered GradingRouter pipeline
(providers/ + subjects/ + checkers/ + router/), as opposed to the
simpler single-call path in the project's existing worker_tasks.py.

Kept as a plain, importable module-level function (not a method/closure)
because RQ pickles a reference to it (module path + function name) to
store in Redis -- the worker process re-imports it to run the job.
"""

from pathlib import Path

from hwchecker.router.grading_router import GradingRouter

_router = None


def _get_router() -> GradingRouter:
    global _router
    if _router is None:
        _router = GradingRouter()
    return _router


def grade_homework_via_router(homework: dict) -> dict:
    """
    homework: same shape GradingRouter.grade() expects. `content.file_path`,
    if present, is passed through as-is (already a str/Path at this point --
    RQ serializes the whole dict through Redis, so callers should pass
    file_path as a string, not a Path object, when enqueuing).
    """
    content = homework.get("content") or {}
    if content.get("file_path"):
        content["file_path"] = Path(content["file_path"])
        homework["content"] = content

    return _get_router().grade(homework)
