"""
hwchecker/infra/queue.py
Queue abstraction per spec section 9 ("SQS/RabbitMQ/Redis Streams
abstraksiyasi -- tanlovni loyihaning mavjud stackiga moslang"). This
project already has a working Redis+RQ queue (jobs.py at the project
root, backing POST /jobs and POST /submissions) -- rather than stand up
a second, competing queue implementation, this module is a thin
adapter that lets hwchecker's GradingRouter submit work through that
SAME queue/worker infrastructure. See jobs.py and worker_tasks.py for
the actual enqueue/consume implementation.
"""

from typing import Optional


def enqueue_grading(homework: dict) -> str:
    """
    Enqueues a homework dict (see GradingRouter.grade()'s docstring for
    its shape) onto the project's existing Redis+RQ queue, to be graded
    by hwchecker.router.grading_router.GradingRouter in a worker
    process. Returns a job_id to poll (see jobs.get_job_status()).
    """
    import jobs as project_jobs  # the project-root jobs.py, not this module

    return project_jobs.get_queue().enqueue(
        "hwchecker.worker_entrypoint.grade_homework_via_router",
        kwargs={"homework": homework},
        job_timeout=300,
        result_ttl=60 * 60 * 24,
    ).id


def get_status(job_id: str) -> Optional[dict]:
    import jobs as project_jobs

    return project_jobs.get_job_status(job_id)
