"""
hwchecker/infra/worker.py
Worker pool concerns per spec section 9. Actual worker PROCESSES are
`rq worker homework_grading` (this project already has that
infrastructure -- see jobs.py / worker_tasks.py at the project root and
the README's "CRM review workflow" section for how to run them). This
module just provides the sizing helper spec section 9 calls for:

    concurrency ~= throughput * average_latency   (Little's Law)

...so you know how many `rq worker` processes to run for a target
throughput, given your observed average grading latency.
"""

import math


def recommended_worker_count(
    target_homeworks_per_hour: int,
    average_latency_seconds: float,
    headroom_factor: float = 1.3,
) -> int:
    """
    Little's Law: concurrency ~= throughput * average latency. Example
    (matching spec section 9's own example): 10,000 homework/hour with
    ~15s average latency needs concurrency ~= (10000/3600) * 15 ~= 42 ->
    round up to ~50 worker processes with the default 1.3x headroom.
    """
    throughput_per_second = target_homeworks_per_hour / 3600.0
    base_concurrency = throughput_per_second * average_latency_seconds
    return max(1, math.ceil(base_concurrency * headroom_factor))
