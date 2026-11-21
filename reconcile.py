"""
reconcile.py
Safety net for the grading queue: finds submissions that have been "pending_ai" for too long
although their RQ job is gone (worker killed by OOM/deploy, Redis flushed, enqueue failed after
the row was saved) and gets them graded, instead of leaving a student's work in limbo forever.

For every submission still pending_ai after config.RECONCILE_REQUEUE_AFTER_SECONDS:
  * its grading job is still queued/running  -> leave it alone (a long queue is not an error);
  * otherwise -> re-queue it, at most config.RECONCILE_MAX_REQUEUES times; after that mark it
    "grading_failed" with an explanation (visible to the teacher, re-gradable via
    POST /submissions/{id}/regrade).

Run it periodically (every minute or two is plenty):

    python reconcile.py                  # one pass, prints a JSON summary
    python reconcile.py --loop 60        # keep running, one pass every 60 seconds
    curl -X POST -H "X-Internal-Api-Key: $KEY" http://localhost:8000/admin/reconcile   # from cron

Safe to run from several places at once: re-queueing goes through jobs.enqueue_submission_grading,
which never creates a second job next to a live one.
"""

import argparse
import json
import logging
import time

import config
import db
import jobs

logger = logging.getLogger("reconcile")
ATTEMPT_KEY_TTL_SECONDS = 7 * 24 * 3600


def _attempt_key(submission_id: str) -> str:
    return f"hw:reconcile:attempts:{submission_id}"


def reconcile_once() -> dict:
    """One pass. Returns counters: checked / in_flight / requeued / failed / errors."""
    summary = {"checked": 0, "in_flight": 0, "requeued": 0, "failed": 0, "errors": 0}
    stuck = db.list_stuck_pending_ai(config.RECONCILE_REQUEUE_AFTER_SECONDS)
    conn = jobs.get_redis_connection()
    for submission_id, age_seconds in stuck:
        summary["checked"] += 1
        try:
            if jobs.grading_job_in_flight(submission_id):
                summary["in_flight"] += 1
                continue
            attempts = conn.incr(_attempt_key(submission_id))
            conn.expire(_attempt_key(submission_id), ATTEMPT_KEY_TTL_SECONDS)
            if attempts > config.RECONCILE_MAX_REQUEUES:
                db.mark_submission_failed(
                    submission_id,
                    f"not graded after {config.RECONCILE_MAX_REQUEUES} automatic re-queues "
                    f"(pending for {int(age_seconds)}s) -- use POST /submissions/{{id}}/regrade to try again",
                )
                summary["failed"] += 1
                logger.warning("submission %s marked grading_failed after %s re-queues", submission_id, attempts - 1)
                continue
            jobs.enqueue_submission_grading(submission_id, force=True)
            summary["requeued"] += 1
            logger.info("submission %s re-queued (attempt %s, pending %ss)", submission_id, attempts, int(age_seconds))
        except Exception:
            summary["errors"] += 1
            logger.exception("could not reconcile submission %s", submission_id)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-queue or fail submissions stuck in pending_ai.")
    parser.add_argument("--loop", type=int, default=0, metavar="SECONDS", help="repeat every SECONDS instead of running once")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    while True:
        print(json.dumps(reconcile_once()), flush=True)
        if args.loop <= 0:
            break
        time.sleep(args.loop)


if __name__ == "__main__":
    main()
