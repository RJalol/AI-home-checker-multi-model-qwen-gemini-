"""
retention.py
Deletes student files that are no longer needed, so homework photos (often of minors) are not kept
forever:

  * the input file of a submission APPROVED more than config.UPLOAD_RETENTION_DAYS days ago (the
    teacher's decision is final; regrading an approved submission is not allowed anyway);
  * ORPHAN files in the uploads directory that no submission refers to and that are older than a
    day (left behind by a failed request).

Run it once a day:

    python retention.py                 # one pass, prints a JSON summary
    curl -X POST -H "X-Internal-Api-Key: $KEY" http://localhost:8000/admin/retention

UPLOAD_RETENTION_DAYS=0 keeps everything (nothing is deleted). The grading result stays in the
database; only the original file goes. training_data/ (opt-in fine-tuning logs) is NOT touched here.
"""

import json
import logging
import time
from pathlib import Path

import config
import db
import utils

logger = logging.getLogger("retention")
ORPHAN_MIN_AGE_SECONDS = 24 * 3600


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def purge_once(now: float = None) -> dict:
    summary = {"approved_files_deleted": 0, "orphan_files_deleted": 0, "errors": 0, "skipped": False}
    days = config.UPLOAD_RETENTION_DAYS
    if days <= 0:
        summary["skipped"] = True
        return summary
    now = time.time() if now is None else now
    upload_dir = Path(utils.UPLOAD_DIR)   # the directory uploads are actually saved to

    for submission_id, file_path in db.list_files_to_purge(days):
        try:
            path = Path(file_path)
            if _inside(path, upload_dir):   # never delete something outside the uploads directory
                path.unlink(missing_ok=True)
            db.clear_input_file(submission_id)
            summary["approved_files_deleted"] += 1
        except Exception:
            summary["errors"] += 1
            logger.exception("could not delete the file of submission %s", submission_id)

    if upload_dir.is_dir():
        referenced = {str(Path(p).resolve()) for p in db.all_referenced_file_paths()}
        for path in upload_dir.iterdir():
            try:
                if not path.is_file() or str(path.resolve()) in referenced:
                    continue
                if now - path.stat().st_mtime < ORPHAN_MIN_AGE_SECONDS:
                    continue   # may belong to a request that is still being saved
                path.unlink()
                summary["orphan_files_deleted"] += 1
            except Exception:
                summary["errors"] += 1
                logger.exception("could not delete orphan file %s", path)
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(json.dumps(purge_once()))
