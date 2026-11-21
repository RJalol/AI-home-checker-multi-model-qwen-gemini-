"""
Integration tests for the CRM review workflow against a REAL PostgreSQL and a REAL Redis/RQ.
Only the LLM SDKs are faked (tests/fakes.py), so these prove queue / database / API behaviour --
never grading quality.

They are skipped unless both services are named explicitly (so a plain `pytest` on a laptop does
not need them, and can never touch a production database by accident):

    export TEST_DATABASE_URL="postgresql+psycopg2://postgres@localhost:5432/homework_checker_test"
    export TEST_REDIS_URL="redis://localhost:6379/15"
    python -m pytest tests/test_crm_integration.py

THE DATABASE NAMED THERE IS WIPED (tables dropped and recreated) AND THE REDIS DB IS FLUSHED: its
name must contain "test" or "audit" or the tests refuse to run.
"""

import http.server
import json
import os
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

TEST_DATABASE_URL = os.environ.get("TEST_DATABASE_URL", "")
TEST_REDIS_URL = os.environ.get("TEST_REDIS_URL", "")

import config  # noqa: E402

from .fakes import FakeSDKTestCase, executable_markup, make_png  # noqa: E402

_SKIP_REASON = ""
if not (TEST_DATABASE_URL and TEST_REDIS_URL):
    _SKIP_REASON = "set TEST_DATABASE_URL and TEST_REDIS_URL to run the PostgreSQL/Redis integration tests"
else:
    db_name = TEST_DATABASE_URL.rsplit("/", 1)[-1].split("?")[0].lower()
    if "test" not in db_name and "audit" not in db_name:
        raise RuntimeError(f"Refusing to wipe database {db_name!r}: its name must contain 'test' or 'audit'.")
    try:
        import psycopg2  # noqa: F401
        import redis as _redis_lib
        import sqlalchemy

        sqlalchemy.create_engine(TEST_DATABASE_URL).connect().close()
        _redis_lib.from_url(TEST_REDIS_URL).ping()
    except Exception as exc:  # services not reachable -> skip rather than fail
        _SKIP_REASON = f"PostgreSQL/Redis not reachable ({type(exc).__name__})"

_REAL_SLEEP = time.sleep   # FakeSDKTestCase turns time.sleep into a no-op; the deadline test needs a real wait

INTERNAL_KEY = "internal-test-key"
MASTER_SECRET = "test-master-secret"
ALL_PROVIDERS_DOWN = {
    "fail_dashscope-intl.aliyuncs.com": True, "fail_gemini": True, "fail_anthropic": True,
    "fail_api.openai.com": True, "fail_api.z.ai": True,
}


class _Receiver:
    """A tiny local HTTP server standing in for the CRM's webhook endpoint."""

    def __init__(self):
        self.received = []
        self.mode = "ok"            # "ok" | "redirect" | "error"
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                outer.received.append({"path": self.path, "headers": dict(self.headers), "raw": body})
                if outer.mode == "redirect" and self.path == "/crm/hook":
                    self.send_response(302)
                    self.send_header("Location", f"http://127.0.0.1:{outer.port}/internal-only")
                elif outer.mode == "error":
                    self.send_response(500)
                else:
                    self.send_response(200)
                self.end_headers()

            def log_message(self, *args):
                pass

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.server.server_address[1]
        self.url = f"http://127.0.0.1:{self.port}/crm/hook"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def hits(self, path):
        return [r for r in self.received if r["path"] == path]


@unittest.skipIf(_SKIP_REASON, _SKIP_REASON)
class CrmIntegrationTest(FakeSDKTestCase):
    """Base: wipes the test database and Redis before every test; wires module globals to them."""

    @classmethod
    def setUpClass(cls):
        import db
        import jobs

        cls._saved = (db.DATABASE_URL, db._engine, db._SessionLocal, jobs.REDIS_URL, jobs._redis_conn, jobs._queue, jobs._webhook_queue)
        db.DATABASE_URL = TEST_DATABASE_URL
        db._engine = db._SessionLocal = None
        jobs.REDIS_URL = TEST_REDIS_URL
        jobs._redis_conn = jobs._queue = jobs._webhook_queue = None
        cls.receiver = _Receiver()

    @classmethod
    def tearDownClass(cls):
        import db
        import jobs

        cls.receiver.server.shutdown()
        if db._engine is not None:
            db._engine.dispose()
        (db.DATABASE_URL, db._engine, db._SessionLocal, jobs.REDIS_URL, jobs._redis_conn, jobs._queue, jobs._webhook_queue) = cls._saved

    def setUp(self):
        super().setUp()
        import api
        import db
        import jobs
        import report_generator
        import training_logger
        import utils
        import worker_tasks
        from fastapi.testclient import TestClient

        self.api, self.db, self.jobs, self.worker_tasks = api, db, jobs, worker_tasks
        db.Base.metadata.drop_all(db.get_engine())
        db.init_db()
        self.redis = jobs.get_redis_connection()
        self.redis.flushdb()
        self.receiver.received.clear()
        self.receiver.mode = "ok"

        uploads = Path(self.tmp.name) / "uploads"
        reports = Path(self.tmp.name) / "reports"
        uploads.mkdir()
        reports.mkdir()
        self.raw_log = Path(self.tmp.name) / "raw_log.jsonl"
        self.gold_log = Path(self.tmp.name) / "gold.jsonl"
        for target, name, value in (
            (api, "INTERNAL_API_KEY", INTERNAL_KEY), (config, "INTERNAL_API_KEY", INTERNAL_KEY),
            (config, "ALLOW_UNAUTHENTICATED_API", False), (config, "WEBHOOK_SIGNING_SECRET", MASTER_SECRET),
            (config, "WEBHOOK_ALLOW_PRIVATE_URLS", False), (config, "COLLECT_TRAINING_DATA", False),
            (worker_tasks, "COLLECT_TRAINING_DATA", False), (config, "RECONCILE_REQUEUE_AFTER_SECONDS", 0),
            (config, "RECONCILE_MAX_REQUEUES", 3), (utils, "UPLOAD_DIR", uploads), (report_generator, "REPORTS_DIR", reports),
            (training_logger, "TRAINING_RAW_LOG", self.raw_log), (training_logger, "TRAINING_GOLD_LOG", self.gold_log),
            (training_logger, "TRAINING_INPUTS_DIR", Path(self.tmp.name) / "train_inputs"),
        ):
            patcher = mock.patch.object(target, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

        self.client = TestClient(api.app, raise_server_exceptions=False)
        self._counter = 0
        self.internal = {"X-Internal-Api-Key": INTERNAL_KEY}

    # -- helpers -----------------------------------------------------------------------------
    def png(self):
        """A valid photo whose bytes are different on every call (distinct result-cache key)."""
        self._counter += 1
        path = make_png(Path(self.tmp.name) / f"p{self._counter}.png", size=(60 + self._counter, 48))
        return path.read_bytes()

    def new_org(self, name="School A", webhook_url=None):
        body = {"name": name}
        if webhook_url:
            body["webhook_url"] = webhook_url
        response = self.client.post("/organizations", json=body, headers=self.internal)
        self.assertEqual(response.status_code, 200, response.text)
        org = response.json()
        org["headers"] = {"X-Organization-Api-Key": org["api_key"]}
        return org

    def submit(self, org, external_id="ext-1", data=None, filename="hw.png", **extra):
        form = {"external_submission_id": external_id, "student_external_id": "stu-1", "student_name": "Ali",
                "teacher_external_id": "t-1", "subject_key": "math"}
        form.update(extra)
        return self.client.post("/submissions", data=form, headers=org["headers"],
                                files={"file": (filename, data if data is not None else self.png(), "image/png")})

    def get(self, org, submission_id):
        return self.client.get(f"/submissions/{submission_id}", headers=org["headers"]).json()

    def review(self, org, submission_id, **body):
        return self.client.post(f"/submissions/{submission_id}/review", json=body, headers=org["headers"])

    def drain(self, queue_name="homework_grading"):
        from rq import Queue, SimpleWorker
        queue = Queue(queue_name, connection=self.redis)
        SimpleWorker([queue], connection=self.redis).work(burst=True, with_scheduler=False, logging_level="CRITICAL")

    def queued(self, queue_name="homework_grading"):
        from rq import Queue
        return Queue(queue_name, connection=self.redis).count

    def graded_submission(self, org, external_id="ext-1"):
        sid = self.submit(org, external_id).json()["id"]
        self.drain()
        self.assertEqual(self.get(org, sid)["status"], "pending_review")
        return sid


@unittest.skipIf(_SKIP_REASON, _SKIP_REASON)
class TestAuthAndOnboarding(CrmIntegrationTest):
    def test_iss001_fails_closed_without_a_configured_key(self):
        with mock.patch.object(self.api, "INTERNAL_API_KEY", ""):
            for method, path in (("post", "/organizations"), ("get", "/backend"), ("get", "/usage"), ("post", "/admin/reconcile"),
                                 ("post", "/reports/html")):
                response = getattr(self.client, method)(path, **({"json": {"name": "x"}} if method == "post" else {}))
                self.assertEqual(response.status_code, 503, f"{method.upper()} {path}")
            self.assertEqual(self.client.get("/health").status_code, 200)
            self.assertEqual(self.client.post("/organizations", json={"name": "x"}).status_code, 503)
            with mock.patch.object(config, "ALLOW_UNAUTHENTICATED_API", True):
                self.assertEqual(self.client.post("/organizations", json={"name": "Dev"}).status_code, 200)

    def test_wrong_and_missing_internal_key_are_rejected(self):
        self.assertEqual(self.client.post("/organizations", json={"name": "x"}).status_code, 401)
        self.assertEqual(self.client.post("/organizations", json={"name": "x"}, headers={"X-Internal-Api-Key": "nope"}).status_code, 401)
        self.assertEqual(self.client.post("/organizations", json={"name": "x"}, headers=self.internal).status_code, 200)

    def test_iss007_organization_input_is_validated(self):
        for body in ({"name": ""}, {"name": "   "}, {"name": "x" * 300},
                     {"name": "x", "webhook_url": "http://169.254.169.254/latest/meta-data/"},
                     {"name": "x", "webhook_url": "https://169.254.169.254/latest/meta-data/"},
                     {"name": "x", "webhook_url": "https://127.0.0.1/hook"},
                     {"name": "x", "webhook_url": "file:///etc/passwd"}, {"name": "x", "webhook_url": "not a url"}):
            response = self.client.post("/organizations", json=body, headers=self.internal)
            self.assertEqual(response.status_code, 422, f"{body} -> {response.status_code} {response.text[:100]}")

    def test_org_api_key_is_stored_hashed_and_secret_is_per_org(self):
        from sqlalchemy import text
        a, b = self.new_org("A"), self.new_org("B")
        with self.db.session_scope() as session:
            stored = session.execute(text("select api_key_hash from organizations where id=:i"), {"i": a["organization_id"]}).scalar()
        self.assertNotEqual(stored, a["api_key"])
        self.assertEqual(len(stored), 64)
        self.assertTrue(a["webhook_secret"] and a["webhook_secret"] != b["webhook_secret"])
        again = self.client.get("/organizations/me/webhook-secret", headers=a["headers"]).json()
        self.assertEqual(again["webhook_secret"], a["webhook_secret"])

    def test_readyz_reports_dependencies(self):
        self.assertEqual(self.client.get("/readyz").status_code, 200)
        with mock.patch.object(self.jobs, "get_redis_connection", side_effect=ConnectionError("down")):
            response = self.client.get("/readyz")
        self.assertEqual(response.status_code, 503)
        self.assertIn("redis", json.dumps(response.json()))

    def test_jobs_endpoint_refuses_a_caller_supplied_provider_key(self):
        response = self.client.post("/jobs", data={"api_key": "sk-live-secret", "subject_key": "math"}, headers=self.internal,
                                    files={"file": ("hw.png", self.png(), "image/png")})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(self.queued(), 0, "nothing may be queued")
        for key in self.redis.scan_iter("*"):
            self.assertNotIn(b"sk-live-secret", self.redis.dump(key) or b"")

    def test_paging_parameters_are_validated(self):
        org = self.new_org()
        for query in ("limit=-1", "offset=-5", "since=yesterday"):
            response = self.client.get(f"/submissions?{query}", headers=org["headers"])
            self.assertIn(response.status_code, (400, 422), query)
        self.assertEqual(self.client.get("/submissions?limit=99999999", headers=org["headers"]).status_code, 200)


@unittest.skipIf(_SKIP_REASON, _SKIP_REASON)
class TestSubmissionWorkflow(CrmIntegrationTest):
    def test_happy_path_submit_grade_review(self):
        org = self.new_org()
        response = self.submit(org)
        self.assertEqual(response.status_code, 200, response.text)
        submission = response.json()
        self.assertEqual(submission["status"], "pending_ai")
        self.assertNotIn("input_file_path", submission, "server filesystem paths must not leak to the CRM")
        self.assertEqual(self.queued(), 1)

        self.drain()
        graded = self.get(org, submission["id"])
        self.assertEqual(graded["status"], "pending_review")
        self.assertIsInstance(graded["ai_result"]["questions"], list)
        self.assertIn("overall_score", graded["ai_result"])
        self.assertIn("needs_attention", graded["ai_result"])
        self.assertNotIn("input_file_path", graded)

        approved = self.review(org, submission["id"], action="approve", reviewer_note="ok")
        self.assertEqual(approved.status_code, 200, approved.text)
        self.assertEqual(approved.json()["status"], "approved")

    def test_iss002_resubmitting_while_in_flight_does_not_queue_a_second_paid_job(self):
        org = self.new_org()
        first = self.submit(org, "same-id").json()
        data = self.png()
        for _ in range(3):
            again = self.submit(org, "same-id", data=data).json()
            self.assertEqual(again["id"], first["id"])
        self.assertEqual(self.queued(), 1)
        self.backend.calls.clear()
        self.drain()
        self.assertEqual(self.get(org, first["id"])["status"], "pending_review")
        graded_calls = len(self.backend.grading())
        self.assertGreaterEqual(graded_calls, 1)
        again = self.submit(org, "same-id", data=data).json()
        self.assertEqual(again["id"], first["id"])
        self.assertEqual(again["status"], "pending_review")
        self.assertEqual(self.queued(), 0, "a graded submission is not re-queued by a client retry")

    def test_assignment_task_and_key_are_shared_stored_once_and_not_echoed(self):
        org = self.new_org()
        extra = {"external_assignment_id": "hw-7", "task_text": "Solve x+1=3", "answer_key": "SECRET-KEY x=2"}
        a = self.submit(org, "s-a", student_external_id="stu-a", **extra)
        # the second student only names the assignment; the stored task/key are reused
        b = self.submit(org, "s-b", student_external_id="stu-b", external_assignment_id="hw-7")
        self.assertEqual(a.status_code, 200, a.text)
        self.assertEqual(b.status_code, 200, b.text)
        self.assertEqual(a.json()["assignment_id"], b.json()["assignment_id"])
        self.assertNotIn("SECRET-KEY", a.text, "the answer key must not be returned by the API")
        self.backend.calls.clear()
        self.drain()
        prompts_sent = [c["user"] for c in self.backend.grading() if c["host"] in ("gemini", "dashscope-intl.aliyuncs.com", "anthropic")]
        self.assertTrue(prompts_sent)
        self.assertTrue(all("SECRET-KEY x=2" in u and "Solve x+1=3" in u for u in prompts_sent), prompts_sent)

    def test_assignment_text_needs_an_assignment_id_and_is_length_limited(self):
        org = self.new_org()
        self.assertEqual(self.submit(org, "e1", task_text="x").status_code, 400)
        too_long = "x" * (self.api.MAX_ASSIGNMENT_TEXT_CHARS + 1)
        self.assertEqual(self.submit(org, "e2", external_assignment_id="hw", answer_key=too_long).status_code, 400)

    def test_iss004_a_late_or_duplicate_job_never_touches_a_reviewed_result(self):
        org = self.new_org()
        sid = self.graded_submission(org)
        self.review(org, sid, action="approve")
        before = self.get(org, sid)
        self.backend.calls.clear()
        self.worker_tasks.grade_submission_task(sid)  # what a duplicate job would run
        self.assertEqual(self.backend.grading(), [], "no paid call for a submission that is no longer pending_ai")
        self.assertEqual(self.get(org, sid), before)
        self.assertFalse(self.db.mark_submission_graded(sid, {"overall_score": 1, "questions": []}))
        self.assertFalse(self.db.mark_submission_failed(sid, "late failure"))
        self.assertEqual(self.get(org, sid)["status"], "approved")

    def test_iss004_review_state_machine(self):
        org = self.new_org()
        pending = self.submit(org, "p").json()["id"]
        self.assertEqual(self.review(org, pending, action="approve").status_code, 409, "AI has not graded it yet")
        sid = self.graded_submission(org, "g")
        self.assertEqual(self.review(org, sid, action="edit").status_code, 400)
        self.assertEqual(self.review(org, sid, action="delete").status_code, 400)
        self.assertEqual(self.review(org, sid, action="approve").status_code, 200)
        final = self.get(org, sid)["final_result"]
        again = self.review(org, sid, action="edit", corrected_result={"overall_score": 1, "questions": []})
        self.assertEqual(again.status_code, 409)
        self.assertEqual(self.get(org, sid)["final_result"], final, "an approved result is final")

    def test_iss004_concurrent_approvals_exactly_one_wins(self):
        org = self.new_org()
        sid = self.graded_submission(org)
        outcomes = []
        barrier = threading.Barrier(2)

        def approve(label):
            barrier.wait()
            try:
                self.db.approve_submission(sid, organization_id=org["organization_id"], corrected_result=None, reviewer_note=label)
                outcomes.append("ok")
            except self.db.InvalidTransition:
                outcomes.append("conflict")

        threads = [threading.Thread(target=approve, args=(n,)) for n in ("a", "b")]
        [t.start() for t in threads]
        [t.join() for t in threads]
        self.assertEqual(sorted(outcomes), ["conflict", "ok"])

    def test_iss005_invalid_teacher_edit_is_rejected_and_nothing_is_stored(self):
        org = self.new_org()
        sid = self.graded_submission(org)
        before = self.get(org, sid)
        for bad in ({"hello": "world"}, {"overall_score": "high", "questions": []}, {"overall_score": 250, "questions": []},
                    {"overall_score": 50, "questions": "none"}, {"overall_score": 50, "questions": [{"score": -3}]}):
            response = self.review(org, sid, action="edit", corrected_result=bad)
            self.assertEqual(response.status_code, 422, f"{bad} -> {response.status_code}")
        self.assertEqual(self.get(org, sid), before)
        good = {"overall_score": 70, "grade": "Good", "questions": [{"question_number": 1, "score": 70, "analysis": "fine"}], "summary": {}}
        response = self.review(org, sid, action="edit", corrected_result=good)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(self.get(org, sid)["final_result"]["overall_score"], 70)

    def test_tenants_cannot_see_each_other(self):
        a, b = self.new_org("A"), self.new_org("B")
        sid = self.graded_submission(a)
        self.assertEqual(self.client.get(f"/submissions/{sid}", headers=b["headers"]).status_code, 404)
        self.assertEqual(self.client.get("/submissions", headers=b["headers"]).json()["count"], 0)
        self.assertEqual(self.review(b, sid, action="approve").status_code, 404)
        self.assertEqual(self.client.post(f"/submissions/{sid}/regrade", headers=b["headers"]).status_code, 404)
        self.assertEqual(self.client.get("/submissions", headers={"X-Organization-Api-Key": "wrong"}).status_code, 401)
        self.assertEqual(self.client.get("/submissions").status_code, 401)
        other = self.submit(b, "ext-1")  # same external id, another tenant
        self.assertEqual(other.status_code, 200)
        self.assertNotEqual(other.json()["id"], sid)

    def test_iss003_nothing_is_written_to_the_training_set_by_default(self):
        org = self.new_org()
        sid = self.graded_submission(org)
        self.review(org, sid, action="approve")
        self.assertFalse(self.raw_log.exists())
        self.assertFalse(self.gold_log.exists())
        self.assertFalse((Path(self.tmp.name) / "train_inputs").exists())

    def test_upload_content_is_checked_not_just_the_file_name(self):
        org = self.new_org()
        fake_image = self.client.post("/submissions", data={"subject_key": "math"}, headers=org["headers"],
                                      files={"file": ("hw.png", b"<html><script>alert(1)</script></html>", "image/png")})
        self.assertEqual(fake_image.status_code, 400)
        wrong_type = self.client.post("/submissions", data={"subject_key": "math"}, headers=org["headers"],
                                      files={"file": ("virus.exe", b"MZ\x90\x00", "application/octet-stream")})
        self.assertEqual(wrong_type.status_code, 400)
        self.assertEqual(self.queued(), 0)


@unittest.skipIf(_SKIP_REASON, _SKIP_REASON)
class TestRecovery(CrmIntegrationTest):
    """ISS-002: a submission can no longer be lost or stuck without anyone noticing."""

    def test_enqueue_failure_answers_503_keeps_the_row_and_a_retry_recovers_it(self):
        org = self.new_org()
        data = self.png()
        with mock.patch.object(self.jobs, "enqueue_submission_grading", side_effect=ConnectionError("redis down")):
            response = self.submit(org, "ext-lost", data=data)
        self.assertEqual(response.status_code, 503)
        sid = response.json()["detail"]["submission_id"]
        self.assertEqual(self.get(org, sid)["status"], "pending_ai")
        retry = self.submit(org, "ext-lost", data=data)
        self.assertEqual(retry.status_code, 200)
        self.assertEqual(retry.json()["id"], sid)
        self.drain()
        self.assertEqual(self.get(org, sid)["status"], "pending_review")

    def test_reconciler_requeues_a_submission_whose_job_was_lost(self):
        from reconcile import reconcile_once
        org = self.new_org()
        sid = self.submit(org).json()["id"]
        self.redis.flushdb()  # the queue is gone (Redis restart / eviction / worker killed)
        self.assertEqual(self.queued(), 0)
        summary = reconcile_once()
        self.assertEqual((summary["checked"], summary["requeued"], summary["errors"]), (1, 1, 0), summary)
        self.assertEqual(self.queued(), 1)
        self.drain()
        self.assertEqual(self.get(org, sid)["status"], "pending_review")

    def test_reconciler_leaves_a_live_job_alone(self):
        from reconcile import reconcile_once
        org = self.new_org()
        self.submit(org)
        summary = reconcile_once()
        self.assertEqual((summary["in_flight"], summary["requeued"]), (1, 0), summary)
        self.assertEqual(self.queued(), 1, "no second job next to the live one")

    def test_reconciler_gives_up_visibly_after_the_limit(self):
        from reconcile import reconcile_once
        org = self.new_org()
        sid = self.submit(org).json()["id"]
        with mock.patch.object(config, "RECONCILE_MAX_REQUEUES", 2):
            results = []
            for _ in range(3):
                self.redis.delete("rq:queue:homework_grading", f"rq:job:grade-{sid}")
                results.append(reconcile_once())
        self.assertEqual([r["requeued"] for r in results], [1, 1, 0], results)
        self.assertEqual(results[2]["failed"], 1)
        failed = self.get(org, sid)
        self.assertEqual(failed["status"], "grading_failed")
        self.assertIn("not graded", failed["reviewer_note"])

    def hang_every_model_call(self, seconds):
        from . import fakes
        calls = []
        original = fakes._fail_if_scripted

        def hang(backend, host):
            calls.append(host)
            _REAL_SLEEP(seconds)

        fakes._fail_if_scripted = hang
        self.addCleanup(setattr, fakes, "_fail_if_scripted", original)
        return calls

    def test_iss008_a_hanging_provider_is_stopped_at_the_job_deadline(self):
        org = self.new_org()
        calls = self.hang_every_model_call(8)
        with mock.patch.object(self.jobs, "JOB_TIMEOUT_SECONDS", 2):
            sid = self.submit(org).json()["id"]
            started = time.monotonic()
            self.drain()
        elapsed = time.monotonic() - started
        self.assertLess(elapsed, 6, f"the job ran {elapsed:.1f}s although its budget was 2s")
        self.assertEqual(len(calls), 1, f"no retry / fallback provider after the deadline: {calls}")
        failed = self.get(org, sid)
        self.assertEqual(failed["status"], "grading_failed")
        self.assertIn("timeout", failed["reviewer_note"].lower())

    def test_iss008_the_same_holds_for_the_original_engine(self):
        import homework_checker
        # subject "general" is not served by the tiered router: it goes to the original Gemini engine
        for name, value in (("GEMINI_API_KEY", "test-key"), ("ENABLE_CONTEXT_CACHING", False)):
            patcher = mock.patch.object(homework_checker, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        org = self.new_org()
        calls = self.hang_every_model_call(8)
        with mock.patch.object(self.jobs, "JOB_TIMEOUT_SECONDS", 2):
            sid = self.submit(org, subject_key="general", custom_subject_name="Drawing").json()["id"]
            started = time.monotonic()
            self.drain()
        elapsed = time.monotonic() - started
        self.assertLess(elapsed, 6, f"the job ran {elapsed:.1f}s although its budget was 2s")
        self.assertEqual(len(calls), 1, calls)
        self.assertEqual(self.get(org, sid)["status"], "grading_failed")

    def test_admin_reconcile_endpoint_requires_the_internal_key(self):
        self.assertEqual(self.client.post("/admin/reconcile").status_code, 401)
        response = self.client.post("/admin/reconcile", headers=self.internal)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["checked"], 0)

    def test_provider_outage_ends_in_grading_failed_and_regrade_recovers(self):
        from hwchecker.infra import circuit_breaker
        org = self.new_org()
        self.backend.cfg.update(ALL_PROVIDERS_DOWN)
        sid = self.submit(org).json()["id"]
        self.drain()
        failed = self.get(org, sid)
        self.assertEqual(failed["status"], "grading_failed")
        self.assertTrue(failed["reviewer_note"].startswith("Grading failed"))

        for key in ALL_PROVIDERS_DOWN:
            self.backend.cfg.pop(key)
        circuit_breaker._memory_state.clear()  # the cool-down has passed
        response = self.client.post(f"/submissions/{sid}/regrade", headers=org["headers"])
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["status"], "pending_ai")
        self.assertIsNone(response.json()["reviewer_note"], "the stale failure message is cleared")
        self.drain()
        self.assertEqual(self.get(org, sid)["status"], "pending_review")

    def test_regrade_really_calls_the_models_again_and_approved_work_is_final(self):
        org = self.new_org()
        sid = self.graded_submission(org)
        first_calls = len(self.backend.grading())
        self.assertGreaterEqual(first_calls, 1)
        self.assertEqual(self.client.post(f"/submissions/{sid}/regrade", headers=org["headers"]).status_code, 200)
        self.drain()
        self.assertEqual(self.get(org, sid)["status"], "pending_review")
        self.assertGreater(len(self.backend.grading()), first_calls, "regrade must bypass the cached verdict")
        self.review(org, sid, action="approve")
        self.assertEqual(self.client.post(f"/submissions/{sid}/regrade", headers=org["headers"]).status_code, 409)
        self.assertEqual(self.client.post("/submissions/does-not-exist/regrade", headers=org["headers"]).status_code, 404)


@unittest.skipIf(_SKIP_REASON, _SKIP_REASON)
class TestWebhook(CrmIntegrationTest):
    """ISS-007: signed, SSRF-safe, no redirects, only for approved results."""

    def setUp(self):
        super().setUp()
        patcher = mock.patch.object(config, "WEBHOOK_ALLOW_PRIVATE_URLS", True)  # the receiver listens on 127.0.0.1
        patcher.start()
        self.addCleanup(patcher.stop)

    def approved(self, org):
        sid = self.graded_submission(org)
        self.assertEqual(self.review(org, sid, action="approve").status_code, 200)
        return sid

    def test_delivered_once_with_a_signature_the_crm_can_verify(self):
        org = self.new_org(webhook_url=self.receiver.url)
        sid = self.approved(org)
        self.drain("webhook_delivery")
        hits = self.receiver.hits("/crm/hook")
        self.assertEqual(len(hits), 1)
        headers = {k.lower(): v for k, v in hits[0]["headers"].items()}
        from webhook_security import sign
        expected = sign(org["webhook_secret"], headers["x-webhook-timestamp"], hits[0]["raw"])
        self.assertEqual(headers["x-webhook-signature"], expected)
        self.assertLess(abs(time.time() - int(headers["x-webhook-timestamp"])), 120)
        body = json.loads(hits[0]["raw"])
        self.assertEqual((body["submission_id"], body["status"]), (sid, "approved"))

        tampered = sign(org["webhook_secret"], headers["x-webhook-timestamp"], hits[0]["raw"] + b" ")
        self.assertNotEqual(headers["x-webhook-signature"], tampered)

    def test_not_sent_for_a_result_that_is_not_approved(self):
        org = self.new_org(webhook_url=self.receiver.url)
        sid = self.graded_submission(org)  # pending_review
        self.worker_tasks.deliver_webhook_task(sid)
        self.assertEqual(self.receiver.received, [])

    def test_private_destination_is_blocked_at_delivery_even_if_stored(self):
        org = self.new_org()
        with mock.patch.object(config, "WEBHOOK_ALLOW_PRIVATE_URLS", False):
            with self.db.session_scope() as session:
                session.execute(__import__("sqlalchemy").text("update organizations set webhook_url=:u where id=:i"),
                                {"u": self.receiver.url.replace("http://", "https://"), "i": org["organization_id"]})
            sid = self.graded_submission(org)
            self.review(org, sid, action="approve")
            with self.assertRaises(Exception):
                self.worker_tasks.deliver_webhook_task(sid)
        self.assertEqual(self.receiver.received, [], "the request must not even be attempted")

    def test_redirects_are_not_followed_and_count_as_failed(self):
        org = self.new_org(webhook_url=self.receiver.url)
        self.receiver.mode = "redirect"
        sid = self.approved(org)
        self.drain("webhook_delivery")
        self.assertEqual(len(self.receiver.hits("/crm/hook")), 1)
        self.assertEqual(self.receiver.hits("/internal-only"), [], "the signed request must not be bounced elsewhere")
        from sqlalchemy import text
        with self.db.session_scope() as session:
            attempts, delivered = session.execute(text("select webhook_attempts, webhook_delivered from submissions where id=:i"), {"i": sid}).first()
        self.assertEqual((attempts, bool(delivered)), (1, False), "a 3xx answer is a failed delivery")

    def test_a_failing_endpoint_is_retried_by_rq_not_dropped(self):
        org = self.new_org(webhook_url=self.receiver.url)
        self.receiver.mode = "error"
        sid = self.approved(org)
        self.drain("webhook_delivery")
        self.assertEqual(len(self.receiver.hits("/crm/hook")), 1)
        from rq import Queue
        scheduled = Queue("webhook_delivery", connection=self.redis).scheduled_job_registry.count
        self.assertEqual(scheduled, 1, "the retry is only *scheduled*: it needs `rq worker --with-scheduler` to ever run")
        from sqlalchemy import text
        with self.db.session_scope() as session:
            attempts, delivered = session.execute(text("select webhook_attempts, webhook_delivered from submissions where id=:i"), {"i": sid}).first()
        self.assertEqual((attempts, bool(delivered)), (1, False), "one failed attempt is recorded once")


@unittest.skipIf(_SKIP_REASON, _SKIP_REASON)
class TestReports(CrmIntegrationTest):
    def hostile_result(self):
        payload = '<script>alert(1)</script><img src=x onerror=alert(1)>'
        return {"overall_score": 50, "grade": "ok", "student_name": payload, "questions": [{
            "question_number": 1, "score": 50, "question": payload, "student_answer": payload, "analysis": payload,
            "mistakes": [payload], "suggestions": [payload]}], "summary": {"strengths": [payload]}}

    def test_iss006_html_report_has_no_script_and_is_served_with_a_locked_down_policy(self):
        response = self.client.post("/reports/html", json=self.hostile_result(), headers=self.internal)
        self.assertEqual(response.status_code, 200, response.text[:200])
        self.assertIn("default-src 'none'", response.headers["content-security-policy"])
        self.assertEqual(response.headers["x-content-type-options"], "nosniff")
        self.assertEqual(executable_markup(response.text), [])
        self.assertIn("&lt;script&gt;", response.text, "the hostile student name is shown as inert text, not dropped silently")

    def test_report_input_must_have_the_grading_shape(self):
        for bad in ({"hello": "world"}, {"overall_score": "x", "questions": []}, {"overall_score": 50, "questions": ["a"]},
                    {"overall_score": 50, "questions": [{"score": 900}]}, {"overall_score": 50, "student_name": "x" * 500, "questions": []}):
            self.assertEqual(self.client.post("/reports/html", json=bad, headers=self.internal).status_code, 422, bad)
        self.assertEqual(self.client.post("/reports/html", json=self.hostile_result()).status_code, 401)

    def test_report_file_names_cannot_escape_the_reports_directory(self):
        result = self.hostile_result()
        result["student_name"] = "../../../../tmp/evil"
        response = self.client.post("/reports/html", json=result, headers=self.internal)
        self.assertEqual(response.status_code, 200)
        reports_dir = Path(self.tmp.name) / "reports"
        written = list(reports_dir.glob("*"))
        self.assertEqual(len(written), 1, written)
        self.assertEqual(written[0].parent, reports_dir)


if __name__ == "__main__":
    unittest.main()
