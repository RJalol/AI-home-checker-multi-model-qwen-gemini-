"""Production hardening: field limits, upload cap, org rate limit, regrade cap, audit trail, erase,
retention, migrations, metrics, production config check."""
import io
import os
import time
import unittest
from pathlib import Path
from unittest import mock

import config
import hardening

from .test_crm_integration import CrmIntegrationTest, _SKIP_REASON, INTERNAL_KEY


class HardeningUnitTests(unittest.TestCase):
    def test_upload_is_read_only_up_to_the_limit(self):
        self.assertEqual(hardening.read_upload_capped(io.BytesIO(b"abc"), max_bytes=3), b"abc")
        with self.assertRaises(hardening.RequestTooLarge):
            hardening.read_upload_capped(io.BytesIO(b"abcd"), max_bytes=3)

    def test_oversized_stream_is_not_read_to_the_end(self):
        class Endless(io.RawIOBase):
            reads = 0

            def read(self, n=-1):
                Endless.reads += 1
                return b"x" * (n if n > 0 else 1024)

        with self.assertRaises(hardening.RequestTooLarge):
            hardening.read_upload_capped(Endless(), max_bytes=3 * 1024 * 1024)
        self.assertLessEqual(Endless.reads, 5)

    def test_field_limits(self):
        hardening.check_field_lengths(student_name="x" * 255, skill_key=None)
        for name, size in (("student_name", 256), ("skill_key", 33), ("language_key", 33), ("subject_key", 65)):
            with self.assertRaises(hardening.FieldTooLong, msg=name):
                hardening.check_field_lengths(**{name: "x" * size})
        with self.assertRaises(hardening.FieldTooLong):
            hardening.check_field_lengths(extra_instructions="x" * (config.MAX_EXTRA_INSTRUCTIONS_CHARS + 1))

    def test_rate_limit_windows_and_zero_disables(self):
        class FakeRedis:
            def __init__(self):
                self.counts = {}

            def incr(self, key):
                self.counts[key] = self.counts.get(key, 0) + 1
                return self.counts[key]

            def expire(self, key, ttl):
                pass

        with mock.patch.object(config, "ORG_MAX_SUBMISSIONS_PER_MINUTE", 2), \
                mock.patch.object(config, "ORG_MAX_SUBMISSIONS_PER_DAY", 0):
            r = FakeRedis()
            hardening.check_org_rate("org", r, now=1000.0)
            hardening.check_org_rate("org", r, now=1001.0)
            with self.assertRaises(hardening.RateLimitExceeded) as ctx:
                hardening.check_org_rate("org", r, now=1002.0)
            self.assertGreaterEqual(ctx.exception.retry_after, 1)
            hardening.check_org_rate("org", r, now=1000.0 + 61)      # next minute: allowed again
            hardening.check_org_rate("other-org", r, now=1002.0)     # another tenant is unaffected
        with mock.patch.object(config, "ORG_MAX_SUBMISSIONS_PER_MINUTE", 0), \
                mock.patch.object(config, "ORG_MAX_SUBMISSIONS_PER_DAY", 0):
            for _ in range(5):
                hardening.check_org_rate("org", FakeRedis())

    def test_rate_limit_fails_open_when_redis_is_down(self):
        class Broken:
            def incr(self, key):
                raise ConnectionError("down")

        with mock.patch.object(config, "ORG_MAX_SUBMISSIONS_PER_MINUTE", 1):
            hardening.check_org_rate("org", Broken())   # no exception

    def test_production_config_checks(self):
        with mock.patch.multiple(config, INTERNAL_API_KEY="", ALLOW_UNAUTHENTICATED_API=True,
                                 WEBHOOK_SIGNING_SECRET="", WEBHOOK_ALLOW_PRIVATE_URLS=True):
            problems = config.production_config_problems()
        self.assertGreaterEqual(len(problems), 4)
        with mock.patch.multiple(config, INTERNAL_API_KEY="a" * 40, ALLOW_UNAUTHENTICATED_API=False,
                                 WEBHOOK_SIGNING_SECRET="b" * 40, WEBHOOK_ALLOW_PRIVATE_URLS=False,
                                 DATABASE_URL="postgresql+psycopg2://app:s3cret@db.internal/hw"):
            self.assertEqual(config.production_config_problems(), [])
        with mock.patch.multiple(config, INTERNAL_API_KEY="same" * 10, WEBHOOK_SIGNING_SECRET="same" * 10):
            self.assertTrue(any("differ" in p for p in config.production_config_problems()))

    def test_webhook_secret_is_not_derived_from_the_admin_key_in_production(self):
        import webhook_security
        with mock.patch.multiple(config, INTERNAL_API_KEY="admin-key", WEBHOOK_SIGNING_SECRET="", IS_PRODUCTION=False):
            self.assertIsNotNone(webhook_security.derive_org_secret("org"))
        with mock.patch.multiple(config, INTERNAL_API_KEY="admin-key", WEBHOOK_SIGNING_SECRET="", IS_PRODUCTION=True):
            self.assertIsNone(webhook_security.derive_org_secret("org"))


@unittest.skipIf(_SKIP_REASON, _SKIP_REASON)
class HardeningApiTests(CrmIntegrationTest):
    def uploads(self):
        return sorted(p.name for p in (Path(self.tmp.name) / "uploads").iterdir())

    # -- input limits ------------------------------------------------------------------------
    def test_over_long_fields_are_422_not_500(self):
        org = self.new_org()
        for field, value in (("student_name", "x" * 300), ("skill_key", "z" * 100),
                             ("extra_instructions", "y" * (config.MAX_EXTRA_INSTRUCTIONS_CHARS + 1))):
            response = self.submit(org, f"f-{field}", **{field: value})
            self.assertEqual(response.status_code, 422, (field, response.text[:200]))
        self.assertEqual(self.submit(org, "e" * 300).status_code, 422)
        self.assertEqual(self.uploads(), [], "a refused request must not leave a file behind")

    def test_file_over_the_limit_is_413(self):
        org = self.new_org()
        with mock.patch.object(config, "MAX_FILE_SIZE_MB", 0.0001):
            response = self.submit(org, "big", data=b"x" * 5000, filename="a.pdf")
        self.assertEqual(response.status_code, 413, response.text[:200])

    def test_image_with_too_many_pixels_is_a_400(self):
        org = self.new_org()
        with mock.patch("utils.MAX_IMAGE_PIXELS", 100):
            response = self.submit(org, "pix")
        self.assertEqual(response.status_code, 400, response.text[:200])
        self.assertIn("too large", response.text)

    # -- abuse protection --------------------------------------------------------------------
    def test_org_rate_limit_returns_429_and_is_per_tenant(self):
        a, b = self.new_org("A"), self.new_org("B")
        with mock.patch.object(config, "ORG_MAX_SUBMISSIONS_PER_MINUTE", 2):
            self.assertEqual(self.submit(a, "1").status_code, 200)
            self.assertEqual(self.submit(a, "2").status_code, 200)
            third = self.submit(a, "3")
            self.assertEqual(third.status_code, 429)
            self.assertIn("retry-after", {k.lower() for k in third.headers})
            self.assertEqual(self.submit(b, "1").status_code, 200)

    def test_regrade_is_capped(self):
        org = self.new_org()
        sid = self.graded_submission(org)
        with mock.patch.object(self.db, "MAX_REGRADES_PER_SUBMISSION", 2):
            for _ in range(2):
                self.assertEqual(self.client.post(f"/submissions/{sid}/regrade", headers=org["headers"]).status_code, 200)
                self.drain()
                self.assertEqual(self.get(org, sid)["status"], "pending_review")
            third = self.client.post(f"/submissions/{sid}/regrade", headers=org["headers"])
            self.assertEqual(third.status_code, 409)
            self.assertIn("already regraded", third.text)

    # -- no orphan files, audit trail, erase --------------------------------------------------
    def test_a_client_retry_does_not_write_another_file(self):
        org = self.new_org()
        self.submit(org, "same")
        self.assertEqual(len(self.uploads()), 1)
        for _ in range(3):
            self.assertEqual(self.submit(org, "same").status_code, 200)
        self.assertEqual(len(self.uploads()), 1)

    def test_review_is_recorded_in_the_audit_trail(self):
        org = self.new_org()
        sid = self.graded_submission(org)
        edited = {"overall_score": 55, "questions": [{"question_number": 1, "score": 55}]}
        response = self.review(org, sid, action="edit", corrected_result=edited, reviewer_note="fixed",
                               reviewer_external_id="teacher-9")
        self.assertEqual(response.status_code, 200, response.text)
        with self.db.session_scope() as session:
            events = session.query(self.db.ReviewEvent).filter_by(submission_id=sid).all()
            self.assertEqual([e.event for e in events], ["edited"])
            self.assertEqual(events[0].actor, "teacher-9")
            self.assertEqual(events[0].score_after, 55)
            self.assertEqual(events[0].note, "fixed")

    def test_erase_removes_row_file_and_keeps_an_anonymous_event(self):
        org = self.new_org()
        sid = self.graded_submission(org)
        self.assertEqual(len(self.uploads()), 1)
        response = self.client.delete(f"/submissions/{sid}?reviewer_external_id=admin-1", headers=org["headers"])
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(self.uploads(), [])
        self.assertEqual(self.client.get(f"/submissions/{sid}", headers=org["headers"]).status_code, 404)
        self.assertEqual(self.client.delete(f"/submissions/{sid}", headers=org["headers"]).status_code, 404)
        with self.db.session_scope() as session:
            events = session.query(self.db.ReviewEvent).filter_by(submission_id=sid).all()
            self.assertEqual([(e.event, e.actor) for e in events], [("erased", "admin-1")])

    def test_erase_is_refused_while_grading_and_is_tenant_scoped(self):
        a, b = self.new_org("A"), self.new_org("B")
        pending = self.submit(a, "p").json()["id"]
        self.assertEqual(self.client.delete(f"/submissions/{pending}", headers=a["headers"]).status_code, 409)
        self.drain()
        self.assertEqual(self.client.delete(f"/submissions/{pending}", headers=b["headers"]).status_code, 404)
        self.assertEqual(self.client.get(f"/submissions/{pending}", headers=a["headers"]).status_code, 200)

    def test_failure_note_does_not_leak_the_provider_error(self):
        org = self.new_org()
        sid = self.submit(org, "leak").json()["id"]
        self.db.mark_submission_failed(sid, "APIError: 401 https://internal.example/v1 key sk-secret-123")
        note = self.get(org, sid)["reviewer_note"]
        self.assertTrue(note.startswith("Grading failed"))
        self.assertNotIn("sk-secret", note)
        self.assertNotIn("internal.example", note)

    # -- retention ---------------------------------------------------------------------------
    def test_retention_deletes_old_approved_files_and_orphans_only(self):
        import retention
        org = self.new_org()
        old = self.graded_submission(org, "old")
        fresh = self.graded_submission(org, "fresh")
        self.assertEqual(self.review(org, old, action="approve").status_code, 200)
        self.assertEqual(self.review(org, fresh, action="approve").status_code, 200)
        from datetime import datetime, timedelta, timezone
        with self.db.session_scope() as session:
            session.get(self.db.Submission, old).approved_at = datetime.now(timezone.utc) - timedelta(days=100)
        uploads = Path(self.tmp.name) / "uploads"
        orphan_new, orphan_old = uploads / "new_orphan.jpg", uploads / "old_orphan.jpg"
        orphan_new.write_bytes(b"x")
        orphan_old.write_bytes(b"x")
        os.utime(orphan_old, (time.time() - 2 * 86400,) * 2)
        self.assertEqual(len(self.uploads()), 4)
        with mock.patch.object(config, "UPLOAD_RETENTION_DAYS", 90):
            summary = retention.purge_once()
        self.assertEqual(summary["approved_files_deleted"], 1, summary)
        self.assertEqual(summary["orphan_files_deleted"], 1, summary)
        self.assertTrue(orphan_new.exists(), "a recent unreferenced file may belong to a request still in flight")
        self.assertFalse(orphan_old.exists())
        self.assertEqual(len(self.uploads()), 2)
        self.assertEqual(self.get(org, old)["status"], "approved", "the result stays; only the file goes")

    def test_retention_zero_keeps_everything(self):
        import retention
        with mock.patch.object(config, "UPLOAD_RETENTION_DAYS", 0):
            self.assertTrue(retention.purge_once()["skipped"])

    def test_retention_endpoint_needs_the_admin_key(self):
        self.assertEqual(self.client.post("/admin/retention").status_code, 401)
        self.assertEqual(self.client.post("/admin/retention", headers=self.internal).status_code, 200)

    # -- migrations / metrics ----------------------------------------------------------------
    def test_migrations_apply_to_an_old_schema_once(self):
        import migrations
        from sqlalchemy import text
        engine = self.db.get_engine()
        with engine.begin() as conn:
            conn.execute(text("ALTER TABLE submissions DROP COLUMN regrade_count"))
            conn.execute(text("DELETE FROM schema_migrations"))
        self.assertEqual(migrations.apply_migrations(engine), [1, 2])
        self.assertEqual(migrations.apply_migrations(engine), [])
        self.assertEqual(migrations.current_version(engine), 2)
        org = self.new_org()
        self.assertEqual(self.submit(org, "after-migration").status_code, 200)

    def test_metrics_endpoint(self):
        self.assertEqual(self.client.get("/metrics").status_code, 401)
        org = self.new_org()
        self.submit(org, "m1")
        response = self.client.get("/metrics", headers=self.internal)
        self.assertEqual(response.status_code, 200, response.text[:200])
        self.assertIn('hw_submissions{status="pending_ai"} 1', response.text)
        self.assertIn('hw_queue_length{queue="homework_grading"} 1', response.text)
        self.assertIn("hw_oldest_pending_ai_seconds", response.text)
        self.assertIn("hw_schema_version 2", response.text)
        self.assertIn('hw_ai_spend_usd{period="day"}', response.text)


if __name__ == "__main__":
    unittest.main()
