"""The approved-result webhook body: unchanged original fields + the gradebook fields (final_score ...)."""

import unittest

import worker_tasks


def submission(**overrides):
    data = {
        "id": "sub-1", "external_submission_id": "crm-hw-001", "status": "approved",
        "ai_result": {"overall_score": 62.5, "questions": []},
        "reviewed_result": None, "reviewer_note": None,
        "reviewed_at": "2026-11-20T10:00:00+00:00", "approved_at": "2026-11-20T10:00:01+00:00",
    }
    data.update(overrides)
    data["final_result"] = data["reviewed_result"] or data["ai_result"]
    return data


class WebhookPayload(unittest.TestCase):
    def test_original_fields_are_unchanged(self):
        payload = worker_tasks.build_webhook_payload(submission())
        self.assertEqual(payload["submission_id"], "sub-1")
        self.assertEqual(payload["external_submission_id"], "crm-hw-001")
        self.assertEqual(payload["status"], "approved")
        self.assertEqual(payload["result"], {"overall_score": 62.5, "questions": []})
        self.assertEqual(payload["approved_at"], "2026-11-20T10:00:01+00:00")

    def test_approved_as_is_carries_the_ai_score_as_final(self):
        payload = worker_tasks.build_webhook_payload(submission())
        self.assertEqual(payload["final_score"], 62.5)
        self.assertEqual(payload["ai_score"], 62.5)
        self.assertFalse(payload["edited_by_teacher"])
        self.assertIsNone(payload["reviewer_note"])

    def test_an_edit_changes_final_score_but_keeps_the_ai_score(self):
        payload = worker_tasks.build_webhook_payload(submission(
            reviewed_result={"overall_score": 80, "questions": []}, reviewer_note="Partial credit for correct method."))
        self.assertEqual(payload["final_score"], 80)
        self.assertEqual(payload["ai_score"], 62.5)
        self.assertTrue(payload["edited_by_teacher"])
        self.assertEqual(payload["reviewer_note"], "Partial credit for correct method.")
        self.assertEqual(payload["result"]["overall_score"], 80)

    def test_a_teacher_edit_to_zero_is_still_an_edit_and_a_zero_score(self):
        payload = worker_tasks.build_webhook_payload(submission(reviewed_result={"overall_score": 0, "questions": []}))
        self.assertEqual(payload["final_score"], 0)
        self.assertTrue(payload["edited_by_teacher"])

    def test_a_missing_or_non_numeric_score_is_null_not_a_crash(self):
        payload = worker_tasks.build_webhook_payload(submission(ai_result={"questions": []}))
        self.assertIsNone(payload["final_score"])
        payload = worker_tasks.build_webhook_payload(submission(ai_result={"overall_score": "85"}))
        self.assertIsNone(payload["final_score"])
        payload = worker_tasks.build_webhook_payload(submission(ai_result={"overall_score": True}))
        self.assertIsNone(payload["final_score"])
        payload = worker_tasks.build_webhook_payload(submission(ai_result=None))
        self.assertIsNone(payload["final_score"])

    def test_the_payload_serialises_to_json(self):
        import json

        json.dumps(worker_tasks.build_webhook_payload(submission()), separators=(",", ":"), ensure_ascii=False)


if __name__ == "__main__":
    unittest.main()
