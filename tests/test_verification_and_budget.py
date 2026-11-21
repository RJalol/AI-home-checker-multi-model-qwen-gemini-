"""
Qwen + Gemini cost plan: second opinions always come from a different vendor than the grader,
the result cache never serves an answer from an older model set-up, the optional output-token
cap, and the spend watch-dog (alert only, never blocks grading).
"""

import logging
import os
import time
import unittest
from unittest import mock

from hwchecker.infra import budget
from hwchecker.providers import registry
from hwchecker.providers.base import output_token_limit
from hwchecker.router.grading_router import GradingRouter

from .fakes import QWEN_HOST, grade_json
from .test_router import RouterTestCase


class CrossVendorVerification(RouterTestCase):
    def low(self, host):
        self.backend.cfg[f"grade_{host}"] = grade_json(confidence=0.4)

    def test_default_set_up_still_uses_haiku_which_is_a_third_vendor(self):
        self.say_difficulty("easy")
        self.low(QWEN_HOST)
        self.grade()
        self.assertEqual([c["model"] for c in self.backend.grading("anthropic")], ["claude-haiku-5-5"])
        self.assertEqual(self.backend.grading("gemini"), [])

    def test_qwen_grade_is_rechecked_by_gemini_when_there_is_no_anthropic_key(self):
        os.environ.pop("ANTHROPIC_API_KEY")
        self.say_difficulty("easy")
        self.low(QWEN_HOST)
        out = self.grade()
        self.assertEqual([c["model"] for c in self.backend.grading("gemini")], ["gemini-3.5-flash-lite"])
        self.assertEqual(self.backend.grading("gemini")[0]["thinking"].thinking_level, "low")
        self.assertIn("verification", out)
        self.assertEqual(len(self.backend.grading(QWEN_HOST)), 1)   # Qwen did not check its own work

    def test_gemini_grade_is_rechecked_by_qwen_never_by_gemini(self):
        os.environ.pop("ANTHROPIC_API_KEY")
        os.environ["HWCHECKER_MEDIUM_PROVIDER"] = "gemini"   # a Gemini grader on a verifiable tier
        os.environ["HWCHECKER_MEDIUM_MODEL"] = "gemini-3.5-flash-lite"
        self.say_difficulty("medium")
        self.backend.cfg["gemini_text"] = grade_json(confidence=0.4)
        out = self.grade()
        self.assertEqual(len(self.backend.grading("gemini")), 1)   # the grade itself; no Gemini second opinion
        self.assertEqual([c["model"] for c in self.backend.grading(QWEN_HOST)], ["qwen3.7-plus"])
        self.assertIn("verification", out)

    def test_only_gemini_key_means_no_second_opinion_and_no_caching(self):
        for key in ("ANTHROPIC_API_KEY", "QWEN_API_KEY", "GLM_API_KEY", "OPENAI_API_KEY"):
            os.environ.pop(key)
        self.backend.cfg["gemini_text"] = grade_json(confidence=0.4)
        first = self.grade()
        self.assertNotIn("verification", first)
        self.grade()
        self.assertEqual(len(self.backend.grading("gemini")), 2)   # unverified results are not cached

    def test_audio_is_not_checked_by_a_vendor_that_cannot_hear(self):
        os.environ.pop("ANTHROPIC_API_KEY")
        audio = os.path.join(self.tmp.name, "answer.mp3")
        with open(audio, "wb") as fh:
            fh.write(b"ID3fake")
        self.say_difficulty("easy")
        self.backend.cfg["gemini_text"] = grade_json(confidence=0.4)
        self.grade(path=audio, subject="english", skill="speaking")
        self.assertEqual(self.backend.grading(QWEN_HOST), [])   # Qwen cannot take audio; Gemini would be self-review

    def test_pick_verifier_skips_the_graders_vendor_and_model(self):
        router = GradingRouter()
        gemini = registry.get_provider("medium_alt")
        self.assertEqual(router._pick_verifier(gemini, "text").name, "claude-haiku-5-5")
        os.environ.pop("ANTHROPIC_API_KEY")
        registry.reset_registry()
        gemini = registry.get_provider("medium_alt")
        self.assertEqual(router._pick_verifier(gemini, "text").name, "qwen3.7-plus")
        for qwen_tier in ("easy", "medium"):
            qwen = registry.get_provider(qwen_tier)
            self.assertEqual(router._pick_verifier(qwen, "text").name, "gemini-3.5-flash-lite")
        audio_grader = registry.get_provider("audio")   # Gemini: only Gemini can hear, and it may not check itself
        self.assertIsNone(router._pick_verifier(audio_grader, "audio"))

    def test_every_provider_is_tagged_with_its_vendor(self):
        for tier in registry.TIER_DEFAULTS:
            provider = registry.get_provider(tier)
            self.assertEqual(provider.vendor, registry.resolve_tier_config(tier)[0])


class CacheKnowsTheModelSetUp(RouterTestCase):
    def test_changing_a_tier_model_does_not_serve_the_old_result(self):
        self.say_difficulty("easy")
        self.assertNotIn("cached", self.grade())
        self.assertTrue(self.grade()["cached"])
        os.environ["HWCHECKER_EASY_MODEL"] = "qwen3.8-flash"
        registry.reset_registry()
        again = self.grade()
        self.assertNotIn("cached", again)
        self.assertEqual(again["provider"], "qwen3.8-flash")

    def test_changing_the_thinking_level_also_misses(self):
        self.say_difficulty("easy")
        self.grade()
        os.environ["HWCHECKER_EASY_THINKING"] = "low"
        registry.reset_registry()
        self.assertNotIn("cached", self.grade())

    def test_fingerprint_changes_only_with_the_set_up(self):
        before = registry.config_fingerprint()
        self.assertEqual(before, registry.config_fingerprint())
        os.environ["HWCHECKER_MEDIUM_THINKING"] = "high"
        self.assertNotEqual(before, registry.config_fingerprint())


class OutputTokenCap(unittest.TestCase):
    def test_default_is_untouched(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HWCHECKER_MAX_OUTPUT_TOKENS", None)
            self.assertEqual(output_token_limit(8192), 8192)

    def test_env_replaces_the_default_but_never_goes_below_1024(self):
        with mock.patch.dict(os.environ, {"HWCHECKER_MAX_OUTPUT_TOKENS": "4096"}):
            self.assertEqual(output_token_limit(8192), 4096)
        with mock.patch.dict(os.environ, {"HWCHECKER_MAX_OUTPUT_TOKENS": "100"}):
            self.assertEqual(output_token_limit(8192), 1024)
        with mock.patch.dict(os.environ, {"HWCHECKER_MAX_OUTPUT_TOKENS": "lots"}):
            self.assertEqual(output_token_limit(8192), 8192)


class BudgetWatchdog(RouterTestCase):
    def alerts(self, cm):
        return [r.getMessage() for r in cm.records if r.levelno == logging.WARNING]

    def test_off_by_default_records_nothing(self):
        budget.record_spend(5.0)
        self.assertEqual(budget.current_spend(), {"day": 0.0, "month": 0.0})

    def test_alerts_at_80_and_100_percent_once_and_never_raises(self):
        os.environ["HWCHECKER_DAILY_BUDGET_USD"] = "10"
        with self.assertLogs("hwchecker.budget", level="WARNING") as cm:
            budget.record_spend(7.0)    # 70%: quiet
            budget.record_spend(1.5)    # 85%: first alert
            budget.record_spend(0.2)    # still 87%: not repeated
            budget.record_spend(2.0)    # 107%: second alert
            budget.record_spend(1.0)    # more spending: not repeated
        messages = self.alerts(cm)
        self.assertEqual(len(messages), 2)
        self.assertIn("80%", messages[0])
        self.assertIn("100%", messages[1])
        self.assertIn("NOT limited", messages[0])

    def test_a_single_big_call_announces_both_levels(self):
        os.environ["HWCHECKER_MONTHLY_BUDGET_USD"] = "10"
        with self.assertLogs("hwchecker.budget", level="WARNING") as cm:
            budget.record_spend(12.0)
        self.assertEqual(len(self.alerts(cm)), 2)

    def test_day_and_month_are_counted_separately(self):
        os.environ["HWCHECKER_DAILY_BUDGET_USD"] = "100"
        os.environ["HWCHECKER_MONTHLY_BUDGET_USD"] = "1000"
        day1 = time.mktime((2026, 11, 20, 12, 0, 0, 0, 0, 0))
        day2 = day1 + 86400
        budget.record_spend(3.0, now=day1)
        budget.record_spend(4.0, now=day2)
        self.assertEqual(budget.current_spend(now=day1), {"day": 3.0, "month": 7.0})
        self.assertEqual(budget.current_spend(now=day2), {"day": 4.0, "month": 7.0})

    def test_bad_budget_values_mean_off(self):
        for bad in ("abc", "-5", "0", ""):
            os.environ["HWCHECKER_DAILY_BUDGET_USD"] = bad
            self.assertEqual(budget.daily_budget_usd(), 0.0)

    def test_grading_feeds_the_watchdog_and_is_never_blocked_by_it(self):
        os.environ["HWCHECKER_DAILY_BUDGET_USD"] = "0.0000001"   # absurdly low: already exceeded
        self.say_difficulty("easy")
        with self.assertLogs("hwchecker.budget", level="WARNING"):
            out = self.grade()
        self.assertEqual(out["provider"], "qwen3.5-flash")   # still graded by the same cheap model
        self.assertGreater(budget.current_spend()["day"], 0)

    def test_a_broken_redis_falls_back_to_memory(self):
        os.environ["HWCHECKER_DAILY_BUDGET_USD"] = "10"

        class Broken:
            def incrbyfloat(self, *a, **k):
                raise RuntimeError("redis down")

            get = set = expire = incrbyfloat

        with mock.patch.object(budget, "get_redis", lambda: Broken()):
            budget.record_spend(2.0)   # must not raise
            self.assertEqual(budget.current_spend()["day"], 2.0)   # the in-process total kept the spend


if __name__ == "__main__":
    unittest.main()
