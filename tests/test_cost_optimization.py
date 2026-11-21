"""
The cost-optimised model set-up: which model/thinking level each tier gets, that the thinking
settings really reach the wire (and degrade gracefully when an endpoint does not know them),
and the router behaviour that keeps the cheaper set-up from making answers worse (no re-check
of the strongest model's verdict, a dedicated audio tier, re-check thresholds).
"""

import os
import unittest
from unittest import mock

from hwchecker.providers import registry
from hwchecker.providers.base import ProviderUnavailableError, normalize_thinking
from hwchecker.router import grading_router
from hwchecker.router.grading_router import GradingRouter

from .fakes import GLM_HOST, OPENAI_HOST, QWEN_HOST, FakeSDKTestCase, grade_json
from .test_router import RouterTestCase


class NormalizeThinkingTests(unittest.TestCase):
    def test_spellings(self):
        self.assertIsNone(normalize_thinking(None))
        self.assertIsNone(normalize_thinking(""))
        self.assertIsNone(normalize_thinking(" Default "))
        for word in ("off", "OFF", "none", "false", "disabled", "0"):
            self.assertEqual(normalize_thinking(word), "off")
        self.assertEqual(normalize_thinking(" Low "), "low")
        self.assertEqual(normalize_thinking("2048"), "2048")


class RegistryDefaults(FakeSDKTestCase):
    def test_default_tier_table(self):
        tiers = registry.describe_tiers()
        self.assertEqual(tiers["router"], {"provider": "qwen", "model": "qwen3.7-flash", "thinking": "off"})
        self.assertEqual(tiers["easy"], {"provider": "qwen", "model": "qwen3.5-flash", "thinking": "off"})
        self.assertEqual(tiers["medium"], {"provider": "qwen", "model": "qwen3.7-plus", "thinking": "low"})
        self.assertEqual(tiers["medium_alt"], {"provider": "gemini", "model": "gemini-3.5-flash-lite", "thinking": "low"})
        self.assertEqual(tiers["hard"], {"provider": "glm", "model": "glm-5.2", "thinking": "high"})
        self.assertEqual(tiers["hard_vision"], {"provider": "gemini", "model": "gemini-3.6-flash", "thinking": "high"})
        self.assertEqual(tiers["audio"], {"provider": "gemini", "model": "gemini-3.6-flash", "thinking": "low"})
        self.assertEqual(tiers["verification"], {"provider": "claude", "model": "claude-haiku-5-5", "thinking": "default"})
        self.assertEqual(tiers["fallback"]["model"], "gpt-5.4-mini")

    def test_every_tier_builds_and_is_tagged_with_its_tier(self):
        for tier in registry.TIER_DEFAULTS:
            self.assertEqual(registry.get_provider(tier).tier, tier)

    def test_model_and_thinking_can_be_overridden_from_the_environment(self):
        os.environ["HWCHECKER_MEDIUM_MODEL"] = "qwen3.8-flash"
        os.environ["HWCHECKER_MEDIUM_THINKING"] = "medium"
        os.environ["HWCHECKER_EASY_THINKING"] = "default"
        self.assertEqual(registry.resolve_tier_config("medium"), ("qwen", "qwen3.8-flash", "medium"))
        self.assertEqual(registry.get_provider("medium").name, "qwen3.8-flash")
        self.assertIsNone(registry.resolve_tier_config("easy")[2])

    def test_provider_override_needs_a_model(self):
        os.environ["HWCHECKER_VERIFICATION_PROVIDER"] = "openai"
        with self.assertRaises(ProviderUnavailableError):
            registry.get_provider("verification")
        os.environ["HWCHECKER_VERIFICATION_MODEL"] = "gpt-5.4-mini"
        registry.reset_registry()
        self.assertEqual(registry.get_provider("verification").name, "gpt-5.4-mini")

    def test_unknown_provider_name_is_reported_not_crashing_describe(self):
        os.environ["HWCHECKER_HARD_PROVIDER"] = "nope"
        self.assertIn("error", registry.describe_tiers()["hard"])
        with self.assertRaises(ProviderUnavailableError):
            registry.get_provider("hard")


class ThinkingReachesTheWire(RouterTestCase):
    def test_qwen_router_and_easy_tier_run_without_thinking(self):
        self.say_difficulty("easy")
        self.grade()
        self.assertEqual(self.backend.router_calls()[0]["extra_body"], {"enable_thinking": False})
        self.assertEqual(self.backend.grading(QWEN_HOST)[0]["extra_body"], {"enable_thinking": False})

    def test_glm_hard_tier_uses_high_reasoning_effort(self):
        import docx

        path = os.path.join(self.tmp.name, "proof.docx")
        document = docx.Document()
        document.add_paragraph("Proof: assume sqrt(2) = p/q ...")
        document.save(path)
        self.say_difficulty("hard")
        self.grade(path=path)
        self.assertEqual(self.backend.grading(GLM_HOST)[0]["extra_body"], {"reasoning_effort": "high"})

    def test_qwen_medium_tier_uses_a_low_thinking_budget(self):
        self.say_difficulty("medium")
        self.grade()
        self.assertEqual(self.backend.grading(QWEN_HOST)[0]["extra_body"], {"enable_thinking": True, "thinking_budget": 1024})

    def test_gemini_hard_vision_tier_uses_high_thinking(self):
        self.say_difficulty("hard")
        self.grade()   # photo -> hard_vision (Gemini 3.6 Flash)
        self.assertEqual(self.backend.grading("gemini")[0]["thinking"].thinking_level, "high")

    def test_claude_tiers_keep_the_model_default_unless_configured(self):
        self.say_difficulty("easy")
        self.backend.cfg[f"grade_{QWEN_HOST}"] = grade_json(confidence=0.4)
        self.grade()   # low confidence -> the Haiku verifier
        self.assertIsNone(self.backend.grading("anthropic")[0]["extra_body"])

    def test_claude_effort_can_be_lowered_for_the_verifier(self):
        os.environ["HWCHECKER_VERIFICATION_THINKING"] = "low"
        self.say_difficulty("easy")
        self.backend.cfg[f"grade_{QWEN_HOST}"] = grade_json(confidence=0.4)
        self.grade()
        self.assertEqual(self.backend.grading("anthropic")[0]["extra_body"], {"output_config": {"effort": "low"}})

    def test_claude_thinking_can_be_switched_off(self):
        os.environ["HWCHECKER_VERIFICATION_THINKING"] = "off"
        self.say_difficulty("easy")
        self.backend.cfg[f"grade_{QWEN_HOST}"] = grade_json(confidence=0.4)
        self.grade()
        self.assertEqual(self.backend.grading("anthropic")[0]["extra_body"], {"thinking": {"type": "disabled"}})

    def test_qwen_levels_and_budgets(self):
        from hwchecker.providers.qwen import QwenProvider

        self.assertEqual(QwenProvider("qwen3.5-flash", thinking="low")._thinking_extra_body(),
                         {"enable_thinking": True, "thinking_budget": 1024})
        self.assertEqual(QwenProvider("qwen3.5-flash", thinking="2048")._thinking_extra_body(),
                         {"enable_thinking": True, "thinking_budget": 2048})
        self.assertIsNone(QwenProvider("qwen3.5-flash", thinking="high")._thinking_extra_body())
        self.assertIsNone(QwenProvider("qwen3.5-flash")._thinking_extra_body())

    def test_glm_off_disables_thinking(self):
        from hwchecker.providers.glm import GLMProvider

        self.assertEqual(GLMProvider(thinking="off")._thinking_extra_body(), {"thinking": {"type": "disabled"}})

    def test_the_openai_fallback_is_sent_no_thinking_fields(self):
        self.say_difficulty("easy")
        self.backend.cfg[f"fail_{QWEN_HOST}"] = True
        self.grade()
        self.assertIsNone(self.backend.grading(OPENAI_HOST)[0]["extra_body"])


class ThinkingDegradesGracefully(RouterTestCase):
    """If an endpoint rejects the new field, grading must still work -- just without the saving."""

    def setUp(self):
        super().setUp()
        self.backend.cfg["reject_thinking"] = True

    def test_qwen_retries_without_the_field(self):
        self.say_difficulty("easy")
        out = self.grade()
        self.assertEqual(out["provider"], "qwen3.5-flash")
        grading = self.backend.grading(QWEN_HOST)
        self.assertEqual([c["extra_body"] for c in grading], [{"enable_thinking": False}, None])
        # the instance remembers: later calls no longer send the field
        again = self.homework()
        again["bypass_cache"] = True
        GradingRouter().grade(again)
        self.assertEqual([c["extra_body"] for c in self.backend.grading(QWEN_HOST)], [{"enable_thinking": False}, None, None])

    def test_gemini_retries_without_thinking_config(self):
        self.say_difficulty("hard")   # photo -> hard_vision (Gemini 3.6 Flash)
        out = self.grade()
        self.assertEqual(out["provider"], "gemini-3.6-flash")
        calls = self.backend.grading("gemini")
        self.assertEqual(len(calls), 2)
        self.assertIsNotNone(calls[0]["thinking"])
        self.assertIsNone(calls[1]["thinking"])

    def test_claude_retries_without_the_field(self):
        # point the hard-photo tier back at Claude Sonnet (the pre-benchmark set-up) with env only
        os.environ["HWCHECKER_HARD_VISION_PROVIDER"] = "claude"
        os.environ["HWCHECKER_HARD_VISION_MODEL"] = "claude-sonnet-5"
        os.environ["HWCHECKER_HARD_VISION_THINKING"] = "high"
        self.say_difficulty("hard")
        out = self.grade()
        self.assertEqual(out["provider"], "claude-sonnet-5")
        calls = self.backend.grading("anthropic")
        self.assertEqual([c["extra_body"] for c in calls], [{"output_config": {"effort": "high"}}, None])


class GeminiTokenAccounting(RouterTestCase):
    def test_thinking_tokens_count_as_output(self):
        self.backend.cfg["gemini_thoughts_tokens"] = 500
        self.say_difficulty("hard")   # photo -> hard_vision (Gemini 3.6 Flash)
        out = self.grade()
        self.assertEqual(out["tokens"], {"input": 90, "output": 540, "cached": 0})   # 40 visible + 500 thinking


class VerificationBehaviour(RouterTestCase):
    def test_second_opinion_comes_from_haiku(self):
        self.say_difficulty("easy")
        self.backend.cfg[f"grade_{QWEN_HOST}"] = grade_json(confidence=0.4)
        self.grade()
        self.assertEqual([c["model"] for c in self.backend.grading("anthropic")], ["claude-haiku-5-5"])

    def test_confidence_just_above_the_new_threshold_is_not_rechecked(self):
        self.say_difficulty("easy")
        self.backend.cfg[f"grade_{QWEN_HOST}"] = grade_json(confidence=0.72)   # was re-checked at the old 0.75
        self.grade()
        self.assertEqual(self.backend.grading("anthropic"), [])

    def test_score_margin_is_two_points(self):
        self.say_difficulty("easy")
        self.backend.cfg[f"grade_{QWEN_HOST}"] = grade_json(score=67.5)   # 2.5 from the 70 boundary: no longer borderline
        self.grade()
        self.assertEqual(self.backend.grading("anthropic"), [])
        self.backend.cfg[f"grade_{QWEN_HOST}"] = grade_json(score=68.5)   # 1.5 away: re-checked
        self.grade(extra="a different submission")
        self.assertEqual(len(self.backend.grading("anthropic")), 1)

    def test_thresholds_come_from_the_environment(self):
        with mock.patch.dict(os.environ, {"HWCHECKER_VERIFY_CONFIDENCE_THRESHOLD": "0.55", "HWCHECKER_VERIFY_SCORE_MARGIN": "x"}):
            self.assertEqual(grading_router._env_float("HWCHECKER_VERIFY_CONFIDENCE_THRESHOLD", 0.70), 0.55)
            self.assertEqual(grading_router._env_float("HWCHECKER_VERIFY_SCORE_MARGIN", 2.0), 2.0)   # unparsable -> default
        self.assertEqual(grading_router._env_float("HWCHECKER_NOT_SET_ANYWHERE", 1.5), 1.5)

    def test_the_strongest_models_verdict_is_never_overruled_and_stays_cached(self):
        self.say_difficulty("hard")   # photo -> hard_vision (Gemini 3.6 Flash)
        self.backend.cfg["gemini_text"] = grade_json(confidence=0.4)   # low confidence, borderline-free
        first = self.grade()
        self.assertEqual(first["provider"], "gemini-3.6-flash")
        self.assertEqual([c["model"] for c in self.backend.grading("gemini")], ["gemini-3.6-flash"])
        self.assertEqual(self.backend.grading("anthropic"), [])   # Haiku not asked
        calls = len(self.backend.calls)
        self.assertTrue(self.grade()["cached"])   # nothing was "left unverified", so the result is cached
        self.assertEqual(len(self.backend.calls), calls)

    def test_needs_verification_ignores_results_from_the_no_verify_tiers(self):
        router = GradingRouter()
        result = mock.Mock(confidence=0.1, score=40.0)
        self.assertTrue(router._needs_verification(result))
        self.assertFalse(router._needs_verification(result, mock.Mock(tier="hard_vision")))
        self.assertFalse(router._needs_verification(result, mock.Mock(tier="verification")))
        self.assertTrue(router._needs_verification(result, mock.Mock(tier="medium")))


class HardPhotoAndAudioRouting(RouterTestCase):
    def test_hard_photo_goes_to_hard_vision_not_to_the_verifier(self):
        self.say_difficulty("hard")
        out = self.grade()
        self.assertEqual(out["provider"], "gemini-3.6-flash")

    def test_hard_vision_model_can_be_overridden(self):
        os.environ["HWCHECKER_HARD_VISION_MODEL"] = "gemini-3.8-flash"
        self.say_difficulty("hard")
        self.assertEqual(self.grade()["provider"], "gemini-3.8-flash")

    def test_hard_photo_can_go_back_to_sonnet_with_env_only(self):
        os.environ["HWCHECKER_HARD_VISION_PROVIDER"] = "claude"
        os.environ["HWCHECKER_HARD_VISION_MODEL"] = "claude-sonnet-5"
        self.say_difficulty("hard")
        self.assertEqual(self.grade()["provider"], "claude-sonnet-5")

    def test_hard_photo_does_not_need_an_anthropic_key(self):
        os.environ.pop("ANTHROPIC_API_KEY")
        self.say_difficulty("hard")
        self.assertEqual(self.grade()["provider"], "gemini-3.6-flash")

    def test_hard_photo_without_a_gemini_key_goes_to_the_medium_tier(self):
        os.environ.pop("GEMINI_API_KEY")
        self.say_difficulty("hard")
        self.assertEqual(self.grade()["provider"], "qwen3.7-plus")

    def test_audio_prefers_the_audio_tier_and_falls_back_to_medium(self):
        audio = os.path.join(self.tmp.name, "answer.mp3")
        with open(audio, "wb") as fh:
            fh.write(b"ID3fake")
        self.say_difficulty("easy")
        self.assertEqual(self.grade(path=audio, subject="english", skill="listening")["provider"], "gemini-3.6-flash")
        self.assertEqual(self.backend.grading("gemini")[0]["thinking"].thinking_level, "low")
        self.backend.cfg["fail_gemini"] = 3   # the audio model fails all of its attempts
        out = self.grade(path=audio, subject="english", skill="speaking")
        self.assertIn("gemini-3.5-flash-lite", out["provider"])


class BenchmarkCandidates(FakeSDKTestCase):
    def test_spec_can_carry_a_thinking_setting(self):
        from hwchecker.benchmarks.compare_candidates import _build_provider

        tuned = _build_provider("gemini:gemini-3.5-flash-lite@low", [])
        self.assertEqual((tuned.model_name, tuned.thinking), ("gemini-3.5-flash-lite", "low"))
        self.assertIsNone(_build_provider("gemini:gemini-3.5-flash-lite", []).thinking)
        self.assertEqual(_build_provider("qwen:qwen3.5-flash@off", []).thinking, "off")
        with self.assertRaises(ValueError):
            _build_provider("qwen:@off", [])

    def test_first_candidate_of_each_tier_is_what_the_registry_uses(self):
        from hwchecker.benchmarks.compare_candidates import CANDIDATES

        for tier, specs in CANDIDATES.items():
            module, model, thinking = registry.resolve_tier_config(tier)
            expected = f"{module}:{model}" + (f"@{thinking}" if thinking else "")
            self.assertEqual(specs[0], expected, tier)


class ImageSizeDefault(unittest.TestCase):
    def test_photos_are_shrunk_to_1568px_by_default(self):
        import config

        if "MAX_IMAGE_SIDE_PX" in os.environ:
            self.skipTest("MAX_IMAGE_SIDE_PX is set in this environment")
        self.assertEqual(config.MAX_IMAGE_SIDE_PX, 1568)


if __name__ == "__main__":
    unittest.main()
