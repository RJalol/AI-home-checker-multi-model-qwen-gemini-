import os
import unittest
from unittest import mock

from hwchecker.providers import registry
from hwchecker.router import grading_router


class OpenAIPresetTests(unittest.TestCase):
    def env(self, **values):
        patcher = mock.patch.dict(os.environ, values, clear=False)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_default_is_unchanged(self):
        self.assertEqual(registry.resolve_tier_config("medium")[0], "qwen")
        self.assertEqual(registry.resolve_tier_config("hard")[0], "glm")

    def test_preset_moves_text_and_photo_tiers_to_openai(self):
        self.env(HWCHECKER_PRESET="openai")
        for tier in ("router", "easy", "medium", "medium_alt", "hard", "hard_vision", "verification", "fallback"):
            self.assertEqual(registry.resolve_tier_config(tier)[0], "openai", tier)
        self.assertEqual(registry.resolve_tier_config("audio")[0], "gemini")   # no OpenAI audio path

    def test_single_tier_override_still_wins(self):
        self.env(HWCHECKER_PRESET="openai", HWCHECKER_HARD_MODEL="gpt-5.4-mini")
        self.assertEqual(registry.resolve_tier_config("hard")[:2], ("openai", "gpt-5.4-mini"))

    def test_unknown_preset_is_reported(self):
        self.env(HWCHECKER_PRESET="nope")
        with self.assertRaises(ValueError):
            registry.resolve_tier_config("medium")

    def test_preset_changes_the_cache_fingerprint(self):
        before = registry.config_fingerprint()
        self.env(HWCHECKER_PRESET="openai")
        self.assertNotEqual(before, registry.config_fingerprint())

    def test_same_vendor_verification_only_when_allowed(self):
        grader = mock.Mock(vendor="openai")
        grader.name = "gpt-6-luna"
        verifier = mock.Mock(vendor="openai")
        verifier.name = "gpt-5.4-mini"

        def fake_get(tier):
            if tier == "verification":
                return verifier
            raise grading_router.AIProviderError("no key")

        router = grading_router.GradingRouter.__new__(grading_router.GradingRouter)
        with mock.patch.object(grading_router, "get_provider", fake_get), \
                mock.patch.object(grading_router, "_can_read", lambda *_: True):
            self.assertIsNone(router._pick_verifier(grader, "text"))          # off by default
            self.env(HWCHECKER_PRESET="openai")
            self.assertIs(router._pick_verifier(grader, "text"), verifier)    # allowed by the preset


if __name__ == "__main__":
    unittest.main()


class VendorDetectionTests(unittest.TestCase):
    def test_real_providers_report_their_vendor(self):
        from hwchecker.providers.gemini import GeminiProvider
        from hwchecker.providers.openai import OpenAIProvider
        from hwchecker.providers.qwen import QwenProvider

        for cls, vendor in ((GeminiProvider, "gemini"), (OpenAIProvider, "openai"), (QwenProvider, "qwen")):
            instance = cls.__new__(cls)        # no API key needed to ask which vendor a class belongs to
            self.assertEqual(grading_router._vendor_of(instance), vendor)

    def test_same_vendor_sibling_is_skipped_without_the_preset(self):
        from hwchecker.providers.gemini import GeminiProvider

        grader, sibling = GeminiProvider.__new__(GeminiProvider), GeminiProvider.__new__(GeminiProvider)
        grader.name, sibling.name = "gemini-3.6-flash", "gemini-3.5-flash-lite"
        router = grading_router.GradingRouter.__new__(grading_router.GradingRouter)
        with mock.patch.object(grading_router, "get_provider", lambda tier: sibling), \
                mock.patch.object(grading_router, "_can_read", lambda *_: True):
            self.assertIsNone(router._pick_verifier(grader, "text"))


class PresetModelTests(unittest.TestCase):
    def test_one_model_for_every_openai_tier(self):
        with mock.patch.dict(os.environ, {"HWCHECKER_PRESET": "openai", "HWCHECKER_PRESET_MODEL": "gpt-5.4-mini"}):
            for tier in ("router", "easy", "medium", "hard", "hard_vision", "verification"):
                self.assertEqual(registry.resolve_tier_config(tier)[:2], ("openai", "gpt-5.4-mini"))
            self.assertEqual(registry.resolve_tier_config("audio")[0], "gemini")


class OpenAIModelSettingsTests(unittest.TestCase):
    def test_temperature_only_for_models_that_accept_it(self):
        from hwchecker.providers.openai import OpenAIProvider
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test"}):
            self.assertTrue(OpenAIProvider("gpt-4.1-mini").sends_sampling_params)
            self.assertFalse(OpenAIProvider("gpt-5-mini").sends_sampling_params)

    def test_known_models_have_real_prices(self):
        from hwchecker.providers.openai import OpenAIProvider
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test"}):
            p = OpenAIProvider("gpt-4.1-mini")
            self.assertAlmostEqual(p.estimate_cost(1_000_000, 1_000_000), 2.0)
