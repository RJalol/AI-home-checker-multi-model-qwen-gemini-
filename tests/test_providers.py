import json
import unittest

from hwchecker.providers.base import (
    AIProviderError, ProviderUnavailableError, build_grading_result, extract_json_object, normalise_questions,
)

from .fakes import GLM_HOST, OPENAI_HOST, QWEN_HOST, FakeSDKTestCase, grade_json


class ExtractJsonObjectTests(unittest.TestCase):
    def test_plain_fenced_prose_and_think_block(self):
        self.assertEqual(extract_json_object('{"score": 1}'), {"score": 1})
        self.assertEqual(extract_json_object('```json\n{"score": 2}\n```'), {"score": 2})
        self.assertEqual(extract_json_object('Here you go:\n{"score": 3, "feedback": "a {b} c"}\nBye'), {"score": 3, "feedback": "a {b} c"})
        self.assertEqual(extract_json_object('<think>maybe {"x": 1}</think>{"classification": "CORRECT"}', prefer_keys=("classification",)),
                         {"classification": "CORRECT"})

    def test_prefers_the_object_with_the_wanted_key(self):
        text = 'Scratch: {"a": 1}\nAnswer: {"classification": "CORRECT", "score": 9}'
        self.assertEqual(extract_json_object(text, prefer_keys=("classification",))["score"], 9)

    def test_rejects_garbage(self):
        for bad in ("not json", "[1, 2]", "", "   ", None):
            with self.assertRaises(ValueError, msg=repr(bad)):
                extract_json_object(bad)


class BuildGradingResultTests(unittest.TestCase):
    def build(self, data):
        return build_grading_result(data, default_confidence=0.8, input_tokens=1, output_tokens=2, latency_ms=3, provider_name="p")

    def test_normalises_loose_model_output(self):
        r = self.build({"classification": "partially correct", "score": "55", "confidence": 85, "error_type": "None"})
        self.assertEqual((r.classification, r.score, r.confidence, r.error_type), ("PARTIALLY_CORRECT", 55.0, 0.85, None))

    def test_clamps_score_and_defaults_confidence(self):
        r = self.build({"classification": "CORRECT", "score": 140})
        self.assertEqual((r.score, r.confidence, r.is_correct), (100.0, 0.8, True))

    def test_missing_or_unknown_classification_is_an_error_not_a_zero(self):
        for data in ({"score": 5}, {"classification": "GREAT", "score": 5}, {"classification": "CORRECT"}, {"classification": "CORRECT", "score": "n/a"}):
            with self.assertRaises(AIProviderError, msg=data):
                self.build(data)

    def test_questions_are_cleaned(self):
        qs = normalise_questions([{"score": "x", "mistakes": "a", "feedback": "fb", "error_types": ["E"]}, "junk", {"question_number": "7", "score": 150}], default_score=40)
        self.assertEqual(len(qs), 2)
        self.assertEqual((qs[0]["score"], qs[0]["mistakes"], qs[0]["analysis"], qs[0]["error_categories"]), (40.0, ["a"], "fb", ["E"]))
        self.assertEqual((qs[1]["question_number"], qs[1]["score"]), (7, 100.0))


class OpenAIProtocolProviderTests(FakeSDKTestCase):
    def content(self, **extra):
        return {"user_prompt": "grade", "system_prompt": "sys", **extra}

    def test_openai_sends_a_request_openai_accepts(self):
        from hwchecker.providers.openai import OpenAIProvider

        result = OpenAIProvider("gpt-5.4-mini").grade_homework(self.content(), {})
        call = self.backend.grading(OPENAI_HOST)[0]
        self.assertIn("max_completion_tokens", call["kwargs"])
        for forbidden in ("max_tokens", "temperature", "top_p"):
            self.assertNotIn(forbidden, call["kwargs"])
        self.assertEqual(result.classification, "PARTIALLY_CORRECT")
        self.assertEqual(len(result.questions), 2)

    def test_qwen_keeps_classic_parameters(self):
        from hwchecker.providers.qwen import QwenProvider

        QwenProvider("qwen3.5-flash").grade_homework(self.content(), {})
        kwargs = self.backend.grading(QWEN_HOST)[0]["kwargs"]
        self.assertIn("max_tokens", kwargs)
        self.assertIn("temperature", kwargs)

    def test_adapts_when_the_endpoint_rejects_a_parameter(self):
        """A provider configured the 'old' way against a strict endpoint learns and retries once."""
        from hwchecker.providers.openai import OpenAIProvider

        provider = OpenAIProvider("gpt-5.4-mini")
        provider._token_param, provider._sampling_ok = "max_tokens", True   # simulate a stale configuration
        provider.grade_homework(self.content(), {})
        self.assertEqual(len(self.backend.grading(OPENAI_HOST)), 3)   # max_tokens rejected, then temperature, then OK
        self.assertEqual((provider._token_param, provider._sampling_ok), ("max_completion_tokens", False))
        provider.grade_homework(self.content(), {})
        self.assertEqual(len(self.backend.grading(OPENAI_HOST)), 4)   # learned: next call succeeds first time

    def test_json_in_a_code_fence_is_accepted(self):
        from hwchecker.providers.glm import GLMProvider

        self.backend.cfg[f"grade_{GLM_HOST}"] = "```json\n" + grade_json(score=77, classification="CORRECT") + "\n```"
        self.assertEqual(GLMProvider("glm-5.2").grade_homework(self.content(), {}).score, 77)

    def test_garbage_reply_is_a_retryable_provider_error(self):
        from hwchecker.providers.qwen import QwenProvider

        self.backend.cfg[f"grade_{QWEN_HOST}"] = "sorry, I cannot do that"
        with self.assertRaises(AIProviderError) as ctx:
            QwenProvider("qwen3.5-flash").grade_homework(self.content(), {})
        self.assertTrue(ctx.exception.retryable)

    def test_text_only_model_refuses_a_photo_without_being_retryable(self):
        from hwchecker.providers.glm import GLMProvider

        with self.assertRaises(AIProviderError) as ctx:
            GLMProvider("glm-5.2").grade_homework(self.content(file_path=self.photo), {})
        self.assertFalse(ctx.exception.retryable)


class OtherProvidersTests(FakeSDKTestCase):
    def test_claude_accepts_a_fenced_reply(self):
        from hwchecker.providers.claude import ClaudeProvider

        self.backend.cfg["claude_text"] = "```json\n" + grade_json(score=20, classification="INCORRECT") + "\n```"
        r = ClaudeProvider("claude-sonnet-5").grade_homework({"user_prompt": "g", "system_prompt": "s"}, {})
        self.assertEqual((r.classification, r.score), ("INCORRECT", 20.0))

    def test_claude_request_carries_a_token_limit(self):
        from hwchecker.providers.claude import ClaudeProvider

        ClaudeProvider("claude-sonnet-5").grade_homework({"user_prompt": "g", "system_prompt": "s"}, {})
        self.assertIn("max_tokens", self.backend.grading("anthropic")[0]["kwargs"])

    def test_missing_key_is_provider_unavailable(self):
        import os
        from hwchecker.providers.claude import ClaudeProvider

        os.environ.pop("ANTHROPIC_API_KEY", None)
        with self.assertRaises(ProviderUnavailableError):
            ClaudeProvider("claude-sonnet-5")


if __name__ == "__main__":
    unittest.main()
