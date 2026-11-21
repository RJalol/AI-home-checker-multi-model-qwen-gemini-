"""
End-to-end behaviour of the real GradingRouter with only the SDKs faked. Scenarios S1-S6 are the
ones that exposed defects when a student's phone photo was sent through the CRM path.
"""

import json
import os
import unittest

from hwchecker.infra import circuit_breaker
from hwchecker.router.grading_router import GradingRouter, GradingRouterError
from prompts import build_user_prompt

from .fakes import GLM_HOST, OPENAI_HOST, QWEN_HOST, FakeSDKTestCase, grade_json


class RouterTestCase(FakeSDKTestCase):
    def homework(self, path=None, subject="math", skill=None, extra=""):
        # exactly what worker_tasks._grade_via_hwchecker_router builds for a CRM upload
        return {"subject_id": subject, "skill": skill,
                "content": {"file_path": path or self.photo, "user_prompt": build_user_prompt(extra)}}

    def grade(self, **kw):
        return GradingRouter().grade(self.homework(**kw))

    def say_difficulty(self, tier):
        self.backend.cfg["router_answer"] = json.dumps({"difficulty": tier})


class S1_OnlyGeminiKey(RouterTestCase):
    keys = ("GEMINI_API_KEY",)

    def test_photo_is_graded_through_gemini(self):
        out = self.grade()
        self.assertEqual(out["provider"], "gemini-3.5-flash-lite")
        self.assertEqual([c["host"] for c in self.backend.calls], ["gemini"])   # no router/Qwen call attempted
        self.assertTrue(self.backend.grading("gemini")[0]["image"])


class S2_EasyPhoto(RouterTestCase):
    def test_router_sees_the_photo_and_answer_is_requested_in_uzbek_per_question(self):
        self.say_difficulty("easy")
        out = self.grade()

        router_call = self.backend.router_calls()[0]
        self.assertTrue(router_call["image"], "the difficulty router must be shown the student's file")
        self.assertEqual(out["provider"], "qwen3.5-flash")
        grading = self.backend.grading(QWEN_HOST)[0]
        self.assertTrue(grading["image"])
        self.assertIn("Uzbek (Latin script)", grading["system"])
        self.assertIn("PER-QUESTION BREAKDOWN", grading["system"])
        self.assertEqual([q["question_number"] for q in out["questions"]], [1, 2])
        self.assertEqual(out["summary"]["weaknesses"], ["multiplication"])

    def test_feedback_language_can_be_changed(self):
        os.environ["FEEDBACK_LANGUAGE"] = "Russian"
        self.say_difficulty("easy")
        self.grade()
        system = self.backend.grading(QWEN_HOST)[0]["system"]
        self.assertIn("in Russian", system)
        self.assertNotIn("Uzbek", system)


class S3_HardPhoto(RouterTestCase):
    def test_text_only_glm_is_skipped_for_a_photo(self):
        self.say_difficulty("hard")
        out = self.grade()
        self.assertEqual(self.backend.to(GLM_HOST), [])
        self.assertEqual(out["provider"], "gemini-3.6-flash")      # the hard-photo tier (a vision model) takes it
        self.assertEqual(len(self.backend.grading("gemini")), 1)

    def test_typed_docx_can_still_use_glm(self):
        import docx

        path = os.path.join(self.tmp.name, "answer.docx")
        document = docx.Document()
        document.add_paragraph("Proof: assume sqrt(2) = p/q ...")
        document.save(path)
        self.say_difficulty("hard")
        out = self.grade(path=path)
        self.assertEqual(out["provider"], "glm-5.2")
        self.assertIn("Proof: assume", self.backend.grading(GLM_HOST)[0]["user"])


class S3b_ProviderOutage(RouterTestCase):
    def test_failed_primary_is_retried_then_the_fallback_gets_a_valid_request(self):
        self.say_difficulty("easy")
        self.backend.cfg[f"fail_{QWEN_HOST}"] = True
        out = self.grade()
        self.assertEqual(len(self.backend.grading(QWEN_HOST)), 3)             # 3 attempts, then give up
        fallback_calls = self.backend.grading(OPENAI_HOST)
        self.assertEqual(len(fallback_calls), 1)
        self.assertIn("max_completion_tokens", fallback_calls[0]["kwargs"])
        self.assertNotIn("temperature", fallback_calls[0]["kwargs"])
        self.assertIn("fallback, qwen3.5-flash failed", out["provider"])

    def test_a_transient_failure_is_retried_on_the_same_provider(self):
        self.say_difficulty("easy")
        self.backend.cfg[f"fail_{QWEN_HOST}"] = 1
        out = self.grade()
        self.assertEqual(out["provider"], "qwen3.5-flash")
        self.assertEqual(len(self.backend.grading(QWEN_HOST)), 2)
        self.assertEqual(self.backend.grading(OPENAI_HOST), [])

    def test_everything_down_raises_one_clear_error(self):
        self.say_difficulty("easy")
        for host in (QWEN_HOST, OPENAI_HOST, "gemini"):
            self.backend.cfg[f"fail_{host}"] = True
        with self.assertRaises(GradingRouterError) as ctx:
            self.grade()
        self.assertIn("every fallback provider failed", str(ctx.exception))

    def test_open_circuit_skips_the_provider_without_calling_it(self):
        self.say_difficulty("easy")
        for _ in range(circuit_breaker.FAILURE_THRESHOLD):
            circuit_breaker.record_failure("qwen3.5-flash")
        out = self.grade()
        self.assertEqual(self.backend.grading(QWEN_HOST), [])
        self.assertIn("fallback", out["provider"])


class S4_Verification(RouterTestCase):
    def test_low_confidence_gets_a_second_opinion_even_when_claude_fences_its_json(self):
        self.say_difficulty("easy")
        self.backend.cfg[f"grade_{QWEN_HOST}"] = grade_json(confidence=0.5)
        self.backend.cfg["claude_text"] = "```json\n" + grade_json(confidence=0.95, score=20, classification="INCORRECT") + "\n```"
        out = self.grade()
        self.assertEqual(out["classification"], "INCORRECT")
        self.assertIn("verified, overrode", out["provider"])
        self.assertEqual(len(self.backend.grading("anthropic")), 1)

    def test_agreeing_second_opinion_averages_scores(self):
        self.say_difficulty("easy")
        self.backend.cfg[f"grade_{QWEN_HOST}"] = grade_json(confidence=0.5, score=60)
        self.backend.cfg["claude_text"] = grade_json(confidence=0.9, score=64)
        out = self.grade()
        self.assertEqual(out["score"], 62.0)
        self.assertIn("(verified)", out["provider"])

    def test_a_model_is_never_asked_to_double_check_itself(self):
        self.say_difficulty("hard")   # photo -> graded by Gemini 3.6 Flash (GLM cannot see it)
        self.backend.cfg["gemini_text"] = grade_json(confidence=0.4)
        self.grade()
        self.assertEqual(len(self.backend.grading("gemini")), 1)   # not asked again ...
        self.assertEqual(self.backend.grading("anthropic"), [])    # ... and a cheaper verifier does not overrule it

    def test_missing_verifier_key_keeps_the_first_result(self):
        os.environ.pop("ANTHROPIC_API_KEY")
        os.environ.pop("GEMINI_API_KEY")   # no other vendor can give a second opinion either
        self.say_difficulty("easy")
        self.backend.cfg[f"grade_{QWEN_HOST}"] = grade_json(confidence=0.5)
        out = self.grade()
        self.assertEqual(out["provider"], "qwen3.5-flash")


class S5_RouterFailsSoft(RouterTestCase):
    def test_unusable_router_answer_means_medium(self):
        self.backend.cfg["router_answer"] = "not json"
        self.assertEqual(self.grade()["provider"], "qwen3.7-plus")

    def test_missing_qwen_key_means_medium_not_a_crash(self):
        os.environ.pop("QWEN_API_KEY")
        self.assertEqual(self.grade()["provider"], "gemini-3.5-flash-lite")

    def test_no_file_and_no_question_text_makes_no_router_call(self):
        from hwchecker.router.difficulty import classify_with_llm

        self.assertEqual(classify_with_llm({"subject_id": "math", "content": {}}, "math"), "medium")
        self.assertEqual(self.backend.calls, [])

    def test_question_text_is_classified_without_attaching_a_file(self):
        from hwchecker.router.difficulty import classify_with_llm

        self.say_difficulty("hard")
        self.assertEqual(classify_with_llm({"question_text": "Prove that ..."}, "math"), "hard")
        self.assertFalse(self.backend.router_calls()[0]["image"])


class S6_UzbekWriting(RouterTestCase):
    def test_language_homework_photo(self):
        self.say_difficulty("easy")
        out = self.grade(subject="uzbek", skill="writing")
        system = self.backend.grading(QWEN_HOST)[0]["system"]
        self.assertIn("WRITING task", system)
        self.assertIn("Uzbek (Latin script)", system)
        self.assertEqual(out["classification"], "PARTIALLY_CORRECT")


class Misc(RouterTestCase):
    keys = ()

    def test_no_provider_configured_explains_what_to_set(self):
        with self.assertRaises(GradingRouterError) as ctx:
            self.grade()
        self.assertIn("API_KEY", str(ctx.exception))
        self.assertIn(".env.example", str(ctx.exception))


class AudioRouting(RouterTestCase):
    def test_audio_goes_to_gemini_only(self):
        audio = os.path.join(self.tmp.name, "answer.mp3")
        with open(audio, "wb") as fh:
            fh.write(b"ID3fake")
        self.say_difficulty("easy")
        out = self.grade(path=audio, subject="english", skill="listening")
        self.assertEqual(out["provider"], "gemini-3.6-flash")   # the dedicated "audio" tier, not the cheap Flash-Lite
        self.assertEqual(self.backend.grading(QWEN_HOST), [])


class CachingTests(RouterTestCase):
    def test_identical_resubmission_is_free(self):
        self.say_difficulty("easy")
        first = self.grade()
        calls_after_first = len(self.backend.calls)
        second = self.grade()
        self.assertEqual(len(self.backend.calls), calls_after_first)
        self.assertTrue(second["cached"])
        self.assertEqual(second["tokens"], {"input": 0, "output": 0})
        self.assertEqual(second["score"], first["score"])
        self.assertNotIn("cached", first)

    def test_different_instructions_or_language_do_not_share_an_entry(self):
        self.say_difficulty("easy")
        self.grade()
        n = len(self.backend.grading())
        self.grade(extra="Be strict about units.")
        self.assertEqual(len(self.backend.grading()), n + 1)
        os.environ["FEEDBACK_LANGUAGE"] = "English"
        self.grade()
        self.assertEqual(len(self.backend.grading()), n + 2)

    def test_cache_can_be_switched_off(self):
        self.say_difficulty("easy")
        router = GradingRouter(enable_cache=False)
        router.grade(self.homework())
        router.grade(self.homework())
        self.assertEqual(len(self.backend.grading()), 2)

    def test_result_that_should_have_been_verified_is_not_cached(self):
        os.environ.pop("ANTHROPIC_API_KEY")   # verifier unavailable ...
        os.environ.pop("GEMINI_API_KEY")      # ... and so is every cross-vendor alternative
        self.say_difficulty("easy")
        self.backend.cfg[f"grade_{QWEN_HOST}"] = grade_json(confidence=0.4)
        self.grade()
        self.grade()
        self.assertEqual(len(self.backend.grading(QWEN_HOST)), 2)


if __name__ == "__main__":
    unittest.main()
