import os
import unittest
from unittest import mock

import prompts
from hwchecker.language import DEFAULT_FEEDBACK_LANGUAGE, get_feedback_language, language_instruction
from hwchecker.pipeline.crm_adapter import adapt_router_result
from hwchecker.pipeline.schema import SchemaValidationError, grading_result_to_schema, validate
from hwchecker.providers.base import GradingResult
from hwchecker.subjects.registry import get_subject


def _schema(**overrides):
    payload = {
        "classification": "PARTIALLY_CORRECT", "score": 72.0, "max_score": 100.0, "error_type": "Calculation Error",
        "error_location": "step 3", "feedback": "Overall comment", "sub_scores": {}, "confidence": 0.9,
        "provider": "gemini-3.5-flash", "tokens": {"input": 10, "output": 5}, "latency_ms": 12,
    }
    payload.update(overrides)
    return payload


class LanguageTests(unittest.TestCase):
    def setUp(self):
        p = mock.patch.dict(os.environ, {}, clear=False)
        p.start()
        self.addCleanup(p.stop)
        os.environ.pop("FEEDBACK_LANGUAGE", None)

    def test_default_is_uzbek_latin(self):
        self.assertEqual(get_feedback_language(), DEFAULT_FEEDBACK_LANGUAGE)
        self.assertEqual(DEFAULT_FEEDBACK_LANGUAGE, "Uzbek (Latin script)")

    def test_env_override_and_auto(self):
        os.environ["FEEDBACK_LANGUAGE"] = "Russian"
        self.assertIn("in Russian", language_instruction())
        os.environ["FEEDBACK_LANGUAGE"] = "auto"
        self.assertIn("in the language the student's work is written in", language_instruction())

    def test_every_hwchecker_prompt_asks_for_the_language_and_the_breakdown(self):
        for subject, skill in [("math", None), ("physics", None), ("chemistry", None), ("biology", None),
                               ("history", None), ("english", "writing"), ("russian", "reading"),
                               ("uzbek", "listening"), ("turkish", "writing")]:
            prompt = get_subject(subject).build_system_prompt(skill)
            self.assertIn("Uzbek (Latin script)", prompt, subject)
            self.assertIn("PER-QUESTION BREAKDOWN", prompt, subject)
            self.assertNotIn("{subject_id}", prompt)

    def test_breakdown_can_be_left_out(self):
        prompt = get_subject("math").build_system_prompt(per_question=False)
        self.assertNotIn("PER-QUESTION BREAKDOWN", prompt)
        self.assertIn("RESPONSE LANGUAGE", prompt)

    def test_original_engine_prompt_also_names_the_language(self):
        for prompt in (prompts.build_system_prompt("math"), prompts.build_system_prompt(language_key="uzbek", skill_key="writing")):
            self.assertIn("Uzbek (Latin script)", prompt)
            self.assertLess(prompt.index("RESPONSE LANGUAGE"), prompt.index("# OUTPUT FORMAT"))


class SchemaTests(unittest.TestCase):
    def test_questions_and_summary_pass_through_only_when_present(self):
        bare = grading_result_to_schema(GradingResult(is_correct=False, classification="INCORRECT", score=10, raw_tokens={"input": 1, "output": 1}))
        self.assertNotIn("questions", bare)
        rich = grading_result_to_schema(GradingResult(
            is_correct=False, classification="INCORRECT", score=10, raw_tokens={"input": 1, "output": 1},
            questions=[{"question_number": 1}], summary={"strengths": ["x"]},
        ))
        self.assertEqual(rich["questions"], [{"question_number": 1}])

    def test_malformed_optional_keys_are_rejected(self):
        with self.assertRaises(SchemaValidationError):
            validate(_schema(questions="nope"))
        with self.assertRaises(SchemaValidationError):
            validate(_schema(summary=[]))


class CrmAdapterTests(unittest.TestCase):
    def test_per_question_breakdown_becomes_one_entry_each(self):
        adapted = adapt_router_result(_schema(
            questions=[
                {"question_number": 1, "question": "2+2", "student_answer": "4", "correct_answer": "4", "score": 100,
                 "mistakes": [], "error_categories": [], "analysis": "ok", "suggestions": []},
                {"question_number": 2, "question": "3*3", "student_answer": "6", "correct_answer": "9", "score": 10,
                 "mistakes": ["wrong"], "error_categories": ["Calculation Error"], "analysis": "bad", "suggestions": ["practise"]},
            ],
            summary={"strengths": ["adding"], "weaknesses": ["times tables"]},
        ))
        self.assertEqual(adapted["overall_score"], 72.0)
        self.assertEqual(adapted["grade"], "Good")                  # band label, not "PARTIALLY_CORRECT"
        self.assertEqual(adapted["classification"], "PARTIALLY_CORRECT")
        self.assertEqual([q["score"] for q in adapted["questions"]], [100, 10])
        self.assertEqual(adapted["questions"][1]["expected_solution"], "9")
        self.assertEqual(adapted["summary"]["weaknesses"], ["times tables"])
        self.assertEqual(adapted["summary"]["recommendations"], [])

    def test_result_without_breakdown_is_one_question(self):
        adapted = adapt_router_result(_schema())
        self.assertEqual(len(adapted["questions"]), 1)
        q = adapted["questions"][0]
        self.assertEqual((q["analysis"], q["mistakes"], q["error_categories"], q["score"]), ("Overall comment", ["step 3"], ["Calculation Error"], 72.0))

    def test_adapted_result_satisfies_the_report_generator_contract(self):
        from utils import parse_and_validate_json  # the keys the original engine's output is validated against
        import json

        adapted = adapt_router_result(_schema(questions=[{"question_number": 1, "score": 50}]))
        parse_and_validate_json(json.dumps(adapted))   # raises InvalidModelResponseError if a required key is missing

    def test_cache_flag_is_carried(self):
        self.assertTrue(adapt_router_result(_schema(cached=True))["cached"])
        self.assertNotIn("cached", adapt_router_result(_schema()))


if __name__ == "__main__":
    unittest.main()
