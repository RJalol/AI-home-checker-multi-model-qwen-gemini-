"""Shared-assignment support: prompt prefix, cached-token accounting, API validation."""
import unittest
from types import SimpleNamespace

import prompts
from hwchecker.providers.base import build_grading_result, count_or_zero
from hwchecker.pipeline.schema import grading_result_to_schema


class PromptPrefixTests(unittest.TestCase):
    def test_without_assignment_prompt_is_unchanged(self):
        self.assertEqual(prompts.build_user_prompt("be strict"), prompts.build_user_prompt("be strict", "", ""))
        self.assertNotIn("assignment given", prompts.build_user_prompt())

    def test_task_and_key_are_included_and_identical_for_every_student(self):
        a = prompts.build_user_prompt("x", task_text="Solve 2+2", answer_key="4")
        b = prompts.build_user_prompt("x", task_text="Solve 2+2", answer_key="4")
        self.assertEqual(a, b)   # byte-identical -> cacheable prefix
        self.assertIn("Solve 2+2", a)
        self.assertIn("reference answers", a)
        self.assertLess(a.index("Solve 2+2"), a.index("reference answers"))

    def test_blank_values_add_nothing(self):
        self.assertEqual(prompts.build_user_prompt("", "  ", "\n"), prompts.build_user_prompt())


class CachedTokenTests(unittest.TestCase):
    def _result(self, cached):
        return build_grading_result(
            {"classification": "CORRECT", "score": 100}, default_confidence=0.9,
            input_tokens=1000, output_tokens=50, latency_ms=1, provider_name="p", cached_input_tokens=cached,
        )

    def test_cached_tokens_reach_the_schema(self):
        self.assertEqual(grading_result_to_schema(self._result(800))["tokens"],
                         {"input": 1000, "output": 50, "cached": 800})

    def test_missing_or_garbage_counts_become_zero(self):
        for bad in (None, "x", True, -5, SimpleNamespace()):
            self.assertEqual(count_or_zero(bad), 0)
        self.assertEqual(self._result(None).raw_tokens["cached"], 0)
        self.assertEqual(count_or_zero(12.0), 12)


if __name__ == "__main__":
    unittest.main()
