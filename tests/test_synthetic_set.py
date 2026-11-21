import json
import unittest
from pathlib import Path

from hwchecker.benchmarks import golden_eval as ge

ROOT = Path(__file__).resolve().parent.parent
SET = ROOT / "eval" / "synthetic" / "golden_synthetic.jsonl"
SUBJECTS = {"math", "physics", "chemistry", "biology", "history", "english", "russian", "uzbek", "turkish"}


class SyntheticSetTests(unittest.TestCase):
    def test_loads_and_covers_every_subject(self):
        cases = ge.load_cases(SET)
        self.assertGreaterEqual(len(cases), 72)
        self.assertEqual({c.subject_id for c in cases}, SUBJECTS)
        self.assertGreaterEqual(len([c for c in cases if c.kind == "normal"]), ge.DEFAULT_THRESHOLDS["min_normal_cases"])

    def test_has_every_kind_of_case(self):
        cases = ge.load_cases(SET)
        tags = {t for c in cases for t in c.tags}
        for needed in ("with_key", "no_key", "docx", "jpg", "blank", "adversarial_no_working", "adversarial_unit_or_sign"):
            self.assertIn(needed, tags)
        scores = {c.teacher_score for c in cases}
        self.assertIn(0.0, scores)
        self.assertIn(100.0, scores)

    def test_expected_score_matches_details(self):
        details = json.loads((SET.parent / "expected_details.json").read_text(encoding="utf-8"))
        by_id = {c.id: c for c in ge.load_cases(SET)}
        for row in details:
            mean = round(sum(q["score"] for q in row["questions"]) / len(row["questions"]))
            self.assertEqual(by_id[row["id"]].teacher_score, mean, row["id"])

    def test_generator_is_deterministic(self):
        import tempfile
        import importlib.util
        spec = importlib.util.spec_from_file_location("mk", ROOT / "eval" / "make_synthetic_set.py")
        mk = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mk)
        with tempfile.TemporaryDirectory() as tmp:
            mk.build(Path(tmp))
            self.assertEqual((Path(tmp) / "golden_synthetic.jsonl").read_text(encoding="utf-8"), SET.read_text(encoding="utf-8"))

    def test_perfect_grader_passes_the_gate(self):
        cases = ge.load_cases(SET)
        report = ge.evaluate(cases, lambda c: c.teacher_score)
        self.assertEqual(report["overall"]["mae"], 0)
        self.assertEqual(ge.check_thresholds(report), [])


if __name__ == "__main__":
    unittest.main()
