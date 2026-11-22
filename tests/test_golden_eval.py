import json
import tempfile
import unittest
from pathlib import Path

from hwchecker.benchmarks import golden_eval as ge


def write_set(directory: Path, rows):
    (directory / "f.txt").write_text("x")
    path = directory / "set.jsonl"
    path.write_text("\n".join(json.dumps({"file": "f.txt", **row}) for row in rows) + "\n", encoding="utf-8")
    return path


def case(i, teacher, **extra):
    return {"id": f"c{i}", "subject_id": "math", "teacher_score": teacher, **extra}


class GoldenEvalTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)

    def run_set(self, rows, scores):
        cases = ge.load_cases(write_set(self.dir, rows))
        return ge.evaluate(cases, lambda c: scores[c.id])

    def test_metrics(self):
        rows = [case(1, 80), case(2, 60), case(3, 40), case(4, 90)]
        report = self.run_set(rows, {"c1": 80, "c2": 70, "c3": 60, "c4": 70})
        o = report["overall"]
        self.assertEqual(o["graded"], 4)
        self.assertEqual(o["mae"], 12.5)                   # (0 + 10 + 20 + 20) / 4
        self.assertEqual(o["bias"], 2.5)                   # (0 + 10 + 20 - 20) / 4
        self.assertEqual(o["within_10"], 0.5)
        self.assertEqual(o["big_error_rate"], 0.5)
        self.assertEqual(o["false_high_rate"], 0.25)       # only c3 is >= 15 ABOVE the teacher

    def test_breakdown_by_subject_and_tag(self):
        rows = [case(1, 50, tags=["photo"]), case(2, 50, tags=["photo", "uz"]),
                {**case(3, 50), "subject_id": "physics"}]
        report = self.run_set(rows, {"c1": 50, "c2": 60, "c3": 90})
        self.assertEqual(report["breakdown"]["tag:photo"]["mae"], 5.0)
        self.assertEqual(report["breakdown"]["tag:uz"]["cases"], 1)
        self.assertEqual(report["breakdown"]["subject:physics"]["mae"], 40.0)

    def test_failures_are_counted_not_fatal(self):
        cases = ge.load_cases(write_set(self.dir, [case(1, 50), case(2, 50)]))

        def grade(c):
            if c.id == "c2":
                raise RuntimeError("provider down")
            return 50

        report = ge.evaluate(cases, grade)
        self.assertEqual(report["overall"]["failure_rate"], 0.5)
        self.assertEqual(report["errors"][0]["id"], "c2")

    def test_injection_cases(self):
        rows = [case(1, 0, kind="injection", max_score=30), case(2, 0, kind="injection"),
                case(3, 50)]
        report = self.run_set(rows, {"c1": 100, "c2": 5, "c3": 50})
        details = {d["id"]: d for d in report["injection"]["details"]}
        self.assertFalse(details["c1"]["passed"])
        self.assertTrue(details["c2"]["passed"])      # default limit = teacher_score + 10
        self.assertEqual(report["injection"]["pass_rate"], 0.5)
        self.assertEqual(report["overall"]["cases"], 1, "injection cases are not mixed into the accuracy metrics")

    def test_thresholds(self):
        rows = [case(i, 50) for i in range(40)]
        good = self.run_set(rows, {f"c{i}": 52 for i in range(40)})
        self.assertEqual(ge.check_thresholds(good), [])
        bad = self.run_set(rows, {f"c{i}": 90 for i in range(40)})
        problems = " | ".join(ge.check_thresholds(bad))
        for expected in ("mean absolute error", "within 10", "unfairly high"):
            self.assertIn(expected, problems)

    def test_a_tiny_set_cannot_pass(self):
        report = self.run_set([case(1, 50)], {"c1": 50})
        self.assertTrue(any("only 1 normal cases" in p for p in ge.check_thresholds(report)))

    def test_baseline_regression(self):
        rows = [case(i, 50) for i in range(40)]
        before = self.run_set(rows, {f"c{i}": 52 for i in range(40)})
        worse = self.run_set(rows, {f"c{i}": 60 for i in range(40)})
        self.assertEqual(ge.compare_to_baseline(before, before), [])
        self.assertTrue(any("mae got worse" in p for p in ge.compare_to_baseline(worse, before)))

    def test_cli_exit_codes(self):
        rows = [case(i, 50) for i in range(40)]
        path = write_set(self.dir, rows)
        out = self.dir / "report.json"
        self.assertEqual(ge.main([str(path), "--out", str(out)], grade_fn=lambda c: 51), 0)
        self.assertEqual(json.loads(out.read_text())["overall"]["cases"], 40)
        self.assertEqual(ge.main([str(path)], grade_fn=lambda c: 100), 1)
        self.assertEqual(ge.main([str(path), "--baseline", str(out)], grade_fn=lambda c: 70), 1)
        self.assertEqual(ge.main([str(self.dir / "missing.jsonl")]), 2)

    def test_malformed_datasets_are_rejected(self):
        (self.dir / "f.txt").write_text("x")
        for bad in ('{"id": "a"}', '{"id": "a", "subject_id": "math", "file": "f.txt", "teacher_score": 150}',
                    '{"id": "a", "subject_id": "math", "file": "nope.txt", "teacher_score": 5}', "not json"):
            path = self.dir / "bad.jsonl"
            path.write_text(bad + "\n")
            with self.assertRaises(ge.GoldenSetError, msg=bad):
                ge.load_cases(path)
        path.write_text(json.dumps({"id": "a", "subject_id": "math", "file": "f.txt", "teacher_score": 5}) + "\n" * 2
                        + json.dumps({"id": "a", "subject_id": "math", "file": "f.txt", "teacher_score": 5}) + "\n")
        with self.assertRaises(ge.GoldenSetError):
            ge.load_cases(path)


class InjectionSetGeneratorTests(unittest.TestCase):
    def test_generated_set_loads(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "make_injection_set", Path(__file__).resolve().parent.parent / "eval" / "make_injection_set.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp:
            count = module.build(Path(tmp))
            cases = ge.load_cases(Path(tmp) / "injection_cases.jsonl")
            self.assertEqual(len(cases), count)
            self.assertTrue(all(c.kind == "injection" and c.file.suffix == ".docx" for c in cases))


if __name__ == "__main__":
    unittest.main()


from .fakes import FakeSDKTestCase  # noqa: E402


class RealPipelineSmokeTest(FakeSDKTestCase):
    """grade_with_router builds the homework like the worker does and returns the pipeline's score
    (the model calls are the project's fake SDKs, so this checks the wiring, not model quality)."""

    def test_grade_with_router_returns_a_score(self):
        from docx import Document
        path = Path(self.tmp.name) / "answer.docx"
        document = Document()
        document.add_paragraph("Question 1. Solve 2x + 6 = 14. Answer: x = 4")
        document.save(path)
        outcome = ge.grade_with_router(ge.Case(id="x", subject_id="math", file=path, teacher_score=100))
        self.assertIsInstance(outcome.score, float)
        self.assertTrue(0 <= outcome.score <= 100)
        self.assertIsNotNone(outcome.usage)
        self.assertEqual(outcome.usage["call_count"], len(outcome.usage["calls"]))


class UsageReportTests(unittest.TestCase):
    def test_usage_table_totals_and_csv(self):
        import csv
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        directory = Path(tmp.name)
        path = write_set(directory, [case(1, 80), case(2, 60)])

        def call(model, i, o, cost):
            return {"provider": "openai", "model": model, "input_tokens": i, "output_tokens": o,
                    "cached_tokens": 0, "estimated_cost_usd": cost, "latency_ms": 100}

        def usage(calls):
            return {"calls": calls, "call_count": len(calls), "input_tokens": sum(c["input_tokens"] for c in calls),
                    "output_tokens": sum(c["output_tokens"] for c in calls), "cached_tokens": 0,
                    "estimated_cost_usd": sum(c["estimated_cost_usd"] for c in calls), "latency_ms": 100 * len(calls)}

        usages = {"c1": usage([call("a", 1000, 200, 0.01)]), "c2": usage([call("a", 3000, 400, 0.03), call("b", 3000, 300, 0.02)])}
        fn = lambda c: ge.GradeOutcome(70, usages[c.id])
        out_csv = directory / "usage.csv"
        report_path = directory / "r.json"
        code = ge.main([str(path), "--usage-out", str(out_csv), "--out", str(report_path)], grade_fn=fn)
        self.assertIn(code, (0, 1))
        report = json.loads(report_path.read_text())
        totals = report["usage"]["totals"]
        self.assertEqual((totals["homeworks"], totals["calls"], totals["input_tokens"]), (2, 3, 7000))
        self.assertEqual(totals["avg_input_tokens"], 3500)
        self.assertEqual(report["usage"]["by_model"]["openai:a"]["calls"], 2)
        rows = list(csv.DictReader(out_csv.open()))
        self.assertEqual([r["id"] for r in rows], ["c1", "c2"])
        self.assertEqual(rows[1]["calls"], "2")
        self.assertEqual(rows[1]["models"], "openai:a + openai:b")

    def test_plain_float_grade_functions_still_work(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        report = ge.evaluate(ge.load_cases(write_set(Path(tmp.name), [case(1, 80)])), lambda c: 80)
        self.assertNotIn("usage", report)
