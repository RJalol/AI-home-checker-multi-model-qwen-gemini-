"""
hwchecker/benchmarks/runner.py
Runs the 60-case math benchmark (loader.py) through the grading
pipeline and computes the metrics from spec section 10:

  - Classification Accuracy: overall % where predicted == expected classification
  - Correct Answer Accuracy: among the 15 "Correct" cases, % correctly
    classified as CORRECT (measures false negatives)
  - Partial Credit Accuracy: among the 15 "Partially Correct" cases, %
    where |predicted_score - expected_score| <= 10
  - Error Flagged Rate (field `error_detection_accuracy`, name kept for compatibility):
    among "Incorrect"/"Misconception" cases, % where ANY error_type was reported. It does NOT
    check that the error identified is the right one -- a grader that labels every error with an
    unrelated string scores 100% -- so read it as "did it notice something", never as accuracy.
  - Error Category Accuracy: among those cases whose dataset "Error Type" falls in one of the
    coarse families below (arithmetic / conceptual / procedural / algebraic), % where the reported
    error_type names the same family. Free text is matched by keyword (English + Uzbek stems), so
    it is a coarse check, not an exact one.
  - False Positive Rate: % of non-CORRECT cases the model called CORRECT
    (the single most important metric per spec section 10 -- an unfair
    "correct" grade is the worst failure mode for a homework checker)
  - False Negative Rate: % of CORRECT cases the model did NOT call CORRECT
  - Score Agreement (MAE): mean absolute error between predicted and expected score

Per spec section 10: "Hech qanday model/tier tanlovi benchmarkdan
o'tmasdan production'ga chiqarilmasin" -- run this BEFORE trusting the
PROVIDER_REGISTRY tier assignments (providers/registry.py) in production.
"""

from dataclasses import dataclass, field
from typing import List, Optional

from hwchecker.pipeline.schema import VALID_CLASSIFICATIONS
from hwchecker.providers.base import AIProviderError
from hwchecker.providers.registry import get_provider

from .loader import BenchmarkCase, load_benchmark


@dataclass
class CaseResult:
    case: BenchmarkCase
    predicted_classification: Optional[str] = None
    predicted_score: Optional[float] = None
    predicted_error_type: Optional[str] = None
    provider_used: str = ""
    error: Optional[str] = None


@dataclass
class BenchmarkReport:
    total_cases: int
    results: List[CaseResult] = field(default_factory=list)
    classification_accuracy: float = 0.0
    correct_answer_accuracy: float = 0.0
    partial_credit_accuracy: float = 0.0
    error_detection_accuracy: float = 0.0     # = Error Flagged Rate (see module docstring)
    error_category_accuracy: float = 0.0
    error_category_cases: int = 0             # how many cases error_category_accuracy is based on
    false_positive_rate: float = 0.0
    false_negative_rate: float = 0.0
    score_mae: float = 0.0

    def summary_text(self) -> str:
        return (
            f"Benchmark results ({self.total_cases} cases):\n"
            f"  Classification Accuracy:    {self.classification_accuracy:.1%}\n"
            f"  Correct Answer Accuracy:    {self.correct_answer_accuracy:.1%}\n"
            f"  Partial Credit Accuracy:    {self.partial_credit_accuracy:.1%}\n"
            f"  Error Flagged Rate:         {self.error_detection_accuracy:.1%}  (any error_type given; NOT proof it was the right one)\n"
            f"  Error Category Accuracy:    {self.error_category_accuracy:.1%}  (coarse family match, {self.error_category_cases} cases)\n"
            f"  False Positive Rate:        {self.false_positive_rate:.1%}  (lower is better -- most important metric)\n"
            f"  False Negative Rate:        {self.false_negative_rate:.1%}\n"
            f"  Score Agreement (MAE):      {self.score_mae:.1f} points\n"
        )


PROMPT_TEMPLATE = """Grade this math homework response.

Problem: {problem}
Gold/correct answer: {gold_answer}
Student's response: {student_answer}
"""


def _grade_one_case(case: BenchmarkCase, provider_tier: str) -> CaseResult:
    provider = get_provider(provider_tier)
    content = {
        "system_prompt": (
            "You are a strict math grading assistant. Classify the student's response as "
            "CORRECT, INCORRECT, PARTIALLY_CORRECT, or INCORRECT_WITH_MISCONCEPTION, and give "
            "a 0-100 score. Return ONLY this JSON: "
            '{"classification": "...", "score": 0-100, "error_type": "string or null", '
            '"error_location": "string or null", "feedback": "...", "sub_scores": {}, "confidence": 0.0-1.0}'
        ),
        "user_prompt": PROMPT_TEMPLATE.format(
            problem=case.problem_text, gold_answer=case.gold_answer, student_answer=case.student_answer
        ),
    }

    try:
        result = provider.grade_homework(content, rubric={})
        return CaseResult(
            case=case,
            predicted_classification=result.classification,
            predicted_score=result.score,
            predicted_error_type=result.error_type,
            provider_used=result.provider_name,
        )
    except AIProviderError as exc:
        return CaseResult(case=case, error=str(exc))


def run_benchmark(md_path, provider_tier: str = "medium", limit: Optional[int] = None) -> BenchmarkReport:
    """
    Runs every parsed case in md_path through `provider_tier` (default:
    "medium", i.e. whatever's assigned to that tier in
    providers/registry.py) and computes accuracy metrics. `limit` caps
    how many cases to run, for a quick smoke test before committing to
    the full 60-case (and therefore full-cost) run.
    """
    cases = load_benchmark(md_path)
    if limit:
        cases = cases[:limit]

    results = [_grade_one_case(case, provider_tier) for case in cases]
    return _compute_metrics(results)


_ERROR_FAMILIES = {
    "arithmetic": ("arithmetic", "calculation", "computation", "hisob", "arifmetik"),
    "conceptual": ("concept", "misconception", "misunderstand", "tushun", "tushuncha"),
    "procedural": ("procedur", "method", "step", "jarayon", "usul"),
    "algebraic": ("algebra", "sign", "belgi", "simplif"),
}


def _error_family(text: Optional[str]) -> Optional[str]:
    """The coarse family a free-text error type belongs to; None if none (or several) match."""
    lowered = (text or "").lower()
    hits = {family for family, stems in _ERROR_FAMILIES.items() if any(stem in lowered for stem in stems)}
    return hits.pop() if len(hits) == 1 else None


def _compute_metrics(results: List[CaseResult]) -> BenchmarkReport:
    valid = [r for r in results if r.error is None and r.predicted_classification in VALID_CLASSIFICATIONS]
    n = len(results)

    correct_classifications = sum(1 for r in valid if r.predicted_classification == r.case.expected_classification)
    classification_accuracy = correct_classifications / n if n else 0.0

    correct_cases = [r for r in valid if r.case.expected_classification == "CORRECT"]
    correct_answer_accuracy = (
        sum(1 for r in correct_cases if r.predicted_classification == "CORRECT") / len(correct_cases)
        if correct_cases else 0.0
    )

    partial_cases = [r for r in valid if r.case.expected_classification == "PARTIALLY_CORRECT"]
    partial_credit_accuracy = (
        sum(1 for r in partial_cases if abs((r.predicted_score or 0) - r.case.expected_score) <= 10) / len(partial_cases)
        if partial_cases else 0.0
    )

    error_cases = [r for r in valid if r.case.expected_classification in ("INCORRECT", "INCORRECT_WITH_MISCONCEPTION")]
    error_detection_accuracy = (
        sum(1 for r in error_cases if r.predicted_error_type) / len(error_cases) if error_cases else 0.0
    )

    family_cases = [r for r in error_cases if _error_family(r.case.expected_error_type)]
    error_category_accuracy = (
        sum(1 for r in family_cases if _error_family(r.predicted_error_type) == _error_family(r.case.expected_error_type))
        / len(family_cases) if family_cases else 0.0
    )

    non_correct_cases = [r for r in valid if r.case.expected_classification != "CORRECT"]
    false_positive_rate = (
        sum(1 for r in non_correct_cases if r.predicted_classification == "CORRECT") / len(non_correct_cases)
        if non_correct_cases else 0.0
    )

    false_negative_rate = (
        sum(1 for r in correct_cases if r.predicted_classification != "CORRECT") / len(correct_cases)
        if correct_cases else 0.0
    )

    scored = [r for r in valid if r.predicted_score is not None]
    score_mae = (
        sum(abs(r.predicted_score - r.case.expected_score) for r in scored) / len(scored) if scored else 0.0
    )

    return BenchmarkReport(
        total_cases=n,
        results=results,
        classification_accuracy=classification_accuracy,
        correct_answer_accuracy=correct_answer_accuracy,
        partial_credit_accuracy=partial_credit_accuracy,
        error_detection_accuracy=error_detection_accuracy,
        error_category_accuracy=error_category_accuracy,
        error_category_cases=len(family_cases),
        false_positive_rate=false_positive_rate,
        false_negative_rate=false_negative_rate,
        score_mae=score_mae,
    )


if __name__ == "__main__":
    import sys

    md_path = sys.argv[1] if len(sys.argv) > 1 else "/mnt/project/claude_math_benchmark_dataset_v1.md"
    tier = sys.argv[2] if len(sys.argv) > 2 else "medium"
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else None

    report = run_benchmark(md_path, provider_tier=tier, limit=limit)
    print(report.summary_text())
