"""
hwchecker/benchmarks/compare_candidates.py
Runs the existing benchmark (runner.py -- same prompt, same metrics) for
SEVERAL candidate models and prints one comparison table, so a tier
decision doesn't need hand-editing providers/registry.py for every model.

Nothing here changes providers/registry.py's tier assignments: candidates
are plugged in under temporary "candidate_N" tier names and removed again.
Per the spec (section 10 / constraint 6), the tier assignments in
registry.py should only change after looking at these numbers -- and with
the project owner's approval.

Usage (from the repo root, with the API keys of the providers you want to
test exported / in .env):

    # 1) Plumbing check, NO API keys needed (stub models with known answers):
    python -m hwchecker.benchmarks.compare_candidates DATASET.md --dry-run

    # 2) Quick smoke test on 8 cases of one tier:
    python -m hwchecker.benchmarks.compare_candidates DATASET.md --tier easy --limit 8

    # 3) Full 60-case run of every candidate whose API key is set:
    python -m hwchecker.benchmarks.compare_candidates DATASET.md --tier all --out results.json

    # 4) Any specific models ("provider:model_id" or "provider:model_id@thinking"):
    python -m hwchecker.benchmarks.compare_candidates DATASET.md --models qwen:qwen3.8-flash@off glm:glm-5.3

Candidates whose API key (or SDK) is missing are reported as SKIPPED, not
treated as failures. Model IDs below are the names seen in provider
price lists in Oct 2026 -- if a provider answers 404 / "model not found",
check its current model list and fix the ID here (nothing else changes).
"""

import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

from hwchecker.pipeline.schema import VALID_CLASSIFICATIONS
from hwchecker.providers import registry
from hwchecker.providers.base import AIProvider, GradingResult

from .loader import BenchmarkCase, load_benchmark
from .runner import run_benchmark

# First entry of every tier = what providers/registry.py uses today
# (the baseline the others have to beat). "provider:model@thinking" runs a model with that
# thinking setting (off | low | medium | high | max | token budget); without "@..." the
# provider's own default applies -- which is what the pre-optimisation set-up used, so the
# second entry of a tier is usually "the same model, the old way". "router" is not listed
# (it does difficulty classification, which this correctness benchmark can't judge); neither
# are "hard_vision" (same job as "hard", for photos -- benchmark it through the "hard" entries
# of a vision-capable model) and "audio" (the benchmark cases are text).
CANDIDATES: Dict[str, List[str]] = {
    "easy": ["qwen:qwen3.5-flash@off", "qwen:qwen3.5-flash", "qwen:qwen3.8-flash@off"],
    "medium": [
        "qwen:qwen3.7-plus@low", "qwen:qwen3.7-plus", "gemini:gemini-3.5-flash-lite@low",
        "gemini:gemini-3.5-flash-lite", "gemini:gemini-3.5-flash", "gemini:gemini-3.8-flash",
        "qwen:qwen3.8-flash@off", "claude:claude-haiku-5-5",
    ],
    "hard": ["glm:glm-5.2@high", "glm:glm-5.2", "glm:glm-5.3@high", "gemini:gemini-3.6-flash@high"],
    "verification": ["claude:claude-haiku-5-5", "claude:claude-sonnet-5", "claude:claude-sonnet-5-5"],
    "fallback": ["openai:gpt-5.4-mini", "openai:gpt-6-luna", "openai:gpt-6.1-terra"],
}

DRY_RUN_SPECS = ["stub:always-correct", "stub:always-incorrect", "stub:oracle"]


# ---------------------------------------------------------------------------
# Wrapper that records tokens / latency / cost of every call
# ---------------------------------------------------------------------------
class _RecordingProvider(AIProvider):
    def __init__(self, inner: AIProvider):
        self.inner = inner
        self.name = getattr(inner, "name", "unknown")
        self._lock = threading.Lock()
        self.calls = 0
        self.in_tokens = 0
        self.out_tokens = 0
        self.latency_ms = 0
        self.cost = 0.0

    def grade_homework(self, content: dict, rubric: dict) -> GradingResult:
        result = self.inner.grade_homework(content, rubric)  # AIProviderError propagates to runner
        tin = int(result.raw_tokens.get("input", 0) or 0)
        tout = int(result.raw_tokens.get("output", 0) or 0)
        with self._lock:
            self.calls += 1
            self.in_tokens += tin
            self.out_tokens += tout
            self.latency_ms += int(result.latency_ms or 0)
            self.cost += self.inner.estimate_cost(tin, tout)
        return result

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        return self.inner.estimate_cost(input_tokens, output_tokens)


# ---------------------------------------------------------------------------
# Stub models for --dry-run (no network, no keys): known behaviour, so the
# metric math can be checked by hand.
# ---------------------------------------------------------------------------
class _StubProvider(AIProvider):
    def __init__(self, mode: str, cases: List[BenchmarkCase]):
        self.name = f"stub-{mode}"
        self.mode = mode
        self.cases = cases

    def _find_case(self, user_prompt: str) -> Optional[BenchmarkCase]:
        for c in self.cases:
            if c.student_answer in user_prompt and c.problem_text in user_prompt:
                return c
        return None

    def grade_homework(self, content: dict, rubric: dict) -> GradingResult:
        tokens = {"input": 5000, "output": 1500}
        if self.mode == "always-correct":
            return GradingResult(True, "CORRECT", 100.0, raw_tokens=tokens, latency_ms=1, provider_name=self.name)
        if self.mode == "always-incorrect":
            return GradingResult(False, "INCORRECT", 0.0, raw_tokens=tokens, latency_ms=1, provider_name=self.name)
        case = self._find_case(content.get("user_prompt", ""))
        if case is None:
            raise AssertionError("stub oracle could not find the case in the prompt")
        is_ok = case.expected_classification == "CORRECT"
        return GradingResult(
            is_ok, case.expected_classification, float(case.expected_score),
            error_type=None if is_ok else "stub-error", raw_tokens=tokens, latency_ms=1, provider_name=self.name,
        )

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        return input_tokens / 1e6 * 0.10 + output_tokens / 1e6 * 0.40


# ---------------------------------------------------------------------------
@dataclass
class Row:
    spec: str
    tiers: str
    status: str = "ok"          # "ok" | "SKIPPED"
    note: str = ""
    n: int = 0                  # cases attempted
    valid: int = 0              # answered with a valid classification
    errors: int = 0             # API/JSON failures
    false_positive_rate: float = 0.0
    false_negative_rate: float = 0.0
    classification_accuracy: float = 0.0
    correct_answer_accuracy: float = 0.0
    partial_credit_accuracy: float = 0.0
    error_detection_accuracy: float = 0.0
    score_mae: float = 0.0
    avg_input_tokens: float = 0.0
    avg_output_tokens: float = 0.0
    avg_latency_s: float = 0.0
    cost_per_1000_calls: float = 0.0


def _build_provider(spec: str, cases: List[BenchmarkCase]) -> AIProvider:
    kind, _, model = spec.partition(":")
    model, _, thinking = model.partition("@")
    if not model:
        raise ValueError(f"'{spec}' must look like provider:model_id or provider:model_id@thinking")
    if kind == "stub":
        return _StubProvider(model, cases)
    return registry._make(kind, model, thinking or None)  # the registry's own lazy factory


def _run_one(spec: str, tiers: str, tier_name: str, rec: _RecordingProvider, md_path, limit) -> Row:
    report = run_benchmark(md_path, provider_tier=tier_name, limit=limit)
    n = report.total_cases
    valid = sum(1 for r in report.results if r.error is None and r.predicted_classification in VALID_CLASSIFICATIONS)
    errors = sum(1 for r in report.results if r.error is not None)
    calls = max(rec.calls, 1)
    first_error = next((r.error for r in report.results if r.error), "")
    return Row(
        spec=spec, tiers=tiers, n=n, valid=valid, errors=errors,
        note=(first_error[:110] if first_error else ""),
        false_positive_rate=report.false_positive_rate,
        false_negative_rate=report.false_negative_rate,
        classification_accuracy=report.classification_accuracy,
        correct_answer_accuracy=report.correct_answer_accuracy,
        partial_credit_accuracy=report.partial_credit_accuracy,
        error_detection_accuracy=report.error_detection_accuracy,
        score_mae=report.score_mae,
        avg_input_tokens=rec.in_tokens / calls,
        avg_output_tokens=rec.out_tokens / calls,
        avg_latency_s=rec.latency_ms / calls / 1000,
        cost_per_1000_calls=rec.cost / calls * 1000,
    )


def run_comparison(md_path, spec_to_tiers: Dict[str, str], limit: Optional[int] = None, parallel: int = 1) -> List[Row]:
    cases = load_benchmark(md_path)
    if limit:
        cases = cases[:limit]
    if not cases:
        raise SystemExit(f"No benchmark cases could be parsed from {md_path}")

    rows: Dict[str, Row] = {}
    jobs = []
    # Build everything up front (in the main thread) so missing keys / SDKs
    # surface immediately and tier registration isn't racing the workers.
    for i, (spec, tiers) in enumerate(spec_to_tiers.items()):
        try:
            provider = _build_provider(spec, cases)
        except Exception as exc:  # missing key, missing SDK, bad spec
            rows[spec] = Row(spec=spec, tiers=tiers, status="SKIPPED", note=f"{type(exc).__name__}: {str(exc)[:100]}")
            continue
        rec = _RecordingProvider(provider)
        tier_name = f"candidate_{i}"
        registry.PROVIDER_REGISTRY[tier_name] = (lambda r=rec: r)
        jobs.append((spec, tiers, tier_name, rec))

    try:
        def work(job):
            spec, tiers, tier_name, rec = job
            return _run_one(spec, tiers, tier_name, rec, md_path, limit)

        with ThreadPoolExecutor(max_workers=max(1, parallel)) as pool:
            for row in pool.map(work, jobs):
                rows[row.spec] = row
    finally:
        for _, _, tier_name, _ in jobs:
            registry.PROVIDER_REGISTRY.pop(tier_name, None)
        registry.reset_registry()

    return [rows[s] for s in spec_to_tiers]  # keep requested order


def _fmt_table(rows: List[Row], baselines: set) -> str:
    head = f"{'model':<36}{'tier(s)':<22}{'valid':>7}{'FPR':>8}{'FNR':>8}{'ClsAcc':>8}{'Partial':>8}{'MAE':>7}{'$/1k calls':>11}{'lat(s)':>8}"
    out = [head, "-" * len(head)]
    for r in rows:
        name = r.spec + (" *" if r.spec in baselines else "")
        if r.status != "ok":
            out.append(f"{name:<36}{r.tiers:<22}  SKIPPED -- {r.note}")
            continue
        invalid = r.n - r.valid - r.errors
        if r.valid == 0:
            # Without a single usable answer FPR would print as 0.0% ("perfect") -- never show that.
            out.append(f"{name:<36}{r.tiers:<22}{r.valid:>3}/{r.n:<3}  NO VALID ANSWERS -- metrics not meaningful")
        else:
            out.append(
                f"{name:<36}{r.tiers:<22}{r.valid:>3}/{r.n:<3}{r.false_positive_rate:>8.1%}{r.false_negative_rate:>8.1%}"
                f"{r.classification_accuracy:>8.1%}{r.partial_credit_accuracy:>8.1%}{r.score_mae:>7.1f}"
                f"{r.cost_per_1000_calls:>11.2f}{r.avg_latency_s:>8.1f}"
            )
        if r.errors:
            out.append(f"{'':<36}  ! {r.errors} call(s) failed, e.g.: {r.note}")
        if invalid:
            out.append(f"{'':<36}  ! {invalid} answer(s) came back with an invalid classification label")
    out.append("")
    out.append("* = what providers/registry.py uses today (the baseline to beat).")
    out.append("FPR = share of NOT-correct answers the model called CORRECT (most important; lower is better).")
    out.append("FPR/FNR are computed over 'valid' answers only -- a model with many failed calls can look better than it is.")
    out.append("With ~45 not-correct cases one extra miss moves FPR by ~2 points: treat small gaps as noise.")
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Compare candidate models on the benchmark dataset.")
    ap.add_argument("dataset", help="path to math_benchmark_dataset_v1.md")
    ap.add_argument("--tier", default="all", choices=[*CANDIDATES, "all"], help="which tier's candidates to run")
    ap.add_argument("--models", nargs="+", metavar="PROVIDER:MODEL[@THINKING]", help="run exactly these instead of the built-in candidates")
    ap.add_argument("--limit", type=int, help="only the first N cases (smoke test)")
    ap.add_argument("--parallel", type=int, default=1, help="candidates to run at the same time (mind provider rate limits)")
    ap.add_argument("--dry-run", action="store_true", help="use stub models; needs no API keys, checks the plumbing and metric math")
    ap.add_argument("--out", help="also write all results to this JSON file")
    args = ap.parse_args(argv)

    spec_to_tiers: Dict[str, str] = {}
    baselines = set()
    if args.dry_run:
        spec_to_tiers = {s: "dry-run" for s in DRY_RUN_SPECS}
    elif args.models:
        spec_to_tiers = {s: "custom" for s in args.models}
    else:
        chosen = CANDIDATES if args.tier == "all" else {args.tier: CANDIDATES[args.tier]}
        for tier, specs in chosen.items():
            baselines.add(specs[0])
            for s in specs:
                spec_to_tiers[s] = (spec_to_tiers[s] + "," + tier) if s in spec_to_tiers else tier

    rows = run_comparison(args.dataset, spec_to_tiers, limit=args.limit, parallel=args.parallel)
    print(_fmt_table(rows, baselines))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump([asdict(r) for r in rows], fh, ensure_ascii=False, indent=2)
        print(f"\nResults written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
