"""
hwchecker/benchmarks/loader.py
Parses claude_math_benchmark_dataset_v1.md (the project's 60-test-case
benchmark, 15 problems x 4 student responses) into structured test
cases runnable through the grading pipeline. Per spec section 10:
"Hech qanday model/tier tanlovi benchmarkdan o'tmasdan production'ga
chiqarilmasin."
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class BenchmarkCase:
    problem_id: str          # "E1", "M3", "H5", ...
    difficulty: str          # "Easy" | "Medium" | "Hard"
    topic: str
    problem_text: str
    gold_answer: str
    student_case_label: str  # "Correct" | "Incorrect" | "Partially Correct" | "Misconception"
    student_answer: str
    expected_classification: str  # "CORRECT" | "INCORRECT" | "PARTIALLY_CORRECT" | "INCORRECT_WITH_MISCONCEPTION"
    expected_score: int
    expected_error_type: str = ""  # the dataset's free-text "Error Type" for this response ("" / "None" = no error)


_PROBLEM_HEADER_RE = re.compile(r"^## (\S+)\s*$", re.MULTILINE)
_STUDENT_HEADER_RE = re.compile(r"^### Student Response \d+ — (.+?)\s*$", re.MULTILINE)
_STUDENT_ANSWER_RE = re.compile(r"\*\*Student Answer:\*\*\s*(.+?)(?=\n\*\*Expected Classification)", re.DOTALL)
_CLASSIFICATION_RE = re.compile(r"\*\*Expected Classification:\*\*\s*`?([A-Z_]+)`?")
_SCORE_RE = re.compile(r"\*\*Expected Score:\*\*\s*(\d+)")
_ERROR_TYPE_RE = re.compile(r"\*\*Error Type:\*\*\s*(.+)")


def _parse_problem_block(problem_id: str, block_text: str) -> List[BenchmarkCase]:
    difficulty_m = re.search(r"\*\*Difficulty:\*\*\s*(\S+)", block_text)
    topic_m = re.search(r"\*\*Topic:\*\*\s*(.+)", block_text)
    problem_m = re.search(r"\*\*Problem:\*\*\s*\n(.+?)\n\n", block_text, re.DOTALL)
    gold_m = re.search(r"\*\*Gold Answer:\*\*\s*(.+)", block_text)

    difficulty = difficulty_m.group(1).strip() if difficulty_m else ""
    topic = topic_m.group(1).strip() if topic_m else ""
    problem_text = problem_m.group(1).strip() if problem_m else ""
    gold_answer = gold_m.group(1).strip() if gold_m else ""

    cases = []
    student_sections = _STUDENT_HEADER_RE.split(block_text)
    # student_sections alternates: [preamble, label1, body1, label2, body2, ...]
    for i in range(1, len(student_sections), 2):
        label = student_sections[i].strip()
        body = student_sections[i + 1]

        answer_m = _STUDENT_ANSWER_RE.search(body)
        classification_m = _CLASSIFICATION_RE.search(body)
        score_m = _SCORE_RE.search(body)
        if not (answer_m and classification_m and score_m):
            continue

        student_answer = answer_m.group(1).strip().strip('"')
        error_type_m = _ERROR_TYPE_RE.search(body)
        expected_error_type = error_type_m.group(1).strip() if error_type_m else ""
        if expected_error_type.lower() in ("none", "-", "n/a"):
            expected_error_type = ""
        cases.append(BenchmarkCase(
            problem_id=problem_id,
            difficulty=difficulty,
            topic=topic,
            problem_text=problem_text,
            gold_answer=gold_answer,
            student_case_label=label,
            student_answer=student_answer,
            expected_classification=classification_m.group(1).strip(),
            expected_score=int(score_m.group(1)),
            expected_error_type=expected_error_type,
        ))
    return cases


def load_benchmark(md_path) -> List[BenchmarkCase]:
    """Parses every problem/student-response pair in the benchmark markdown into BenchmarkCase objects."""
    text = Path(md_path).read_text(encoding="utf-8")

    # Only look at "QISM 1/2/3" content (individual problem sections),
    # not "QISM 4" (the summary table, which would double-count cases).
    qism4_split = re.split(r"^# QISM 4", text, maxsplit=1, flags=re.MULTILINE)
    body = qism4_split[0]

    headers = list(_PROBLEM_HEADER_RE.finditer(body))
    cases = []
    for i, m in enumerate(headers):
        problem_id = m.group(1)
        start = m.end()
        end = headers[i + 1].start() if i + 1 < len(headers) else len(body)
        block = body[start:end]
        cases.extend(_parse_problem_block(problem_id, block))
    return cases
