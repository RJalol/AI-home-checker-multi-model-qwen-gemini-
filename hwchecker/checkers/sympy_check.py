"""
hwchecker/checkers/sympy_check.py
Deterministic (no LLM, $0) final-answer verification for math/physics
problems with a clean numeric or symbolic gold answer. Used as
SubjectProfile.deterministic_checker for math (subjects/math.py).

This is intentionally conservative: it only returns a result when it is
HIGHLY confident the comparison is meaningful (both the gold answer and
the extracted student answer parse cleanly as the same kind of
expression). Any parsing ambiguity, and it returns None so the homework
falls through to the router/grading_router.py's normal LLM-based path --
a wrong "confident" auto-grade would be far worse than just deferring to
the LLM tier.
"""

import ast
import re
from typing import Optional

from hwchecker.providers.base import GradingResult


def _try_import_sympy():
    try:
        import sympy
        from sympy.parsing.sympy_parser import convert_xor, parse_expr, standard_transformations

        # convert_xor: "2^3" must mean 2**3 (Python's ^ is XOR, which silently turned
        # 2^3 into 1 and let a wrong answer match). The two-name return is kept for callers.
        def parse(text):
            return parse_expr(text, evaluate=True, transformations=standard_transformations + (convert_xor,))

        return sympy, parse
    except ImportError:
        return None, None


# ---- guard against hostile input (the student's text is attacker-controlled) -------------------
# sympy would happily try to evaluate 9**9**9**9 or factorial(99999999) and hang a worker. Before
# anything reaches sympy the text is parsed as a plain Python expression and only a tiny,
# bounded grammar is allowed: + - * / ** (small numeric exponents), single-letter variables,
# pi/e, and a few functions with small arguments. Anything else -> None (the AI tiers decide).
_ALLOWED_FUNCTIONS = {"sqrt", "sin", "cos", "tan", "log", "ln", "exp", "abs", "factorial"}
_ALLOWED_CONSTANTS = {"pi", "e"}
_MAX_CONSTANT = 10 ** 12
_MAX_EXPONENT = 100
_MAX_FACTORIAL_ARG = 100
_MAX_NODES = 120


def _small_constant(node, limit) -> bool:
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        return _small_constant(node.operand, limit)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return abs(node.value) <= limit
    return False


def _is_safe_expression(text: str) -> bool:
    try:
        tree = ast.parse(text.replace("^", "**"), mode="eval")
    except (SyntaxError, ValueError, MemoryError, RecursionError):
        return False
    nodes = list(ast.walk(tree))
    if len(nodes) > _MAX_NODES:
        return False
    for node in nodes:
        if isinstance(node, (ast.Expression, ast.Load, ast.operator, ast.unaryop)):
            if isinstance(node, ast.operator) and not isinstance(node, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)):
                return False
            continue
        if isinstance(node, ast.BinOp):
            if isinstance(node.op, ast.Pow):
                exponent = node.right
                simple_fraction = (
                    isinstance(exponent, ast.BinOp) and isinstance(exponent.op, ast.Div)
                    and _small_constant(exponent.left, _MAX_EXPONENT) and _small_constant(exponent.right, _MAX_EXPONENT)
                )
                if not (_small_constant(exponent, _MAX_EXPONENT) or simple_fraction or isinstance(exponent, ast.Name)):
                    return False   # a tower like 9**9**9, or a huge/unknown exponent
            continue
        if isinstance(node, ast.UnaryOp):
            if not isinstance(node.op, (ast.USub, ast.UAdd)):
                return False
            continue
        if isinstance(node, ast.Constant):
            if not isinstance(node.value, (int, float)) or isinstance(node.value, bool) or abs(node.value) > _MAX_CONSTANT:
                return False
            continue
        if isinstance(node, ast.Name):
            if not (len(node.id) == 1 and node.id.isalpha() and node.id.isascii()) and node.id not in _ALLOWED_CONSTANTS:
                if node.id not in _ALLOWED_FUNCTIONS:   # a function name is only valid as the callee
                    return False
            continue
        if isinstance(node, ast.Call):
            if not (isinstance(node.func, ast.Name) and node.func.id in _ALLOWED_FUNCTIONS and len(node.args) == 1 and not node.keywords):
                return False
            if node.func.id == "factorial" and not (
                _small_constant(node.args[0], _MAX_FACTORIAL_ARG) or isinstance(node.args[0], ast.Name)
            ):
                return False
            continue
        return False   # anything else: attribute access, subscripts, comprehensions, lambdas, strings...
    return True


_NUMBER_OR_EXPR_RE = re.compile(r"^[\s\d\.\-\+\*/\^\(\)xXyYzZa-zA-Z=,]+$")


def _looks_parseable(text: str) -> bool:
    """Cheap pre-filter: only attempt SymPy parsing on short, formula-
    like strings -- avoids wasting time (and risking weird parses) on
    long prose that happens to contain a number."""
    text = text.strip()
    if not text or len(text) > 80:
        return False
    return bool(_NUMBER_OR_EXPR_RE.match(text))


def check(homework: dict) -> Optional[GradingResult]:
    """
    homework: {"gold_answer": str, "student_answer": str, ...}
    Returns a GradingResult if both answers parse as equivalent (or
    unambiguously different) symbolic/numeric expressions, else None
    (meaning: fall through to the LLM tiers).
    """
    sympy, parse = _try_import_sympy()
    if sympy is None:
        return None  # sympy not installed -- fall through, don't crash the pipeline

    gold = (homework.get("gold_answer") or "").strip()
    student = (homework.get("student_answer") or "").strip()

    if not gold or not student:
        return None
    if not _looks_parseable(gold) or not _looks_parseable(student):
        return None
    if not _is_safe_expression(gold) or not _is_safe_expression(student):
        return None   # hostile or unsupported syntax: never hand it to sympy

    try:
        gold_expr = parse(gold)
        student_expr = parse(student)
    except Exception:
        return None  # not cleanly parseable -- defer to the LLM tiers

    try:
        is_equal = bool(sympy.simplify(gold_expr - student_expr) == 0)
    except Exception:
        try:
            is_equal = gold_expr == student_expr
        except Exception:
            return None

    if is_equal:
        return GradingResult(
            is_correct=True,
            classification="CORRECT",
            score=100.0,
            error_type=None,
            error_location=None,
            feedback="Final answer matches the expected result (verified symbolically, no AI call needed).",
            sub_scores={},
            confidence=1.0,
            raw_tokens={"input": 0, "output": 0},
            latency_ms=0,
            provider_name="sympy_check",
        )

    # A confident WRONG classification from a $0 symbolic check is risky
    # -- the student's method/partial credit still needs human-quality
    # judgment. Only auto-resolve the CORRECT case; let a mismatch fall
    # through to the LLM so partial credit and error explanation still happen.
    return None
