"""
hwchecker/checkers/unit_check.py
Deterministic (no LLM, $0) numeric-value-with-units verification for
physics, using Pint. Used as SubjectProfile.deterministic_checker for
physics (subjects/physics.py).

Like sympy_check.py, this is conservative: only resolves the CORRECT
case with high confidence (matching value within tolerance, after unit
conversion); any parsing failure or mismatch falls through to the LLM
tiers so method/partial-credit judgment still happens.
"""

import re
from typing import Optional

from hwchecker.providers.base import GradingResult

_VALUE_UNIT_RE = re.compile(r"^\s*(-?\d+\.?\d*(?:[eE]-?\d+)?)\s*([a-zA-Z/\^0-9\*\.\s]*)\s*$")
DEFAULT_RELATIVE_TOLERANCE = 0.01  # 1% -- generous enough for rounding, strict enough to catch real errors


def _try_import_pint():
    try:
        import pint

        return pint
    except ImportError:
        return None


_UNIT_REGISTRY = None


def _registry(pint):
    """Pint's UnitRegistry takes ~150 ms to build; build it once per process, not per answer."""
    global _UNIT_REGISTRY
    if _UNIT_REGISTRY is None:
        _UNIT_REGISTRY = pint.UnitRegistry()
    return _UNIT_REGISTRY


def _parse_value_unit(text: str):
    match = _VALUE_UNIT_RE.match(text.strip())
    if not match:
        return None, None
    value_str, unit_str = match.groups()
    try:
        value = float(value_str)
    except ValueError:
        return None, None
    return value, unit_str.strip() or None


def check(homework: dict, relative_tolerance: float = DEFAULT_RELATIVE_TOLERANCE) -> Optional[GradingResult]:
    """
    homework: {"gold_answer": "9.8 m/s^2", "student_answer": "9.81 m/s^2", ...}
    Returns a GradingResult only for a confident, unit-aware CORRECT
    match; None otherwise (defer to the LLM tiers).
    """
    pint = _try_import_pint()
    if pint is None:
        return None

    gold = (homework.get("gold_answer") or "").strip()
    student = (homework.get("student_answer") or "").strip()
    if not gold or not student:
        return None

    gold_value, gold_unit = _parse_value_unit(gold)
    student_value, student_unit = _parse_value_unit(student)
    if gold_value is None or student_value is None:
        return None

    ureg = _registry(pint)
    try:
        gold_qty = ureg.Quantity(gold_value, gold_unit) if gold_unit else ureg.Quantity(gold_value)
        student_qty = ureg.Quantity(student_value, student_unit) if student_unit else ureg.Quantity(student_value)
        student_qty_converted = student_qty.to(gold_qty.units)
    except Exception:
        return None  # incompatible/unparseable units -- defer to the LLM tiers (which can explain the unit error)

    if gold_qty.magnitude == 0:
        is_close = abs(student_qty_converted.magnitude) < 1e-9
    else:
        is_close = abs(student_qty_converted.magnitude - gold_qty.magnitude) / abs(gold_qty.magnitude) <= relative_tolerance

    if is_close:
        return GradingResult(
            is_correct=True,
            classification="CORRECT",
            score=100.0,
            error_type=None,
            error_location=None,
            feedback=f"Final answer ({student}) matches the expected result within tolerance, units verified (no AI call needed).",
            sub_scores={},
            confidence=1.0,
            raw_tokens={"input": 0, "output": 0},
            latency_ms=0,
            provider_name="unit_check",
        )

    return None  # mismatch -- defer to the LLM so method/partial-credit judgment still happens
