"""
hwchecker/checkers/equation_balance.py
Deterministic (no LLM, $0) balanced-chemical-equation verification.
Used as SubjectProfile.deterministic_checker for chemistry
(subjects/chemistry.py) when the student's answer is itself a chemical
equation (not every chemistry question is -- most fall through to the
LLM tiers, which is expected and fine).

Parses simple equations like "2H2 + O2 -> 2H2O" or "CH4 + 2O2 = CO2 + 2H2O",
counts atoms of each element on both sides, and confirms they match.
This only ever resolves the "student wrote THE expected equation, correctly
balanced" case -- an unbalanced-looking parse could just as easily be a
parsing limitation as a real student error, so everything else always
defers to the LLM tier for a real judgment call.
"""

import math
import re
from collections import Counter
from typing import Optional

from hwchecker.providers.base import GradingResult

_TERM_RE = re.compile(r"^\s*(\d*)\s*([A-Za-z0-9\(\)]+)\s*$")
_ELEMENT_RE = re.compile(r"([A-Z][a-z]?)(\d*)")
_GROUP_RE = re.compile(r"\(([^()]+)\)(\d*)")


def _parse_formula(formula: str) -> Optional[Counter]:
    """Counts atoms in a single chemical formula, e.g. 'Ca(OH)2' -> {Ca:1, O:2, H:2}."""
    formula = formula.strip()
    if not formula:
        return None

    counts = Counter()

    def _expand_group(m):
        inner, mult = m.group(1), m.group(2)
        mult = int(mult) if mult else 1
        return inner * mult

    prev = None
    while prev != formula:
        prev = formula
        formula = _GROUP_RE.sub(_expand_group, formula)

    matches = _ELEMENT_RE.findall(formula)
    if not matches:
        return None
    for element, count_str in matches:
        if not element:
            continue
        count = int(count_str) if count_str else 1
        counts[element] += count
    return counts


def _parse_side(side: str) -> Optional[Counter]:
    """Counts total atoms across all terms on one side of an equation,
    e.g. '2H2 + O2' -> {H:4, O:2}."""
    total = Counter()
    for term in side.split("+"):
        m = _TERM_RE.match(term)
        if not m:
            return None
        coeff_str, formula = m.groups()
        coeff = int(coeff_str) if coeff_str else 1
        atoms = _parse_formula(formula)
        if atoms is None:
            return None
        for element, count in atoms.items():
            total[element] += count * coeff
    return total


def _species(side: str):
    """{formula: coefficient} for one side, or None if it cannot be parsed. A formula repeated on a
    side makes the comparison ambiguous, so that returns None as well."""
    result = {}
    for term in side.split("+"):
        m = _TERM_RE.match(term)
        if not m:
            return None
        coeff_str, formula = m.groups()
        if formula in result:
            return None
        result[formula] = int(coeff_str) if coeff_str else 1
    return result


def check(homework: dict) -> Optional[GradingResult]:
    """
    homework: {"student_answer": "2H2 + O2 -> 2H2O", "gold_answer": "2H2 + O2 -> 2H2O", ...}
    Returns a GradingResult only when ALL of these hold; otherwise None (defer to the LLM tiers):
      * a reference equation is supplied as `gold_answer` -- an atom-balanced equation is not
        necessarily THE asked-for reaction ("CH4 -> CH4" and "Na + Cl2 = NaCl2" balance too);
      * the student's equation has exactly the same reactants and products as the reference;
      * it is balanced (atom counts match on both sides);
      * its coefficients are in lowest terms (4H2 + 2O2 -> 4H2O is balanced but not the answer asked for).
    An equation that does not confirm is never marked wrong here: a parse limitation is as likely
    as a real error, and the LLM can tell the difference and explain it.
    """
    student = (homework.get("student_answer") or "").strip()
    gold = (homework.get("gold_answer") or "").strip()
    if not student or not gold:
        return None

    student_sides = re.split(r"->|=|\u2192", student)
    gold_sides = re.split(r"->|=|\u2192", gold)
    if len(student_sides) != 2 or len(gold_sides) != 2:
        return None

    student_species = [_species(side) for side in student_sides]
    gold_species = [_species(side) for side in gold_sides]
    if None in student_species or None in gold_species:
        return None
    if [set(side) for side in student_species] != [set(side) for side in gold_species]:
        return None   # different substances than the reference reaction

    left_counts = _parse_side(student_sides[0])
    right_counts = _parse_side(student_sides[1])
    if left_counts is None or right_counts is None or left_counts != right_counts:
        return None

    coefficients = [c for side in student_species for c in side.values()]
    if math.gcd(*coefficients) != 1:
        return None   # balanced, but not in the smallest whole numbers

    return GradingResult(
        is_correct=True,
        classification="CORRECT",
        score=100.0,
        error_type=None,
        error_location=None,
        feedback="The equation has the expected reactants and products and is balanced (verified programmatically, no AI call needed).",
        sub_scores={},
        confidence=1.0,
        raw_tokens={"input": 0, "output": 0},
        latency_ms=0,
        provider_name="equation_balance",
    )
