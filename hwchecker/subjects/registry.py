"""
hwchecker/subjects/registry.py
SUBJECT_REGISTRY: dict[str, SubjectProfile] -- 9 initial entries (math,
physics, chemistry, biology, history, english, russian, uzbek,
turkish). Adding a new subject = adding ONE new entry here; nothing in
router/, providers/, pipeline/ should ever need to change.

deterministic_checker functions are wired in HERE (not inside each
subjects/*.py module) specifically to avoid a circular import:
checkers/*.py imports GradingResult from providers/base.py, and
subjects/*.py profiles are meant to stay free of any dependency on
checkers/ or providers/ -- keeping this wiring in one place (here)
means subjects/math.py etc. stay pure data.
"""

from typing import Dict

from .base import SubjectProfile
from .biology import BIOLOGY_PROFILE
from .chemistry import CHEMISTRY_PROFILE
from .history import HISTORY_PROFILE
from .languages import ENGLISH_PROFILE, RUSSIAN_PROFILE, TURKISH_PROFILE, UZBEK_PROFILE
from .math import MATH_PROFILE
from .physics import PHYSICS_PROFILE


def _wire_deterministic_checkers() -> None:
    from hwchecker.checkers import equation_balance, fuzzy_match, sympy_check, unit_check

    MATH_PROFILE.deterministic_checker = sympy_check.check
    PHYSICS_PROFILE.deterministic_checker = unit_check.check
    CHEMISTRY_PROFILE.deterministic_checker = equation_balance.check
    BIOLOGY_PROFILE.deterministic_checker = fuzzy_match.check
    # HISTORY_PROFILE: intentionally left without a deterministic_checker
    # (see subjects/history.py docstring -- historical analysis isn't
    # reducible to a symbolic/exact-match check).

    from hwchecker.checkers import languagetool_check

    ENGLISH_PROFILE.deterministic_checker = languagetool_check.check
    RUSSIAN_PROFILE.deterministic_checker = languagetool_check.check
    TURKISH_PROFILE.deterministic_checker = languagetool_check.check
    # UZBEK_PROFILE: intentionally left without languagetool_check --
    # LanguageTool doesn't meaningfully support Uzbek (see
    # subjects/languages.py) -- LLM-only grammar checking for now.


_wire_deterministic_checkers()

SUBJECT_REGISTRY: Dict[str, SubjectProfile] = {
    "math": MATH_PROFILE,
    "physics": PHYSICS_PROFILE,
    "chemistry": CHEMISTRY_PROFILE,
    "biology": BIOLOGY_PROFILE,
    "history": HISTORY_PROFILE,
    "english": ENGLISH_PROFILE,
    "russian": RUSSIAN_PROFILE,
    "uzbek": UZBEK_PROFILE,
    "turkish": TURKISH_PROFILE,
}


def get_subject(subject_id: str) -> SubjectProfile:
    if subject_id not in SUBJECT_REGISTRY:
        raise ValueError(f"Unknown subject_id '{subject_id}'. Known subjects: {list(SUBJECT_REGISTRY)}")
    return SUBJECT_REGISTRY[subject_id]
