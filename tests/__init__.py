"""
Test suite -- run from the repository root:

    python -m unittest discover -s tests -t . -v
    (or: pytest tests)

No API keys, network or Redis are needed: the OpenAI / Anthropic / Google SDKs are replaced by
in-process fakes (tests/fakes.py) that record every request and answer from a script, so
the REAL router, providers, prompts and infra code run. What these tests cannot tell you is
how good a real model's grading is -- that is what hwchecker/benchmarks/ is for.
"""

import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
