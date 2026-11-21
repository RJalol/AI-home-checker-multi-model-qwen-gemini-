"""
setup_env.py
Creates `.env` (from `.env.example`) with freshly generated secrets, so the API does not start in its
"fail closed" state (INTERNAL_API_KEY empty -> every protected endpoint answers 503).

    python setup_env.py              # creates / completes ./.env
    python setup_env.py --path /etc/homework/.env

What it does, and only this:
  * INTERNAL_API_KEY       -- generated if missing or empty (the key your CRM backend sends as X-Internal-Api-Key)
  * WEBHOOK_SIGNING_SECRET -- generated if missing or empty (signs webhooks; separate from the admin key, so
                              rotating one does not invalidate the other)
  * Values that are already set are NEVER changed, and the secrets are NEVER printed (read them from the file).
  * A newly created file is readable by its owner only (0600).
It does not touch your provider API keys: add GEMINI_API_KEY etc. to the file yourself.

Changing WEBHOOK_SIGNING_SECRET later changes every organization's webhook secret
(GET /organizations/me/webhook-secret): tell your CRM.
"""

import argparse
import os
import re
import secrets
import shutil
import sys
from pathlib import Path
from typing import Callable, List

GENERATED_KEYS = ("INTERNAL_API_KEY", "WEBHOOK_SIGNING_SECRET")
HERE = Path(__file__).resolve().parent


def _line_regex(key: str, commented: bool) -> "re.Pattern":
    prefix = r"\s*#\s*" if commented else r"\s*"
    return re.compile(rf"^{prefix}{re.escape(key)}\s*=(?P<value>[^#\n]*)(?P<tail>\s*#.*)?$")


def ensure_env(path: Path, example: Path, make_secret: Callable[[], str] = lambda: secrets.token_urlsafe(32)) -> List[str]:
    """Creates `path` from `example` if needed and fills the empty secrets. Returns the names that were generated."""
    path = Path(path)
    if not path.exists():
        if example.exists():
            shutil.copyfile(example, path)
        else:
            path.write_text("", encoding="utf-8")
        os.chmod(path, 0o600)

    lines = path.read_text(encoding="utf-8").splitlines()
    generated = []
    for key in GENERATED_KEYS:
        active, commented = _line_regex(key, False), _line_regex(key, True)
        active_at = next((i for i, line in enumerate(lines) if active.match(line)), None)
        if active_at is not None:
            if active.match(lines[active_at]).group("value").strip():
                continue   # already set: leave it alone
            target = active_at
        else:
            target = next((i for i, line in enumerate(lines) if commented.match(line)), None)
        new_line = f"{key}={make_secret()}"
        if target is None:
            lines.append(new_line)
        else:
            lines[target] = new_line
        generated.append(key)

    if generated:
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return generated


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Create .env with generated INTERNAL_API_KEY and WEBHOOK_SIGNING_SECRET.")
    parser.add_argument("--path", default=str(HERE / ".env"), help="the .env file to create or complete (default: next to this script)")
    args = parser.parse_args(argv)
    path = Path(args.path)
    existed = path.exists()
    generated = ensure_env(path, HERE / ".env.example")
    print(f"{'Updated' if existed else 'Created'} {path}")
    for key in GENERATED_KEYS:
        print(f"  {key}: {'generated (value is in the file, not shown here)' if key in generated else 'already set, left unchanged'}")
    print("Next: add your provider key(s) (e.g. GEMINI_API_KEY) to the file, then start everything with `honcho start` "
          "(or run the three commands in the Procfile yourself).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
