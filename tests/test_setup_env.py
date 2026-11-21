import io
import os
import stat
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import setup_env

REPO_ROOT = Path(__file__).resolve().parent.parent


class SetupEnvTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / ".env"
        self.counter = 0

    def secret(self):
        self.counter += 1
        return f"generated-{self.counter}"

    def values(self):
        out = {}
        for line in self.path.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                key, _, value = line.partition("=")
                out[key.strip()] = value.split("#")[0].strip()
        return out

    def test_creates_the_file_from_the_example_with_both_secrets(self):
        generated = setup_env.ensure_env(self.path, REPO_ROOT / ".env.example", self.secret)
        self.assertEqual(generated, ["INTERNAL_API_KEY", "WEBHOOK_SIGNING_SECRET"])
        values = self.values()
        self.assertEqual(values["INTERNAL_API_KEY"], "generated-1")
        self.assertEqual(values["WEBHOOK_SIGNING_SECRET"], "generated-2")
        self.assertIn("GEMINI_API_KEY", self.path.read_text(), "the rest of .env.example is kept")
        self.assertEqual(self.path.read_text().count("WEBHOOK_SIGNING_SECRET="), 1, "the commented example line is replaced, not duplicated")

    def test_new_file_is_private(self):
        setup_env.ensure_env(self.path, REPO_ROOT / ".env.example", self.secret)
        self.assertEqual(stat.S_IMODE(os.stat(self.path).st_mode), 0o600)

    def test_existing_values_are_never_changed_and_a_second_run_does_nothing(self):
        self.path.write_text("GEMINI_API_KEY=abc  # mine\nINTERNAL_API_KEY=keep-me\n")
        generated = setup_env.ensure_env(self.path, REPO_ROOT / ".env.example", self.secret)
        self.assertEqual(generated, ["WEBHOOK_SIGNING_SECRET"])
        before = self.path.read_text()
        self.assertEqual(self.values()["INTERNAL_API_KEY"], "keep-me")
        self.assertEqual(setup_env.ensure_env(self.path, REPO_ROOT / ".env.example", self.secret), [])
        self.assertEqual(self.path.read_text(), before)
        self.assertIn("GEMINI_API_KEY=abc  # mine", before)

    def test_empty_value_is_filled_and_missing_key_is_appended(self):
        self.path.write_text("INTERNAL_API_KEY=   # set me\nOTHER=1\n")
        setup_env.ensure_env(self.path, REPO_ROOT / ".env.example", self.secret)
        values = self.values()
        self.assertEqual((values["INTERNAL_API_KEY"], values["WEBHOOK_SIGNING_SECRET"], values["OTHER"]), ("generated-1", "generated-2", "1"))

    def test_real_generator_makes_long_distinct_secrets_and_cli_never_prints_them(self):
        out = io.StringIO()
        with redirect_stdout(out):
            self.assertEqual(setup_env.main(["--path", str(self.path)]), 0)
        values = self.values()
        self.assertGreaterEqual(len(values["INTERNAL_API_KEY"]), 40)
        self.assertNotEqual(values["INTERNAL_API_KEY"], values["WEBHOOK_SIGNING_SECRET"])
        for secret in (values["INTERNAL_API_KEY"], values["WEBHOOK_SIGNING_SECRET"]):
            self.assertNotIn(secret, out.getvalue())

    def test_procfile_runs_the_three_processes_the_readme_requires(self):
        procfile = (REPO_ROOT / "Procfile").read_text()
        self.assertIn("uvicorn api:app", procfile)
        self.assertIn("rq worker homework_grading webhook_delivery --with-scheduler", procfile)
        self.assertIn("python reconcile.py --loop", procfile)


if __name__ == "__main__":
    unittest.main()
