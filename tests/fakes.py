"""
Fake provider SDKs + a unittest base class that wires them in.

The fakes mimic the parts of the real SDKs the providers use. The fake OpenAI endpoint is
deliberately as strict as OpenAI's reasoning models: it rejects `max_tokens`, `temperature`
and `top_p` on api.openai.com, so a provider that still sends them fails the test.
"""

import json
import os
import struct
import sys
import tempfile
import types
import unittest
import zlib
from pathlib import Path
from unittest import mock

ALL_KEY_VARS = ("QWEN_API_KEY", "GEMINI_API_KEY", "GLM_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY")
OTHER_VARS = ("FEEDBACK_LANGUAGE", "HWCHECKER_ENABLE_CACHE", "HWCHECKER_RATE_LIMIT_PER_MINUTE",
              "HWCHECKER_DAILY_BUDGET_USD", "HWCHECKER_MONTHLY_BUDGET_USD", "HWCHECKER_MAX_OUTPUT_TOKENS")
# Per-tier model/provider/thinking overrides (see providers/registry.py); a test must never inherit them.
TIER_VARS = tuple(
    f"HWCHECKER_{tier}_{what}"
    for tier in ("ROUTER", "EASY", "MEDIUM", "MEDIUM_ALT", "HARD", "HARD_VISION", "AUDIO", "VERIFICATION",
                 "VERIFICATION_GEMINI", "VERIFICATION_QWEN", "FALLBACK")
    for what in ("MODEL", "PROVIDER", "THINKING")
)

QWEN_HOST = "dashscope-intl.aliyuncs.com"
GLM_HOST = "api.z.ai"
OPENAI_HOST = "api.openai.com"


def grade_json(**overrides) -> str:
    data = {
        "classification": "PARTIALLY_CORRECT", "score": 55, "error_type": None, "error_location": None,
        "feedback": "ok", "sub_scores": {}, "confidence": 0.9,
        "questions": [
            {"question_number": 1, "question": "2+2", "student_answer": "4", "correct_answer": "4", "score": 100,
             "mistakes": [], "error_categories": [], "analysis": "Correct.", "suggestions": []},
            {"question_number": 2, "question": "3*3", "student_answer": "6", "correct_answer": "9", "score": 10,
             "mistakes": ["3*3 is 9, not 6"], "error_categories": ["Calculation Error"], "analysis": "Wrong product.",
             "suggestions": ["Review the multiplication table"]},
        ],
        "summary": {"strengths": ["addition"], "weaknesses": ["multiplication"], "topics_to_review": [], "recommendations": []},
    }
    data.update(overrides)
    return json.dumps(data)


def make_png(path: Path, size=(64, 48)) -> Path:
    """A real, valid PNG (a 'student photo')."""
    w, h = size
    raw = b"".join(b"\x00" + b"\xff\xff\xff" * w for _ in range(h))

    def chunk(kind, data):
        body = struct.pack(">I", len(data)) + kind + data
        return body + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    path.write_bytes(
        b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")
    )
    return path


class Backend:
    """Everything the fake SDKs record and are scripted with, for one test."""

    def __init__(self):
        self.calls = []   # one dict per outbound request
        self.cfg = {}
        self.client_kwargs = []   # kwargs each fake SDK client was constructed with (timeouts etc.)

    def to(self, host):
        return [c for c in self.calls if c["host"] == host]

    def grading(self, host=None):
        return [c for c in self.calls if c["kind"] == "grade" and (host is None or c["host"] == host)]

    def router_calls(self):
        return [c for c in self.calls if c["kind"] == "router"]


def _fail_if_scripted(backend, host):
    """cfg['fail_<host>'] = True (always) or an int N (the first N grading calls)."""
    rule = backend.cfg.get(f"fail_{host}")
    if rule is True:
        raise RuntimeError(f"{host} is down")
    if isinstance(rule, int) and rule > 0:
        backend.cfg[f"fail_{host}"] = rule - 1
        raise RuntimeError(f"{host} is down")


def install_fake_sdks(backend: Backend) -> dict:
    """Puts fake `openai`, `anthropic` and `google.genai` into sys.modules; returns the originals."""
    saved = {name: sys.modules.get(name) for name in ("openai", "anthropic", "google", "google.genai", "google.genai.types")}

    def response(text, tin=100, tout=50):
        message = types.SimpleNamespace(content=text)
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=message, finish_reason="stop")],
            usage=types.SimpleNamespace(prompt_tokens=tin, completion_tokens=tout),
        )

    class FakeOpenAI:
        def __init__(self, base_url=None, api_key=None, **kw):
            self.base_url = base_url
            backend.client_kwargs.append({"sdk": "openai", **kw})
            self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self._create))

        def _create(self, **kw):
            host = self.base_url.split("//")[1].split("/")[0]
            messages = kw["messages"]
            user = messages[-1]["content"]
            user_text = user if isinstance(user, str) else user[0]["text"]
            has_image = isinstance(user, list) and any(part.get("type") == "image_url" for part in user)
            is_router = "classify" in user_text.lower() and "difficulty" in user_text.lower()
            backend.calls.append({
                "kind": "router" if is_router else "grade", "host": host, "model": kw["model"], "image": has_image,
                "kwargs": sorted(k for k in kw if k not in ("messages", "model")),
                "extra_body": kw.get("extra_body"),
                "system": next((m["content"] for m in messages if m["role"] == "system"), ""), "user": user_text,
            })
            if kw.get("extra_body") and backend.cfg.get("reject_thinking"):   # an endpoint that does not know the field
                raise ValueError("Unrecognized request argument supplied: enable_thinking")
            if host == OPENAI_HOST:   # as strict as the reasoning-model families
                if "max_tokens" in kw:
                    raise ValueError("Unsupported parameter: 'max_tokens' is not supported with this model. Use 'max_completion_tokens' instead.")
                for name in ("temperature", "top_p"):
                    if name in kw:
                        raise ValueError(f"Unsupported value: '{name}' does not support this value with this model. Only the default (1) value is supported.")
            if is_router:
                return response(backend.cfg.get("router_answer", json.dumps({"difficulty": "easy"})), 80, 5)
            _fail_if_scripted(backend, host)
            return response(backend.cfg.get(f"grade_{host}", grade_json()))

    class FakeAnthropic:
        def __init__(self, api_key=None, **kw):
            backend.client_kwargs.append({"sdk": "anthropic", **kw})
            self.messages = types.SimpleNamespace(create=self._create)

        def _create(self, **kw):
            blocks = kw["messages"][0]["content"]
            backend.calls.append({
                "kind": "grade", "host": "anthropic", "model": kw["model"],
                "image": any(b["type"] in ("image", "document") for b in blocks),
                "kwargs": sorted(k for k in kw if k not in ("messages", "model", "system")), "system": kw["system"], "user": blocks[0]["text"],
                "extra_body": kw.get("extra_body"),
            })
            if kw.get("extra_body") and backend.cfg.get("reject_thinking"):
                raise ValueError("output_config.effort: Extra inputs are not permitted")
            _fail_if_scripted(backend, "anthropic")
            text = backend.cfg.get("claude_text", grade_json(confidence=0.95))
            return types.SimpleNamespace(
                content=[types.SimpleNamespace(type="text", text=text)],
                usage=types.SimpleNamespace(input_tokens=120, output_tokens=60),
            )

    class FakeGeminiModels:
        def generate_content(self, model, contents, config):
            backend.calls.append({
                "kind": "grade", "host": "gemini", "model": model,
                "image": any(getattr(p, "mime", "").startswith(("image", "audio", "application/pdf")) for p in contents),
                "kwargs": [], "system": config.system_instruction, "user": contents[0].text,
                "thinking": getattr(config, "thinking_config", None),
            })
            if getattr(config, "thinking_config", None) is not None and backend.cfg.get("reject_thinking"):
                raise ValueError("400 INVALID_ARGUMENT: thinking_level is not supported by this model")
            _fail_if_scripted(backend, "gemini")
            return types.SimpleNamespace(
                text=backend.cfg.get("gemini_text", grade_json()),
                usage_metadata=types.SimpleNamespace(
                    prompt_token_count=90, candidates_token_count=40,
                    thoughts_token_count=backend.cfg.get("gemini_thoughts_tokens"),
                ),
            )

    openai_mod = types.ModuleType("openai")
    openai_mod.OpenAI = FakeOpenAI
    anthropic_mod = types.ModuleType("anthropic")
    anthropic_mod.Anthropic = FakeAnthropic
    google_mod, genai_mod, types_mod = types.ModuleType("google"), types.ModuleType("google.genai"), types.ModuleType("google.genai.types")
    def _fake_genai_client(**kw):
        backend.client_kwargs.append({"sdk": "gemini", **kw})
        return types.SimpleNamespace(models=FakeGeminiModels())

    genai_mod.Client = _fake_genai_client
    types_mod.HttpOptions = lambda **kw: types.SimpleNamespace(**kw)
    types_mod.Part = types.SimpleNamespace(
        from_text=lambda text: types.SimpleNamespace(text=text, mime=""),
        from_bytes=lambda data, mime_type: types.SimpleNamespace(text="", mime=mime_type),
    )
    types_mod.GenerateContentConfig = lambda **kw: types.SimpleNamespace(**kw)
    types_mod.ThinkingConfig = lambda **kw: types.SimpleNamespace(**kw)
    google_mod.genai = genai_mod
    genai_mod.types = types_mod
    sys.modules.update({
        "openai": openai_mod, "anthropic": anthropic_mod,
        "google": google_mod, "google.genai": genai_mod, "google.genai.types": types_mod,
    })
    return saved


def restore_sdks(saved: dict) -> None:
    for name, module in saved.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


def executable_markup(html_text: str) -> list:
    """Everything in `html_text` that a browser could execute or load: script/iframe/... elements,
    on* event attributes and javascript: URLs. The text is PARSED (not substring-matched), so
    escaped text such as `&lt;script&gt;` -- which is inert -- is correctly not reported."""
    from html.parser import HTMLParser

    found = []
    risky_tags = {"script", "iframe", "frame", "object", "embed", "applet", "img", "svg", "math", "link", "base",
                  "form", "input", "button", "video", "audio", "source"}

    class Parser(HTMLParser):
        def handle_starttag(self, tag, attrs):
            if tag in risky_tags:
                found.append(f"<{tag}>")
            for name, value in attrs:
                if name.lower().startswith("on"):
                    found.append(f"{tag}[{name}]")
                if (value or "").strip().lower().startswith(("javascript:", "data:text/html", "vbscript:")):
                    found.append(f"{tag}[{name}={value[:20]}]")

    Parser().feed(html_text)
    return found


class FakeSDKTestCase(unittest.TestCase):
    """Fake SDKs, a clean environment, no Redis, no real sleeping, fresh in-process infra state."""

    keys = ALL_KEY_VARS   # which API keys exist in this test; override per test class / test

    def setUp(self):
        from hwchecker.infra import budget, cache, circuit_breaker, rate_limiter, _redis
        from hwchecker.providers import registry

        import logging

        logging.getLogger("hwchecker.monitoring").setLevel(logging.CRITICAL)   # keep the test output readable
        self.backend = Backend()
        self._saved_sdks = install_fake_sdks(self.backend)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.addCleanup(restore_sdks, self._saved_sdks)

        env = {k: "test-key" for k in self.keys}
        patcher = mock.patch.dict(os.environ, env, clear=False)
        patcher.start()
        self.addCleanup(patcher.stop)
        for name in ALL_KEY_VARS + OTHER_VARS:
            if name not in env:
                os.environ.pop(name, None)
        # mock.patch.dict(clear=False) restores os.environ on cleanup, so popping/setting here is test-local
        for name in TIER_VARS:
            os.environ.pop(name, None)

        for module in (budget, cache, circuit_breaker, rate_limiter):
            p = mock.patch.object(module, "get_redis", lambda: None)
            p.start()
            self.addCleanup(p.stop)
        _redis.reset()
        cache._memory_cache.clear()
        circuit_breaker._memory_state.clear()
        rate_limiter._memory_windows.clear()
        budget.reset()

        p = mock.patch("time.sleep", lambda s: None)   # retry backoff must not slow the suite down
        p.start()
        self.addCleanup(p.stop)

        registry.reset_registry()
        self.addCleanup(registry.reset_registry)

        self.photo = make_png(Path(self.tmp.name) / "homework.png")
