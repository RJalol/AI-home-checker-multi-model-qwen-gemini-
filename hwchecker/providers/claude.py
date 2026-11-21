"""
hwchecker/providers/claude.py
ClaudeProvider -- used for Secondary Verification per the model-
selection table: re-checks low-confidence/borderline-score results from
the primary tier's provider. Uses Anthropic's native Messages API
(document/image content blocks), not the OpenAI protocol, since Claude's
PDF support is native (no page-to-image conversion needed).
"""

import base64
import logging
from pathlib import Path
from typing import Optional

from hwchecker.infra.deadline import JobDeadline
from hwchecker.safety import wrap_student_text

from .base import (
    PROVIDER_TIMEOUT_SECONDS,
    AIProvider,
    AIProviderError,
    GradingResult,
    ProviderUnavailableError,
    UnsupportedInputError,
    build_grading_result,
    extract_json_object,
    normalize_thinking,
    output_token_limit,
)

# Rough published per-1M-token pricing (USD), for cost estimation/logging
# only. Update if the provider changes pricing.
#
# Oct 2026, via third-party price trackers -- verify on Anthropic's pricing
# page. Sonnet 5 is $2/$10 (the old 3.00/15.00 was the Sonnet 4.x price).
# Haiku 5.5 is $0.10/$0.50 for prompts up to 100K tokens ($0.50/$2.50 above).
_PRICING_PER_1M = {
    "claude-sonnet-5": {"input": 2.00, "output": 10.00},
    "claude-sonnet-5-5": {"input": 2.00, "output": 10.00},
    "claude-haiku-5-5": {"input": 0.10, "output": 0.50},
    "claude-haiku-4-5-20251001": {"input": 0.80, "output": 4.00},
}

_log = logging.getLogger("hwchecker.providers")

MAX_OUTPUT_TOKENS = 8192

# Claude's adaptive thinking is steered with an effort level; "off" switches thinking off.
_EFFORT_LEVELS = {"minimal": "low", "low": "low", "medium": "medium", "high": "high", "xhigh": "xhigh", "max": "max"}


class ClaudeProvider(AIProvider):
    supports_vision = True
    supports_audio = False

    def __init__(self, model_name: str = "claude-sonnet-5", api_key: Optional[str] = None, thinking: Optional[str] = None):
        import os

        self.name = model_name
        self.model_name = model_name
        #: Effort/thinking setting (see base.normalize_thinking); None = the model's own default.
        self.thinking = normalize_thinking(thinking)
        self._extra_ok = True   # flipped off if the API rejects the thinking/effort field
        resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not resolved_key:
            raise ProviderUnavailableError("ANTHROPIC_API_KEY is not set and no api_key was provided.")

        try:
            import anthropic
        except ImportError as exc:
            raise ProviderUnavailableError("The 'anthropic' package is required. Install with: pip install anthropic") from exc

        self.client = anthropic.Anthropic(api_key=resolved_key, timeout=PROVIDER_TIMEOUT_SECONDS, max_retries=0)
        self._send_temperature = True

    def _thinking_extra_body(self) -> Optional[dict]:
        """Request fields for self.thinking. Sent through `extra_body` so they reach the API even
        when the installed SDK version has no typed parameter for them yet."""
        level = self.thinking
        if level is None or not self._extra_ok:
            return None
        if level == "off":
            return {"thinking": {"type": "disabled"}}
        effort = _EFFORT_LEVELS.get(level)
        return {"output_config": {"effort": effort}} if effort else None

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        rates = _PRICING_PER_1M.get(self.model_name, {"input": 3.00, "output": 15.00})
        return (input_tokens / 1_000_000) * rates["input"] + (output_tokens / 1_000_000) * rates["output"]

    def _build_content_blocks(self, content: dict) -> list:
        file_path = content.get("file_path")
        user_prompt = content.get("user_prompt", "")
        blocks = [{"type": "text", "text": user_prompt}]

        if not file_path:
            return blocks

        file_path = Path(file_path)
        ext = file_path.suffix.lower()

        if ext == ".docx":
            from utils import extract_text_from_docx

            blocks[0]["text"] += "\n\nExtracted document text:\n\n" + wrap_student_text(extract_text_from_docx(file_path))
            return blocks

        if ext in (".mp3", ".wav", ".m4a", ".ogg"):
            raise UnsupportedInputError("Claude models do not accept audio input.")

        b64 = base64.b64encode(file_path.read_bytes()).decode("ascii")
        if ext == ".pdf":
            blocks.append({"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": b64}})
        else:
            from utils import guess_mime_type

            mime_type = guess_mime_type(file_path) or "image/png"
            blocks.append({"type": "image", "source": {"type": "base64", "media_type": mime_type, "data": b64}})

        return blocks

    def _create(self, messages: list, system_prompt: str):
        last_exc = None
        for _ in range(3):   # up to two distinct parameters may need dropping, then the real attempt
            kwargs = {
                "model": self.model_name,
                "max_tokens": output_token_limit(MAX_OUTPUT_TOKENS),
                "system": system_prompt,
                "messages": messages,
            }
            if self._send_temperature:
                kwargs["temperature"] = 0.1
            extra = self._thinking_extra_body()
            if extra:
                kwargs["extra_body"] = extra
            try:
                return self.client.messages.create(**kwargs)
            except JobDeadline:
                raise
            except Exception as exc:
                last_exc = exc
                message = str(exc).lower()
                adapted = False
                # Some newer Claude models no longer accept a temperature setting, and the
                # thinking/effort fields differ between models. If the API says so, drop the
                # offending field (for this provider instance from now on) and retry.
                if self._send_temperature and "temperature" in message:
                    self._send_temperature = False
                    adapted = True
                if extra and any(name in message for name in ("extra_body", "output_config", "effort", "thinking")):
                    self._extra_ok = False
                    adapted = True
                    _log.warning("%s rejected the thinking setting %r (%s); continuing without it, "
                                 "so this tier loses its thinking-token saving.", self.name, self.thinking, str(exc)[:200])
                if not adapted:
                    raise
        raise last_exc

    def grade_homework(self, content: dict, rubric: dict) -> GradingResult:
        blocks = self._build_content_blocks(content)
        system_prompt = content.get("system_prompt", "")
        messages = [{"role": "user", "content": blocks}]

        try:
            response, latency_ms = self._timed_call(self._create, messages, system_prompt)
        except JobDeadline:
            raise   # the job's time budget is spent: not a provider failure
        except Exception as exc:
            raise AIProviderError(f"Claude grading call failed: {exc}") from exc

        raw_text = "".join(b.text for b in response.content if getattr(b, "type", "") == "text")
        try:
            data = extract_json_object(raw_text, prefer_keys=("classification", "score"))
        except ValueError as exc:
            raise AIProviderError(f"Claude returned non-JSON output: {exc}\n{raw_text[:300]}") from exc

        usage = getattr(response, "usage", None)
        return build_grading_result(
            data,
            default_confidence=0.85,
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
            latency_ms=latency_ms,
            provider_name=self.name,
        )
