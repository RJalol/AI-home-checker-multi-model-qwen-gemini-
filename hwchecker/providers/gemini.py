"""
hwchecker/providers/gemini.py
GeminiProvider -- wraps Google's native google-genai SDK. Used for OCR/
document parsing (vision on handwritten/scanned pages) and as the
Primary-Medium tier per the model-selection table (providers/registry.py).
"""

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
    build_grading_result,
    extract_json_object,
    normalize_thinking,
    output_token_limit,
)

# Rough published per-1M-token pricing (USD) for cost estimation/logging
# only -- NOT used for billing. Update if the provider changes pricing.
#
# List prices, Oct 2026, from third-party price trackers -- verify on
# ai.google.dev/gemini-api/docs/pricing. The old gemini-3.5-flash figure
# (0.30/2.50) is actually gemini-3.5-flash-LITE's price.
_PRICING_PER_1M = {
    # "gemini-3.5-flash" is a deprecated alias that may be served (and billed) as
    # gemini-3.6-flash ($0.75/$3.75); the 3.5 Flash list price is used so costs
    # aren't under-estimated. Check the Google AI Studio invoice to see which applies.
    "gemini-3.5-flash": {"input": 1.50, "output": 9.00},
    "gemini-3.6-flash": {"input": 0.75, "output": 3.75},
    # Introductory price through 2026-12-31; doubles to 1.50/7.50 on 2027-01-01.
    "gemini-3.8-flash": {"input": 0.75, "output": 3.75},
    "gemini-3.5-flash-lite": {"input": 0.30, "output": 2.50},
    "gemini-3.1-flash-lite": {"input": 0.25, "output": 1.50},   # shuts down 2027-05-07
    "gemini-3.1-pro-preview": {"input": 1.25, "output": 10.00},   # not re-checked
}

_log = logging.getLogger("hwchecker.providers")

MAX_OUTPUT_TOKENS = 8192

# Gemini 3 takes a `thinking_level` (default "high"); thinking tokens are billed as output.
# "off" cannot truly disable it on Gemini 3, so it maps to the lowest level.
_THINKING_LEVELS = {
    "off": "minimal", "minimal": "minimal", "low": "low", "medium": "medium",
    "high": "high", "xhigh": "high", "max": "high",
}


class GeminiProvider(AIProvider):
    supports_vision = True
    supports_audio = True   # Gemini reads audio natively

    def __init__(self, model_name: str = "gemini-3.5-flash", api_key: Optional[str] = None, thinking: Optional[str] = None):
        import os

        self.name = model_name
        self.model_name = model_name
        #: Hidden-reasoning level (see base.normalize_thinking); None = Gemini's own default.
        self.thinking = normalize_thinking(thinking)
        self._thinking_ok = True   # flipped off if the API/SDK turns out not to accept thinking_config
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        if not self.api_key:
            raise ProviderUnavailableError("GEMINI_API_KEY is not set and no api_key was provided.")

        try:
            from google import genai
        except ImportError as exc:
            raise ProviderUnavailableError(
                "The 'google-genai' package is required. Install with: pip install google-genai"
            ) from exc

        self._genai = genai
        self._types = __import__("google.genai.types", fromlist=["types"])
        try:  # google-genai takes the request timeout in MILLISECONDS
            http_options = self._types.HttpOptions(timeout=int(PROVIDER_TIMEOUT_SECONDS * 1000))
            self.client = genai.Client(api_key=self.api_key, http_options=http_options)
        except (AttributeError, TypeError, ValueError):   # an older SDK without HttpOptions: keep working
            self.client = genai.Client(api_key=self.api_key)

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        rates = _PRICING_PER_1M.get(self.model_name, {"input": 0.30, "output": 2.50})
        return (input_tokens / 1_000_000) * rates["input"] + (output_tokens / 1_000_000) * rates["output"]

    def health_check(self) -> bool:
        try:
            self.client.models.list()
            return True
        except Exception:
            return False

    def _build_contents(self, content: dict) -> list:
        types = self._types
        file_path = content.get("file_path")
        user_prompt = content.get("user_prompt", "")
        parts = [types.Part.from_text(text=user_prompt)]

        if file_path:
            file_path = Path(file_path)
            ext = file_path.suffix.lower()
            if ext == ".docx":
                from utils import extract_text_from_docx

                parts[0] = types.Part.from_text(
                    text=user_prompt + "\n\nExtracted document text:\n\n" + wrap_student_text(extract_text_from_docx(file_path))
                )
            else:
                from utils import guess_mime_type

                mime_type = guess_mime_type(file_path) or "application/octet-stream"
                parts.append(types.Part.from_bytes(data=file_path.read_bytes(), mime_type=mime_type))

        return parts

    def _thinking_config(self):
        """`ThinkingConfig` for self.thinking, or None (not requested, or this SDK has no such type)."""
        level = _THINKING_LEVELS.get(self.thinking or "")
        if level is None or not self._thinking_ok:
            return None
        try:
            return self._types.ThinkingConfig(thinking_level=level)
        except (AttributeError, TypeError, ValueError):   # an older google-genai without thinking levels
            self._thinking_ok = False
            return None

    def _generate(self, contents: list, system_prompt: str):
        """One generate_content call; if the API rejects the thinking setting, retry once without it."""
        types = self._types
        for _ in range(2):
            settings = dict(
                temperature=0.1,
                top_p=0.9,
                max_output_tokens=output_token_limit(MAX_OUTPUT_TOKENS),
                response_mime_type="application/json",
                system_instruction=system_prompt,
            )
            thinking_config = self._thinking_config()
            if thinking_config is not None:
                settings["thinking_config"] = thinking_config
            config = types.GenerateContentConfig(**settings)
            try:
                return self.client.models.generate_content(model=self.model_name, contents=contents, config=config)
            except JobDeadline:
                raise
            except Exception as exc:
                if thinking_config is not None and "thinking" in str(exc).lower():
                    self._thinking_ok = False   # keep working, just without the cost saving
                    _log.warning("%s rejected the thinking setting %r (%s); continuing without it, "
                                 "so this tier loses its thinking-token saving.", self.name, self.thinking, str(exc)[:200])
                    continue
                raise
        raise AIProviderError("Gemini call failed after dropping the thinking setting.")   # not reached in practice

    def grade_homework(self, content: dict, rubric: dict) -> GradingResult:
        contents = self._build_contents(content)
        system_prompt = content.get("system_prompt", "")

        def _call():
            return self._generate(contents, system_prompt)

        try:
            response, latency_ms = self._timed_call(_call)
        except JobDeadline:
            raise   # the job's time budget is spent: not a provider failure
        except Exception as exc:
            raise AIProviderError(f"Gemini grading call failed: {exc}") from exc

        # response.text is None when Gemini blocks the reply (safety filter) or returns no candidates.
        raw_text = getattr(response, "text", None)
        try:
            data = extract_json_object(raw_text, prefer_keys=("classification", "score"))
        except ValueError as exc:
            raise AIProviderError(f"Gemini returned non-JSON output: {exc}\n{str(raw_text)[:300]}") from exc

        usage = getattr(response, "usage_metadata", None)
        return build_grading_result(
            data,
            default_confidence=0.8,
            input_tokens=getattr(usage, "prompt_token_count", 0) or 0,
            # Thinking tokens are billed as output but reported separately from the visible answer.
            output_tokens=(getattr(usage, "candidates_token_count", 0) or 0) + (getattr(usage, "thoughts_token_count", 0) or 0),
            latency_ms=latency_ms,
            provider_name=self.name,
            cached_input_tokens=getattr(usage, "cached_content_token_count", 0),
        )
