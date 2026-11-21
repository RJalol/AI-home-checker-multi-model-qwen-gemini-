"""
hwchecker/providers/_openai_protocol_base.py
Internal shared implementation for providers that speak the OpenAI chat-
completions protocol: Qwen (DashScope), GLM (Zhipu AI's "bigmodel"
endpoint), and OpenAI itself. Not one of the spec's named provider
files on purpose -- qwen.py, glm.py, and openai.py each stay as thin,
clearly-named subclasses of this, so PROVIDER_REGISTRY and everything
importing from providers.qwen / providers.glm / providers.openai sees
exactly the module layout the spec calls for, while the actual HTTP-
calling logic (which is identical across these three) isn't triplicated.
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

_log = logging.getLogger("hwchecker.providers")

PDF_PAGE_DPI = 150
MAX_PDF_PAGES = 12


def _pdf_to_image_data_urls(file_path: Path) -> list:
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:
        raise ProviderUnavailableError(
            "PyMuPDF is required to send PDFs to this provider. Install with: pip install pymupdf"
        ) from exc

    doc = fitz.open(str(file_path))
    try:
        zoom = PDF_PAGE_DPI / 72
        matrix = fitz.Matrix(zoom, zoom)
        urls = []
        for i, page in enumerate(doc):
            if i >= MAX_PDF_PAGES:
                break
            pix = page.get_pixmap(matrix=matrix)
            b64 = base64.b64encode(pix.tobytes("png")).decode("ascii")
            urls.append(f"data:image/png;base64,{b64}")
        return urls
    finally:
        doc.close()


class OpenAIProtocolProvider(AIProvider):
    """
    Base class for any provider reachable via the OpenAI chat-completions
    wire protocol. Subclasses just set base_url / api_key_env / pricing /
    supports_vision and a `name`.
    """

    base_url: str = ""
    api_key_env: str = ""
    supports_vision: bool = True
    supports_audio: bool = False
    pricing_per_1m: dict = {"input": 0.0, "output": 0.0}

    #: Request-shape knobs. Not every OpenAI-protocol endpoint accepts the same parameters:
    #: OpenAI's newer models want `max_completion_tokens` and reject non-default
    #: temperature/top_p, while Qwen/GLM take the classic `max_tokens` + sampling params.
    #: Subclasses set the right defaults; _create_completion() also adapts at runtime if the
    #: endpoint answers with a "unsupported parameter" error, so a model-family change on the
    #: provider's side does not turn every grading call into a failure.
    token_limit_param: str = "max_tokens"
    sends_sampling_params: bool = True
    #: Output budget. Generous on purpose: reasoning models spend part of it on hidden
    #: thinking, and a per-question breakdown is longer than a single verdict.
    max_output_tokens: int = 8192

    def __init__(self, model_name: str, api_key: Optional[str] = None, thinking: Optional[str] = None):
        import os

        self.name = model_name
        self.model_name = model_name
        #: How much hidden reasoning to ask for (see base.normalize_thinking); None = provider default.
        #: Subclasses that have such a knob turn it into request fields in _thinking_extra_body().
        self.thinking = normalize_thinking(thinking)
        resolved_key = api_key or os.environ.get(self.api_key_env, "")
        if not resolved_key:
            raise ProviderUnavailableError(
                f"{self.api_key_env} is not set and no api_key was provided for {self.__class__.__name__}."
            )

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ProviderUnavailableError(
                f"The 'openai' package is required for {self.__class__.__name__}. Install with: pip install openai"
            ) from exc

        # Explicit timeout (the SDK default is 10 minutes) and no hidden SDK-level retries: the
        # router already retries with back-off, and SDK retries would multiply the worst case.
        self.client = OpenAI(
            base_url=self.base_url, api_key=resolved_key, timeout=PROVIDER_TIMEOUT_SECONDS, max_retries=0
        )
        # Per-instance copies: they may be changed by _adapt_to_parameter_error().
        self._token_param = self.token_limit_param
        self._sampling_ok = self.sends_sampling_params
        self._extra_ok = True

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        return (
            (input_tokens / 1_000_000) * self.pricing_per_1m.get("input", 0.0)
            + (output_tokens / 1_000_000) * self.pricing_per_1m.get("output", 0.0)
        )

    def health_check(self) -> bool:
        try:
            self.client.models.list()
            return True
        except Exception:
            return False

    # -- request building ------------------------------------------------
    def _thinking_extra_body(self) -> Optional[dict]:
        """Provider-specific request fields that implement `self.thinking`, or None. Overridden by
        the providers that have such a knob (Qwen, GLM); the base class has none."""
        return None

    def _build_messages(self, content: dict) -> list:
        file_path = content.get("file_path")
        user_prompt = content.get("user_prompt", "")
        system_prompt = content.get("system_prompt", "")

        user_content = [{"type": "text", "text": user_prompt}]

        if file_path:
            file_path = Path(file_path)
            ext = file_path.suffix.lower()

            if ext == ".docx":
                from utils import extract_text_from_docx

                user_content[0]["text"] += "\n\nExtracted document text:\n\n" + wrap_student_text(extract_text_from_docx(file_path))
            elif ext in (".mp3", ".wav", ".m4a", ".ogg"):
                if not self.supports_audio:
                    raise UnsupportedInputError(f"{self.name} does not support audio input.")
                b64 = base64.b64encode(file_path.read_bytes()).decode("ascii")
                user_content.append({
                    "type": "input_audio",
                    "input_audio": {"data": b64, "format": ext.lstrip(".")},
                })
            elif ext == ".pdf":
                if not self.supports_vision:
                    raise UnsupportedInputError(
                        f"{self.name} is a text-only model and cannot read PDF/image homework. "
                        f"Route this subject to a vision-capable provider, or submit as DOCX."
                    )
                for data_url in _pdf_to_image_data_urls(file_path):
                    user_content.append({"type": "image_url", "image_url": {"url": data_url}})
            else:
                if not self.supports_vision:
                    raise UnsupportedInputError(f"{self.name} is a text-only model and cannot read image homework.")
                from utils import guess_mime_type

                mime_type = guess_mime_type(file_path) or "image/png"
                b64 = base64.b64encode(file_path.read_bytes()).decode("ascii")
                user_content.append({"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64}"}})

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

    def _request_kwargs(self, messages: list, *, max_tokens: int, temperature, top_p, json_mode: bool) -> dict:
        kwargs = {"model": self.model_name, "messages": messages, self._token_param: max_tokens}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if self._sampling_ok:
            if temperature is not None:
                kwargs["temperature"] = temperature
            if top_p is not None:
                kwargs["top_p"] = top_p
        extra = self._thinking_extra_body() if self._extra_ok else None
        if extra:
            kwargs["extra_body"] = extra
        return kwargs

    def _adapt_to_parameter_error(self, exc: Exception) -> bool:
        """If `exc` is the endpoint rejecting a request parameter we can drop or rename,
        adjust this instance's settings and return True (caller retries once)."""
        message = str(exc).lower()
        changed = False
        if self._token_param == "max_tokens" and "max_completion_tokens" in message:
            self._token_param = "max_completion_tokens"
            changed = True
        elif self._token_param == "max_completion_tokens" and "max_tokens" in message and "max_completion_tokens" not in message:
            self._token_param = "max_tokens"
            changed = True
        if self._sampling_ok and ("temperature" in message or "top_p" in message):
            self._sampling_ok = False
            changed = True
        if self._extra_ok and self._thinking_extra_body():
            # The endpoint does not know our thinking field(s): stop sending them. Grading then
            # still works, just without the cost saving -- better than every call failing.
            names = ("extra_body", "enable_thinking", "thinking", "reasoning_effort", "reasoning")
            if any(name in message for name in names):
                self._extra_ok = False
                changed = True
                _log.warning("%s rejected the thinking setting %r (%s); continuing without it, "
                             "so this tier loses its thinking-token saving.", self.name, self.thinking, str(exc)[:200])
        return changed

    def _create_completion(self, messages: list, *, max_tokens: int, temperature=None, top_p=None, json_mode: bool = True):
        last_exc = None
        for _ in range(4):   # up to three distinct parameters may need dropping/renaming, then the real attempt
            kwargs = self._request_kwargs(messages, max_tokens=max_tokens, temperature=temperature, top_p=top_p, json_mode=json_mode)
            try:
                return self.client.chat.completions.create(**kwargs)
            except JobDeadline:
                raise
            except Exception as exc:
                last_exc = exc
                if not self._adapt_to_parameter_error(exc):
                    raise
        raise last_exc

    # -- plain classification-style call (used by router/difficulty.py) -------
    def complete_json(self, prompt: str, *, file_path=None, max_tokens: int = 256) -> str:
        """One cheap JSON-mode completion that is NOT a grading call, optionally with the
        student's file attached. Returns the raw reply text."""
        messages = self._build_messages({"user_prompt": prompt, "system_prompt": "", "file_path": file_path})
        messages = [m for m in messages if not (m["role"] == "system" and not m["content"])]
        try:
            response = self._create_completion(messages, max_tokens=max_tokens, temperature=0.0)
        except JobDeadline:
            raise   # the job's time budget is spent: not a provider failure
        except Exception as exc:
            raise AIProviderError(f"{self.name} call failed: {exc}") from exc
        if not getattr(response, "choices", None):
            raise AIProviderError(f"{self.name} returned no choices.")
        return response.choices[0].message.content or ""

    # -- grading -----------------------------------------------------------
    def grade_homework(self, content: dict, rubric: dict) -> GradingResult:
        messages = self._build_messages(content)

        def _call():
            return self._create_completion(messages, max_tokens=output_token_limit(self.max_output_tokens), temperature=0.1, top_p=0.9)

        try:
            response, latency_ms = self._timed_call(_call)
        except JobDeadline:
            raise   # the job's time budget is spent: not a provider failure
        except Exception as exc:
            raise AIProviderError(f"{self.name} grading call failed: {exc}") from exc

        if not getattr(response, "choices", None):
            raise AIProviderError(f"{self.name} returned no choices (the request may have been blocked by a content filter).")
        choice = response.choices[0]
        raw_text = choice.message.content
        try:
            data = extract_json_object(raw_text, prefer_keys=("classification", "score"))
        except ValueError as exc:
            finish = getattr(choice, "finish_reason", None)
            hint = " (the reply was cut off at the output-token limit)" if finish == "length" else ""
            raise AIProviderError(f"{self.name} returned non-JSON output{hint}: {exc}\n{str(raw_text)[:300]}") from exc

        usage = getattr(response, "usage", None)
        return build_grading_result(
            data,
            default_confidence=0.8,
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
            latency_ms=latency_ms,
            provider_name=self.name,
            cached_input_tokens=getattr(getattr(usage, "prompt_tokens_details", None), "cached_tokens", 0),
        )
