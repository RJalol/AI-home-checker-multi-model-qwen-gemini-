"""
cloud_checker.py
A single grading backend for every "OpenAI-protocol" cloud provider:
OpenAI itself, DeepSeek, and Qwen (via Alibaba's DashScope). These are
all reached the exact same way -- the `openai` python client pointed at
a different base_url/api_key/model -- so one class covers all three
instead of duplicating near-identical code per provider.

(config.LOCAL_BACKEND in local_checker.py covers your own self-hosted
server the same way, but stays a separate module since it also handles
PDF-to-image conversion the exact same way and predates this file --
no need to disturb an already-tested module to share logic that's this small.)

Vision support varies by provider (config.PROVIDERS[...]["supports_vision"]):
if a provider is text-only (e.g. DeepSeek's standard API models), this
checker raises a clear error rather than silently sending a request the
provider will just fail on some other way -- so a misconfigured
SUBJECT_MODEL_MAP entry (a vision-required subject routed to a
text-only provider) is easy to diagnose.
"""

import base64
import time
from pathlib import Path
from typing import Optional

from hwchecker.infra.deadline import JobDeadline
import token_usage_logger
from config import GENERATION_CONFIG, LOCAL_MODEL_API_KEY, LOCAL_MODEL_BASE_URL, PROVIDERS
from prompts import DEFAULT_SUBJECT_KEY, build_system_prompt, build_user_prompt
from utils import (
    InvalidModelResponseError,
    extract_text_from_docx,
    guess_mime_type,
    parse_and_validate_json,
)

AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".ogg"}
PDF_PAGE_DPI = 150
MAX_PDF_PAGES = 12


class CloudCheckerError(Exception):
    pass


def _pdf_to_image_data_urls(file_path: Path) -> list:
    """Renders each PDF page to a PNG data URL -- providers here take
    images, not raw PDF bytes (unlike Gemini/Claude's native PDF support)."""
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:
        raise CloudCheckerError(
            "PyMuPDF is required to send PDFs to this provider. Install with: pip install pymupdf"
        ) from exc

    doc = fitz.open(str(file_path))
    try:
        zoom = PDF_PAGE_DPI / 72
        matrix = fitz.Matrix(zoom, zoom)
        data_urls = []
        for i, page in enumerate(doc):
            if i >= MAX_PDF_PAGES:
                break
            pix = page.get_pixmap(matrix=matrix)
            b64 = base64.b64encode(pix.tobytes("png")).decode("ascii")
            data_urls.append(f"data:image/png;base64,{b64}")
        return data_urls
    finally:
        doc.close()


def _image_to_data_url(file_path: Path) -> str:
    mime_type = guess_mime_type(file_path) or "image/png"
    b64 = base64.b64encode(file_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{b64}"


class CloudModelChecker:
    """
    Grading backend for OpenAI, DeepSeek, Qwen (DashScope), or any other
    OpenAI-protocol-compatible cloud API. Mirrors HomeworkChecker's
    (Gemini) public interface so homework_checker.get_checker() can hand
    back any of them interchangeably.

    Usage:
        checker = CloudModelChecker(provider_key="openai", subject_key="computer_science")
        result = checker.analyze(Path("uploads/hw1.pdf"))
    """

    def __init__(
        self,
        provider_key: str,
        api_key: Optional[str] = None,
        model_name: str = "",
        subject_key: str = DEFAULT_SUBJECT_KEY,
        custom_subject_name: str = "",
        language_key: str = "",
        skill_key: str = "",
    ):
        if provider_key not in PROVIDERS:
            raise CloudCheckerError(f"Unknown provider '{provider_key}'. Known providers: {list(PROVIDERS)}")

        self.provider_key = provider_key
        self.provider_config = PROVIDERS[provider_key]
        self.subject_key = subject_key
        self.custom_subject_name = custom_subject_name
        self.language_key = language_key
        self.skill_key = skill_key
        self.model_name = model_name
        self.last_usage = {"input_tokens": 0, "output_tokens": 0, "cached_tokens": 0, "total_tokens": 0}

        if not self.model_name:
            raise CloudCheckerError(f"model_name is required for provider '{provider_key}'.")

        base_url = self.provider_config.get("base_url") or LOCAL_MODEL_BASE_URL
        if api_key:
            resolved_api_key = api_key
        elif provider_key == "local":
            resolved_api_key = LOCAL_MODEL_API_KEY
        else:
            import os

            env_var = self.provider_config.get("api_key_env")
            resolved_api_key = os.environ.get(env_var, "") if env_var else ""
            if not resolved_api_key:
                raise CloudCheckerError(
                    f"No API key found for provider '{provider_key}'. Set the {env_var} "
                    f"environment variable, or pass api_key explicitly."
                )

        from openai import OpenAI

        self.client = OpenAI(base_url=base_url, api_key=resolved_api_key)
        self.system_prompt = build_system_prompt(
            subject_key=subject_key,
            custom_subject_name=custom_subject_name,
            language_key=language_key,
            skill_key=skill_key,
        )

    def _build_user_content(self, file_path: Path, extra_instructions: str) -> list:
        ext = file_path.suffix.lower()
        text = build_user_prompt(extra_instructions)
        self.last_user_prompt = text
        content = []

        if ext == ".docx":
            text += (
                "\n\nThe homework was submitted as a Word document. Here is "
                "the extracted text:\n\n" + extract_text_from_docx(file_path)
            )
            content.append({"type": "text", "text": text})
            return content

        needs_vision = ext in (".pdf", ".png", ".jpg", ".jpeg", ".webp") or ext in AUDIO_EXTENSIONS
        if needs_vision and not self.provider_config.get("supports_vision"):
            raise CloudCheckerError(
                f"Provider '{self.provider_key}' does not support image/audio input (text-only API). "
                f"This subject/skill was routed here by SUBJECT_MODEL_MAP or LANGUAGE_SKILL_MODEL_MAP "
                f"in config.py -- route it to a vision-capable provider (gemini, anthropic, openai, "
                f"qwen, or local) instead, or submit this file type as a DOCX/text extract."
            )

        content.append({"type": "text", "text": text})

        if ext in AUDIO_EXTENSIONS:
            audio_format = ext.lstrip(".")
            b64 = base64.b64encode(file_path.read_bytes()).decode("ascii")
            content.append({"type": "input_audio", "input_audio": {"data": b64, "format": audio_format}})
        elif ext == ".pdf":
            for data_url in _pdf_to_image_data_urls(file_path):
                content.append({"type": "image_url", "image_url": {"url": data_url}})
        else:
            content.append({"type": "image_url", "image_url": {"url": _image_to_data_url(file_path)}})

        return content

    def analyze(self, file_path: Path, extra_instructions: str = "", max_retries: int = 2) -> dict:
        file_path = Path(file_path)
        try:
            user_content = self._build_user_content(file_path, extra_instructions)
        except CloudCheckerError:
            raise
        except Exception as exc:
            raise CloudCheckerError(f"Failed to prepare '{file_path.name}' for {self.provider_key}: {exc}") from exc

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]

        last_error = None
        raw_text = ""
        for attempt in range(max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=GENERATION_CONFIG.get("temperature", 0.1),
                    top_p=GENERATION_CONFIG.get("top_p", 0.9),
                    max_tokens=GENERATION_CONFIG.get("max_output_tokens", 8192),
                    response_format={"type": "json_object"},
                )
                raw_text = response.choices[0].message.content
                result = parse_and_validate_json(raw_text)

                usage = getattr(response, "usage", None)
                self.last_usage = {
                    "input_tokens": getattr(usage, "prompt_tokens", 0) or 0,
                    "output_tokens": getattr(usage, "completion_tokens", 0) or 0,
                    "cached_tokens": 0,
                    "total_tokens": getattr(usage, "total_tokens", 0) or 0,
                }
                token_usage_logger.log_usage(
                    input_tokens=self.last_usage["input_tokens"],
                    output_tokens=self.last_usage["output_tokens"],
                    total_tokens=self.last_usage["total_tokens"],
                    backend=self.provider_key,
                    model_name=self.model_name,
                    subject_key=self.subject_key,
                    custom_subject_name=self.custom_subject_name,
                    language_key=self.language_key,
                    skill_key=self.skill_key,
                )
                return result
            except InvalidModelResponseError as exc:
                last_error = exc
                messages.append({"role": "assistant", "content": raw_text})
                messages.append({
                    "role": "user",
                    "content": "Your previous response was not valid JSON matching the "
                               "required schema. Return ONLY the corrected JSON object, nothing else.",
                })
                time.sleep(1)
            except JobDeadline:
                raise   # the job's time budget is spent: do not retry
            except Exception as exc:
                last_error = exc
                time.sleep(1.5 * (attempt + 1))

        raise CloudCheckerError(
            f"Failed to get a valid analysis from {self.provider_key} after {max_retries + 1} attempts: {last_error}"
        )

    def analyze_batch(self, file_paths, extra_instructions: str = ""):
        results = []
        for path in file_paths:
            try:
                result = self.analyze(path, extra_instructions=extra_instructions)
                results.append((path, result, None))
            except Exception as exc:
                results.append((path, None, str(exc)))
        return results
