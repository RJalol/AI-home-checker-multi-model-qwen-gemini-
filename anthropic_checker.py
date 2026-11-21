"""
anthropic_checker.py
Grading backend for Anthropic's Claude models, via the native Messages
API (not the OpenAI-protocol path other cloud providers use, since
Claude's request/response shape is different enough to warrant its own
small client).

Two things Claude's API does natively, which simplifies this module
compared to cloud_checker.py:
  - PDFs are sent directly as "document" content blocks -- no PyMuPDF
    page-to-image conversion needed, unlike OpenAI/DeepSeek/Qwen.
  - Images are sent as standard "image" content blocks.

One thing it does NOT support: audio input. If a Speaking submission
somehow gets routed to Claude (it shouldn't, under normal
config.LANGUAGE_SKILL_MODEL_MAP defaults), this raises a clear error
rather than silently failing some other way.
"""

import base64
import time
from pathlib import Path
from typing import Optional

from hwchecker.infra.deadline import JobDeadline
import token_usage_logger
from config import GENERATION_CONFIG
from prompts import DEFAULT_SUBJECT_KEY, build_system_prompt, build_user_prompt
from utils import (
    InvalidModelResponseError,
    extract_text_from_docx,
    guess_mime_type,
    parse_and_validate_json,
)

AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".ogg"}


class AnthropicCheckerError(Exception):
    pass


class AnthropicChecker:
    """
    Mirrors HomeworkChecker's (Gemini) public interface so
    homework_checker.get_checker() can hand back either interchangeably.

    Usage:
        checker = AnthropicChecker(subject_key="essay")
        result = checker.analyze(Path("uploads/essay.pdf"))
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "",
        subject_key: str = DEFAULT_SUBJECT_KEY,
        custom_subject_name: str = "",
        language_key: str = "",
        skill_key: str = "",
    ):
        import os

        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not self.api_key:
            raise AnthropicCheckerError(
                "No Anthropic API key found. Set the ANTHROPIC_API_KEY environment "
                "variable, or pass api_key explicitly."
            )
        if not model_name:
            raise AnthropicCheckerError("model_name is required for the Anthropic provider.")

        self.model_name = model_name
        self.subject_key = subject_key
        self.custom_subject_name = custom_subject_name
        self.language_key = language_key
        self.skill_key = skill_key
        self.last_usage = {"input_tokens": 0, "output_tokens": 0, "cached_tokens": 0, "total_tokens": 0}

        import anthropic

        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.system_prompt = build_system_prompt(
            subject_key=subject_key,
            custom_subject_name=custom_subject_name,
            language_key=language_key,
            skill_key=skill_key,
        )

    def _build_content_blocks(self, file_path: Path, extra_instructions: str) -> list:
        ext = file_path.suffix.lower()
        text = build_user_prompt(extra_instructions)
        self.last_user_prompt = text

        if ext in AUDIO_EXTENSIONS:
            raise AnthropicCheckerError(
                "Claude models do not accept audio input. This Speaking submission was "
                "routed to the Anthropic provider -- check config.LANGUAGE_SKILL_MODEL_MAP "
                "and route Speaking to gemini (or another audio-capable provider) instead."
            )

        if ext == ".docx":
            text += (
                "\n\nThe homework was submitted as a Word document. Here is "
                "the extracted text:\n\n" + extract_text_from_docx(file_path)
            )
            return [{"type": "text", "text": text}]

        blocks = [{"type": "text", "text": text}]
        file_bytes = file_path.read_bytes()
        b64 = base64.b64encode(file_bytes).decode("ascii")

        if ext == ".pdf":
            blocks.append({
                "type": "document",
                "source": {"type": "base64", "media_type": "application/pdf", "data": b64},
            })
        else:
            mime_type = guess_mime_type(file_path) or "image/png"
            blocks.append({
                "type": "image",
                "source": {"type": "base64", "media_type": mime_type, "data": b64},
            })

        return blocks

    def analyze(self, file_path: Path, extra_instructions: str = "", max_retries: int = 2) -> dict:
        file_path = Path(file_path)
        try:
            content_blocks = self._build_content_blocks(file_path, extra_instructions)
        except AnthropicCheckerError:
            raise
        except Exception as exc:
            raise AnthropicCheckerError(f"Failed to prepare '{file_path.name}' for Claude: {exc}") from exc

        messages = [{"role": "user", "content": content_blocks}]
        last_error = None
        raw_text = ""

        for attempt in range(max_retries + 1):
            try:
                response = self.client.messages.create(
                    model=self.model_name,
                    max_tokens=GENERATION_CONFIG.get("max_output_tokens", 8192),
                    temperature=GENERATION_CONFIG.get("temperature", 0.1),
                    system=self.system_prompt,
                    messages=messages,
                )
                raw_text = "".join(
                    block.text for block in response.content if getattr(block, "type", "") == "text"
                )
                result = parse_and_validate_json(raw_text)

                usage = getattr(response, "usage", None)
                input_tokens = getattr(usage, "input_tokens", 0) or 0
                output_tokens = getattr(usage, "output_tokens", 0) or 0
                cached_tokens = getattr(usage, "cache_read_input_tokens", 0) or 0
                self.last_usage = {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "cached_tokens": cached_tokens,
                    "total_tokens": input_tokens + output_tokens,
                }
                token_usage_logger.log_usage(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cached_tokens=cached_tokens,
                    total_tokens=input_tokens + output_tokens,
                    backend="anthropic",
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

        raise AnthropicCheckerError(
            f"Failed to get a valid analysis from Claude after {max_retries + 1} attempts: {last_error}"
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
