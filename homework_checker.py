"""
homework_checker.py
The core backend: takes an uploaded homework file, sends it (plus the
grading system prompt) to Gemini, and returns validated, structured JSON.

Uses the current, officially-supported "google-genai" SDK (the older
"google-generativeai" package is deprecated by Google). This gives us
access to explicit context caching (huge input-token savings on the
system prompt, which is identical across every request for a given
subject/skill) in addition to normal grading.

PDFs and images are sent directly to Gemini as multimodal input, since
the model can read handwriting and typeset math from the page itself.
DOCX files are text-extracted first (Gemini has no native docx reader).
"""

import logging
import re
import time
from pathlib import Path
from typing import Optional

from hwchecker.infra.deadline import JobDeadline
import cache_registry
import token_usage_logger
from config import (
    CACHE_TTL_SECONDS,
    ENABLE_CONTEXT_CACHING,
    GEMINI_API_KEY,
    DEFAULT_MODEL,
    GENERATION_CONFIG,
)
from prompts import build_system_prompt, build_user_prompt, DEFAULT_SUBJECT_KEY
from utils import (
    extract_text_from_docx,
    parse_and_validate_json,
    guess_mime_type,
    InvalidModelResponseError,
)

# Timing diagnostics: shows in the terminal running `streamlit run app.py`
# or `uvicorn api:app`, so you can see exactly where time goes for a slow
# request (file prep vs. which Gemini call vs. how many retries fired).
#
# IMPORTANT: we configure ONLY this module's own logger (with its own
# handler, propagate=False) rather than calling logging.basicConfig().
# basicConfig() configures the ROOT logger, which affects every other
# library using the standard logging module too (fontTools, PIL, etc.) --
# that's what caused unrelated font-subsetting logs to flood the console
# when PDF reports were generated. This way only our own timing lines print.
logger = logging.getLogger("homework_checker")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(message)s"))
    logger.addHandler(_handler)
    logger.propagate = False

# Gemini accepts small files as "inline data" (raw bytes in the request
# body) or as uploads via the resumable Files API. Inline is simpler and
# avoids the Files API's extra network round-trip. We use inline data for
# anything reasonably small (homework pages, short audio recordings) and
# only fall back to the Files API for larger files, per Google's own
# guidance (inline is recommended for requests under ~20MB total).
INLINE_SIZE_LIMIT_BYTES = 15 * 1024 * 1024  # 15 MB


class HomeworkCheckerError(Exception):
    pass


def _safe_cache_display_name(*parts: str) -> str:
    """Builds a short, safe display_name for a Gemini cache from key parts."""
    raw = "hwck-" + "-".join(p for p in parts if p)
    slug = re.sub(r"[^a-zA-Z0-9\-]+", "-", raw).strip("-").lower()
    return slug[:100] or "hwck-default"


class HomeworkChecker:
    """
    Thin wrapper around the Gemini API dedicated to homework grading.

    Usage:
        # Standard subject:
        checker = HomeworkChecker(api_key="...", model_name="gemini-3.5-flash",
                                   subject_key="physics")
        # Language skill:
        checker = HomeworkChecker(api_key="...", model_name="gemini-3.5-flash",
                                   language_key="english", skill_key="speaking")
        result = checker.analyze(Path("uploads/hw1.pdf"))
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = DEFAULT_MODEL,
        subject_key: str = DEFAULT_SUBJECT_KEY,
        custom_subject_name: str = "",
        language_key: str = "",
        skill_key: str = "",
    ):
        self.api_key = api_key or GEMINI_API_KEY
        if not self.api_key:
            raise HomeworkCheckerError(
                "No Gemini API key found. Set the GEMINI_API_KEY environment "
                "variable or enter a key in the sidebar."
            )
        self.model_name = model_name
        self.subject_key = subject_key
        self.custom_subject_name = custom_subject_name
        self.language_key = language_key
        self.skill_key = skill_key

        # Imported lazily so the rest of the app (and tests) can run
        # without the google-genai package installed.
        from google import genai

        self._genai = genai
        self._types = __import__("google.genai.types", fromlist=["types"])
        self.client = genai.Client(api_key=self.api_key)

        self.system_prompt = build_system_prompt(
            subject_key=subject_key,
            custom_subject_name=custom_subject_name,
            language_key=language_key,
            skill_key=skill_key,
        )
        # Stable key identifying this exact (model, subject/skill)
        # combination -- used to look up / share a Gemini explicit cache
        # for the system prompt across processes. See cache_registry.py.
        self.cache_key = "|".join([
            model_name, subject_key, custom_subject_name.lower().strip(),
            language_key, skill_key,
        ])
        self.cache_display_name = _safe_cache_display_name(
            model_name, subject_key or "lang", language_key, skill_key
        )
        self.last_usage = {"input_tokens": 0, "output_tokens": 0, "cached_tokens": 0, "total_tokens": 0}

    def _get_generation_config_kwargs(self) -> dict:
        """
        Decides whether to use a cached system prompt (cheaper) or send it
        inline (always works). Returns kwargs to merge into
        GenerateContentConfig: either {"cached_content": name} or
        {"system_instruction": self.system_prompt}, never both -- Gemini
        does not allow specifying system_instruction alongside cached_content.
        """
        types = self._types

        if not ENABLE_CONTEXT_CACHING:
            return {"system_instruction": self.system_prompt}

        status, cache_name = cache_registry.lookup(self.cache_key)

        if status == "cached":
            logger.info(f"  Using cached system prompt (cache={cache_name})")
            return {"cached_content": cache_name}

        if status == "uncacheable":
            return {"system_instruction": self.system_prompt}

        # status == "unknown" -- try creating a new cache.
        try:
            t0 = time.monotonic()
            cache = self.client.caches.create(
                model=self.model_name,
                config=types.CreateCachedContentConfig(
                    system_instruction=self.system_prompt,
                    ttl=f"{CACHE_TTL_SECONDS}s",
                    display_name=self.cache_display_name,
                ),
            )
            cache_registry.set_cache(self.cache_key, cache.name, CACHE_TTL_SECONDS)
            logger.info(
                f"  Created new context cache '{cache.name}' in "
                f"{time.monotonic() - t0:.1f}s (will be reused for {CACHE_TTL_SECONDS}s)"
            )
            return {"cached_content": cache.name}
        except Exception as exc:
            # Most common cause: prompt is below Gemini's minimum cacheable
            # token count. Remember this so we don't retry on every request.
            cache_registry.mark_uncacheable(self.cache_key)
            logger.info(f"  Could not create context cache, falling back to uncached: {exc}")
            return {"system_instruction": self.system_prompt}

    def analyze(
        self,
        file_path: Path,
        extra_instructions: str = "",
        max_retries: int = 2,
    ) -> dict:
        """
        Analyzes a single homework file and returns validated JSON (dict).
        Retries once or twice if the model returns malformed JSON, asking
        it to correct itself.
        """
        types = self._types
        file_path = Path(file_path)
        ext = file_path.suffix.lower()
        user_prompt = build_user_prompt(extra_instructions)
        self.last_user_prompt = user_prompt

        t_start = time.monotonic()
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        logger.info(f"analyze() start: file={file_path.name} size={file_size_mb:.2f}MB model={self.model_name}")

        if ext == ".docx":
            contents = [
                types.Part.from_text(text=user_prompt),
                types.Part.from_text(
                    text="The homework was submitted as a Word document. "
                    "Here is the extracted text:\n\n" + extract_text_from_docx(file_path)
                ),
            ]
        else:
            mime_type = guess_mime_type(file_path)
            file_size = file_path.stat().st_size
            try:
                if file_size <= INLINE_SIZE_LIMIT_BYTES:
                    file_bytes = file_path.read_bytes()
                    contents = [
                        types.Part.from_text(text=user_prompt),
                        types.Part.from_bytes(data=file_bytes, mime_type=mime_type),
                    ]
                else:
                    t_upload_start = time.monotonic()
                    uploaded = self.client.files.upload(
                        file=str(file_path),
                        config=types.UploadFileConfig(mime_type=mime_type),
                    )
                    logger.info(f"  Files API upload took {time.monotonic() - t_upload_start:.1f}s")
                    contents = [
                        types.Part.from_text(text=user_prompt),
                        types.Part.from_uri(file_uri=uploaded.uri, mime_type=mime_type),
                    ]
            except Exception as exc:
                raise HomeworkCheckerError(
                    f"Failed to prepare '{file_path.name}' (detected mime type: "
                    f"{mime_type}) for Gemini: {exc}"
                ) from exc

        logger.info(f"  File prep took {time.monotonic() - t_start:.1f}s")

        last_error = None
        for attempt in range(max_retries + 1):
            t_attempt_start = time.monotonic()
            logger.info(f"  Attempt {attempt + 1}/{max_retries + 1}: calling Gemini...")
            try:
                config_kwargs = self._get_generation_config_kwargs()
                config = types.GenerateContentConfig(
                    temperature=GENERATION_CONFIG.get("temperature", 0.1),
                    top_p=GENERATION_CONFIG.get("top_p", 0.9),
                    max_output_tokens=GENERATION_CONFIG.get("max_output_tokens", 8192),
                    response_mime_type=GENERATION_CONFIG.get("response_mime_type", "application/json"),
                    **config_kwargs,
                )
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                    config=config,
                )
                attempt_elapsed = time.monotonic() - t_attempt_start
                usage = getattr(response, "usage_metadata", None)
                cached_tokens = getattr(usage, "cached_content_token_count", None)
                cache_note = f", cached_tokens={cached_tokens}" if cached_tokens else ""
                logger.info(
                    f"  Attempt {attempt + 1} got a response in {attempt_elapsed:.1f}s"
                    f"{cache_note}, validating JSON..."
                )
                result = parse_and_validate_json(response.text)
                logger.info(f"analyze() SUCCESS total={time.monotonic() - t_start:.1f}s (attempt {attempt + 1})")

                self.last_usage = {
                    "input_tokens": getattr(usage, "prompt_token_count", 0) or 0,
                    "output_tokens": getattr(usage, "candidates_token_count", 0) or 0,
                    "cached_tokens": cached_tokens or 0,
                    "total_tokens": getattr(usage, "total_token_count", 0) or 0,
                }
                token_usage_logger.log_usage(
                    input_tokens=self.last_usage["input_tokens"],
                    output_tokens=self.last_usage["output_tokens"],
                    cached_tokens=self.last_usage["cached_tokens"],
                    total_tokens=self.last_usage["total_tokens"],
                    backend="gemini",
                    model_name=self.model_name,
                    subject_key=self.subject_key,
                    custom_subject_name=self.custom_subject_name,
                    language_key=self.language_key,
                    skill_key=self.skill_key,
                )
                return result
            except InvalidModelResponseError as exc:
                attempt_elapsed = time.monotonic() - t_attempt_start
                logger.info(f"  Attempt {attempt + 1} failed JSON validation after {attempt_elapsed:.1f}s: {exc}")
                last_error = exc
                # Ask the model to fix its own output on retry.
                contents = contents + [
                    types.Part.from_text(
                        text="Your previous response was not valid JSON matching the "
                        "required schema. Return ONLY the corrected JSON object, "
                        "nothing else."
                    )
                ]
                time.sleep(1)
            except JobDeadline:
                raise   # the job's time budget is spent: do not retry
            except Exception as exc:  # network / API errors
                attempt_elapsed = time.monotonic() - t_attempt_start
                logger.info(f"  Attempt {attempt + 1} raised {type(exc).__name__} after {attempt_elapsed:.1f}s: {exc}")
                last_error = exc
                time.sleep(1.5 * (attempt + 1))

        total_elapsed = time.monotonic() - t_start
        logger.info(f"analyze() FAILED total={total_elapsed:.1f}s after {max_retries + 1} attempts")
        raise HomeworkCheckerError(
            f"Failed to get a valid analysis after {max_retries + 1} attempts: {last_error}"
        )

    def analyze_batch(self, file_paths, extra_instructions: str = ""):
        """Analyzes multiple files, returning a list of (path, result_or_error)."""
        results = []
        for path in file_paths:
            try:
                result = self.analyze(path, extra_instructions=extra_instructions)
                results.append((path, result, None))
            except Exception as exc:
                results.append((path, None, str(exc)))
        return results


def get_checker(
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
    subject_key: str = DEFAULT_SUBJECT_KEY,
    custom_subject_name: str = "",
    language_key: str = "",
    skill_key: str = "",
    backend: Optional[str] = None,
):
    """
    Factory that returns either the Gemini-backed HomeworkChecker or the
    self-hosted LocalModelChecker, or one of the additional cloud
    providers (OpenAI, Anthropic, DeepSeek, Qwen), depending on
    config.MODEL_BACKEND (or the explicit `backend` argument, which
    overrides it for one call).

    Every checker class exposes the identical .analyze(file_path,
    extra_instructions) interface and returns the same JSON schema, so
    callers (app.py, api.py) never need to know or care which one they
    got back. This is the ONE place in the codebase where the
    which-provider-grades-this decision is made.

    Backend modes (config.MODEL_BACKEND, or the `backend` argument):
      - "auto" (the default) -- routes each subject/language+skill to
        the provider+model config.SUBJECT_MODEL_MAP /
        LANGUAGE_SKILL_MODEL_MAP recommends for it. This is what enables
        genuinely different companies' models grading different subjects.
      - "local" -- always use your self-hosted server (Phase 2),
        regardless of subject.
      - a specific provider key ("gemini", "openai", "anthropic",
        "deepseek", "qwen") -- forces that one provider for everything,
        ignoring the per-subject mapping. Useful for testing one
        provider's quality/cost in isolation.

    An explicit `model_name` argument always wins over any of this.

    Special case: if config.FORCE_GEMINI_FOR_SPEAKING is true (the
    default) and this call is for skill_key="speaking", Gemini is used
    even if routing would otherwise pick a different provider. Native
    audio understanding is still uncommon outside Gemini (and a
    self-hosted audio model you've explicitly deployed) -- this avoids a
    silent dead end when Speaking gets routed to a provider that simply
    can't accept audio input.
    """
    from config import (
        FORCE_GEMINI_FOR_SPEAKING,
        MODEL_BACKEND,
        PROVIDER_DEFAULT_MODEL,
        PROVIDERS,
        resolve_provider_and_model,
    )

    chosen_backend = (backend or MODEL_BACKEND or "auto").lower()

    if chosen_backend == "auto":
        resolved_provider, subject_default_model = resolve_provider_and_model(
            subject_key, language_key, skill_key
        )
    elif chosen_backend == "local":
        resolved_provider, subject_default_model = "local", ""
    elif chosen_backend in PROVIDERS:
        resolved_provider = chosen_backend
        subject_default_model = PROVIDER_DEFAULT_MODEL.get(chosen_backend, "")
    else:
        raise ValueError(
            f"Unknown MODEL_BACKEND/backend '{chosen_backend}'. Expected 'auto', 'local', "
            f"or one of: {list(PROVIDERS)}"
        )

    if skill_key == "speaking" and resolved_provider != "gemini" and FORCE_GEMINI_FOR_SPEAKING:
        resolved_provider = "gemini"
        subject_default_model = PROVIDER_DEFAULT_MODEL.get("gemini", "")

    resolved_model = model_name or subject_default_model

    if resolved_provider == "local":
        from local_checker import LocalModelChecker

        return LocalModelChecker(
            api_key=api_key,
            model_name=resolved_model,
            subject_key=subject_key,
            custom_subject_name=custom_subject_name,
            language_key=language_key,
            skill_key=skill_key,
        )

    provider_kind = PROVIDERS[resolved_provider]["kind"]

    if provider_kind == "gemini":
        return HomeworkChecker(
            api_key=api_key,
            model_name=resolved_model,
            subject_key=subject_key,
            custom_subject_name=custom_subject_name,
            language_key=language_key,
            skill_key=skill_key,
        )

    if provider_kind == "anthropic":
        from anthropic_checker import AnthropicChecker

        return AnthropicChecker(
            api_key=api_key,
            model_name=resolved_model,
            subject_key=subject_key,
            custom_subject_name=custom_subject_name,
            language_key=language_key,
            skill_key=skill_key,
        )

    if provider_kind == "openai_protocol":
        from cloud_checker import CloudModelChecker

        return CloudModelChecker(
            provider_key=resolved_provider,
            api_key=api_key,
            model_name=resolved_model,
            subject_key=subject_key,
            custom_subject_name=custom_subject_name,
            language_key=language_key,
            skill_key=skill_key,
        )

    raise ValueError(f"Unhandled provider kind '{provider_kind}' for provider '{resolved_provider}'.")
