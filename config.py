"""
config.py
Central configuration for the Homework Checker AI app.
All values can be overridden via environment variables so the app
works the same way locally, in Docker, or on a hosting platform.
"""

import os
from pathlib import Path

# If a .env file exists (see .env.example), load it into the environment
# automatically -- this is what makes `cp .env.example .env` + editing it
# actually work, on Windows/macOS/Linux alike, without needing to run
# export/set/$env: commands by hand every session. Real environment
# variables (e.g. set by your OS, shell profile, or process manager)
# still take precedence and are never overwritten by .env.
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed -- fall back to real env vars only

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
REPORTS_DIR = BASE_DIR / "reports"
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

for _dir in (UPLOAD_DIR, REPORTS_DIR, TEMPLATES_DIR, STATIC_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Gemini API
# ---------------------------------------------------------------------------
# Never hard-code a real key here. Set it as an environment variable:
#   export GEMINI_API_KEY="your-key-here"
# or type it into the sidebar text box when running the Streamlit app.
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# Available models the teacher can pick from in the UI.
# NOTE: Google retires/renames Gemini model IDs periodically. If you get a
# "404 ... is no longer available to new users" error, check the current
# list at https://ai.google.dev/gemini-api/docs/models and update below.
AVAILABLE_MODELS = [
    "gemini-3.5-flash",        # stable, best price-performance default
    "gemini-3.1-flash-lite",   # stable, fastest/cheapest option
    "gemini-3.1-pro-preview",  # preview, strongest reasoning for hard math
]
DEFAULT_MODEL = "gemini-3.5-flash"

# ---------------------------------------------------------------------------
# Per-subject model selection (accuracy vs cost)
# ---------------------------------------------------------------------------
# Not every subject needs the same model. Subjects that mostly involve
# checking facts/terminology against a completed answer are handled well
# by a fast, cheap model; subjects that require following multi-step
# reasoning to judge whether an approach (not just the final number) was
# valid benefit more from a stronger model, even at higher cost per
# request. This mapping is the default used whenever a caller doesn't
# explicitly request a specific model_name -- an explicit model_name
# ALWAYS wins over this.
#
# These are reasonable starting points, not settled science -- once you
# have enough teacher-reviewed "gold" data (training_data/gold.jsonl) per
# subject, compare AI-vs-teacher agreement rates across models for that
# subject and adjust this mapping based on real evidence rather than
# assumption.
# ---------------------------------------------------------------------------
# Multi-provider registry (Phase 4: not just Gemini)
# ---------------------------------------------------------------------------
# Each provider entry describes how to reach it and what it can do.
# "kind" selects which checker class handles it (see homework_checker.
# get_checker()): "gemini" (native google-genai SDK, supports caching),
# "anthropic" (native Claude Messages API, native PDF/image support), or
# "openai_protocol" (the openai python client pointed at a different
# base_url -- covers OpenAI itself, DeepSeek, Qwen/DashScope, and your
# own self-hosted server, since they all speak the same wire protocol).
#
# NOTE: every AI company renames/retires model IDs frequently -- these
# are reasonable defaults as of when this was last updated, not a
# permanent guarantee. If a model ID here 404s, check that provider's
# current model list and update PROVIDERS (or just override via
# SUBJECT_MODEL_MAP / an explicit model_name -- nothing else needs to change).
PROVIDERS = {
    "gemini": {
        "kind": "gemini",
        "api_key_env": "GEMINI_API_KEY",
        "supports_vision": True,
        "supports_pdf_native": True,   # Gemini accepts raw PDF bytes directly
    },
    "anthropic": {
        "kind": "anthropic",
        "api_key_env": "ANTHROPIC_API_KEY",
        "supports_vision": True,
        "supports_pdf_native": True,   # Claude's Messages API accepts PDF documents directly
    },
    "openai": {
        "kind": "openai_protocol",
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
        "supports_vision": True,       # images only -- PDFs are converted to page images first
        "supports_pdf_native": False,
    },
    "deepseek": {
        "kind": "openai_protocol",
        "base_url": "https://api.deepseek.com/v1",
        "api_key_env": "DEEPSEEK_API_KEY",
        "supports_vision": False,      # DeepSeek's standard API models are text-only as of writing
        "supports_pdf_native": False,
    },
    "qwen": {
        "kind": "openai_protocol",
        "base_url": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        "api_key_env": "QWEN_API_KEY",
        "supports_vision": True,       # images only -- PDFs are converted to page images first
        "supports_pdf_native": False,
    },
    "local": {
        "kind": "openai_protocol",
        "base_url": None,  # resolved at call time from LOCAL_MODEL_BASE_URL
        "api_key_env": None,
        "supports_vision": True,
        "supports_pdf_native": False,
    },
}

# ---------------------------------------------------------------------------
# Per-subject model selection (accuracy vs cost, across ALL providers)
# ---------------------------------------------------------------------------
# Not every subject needs the same model -- or even the same company's
# model. Subjects that mostly involve checking facts/terminology against
# a completed answer are handled well by a fast, cheap model; subjects
# that require following multi-step reasoning to judge whether an
# APPROACH (not just the final number) was valid benefit more from a
# stronger model, regardless of provider, even at higher cost per
# request. Each entry is (provider_key, model_name). This mapping is the
# default used whenever a caller doesn't explicitly request a specific
# model_name -- an explicit model_name (+ optional provider) ALWAYS wins
# over this.
#
# These are reasonable starting points, not settled science -- once you
# have enough teacher-reviewed "gold" data (training_data/gold.jsonl) per
# subject, compare AI-vs-teacher agreement rates across providers/models
# for that subject and adjust this mapping based on real evidence.
SUBJECT_MODEL_MAP = {
    "math": ("gemini", "gemini-3.5-flash"),
    "physics": ("gemini", "gemini-3.5-flash"),
    # NOTE: chemistry deliberately does NOT default to DeepSeek even
    # though its models are excellent value for technical reasoning --
    # DeepSeek's standard API models are text-only (see PROVIDERS above),
    # and real homework is very often a photo/scan, not typed text. Qwen
    # gives similar cost savings while still handling images/PDFs. If
    # your deployment only ever receives typed/DOCX chemistry work,
    # ("deepseek", "deepseek-v4-pro") is a substantially cheaper choice --
    # just be deliberate about that tradeoff rather than defaulting to it.
    "chemistry": ("qwen", "qwen3.7-plus"),
    "biology": ("qwen", "qwen3.7-plus"),                  # mostly terminology/fact checking, cheap multimodal
    "computer_science": ("openai", "gpt-5.6"),            # logic/code-correctness reasoning benefits most from a top-tier model
    "essay": ("anthropic", "claude-sonnet-5"),            # nuanced writing/argumentation feedback is a strength of Claude models
    "history": ("qwen", "qwen3.7-plus"),                  # mostly factual/terminology checking, cheap multimodal
    "general": ("gemini", "gemini-3.5-flash"),
}

# Keyed by (language_key, skill_key); "*" matches any language for that
# skill, so per-language overrides (e.g. a specific language needing a
# stronger model) can be added without duplicating every language.
LANGUAGE_SKILL_MODEL_MAP = {
    ("*", "writing"): ("anthropic", "claude-sonnet-5"),   # nuanced language feedback
    ("*", "reading"): ("qwen", "qwen3.7-plus"),           # comprehension-check grading, cheap
    ("*", "listening"): ("qwen", "qwen3.7-plus"),         # comprehension-check grading, cheap
    ("*", "speaking"): ("gemini", "gemini-3.5-flash"),    # needs native audio understanding
}


def resolve_provider_and_model(subject_key: str = "", language_key: str = "", skill_key: str = "") -> tuple:
    """
    Returns (provider_key, model_name) -- the recommended provider AND
    model for a subject or language+skill, per SUBJECT_MODEL_MAP /
    LANGUAGE_SKILL_MODEL_MAP above. Used as the default in
    homework_checker.get_checker() whenever the caller doesn't pass an
    explicit provider/model_name -- explicit values always override this.
    """
    if language_key and skill_key:
        return (
            LANGUAGE_SKILL_MODEL_MAP.get((language_key, skill_key))
            or LANGUAGE_SKILL_MODEL_MAP.get(("*", skill_key))
            or ("gemini", DEFAULT_MODEL)
        )
    return SUBJECT_MODEL_MAP.get(subject_key, ("gemini", DEFAULT_MODEL))


def resolve_model_name(subject_key: str = "", language_key: str = "", skill_key: str = "") -> str:
    """Backwards-compatible helper returning just the model name (Gemini
    call sites that don't need the provider). Prefer
    resolve_provider_and_model() for new code."""
    return resolve_provider_and_model(subject_key, language_key, skill_key)[1]

# Generation settings
GENERATION_CONFIG = {
    "temperature": 0.1,       # low temperature -> consistent, strict grading
    "top_p": 0.9,
    "max_output_tokens": 8192,
    "response_mime_type": "application/json",
}

# ---------------------------------------------------------------------------
# Gemini explicit context caching (cost optimization)
# ---------------------------------------------------------------------------
# The system prompt for a given (subject/language+skill) is identical
# across every request -- a textbook case for Gemini's explicit context
# caching, which bills cached input at roughly 10% of the normal rate.
# See cache_registry.py for the shared (Redis-backed) registry that lets
# multiple worker processes reuse one cache instead of each creating
# their own. Has no effect on the local/self-hosted backend.
ENABLE_CONTEXT_CACHING = os.environ.get("ENABLE_CONTEXT_CACHING", "true").lower() == "true"
# How long a created cache stays valid before it must be recreated.
# Longer TTLs amortize the (small) cache-write cost over more requests,
# but also mean a prompt change in prompts.py takes up to this long to
# actually take effect in production (old cached prompt keeps being used
# until it expires). 1 hour is a reasonable default for active use.
CACHE_TTL_SECONDS = int(os.environ.get("CACHE_TTL_SECONDS", str(60 * 60)))

# ---------------------------------------------------------------------------
# Gemini Batch API (cost optimization for non-real-time bulk grading)
# ---------------------------------------------------------------------------
# The Batch API gives a flat 50% discount on input+output, in exchange for
# asynchronous (up to ~24h) turnaround instead of an immediate response.
# Use this for bulk/overnight grading runs (e.g. a CRM exporting a whole
# day's submissions at once) rather than for a teacher waiting on a
# single result in the Streamlit UI -- see batch_processor.py.
BATCH_POLL_INTERVAL_SECONDS = int(os.environ.get("BATCH_POLL_INTERVAL_SECONDS", "20"))
BATCH_MAX_WAIT_SECONDS = int(os.environ.get("BATCH_MAX_WAIT_SECONDS", str(60 * 60 * 24)))  # 24h

# ---------------------------------------------------------------------------
# Model backend switch: which provider(s) grade homework
# ---------------------------------------------------------------------------
# This is the ONE setting that determines who grades homework. Nothing
# else in the app (app.py, api.py, report_generator.py) needs to change
# when you switch it.
#
#   "auto" (default) -- route each subject/language+skill to whichever
#     provider+model config.SUBJECT_MODEL_MAP / LANGUAGE_SKILL_MODEL_MAP
#     recommends for it (see above) -- this is what lets different
#     companies' models grade different subjects.
#   "local" -- always use your own self-hosted server (Phase 2), for
#     every subject, regardless of the per-subject mapping.
#   a specific provider key ("gemini", "openai", "anthropic",
#     "deepseek", "qwen") -- forces that ONE provider for everything,
#     ignoring the per-subject mapping. Useful for testing a single
#     provider's quality/cost/latency in isolation.
#
#   export MODEL_BACKEND="local"
MODEL_BACKEND = os.environ.get("MODEL_BACKEND", "auto")

# Fallback model used when a specific provider is force-selected via
# MODEL_BACKEND (above) rather than resolved per-subject -- e.g.
# MODEL_BACKEND=openai with no explicit model_name needs *some* default
# OpenAI model to use for every subject.
PROVIDER_DEFAULT_MODEL = {
    "gemini": DEFAULT_MODEL,
    "anthropic": "claude-sonnet-5",
    "openai": "gpt-5.6",
    "deepseek": "deepseek-v4-flash",
    "qwen": "qwen3.7-plus",
}

# Even when Speaking would otherwise be routed to a provider that can't
# accept audio input (most cloud providers besides Gemini, and any local
# model unless you've specifically deployed an audio-capable one),
# Speaking grading keeps using Gemini by default. Set to "false" once
# you have a real audio-capable alternative configured and want Speaking
# to use it too.
#   export FORCE_GEMINI_FOR_SPEAKING="false"
FORCE_GEMINI_FOR_SPEAKING = os.environ.get("FORCE_GEMINI_FOR_SPEAKING", "true").lower() == "true"

# ---------------------------------------------------------------------------
# Async job queue (Redis + RQ) -- for POST /jobs in api.py
# ---------------------------------------------------------------------------
# When many students submit at once (e.g. via a CRM), grading them
# synchronously inside the HTTP request risks slow responses, CRM
# timeouts, and Gemini rate-limit errors under load. POST /jobs instead
# enqueues the work and returns a job_id immediately; one or more
# background workers (run with `rq worker homework_grading`) process the
# queue, and the caller polls GET /jobs/{job_id} for the result.
#   export REDIS_URL="redis://localhost:6379/0"
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
JOB_QUEUE_NAME = os.environ.get("JOB_QUEUE_NAME", "homework_grading")
JOB_TIMEOUT_SECONDS = int(os.environ.get("JOB_TIMEOUT_SECONDS", "300"))  # max time per job
JOB_RESULT_TTL_SECONDS = int(os.environ.get("JOB_RESULT_TTL_SECONDS", str(60 * 60 * 24)))  # how long results stay fetchable

# Recovery of lost grading jobs (worker killed by OOM/deploy, Redis flushed, enqueue failed):
# reconcile.py looks for submissions still "pending_ai" after RECONCILE_REQUEUE_AFTER_SECONDS whose
# RQ job is no longer queued/running, and re-queues them -- at most RECONCILE_MAX_REQUEUES times,
# after which the submission is marked "grading_failed" (visible, and re-gradable via
# POST /submissions/{id}/regrade) instead of sitting in "pending_ai" forever.
RECONCILE_REQUEUE_AFTER_SECONDS = int(os.environ.get("RECONCILE_REQUEUE_AFTER_SECONDS", "300"))
RECONCILE_MAX_REQUEUES = int(os.environ.get("RECONCILE_MAX_REQUEUES", "3"))

# Per-request timeout (seconds) applied to every LLM provider SDK call, so one hung provider
# cannot hold a worker for longer than this (the SDK defaults are many minutes).
PROVIDER_TIMEOUT_SECONDS = float(os.environ.get("PROVIDER_TIMEOUT_SECONDS", "120"))

# ---------------------------------------------------------------------------
# PostgreSQL (Phase 3: CRM review workflow -- organizations, teachers,
# students, submissions). See db.py for the schema.
# ---------------------------------------------------------------------------
# Format: postgresql+psycopg2://user:password@host:port/dbname
#   export DATABASE_URL="postgresql+psycopg2://postgres:postgres@localhost:5432/homework_checker"
DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql+psycopg2://postgres:postgres@localhost:5432/homework_checker"
)

# ---------------------------------------------------------------------------
# Webhook delivery (pushing approved results back to a CRM)
# ---------------------------------------------------------------------------
# When a teacher approves a submission, we POST the final result to the
# organization's registered webhook_url (see db.py Organization.webhook_url).
# Delivery retries with backoff; GET /submissions?status=approved remains
# available as a polling fallback for CRMs that miss a webhook or don't
# want to run a receiving endpoint at all.
WEBHOOK_MAX_RETRIES = int(os.environ.get("WEBHOOK_MAX_RETRIES", "5"))
WEBHOOK_TIMEOUT_SECONDS = int(os.environ.get("WEBHOOK_TIMEOUT_SECONDS", "10"))

# Webhook authenticity: every delivery carries `X-Webhook-Timestamp` and
# `X-Webhook-Signature: sha256=<hex>` = HMAC-SHA256(secret, f"{timestamp}.{raw_body}").
# `secret` is derived per organization from this master value (no extra DB column), and the
# organization can read its own secret from GET /organizations/me/webhook-secret. If this is
# empty the INTERNAL_API_KEY is used as the master value; if both are empty no signature is sent.
WEBHOOK_SIGNING_SECRET = os.environ.get("WEBHOOK_SIGNING_SECRET", "")
# Webhook destinations must be https and resolve to public addresses (SSRF protection).
# Set to "true" only for local development/tests (allows http and loopback/private addresses).
WEBHOOK_ALLOW_PRIVATE_URLS = os.environ.get("WEBHOOK_ALLOW_PRIVATE_URLS", "false").lower() == "true"
# Comma-separated hostnames that are always allowed even if they resolve to a private address
# (e.g. a CRM running inside your own network): WEBHOOK_ALLOWED_HOSTS="crm.internal.example"
WEBHOOK_ALLOWED_HOSTS = [h.strip().lower() for h in os.environ.get("WEBHOOK_ALLOWED_HOSTS", "").split(",") if h.strip()]

# ---------------------------------------------------------------------------
# Local model server (Phase 2: your own GPU server, e.g. vLLM)
# ---------------------------------------------------------------------------
# vLLM (and most self-hosted inference servers) expose an OpenAI-compatible
# REST API, so we talk to it the same way we'd talk to OpenAI's API --
# just pointed at your own server's address instead.
#   export LOCAL_MODEL_BASE_URL="http://YOUR_SERVER_IP:8000/v1"
LOCAL_MODEL_BASE_URL = os.environ.get("LOCAL_MODEL_BASE_URL", "http://localhost:8000/v1")
# The API "key" your local server expects, if any. vLLM ignores this by
# default (any non-empty string works) unless you've configured it with
# --api-key yourself.
LOCAL_MODEL_API_KEY = os.environ.get("LOCAL_MODEL_API_KEY", "not-needed")

# Model names as registered on your local server. These are placeholders --
# set them to whatever you actually deploy (base model at first, later your
# fine-tuned checkpoint's name/path).
LOCAL_TEXT_MODEL = os.environ.get("LOCAL_TEXT_MODEL", "Qwen2.5-VL-32B-Instruct")
LOCAL_AUDIO_MODEL = os.environ.get("LOCAL_AUDIO_MODEL", "Qwen2-Audio-7B-Instruct")

# ---------------------------------------------------------------------------
# Training-data collection (for future fine-tuning)
# ---------------------------------------------------------------------------
# When enabled, every graded submission (input file + prompt + model
# output) is logged to TRAINING_DATA_DIR so you can build a fine-tuning
# dataset over time, with an optional teacher-correction step to turn raw
# model output into "gold" (verified-correct) training examples.
# OFF BY DEFAULT: the files are students' homework (names, handwriting, grades) stored
# unencrypted on local disk outside the database. Turn it on only with the consent of the
# school/parents and a retention plan.
#   export COLLECT_TRAINING_DATA="true"    # to enable
COLLECT_TRAINING_DATA = os.environ.get("COLLECT_TRAINING_DATA", "false").lower() == "true"
TRAINING_DATA_DIR = BASE_DIR / "training_data"
TRAINING_INPUTS_DIR = TRAINING_DATA_DIR / "inputs"
TRAINING_RAW_LOG = TRAINING_DATA_DIR / "raw_log.jsonl"
TRAINING_GOLD_LOG = TRAINING_DATA_DIR / "gold.jsonl"

for _dir in (TRAINING_DATA_DIR, TRAINING_INPUTS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Token usage tracking (cost visibility / reporting)
# ---------------------------------------------------------------------------
# Every graded submission logs how many input/output/cached tokens it
# used to TOKEN_USAGE_LOG, so you can see actual token spend per subject,
# per model, per day, etc. See token_usage_logger.py.
#   export TRACK_TOKEN_USAGE="false"   # to disable
TRACK_TOKEN_USAGE = os.environ.get("TRACK_TOKEN_USAGE", "true").lower() == "true"
TOKEN_USAGE_LOG = TRAINING_DATA_DIR / "token_usage_log.jsonl"

# ---------------------------------------------------------------------------
# Upload constraints
# ---------------------------------------------------------------------------
# Audio extensions are needed for "Speaking" language-skill assessments,
# where the student uploads a recording of themselves talking.
# .heic/.heif: the default photo format of iPhones -- converted to JPEG on upload
# (see utils.normalize_image), so every provider only ever sees a format it accepts.
ALLOWED_EXTENSIONS = {
    ".pdf", ".png", ".jpg", ".jpeg", ".webp", ".heic", ".heif", ".docx",
    ".mp3", ".wav", ".m4a", ".ogg",
}
AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".ogg"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".heic", ".heif"}
MAX_FILE_SIZE_MB = 25
# A photo with more pixels than this is refused BEFORE it is decoded (a small file can expand to
# hundreds of MB in memory: "decompression bomb"). 50 million pixels = a 7000x7000 picture.
MAX_IMAGE_PIXELS = int(os.environ.get("MAX_IMAGE_PIXELS", "50000000"))

# Field limits of POST /submissions (the database columns are 255 / 64 / 32 characters wide; a longer
# value used to end in HTTP 500). extra_instructions goes into every prompt, so it is capped too.
MAX_EXTRA_INSTRUCTIONS_CHARS = int(os.environ.get("MAX_EXTRA_INSTRUCTIONS_CHARS", "4000"))
MAX_REVIEWER_NOTE_CHARS = 4000

# A submission can be sent back for grading at most this many times (each one is a paid call that
# bypasses the result cache). 0 = unlimited.
MAX_REGRADES_PER_SUBMISSION = int(os.environ.get("MAX_REGRADES_PER_SUBMISSION", "3"))

# Per-organization abuse protection for POST /submissions (a leaked key or a CRM retry loop must not
# create unlimited paid grading). These are SAFETY nets: set them well above your real peak.
# Fixed windows counted in Redis; 0 disables a limit.
ORG_MAX_SUBMISSIONS_PER_MINUTE = int(os.environ.get("ORG_MAX_SUBMISSIONS_PER_MINUTE", "300"))
ORG_MAX_SUBMISSIONS_PER_DAY = int(os.environ.get("ORG_MAX_SUBMISSIONS_PER_DAY", "50000"))

# Original student files (photos of minors' homework) are deleted this many days after the teacher
# approved the result; orphan files (no submission refers to them) after one day. 0 = keep forever.
# Run `python retention.py` (daily) or POST /admin/retention.
UPLOAD_RETENTION_DAYS = int(os.environ.get("UPLOAD_RETENTION_DAYS", "90"))

# Uploaded photos are normalised before grading: rotated upright according to their
# EXIF tag, and shrunk if the longest side exceeds this many pixels (a 12-megapixel phone
# photo is both slower and pricier to send than ~1568px, which is enough to read handwriting).
# 1568 is also the longest side Claude itself downsizes images to, so for Claude nothing is
# lost; for Gemini/Qwen it cuts the image tokens (the biggest input cost of photo homework).
# Raise it (e.g. MAX_IMAGE_SIDE_PX=2400) if small handwriting starts being misread.
MAX_IMAGE_SIDE_PX = int(os.environ.get("MAX_IMAGE_SIDE_PX", "1568"))

# ---------------------------------------------------------------------------
# REST API (api.py / FastAPI)
# ---------------------------------------------------------------------------
# If set, callers of the REST API must send this value in an
# "X-Internal-Api-Key" header. This protects your Gemini quota/billing
# from being hit by anonymous internet traffic once you deploy api.py
# somewhere public.
#   export INTERNAL_API_KEY="some-long-random-string"
INTERNAL_API_KEY = os.environ.get("INTERNAL_API_KEY", "")
# The API FAILS CLOSED: with INTERNAL_API_KEY empty, every protected endpoint answers 503
# ("not configured") instead of silently being open to anyone. For local development only,
# you can explicitly opt in to an open API:
#   export ALLOW_UNAUTHENTICATED_API="true"
ALLOW_UNAUTHENTICATED_API = os.environ.get("ALLOW_UNAUTHENTICATED_API", "false").lower() == "true"
# Browser origins allowed to call the API (CORS), comma-separated. Empty (default) = no CORS
# headers at all: server-to-server callers (a CRM backend, the Streamlit app) don't need CORS.
#   export CORS_ALLOW_ORIGINS="https://crm.example.com,https://teacher.example.com"
CORS_ALLOW_ORIGINS = [o.strip() for o in os.environ.get("CORS_ALLOW_ORIGINS", "").split(",") if o.strip()]

# ---------------------------------------------------------------------------
# Production mode
# ---------------------------------------------------------------------------
# APP_ENV=production makes the API REFUSE TO START with an unsafe configuration (see
# production_config_problems) instead of running with silent fallbacks. Leave it unset for
# local development and tests.
APP_ENV = os.environ.get("APP_ENV", "development").strip().lower()
IS_PRODUCTION = APP_ENV == "production"

_DEFAULT_DATABASE_URL = "postgresql+psycopg2://postgres:postgres@localhost:5432/homework_checker"


def production_config_problems() -> list:
    """Human-readable reasons this configuration is not safe for production (empty list = fine)."""
    problems = []
    if not INTERNAL_API_KEY:
        problems.append("INTERNAL_API_KEY is empty (the admin endpoints would answer 503).")
    if ALLOW_UNAUTHENTICATED_API:
        problems.append("ALLOW_UNAUTHENTICATED_API=true opens the paid endpoints to anyone.")
    if not WEBHOOK_SIGNING_SECRET:
        problems.append(
            "WEBHOOK_SIGNING_SECRET is empty: webhooks would be unsigned or signed with the admin key, "
            "and rotating the admin key would silently break every CRM's signature check."
        )
    elif WEBHOOK_SIGNING_SECRET == INTERNAL_API_KEY:
        problems.append("WEBHOOK_SIGNING_SECRET must differ from INTERNAL_API_KEY.")
    if DATABASE_URL == _DEFAULT_DATABASE_URL:
        problems.append("DATABASE_URL is the development default (postgres:postgres@localhost).")
    if WEBHOOK_ALLOW_PRIVATE_URLS:
        problems.append("WEBHOOK_ALLOW_PRIVATE_URLS=true lets tenants point webhooks at your internal network.")
    return problems


# ---------------------------------------------------------------------------
# Grade scale (kept here so report_generator + app agree on labels/colors)
# ---------------------------------------------------------------------------
GRADE_SCALE = [
    (90, 100, "Excellent", "#16a34a"),
    (80, 89, "Very Good", "#22c55e"),
    (70, 79, "Good", "#84cc16"),
    (60, 69, "Satisfactory", "#eab308"),
    (50, 59, "Needs Improvement", "#f97316"),
    (0, 49, "Significant Improvement Needed", "#ef4444"),
]


def grade_for_score(score: float):
    for low, high, label, color in GRADE_SCALE:
        if low <= score <= high:
            return label, color
    return "Unknown", "#6b7280"
