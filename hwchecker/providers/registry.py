"""
hwchecker/providers/registry.py
PROVIDER_REGISTRY: dict[str, AIProvider] -- the ONE place tier-to-
provider assignments live, per spec section 5 ("Model tanlovi -- QAT'IY
belgilangan"). Swapping a provider is meant to be a one-line change
here; router/grading_router.py and everything else looks providers up
by tier name ("router", "easy", "medium", "hard", "hard_vision", "audio",
"verification", "fallback"), never by importing a concrete provider class
directly.

Cost-optimised defaults (see README "Narxni optimallashtirish"):

| Tier          | Model                 | Thinking | Role                                       |
|---------------|-----------------------|----------|---------------------------------------------|
| router        | Qwen3.7 Flash         | off      | difficulty classification (cheap)           |
| easy          | Qwen3.5 Flash         | off      | correct/incorrect, basic error (~60%)       |
| medium        | Qwen3.7 Plus          | low      | multi-step grading, partial credit (~30%)   |
| medium_alt    | Gemini 3.5 Flash-Lite | low      | medium tier's stand-in without a Qwen key   |
| hard          | GLM-5.2 (text only)   | high     | proof/optimization-level grading (~10%)     |
| hard_vision   | Gemini 3.6 Flash      | high     | hard questions that arrive as photo / scan  |
| audio         | Gemini 3.6 Flash      | low      | recordings (Flash-Lite is too weak for them)|
| verification  | Claude Haiku 5.5      | default  | second opinion on low-confidence results    |
| verification_gemini | Gemini 3.5 Flash-Lite | low | second opinion when Claude is not available / graded it |
| verification_qwen   | Qwen3.7 Plus          | low      | second opinion for results graded by Gemini  |
| fallback      | GPT-5.4 mini          | default  | cross-provider fallback if primary down     |

A second opinion always comes from a DIFFERENT vendor than the model that graded the work (a
Gemini model re-checking a Gemini grade shares its blind spots). The router tries
verification -> verification_gemini -> verification_qwen and takes the first one that is
configured, can read the input and is not from the grader's vendor; with only Qwen and Gemini
keys the two vendors check each other (Qwen's grades by Gemini, Gemini's by Qwen).

Every tier can be re-pointed WITHOUT a code change, with environment variables
(TIER is the upper-cased tier name, e.g. HARD_VISION):

    HWCHECKER_<TIER>_MODEL      model name, same provider as the default
    HWCHECKER_<TIER>_PROVIDER   qwen | gemini | glm | claude | openai (needs _MODEL too)
    HWCHECKER_<TIER>_THINKING   off | low | medium | high | max | <token budget> | default

Providers are instantiated lazily (only when first requested, and only
once, cached) so a missing API key for a provider tier you never
actually use doesn't crash the whole app at import time -- you only hit
that error when something actually tries to use that tier.
"""

import os
from typing import Callable, Dict, Optional, Tuple

from .base import AIProvider, AIProviderError, ProviderUnavailableError, normalize_thinking

# tier -> (provider module, model name, thinking setting). This table is the single source of
# truth for the defaults; the environment variables above override single entries.
TIER_DEFAULTS: Dict[str, Tuple[str, str, Optional[str]]] = {
    "router": ("qwen", "qwen3.7-flash", "off"),
    "easy": ("qwen", "qwen3.5-flash", "off"),
    "medium": ("qwen", "qwen3.7-plus", "low"),
    # The Gemini all-rounder: the medium tier's stand-in when there is no Qwen key (a Gemini-only
    # deployment still grades everything) and its second choice for photos/audio after a failure.
    "medium_alt": ("gemini", "gemini-3.5-flash-lite", "low"),
    "hard": ("glm", "glm-5.2", "high"),
    "hard_vision": ("gemini", "gemini-3.6-flash", "high"),
    "audio": ("gemini", "gemini-3.6-flash", "low"),
    "verification": ("claude", "claude-haiku-5-5", None),
    # Cross-vendor second opinions for deployments without an Anthropic key (see the router's
    # _pick_verifier): used in this order after "verification", never for the grader's own vendor.
    "verification_gemini": ("gemini", "gemini-3.5-flash-lite", "low"),
    "verification_qwen": ("qwen", "qwen3.7-plus", "low"),
    "fallback": ("openai", "gpt-5.4-mini", None),
}

_KNOWN_MODULES = ("qwen", "gemini", "glm", "claude", "openai")


def resolve_tier_config(tier: str) -> Tuple[str, str, Optional[str]]:
    """(provider module, model name, normalised thinking) for `tier`, defaults + env overrides."""
    if tier not in TIER_DEFAULTS:
        raise ValueError(f"Unknown provider tier '{tier}'. Known tiers: {list(TIER_DEFAULTS)}")
    module, model, thinking = TIER_DEFAULTS[tier]
    prefix = f"HWCHECKER_{tier.upper()}_"
    env_module = os.environ.get(prefix + "PROVIDER", "").strip().lower()
    env_model = os.environ.get(prefix + "MODEL", "").strip()
    if env_module:
        if env_module not in _KNOWN_MODULES:
            raise ValueError(f"{prefix}PROVIDER='{env_module}' is not one of {list(_KNOWN_MODULES)}.")
        if env_module != module and not env_model:
            raise ValueError(f"{prefix}PROVIDER={env_module} also needs {prefix}MODEL (the default model belongs to '{module}').")
        module = env_module
    if env_model:
        model = env_model
    env_thinking = os.environ.get(prefix + "THINKING")
    if env_thinking is not None and env_thinking.strip():
        thinking = env_thinking
    return module, model, normalize_thinking(thinking)


def config_fingerprint() -> str:
    """A short string that changes whenever any tier's provider, model or thinking level changes.
    The result cache includes it in its key, so switching models never serves an answer that an
    earlier (possibly different, cheaper or weaker) set-up produced."""
    parts = []
    for tier in TIER_DEFAULTS:
        try:
            module, model, thinking = resolve_tier_config(tier)
            parts.append(f"{tier}={module}:{model}:{thinking or 'default'}")
        except ValueError as exc:   # a broken setting must not stop grading; it still changes the key
            parts.append(f"{tier}=invalid:{exc}")
    return "|".join(parts)


def describe_tiers() -> Dict[str, dict]:
    """What every tier is currently set to (defaults + env overrides); for logs, health pages, docs."""
    out = {}
    for tier in TIER_DEFAULTS:
        try:
            module, model, thinking = resolve_tier_config(tier)
            out[tier] = {"provider": module, "model": model, "thinking": thinking or "default"}
        except ValueError as exc:
            out[tier] = {"error": str(exc)}
    return out


# Provider name is read from config/env, never hardcoded elsewhere in the
# codebase -- this dict of factories is the single source of truth.
_PROVIDER_FACTORIES: Dict[str, Callable[[], AIProvider]] = {
    tier: (lambda tier=tier: _build_tier(tier)) for tier in TIER_DEFAULTS
}

_instances: Dict[str, AIProvider] = {}


def _build_tier(tier: str) -> AIProvider:
    module, model, thinking = resolve_tier_config(tier)
    provider = _make(module, model, thinking)
    provider.tier = tier   # lets the router tell which tier produced a result (e.g. to skip a pointless re-check)
    provider.vendor = module   # lets the router pick a second opinion from a different vendor
    return provider


def _make(provider_module: str, model_name: str, thinking: Optional[str] = None) -> AIProvider:
    """Imports the concrete provider class lazily and instantiates it."""
    if provider_module == "qwen":
        from .qwen import QwenProvider

        return QwenProvider(model_name=model_name, thinking=thinking)
    if provider_module == "gemini":
        from .gemini import GeminiProvider

        return GeminiProvider(model_name=model_name, thinking=thinking)
    if provider_module == "glm":
        from .glm import GLMProvider

        return GLMProvider(model_name=model_name, thinking=thinking)
    if provider_module == "claude":
        from .claude import ClaudeProvider

        return ClaudeProvider(model_name=model_name, thinking=thinking)
    if provider_module == "openai":
        from .openai import OpenAIProvider

        return OpenAIProvider(model_name=model_name, thinking=thinking)
    raise ValueError(f"Unknown provider module '{provider_module}'.")


def get_provider(tier: str) -> AIProvider:
    """
    Looks up (and lazily instantiates + caches) the provider assigned to
    a tier ("router" | "easy" | "medium" | "hard" | "hard_vision" | "audio" |
    "verification" | "fallback"). This is the ONLY function router/grading_router.py
    should call to obtain a provider -- never import a concrete provider
    class directly outside this module.
    """
    if tier not in _PROVIDER_FACTORIES:
        raise ValueError(f"Unknown provider tier '{tier}'. Known tiers: {list(_PROVIDER_FACTORIES)}")
    if tier not in _instances:
        try:
            _instances[tier] = _PROVIDER_FACTORIES[tier]()
        except AIProviderError:
            raise
        except Exception as exc:   # e.g. an ImportError from a half-installed SDK, or a bad HWCHECKER_*_MODEL setting
            raise ProviderUnavailableError(f"Could not initialise the '{tier}' provider: {exc}") from exc
    return _instances[tier]


def is_tier_available(tier: str) -> bool:
    """True if the provider for `tier` can be created right now (API key set, SDK installed)."""
    try:
        get_provider(tier)
        return True
    except AIProviderError:
        return False


def reset_registry() -> None:
    """Clears cached instances -- mainly useful for tests that swap env vars/mocks between runs."""
    _instances.clear()


# Exposed for introspection/tests; prefer get_provider(tier) in real code.
PROVIDER_REGISTRY = _PROVIDER_FACTORIES
