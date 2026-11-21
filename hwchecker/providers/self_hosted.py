"""
hwchecker/providers/self_hosted.py
SelfHostedProvider -- extension point for a future self-hosted GPU
server (vLLM/Ollama serving an open-weight model), per spec section 1
("Buyurtmachi hozircha o'z serverida model ishga tushirishga tayyor
emas") and constraint #5 ("Self-hosted provider'ni default yoqilgan
holda joriy qilmang -- feature-flag orqali o'chirilgan holatda
qoldiring").

This class is intentionally NOT registered in providers/registry.py's
PROVIDER_REGISTRY by default. To enable it later:
  1. Stand up an OpenAI-compatible inference server (see the project's
     existing local_checker.py / README "Phase 2" section for a fully
     worked example using vLLM or Ollama).
  2. Set HWCHECKER_ENABLE_SELF_HOSTED=true and SELF_HOSTED_BASE_URL.
  3. Add an entry for it in providers/registry.py's PROVIDER_REGISTRY --
     that one-line change is the only place this needs wiring in.
"""

import os

from ._openai_protocol_base import OpenAIProtocolProvider
from .base import ProviderUnavailableError

SELF_HOSTED_ENABLED = os.environ.get("HWCHECKER_ENABLE_SELF_HOSTED", "false").lower() == "true"


class SelfHostedProvider(OpenAIProtocolProvider):
    """
    Points at your own OpenAI-compatible inference server. Disabled by
    default (see SELF_HOSTED_ENABLED above) -- instantiating this while
    disabled raises immediately, so it can't be reached accidentally via
    a misconfigured PROVIDER_REGISTRY entry.
    """

    api_key_env = "SELF_HOSTED_API_KEY"  # optional; most self-hosted servers don't check this
    supports_vision = True

    def __init__(self, model_name: str, api_key=None, base_url: str = "", thinking=None):
        if not SELF_HOSTED_ENABLED:
            raise ProviderUnavailableError(
                "SelfHostedProvider is disabled (HWCHECKER_ENABLE_SELF_HOSTED is not 'true'). "
                "This is intentional per spec constraint #5 -- flip the flag once you actually "
                "have a self-hosted server running and have decided to use it."
            )
        self.base_url = base_url or os.environ.get("SELF_HOSTED_BASE_URL", "")
        if not self.base_url:
            raise ProviderUnavailableError("SELF_HOSTED_BASE_URL is not set.")
        super().__init__(model_name=model_name, api_key=api_key or "not-needed", thinking=thinking)

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        # Self-hosted compute isn't billed per-token -- report 0 for the
        # per-request cost-tracking log; your actual cost is GPU-hour
        # amortization, tracked separately (see infra/monitoring.py).
        return 0.0
