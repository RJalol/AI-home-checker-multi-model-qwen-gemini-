"""
hwchecker/providers/openai.py
OpenAIProvider -- used as the Fallback (cross-provider) tier per the
model-selection table: if the primary provider for a tier is down, this
is what the circuit breaker (infra/circuit_breaker.py) switches to.
"""

from ._openai_protocol_base import OpenAIProtocolProvider

# Rough published per-1M-token pricing (USD), for cost estimation/logging
# only. Update if the provider changes pricing.
_PRICING_PER_1M = {
    "gpt-5.4-mini": {"input": 0.75, "output": 4.50},
    "gpt-5.6": {"input": 4.00, "output": 20.00},
    "gpt-5.6-luna": {"input": 0.20, "output": 1.20},
    # Added Oct 2026 (third-party price trackers -- verify on openai.com/api/pricing):
    "gpt-6-luna": {"input": 0.10, "output": 0.50},
    "gpt-6.1-terra": {"input": 1.00, "output": 5.00},
}


class OpenAIProvider(OpenAIProtocolProvider):
    base_url = "https://api.openai.com/v1"
    api_key_env = "OPENAI_API_KEY"
    supports_vision = True

    # OpenAI's current models take `max_completion_tokens` (the older `max_tokens` is
    # rejected by the reasoning-model families) and only accept the default sampling
    # settings, so none are sent. The base class still adapts if the API disagrees.
    # The budget also covers hidden reasoning tokens, hence the larger number.
    token_limit_param = "max_completion_tokens"
    sends_sampling_params = False
    max_output_tokens = 16384

    def __init__(self, model_name: str = "gpt-5.4-mini", api_key=None, thinking=None):
        # `thinking` is accepted so the registry can treat every tier alike, but this provider has no
        # such knob: it is the cross-provider fallback and keeps OpenAI's own reasoning default.
        self.pricing_per_1m = _PRICING_PER_1M.get(model_name, {"input": 0.75, "output": 4.50})
        super().__init__(model_name=model_name, api_key=api_key, thinking=thinking)
