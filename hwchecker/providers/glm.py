"""
hwchecker/providers/glm.py
GLMProvider -- Zhipu AI (Z.ai)'s GLM models, via their OpenAI-compatible
endpoint. Used for the Primary-Hard tier (GLM-5.2) per the model-
selection table: proof/optimization-level reasoning.
"""

from typing import Optional

from ._openai_protocol_base import OpenAIProtocolProvider

# Rough published per-1M-token pricing (USD), for cost estimation/logging
# only. Update if the provider changes pricing.
#
# Z.ai list prices, Oct 2026 (via third-party price trackers -- verify on
# docs.z.ai). The old 0.60/2.20 figures were GLM-4.x prices.
_PRICING_PER_1M = {
    "glm-5.2": {"input": 1.40, "output": 4.40},
    "glm-5.3": {"input": 1.40, "output": 4.40},
}


class GLMProvider(OpenAIProtocolProvider):
    base_url = "https://api.z.ai/api/paas/v4"
    api_key_env = "GLM_API_KEY"
    supports_vision = False  # GLM's standard chat models are text-only; use a GLM-V variant if you need vision.
    # The router checks this flag: photos/PDF scans are never sent to this tier (a vision-capable
    # tier is used instead), typed DOCX homework still can be.

    def __init__(self, model_name: str = "glm-5.2", api_key=None, thinking: Optional[str] = None):
        self.pricing_per_1m = _PRICING_PER_1M.get(model_name, {"input": 1.40, "output": 4.40})
        super().__init__(model_name=model_name, api_key=api_key, thinking=thinking)

    def _thinking_extra_body(self) -> Optional[dict]:
        """GLM-5.x thinks by default at its maximum effort. "off" -> thinking disabled; a level ->
        `reasoning_effort` (docs.z.ai: max is the default, "high" is the next step down)."""
        level = self.thinking
        if level is None:
            return None
        if level == "off":
            return {"thinking": {"type": "disabled"}}
        if level in ("minimal", "low", "medium", "high", "max"):
            return {"reasoning_effort": level}
        return None
