"""
hwchecker/providers/qwen.py
QwenProvider -- Alibaba Cloud's Qwen models via DashScope's OpenAI-
compatible endpoint. Used for the Difficulty Router (Qwen3.7 Flash) and
Primary-Easy tier (Qwen3.5 Flash) per the model-selection table.
"""

from typing import Optional

from ._openai_protocol_base import OpenAIProtocolProvider

# Rough published per-1M-token pricing (USD), for cost estimation/logging
# only. Update if the provider changes pricing.
#
# Alibaba Cloud Model Studio, International (Singapore), standard tier,
# checked Oct 2026. qwen3.7-flash is tiered by prompt length: $0.03/$0.13
# up to 32K input tokens ($0.10/$0.40 for 32K-256K) -- grading prompts are
# well under 32K, so the first tier applies.
_PRICING_PER_1M = {
    "qwen3.7-flash": {"input": 0.03, "output": 0.13},
    "qwen3.5-flash": {"input": 0.10, "output": 0.40},
    "qwen3.8-flash": {"input": 0.15, "output": 0.47},
    "qwen3.7-plus": {"input": 0.20, "output": 1.60},   # not re-checked (the list price on the pricing page carries a promo)
}


# Thinking-token budgets for the named levels. "high"/"max" leave the budget to the model.
_THINKING_BUDGETS = {"minimal": 512, "low": 1024, "medium": 4096}


class QwenProvider(OpenAIProtocolProvider):
    base_url = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    api_key_env = "QWEN_API_KEY"
    supports_vision = True

    def __init__(self, model_name: str = "qwen3.5-flash", api_key=None, thinking: Optional[str] = None):
        self.pricing_per_1m = _PRICING_PER_1M.get(model_name, {"input": 0.10, "output": 0.20})
        super().__init__(model_name=model_name, api_key=api_key, thinking=thinking)

    def _thinking_extra_body(self) -> Optional[dict]:
        """Qwen3.5/3.7 Flash think by default, and the thinking tokens are billed as output.
        "off" -> `enable_thinking: false`; a level or a number -> a capped `thinking_budget`."""
        level = self.thinking
        if level is None:
            return None
        if level == "off":
            return {"enable_thinking": False}
        budget = int(level) if level.isdigit() else _THINKING_BUDGETS.get(level)
        if budget:
            return {"enable_thinking": True, "thinking_budget": budget}
        return None   # "high" / "max" / unknown word: keep the provider default
