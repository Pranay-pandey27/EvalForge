"""Optional LiteLLM adapter with strict structured-output parsing."""

from __future__ import annotations

import json
import re
from typing import Any

from evalforge.models import AgentDecision, ProviderResponse, Usage
from evalforge.prompts import SYSTEM_PROMPT, build_user_prompt
from evalforge.providers.base import AgentContext

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


class LiteLLMProvider:
    def __init__(
        self,
        model: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 1400,
        seed: int | None = None,
    ) -> None:
        if not model.strip():
            raise ValueError("model must not be empty")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed

    @property
    def exhausted(self) -> bool:
        return False

    def next_action(self, context: AgentContext) -> ProviderResponse:
        try:
            from litellm import completion
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("install EvalForge with the 'llm' extra") from exc
        kwargs: dict[str, Any] = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(context)},
            ],
        }
        if self.seed is not None:
            kwargs["seed"] = self.seed
        response = completion(**kwargs)
        text = response.choices[0].message.content or ""
        cleaned = _FENCE.sub("", text.strip()).strip()
        try:
            decision = AgentDecision.model_validate(json.loads(cleaned))
        except (json.JSONDecodeError, ValueError) as exc:
            raise RuntimeError(f"provider returned invalid decision JSON: {exc}") from exc
        usage = getattr(response, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", None) if usage else None
        completion_tokens = getattr(usage, "completion_tokens", None) if usage else None
        cost = getattr(response, "_hidden_params", {}).get("response_cost")
        return ProviderResponse(
            decision=decision,
            raw_text=text,
            usage=Usage(
                input_tokens=prompt_tokens,
                output_tokens=completion_tokens,
                estimated_cost_usd=cost,
            ),
        )
