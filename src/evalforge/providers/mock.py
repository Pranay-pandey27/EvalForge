"""Deterministic provider used by offline tests and demonstrations."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable

from evalforge.models import AgentDecision, ProviderResponse
from evalforge.providers.base import AgentContext


class ScriptedProvider:
    """Replay pre-validated decisions without model credentials."""

    def __init__(self, decisions: Iterable[AgentDecision | dict[str, object]]) -> None:
        self.model: str | None = None
        self._decisions = deque(
            decision
            if isinstance(decision, AgentDecision)
            else AgentDecision.model_validate(decision)
            for decision in decisions
        )

    @property
    def exhausted(self) -> bool:
        return not self._decisions

    def next_action(self, context: AgentContext) -> ProviderResponse:
        del context
        if self.exhausted:
            raise RuntimeError("scripted provider has no remaining actions")
        return ProviderResponse(decision=self._decisions.popleft())
