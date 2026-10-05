"""Provider-independent agent interface."""

from __future__ import annotations

from typing import Protocol

from pydantic import Field

from evalforge.models import (
    AgentState,
    ProviderResponse,
    StrictModel,
    ToolResult,
    VerificationResult,
)


class AgentContext(StrictModel):
    task_id: str
    issue: str
    state: AgentState
    repository_files: list[str] = Field(default_factory=list)
    previous_results: list[ToolResult] = Field(default_factory=list)
    latest_verification: VerificationResult | None = None
    iteration: int = 0
    tool_calls: int = 0
    memory: list[str] = Field(default_factory=list)
    require_structured_diagnosis: bool = False


class Provider(Protocol):
    model: str | None

    @property
    def exhausted(self) -> bool:
        """Whether the provider can produce another decision."""

    def next_action(self, context: AgentContext) -> ProviderResponse:
        """Choose the next structured agent action."""
