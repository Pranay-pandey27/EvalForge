"""Explicit, validated state transitions for repair runs."""

from __future__ import annotations

from dataclasses import dataclass

from evalforge.models import AgentState


class InvalidTransition(ValueError):
    """Raised when a run attempts an impossible state transition."""


ALLOWED_TRANSITIONS: dict[AgentState, frozenset[AgentState]] = {
    AgentState.TASK_RECEIVED: frozenset({AgentState.INSPECT_REPOSITORY}),
    AgentState.INSPECT_REPOSITORY: frozenset({AgentState.RETRIEVE_MEMORY}),
    AgentState.RETRIEVE_MEMORY: frozenset({AgentState.PLAN_REPAIR}),
    AgentState.PLAN_REPAIR: frozenset(
        {AgentState.EDIT_FILES, AgentState.RUN_VERIFIER, AgentState.FINISHED}
    ),
    AgentState.EDIT_FILES: frozenset(
        {AgentState.EDIT_FILES, AgentState.RUN_VERIFIER, AgentState.DIAGNOSE_RESULT}
    ),
    AgentState.RUN_VERIFIER: frozenset({AgentState.DIAGNOSE_RESULT, AgentState.FINISHED}),
    AgentState.DIAGNOSE_RESULT: frozenset(
        {
            AgentState.PLAN_REPAIR,
            AgentState.EDIT_FILES,
            AgentState.RUN_VERIFIER,
            AgentState.FINISHED,
        }
    ),
    AgentState.FINISHED: frozenset(),
}


@dataclass(slots=True)
class StateMachine:
    state: AgentState = AgentState.TASK_RECEIVED

    def transition(self, target: AgentState) -> AgentState:
        if target not in ALLOWED_TRANSITIONS[self.state]:
            raise InvalidTransition(f"cannot transition from {self.state} to {target}")
        self.state = target
        return self.state
