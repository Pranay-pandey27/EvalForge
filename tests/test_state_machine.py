from __future__ import annotations

import pytest

from evalforge.models import AgentState
from evalforge.state_machine import InvalidTransition, StateMachine


def test_valid_initial_path() -> None:
    machine = StateMachine()
    machine.transition(AgentState.INSPECT_REPOSITORY)
    machine.transition(AgentState.RETRIEVE_MEMORY)
    machine.transition(AgentState.PLAN_REPAIR)
    assert machine.state is AgentState.PLAN_REPAIR


def test_cannot_skip_directly_to_finished() -> None:
    machine = StateMachine()
    with pytest.raises(InvalidTransition):
        machine.transition(AgentState.FINISHED)
