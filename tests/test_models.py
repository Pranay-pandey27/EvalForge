from __future__ import annotations

import pytest
from pydantic import ValidationError

from evalforge.models import ActionKind, AgentDecision, Budget


def test_budget_rejects_zero_limits() -> None:
    with pytest.raises(ValidationError):
        Budget(max_iterations=0)


def test_tool_decision_requires_a_tool_name() -> None:
    with pytest.raises(ValidationError):
        AgentDecision(kind=ActionKind.TOOL)


def test_submit_cannot_smuggle_a_tool_name() -> None:
    with pytest.raises(ValidationError):
        AgentDecision(kind=ActionKind.SUBMIT, tool_name="run_tests")
