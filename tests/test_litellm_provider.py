from __future__ import annotations

import sys
from types import SimpleNamespace

from evalforge.models import AgentState
from evalforge.providers.base import AgentContext
from evalforge.providers.litellm import LiteLLMProvider


def test_litellm_provider_parses_fenced_structured_decision(monkeypatch) -> None:
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content='```json\n{"kind":"tool","tool_name":"run_tests","arguments":{}}\n```'
                )
            )
        ],
        usage=SimpleNamespace(prompt_tokens=11, completion_tokens=7),
        _hidden_params={"response_cost": 0.02},
    )
    monkeypatch.setitem(
        sys.modules,
        "litellm",
        SimpleNamespace(completion=lambda **kwargs: response),
    )
    provider = LiteLLMProvider("provider/model", seed=3)
    result = provider.next_action(
        AgentContext(task_id="task", issue="fix", state=AgentState.PLAN_REPAIR)
    )
    assert result.decision.tool_name == "run_tests"
    assert result.usage.total_tokens == 18
    assert result.usage.estimated_cost_usd == 0.02
