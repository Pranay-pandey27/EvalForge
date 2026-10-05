from __future__ import annotations

import json
from pathlib import Path

from evalforge.benchmark import load_task, read_script
from evalforge.models import AgentDecision, TerminationReason
from evalforge.orchestrator import Orchestrator
from evalforge.providers.mock import ScriptedProvider

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TASK_ROOT = PROJECT_ROOT / "benchmarks" / "development" / "task_001"


def test_scripted_repair_requires_both_verifiers(tmp_path: Path) -> None:
    task = load_task(TASK_ROOT)
    decisions = [
        AgentDecision.model_validate(item)
        for item in read_script(TASK_ROOT / "scripted_actions.json")
    ]
    summary = Orchestrator(
        task=task,
        provider=ScriptedProvider(decisions),
        run_root=tmp_path,
    ).run()

    assert summary.resolved
    assert summary.termination_reason is TerminationReason.TESTS_PASSED
    assert summary.visible_verification is not None
    assert summary.visible_verification.passed
    assert summary.evaluator_verification is not None
    assert summary.evaluator_verification.passed
    assert summary.changed_files == ["scientific_stats/normalization.py"]
    events = [json.loads(line) for line in Path(summary.trace_path).read_text().splitlines()]
    assert any(event["event_type"] == "submission_rejected" for event in events) is False


def test_submit_without_passing_tests_is_rejected(tmp_path: Path) -> None:
    task = load_task(TASK_ROOT)
    provider = ScriptedProvider([AgentDecision(kind="submit")])
    summary = Orchestrator(task=task, provider=provider, run_root=tmp_path).run()
    assert not summary.resolved
    assert summary.termination_reason is TerminationReason.INVALID_ACTION
