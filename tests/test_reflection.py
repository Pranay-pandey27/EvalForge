from __future__ import annotations

import json
from pathlib import Path

from evalforge.benchmark import load_task, read_script
from evalforge.models import AgentDecision, Condition, TerminationReason
from evalforge.orchestrator import Orchestrator
from evalforge.providers.mock import ScriptedProvider

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TASK_ROOT = PROJECT_ROOT / "benchmarks" / "development" / "task_001"


def test_verifier_condition_rejects_unstructured_diagnosis(tmp_path: Path) -> None:
    task = load_task(TASK_ROOT)
    patch = (TASK_ROOT / "reference_patch.diff").read_text(encoding="utf-8")
    decisions = [
        AgentDecision(kind="tool", tool_name="run_tests"),
        AgentDecision(
            kind="tool",
            tool_name="apply_patch",
            diagnosis="the test failed",
            arguments={"patch": patch},
        ),
    ]
    summary = Orchestrator(
        task=task,
        provider=ScriptedProvider(decisions),
        run_root=tmp_path,
        condition=Condition.MEMORY_VERIFIER,
    ).run()
    assert not summary.resolved
    assert summary.termination_reason is TerminationReason.PROVIDER_ABORT
    events = [json.loads(line) for line in Path(summary.trace_path).read_text().splitlines()]
    assert any(event["event_type"] == "diagnosis_rejected" for event in events)
    assert summary.changed_files == []


def test_evidence_linked_script_passes_verifier_condition(tmp_path: Path) -> None:
    task = load_task(TASK_ROOT)
    decisions = [
        AgentDecision.model_validate(item)
        for item in read_script(TASK_ROOT / "scripted_actions.json")
    ]
    summary = Orchestrator(
        task=task,
        provider=ScriptedProvider(decisions),
        run_root=tmp_path,
        condition=Condition.MEMORY_VERIFIER,
    ).run()
    assert summary.resolved


def test_diagnosis_must_quote_latest_output(tmp_path: Path) -> None:
    task = load_task(TASK_ROOT)
    patch = (TASK_ROOT / "reference_patch.diff").read_text(encoding="utf-8")
    decisions = [
        AgentDecision(kind="tool", tool_name="run_tests"),
        AgentDecision(
            kind="tool",
            tool_name="apply_patch",
            diagnosis={
                "failed_test": "invented::test",
                "evidence": ["invented output"],
                "mechanism": "invented mechanism",
                "next_change": "change an axis",
            },
            arguments={"patch": patch},
        ),
    ]
    summary = Orchestrator(
        task=task,
        provider=ScriptedProvider(decisions),
        run_root=tmp_path,
        condition=Condition.MEMORY_VERIFIER,
    ).run()
    assert not summary.resolved
    assert summary.changed_files == []
