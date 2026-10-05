from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from evalforge.evaluation import aggregate, analyze_runs
from evalforge.models import (
    Condition,
    RunSummary,
    TerminationReason,
    Usage,
    VerificationResult,
)


def summary(run_id: str, condition: Condition, resolved: bool) -> RunSummary:
    verification = VerificationResult(
        command=["pytest"],
        exit_code=0 if resolved else 1,
        passed=resolved,
        output="4 passed" if resolved else "1 failed, 3 passed",
        runtime_seconds=0.1,
        tests_collected=4,
        tests_passed=4 if resolved else 3,
        tests_failed=0 if resolved else 1,
    )
    now = datetime.now(UTC)
    return RunSummary(
        run_id=run_id,
        task_id=f"task-{run_id}",
        condition=condition,
        started_at=now,
        finished_at=now,
        termination_reason=(
            TerminationReason.TESTS_PASSED if resolved else TerminationReason.EVALUATOR_FAILED
        ),
        resolved=resolved,
        evaluator_verification=verification,
        iterations=2,
        tool_calls=4,
        usage=Usage(input_tokens=10, output_tokens=5, estimated_cost_usd=0.01),
        runtime_seconds=1.0,
        trace_path="trace.jsonl",
        workspace_path="workspace",
        provider="FakeProvider",
    )


def test_aggregate_uses_evaluator_test_counts() -> None:
    rows = aggregate(
        [
            summary("a", Condition.BASELINE, True),
            summary("b", Condition.BASELINE, False),
        ]
    )
    assert rows[0]["task_resolution_rate"] == 0.5
    assert rows[0]["test_pass_rate"] == 7 / 8
    assert rows[0]["average_tokens"] == 15


def test_analysis_writes_trace_derived_artifacts(tmp_path: Path) -> None:
    run_root = tmp_path / "runs"
    target = run_root / "one"
    target.mkdir(parents=True)
    (target / "summary.json").write_text(
        summary("a", Condition.BASELINE, True).model_dump_json(), encoding="utf-8"
    )
    output = tmp_path / "analysis"
    analyze_runs(run_root, output)
    assert (output / "aggregate.json").exists()
    assert (output / "ablation_table.csv").exists()
    assert (output / "results.md").exists()
