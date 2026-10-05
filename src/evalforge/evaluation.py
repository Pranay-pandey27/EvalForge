"""Trace-derived metrics, tables, plots, and results report."""

from __future__ import annotations

import csv
import json
import math
import os
import statistics
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from evalforge.models import RunSummary


def load_summaries(run_root: Path) -> list[RunSummary]:
    summaries: list[RunSummary] = []
    for path in sorted(run_root.resolve().rglob("summary.json")):
        summaries.append(RunSummary.model_validate_json(path.read_text(encoding="utf-8")))
    if not summaries:
        raise ValueError(f"no summary.json files found under {run_root}")
    return summaries


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def aggregate(summaries: list[RunSummary]) -> list[dict[str, Any]]:
    grouped: dict[str, list[RunSummary]] = defaultdict(list)
    for summary in summaries:
        grouped[summary.condition.value].append(summary)
    rows: list[dict[str, Any]] = []
    for condition, runs in sorted(grouped.items()):
        evaluator_results = [run.evaluator_verification for run in runs]
        tests_passed = sum(
            result.tests_passed or 0 for result in evaluator_results if result is not None
        )
        tests_collected = sum(
            result.tests_collected or 0 for result in evaluator_results if result is not None
        )
        token_values = [
            float(run.usage.total_tokens) for run in runs if run.usage.total_tokens is not None
        ]
        cost_values = [
            float(run.usage.estimated_cost_usd)
            for run in runs
            if run.usage.estimated_cost_usd is not None
        ]
        failure_counts = Counter(run.termination_reason.value for run in runs if not run.resolved)
        rows.append(
            {
                "condition": condition,
                "runs": len(runs),
                "unique_tasks": len({run.task_id for run in runs}),
                "task_resolution_rate": sum(run.resolved for run in runs) / len(runs),
                "test_pass_rate": (tests_passed / tests_collected if tests_collected else None),
                "patch_validity_rate": sum(run.patch_valid for run in runs) / len(runs),
                "average_iterations": _mean([float(run.iterations) for run in runs]),
                "average_runtime_seconds": _mean([run.runtime_seconds for run in runs]),
                "average_tokens": _mean(token_values),
                "average_cost_usd": _mean(cost_values),
                "regression_rate": sum(run.regression_detected for run in runs) / len(runs),
                "failure_taxonomy": dict(sorted(failure_counts.items())),
            }
        )
    return rows


def analyze_runs(run_root: Path, output_dir: Path) -> list[dict[str, Any]]:
    summaries = load_summaries(run_root)
    rows = aggregate(summaries)
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "aggregate.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    _write_run_csv(summaries, output_dir / "runs.csv")
    _write_aggregate_csv(rows, output_dir / "ablation_table.csv")
    _write_report(rows, summaries, output_dir / "results.md")
    _write_plots(rows, output_dir)
    return rows


def _write_run_csv(summaries: list[RunSummary], path: Path) -> None:
    fieldnames = [
        "run_id",
        "task_id",
        "condition",
        "resolved",
        "termination_reason",
        "patch_valid",
        "regression_detected",
        "iterations",
        "tool_calls",
        "runtime_seconds",
        "tokens",
        "estimated_cost_usd",
        "provider",
        "model",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            writer.writerow(
                {
                    "run_id": summary.run_id,
                    "task_id": summary.task_id,
                    "condition": summary.condition.value,
                    "resolved": summary.resolved,
                    "termination_reason": summary.termination_reason.value,
                    "patch_valid": summary.patch_valid,
                    "regression_detected": summary.regression_detected,
                    "iterations": summary.iterations,
                    "tool_calls": summary.tool_calls,
                    "runtime_seconds": summary.runtime_seconds,
                    "tokens": summary.usage.total_tokens,
                    "estimated_cost_usd": summary.usage.estimated_cost_usd,
                    "provider": summary.provider,
                    "model": summary.model,
                }
            )


def _write_aggregate_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = [key for key in rows[0] if key != "failure_taxonomy"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fields})


def _format(value: Any, *, percent: bool = False) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "N/A"
    if percent:
        return f"{100 * float(value):.1f}%"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _write_report(rows: list[dict[str, Any]], summaries: list[RunSummary], path: Path) -> None:
    providers = sorted({summary.provider for summary in summaries})
    scripted_only = providers == ["ScriptedProvider"]
    lines = [
        "# EvalForge results",
        "",
        (
            "> **Infrastructure validation only.** These runs used deterministic "
            "reference scripts and are not evidence of LLM coding performance."
            if scripted_only
            else "> Results below were computed directly from saved run summaries."
        ),
        "",
        f"Providers: {', '.join(providers)}",
        "",
        "| Condition | Runs | Resolution | Test pass | Patch validity | "
        "Avg iterations | Avg runtime (s) | Avg tokens | Avg cost | Regression |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {condition} | {runs} | {resolution} | {test_pass} | {patch} | "
            "{iterations} | {runtime} | {tokens} | {cost} | {regression} |".format(
                condition=row["condition"],
                runs=row["runs"],
                resolution=_format(row["task_resolution_rate"], percent=True),
                test_pass=_format(row["test_pass_rate"], percent=True),
                patch=_format(row["patch_validity_rate"], percent=True),
                iterations=_format(row["average_iterations"]),
                runtime=_format(row["average_runtime_seconds"]),
                tokens=_format(row["average_tokens"]),
                cost=_format(row["average_cost_usd"]),
                regression=_format(row["regression_rate"], percent=True),
            )
        )
    lines.extend(["", "## Failure taxonomy", ""])
    for row in rows:
        lines.append(
            f"- **{row['condition']}:** {json.dumps(row['failure_taxonomy'], sort_keys=True)}"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "The table is descriptive. With a small controlled task set, condition "
            "differences are exploratory. Causal claims require identical model settings, "
            "frozen memory, repeated runs, and uncertainty analysis.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_plots(rows: list[dict[str, Any]], output_dir: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "evalforge-matplotlib"))
    try:
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - analysis extra is optional
        return
    conditions = [row["condition"] for row in rows]
    resolution = [row["task_resolution_rate"] for row in rows]
    runtime = [row["average_runtime_seconds"] or 0 for row in rows]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(conditions, resolution, color="#3465a4")
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Resolution rate")
    axes[0].set_title("Task resolution")
    axes[1].bar(conditions, runtime, color="#4e9a06")
    axes[1].set_ylabel("Seconds")
    axes[1].set_title("Average runtime")
    for axis in axes:
        axis.tick_params(axis="x", rotation=20)
    figure.tight_layout()
    figure.savefig(output_dir / "ablation_overview.png", dpi=180)
    plt.close(figure)
