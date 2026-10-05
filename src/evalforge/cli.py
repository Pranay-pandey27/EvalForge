"""EvalForge command-line interface."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Annotated

import typer

from evalforge.benchmark import load_task, read_script, validate_task_behavior
from evalforge.evaluation import analyze_runs
from evalforge.experiment import ExperimentRunner, discover_tasks
from evalforge.memory.store import MemoryStore
from evalforge.models import AgentDecision, Budget, Condition, ExperimentConfig
from evalforge.orchestrator import Orchestrator
from evalforge.providers.litellm import LiteLLMProvider
from evalforge.providers.mock import ScriptedProvider
from evalforge.runners.docker import DockerRunner

app = typer.Typer(
    name="evalforge",
    no_args_is_help=True,
    help="Evaluation-guided coding-agent harness for scientific Python.",
)


@app.command("validate-task")
def validate_task_command(
    task_dir: Path,
    execute: Annotated[
        bool,
        typer.Option(
            "--execute/--structure-only",
            help="Execute the buggy and reference-patched verification gates.",
        ),
    ] = True,
) -> None:
    """Validate one benchmark task's structure and expected behavior."""

    task = load_task(task_dir)
    if execute:
        result = validate_task_behavior(task)
        typer.echo(json.dumps(result.model_dump(mode="json"), indent=2))
        if not result.valid:
            raise typer.Exit(code=1)
        return
    typer.echo(
        json.dumps(
            {
                "task_id": task.metadata.task_id,
                "category": task.metadata.category,
                "difficulty": task.metadata.difficulty,
                "valid": True,
                "behavior_executed": False,
            },
            indent=2,
        )
    )


@app.command("validate-suite")
def validate_suite_command(
    suite_dir: Path,
    execute: Annotated[
        bool,
        typer.Option("--execute/--structure-only", help="Run behavioral validation."),
    ] = True,
) -> None:
    """Validate every task in a benchmark suite."""

    rows = []
    for task_dir in discover_tasks(suite_dir):
        task = load_task(task_dir)
        if execute:
            result = validate_task_behavior(task)
            rows.append(result.model_dump(mode="json"))
        else:
            rows.append({"task_id": task.metadata.task_id, "valid": True})
    valid = all(row["valid"] for row in rows)
    typer.echo(json.dumps({"valid": valid, "tasks": rows}, indent=2))
    if not valid:
        raise typer.Exit(code=1)


@app.command("run-scripted")
def run_scripted(
    task_dir: Path,
    script: Path | None = None,
    run_root: Annotated[Path, typer.Option(help="Run artifact directory.")] = Path(
        "experiments/runs"
    ),
    condition: Condition = Condition.SCRIPTED,
    memory_db: Annotated[Path | None, typer.Option()] = None,
    persist_memory: Annotated[bool, typer.Option()] = False,
    max_iterations: Annotated[int, typer.Option(min=1)] = 8,
    max_tool_calls: Annotated[int, typer.Option(min=1)] = 40,
) -> None:
    """Run a credential-free deterministic repair script."""

    task = load_task(task_dir)
    script = script or task_dir / "scripted_actions.json"
    decisions = [AgentDecision.model_validate(item) for item in read_script(script)]
    summary = Orchestrator(
        task=task,
        provider=ScriptedProvider(decisions),
        run_root=run_root,
        budget=Budget(max_iterations=max_iterations, max_tool_calls=max_tool_calls),
        condition=condition,
        memory_store=MemoryStore(memory_db) if memory_db else None,
        persist_memory=persist_memory,
    ).run()
    typer.echo(json.dumps(summary.model_dump(mode="json"), indent=2))
    if not summary.resolved:
        raise typer.Exit(code=1)


@app.command("run-task")
def run_task(
    task_dir: Path,
    model: Annotated[str, typer.Option(help="LiteLLM model identifier.")],
    condition: Condition = Condition.BASELINE,
    run_root: Annotated[Path, typer.Option()] = Path("experiments/runs"),
    memory_db: Annotated[Path | None, typer.Option()] = None,
    runner: Annotated[str, typer.Option(help="local or docker")] = "local",
    docker_image: Annotated[str, typer.Option()] = "evalforge-bench:latest",
    max_iterations: Annotated[int, typer.Option(min=1)] = 8,
    max_tool_calls: Annotated[int, typer.Option(min=1)] = 40,
    max_tokens: Annotated[int | None, typer.Option(min=1)] = None,
    max_cost_usd: Annotated[float | None, typer.Option(min=0.0)] = None,
) -> None:
    """Run one task with a real model through LiteLLM."""

    if runner not in {"local", "docker"}:
        raise typer.BadParameter("runner must be local or docker")
    task = load_task(task_dir)
    budget = Budget(
        max_iterations=max_iterations,
        max_tool_calls=max_tool_calls,
        max_tokens=max_tokens,
        max_cost_usd=max_cost_usd,
    )
    factory = None
    if runner == "docker":

        def docker_factory(root: Path) -> DockerRunner:
            return DockerRunner(
                root,
                image=docker_image,
                timeout_seconds=budget.command_timeout_seconds,
            )

        factory = docker_factory
    summary = Orchestrator(
        task=task,
        provider=LiteLLMProvider(model),
        run_root=run_root,
        budget=budget,
        condition=condition,
        memory_store=MemoryStore(memory_db) if memory_db else None,
        persist_memory=False,
        verifier_runner_factory=factory,
    ).run()
    typer.echo(json.dumps(summary.model_dump(mode="json"), indent=2))
    if not summary.resolved:
        raise typer.Exit(code=1)


@app.command("run-experiment")
def run_experiment(config: Path) -> None:
    """Run a reproducible multi-condition experiment from JSON config."""

    parsed = ExperimentConfig.model_validate_json(config.read_text(encoding="utf-8"))
    summaries = ExperimentRunner(parsed).run()
    typer.echo(
        json.dumps(
            {
                "runs": len(summaries),
                "resolved": sum(summary.resolved for summary in summaries),
                "output_dir": str(parsed.output_dir.resolve()),
            },
            indent=2,
        )
    )


@app.command("analyze")
def analyze(
    run_root: Path,
    output_dir: Annotated[Path, typer.Option()] = Path("experiments/analysis"),
) -> None:
    """Generate trace-derived tables, plots, and a report."""

    rows = analyze_runs(run_root, output_dir)
    typer.echo(json.dumps(rows, indent=2))


@app.command("inspect-memory")
def inspect_memory(memory_db: Path) -> None:
    """Print memory records with provenance for audit."""

    records = MemoryStore(memory_db).all()
    typer.echo(json.dumps([record.model_dump(mode="json") for record in records], indent=2))


@app.command("freeze-memory")
def freeze_memory(source: Path, destination: Path) -> None:
    """Create a consistent read-only experiment snapshot of a memory database."""

    if not source.is_file():
        raise typer.BadParameter(f"source database does not exist: {source}")
    if destination.exists():
        raise typer.BadParameter(f"destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(source) as source_connection:
        with sqlite3.connect(destination) as destination_connection:
            source_connection.backup(destination_connection)
    typer.echo(str(destination.resolve()))


if __name__ == "__main__":  # pragma: no cover
    app()
