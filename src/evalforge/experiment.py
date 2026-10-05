"""Controlled multi-condition experiment execution."""

from __future__ import annotations

import hashlib
import json
import random
from datetime import UTC, datetime
from pathlib import Path

from evalforge.benchmark import load_task, read_script
from evalforge.memory.store import MemoryStore
from evalforge.models import AgentDecision, Condition, ExperimentConfig, RunSummary
from evalforge.orchestrator import Orchestrator
from evalforge.providers.litellm import LiteLLMProvider
from evalforge.providers.mock import ScriptedProvider
from evalforge.runners.docker import DockerRunner


def discover_tasks(suite_dir: Path) -> list[Path]:
    tasks = sorted(path for path in suite_dir.resolve().glob("task_*") if path.is_dir())
    if not tasks:
        raise ValueError(f"no task_* directories found under {suite_dir}")
    return tasks


class ExperimentRunner:
    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config

    def run(self) -> list[RunSummary]:
        tasks = discover_tasks(self.config.suite_dir)
        output = self.config.output_dir.resolve()
        output.mkdir(parents=True, exist_ok=True)
        memory_store = MemoryStore(self.config.memory_db) if self.config.memory_db else None
        summaries: list[RunSummary] = []
        randomizer = random.Random(self.config.seed)
        schedule = [
            (condition, repeat, task_dir)
            for condition in self.config.conditions
            for repeat in range(self.config.repeats)
            for task_dir in tasks
        ]
        # A recorded deterministic shuffle prevents easy/hard ordering from coupling
        # to transient provider behavior while remaining reproducible.
        randomizer.shuffle(schedule)
        for condition, repeat, task_dir in schedule:
            task = load_task(task_dir)
            provider = self._provider(task_dir, repeat)
            runner_factory = None
            if self.config.runner == "docker":

                def docker_factory(root: Path) -> DockerRunner:
                    return DockerRunner(
                        root,
                        image=self.config.docker_image,
                        timeout_seconds=self.config.budget.command_timeout_seconds,
                    )

                runner_factory = docker_factory
            run_root = output / condition.value / f"repeat_{repeat:02d}"
            summary = Orchestrator(
                task=task,
                provider=provider,
                run_root=run_root,
                budget=self.config.budget,
                condition=condition,
                memory_store=memory_store,
                persist_memory=self.config.persist_memory,
                verifier_runner_factory=runner_factory,
            ).run()
            summaries.append(summary)
        self._write_manifest(output, summaries, schedule)
        return summaries

    def _provider(self, task_dir: Path, repeat: int):
        if self.config.provider == "scripted":
            script = task_dir / "scripted_actions.json"
            if not script.exists():
                raise FileNotFoundError(f"scripted provider requires {script}")
            decisions = [AgentDecision.model_validate(item) for item in read_script(script)]
            return ScriptedProvider(decisions)
        return LiteLLMProvider(
            self.config.model or "",
            temperature=0.0,
            seed=self.config.seed + repeat,
        )

    def _write_manifest(
        self,
        output: Path,
        summaries: list[RunSummary],
        schedule: list[tuple[Condition, int, Path]],
    ) -> None:
        canonical = self.config.model_dump_json(exclude={"output_dir"})
        payload = {
            "schema_version": "1.0",
            "created_at": datetime.now(UTC).isoformat(),
            "experiment_kind": (
                "infrastructure_validation"
                if self.config.provider == "scripted"
                else "model_ablation"
            ),
            "config": self.config.model_dump(mode="json"),
            "config_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
            "schedule": [
                {
                    "condition": condition,
                    "repeat": repeat,
                    "task": task.name,
                }
                for condition, repeat, task in schedule
            ],
            "run_ids": [summary.run_id for summary in summaries],
        }
        (output / "manifest.json").write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
