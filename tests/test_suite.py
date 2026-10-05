from __future__ import annotations

from pathlib import Path

from evalforge.benchmark import load_task
from evalforge.experiment import discover_tasks

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_heldout_suite_has_twenty_unique_tasks() -> None:
    task_dirs = discover_tasks(PROJECT_ROOT / "benchmarks" / "evaluation")
    tasks = [load_task(path) for path in task_dirs]
    assert len(tasks) == 20
    assert len({task.metadata.task_id for task in tasks}) == 20
    assert all(task.root.joinpath("reference_patch.diff").stat().st_size > 0 for task in tasks)


def test_development_and_evaluation_ids_do_not_overlap() -> None:
    development = {
        load_task(path).metadata.task_id
        for path in discover_tasks(PROJECT_ROOT / "benchmarks" / "development")
    }
    evaluation = {
        load_task(path).metadata.task_id
        for path in discover_tasks(PROJECT_ROOT / "benchmarks" / "evaluation")
    }
    assert development.isdisjoint(evaluation)
