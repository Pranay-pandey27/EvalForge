from __future__ import annotations

from pathlib import Path

from evalforge.benchmark import TaskWorkspace, load_task, validate_task_behavior

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TASK = PROJECT_ROOT / "benchmarks" / "development" / "task_001"


def test_loads_first_development_task() -> None:
    task = load_task(TASK)
    assert task.metadata.task_id == "numpy_broadcast_001"
    assert task.repo_dir.is_dir()
    assert task.evaluator_tests_dir.is_dir()


def test_workspace_is_a_fresh_git_copy(tmp_path: Path) -> None:
    task = load_task(TASK)
    workspace = TaskWorkspace.create(task, tmp_path, "test-run")
    assert (workspace.path / ".git").is_dir()
    assert workspace.changed_files() == []


def test_reference_patch_repairs_first_development_task() -> None:
    result = validate_task_behavior(load_task(TASK))
    assert result.valid, result.error
    assert result.buggy_state_failed
    assert result.reference_patch_applied
    assert result.repaired_state_passed
