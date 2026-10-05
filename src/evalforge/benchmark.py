"""Benchmark loading, validation, and isolated workspace creation."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from evalforge.models import TaskMetadata, TaskSpec, TaskValidationResult

REQUIRED_TASK_ENTRIES = (
    "repo",
    "tests",
    "issue.md",
    "metadata.json",
    "reference_patch.diff",
)


class InvalidTask(ValueError):
    """Raised when a benchmark task is incomplete or malformed."""


def load_task(task_dir: Path) -> TaskSpec:
    root = task_dir.resolve()
    missing = [name for name in REQUIRED_TASK_ENTRIES if not (root / name).exists()]
    if missing:
        raise InvalidTask(f"missing required task entries: {', '.join(missing)}")
    metadata = TaskMetadata.model_validate_json((root / "metadata.json").read_text())
    issue = (root / "issue.md").read_text(encoding="utf-8").strip()
    if not issue:
        raise InvalidTask("issue.md must not be empty")
    return TaskSpec(root=root, issue=issue, metadata=metadata)


@dataclass(frozen=True, slots=True)
class TaskWorkspace:
    run_id: str
    path: Path

    @classmethod
    def create(cls, task: TaskSpec, run_root: Path, run_id: str) -> TaskWorkspace:
        destination = (run_root.resolve() / run_id / "workspace").resolve()
        if destination.exists():
            raise FileExistsError(f"run workspace already exists: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=False)
        shutil.copytree(task.repo_dir, destination)
        cls._initialize_git(destination)
        return cls(run_id=run_id, path=destination)

    @staticmethod
    def _initialize_git(path: Path) -> None:
        commands = (
            ["git", "init", "-q"],
            ["git", "config", "user.name", "EvalForge"],
            ["git", "config", "user.email", "evalforge@localhost"],
            ["git", "add", "--all"],
            ["git", "commit", "-q", "-m", "buggy task baseline"],
        )
        for command in commands:
            completed = subprocess.run(command, cwd=path, capture_output=True, text=True)
            if completed.returncode != 0:
                raise RuntimeError(
                    f"failed to initialize isolated Git workspace: {completed.stderr.strip()}"
                )

    def changed_files(self) -> list[str]:
        modified = subprocess.run(
            ["git", "diff", "--name-only", "HEAD"],
            cwd=self.path,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        untracked = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard"],
            cwd=self.path,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        return sorted(set(modified + untracked))

    def patch_is_valid(self) -> bool:
        """Check textual patch integrity and the presence of the Git baseline."""

        if not (self.path / ".git").is_dir():
            return False
        completed = subprocess.run(
            ["git", "diff", "--check", "HEAD"],
            cwd=self.path,
            capture_output=True,
            text=True,
        )
        return completed.returncode == 0


def read_script(path: Path) -> list[dict[str, object]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("scripted actions must be a JSON list")
    return data


def validate_task_behavior(task: TaskSpec, *, timeout_seconds: float = 60) -> TaskValidationResult:
    """Prove that the buggy state fails and the reference repair passes all tests."""

    from evalforge.tools.filesystem import FileSystemTools
    from evalforge.tools.testing import TestVerifier

    with tempfile.TemporaryDirectory(prefix="evalforge-task-validation-") as temporary_root:
        workspace = TaskWorkspace.create(task, Path(temporary_root), "validation")
        verifier = TestVerifier(workspace.path, timeout_seconds=timeout_seconds)
        buggy = verifier.run(task.metadata.visible_test_command)
        buggy_state_failed = not buggy.passed and not buggy.environment_error
        patch = (task.root / "reference_patch.diff").read_text(encoding="utf-8")
        patch_result = FileSystemTools(workspace.path).apply_patch(patch)
        if not patch_result.ok:
            return TaskValidationResult(
                task_id=task.metadata.task_id,
                valid=False,
                buggy_state_failed=buggy_state_failed,
                reference_patch_applied=False,
                repaired_state_passed=False,
                buggy_verification=buggy,
                error=patch_result.output,
            )
        overlay = workspace.path / ".evalforge_evaluator_tests"
        shutil.copytree(task.evaluator_tests_dir, overlay)
        try:
            repaired = verifier.run(task.metadata.evaluator_test_command)
        finally:
            shutil.rmtree(overlay, ignore_errors=True)
        repaired_state_passed = repaired.passed and not repaired.environment_error
        return TaskValidationResult(
            task_id=task.metadata.task_id,
            valid=buggy_state_failed and repaired_state_passed,
            buggy_state_failed=buggy_state_failed,
            reference_patch_applied=True,
            repaired_state_passed=repaired_state_passed,
            buggy_verification=buggy,
            repaired_verification=repaired,
        )
