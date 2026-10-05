"""Repository-contained read, search, and patch operations."""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
import time
from pathlib import Path

from evalforge.models import ToolResult


class PathViolation(ValueError):
    """Raised when an operation attempts to escape the task workspace."""


class FileSystemTools:
    def __init__(self, root: Path, *, max_output_chars: int = 30_000) -> None:
        self.root = root.resolve()
        self.max_output_chars = max_output_chars

    def _resolve(self, relative_path: str) -> Path:
        if not relative_path or "\x00" in relative_path:
            raise PathViolation("path must be non-empty and NUL-free")
        candidate = (self.root / relative_path).resolve()
        try:
            candidate.relative_to(self.root)
        except ValueError as exc:
            raise PathViolation(f"path escapes workspace: {relative_path}") from exc
        return candidate

    def _bounded(self, value: str) -> tuple[str, bool]:
        if len(value) <= self.max_output_chars:
            return value, False
        return value[: self.max_output_chars] + "\n... output truncated ...", True

    def list_files(self, pattern: str = "*") -> ToolResult:
        started = time.monotonic()
        paths = sorted(
            str(path.relative_to(self.root))
            for path in self.root.rglob(pattern)
            if path.is_file() and ".git" not in path.parts
        )
        output, truncated = self._bounded("\n".join(paths))
        return ToolResult(
            tool_name="list_files",
            ok=True,
            output=output,
            runtime_seconds=time.monotonic() - started,
            truncated=truncated,
        )

    def read_file(self, path: str) -> ToolResult:
        started = time.monotonic()
        target = self._resolve(path)
        if not target.is_file():
            return ToolResult(
                tool_name="read_file",
                ok=False,
                output=f"file not found: {path}",
                runtime_seconds=time.monotonic() - started,
            )
        output, truncated = self._bounded(target.read_text(encoding="utf-8"))
        return ToolResult(
            tool_name="read_file",
            ok=True,
            output=output,
            runtime_seconds=time.monotonic() - started,
            truncated=truncated,
        )

    def search_text(self, query: str, pattern: str = "*.py") -> ToolResult:
        started = time.monotonic()
        try:
            expression = re.compile(query)
        except re.error as exc:
            return ToolResult(
                tool_name="search_text",
                ok=False,
                output=f"invalid regular expression: {exc}",
                runtime_seconds=time.monotonic() - started,
            )
        matches: list[str] = []
        for path in sorted(self.root.rglob(pattern)):
            if not path.is_file() or ".git" in path.parts:
                continue
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except UnicodeDecodeError:
                continue
            for number, line in enumerate(lines, start=1):
                if expression.search(line):
                    matches.append(f"{path.relative_to(self.root)}:{number}:{line}")
        output, truncated = self._bounded("\n".join(matches))
        return ToolResult(
            tool_name="search_text",
            ok=True,
            output=output,
            runtime_seconds=time.monotonic() - started,
            truncated=truncated,
        )

    def apply_patch(self, patch: str) -> ToolResult:
        started = time.monotonic()
        if not patch.strip():
            return ToolResult(
                tool_name="apply_patch",
                ok=False,
                output="patch must not be empty",
                runtime_seconds=time.monotonic() - started,
            )
        descriptor, patch_path = tempfile.mkstemp(prefix="evalforge-", suffix=".diff")
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(patch)
            check = subprocess.run(
                ["git", "apply", "--check", patch_path],
                cwd=self.root,
                capture_output=True,
                text=True,
            )
            if check.returncode != 0:
                return ToolResult(
                    tool_name="apply_patch",
                    ok=False,
                    output=check.stderr.strip() or check.stdout.strip(),
                    exit_code=check.returncode,
                    runtime_seconds=time.monotonic() - started,
                )
            applied = subprocess.run(
                ["git", "apply", patch_path],
                cwd=self.root,
                capture_output=True,
                text=True,
            )
            return ToolResult(
                tool_name="apply_patch",
                ok=applied.returncode == 0,
                output=applied.stderr.strip() or applied.stdout.strip() or "patch applied",
                exit_code=applied.returncode,
                runtime_seconds=time.monotonic() - started,
            )
        finally:
            Path(patch_path).unlink(missing_ok=True)
