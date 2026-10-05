"""Tokenized, allowlisted subprocess execution inside a task workspace."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

from evalforge.models import ToolResult

DEFAULT_ALLOWED_EXECUTABLES = frozenset({"python", "python3", "pytest", "git", "ruff", "mypy"})
SAFE_ENV_KEYS = frozenset(
    {"PATH", "LANG", "LC_ALL", "PYTHONPATH", "PYTHONUTF8", "VIRTUAL_ENV", "SYSTEMROOT"}
)
READ_ONLY_GIT_SUBCOMMANDS = frozenset({"diff", "grep", "ls-files", "show", "status"})
ALLOWED_PYTHON_MODULES = frozenset({"mypy", "pytest", "ruff"})


class ControlledShell:
    def __init__(
        self,
        root: Path,
        *,
        timeout_seconds: float = 60,
        max_output_chars: int = 50_000,
        allowed_executables: frozenset[str] = DEFAULT_ALLOWED_EXECUTABLES,
    ) -> None:
        self.root = root.resolve()
        self.timeout_seconds = timeout_seconds
        self.max_output_chars = max_output_chars
        self.allowed_executables = allowed_executables

    def run(self, command: list[str], cwd: str = ".") -> ToolResult:
        started = time.monotonic()
        if not command:
            return ToolResult(tool_name="shell", ok=False, output="command must not be empty")
        command = list(command)
        executable = Path(command[0]).name
        if executable not in self.allowed_executables:
            return ToolResult(
                tool_name="shell",
                ok=False,
                output=f"executable is not allowlisted: {executable}",
            )
        policy_error = self._validate_profile(executable, command[1:])
        if policy_error is not None:
            return ToolResult(tool_name="shell", ok=False, output=policy_error)
        if executable in {"python", "python3"}:
            command[0] = sys.executable
        elif executable == "pytest":
            command = [sys.executable, "-m", "pytest", *command[1:]]
        working_directory = (self.root / cwd).resolve()
        try:
            working_directory.relative_to(self.root)
        except ValueError:
            return ToolResult(tool_name="shell", ok=False, output="cwd escapes workspace")
        if not working_directory.is_dir():
            return ToolResult(tool_name="shell", ok=False, output="cwd does not exist")
        environment = {key: value for key, value in os.environ.items() if key in SAFE_ENV_KEYS}
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        try:
            completed = subprocess.run(
                command,
                cwd=working_directory,
                env=environment,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
            )
            output = completed.stdout + completed.stderr
            truncated = len(output) > self.max_output_chars
            if truncated:
                output = output[: self.max_output_chars] + "\n... output truncated ..."
            return ToolResult(
                tool_name="shell",
                ok=completed.returncode == 0,
                output=output,
                exit_code=completed.returncode,
                runtime_seconds=time.monotonic() - started,
                truncated=truncated,
            )
        except subprocess.TimeoutExpired as exc:
            captured = (exc.stdout or "") + (exc.stderr or "")
            return ToolResult(
                tool_name="shell",
                ok=False,
                output=f"command timed out after {self.timeout_seconds}s\n{captured}",
                exit_code=124,
                runtime_seconds=time.monotonic() - started,
            )

    @staticmethod
    def _validate_profile(executable: str, arguments: list[str]) -> str | None:
        for argument in arguments:
            candidate = argument.split("=", maxsplit=1)[-1]
            path = Path(candidate)
            if path.is_absolute() or ".." in path.parts:
                return "command arguments may not address paths outside the workspace"
            if "\x00" in argument:
                return "command arguments must be NUL-free"
        if executable in {"python", "python3"}:
            if arguments in (["-V"], ["--version"]):
                return None
            if len(arguments) >= 2 and arguments[0] == "-m":
                if arguments[1] in ALLOWED_PYTHON_MODULES:
                    return None
            return "python is restricted to approved tool modules; inline code is disabled"
        if executable == "git":
            if not arguments or arguments[0] not in READ_ONLY_GIT_SUBCOMMANDS:
                return "git is restricted to read-only inspection subcommands"
        return None
