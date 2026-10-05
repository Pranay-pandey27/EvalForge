"""Network-disabled Docker verifier backend."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

from evalforge.models import ToolResult
from evalforge.tools.shell import DEFAULT_ALLOWED_EXECUTABLES


class DockerRunner:
    def __init__(
        self,
        root: Path,
        *,
        image: str = "evalforge-bench:latest",
        timeout_seconds: float = 60,
        max_output_chars: int = 50_000,
    ) -> None:
        self.root = root.resolve()
        self.image = image
        self.timeout_seconds = timeout_seconds
        self.max_output_chars = max_output_chars

    def run(self, command: list[str], cwd: str = ".") -> ToolResult:
        started = time.monotonic()
        if not command:
            return ToolResult(tool_name="docker", ok=False, output="command must not be empty")
        if Path(command[0]).name not in DEFAULT_ALLOWED_EXECUTABLES:
            return ToolResult(
                tool_name="docker",
                ok=False,
                output=f"executable is not allowlisted: {Path(command[0]).name}",
            )
        working = (self.root / cwd).resolve()
        try:
            relative = working.relative_to(self.root)
        except ValueError:
            return ToolResult(tool_name="docker", ok=False, output="cwd escapes workspace")
        container_cwd = "/workspace"
        if str(relative) != ".":
            container_cwd = f"/workspace/{relative.as_posix()}"
        docker_command = [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "--read-only",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,size=128m",
            "--mount",
            f"type=bind,src={self.root},dst=/workspace,rw",
            "--workdir",
            container_cwd,
            self.image,
            *command,
        ]
        try:
            completed = subprocess.run(
                docker_command,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
            )
            output = completed.stdout + completed.stderr
            truncated = len(output) > self.max_output_chars
            if truncated:
                output = output[: self.max_output_chars] + "\n... output truncated ..."
            return ToolResult(
                tool_name="docker",
                ok=completed.returncode == 0,
                output=output,
                exit_code=completed.returncode,
                runtime_seconds=time.monotonic() - started,
                truncated=truncated,
            )
        except FileNotFoundError:
            return ToolResult(
                tool_name="docker",
                ok=False,
                output="docker executable was not found",
                exit_code=127,
                runtime_seconds=time.monotonic() - started,
            )
        except subprocess.TimeoutExpired as exc:
            output = (exc.stdout or "") + (exc.stderr or "")
            return ToolResult(
                tool_name="docker",
                ok=False,
                output=f"container timed out after {self.timeout_seconds}s\n{output}",
                exit_code=124,
                runtime_seconds=time.monotonic() - started,
            )
