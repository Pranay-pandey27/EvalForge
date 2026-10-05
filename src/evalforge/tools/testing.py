"""Test execution and normalized verifier evidence."""

from __future__ import annotations

import re
from pathlib import Path

from evalforge.models import VerificationResult
from evalforge.runners.base import CommandRunner
from evalforge.tools.shell import ControlledShell

_COUNT_PATTERNS = {
    "passed": re.compile(r"(?P<count>\d+) passed"),
    "failed": re.compile(r"(?P<count>\d+) failed"),
    "collected": re.compile(r"collected (?P<count>\d+) items?"),
}


class TestVerifier:
    def __init__(
        self,
        root: Path,
        *,
        timeout_seconds: float = 60,
        runner: CommandRunner | None = None,
    ) -> None:
        self.shell = runner or ControlledShell(root, timeout_seconds=timeout_seconds)

    @staticmethod
    def _count(output: str, label: str) -> int | None:
        matches = list(_COUNT_PATTERNS[label].finditer(output))
        return int(matches[-1].group("count")) if matches else None

    def run(self, command: list[str]) -> VerificationResult:
        result = self.shell.run(command)
        output_lower = result.output.lower()
        environment_error = result.exit_code in {124, 126, 127} or any(
            marker in output_lower
            for marker in ("no module named pytest", "command not found", "internalerror")
        )
        passed_count = self._count(result.output, "passed")
        failed_count = self._count(result.output, "failed")
        collected_count = self._count(result.output, "collected")
        if collected_count is None and (passed_count is not None or failed_count is not None):
            collected_count = (passed_count or 0) + (failed_count or 0)
        return VerificationResult(
            command=command,
            exit_code=result.exit_code if result.exit_code is not None else 1,
            passed=result.ok and not environment_error,
            output=result.output,
            runtime_seconds=result.runtime_seconds,
            tests_collected=collected_count,
            tests_passed=passed_count,
            tests_failed=failed_count,
            environment_error=environment_error,
        )
