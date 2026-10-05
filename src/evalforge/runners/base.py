"""Execution backend protocol."""

from __future__ import annotations

from typing import Protocol

from evalforge.models import ToolResult


class CommandRunner(Protocol):
    def run(self, command: list[str], cwd: str = ".") -> ToolResult:
        """Run a tokenized command and return normalized evidence."""
