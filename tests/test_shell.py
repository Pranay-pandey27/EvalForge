from __future__ import annotations

from pathlib import Path

from evalforge.tools.shell import ControlledShell


def test_shell_rejects_unlisted_executable(tmp_path: Path) -> None:
    result = ControlledShell(tmp_path).run(["curl", "https://example.com"])
    assert not result.ok
    assert "not allowlisted" in result.output


def test_shell_rejects_cwd_escape(tmp_path: Path) -> None:
    result = ControlledShell(tmp_path).run(["python", "-V"], cwd="..")
    assert not result.ok
    assert "escapes" in result.output


def test_shell_rejects_inline_python(tmp_path: Path) -> None:
    result = ControlledShell(tmp_path).run(["python", "-c", "print('unsafe')"])
    assert not result.ok
    assert "inline code is disabled" in result.output


def test_shell_restricts_git_to_read_only_subcommands(tmp_path: Path) -> None:
    result = ControlledShell(tmp_path).run(["git", "commit", "-m", "unexpected"])
    assert not result.ok
    assert "read-only" in result.output
