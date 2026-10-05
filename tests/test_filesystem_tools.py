from __future__ import annotations

from pathlib import Path

import pytest

from evalforge.tools.filesystem import FileSystemTools, PathViolation


def test_read_rejects_workspace_escape(tmp_path: Path) -> None:
    tools = FileSystemTools(tmp_path)
    with pytest.raises(PathViolation):
        tools.read_file("../outside.txt")


def test_apply_patch_changes_tracked_file(tmp_path: Path) -> None:
    target = tmp_path / "value.py"
    target.write_text("VALUE = 1\n", encoding="utf-8")
    tools = FileSystemTools(tmp_path)
    patch = """diff --git a/value.py b/value.py
--- a/value.py
+++ b/value.py
@@ -1 +1 @@
-VALUE = 1
+VALUE = 2
"""
    result = tools.apply_patch(patch)
    assert result.ok, result.output
    assert target.read_text(encoding="utf-8") == "VALUE = 2\n"
