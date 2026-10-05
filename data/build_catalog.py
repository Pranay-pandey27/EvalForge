"""Export a compact JSONL catalogue of EvalForge benchmark tasks."""

from __future__ import annotations

import json
from pathlib import Path


def build(root: Path, output: Path) -> None:
    rows = []
    for suite in ["development", "evaluation"]:
        for metadata_path in sorted((root / "benchmarks" / suite).glob("task_*/metadata.json")):
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            rows.append({
                "suite": suite,
                "task_directory": metadata_path.parent.name,
                "task_id": metadata["task_id"],
                "title": metadata.get("title", metadata["task_id"]),
                "category": metadata["category"],
                "difficulty": metadata["difficulty"],
                "visible_test_command": metadata["visible_test_command"],
                "evaluator_test_command": metadata["evaluator_test_command"],
            })
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    project = Path(__file__).parents[1]
    build(project, Path(__file__).with_name("benchmark_catalog.jsonl"))
