"""Export versioned JSON Schemas from EvalForge's Pydantic contracts."""

from __future__ import annotations

import json
from pathlib import Path

from evalforge.models import ExperimentConfig, MemoryRecord, RunSummary, TraceEvent


def main() -> None:
    root = Path(__file__).resolve().parents[1] / "schemas"
    root.mkdir(exist_ok=True)
    models = {
        "trace.schema.json": TraceEvent,
        "run-summary.schema.json": RunSummary,
        "memory.schema.json": MemoryRecord,
        "experiment-config.schema.json": ExperimentConfig,
    }
    for filename, model in models.items():
        (root / filename).write_text(
            json.dumps(model.model_json_schema(), indent=2) + "\n", encoding="utf-8"
        )
    print(f"exported {len(models)} schemas")


if __name__ == "__main__":
    main()
