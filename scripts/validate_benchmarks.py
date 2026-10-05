"""Validate all benchmark behaviors and write a compact evidence report."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from evalforge.benchmark import load_task, validate_task_behavior
from evalforge.experiment import discover_tasks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/artifacts/benchmark-validation.json"),
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    records = []
    for suite in ("development", "evaluation"):
        for task_dir in discover_tasks(root / "benchmarks" / suite):
            result = validate_task_behavior(load_task(task_dir))
            records.append(
                {
                    "suite": suite,
                    "task_id": result.task_id,
                    "valid": result.valid,
                    "buggy_state_failed": result.buggy_state_failed,
                    "reference_patch_applied": result.reference_patch_applied,
                    "repaired_state_passed": result.repaired_state_passed,
                    "buggy_tests_collected": result.buggy_verification.tests_collected,
                    "repaired_tests_collected": (
                        result.repaired_verification.tests_collected
                        if result.repaired_verification
                        else None
                    ),
                }
            )
    payload = {
        "schema_version": "1.0",
        "created_at": datetime.now(UTC).isoformat(),
        "valid": all(record["valid"] for record in records),
        "task_count": len(records),
        "tasks": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"valid": payload["valid"], "task_count": len(records)}))
    if not payload["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
