"""Copy auditable run evidence without transient Git workspaces."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from evalforge.memory.store import MemoryStore


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--memory-db", type=Path)
    args = parser.parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    copied = 0
    manifest = source / "manifest.json"
    if manifest.exists():
        shutil.copy2(manifest, output / "manifest.json")
    for summary in source.rglob("summary.json"):
        relative_parent = summary.parent.relative_to(source)
        destination = output / relative_parent
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copy2(summary, destination / "summary.json")
        trace = summary.with_name("trace.jsonl")
        if trace.exists():
            shutil.copy2(trace, destination / "trace.jsonl")
        copied += 1
    if args.memory_db:
        records = MemoryStore(args.memory_db).all()
        (output / "memory_records.json").write_text(
            json.dumps([record.model_dump(mode="json") for record in records], indent=2) + "\n",
            encoding="utf-8",
        )
    print(f"copied {copied} run summaries and traces")


if __name__ == "__main__":
    main()
