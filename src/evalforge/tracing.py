"""Append-only JSONL traces with a separate final run summary."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from evalforge.models import AgentState, RunSummary, TraceEvent


class TraceWriter:
    def __init__(self, path: Path) -> None:
        self.path = path.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._sequence = 0

    def record(self, event_type: str, state: AgentState, **payload: Any) -> TraceEvent:
        event = TraceEvent(
            sequence=self._sequence,
            event_type=event_type,
            state=state,
            payload=payload,
        )
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(event.model_dump_json() + "\n")
        self._sequence += 1
        return event

    def write_summary(self, summary: RunSummary) -> Path:
        summary_path = self.path.with_name("summary.json")
        summary_path.write_text(
            json.dumps(summary.model_dump(mode="json"), indent=2) + "\n",
            encoding="utf-8",
        )
        return summary_path
