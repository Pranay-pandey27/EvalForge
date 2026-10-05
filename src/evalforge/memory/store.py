"""SQLite-backed storage for concise repair lessons."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from evalforge.models import MemoryRecord


class MemoryStore:
    """Own the memory schema and preserve verifier provenance."""

    def __init__(self, path: Path) -> None:
        self.path = path.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize(self) -> None:
        with self.connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    memory_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    repository TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    task_category TEXT NOT NULL,
                    error_signature TEXT NOT NULL,
                    lesson TEXT NOT NULL,
                    repair_pattern TEXT NOT NULL DEFAULT '',
                    failed_approach TEXT NOT NULL DEFAULT '',
                    trace_id TEXT NOT NULL,
                    verifier_evidence TEXT NOT NULL,
                    outcome TEXT NOT NULL CHECK (outcome IN ('resolved', 'failed')),
                    tags_json TEXT NOT NULL DEFAULT '[]',
                    created_at TEXT NOT NULL,
                    schema_version TEXT NOT NULL,
                    UNIQUE(task_id, trace_id, lesson)
                );
                CREATE INDEX IF NOT EXISTS idx_memories_repository
                    ON memories(repository);
                CREATE INDEX IF NOT EXISTS idx_memories_category
                    ON memories(task_category);
                CREATE INDEX IF NOT EXISTS idx_memories_task
                    ON memories(task_id);
                """
            )
            try:
                connection.execute(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                        lesson,
                        repair_pattern,
                        error_signature,
                        tags
                    )
                    """
                )
                connection.executescript(
                    """
                    CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
                      INSERT INTO memories_fts(rowid, lesson, repair_pattern, error_signature, tags)
                      VALUES (new.memory_id, new.lesson, new.repair_pattern,
                              new.error_signature, new.tags_json);
                    END;
                    CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
                      DELETE FROM memories_fts WHERE rowid = old.memory_id;
                    END;
                    """
                )
            except sqlite3.OperationalError:
                # Some minimal SQLite builds omit FTS5. Retrieval has a LIKE fallback.
                pass

    def add(self, record: MemoryRecord) -> MemoryRecord:
        if not record.verifier_evidence.strip():
            raise ValueError("memory requires verifier evidence")
        values = (
            record.repository,
            record.task_id,
            record.task_category,
            record.error_signature,
            record.lesson,
            record.repair_pattern,
            record.failed_approach,
            record.trace_id,
            record.verifier_evidence,
            record.outcome,
            json.dumps(record.tags, sort_keys=True),
            record.created_at.isoformat(),
            record.schema_version,
        )
        with self.connect() as connection:
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO memories (
                    repository, task_id, task_category, error_signature, lesson,
                    repair_pattern, failed_approach, trace_id, verifier_evidence,
                    outcome, tags_json, created_at, schema_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                values,
            )
            if cursor.lastrowid:
                memory_id = int(cursor.lastrowid)
            else:
                row = connection.execute(
                    """SELECT memory_id FROM memories
                       WHERE task_id = ? AND trace_id = ? AND lesson = ?""",
                    (record.task_id, record.trace_id, record.lesson),
                ).fetchone()
                if row is None:
                    raise RuntimeError("failed to persist memory")
                memory_id = int(row["memory_id"])
        return record.model_copy(update={"memory_id": memory_id})

    def get(self, memory_id: int) -> MemoryRecord | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM memories WHERE memory_id = ?", (memory_id,)
            ).fetchone()
        return self._from_row(row) if row else None

    def all(self) -> list[MemoryRecord]:
        with self.connect() as connection:
            rows = connection.execute("SELECT * FROM memories ORDER BY memory_id").fetchall()
        return [self._from_row(row) for row in rows]

    @staticmethod
    def _from_row(row: sqlite3.Row) -> MemoryRecord:
        return MemoryRecord(
            memory_id=row["memory_id"],
            repository=row["repository"],
            task_id=row["task_id"],
            task_category=row["task_category"],
            error_signature=row["error_signature"],
            lesson=row["lesson"],
            repair_pattern=row["repair_pattern"],
            failed_approach=row["failed_approach"],
            trace_id=row["trace_id"],
            verifier_evidence=row["verifier_evidence"],
            outcome=row["outcome"],
            tags=json.loads(row["tags_json"]),
            created_at=row["created_at"],
            schema_version=row["schema_version"],
        )
