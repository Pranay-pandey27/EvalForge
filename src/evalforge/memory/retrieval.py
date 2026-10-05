"""Leakage-aware lexical retrieval over repair memory."""

from __future__ import annotations

import re
import sqlite3

from evalforge.memory.store import MemoryStore
from evalforge.models import MemoryRecord

_TOKEN = re.compile(r"[A-Za-z0-9_]{3,}")


class MemoryRetriever:
    def __init__(self, store: MemoryStore) -> None:
        self.store = store

    def retrieve(
        self,
        *,
        query: str,
        repository: str,
        category: str,
        exclude_task_id: str,
        top_k: int = 4,
    ) -> list[MemoryRecord]:
        if top_k < 1 or top_k > 20:
            raise ValueError("top_k must be between 1 and 20")
        terms = list(dict.fromkeys(token.lower() for token in _TOKEN.findall(query)))[:20]
        with self.store.connect() as connection:
            if terms:
                try:
                    rows = connection.execute(
                        """
                        SELECT m.*, bm25(memories_fts) AS lexical_score,
                               CASE WHEN m.repository = ? THEN -2.0 ELSE 0.0 END AS repo_bonus,
                               CASE WHEN m.task_category = ?
                                    THEN -1.5 ELSE 0.0 END AS category_bonus
                        FROM memories_fts
                        JOIN memories AS m ON m.memory_id = memories_fts.rowid
                        WHERE memories_fts MATCH ?
                          AND m.task_id != ?
                          AND m.outcome = 'resolved'
                        ORDER BY lexical_score + repo_bonus + category_bonus ASC,
                                 m.memory_id DESC
                        LIMIT ?
                        """,
                        (
                            repository,
                            category,
                            " OR ".join(f'"{term}"' for term in terms),
                            exclude_task_id,
                            top_k,
                        ),
                    ).fetchall()
                    return [self.store._from_row(row) for row in rows]
                except sqlite3.OperationalError:
                    pass
            like_terms = terms[:6] or [category.lower()]
            clauses = " OR ".join(
                "lower(lesson || ' ' || repair_pattern || ' ' || "
                "error_signature || ' ' || tags_json) LIKE ?"
                for _ in like_terms
            )
            rows = connection.execute(
                f"""
                SELECT * FROM memories
                WHERE task_id != ? AND outcome = 'resolved' AND ({clauses})
                ORDER BY CASE WHEN repository = ? THEN 0 ELSE 1 END,
                         CASE WHEN task_category = ? THEN 0 ELSE 1 END,
                         memory_id DESC
                LIMIT ?
                """,
                (
                    exclude_task_id,
                    *(f"%{term}%" for term in like_terms),
                    repository,
                    category,
                    top_k,
                ),
            ).fetchall()
        return [self.store._from_row(row) for row in rows]
