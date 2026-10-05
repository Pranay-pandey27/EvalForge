from __future__ import annotations

from pathlib import Path

from evalforge.memory.retrieval import MemoryRetriever
from evalforge.memory.store import MemoryStore
from evalforge.models import MemoryRecord


def record(task_id: str, category: str, lesson: str) -> MemoryRecord:
    return MemoryRecord(
        repository="scientific-mini",
        task_id=task_id,
        task_category=category,
        error_signature="broadcast shape mismatch",
        lesson=lesson,
        repair_pattern="preserve reduced dimensions",
        trace_id=f"trace-{task_id}",
        verifier_evidence="4 passed in 0.02s",
        outcome="resolved",
        tags=["numpy", "broadcasting"],
    )


def test_memory_round_trip_has_provenance(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "memory.sqlite")
    saved = store.add(record("task-a", "numpy_shape", "Use keepdims for broadcasting"))
    assert saved.memory_id is not None
    loaded = store.get(saved.memory_id)
    assert loaded is not None
    assert loaded.trace_id == "trace-task-a"
    assert loaded.verifier_evidence == "4 passed in 0.02s"


def test_retrieval_excludes_current_task_and_prefers_relevance(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "memory.sqlite")
    current = store.add(record("task-current", "numpy_shape", "Use keepdims for axes"))
    relevant = store.add(record("task-related", "numpy_shape", "Fix broadcasting with keepdims"))
    store.add(record("task-metric", "regression_metrics", "Take square root for RMSE"))
    results = MemoryRetriever(store).retrieve(
        query="NumPy broadcasting axis keepdims",
        repository="scientific-mini",
        category="numpy_shape",
        exclude_task_id="task-current",
        top_k=2,
    )
    ids = [item.memory_id for item in results]
    assert current.memory_id not in ids
    assert relevant.memory_id in ids


def test_duplicate_memory_is_idempotent(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "memory.sqlite")
    first = store.add(record("task-a", "numpy_shape", "Use keepdims"))
    second = store.add(record("task-a", "numpy_shape", "Use keepdims"))
    assert first.memory_id == second.memory_id
    assert len(store.all()) == 1
