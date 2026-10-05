"""Typed contracts shared across the EvalForge harness."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp."""

    return datetime.now(UTC)


class StrictModel(BaseModel):
    """Base model that rejects undocumented fields."""

    model_config = ConfigDict(extra="forbid")


class AgentState(StrEnum):
    TASK_RECEIVED = "task_received"
    INSPECT_REPOSITORY = "inspect_repository"
    RETRIEVE_MEMORY = "retrieve_memory"
    PLAN_REPAIR = "plan_repair"
    EDIT_FILES = "edit_files"
    RUN_VERIFIER = "run_verifier"
    DIAGNOSE_RESULT = "diagnose_result"
    FINISHED = "finished"


class TerminationReason(StrEnum):
    TESTS_PASSED = "tests_passed"
    MAX_ITERATIONS = "max_iterations"
    MAX_TOOL_CALLS = "max_tool_calls"
    MAX_TOKENS = "max_tokens"
    MAX_COST = "max_cost"
    ENVIRONMENT_FAILURE = "environment_failure"
    PROVIDER_ABORT = "provider_abort"
    INVALID_ACTION = "invalid_action"
    EVALUATOR_FAILED = "evaluator_failed"


class ActionKind(StrEnum):
    TOOL = "tool"
    SUBMIT = "submit"
    ABORT = "abort"


class Condition(StrEnum):
    BASELINE = "baseline"
    MEMORY = "memory"
    MEMORY_VERIFIER = "memory_verifier"
    SCRIPTED = "scripted"


class Budget(StrictModel):
    max_iterations: int = Field(default=8, ge=1)
    max_tool_calls: int = Field(default=40, ge=1)
    max_tokens: int | None = Field(default=None, ge=1)
    max_cost_usd: float | None = Field(default=None, gt=0)
    command_timeout_seconds: float = Field(default=60.0, gt=0, le=600)


class TaskMetadata(StrictModel):
    task_id: str = Field(min_length=1, pattern=r"^[a-zA-Z0-9_-]+$")
    title: str = Field(min_length=1)
    repository: str = Field(min_length=1)
    category: str = Field(min_length=1)
    difficulty: Literal["easy", "medium", "hard"]
    visible_test_command: list[str] = Field(min_length=1)
    evaluator_test_command: list[str] = Field(min_length=1)
    expected_failure: str = Field(min_length=1)
    dependencies: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    schema_version: str = "1.0"

    @field_validator("visible_test_command", "evaluator_test_command")
    @classmethod
    def command_must_be_tokenized(cls, value: list[str]) -> list[str]:
        if any(not token or "\x00" in token for token in value):
            raise ValueError("commands must contain non-empty, NUL-free tokens")
        return value


class TaskSpec(StrictModel):
    root: Path
    issue: str = Field(min_length=1)
    metadata: TaskMetadata

    @property
    def repo_dir(self) -> Path:
        return self.root / "repo"

    @property
    def evaluator_tests_dir(self) -> Path:
        return self.root / "tests"


class FailureDiagnosis(StrictModel):
    """Evidence-linked diagnosis required by the verifier-loop treatment."""

    failed_test: str = Field(min_length=1)
    evidence: list[str] = Field(min_length=1)
    mechanism: str = Field(min_length=1)
    next_change: str = Field(min_length=1)

    @field_validator("evidence")
    @classmethod
    def evidence_must_be_concrete(cls, value: list[str]) -> list[str]:
        if any(len(item.strip()) < 4 for item in value):
            raise ValueError("diagnosis evidence items must contain concrete output")
        return [item.strip() for item in value]


class AgentDecision(StrictModel):
    kind: ActionKind
    plan: str | None = None
    diagnosis: FailureDiagnosis | str | None = None
    lesson: str | None = Field(default=None, min_length=1, max_length=1200)
    tool_name: str | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_shape(self) -> AgentDecision:
        if self.kind is ActionKind.TOOL and not self.tool_name:
            raise ValueError("tool actions require tool_name")
        if self.kind is not ActionKind.TOOL and self.tool_name is not None:
            raise ValueError("only tool actions may include tool_name")
        return self


class Usage(StrictModel):
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    estimated_cost_usd: float | None = Field(default=None, ge=0)

    @property
    def total_tokens(self) -> int | None:
        if self.input_tokens is None and self.output_tokens is None:
            return None
        return (self.input_tokens or 0) + (self.output_tokens or 0)


class ProviderResponse(StrictModel):
    decision: AgentDecision
    raw_text: str | None = None
    usage: Usage = Field(default_factory=Usage)


class ToolResult(StrictModel):
    tool_name: str
    ok: bool
    output: str = ""
    exit_code: int | None = None
    runtime_seconds: float = Field(default=0.0, ge=0)
    truncated: bool = False


class VerificationResult(StrictModel):
    command: list[str]
    exit_code: int
    passed: bool
    output: str
    runtime_seconds: float = Field(ge=0)
    tests_collected: int | None = Field(default=None, ge=0)
    tests_passed: int | None = Field(default=None, ge=0)
    tests_failed: int | None = Field(default=None, ge=0)
    environment_error: bool = False


class TaskValidationResult(StrictModel):
    task_id: str
    valid: bool
    buggy_state_failed: bool
    reference_patch_applied: bool
    repaired_state_passed: bool
    buggy_verification: VerificationResult
    repaired_verification: VerificationResult | None = None
    error: str | None = None


class TraceEvent(StrictModel):
    sequence: int = Field(ge=0)
    timestamp: datetime = Field(default_factory=utc_now)
    event_type: str = Field(min_length=1)
    state: AgentState
    payload: dict[str, Any] = Field(default_factory=dict)
    schema_version: str = "1.0"


class RunSummary(StrictModel):
    run_id: str
    task_id: str
    condition: Condition
    started_at: datetime
    finished_at: datetime
    termination_reason: TerminationReason
    resolved: bool
    visible_verification: VerificationResult | None = None
    evaluator_verification: VerificationResult | None = None
    changed_files: list[str] = Field(default_factory=list)
    patch_valid: bool = True
    regression_detected: bool = False
    iterations: int = Field(ge=0)
    tool_calls: int = Field(ge=0)
    usage: Usage = Field(default_factory=Usage)
    runtime_seconds: float = Field(ge=0)
    trace_path: str
    workspace_path: str
    provider: str = "unknown"
    model: str | None = None
    retrieved_memory_ids: list[int] = Field(default_factory=list)
    schema_version: str = "1.0"


class MemoryRecord(StrictModel):
    """A concise, provenance-bearing lesson from a completed repair attempt."""

    memory_id: int | None = Field(default=None, ge=1)
    repository: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    task_category: str = Field(min_length=1)
    error_signature: str = Field(min_length=1)
    lesson: str = Field(min_length=1, max_length=2000)
    repair_pattern: str = Field(default="", max_length=2000)
    failed_approach: str = Field(default="", max_length=2000)
    trace_id: str = Field(min_length=1)
    verifier_evidence: str = Field(min_length=1, max_length=4000)
    outcome: Literal["resolved", "failed"]
    tags: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)
    schema_version: str = "1.0"

    def as_context(self) -> str:
        return (
            f"[{self.task_category}] {self.lesson} "
            f"Repair pattern: {self.repair_pattern or 'not recorded'}. "
            f"Evidence source: {self.trace_id}."
        )


class ExperimentConfig(StrictModel):
    """Reproducible experiment configuration loaded from JSON."""

    suite_dir: Path
    output_dir: Path = Path("experiments/runs")
    conditions: list[Condition] = Field(
        default_factory=lambda: [
            Condition.BASELINE,
            Condition.MEMORY,
            Condition.MEMORY_VERIFIER,
        ],
        min_length=1,
    )
    provider: Literal["litellm", "scripted"] = "litellm"
    model: str | None = None
    repeats: int = Field(default=1, ge=1, le=100)
    seed: int = 17
    budget: Budget = Field(default_factory=Budget)
    memory_db: Path | None = None
    persist_memory: bool = False
    runner: Literal["local", "docker"] = "local"
    docker_image: str = "evalforge-bench:latest"
    schema_version: str = "1.0"

    @model_validator(mode="after")
    def model_required_for_litellm(self) -> ExperimentConfig:
        if self.provider == "litellm" and not self.model:
            raise ValueError("model is required when provider is litellm")
        return self
