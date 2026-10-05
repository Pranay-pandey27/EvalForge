"""Verifier-first orchestration with memory, reflection, and hard budgets."""

from __future__ import annotations

import shutil
import time
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evalforge.benchmark import TaskWorkspace
from evalforge.memory.retrieval import MemoryRetriever
from evalforge.memory.store import MemoryStore
from evalforge.models import (
    ActionKind,
    AgentDecision,
    AgentState,
    Budget,
    Condition,
    FailureDiagnosis,
    MemoryRecord,
    RunSummary,
    TaskSpec,
    TerminationReason,
    ToolResult,
    Usage,
    VerificationResult,
)
from evalforge.providers.base import AgentContext, Provider
from evalforge.runners.base import CommandRunner
from evalforge.state_machine import InvalidTransition, StateMachine
from evalforge.tools.filesystem import FileSystemTools, PathViolation
from evalforge.tools.shell import ControlledShell
from evalforge.tools.testing import TestVerifier
from evalforge.tracing import TraceWriter


class Orchestrator:
    def __init__(
        self,
        *,
        task: TaskSpec,
        provider: Provider,
        run_root: Path,
        budget: Budget | None = None,
        condition: Condition | str = Condition.SCRIPTED,
        memory_store: MemoryStore | None = None,
        persist_memory: bool = True,
        memory_top_k: int = 4,
        verifier_runner_factory: Callable[[Path], CommandRunner] | None = None,
    ) -> None:
        self.task = task
        self.provider = provider
        self.run_root = run_root
        self.budget = budget or Budget()
        self.condition = Condition(condition)
        self.memory_store = memory_store
        self.persist_memory = persist_memory
        self.memory_top_k = memory_top_k
        self.verifier_runner_factory = verifier_runner_factory

    def run(self) -> RunSummary:
        started_at = datetime.now(UTC)
        started_clock = time.monotonic()
        run_id = f"{self.task.metadata.task_id}-{uuid.uuid4().hex[:10]}"
        workspace = TaskWorkspace.create(self.task, self.run_root, run_id)
        trace = TraceWriter(workspace.path.parent / "trace.jsonl")
        machine = StateMachine()
        fs = FileSystemTools(workspace.path)
        shell = ControlledShell(
            workspace.path,
            timeout_seconds=self.budget.command_timeout_seconds,
        )
        runner = (
            self.verifier_runner_factory(workspace.path) if self.verifier_runner_factory else None
        )
        verifier = TestVerifier(
            workspace.path,
            timeout_seconds=self.budget.command_timeout_seconds,
            runner=runner,
        )
        results: list[ToolResult] = []
        visible: VerificationResult | None = None
        evaluator: VerificationResult | None = None
        iterations = 0
        tool_calls = 0
        usage = Usage()
        termination: TerminationReason | None = None
        diagnosis_required = False
        last_submit: AgentDecision | None = None
        last_diagnosis: FailureDiagnosis | None = None
        memories: list[MemoryRecord] = []

        trace.record("state_entered", machine.state, task_id=self.task.metadata.task_id)
        machine.transition(AgentState.INSPECT_REPOSITORY)
        trace.record("state_entered", machine.state)
        machine.transition(AgentState.RETRIEVE_MEMORY)
        trace.record("state_entered", machine.state)
        memories = self._retrieve_memory(trace, machine.state)
        machine.transition(AgentState.PLAN_REPAIR)
        trace.record("state_entered", machine.state)

        while termination is None:
            if iterations >= self.budget.max_iterations:
                termination = TerminationReason.MAX_ITERATIONS
                break
            if tool_calls >= self.budget.max_tool_calls:
                termination = TerminationReason.MAX_TOOL_CALLS
                break
            if self.provider.exhausted:
                termination = TerminationReason.PROVIDER_ABORT
                break

            context = AgentContext(
                task_id=self.task.metadata.task_id,
                issue=self.task.issue,
                state=machine.state,
                repository_files=fs.list_files().output.splitlines(),
                previous_results=results[-5:],
                latest_verification=visible,
                iteration=iterations,
                tool_calls=tool_calls,
                memory=[record.as_context() for record in memories],
                require_structured_diagnosis=(
                    self.condition is Condition.MEMORY_VERIFIER and diagnosis_required
                ),
            )
            try:
                response = self.provider.next_action(context)
            except Exception as exc:  # providers are an external failure boundary
                trace.record(
                    "provider_error",
                    machine.state,
                    error_type=type(exc).__name__,
                    message=str(exc),
                )
                termination = TerminationReason.PROVIDER_ABORT
                break
            trace.record(
                "provider_decision",
                machine.state,
                decision=response.decision.model_dump(mode="json"),
                usage=response.usage.model_dump(mode="json"),
            )
            usage = self._add_usage(usage, response.usage)
            budget_reason = self._usage_budget_reason(usage)
            if budget_reason is not None:
                termination = budget_reason
                break

            decision = response.decision
            if isinstance(decision.diagnosis, FailureDiagnosis):
                last_diagnosis = decision.diagnosis
            if decision.kind is ActionKind.ABORT:
                termination = TerminationReason.PROVIDER_ABORT
                break
            if decision.kind is ActionKind.SUBMIT:
                last_submit = decision
                if visible is None or not visible.passed:
                    trace.record(
                        "submission_rejected",
                        machine.state,
                        reason="latest visible verification did not pass",
                    )
                    termination = TerminationReason.INVALID_ACTION
                    break
                evaluator = self._run_evaluator(workspace, verifier, trace, machine.state)
                if evaluator.environment_error:
                    termination = TerminationReason.ENVIRONMENT_FAILURE
                elif evaluator.passed and workspace.patch_is_valid():
                    termination = TerminationReason.TESTS_PASSED
                else:
                    termination = TerminationReason.EVALUATOR_FAILED
                break

            tool_calls += 1
            if (
                decision.tool_name == "apply_patch"
                and self.condition is Condition.MEMORY_VERIFIER
                and diagnosis_required
            ):
                diagnosis_error = self._diagnosis_error(decision.diagnosis, visible)
                if diagnosis_error:
                    result = ToolResult(
                        tool_name="apply_patch",
                        ok=False,
                        output=f"edit rejected: {diagnosis_error}",
                    )
                    results.append(result)
                    trace.record(
                        "diagnosis_rejected",
                        machine.state,
                        reason=diagnosis_error,
                    )
                    continue

            try:
                result, visible = self._execute_tool(
                    decision.tool_name or "", decision.arguments, fs, shell, verifier, visible
                )
            except (PathViolation, TypeError, ValueError) as exc:
                result = ToolResult(
                    tool_name=decision.tool_name or "unknown",
                    ok=False,
                    output=f"invalid tool request: {exc}",
                )
            results.append(result)
            trace.record("tool_result", machine.state, result=result.model_dump(mode="json"))

            if result.tool_name == "apply_patch" and result.ok:
                diagnosis_required = False
                self._transition_if_needed(machine, AgentState.EDIT_FILES, trace)
            elif result.tool_name == "run_tests":
                iterations += 1
                self._transition_if_needed(machine, AgentState.RUN_VERIFIER, trace)
                if visible and visible.environment_error:
                    termination = TerminationReason.ENVIRONMENT_FAILURE
                    break
                if visible and not visible.passed:
                    diagnosis_required = True
                    machine.transition(AgentState.DIAGNOSE_RESULT)
                    trace.record(
                        "state_entered",
                        machine.state,
                        verifier_output=visible.output,
                    )
                elif visible and visible.passed:
                    diagnosis_required = False
                    trace.record("visible_tests_passed", machine.state)

        if machine.state is not AgentState.FINISHED:
            try:
                machine.transition(AgentState.FINISHED)
            except InvalidTransition:
                machine.state = AgentState.FINISHED
        trace.record(
            "run_terminated",
            machine.state,
            termination_reason=termination,
        )

        finished_at = datetime.now(UTC)
        patch_valid = workspace.patch_is_valid()
        resolved = termination is TerminationReason.TESTS_PASSED and patch_valid
        summary = RunSummary(
            run_id=run_id,
            task_id=self.task.metadata.task_id,
            condition=self.condition,
            started_at=started_at,
            finished_at=finished_at,
            termination_reason=termination or TerminationReason.ENVIRONMENT_FAILURE,
            resolved=resolved,
            visible_verification=visible,
            evaluator_verification=evaluator,
            changed_files=workspace.changed_files(),
            patch_valid=patch_valid,
            regression_detected=bool(visible and not visible.passed),
            iterations=iterations,
            tool_calls=tool_calls,
            usage=usage,
            runtime_seconds=time.monotonic() - started_clock,
            trace_path=str(trace.path),
            workspace_path=str(workspace.path),
            provider=type(self.provider).__name__,
            model=getattr(self.provider, "model", None),
            retrieved_memory_ids=[
                record.memory_id for record in memories if record.memory_id is not None
            ],
        )
        trace.write_summary(summary)
        if resolved and last_submit is not None:
            self._save_memory(summary, last_submit, last_diagnosis)
        return summary

    def _retrieve_memory(self, trace: TraceWriter, state: AgentState) -> list[MemoryRecord]:
        if self.condition not in {Condition.MEMORY, Condition.MEMORY_VERIFIER}:
            trace.record("memory_skipped", state, reason="condition disables memory")
            return []
        if self.memory_store is None:
            trace.record("memory_skipped", state, reason="no memory store configured")
            return []
        query = " ".join(
            [
                self.task.issue,
                self.task.metadata.category,
                self.task.metadata.expected_failure,
                *self.task.metadata.tags,
            ]
        )
        memories = MemoryRetriever(self.memory_store).retrieve(
            query=query,
            repository=self.task.metadata.repository,
            category=self.task.metadata.category,
            exclude_task_id=self.task.metadata.task_id,
            top_k=self.memory_top_k,
        )
        trace.record(
            "memory_retrieved",
            state,
            memory_ids=[memory.memory_id for memory in memories],
            count=len(memories),
        )
        return memories

    def _save_memory(
        self,
        summary: RunSummary,
        decision: AgentDecision,
        diagnosis: FailureDiagnosis | None,
    ) -> None:
        lesson = decision.lesson
        if not lesson and diagnosis:
            lesson = f"{diagnosis.mechanism} Next repair: {diagnosis.next_change}"
        if (
            self.condition not in {Condition.MEMORY, Condition.MEMORY_VERIFIER}
            or self.memory_store is None
            or not self.persist_memory
            or not lesson
            or summary.evaluator_verification is None
        ):
            return
        evidence = summary.evaluator_verification.output[-3500:].strip()
        record = MemoryRecord(
            repository=self.task.metadata.repository,
            task_id=self.task.metadata.task_id,
            task_category=self.task.metadata.category,
            error_signature=self.task.metadata.expected_failure,
            lesson=lesson,
            repair_pattern=f"Changed: {', '.join(summary.changed_files)}",
            trace_id=summary.run_id,
            verifier_evidence=evidence or "evaluator passed with exit code 0",
            outcome="resolved",
            tags=self.task.metadata.tags,
        )
        self.memory_store.add(record)

    @staticmethod
    def _diagnosis_error(
        diagnosis: FailureDiagnosis | str | None,
        verification: VerificationResult | None,
    ) -> str | None:
        if not isinstance(diagnosis, FailureDiagnosis):
            return "a structured FailureDiagnosis is required before editing"
        if verification is None or verification.passed:
            return "no failing verifier evidence is available"
        output = verification.output
        if not any(item in output for item in diagnosis.evidence):
            return "at least one evidence item must exactly cite the latest test output"
        if diagnosis.failed_test not in output:
            return "failed_test must occur in the latest test output"
        return None

    @staticmethod
    def _transition_if_needed(
        machine: StateMachine, target: AgentState, trace: TraceWriter
    ) -> None:
        if machine.state is target:
            return
        try:
            machine.transition(target)
        except InvalidTransition:
            if machine.state is AgentState.DIAGNOSE_RESULT and target is AgentState.EDIT_FILES:
                machine.transition(AgentState.EDIT_FILES)
            elif machine.state is AgentState.PLAN_REPAIR and target is AgentState.RUN_VERIFIER:
                machine.transition(AgentState.RUN_VERIFIER)
            else:
                raise
        trace.record("state_entered", machine.state)

    def _execute_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        fs: FileSystemTools,
        shell: ControlledShell,
        verifier: TestVerifier,
        latest_verification: VerificationResult | None,
    ) -> tuple[ToolResult, VerificationResult | None]:
        if tool_name == "list_files":
            return fs.list_files(**arguments), latest_verification
        if tool_name == "read_file":
            return fs.read_file(**arguments), latest_verification
        if tool_name == "search_text":
            return fs.search_text(**arguments), latest_verification
        if tool_name == "apply_patch":
            return fs.apply_patch(**arguments), latest_verification
        if tool_name == "shell":
            return shell.run(**arguments), latest_verification
        if tool_name == "run_tests":
            verification = verifier.run(self.task.metadata.visible_test_command)
            return (
                ToolResult(
                    tool_name="run_tests",
                    ok=verification.passed,
                    output=verification.output,
                    exit_code=verification.exit_code,
                    runtime_seconds=verification.runtime_seconds,
                ),
                verification,
            )
        raise ValueError(f"unknown tool: {tool_name}")

    def _run_evaluator(
        self,
        workspace: TaskWorkspace,
        verifier: TestVerifier,
        trace: TraceWriter,
        state: AgentState,
    ) -> VerificationResult:
        overlay = workspace.path / ".evalforge_evaluator_tests"
        shutil.copytree(self.task.evaluator_tests_dir, overlay)
        try:
            result = verifier.run(self.task.metadata.evaluator_test_command)
            trace.record("evaluator_result", state, result=result.model_dump(mode="json"))
            return result
        finally:
            shutil.rmtree(overlay, ignore_errors=True)

    def _usage_budget_reason(self, usage: Usage) -> TerminationReason | None:
        if self.budget.max_tokens is not None:
            total = usage.total_tokens or 0
            if total > self.budget.max_tokens:
                return TerminationReason.MAX_TOKENS
        if (
            self.budget.max_cost_usd is not None
            and (usage.estimated_cost_usd or 0) > self.budget.max_cost_usd
        ):
            return TerminationReason.MAX_COST
        return None

    @staticmethod
    def _add_usage(left: Usage, right: Usage) -> Usage:
        def add_optional(a: int | float | None, b: int | float | None) -> int | float | None:
            if a is None and b is None:
                return None
            return (a or 0) + (b or 0)

        return Usage(
            input_tokens=add_optional(left.input_tokens, right.input_tokens),
            output_tokens=add_optional(left.output_tokens, right.output_tokens),
            estimated_cost_usd=add_optional(left.estimated_cost_usd, right.estimated_cost_usd),
        )
