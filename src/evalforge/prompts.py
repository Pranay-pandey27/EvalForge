"""Provider-neutral prompts for structured coding-agent decisions."""

from __future__ import annotations

import json

from evalforge.providers.base import AgentContext

SYSTEM_PROMPT = """You are the decision component inside EvalForge, a controlled
scientific-Python repair harness. Return exactly one JSON object and no prose.
You may request one tool: list_files, read_file, search_text, apply_patch, shell,
or run_tests. Never claim success without a passing run_tests result. Submit only
after the latest visible verifier passed. Keep edits minimal and evidence-based.

Decision schema:
{
  "kind": "tool|submit|abort",
  "plan": "short plan or null",
  "diagnosis": null OR {
    "failed_test": "test node or failure label",
    "evidence": ["exact excerpt from the latest test output"],
    "mechanism": "causal explanation",
    "next_change": "specific next change"
  },
  "lesson": "concise reusable lesson on submit, or null",
  "tool_name": "tool name or null",
  "arguments": {}
}
"""


def build_user_prompt(context: AgentContext) -> str:
    results = [result.model_dump(mode="json") for result in context.previous_results]
    payload = {
        "task_id": context.task_id,
        "issue": context.issue,
        "state": context.state,
        "repository_files": context.repository_files,
        "recent_tool_results": results,
        "latest_verification": (
            context.latest_verification.model_dump(mode="json")
            if context.latest_verification
            else None
        ),
        "iteration": context.iteration,
        "tool_calls": context.tool_calls,
        "relevant_memory": context.memory,
        "structured_diagnosis_required_before_edit": context.require_structured_diagnosis,
    }
    return json.dumps(payload, indent=2)
