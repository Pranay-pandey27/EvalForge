# EvalForge technical report

## Abstract

EvalForge is a verifier-first coding-agent research harness for compact
scientific Python and machine-learning repair tasks. It tests whether persistent
task memory and evidence-linked reflection improve resolution relative to a
baseline agent. The system contributes controlled task execution, protected
evaluation, provenance-bearing memory, trace-derived metrics, and a 20-task
held-out benchmark. Credential-free reference runs validate the apparatus but
are explicitly excluded from model-performance claims.

## Research design

The independent variable is controller support:

1. baseline tools and visible tests;
2. the same controller plus top-k persistent memory;
3. memory plus a mandatory structured diagnosis grounded in the latest failure output.

The dependent variables are task resolution, evaluator test pass rate, patch
validity, iterations, runtime, usage, cost, regression rate, and termination
taxonomy. Model, prompt core, task order seed, starting state, tools, budgets,
and verifier commands are held constant.

## System method

Each task is copied into a fresh Git repository. A provider returns one typed
decision per turn. The orchestrator validates the action, accounts for budgets,
executes a contained tool, and returns normalized evidence. A submit action is
rejected unless the latest visible test run passed. Protected tests are then
overlaid for one evaluator invocation and removed.

Memory is stored in SQLite rather than chat history. Records contain concise
lessons plus repository, category, error, trace, outcome, and verifier
provenance. FTS5 retrieval supplies only a bounded relevant subset and excludes
the current task. Evaluation uses a frozen database to prevent task order from
changing the treatment.

## Benchmark method

The task suite targets failure mechanisms frequently hidden by superficial code
generation demos: axis semantics, broadcasting, leakage, metric definitions,
training/evaluation mode, gradient lifecycle, device and dtype mismatches,
overflow, division by zero, global RNG mutation, physical units, PDE boundary
conditions, and nonstandard JSON types.

Each reference repair is validated independently. A task is admitted only when
the buggy version fails and the patched version passes all visible and evaluator
tests in a fresh workspace.

## Apparatus validation

The development suite produced eight resolved scripted traces and eight memory
records. The held-out suite then produced 60 resolved traces across three
conditions. All 60 patches passed the protected evaluator and repository
integrity check. This validates state transitions, treatment routing, memory
retrieval, reflection enforcement, evaluator isolation, termination, and metric
generation.

Because reference actions are deterministic and contain the known repair, this
validation cannot answer the research question.

## Required real-model protocol

Before publishing performance numbers:

1. Select one immutable provider model/version and decoding configuration.
2. Run development tasks to create memory with that same model.
3. Audit records for hidden-test or reference-patch leakage and freeze the
   database with `evalforge freeze-memory`.
4. Run all three conditions on all 20 evaluation tasks with identical budgets.
5. Prefer at least three repeats per task and condition.
6. Preserve the manifest, traces, summaries, model identifier, and pricing rule.
7. Generate tables from `evalforge analyze`; do not transcribe values manually.
8. Report uncertainty and paired task-level differences as exploratory.

## Threats to validity

- Twenty tasks provide limited statistical power and domain coverage.
- Public benchmark release weakens long-term contamination resistance.
- Lexical retrieval quality can confound the value of memory itself.
- Requiring reflection changes token use as well as reasoning structure.
- Protocol fakes test PyTorch control semantics but not autograd numerics.
- Provider APIs may change model behavior without a new public model name.
- Evaluator tests reduce visible-test overfitting but cannot prove semantic correctness.

## Conclusion

EvalForge is complete as an auditable experimental apparatus and benchmark. Its
scientific conclusion remains intentionally open until provider-backed ablation
runs are funded and executed.
