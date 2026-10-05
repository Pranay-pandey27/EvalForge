# Results

> **Infrastructure validation only.** These runs used deterministic reference
> scripts and are not evidence of LLM coding performance.

The checked-in validation exercised the 20 held-out tasks under all three
controller conditions. The eight development tasks first seeded a SQLite memory
database. Evaluation memory was then frozen.

| Condition | Runs | Resolution | Test pass | Patch validity | Avg iterations | Avg runtime (s) | Regression |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 20 | 100.0% | 100.0% | 100.0% | 2.000 | 3.176 | 0.0% |
| Memory | 20 | 100.0% | 100.0% | 100.0% | 2.000 | 3.195 | 0.0% |
| Memory + verifier | 20 | 100.0% | 100.0% | 100.0% | 2.000 | 3.143 | 0.0% |

Additional apparatus evidence:

- 60 of 60 reference-script runs resolved.
- Every run passed protected evaluator tests and `git diff --check`.
- Memory and memory-plus-verifier runs each retrieved four memories.
- Baseline runs retrieved no memories.
- Memory-plus-verifier repairs passed exact verifier-evidence enforcement.
- Scripted token and cost fields are unavailable because no model was called.

The compact summaries, traces, manifest, memory export, CSV table, JSON
aggregate, and plot are in `experiments/artifacts/scripted-validation/` and
`experiments/analysis/scripted-validation/`.

## What these numbers mean

They show that every experimental pathway can execute the same controlled task
set without corrupting workspaces, leaking evaluator tests, bypassing the
verifier, or losing provenance.

## What these numbers do not mean

They do not show that memory improves a coding model. The scripted provider is
given each reference patch and therefore has no meaningful uncertainty. Real
ablation claims require one fixed model configuration, the same frozen memory,
the same budgets, repeated runs, and results generated from those new traces.
