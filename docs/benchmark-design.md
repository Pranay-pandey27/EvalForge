# Benchmark design

EvalForge contains two deliberately separate suites:

- eight development tasks for harness construction and memory seeding;
- 20 held-out tasks for the primary three-condition comparison.

## Task contract

```text
task_NNN/
├── repo/                  # buggy state and visible tests
├── tests/                 # evaluator-only tests
├── issue.md               # provider-visible objective
├── metadata.json          # category, commands, dependencies, difficulty
├── reference_patch.diff   # task validation oracle, never provider-visible
└── scripted_actions.json  # apparatus validation, never a model result
```

Every task must satisfy four executable invariants:

1. The buggy state fails relevant visible tests.
2. The reference patch applies to a fresh Git copy.
3. The repaired state passes visible and evaluator tests.
4. The task runs without network, GPU, large data, or proprietary credentials.

`evalforge validate-suite ... --execute` proves these properties rather than
trusting metadata.

## Held-out distribution

| Primary mechanism | Count | Example |
|---|---:|---|
| NumPy shape/axis/broadcasting | 3 | row-wise stable softmax |
| Pandas leakage/split handling | 2 | train-fitted categories |
| Classification/regression metrics | 3 | macro F1 versus accuracy |
| PyTorch loop/mode/gradient semantics | 4 | accumulation windows |
| Device and dtype mismatch | 2 | nested batch movement |
| Numerical stability | 2 | stable log-sum-exp |
| Random seeding | 1 | deterministic split indices |
| Units and boundary conditions | 2 | periodic Laplacian |
| Result serialization | 1 | NumPy JSON and nonfinite values |

## Why protocol-compatible PyTorch tests

Training-loop correctness is primarily about observable protocol calls and
ordering. Lightweight fakes test `train`, `eval`, `zero_grad`, `backward`,
`step`, `to`, `float`, and `long` without installing a multi-hundred-megabyte
framework or requiring accelerators. The production functions remain directly
compatible with real PyTorch objects because they use those same public
protocols.

## Leakage and contamination controls

- Development and evaluation task IDs are disjoint.
- Evaluation repairs differ from development repairs even when categories overlap.
- Reference patches are never inserted into prompts or memory.
- Evaluator tests appear only after submission.
- Memory retrieval excludes the current task and returns source trace IDs.
- Evaluation memory is frozen before condition runs.
- The same starting repository is recreated for every condition and repeat.

Public release makes evaluator source inspectable to humans. The benchmark is
therefore best for controlled local experiments where the model is not given
repository-wide access to the benchmark definition itself. A future external
test service would provide stronger secrecy.
