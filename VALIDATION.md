# Validation record

## Publication check — 2026-10-05

On Linux with Python 3.12.3, `uv sync --python /usr/bin/python3 --extra dev`
installed the dependencies from `uv.lock`. The project then passed:

- `pytest -q`: **27 passed**.
- `ruff check src tests`: **all checks passed**.
- Executed behavioral validation of the development and evaluation suites:
  all **28 tasks** passed the buggy-state and reference-repair checks.

These checks use the deterministic scripted reference repairs. They do not
measure LLM agent performance. Docker and hosted GitHub Actions were not run
in this publication check.

## Earlier validation — 2026-09-11

Date: 2026-09-11. Environment: Linux, Python 3.12.14.

- Existing automated project tests: **27 passed**.
- Canonical notebook: **6 code cells passed** in sequential Python execution from the repository root.
- Notebook metrics and model artifacts were regenerated. Existing scientific result limitations remain applicable.
- All notebook files passed structural validation.
- No high-confidence credential patterns were found in the inspected text files.

- Package and test source passed Ruff. The canonical notebook validated one repair task and executed one scripted repair; the complete 28-task/60-run historical report remains included. No new real-model experiment was run.

The Jupyter kernel transport could not start in this execution environment.
Notebook code was therefore executed using the existing code-cell runner,
not through a live Jupyter frontend. Docker builds and hosted GitHub Actions
have not been run. The standalone EvalForge notebook, when present, was
structurally checked but was not rerun as part of this publication pass.

Direct dependency versions are recorded in `requirements-validated.txt`.
For matching saved scikit-learn model serialization, install that file before
the development requirements or regenerate models after changing versions.

These checks validate the included project scope; they do not certify a
production service or resolve all research and evaluation limitations.
