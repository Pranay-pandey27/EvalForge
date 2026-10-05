# Known evaluation limitations

The preceding notebook audit identified these limitations. This publication
preparation documents them; it does not claim they have been repaired.

- The verifier treatment requires diagnosis after an observed failure, but the
  first edit can occur before an initial failing test.
- Exit-code-based verification can accept a run in which tests were skipped.
- Visible tests and pytest configuration are not fully protected against edits.
- Memory snapshots are writable; held-out evaluation relies on the no-write
  workflow convention rather than enforced database immutability.
- Command timeouts do not provide a whole-run wall-clock deadline.
- The local runner is for trusted examples, not arbitrary adversarial code.
- Saved 60/60 results are reference-script harness checks, not LLM performance
  or evidence that memory improves repair accuracy.

The standalone notebook also lacks the package's strict Pydantic contracts,
real-provider adapter, Docker runner, CLI and separate automated tests.
