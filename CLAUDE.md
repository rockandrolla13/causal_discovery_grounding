# CLAUDE.md

Guidance for Claude Code and other agents working in this repository.

## What this repository is

The code of the paper *Causal Discovery Grounding and the Naturalness Assumption*: the
`causal_grounding` package and the scripts that reproduce the paper's tables and figures.
README.md describes the method; paper/README.md maps each script to the results it produces.

The paper's numbers must not change by accident. A refactor that is meant to leave results
unchanged must keep `tests/test_golden_toy_example.py` green. When code and paper disagree,
say so and ask. Do not silently change either to match the other.

## Environment

Use a project environment, never the global interpreter. The paper's runs used Python 3.11,
torch 2.5.1 and the pins of `requirements.txt`:

```bash
python3.11 -m venv .venv
./.venv/bin/python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1+cpu
./.venv/bin/python -m pip install -r requirements-dev.txt
```

Never `pip install` for this project into a shared (conda base, system) environment: it can
change numpy for unrelated projects, and test results from an environment nobody else has are
not trustworthy. Tests that need an optional package (`tabicl`, `numpyro`, `pyreadstat`, ...)
skip when it is missing, so a green run in a thin environment is not full coverage.

## Tests and lint

```bash
./.venv/bin/python -m pytest tests/test_causal_grounding/ -q    # the package, a few minutes
./.venv/bin/python -m pytest tests/test_golden_toy_example.py   # result guard, about 2 minutes
./.venv/bin/python -m pytest -q                                 # full suite, about 25 minutes
ruff check .
```

Run targeted test files while developing; keep the full suite for CI or a deliberate long run.
CI (`.github/workflows/tests.yml`) runs both on every push and pull request to `main`.

## Code conventions

- Python 3.11+, type hints, `X | None` rather than `Optional[X]` in new code.
- NumPy-style docstrings; scikit-learn conventions (`fit` returns `self`, fitted attributes end
  in `_`) for estimator-like classes.
- Seed every random step through an explicit `random_seed` / `numpy.random.Generator`; do not
  call the global `np.random.seed`.
- The package never imports `experiments/`. Experiment scripts must not import each other
  either: shared logic goes into the package or into a domain-named module under
  `experiments/` that imports no experiment script.
- Heavy optional dependencies (torch, tabicl, jax, numpyro) are imported lazily, inside the
  function that needs them, so `pip install .` works with numpy, pandas, scipy and
  scikit-learn only.
- Paper grids and constants (epsilon/tau/beta grids, thresholds) have one definition; import
  it rather than copying the literal.
