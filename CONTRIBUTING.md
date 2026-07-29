# Contributing

Stochastic Program IR is in alpha. Please open an issue before starting broad
API or semantic changes so the proposal can be checked against the
[v0.1 specification](https://github.com/upriyam-cmu/Stochastic-Program-IR/blob/main/specs/specification-v0.1.md).

## Development setup

The repository uses [uv](https://docs.astral.sh/uv/) for locked Python
environments:

```console
uv sync --locked --all-groups
```

Run the complete local validation suite before opening a pull request:

```console
uv run --locked --group dev ruff check .
uv run --locked --group dev ruff format --check .
uv run --locked --group dev -- ty check --python-version 3.10
uv run --locked --group dev pytest -W error --cov --cov-report=term-missing
uv run --locked --group docs sphinx-build -W --keep-going -b html docs docs/_build/html
uv run --locked --group docs python scripts/check_docs.py docs/_build/html
uv build
uv run --locked --group dev python scripts/check_wheel.py dist/*.whl
```

Tests belong in `tests/`, runtime code in `src/stoch_ir/`, normative behavior
in `specs/`, and user or contributor documentation in `docs/`.

## Pull requests

Keep changes narrowly scoped and update tests, documentation, and the changelog
when public behavior changes. CI validates Python 3.10, Python 3.11, and the
latest supported Python.

## Release checklist

The complete Trusted Publishing setup and release procedure is documented in
[Publishing releases](https://upriyam-cmu.github.io/Stochastic-Program-IR/releasing.html).

Before tagging a release:

1. Update the version in `pyproject.toml` and move the changelog entry from
   `Unreleased` to the release date.
2. Confirm all required CI checks pass on `main`.
3. Confirm the GitHub Pages documentation matches the release.
4. Create the `v<version>` tag only from the validated `main` commit.
5. Inspect and install the resulting TestPyPI upload in a clean environment.
6. Approve the protected `pypi` deployment only after validating that TestPyPI
   upload.
