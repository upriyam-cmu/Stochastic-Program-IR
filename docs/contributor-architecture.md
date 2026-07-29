# Contributor architecture

The public random-variable algebra is backed by immutable expression nodes,
canonical named-plate metadata, and an immutable materialization rewrite.

These pages describe internals for contributors; they are not extension
contracts:

- [Graph hashing and materialization](graph-hashing-and-materialization.md)
- [v0.1 implementation plan](implementation-plan-v0.1.md)

Concrete node classes, dependency reconstruction, operation implementations,
plate layouts, and stochastic hash passes remain internal in v0.1.

The normative product specification remains in the
[source repository](https://github.com/upriyam-cmu/Stochastic-Program-IR/blob/main/specs/specification-v0.1.md).

## Local validation

Install every development group from the lockfile:

```console
uv sync --locked --all-groups
```

Run the same core checks as CI:

```console
uv run --locked --group dev ruff check .
uv run --locked --group dev ruff format --check .
uv run --locked --group dev -- ty check --python-version 3.10
uv run --locked --group dev pytest -W error --cov --cov-report=term-missing
```

Build the documentation with warnings treated as errors:

```console
uv run --locked --group docs sphinx-build -W --keep-going -b html docs docs/_build/html
uv run --locked --group docs python scripts/check_docs.py docs/_build/html
```

The generated `docs/_build/` directory is intentionally ignored. CI builds the
site for every branch and pull request; pushes to `main` additionally publish
the checked HTML through GitHub Pages.
