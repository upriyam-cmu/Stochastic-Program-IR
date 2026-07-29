# Stochastic Program IR

Stochastic Program IR is a small immutable random-variable algebra for
readable, structurally verifiable forward-sampling programs.

Named plates replace positional shape reasoning, explicit reductions replace
hidden contractions, and staged materialization replaces mutable RNG state.
Operations whose semantics cannot be understood locally—especially arbitrary
indexing—are intentionally excluded.

```{toctree}
:maxdepth: 2

getting-started
plates
phases
equality
values
alpha-stability
api-reference
contributor-architecture
```

```{toctree}
:hidden:

api-v0.1
graph-hashing-and-materialization
implementation-plan-v0.1
```

## License

Stochastic Program IR is distributed under the
[BSD 3-Clause License](https://opensource.org/license/bsd-3-clause).
