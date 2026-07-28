# Structural equality and stochastic equality

Expression equality answers whether two graphs represent the same computation
structure. It ignores Python object aliasing and unresolved RNG state.

```python
from stoch_ir import Normal

source = Normal(0.0, 1.0)
shared = source + source
independent = Normal(0.0, 1.0) + Normal(0.0, 1.0)

assert shared == independent
```

This is exact structural comparison, not numerical closeness, algebraic
equivalence, or equality of probability laws.

Comparison uses a memo table keyed by pairs of node identities. Aliasing is
still ignored semantically, but a shared subgraph is traversed only once for
each counterpart rather than expanded repeatedly.

Graph-aware entropy is resolved as part of materialization. A checkpoint can
therefore compare stochastic sharing as well:

```python
shared_state = shared.materialize(seed=1, phases=())
independent_state = independent.materialize(seed=1, phases=())

assert not shared_state.stochastically_equal(independent_state)
```

An optional `rng_label` is semantic metadata mixed into graph-derived entropy.
It never replaces the structural contribution or forces two distinct nodes to
share randomness.
