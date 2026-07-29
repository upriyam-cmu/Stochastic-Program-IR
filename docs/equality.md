# Structural equality and stochastic equality

Expression equality answers whether two graphs represent the same computation
structure. It ignores Python object aliasing and unresolved RNG state.

```python
from stoch_ir import Normal, sampling_phase

with sampling_phase("draw"):
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

This comparison concerns the checkpoint's current state. Once distributions
are sampled, they become constants and their historical RNG provenance is not
retained. Fully materialized checkpoints with equal values and plate sizes are
therefore stochastically equal. Keep distributions blocked behind phases when
the comparison should inspect unresolved sharing and RNG addresses.

An optional `rng_label` is semantic metadata mixed into graph-derived entropy.
It never replaces the structural contribution or forces two distinct nodes to
share randomness.
