# Stochastic Program IR

> Status: v0.1 alpha. The public API is still allowed to change.

Stochastic Program IR is a small, immutable random-variable algebra for writing
forward-sampling programs that read like ordinary mathematical array code.
Named plates replace positional shape reasoning; explicit reductions replace
hidden contractions; staged materialization replaces mutable RNG state.
Features whose semantics cannot be understood locally—especially arbitrary
indexing—are intentionally excluded.

The project is designed for simulations, generators, and agent-authored
stochastic programs where probabilistic structure should remain easy to read,
compare, and validate. It performs forward sampling only; it is not a Bayesian
inference framework.

## Why an IR?

“IR” describes the immutable expression graph that the API constructs. The
graph preserves:

- which values are sampled and which operations are deterministic;
- where independent replication is introduced;
- which named plates are preserved, checked, or reduced;
- which sampling phase owns each unresolved distribution;
- structural equality independently from resolved stochastic sharing; and
- partially materialized values without mutable runtime sessions.

The name does not imply that v0.1 includes serialization, optimization passes,
compilation, or multiple numeric backends.

## Quick start

```python
from stoch_ir import Normal, sampling_phase, softplus

with sampling_phase("latent"):
    weights = Normal(0.0, 1.0, rng_label="weights").add_plates("layer")

with sampling_phase("observation"):
    activations = Normal(
        mu=weights,
        sigma=softplus(weights) + 0.1,
        rng_label="activations",
    ).add_plates("batch", expect=("layer",))

layer_score = activations.mean("batch").check_plates("layer")
```

`add_plates(..., expect=...)` states both sides of the structural change:
`activations` must already vary over `"layer"`, and the operation intentionally
introduces independent samples over `"batch"`.

The same graph can be realized in stages:

```python
sizes = {"layer": 8, "batch": 32}

fixed_latent = layer_score.materialize(
    phases=("latent",),
    seed=100,
    plate_sizes=sizes,
)

sample_a = fixed_latent.realize(seed=200)
sample_b = fixed_latent.realize(seed=201)
```

`fixed_latent` is an opaque immutable checkpoint. Its latent draws have become
constants, while its observation draws remain symbolic. Branching from that
checkpoint reuses the latent values and resamples only the remaining
uncertainty.

## Explicit contractions

Named alignment makes familiar array operations readable without positional
axis bookkeeping. For example, a matrix product is elementwise multiplication
followed by an explicit reduction:

```python
from stoch_ir import Normal

# left varies over {"row", "inner"}
left = Normal(0.0, 1.0).add_plates("row", "inner")

# right varies over {"inner", "col"}
right = Normal(0.0, 1.0).add_plates("inner", "col")

product = (
    (left * right)
    .sum("inner")
    .check_plates("row", "col")
)
```

This replaces an implicit `left @ right` contraction with source code that
names the contracted plate. The missing `@` operator is therefore not a
capability gap in v0.1: the primitive expression is more explicit about the
stochastic and array structure.

## Distributions and concrete inputs

v0.1 includes normal, arbitrary-bound continuous uniform, and Bernoulli draws:

```python
from stoch_ir import Bernoulli, Uniform, sampling_phase

with sampling_phase("probability"):
    probability = Uniform().add_plates("group")

with sampling_phase("trial"):
    trial = Bernoulli(probability).add_plates(
        "trial",
        expect=("group",),
    )

rate = trial.mean("trial").check_plates("group")
```

Existing NumPy values enter through one explicit boundary:

```python
import numpy as np

from stoch_ir import constant

offset = constant(
    np.array([0.1, 0.2], dtype=np.float32),
    plates=("group",),
)
```

`constant` infers Boolean, integer, or floating metadata and stores values
canonically as `np.bool_`, `np.int64`, or `np.float64`. Complex, object, string,
and unnamed multidimensional values are rejected.

## Plate algebra

Plate names are strings. Sizes are supplied only when a graph is materialized.
The public operations are deliberately small:

- `add_plates(*new, expect=None)` introduces independent replication;
- `check_plates(*expected)` validates the complete plate set;
- `reduce_plates(*plates, reduction=...)` explicitly contracts plates; and
- `mean`, `sum`, `max`, `min`, `prod`, and `logsumexp` provide named
  convenience reductions.

`expr.plates` returns the canonical lexicographic tuple used to align concrete
array axes. Contract methods accept plate names in any order.

## Structural and stochastic equality

Expression equality compares exact computation structure—not numerical,
algebraic, or absolute probabilistic equivalence. It intentionally ignores
object aliasing and unresolved randomness:

```python
from stoch_ir import Normal

x = Normal(0.0, 1.0)

shared = x + x
independent = Normal(0.0, 1.0) + Normal(0.0, 1.0)

assert shared == independent
```

Graph-derived node entropy is resolved only during materialization. The
resulting checkpoints therefore expose `stochastically_equal`, which also
checks stochastic sharing and resolved RNG state:

```python
shared_checkpoint = shared.materialize(seed=1, phases=())
independent_checkpoint = independent.materialize(seed=1, phases=())

assert not shared_checkpoint.stochastically_equal(independent_checkpoint)
```

An `rng_label` supplements graph-derived entropy but never replaces it or opts
distinct nodes into shared randomness.

## Intentional v0.1 boundaries

The project intentionally excludes arbitrary indexing, positional-axis
operations, hidden contractions, mutable RNG sessions, probabilistic inference,
autodiff, JIT compilation, serialization, graph optimization, custom nodes or
reductions, and alternate numeric backends.

v0.1 uses NumPy internally. Conversion to other array libraries belongs at the
API boundary.

## Documentation and development

- [Documentation source](docs/index.md)
- [v0.1 specification](specs/specification-v0.1.md)
- [Public API contract](docs/api-v0.1.md)
- [Hashing and materialization architecture](docs/graph-hashing-and-materialization.md)
- [Implementation plan](docs/implementation-plan-v0.1.md)

The package ships inline annotations with `py.typed`. CI checks Ruff, ty, tests
with a 95% branch-coverage floor, built-wheel imports, and the documentation
site.
