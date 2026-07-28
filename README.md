# Stochastic Program IR

> Status: v0.1 alpha. The API is still under active development.

Stochastic Program IR is a structural intermediate representation for authoring
stochastic programs that are easy to read, compare, validate, and realize in
stages.

The central idea is to preserve stochastic intent in the program itself. Distribution dependencies remain symbolic, independent replication is expressed with named plates, and sampling is grouped into phases that can be materialized independently. This makes important relationships visible to both human readers and code-generating agents instead of hiding them behind mutable RNG state and positional tensor axes.

The project is deliberately not a probabilistic inference framework. v0.1 is aimed at simulations, generators, and other forward-sampling programs where structural correctness and reproducibility matter.

## What the API is designed to make explicit

- which values are sampled and which expressions are deterministic;
- which samples are shared and which are independently replicated;
- which named plates are added, preserved, checked, or reduced;
- which phase of a program may be sampled at each materialization step;
- whether two programs have the same computation structure and, after key
  resolution, the same stochastic sharing relationships;
- how to freeze an early phase and resample later phases from it.

## v0.1 API

```python
from stoch_ir import (
    Bernoulli,
    Normal,
    Uniform,
    constant,
    reductions,
    sampling_phase,
    softplus,
)

with sampling_phase("latent"):
    weights = Normal(0.0, 1.0, rng_label="weights").add_plates("layer")

with sampling_phase("observation"):
    activations = Normal(
        mu=weights,
        sigma=softplus(weights) + 0.1,
        rng_label="activations",
    ).add_plates("batch", expect=("layer",))

layer_score = (
    activations
    .mean("batch")
    .check_plates("layer")
)
```

`add_plates(..., expect=...)` combines an exact precondition with an explicit stochastic transformation. The example states that `activations` already varies over `"layer"`, verifies that claim, and then introduces independent samples over `"batch"`.

v0.1 also includes arbitrary-bound continuous uniforms and Boolean Bernoulli
draws:

```python
with sampling_phase("probability"):
    probability = Uniform().add_plates("group")

with sampling_phase("trial"):
    trial = Bernoulli(probability).add_plates(
        "trial",
        expect=("group",),
    )

rate = trial.reduce_plates("trial", reduction=reductions.MEAN)
```

Existing NumPy values enter through one explicit boundary:

```python
import numpy as np

offset = constant(
    np.array([0.1, 0.2], dtype=np.float32),
    plates=("group",),
)
```

`constant` infers Boolean, integer, or floating metadata unless `dtype=` is
given. Storage is immediately coerced to `np.bool_`, `np.int64`, or
`np.float64`, respectively. Complex, object, string, and unnamed
multidimensional values are rejected.

The same graph can be materialized in phases:

```python
sizes = {"layer": 8, "batch": 32}

fixed_latent = layer_score.materialize(
    phases=("latent",),
    seed=100,
    plate_sizes=sizes,
)

sample_a = fixed_latent.realize(seed=200, plate_sizes=sizes)
sample_b = fixed_latent.realize(seed=201, plate_sizes=sizes)
```

`fixed_latent` is an opaque immutable checkpoint around a rewritten expression
graph. Sampled latent nodes have been replaced by constants; observation nodes
remain symbolic. The two calls reuse the embedded latent values while
intentionally drawing different observation values. Reusing the same seed
reproduces the same result.

## Core model

The v0.1 graph has six node families:

1. constants;
2. distributions;
3. unary operators;
4. binary operators;
5. plate introduction;
6. plate reduction.

The public plate operations are intentionally small:

- `add_plates(*new, expect=None)` introduces independent replication. If `expect` is supplied, the current plates must match it exactly before the new plates are added.
- `check_plates(*expected)` validates the exact current plate set and returns the unchanged expression.
- `reduce_plates(*plates, reduction=reductions.MEAN)` removes plates through a canonical immutable implementation object.
- `mean`, `sum`, `max`, `min`, `prod`, and `logsumexp` are convenience methods over `reduce_plates`.

Plate identifiers are strings. Their sizes are deliberately not stored in the symbolic graph; a `plate_sizes` mapping supplies concrete extents when a graph is materialized.

## Structural equality

Expression equality is intended for computation-structure verification, not
algebraic or probabilistic equivalence. It compares node kinds, deterministic
arguments, distribution kinds, and plate operations while deliberately ignoring
object aliasing and resolved randomness.

```python
x = Normal(0.0, 1.0)

shared = x + x
independent = (
    Normal(0.0, 1.0)
    + Normal(0.0, 1.0)
)

assert shared == independent

shared_checkpoint = shared.materialize(seed=1, phases=())
independent_checkpoint = independent.materialize(seed=1, phases=())

assert not shared_checkpoint.stochastically_equal(independent_checkpoint)
```

Graph-aware node entropy resolution occurs only as part of materialization. The
returned checkpoint therefore exposes `stochastically_equal`, which additionally
compares graph-derived entropy and any bound sampling seeds and catches the
shared-versus-independent distinction. A user-facing `rng_label` may supplement
that entropy, but never replaces the graph contribution or opts distinct nodes
into shared randomness. Raw expression nodes intentionally do not expose
stochastic comparison.

## Backend boundary

The graph engine owns symbolic structure, plate alignment, phase handling,
deterministic RNG entropy derivation, and immutable materialization. v0.1 performs
numeric propagation and distribution sampling with NumPy internally. Conversion
to other array libraries belongs at API boundaries; a pluggable backend protocol
is not part of the current implementation.

## Typing and validation

The package is marked with `py.typed` and keeps its annotations inline in the
runtime modules; no parallel `.pyi` API is shipped. CI type-checks a consumer
that imports only public names, builds the wheel, verifies that the marker (and
no stub files) ships, and smoke-tests the installed public API. The test suite
uses branch coverage with a 95% release floor.

## Project documents

- [v0.1 specification](specs/specification-v0.1.md)
- [v0.1 implementation plan](docs/implementation-plan-v0.1.md)
- [public API contract](docs/api-v0.1.md)
- [graph hashing and materialization architecture](docs/graph-hashing-and-materialization.md)
- The typed implementation lives in [`src/stoch_ir`](src/stoch_ir).

## v0.1 boundaries

The MVP does not include inference, autodiff, JIT compilation, graph optimization, serialization, visualization, arbitrary user-defined distributions or reductions, or backend-specific acceleration. Those features are intentionally excluded until the core authoring and verification model has been validated.
