# Stochastic Programming Library

> Status: v0.1 API and design specification only. The package is not implemented yet.

Stochastic Programming Library is a graph-first Python API for authoring stochastic programs that are easy to read, compare, validate, and realize in stages.

The central idea is to preserve stochastic intent in the program itself. Distribution dependencies remain symbolic, independent replication is expressed with named plates, and sampling is grouped into phases that can be materialized independently. This makes important relationships visible to both human readers and code-generating agents instead of hiding them behind mutable RNG state and positional tensor axes.

The project is deliberately not a probabilistic inference framework. v0.1 is aimed at simulations, generators, and other forward-sampling programs where structural correctness and reproducibility matter.

## What the API is designed to make explicit

- which values are sampled and which expressions are deterministic;
- which samples are shared and which are independently replicated;
- which named plates are added, preserved, checked, or reduced;
- which phase of a program may be sampled at each materialization step;
- whether two programs have the same graph structure, including shared-node relationships;
- how to freeze an early phase and resample later phases from it.

## Proposed v0.1 API

```python
from stochastic_programming_library import (
    Normal,
    sampling_phase,
    softplus,
)

with sampling_phase("latent"):
    weights = Normal(0.0, 1.0, rng_name="weights").add_plates("layer")

with sampling_phase("observation"):
    activations = Normal(
        mu=weights,
        sigma=softplus(weights) + 0.1,
        rng_name="activations",
    ).add_plates("batch", expect=("layer",))

layer_score = (
    activations
    .mean("batch")
    .check_plates("layer")
)
```

`add_plates(..., expect=...)` combines an exact precondition with an explicit stochastic transformation. The example states that `activations` already varies over `"layer"`, verifies that claim, and then introduces independent samples over `"batch"`.

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

`fixed_latent` is another immutable expression graph. Sampled latent nodes have been replaced by constants; observation nodes remain symbolic. The two calls reuse the embedded latent values while intentionally drawing different observation values. Reusing the same seed reproduces the same result.

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
- `reduce_plates(*plates, reduction=...)` removes plates through a named aggregation.
- `mean`, `sum`, `max`, `min`, `prod`, and `logsumexp` are convenience methods over `reduce_plates`.

Plate identifiers are strings. Their sizes are deliberately not stored in the symbolic graph; a `plate_sizes` mapping supplies concrete extents when a graph is materialized.

## Structural equality

Expression equality is intended for exact structural verification, not algebraic equivalence. It compares node kinds, arguments, plates, phases, RNG names, and graph topology. In particular, it distinguishes a shared sample used twice from two separate but textually identical samples.

```python
x = Normal(0.0, 1.0, rng_name="x")

shared = x + x
independent = (
    Normal(0.0, 1.0, rng_name="left")
    + Normal(0.0, 1.0, rng_name="right")
)

assert shared != independent
```

This exactness is intentional: the library is designed to test whether a hand-written or agent-generated stochastic program represents the expected dependency graph.

## Backend boundary

The graph engine owns symbolic structure, plate alignment, phase handling, deterministic RNG-key derivation, and immutable materialization. A backend only receives a concrete distribution request and an opaque RNG key, and returns a value. The first implementation target is NumPy; JAX and PyTorch are outside the v0.1 deliverable.

## Project documents

- [v0.1 specification](specs/specification-v0.1.md)
- [v0.1 implementation plan](docs/implementation-plan-v0.1.md)
- [public API contract](docs/api-v0.1.md)
- Type-level API stubs live in [`src/stochastic_programming_library`](src/stochastic_programming_library).

## v0.1 boundaries

The MVP does not include inference, autodiff, JIT compilation, graph optimization, serialization, visualization, arbitrary user-defined distributions or reductions, or backend-specific acceleration. Those features are intentionally excluded until the core authoring and verification model has been validated.
