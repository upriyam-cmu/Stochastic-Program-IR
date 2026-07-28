# v0.1 Public API Contract

The typed runtime under `src/stochastic_programming_library` is the
machine-readable API contract. This page summarizes the currently implemented
surface.

## Top-level imports

```python
from stochastic_programming_library import (
    Bernoulli,
    BernoulliDistribution,
    ConcreteValue,
    Constant,
    DataType,
    Gaussian,
    Normal,
    PlateLayout,
    RandomVariable,
    SamplingCheckpoint,
    Uniform,
    UniformDistribution,
    ValueMeta,
    ValueSupport,
    bernoulli,
    constant,
    current_sampling_phase,
    exp,
    log,
    normal,
    reductions,
    sampling_phase,
    softplus,
    uniform,
)
```

The package is inline-typed and ships `py.typed`. No public symbol exists only
in a `.pyi` file.

## Construction

```python
constant(value, *, plates=(), dtype=None)
Constant.of(value, dtype)
Constant.array(array, dtype, plate_layout)
Normal(mu, sigma, *, rng_label=None)
Uniform(low=0.0, high=1.0, *, rng_label=None)
Bernoulli(p, *, rng_label=None)
```

Python scalar distribution parameters are coerced to constants. A distribution records the active `sampling_phase` at construction.
`Constant.of` and `Constant.array` are retained convenience constructors and
delegate to the same concrete boundary as `constant`.

`ValueMeta.dtype` is authoritative at that boundary:

| Metadata | NumPy storage |
| --- | --- |
| `DataType.BOOL` | `np.bool_` |
| `DataType.INT` | `np.int64` |
| `DataType.FLOAT` | `np.float64` |

Supported scalar and NumPy Boolean, integer, and floating kinds are inferred
when `dtype` is omitted. Complex, object, and string values are rejected.
Non-scalar rank must match the number of named plates. Narrow support metadata
is checked against the coerced value.

## Expression properties

```python
expr.plates
expr.plate_layout
expr.pending_phases
expr.has_value
expr.dependencies
```

## Expression transforms

```python
expr + other
expr - other
expr * other
expr / other
expr // other

exp(expr)
log(expr)
softplus(expr)
expr.abs()
```

## Plate methods

```python
expr.add_plates(*new, expect=None)
expr.check_plates(*expected)
expr.reduce_plates(*plates, reduction=reductions.MEAN)

expr.mean(*plates)
expr.sum(*plates)
expr.max(*plates)
expr.min(*plates)
expr.prod(*plates)
expr.logsumexp(*plates)
```

The canonical objects are `reductions.MEAN`, `SUM`, `MAX`, `MIN`, `PROD`, and
`LOGSUMEXP`. They are immutable singleton implementations; implementation
classes and caller-defined reductions are not public v0.1 extension points.

## Structural comparison

```python
expr == other
expr.structurally_equal(other)
```

Comparison checks computation structure while ignoring object aliasing and RNG
resolution. It is not numerical closeness, algebraic equivalence, or
probabilistic equality.

Nodes also implement an exact, node-owned dependency reconstruction contract:

```python
expr.rewrite_dependencies({"dependency_name": rewritten_expr, ...})
```

The mapping must contain every dependency exactly once. This method supports
internal immutable graph rewrites; general custom-node support remains outside
the v0.1 public extension contract.

## Staged execution

```python
partial = expr.materialize(
    phases=("latent",),
    seed=10,
    plate_sizes={"row": 4},
)

value = partial.realize(
    seed=11,
    plate_sizes={"row": 4},
)

completed = partial.materialize(seed=11)
value = completed.value()
```

`materialize` returns an opaque, non-composable `SamplingCheckpoint`.
`phases=None` enables every remaining named phase. Unphased distributions have
no barrier and execute as soon as their dependencies are concrete.

Only checkpoints expose `value()` and stochastic comparison because graph-aware
node entropy is resolved as part of materialization:

```python
partial.stochastically_equal(other_partial)
```

This comparison first requires structural equality and then compares relevant
plate sizes, remaining phase requirements, node entropy, and bound sampling
seeds.

When an enabled phase is blocked by another stochastic dependency, its sampling
seed is still fixed during that materialization call. Clearing the dependency
later does not silently replace it with the later call's seed.

## Numeric execution

v0.1 stores concrete values as NumPy arrays and samples through
`numpy.random.Generator`. There is no public backend protocol in the current
surface.
