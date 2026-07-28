# v0.1 Public API Contract

The inline-typed runtime is the machine-readable contract. This page defines
the intentionally curated alpha surface.

## Top-level imports

```python
from stoch_ir import (
    Bernoulli,
    ConcreteValue,
    DataType,
    Normal,
    RandomVariable,
    SamplingCheckpoint,
    Uniform,
    ValueMeta,
    ValueSupport,
    constant,
    errors,
    exp,
    log,
    reductions,
    sampling_phase,
    softplus,
)

from stoch_ir import __version__
```

Concrete node classes, plate layouts, dependency-rewrite hooks, current phase
state, lowercase distribution aliases, and generic operation implementations
are internal.

## Construction and transforms

```python
constant(value, *, plates=(), dtype=None)
Normal(mu, sigma, *, rng_label=None)
Uniform(low=0.0, high=1.0, *, rng_label=None)
Bernoulli(p, *, rng_label=None)

exp(expr)
log(expr)
softplus(expr)

expr.exp()
expr.log()
expr.softplus()
expr.abs()
abs(expr)
```

Free and fluent forms construct structurally equal graphs. Arithmetic supports
`+`, `-`, `*`, `/`, and `//`, including their reverse forms.

`constant` accepts supported Python and NumPy Boolean, integer, and floating
values. Metadata is authoritative and storage is canonical:

| Metadata | NumPy storage |
| --- | --- |
| `DataType.BOOL` | `np.bool_` |
| `DataType.INT` | `np.int64` |
| `DataType.FLOAT` | `np.float64` |

Non-scalar rank must match the number of named plates. Complex, object, and
string values are rejected.

## Expression inspection

Every `RandomVariable` exposes:

```python
expr.dependencies     # immutable Mapping[str, RandomVariable]
expr.plates           # canonical tuple[str, ...]
expr.pending_phases   # frozenset[str]
expr.has_value        # bool
expr.value_meta       # ValueMeta
```

Dependencies are name-sorted and preserve object aliasing in their values.
Internal plate-layout and dependency-reconstruction objects are deliberately
not part of the public contract.

## Plates and reductions

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

`add_plates` introduces independent replication. If `expect` is supplied, the
existing plate set must match exactly before the new plates are added.
`check_plates` validates without changing the graph. Reductions explicitly
remove named plates.

The public immutable reduction objects are:

```python
reductions.MEAN
reductions.SUM
reductions.MAX
reductions.MIN
reductions.PROD
reductions.LOGSUMEXP
```

Their common opaque type is `reductions.Reduction`. Caller-defined reductions
are not a v0.1 extension point.

## Phases and materialization

```python
with sampling_phase("latent"):
    latent = Normal(0.0, 1.0)

partial = latent.materialize(
    phases=("latent",),
    seed=10,
    plate_sizes=None,
)

value = partial.realize(seed=11)
```

Phase names have no intrinsic order. Enabling a phase permanently clears that
barrier in the returned immutable graph, but sampling still waits for concrete
dependencies.

`RandomVariable.materialize` returns an opaque, non-composable
`SamplingCheckpoint`. A checkpoint exposes:

```python
checkpoint.pending_phases
checkpoint.is_fully_materialized
checkpoint.materialize(...)
checkpoint.value()
checkpoint.realize(...)
checkpoint.structurally_equal(other)
checkpoint.stochastically_equal(other)
```

`value()` succeeds only after the checkpoint root is concrete. `realize()`
enables every remaining phase and returns a `ConcreteValue`.

## Concrete values

`ConcreteValue` contains an immutable, read-only NumPy array and exposes:

```python
value.data
value.plates
value.shape
value.meta
value.dtype
value.support
```

The plate tuple is the canonical axis order of `data`.

## Equality

```python
expr == other
expr.structurally_equal(other)
checkpoint.stochastically_equal(other_checkpoint)
```

Structural equality compares the represented computation while ignoring object
aliasing and RNG resolution. It is not algebraic or probabilistic equality.
Stochastic equality is available only after materialization and additionally
checks graph-derived node entropy, sharing, fixed plate sizes, and bound
sampling seeds.

## Errors

Documented failures live under `stoch_ir.errors`. All public exceptions derive
from `errors.StochIRError`. The curated hierarchy covers graph validation,
plates, phases, materialization, concrete-value validation, and invalid
distribution support.
