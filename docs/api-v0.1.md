# v0.1 Public API Contract

The typed runtime under `src/stochastic_programming_library` is the
machine-readable API contract. This page summarizes the currently implemented
surface.

## Top-level imports

```python
from stochastic_programming_library import (
    ConcreteValue,
    Constant,
    DataType,
    Gaussian,
    Normal,
    PlateLayout,
    RandomVariable,
    SamplingCheckpoint,
    ValueMeta,
    ValueSupport,
    current_sampling_phase,
    exp,
    log,
    normal,
    sampling_phase,
    softplus,
)
```

## Construction

```python
Constant.of(value, dtype)
Constant.array(array, dtype, plate_layout)
Normal(mu, sigma, *, rng_key=None)
```

Python scalar distribution parameters are coerced to constants. A distribution records the active `sampling_phase` at construction.

## Expression properties

```python
expr.plates
expr.plate_layout
expr.pending_phases
expr.has_value
```

## Expression transforms

```python
expr + other
expr - other
expr * other
expr / other

exp(expr)
log(expr)
softplus(expr)
```

## Plate methods

```python
expr.add_plates(*new, expect=None)
expr.check_plates(*expected)
expr.reduce_plates(*plates, reduction=Reduction.MEAN)

expr.mean(*plates)
expr.sum(*plates)
expr.max(*plates)
expr.min(*plates)
expr.prod(*plates)
expr.logsumexp(*plates)
```

## Structural comparison

```python
expr == other
expr.structurally_equal(other)
```

Comparison checks computation structure while ignoring object aliasing and RNG
resolution. It is not numerical closeness, algebraic equivalence, or
probabilistic equality.

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
```

`materialize` returns an opaque, non-composable `SamplingCheckpoint`.
`phases=None` enables every remaining named phase. Unphased distributions have
no barrier and execute as soon as their dependencies are concrete.

Only checkpoints expose stochastic comparison because RNG keys are resolved as
part of materialization:

```python
partial.stochastically_equal(other_partial)
```

This comparison first requires structural equality and then compares relevant
plate sizes, remaining phase requirements, and resolved RNG keys.

## Numeric execution

v0.1 stores concrete values as NumPy arrays and samples through
`numpy.random.Generator`. There is no public backend protocol in the current
surface.
