# v0.1 Public API Contract

The `.pyi` files under `src/stochastic_programming_library` are the machine-readable API contract. This page summarizes the intended imports and behavior. None of these runtime objects is implemented yet.

## Top-level imports

```python
from stochastic_programming_library import (
    BackendError,
    Bernoulli,
    Constant,
    DistributionKind,
    DuplicatePlateError,
    DuplicateRNGNameError,
    Expr,
    GraphCycleError,
    GraphValidationError,
    MaterializationError,
    MissingPlateSizeError,
    Normal,
    NumPyBackend,
    PhaseError,
    PlateError,
    PlateExpectationError,
    Reduction,
    RNGKey,
    SampleRequest,
    SamplingBackend,
    StochasticProgrammingError,
    Uniform,
    UnknownPlateError,
    UnrealizedGraphError,
    current_sampling_phase,
    exp,
    log,
    sampling_phase,
    softplus,
)
```

## Construction

```python
Constant(value, *, plates=())
Normal(mu, sigma, *, rng_name=None)
Uniform(low, high, *, rng_name=None)
Bernoulli(p, *, rng_name=None)
```

Python scalar distribution parameters are coerced to constants. A distribution records the active `sampling_phase` at construction.

## Expression properties

```python
expr.plates
expr.plate_order
expr.pending_phases
expr.enabled_phases
expr.is_fully_realized
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

Comparison is exact and alias-preserving. It is not numerical closeness or algebraic equivalence.

## Staged execution

```python
partial = expr.materialize(
    phases=("latent",),
    seed=10,
    backend=None,
    plate_sizes={"row": 4},
)

value = partial.realize(
    seed=11,
    backend=None,
    plate_sizes={"row": 4},
)
```

Passing `backend=None` selects the package's eventual default `NumPyBackend`. Passing `phases=None` enables all remaining phases.

Extraction without new sampling is explicit:

```python
expr.assert_fully_realized()
value = expr.value()
```

Both calls fail when a reachable distribution node remains.

## Backend protocol

```python
class SamplingBackend(Protocol):
    def sample(
        self,
        request: SampleRequest,
        *,
        rng_key: RNGKey,
    ) -> object: ...
```

The request contains a distribution enum, aligned concrete parameters, and an unnamed output shape. Backend implementations do not traverse graphs or interpret plates and phases.
