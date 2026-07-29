# Sampling phases and partial materialization

Phases describe when distribution nodes may sample. They are string labels
without an intrinsic order; dependencies remain the source of execution order.

```python
from stoch_ir import Normal, sampling_phase

with sampling_phase("latent"):
    latent = Normal(0.0, 1.0)

with sampling_phase("observation"):
    observation = Normal(latent, 1.0)
```

## Freeze one phase

```python
fixed_latent = observation.materialize(
    phases=("latent",),
    seed=10,
)
```

Materialization immutably rewrites sampled distributions into constants and
returns a {class}`stoch_ir.SamplingCheckpoint`. The source expression remains
unchanged.

```python
first = fixed_latent.realize(seed=20)
second = fixed_latent.realize(seed=21)
```

Both branches reuse the embedded latent draw and independently sample the
remaining observation phase.

Enabling a phase clears its barrier in the returned checkpoint even when an
unresolved dependency temporarily prevents sampling. Its run seed is fixed at
that enabling step and is not silently replaced by a later call.
