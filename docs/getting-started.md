# Getting started

## Installation

Stochastic Program IR requires Python 3.10 or newer:

```console
pip install stoch-ir
```

The package is in alpha. Pin an exact prerelease version when reproducibility
matters.

## Build a graph

Distribution constructors return immutable symbolic expressions:

```python
from stoch_ir import Normal, sampling_phase, softplus

with sampling_phase("latent"):
    row_effect = Normal(0.0, 1.0, plates="row")

with sampling_phase("observation"):
    observation = Normal(
        mu=row_effect,
        sigma=softplus(row_effect) + 0.1,
        plates=("row", "replicate"),
    )

row_score = observation.mean("replicate").check_plates("row")
```

No sample is drawn while this graph is authored.

## Realize a value

```python
value = row_score.realize(
    seed=10,
    plate_sizes={"row": 4, "replicate": 32},
)

print(value.data)
print(value.plates)
```

The result is a {class}`stoch_ir.ConcreteValue` with immutable canonical NumPy
storage.

## Distributions

v0.1 exposes:

- {func}`stoch_ir.Normal` for univariate normal draws;
- {func}`stoch_ir.Uniform` for continuous draws over symbolic bounds; and
- {func}`stoch_ir.Bernoulli` for Boolean draws from symbolic probabilities.

Stochastic Program IR is a forward-sampling library. It does not implement
conditioning or posterior inference.
