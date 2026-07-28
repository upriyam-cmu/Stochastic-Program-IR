"""Static consumer of inline annotations exposed by the public package."""

from stochastic_programming_library import (
    Bernoulli,
    ConcreteValue,
    Normal,
    RandomVariable,
    SamplingCheckpoint,
    Uniform,
    constant,
    reductions,
    sampling_phase,
)

base: RandomVariable = constant(0.5)
with sampling_phase("typed"):
    probability: RandomVariable = Uniform(base, 1.0, rng_label="probability")
    trial: RandomVariable = Bernoulli(probability, rng_label="trial")
    observation: RandomVariable = Normal(trial, 1.0)

reduced: RandomVariable = observation.reduce_plates(reduction=reductions.MEAN)
checkpoint: SamplingCheckpoint = reduced.materialize(seed=1)
value: ConcreteValue = checkpoint.realize(seed=2)
