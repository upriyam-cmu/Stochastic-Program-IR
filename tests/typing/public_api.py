"""Static consumer of inline annotations exposed by the public package."""

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

base: RandomVariable = constant(0.5)
with sampling_phase("typed"):
    probability: RandomVariable = Uniform(base, 1.0, rng_label="probability")
    trial: RandomVariable = Bernoulli(probability, rng_label="trial")
    observation: RandomVariable = Normal(trial, 1.0)

reduced: RandomVariable = observation.reduce_plates(reduction=reductions.MEAN)
checkpoint: SamplingCheckpoint = reduced.materialize(seed=1)
value: ConcreteValue = checkpoint.realize(seed=2)
plates: tuple[str, ...] = observation.plates
dependencies: dict[str, RandomVariable] = dict(observation.dependencies)
dtype: DataType = value.dtype
support: ValueSupport = value.support
meta: ValueMeta = value.meta
transformed: RandomVariable = exp(log(softplus(observation) + 1.0))
error_type: type[errors.StochIRError] = errors.StochIRError
reduction: reductions.Reduction = reductions.MEAN
