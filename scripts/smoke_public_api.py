"""Import the supported API from an installed package."""

import stoch_ir
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

with sampling_phase("smoke"):
    probability = Uniform()
    trial = Bernoulli(probability)

assert Normal(0.0, 1.0).value_meta.dtype.name == "FLOAT"
assert constant(1).value_meta.dtype.name == "INT"
assert trial.reduce_plates(reduction=reductions.MEAN) is trial
assert isinstance(exp(log(softplus(Normal(0.0, 1.0)) + 1.0)), RandomVariable)
assert issubclass(errors.PhaseError, errors.StochIRError)
assert stoch_ir.__all__

# Ensure every curated runtime type is present in the installed wheel.
_ = (
    ConcreteValue,
    DataType,
    SamplingCheckpoint,
    ValueMeta,
    ValueSupport,
)
