"""Import the supported API from an installed package."""

from stochastic_programming_library import (
    Bernoulli,
    Normal,
    Uniform,
    constant,
    reductions,
    sampling_phase,
)

with sampling_phase("smoke"):
    probability = Uniform()
    trial = Bernoulli(probability)

assert Normal(0.0, 1.0).value_meta.dtype.name == "FLOAT"
assert constant(1).value_meta.dtype.name == "INT"
assert trial.reduce_plates(reduction=reductions.MEAN) is trial
