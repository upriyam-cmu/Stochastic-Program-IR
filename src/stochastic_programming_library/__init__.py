"""Graph-first authoring and staged realization for stochastic programs."""

from . import reductions
from .distributions import (
    Bernoulli,
    BernoulliDistribution,
    Gaussian,
    Normal,
    Uniform,
    UniformDistribution,
    bernoulli,
    normal,
    uniform,
)
from .expr import (
    ConcreteValue,
    Constant,
    DataType,
    PlateLayout,
    RandomVariable,
    SamplingCheckpoint,
    ValueMeta,
    ValueSupport,
    constant,
)
from .phases import current_sampling_phase, sampling_phase
from .transforms import exp, log, softplus

__version__ = "0.1.0"

__all__ = [
    "Bernoulli",
    "BernoulliDistribution",
    "ConcreteValue",
    "Constant",
    "DataType",
    "Gaussian",
    "Normal",
    "PlateLayout",
    "RandomVariable",
    "SamplingCheckpoint",
    "Uniform",
    "UniformDistribution",
    "ValueMeta",
    "ValueSupport",
    "current_sampling_phase",
    "bernoulli",
    "constant",
    "exp",
    "log",
    "normal",
    "reductions",
    "sampling_phase",
    "softplus",
    "uniform",
]
