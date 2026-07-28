"""Stochastic Program IR: graph-first stochastic authoring and realization."""

from importlib.metadata import version as _distribution_version

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

__version__ = _distribution_version("stoch-ir")

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
    "bernoulli",
    "constant",
    "current_sampling_phase",
    "exp",
    "log",
    "normal",
    "reductions",
    "sampling_phase",
    "softplus",
    "uniform",
]
