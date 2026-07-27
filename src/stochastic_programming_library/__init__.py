"""Graph-first authoring and staged realization for stochastic programs."""

from .distributions import Gaussian, Normal, normal
from .expr import (
    ConcreteValue,
    Constant,
    DataType,
    PlateLayout,
    RandomVariable,
    SamplingCheckpoint,
    ValueMeta,
    ValueSupport,
)
from .phases import current_sampling_phase, sampling_phase
from .transforms import exp, log, softplus

__version__ = "0.1.0"

__all__ = [
    "ConcreteValue",
    "Constant",
    "DataType",
    "Gaussian",
    "Normal",
    "PlateLayout",
    "RandomVariable",
    "SamplingCheckpoint",
    "ValueMeta",
    "ValueSupport",
    "current_sampling_phase",
    "exp",
    "log",
    "normal",
    "sampling_phase",
    "softplus",
]
