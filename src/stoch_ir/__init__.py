"""Stochastic Program IR: graph-first stochastic authoring and realization."""

from importlib.metadata import version as _distribution_version

from . import errors, reductions
from .distributions import Bernoulli, Normal, Uniform
from .expr import (
    ConcreteValue,
    DataType,
    RandomVariable,
    SamplingCheckpoint,
    ValueMeta,
    ValueSupport,
    constant,
)
from .phases import sampling_phase
from .transforms import exp, log, softplus

__version__ = _distribution_version("stoch-ir")

__all__ = [
    "Bernoulli",
    "ConcreteValue",
    "DataType",
    "Normal",
    "RandomVariable",
    "SamplingCheckpoint",
    "Uniform",
    "ValueMeta",
    "ValueSupport",
    "constant",
    "errors",
    "exp",
    "log",
    "reductions",
    "sampling_phase",
    "softplus",
]
