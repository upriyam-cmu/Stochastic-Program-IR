"""Public exception hierarchy for Stochastic Program IR."""


class StochIRError(Exception):
    """Base class for all documented Stochastic Program IR errors."""


class StochIRWarning(UserWarning):
    """Base class for all documented Stochastic Program IR warnings."""


class GraphValidationError(StochIRError):
    """Raised when an expression graph violates a structural invariant."""


class GraphCycleError(GraphValidationError):
    """Raised when a cycle is found in an expression graph."""


class DependencyRewriteError(GraphValidationError):
    """Raised when an internal immutable dependency rewrite is invalid."""


class RngLabelError(GraphValidationError):
    """Raised when a distribution receives an invalid RNG label."""


class PlateError(StochIRError):
    """Base class for named-plate validation errors."""


class DuplicatePlateError(PlateError):
    """Raised when the same plate is introduced more than once."""


class PlateExpectationError(PlateError):
    """Raised when an expression does not have the expected plates."""


class UnknownPlateError(PlateError):
    """Raised when an operation names a plate that is not present."""


class MissingPlateSizeError(PlateError):
    """Raised when materialization needs a plate size that was not supplied."""


class PhaseError(StochIRError):
    """Raised when a sampling phase name or selection is invalid."""


class MaterializationError(StochIRError):
    """Base class for staged materialization and realization failures."""


class ValueValidationError(StochIRError):
    """Raised when a concrete value violates dtype, rank, or support rules."""


class InvalidSupportError(ValueValidationError):
    """Raised when an operation or distribution receives invalid support."""


class PossibleInvalidSupportWarning(StochIRWarning):
    """Warns when metadata cannot prove that a parameter is valid."""


class UnrealizedGraphError(MaterializationError):
    """Raised when concrete data is requested from an unrealized graph."""


class PlateSizeMismatchError(PlateError, MaterializationError):
    """Raised when materialization conflicts with a resolved plate size."""


class UnresolvedRandomnessError(MaterializationError):
    """Raised when internal RNG resolution has not completed."""


__all__ = [
    "DuplicatePlateError",
    "GraphCycleError",
    "GraphValidationError",
    "InvalidSupportError",
    "MaterializationError",
    "MissingPlateSizeError",
    "PhaseError",
    "PlateError",
    "PlateExpectationError",
    "PlateSizeMismatchError",
    "PossibleInvalidSupportWarning",
    "RngLabelError",
    "StochIRError",
    "StochIRWarning",
    "UnknownPlateError",
    "UnrealizedGraphError",
    "ValueValidationError",
]
