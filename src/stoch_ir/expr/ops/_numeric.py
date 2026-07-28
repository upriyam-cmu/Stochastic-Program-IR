"""Internal checked arithmetic helpers for canonical integer storage."""

from collections.abc import Callable

import numpy as np

from ...errors import ValueValidationError


def checked_int64_result(value: np.ndarray) -> np.ndarray:
    """Convert an exact integer result to int64 or reject overflow."""

    result = np.asarray(value, dtype=object)
    bounds = np.iinfo(np.int64)
    if bool(np.any(result < bounds.min) or np.any(result > bounds.max)):
        raise ValueValidationError("integer operation result is outside int64 range")
    return np.asarray(result, dtype=np.int64)


def checked_int64_binary(
    lhs: np.ndarray,
    rhs: np.ndarray,
    operation: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> np.ndarray:
    """Apply an integer binary operation without fixed-width wraparound."""

    return checked_int64_result(
        operation(lhs.astype(object), rhs.astype(object)),
    )


__all__ = ["checked_int64_binary", "checked_int64_result"]
