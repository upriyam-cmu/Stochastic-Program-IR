from enum import Enum
from functools import cached_property as cached_property_bad_typing
from typing import Any, Literal, TypeAlias, cast, overload

import numpy as np
from typing_extensions import TypeIs

from ..errors import DuplicatePlateError

Plate: TypeAlias = str
Phase: TypeAlias = str | None
Scalar: TypeAlias = bool | int | float
Data: TypeAlias = np.ndarray | np.floating | np.integer


def is_scalar(obj: Any) -> TypeIs[Scalar]:
    return isinstance(obj, (bool, int, float))


def normalize_plates(*plates: Plate) -> tuple[Plate, ...]:
    unique_plates = set(plates)
    if len(unique_plates) != len(plates):
        raise DuplicatePlateError(f"{plates = } contains duplicates")
    return plates


class Reduction(Enum):
    MEAN = "mean"
    SUM = "sum"
    MAX = "max"
    MIN = "min"
    PROD = "prod"
    LOGSUMEXP = "logsumexp"
