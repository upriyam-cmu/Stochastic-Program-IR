from dataclasses import dataclass
from enum import Enum
from typing import Mapping, TypeAlias

import numpy as np

Plate: TypeAlias = str
Phase: TypeAlias = str | None
Scalar: TypeAlias = bool | int | float
Data: TypeAlias = Scalar | np.generic | np.ndarray
PlateSizes: TypeAlias = Mapping[Plate, int]

class ValueSupport(str, Enum):
    # Conservative guarantees rather than an exhaustive interval algebra.
    # POSITIVE and NEGATIVE are strict; UNIT_INTERVAL includes both endpoints.
    POSITIVE: str
    NEGATIVE: str
    UNIT_INTERVAL: str
    REAL: str

@dataclass(frozen=True, slots=True)
class ValueMeta:
    dtype: np.dtype
    support: ValueSupport

    @classmethod
    def from_value(cls, value: Data) -> ValueMeta: ...

@dataclass(frozen=True, slots=True)
class PlateLayout:
    # `plates` is always stored in canonical lexicographic order.
    plates: tuple[Plate, ...]

    @classmethod
    def canonical(cls, *plates: Plate) -> PlateLayout: ...
    @classmethod
    def union(cls, *layouts: PlateLayout) -> PlateLayout: ...
    @property
    def as_set(self) -> frozenset[Plate]: ...
    def axis(self, plate: Plate) -> int: ...
    def without(self, *plates: Plate) -> PlateLayout: ...

@dataclass(frozen=True, slots=True)
class ConcreteValue:
    data: np.ndarray
    layout: PlateLayout
    meta: ValueMeta

    @property
    def shape(self) -> tuple[int, ...]: ...
