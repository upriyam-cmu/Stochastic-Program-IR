import operator
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from enum import Enum, IntEnum
from functools import cached_property, reduce
from types import MappingProxyType
from typing import Final, TypeAlias

import numpy as np

from ..errors import BackendError, DuplicatePlateError, UnknownPlateError

Plate: TypeAlias = str
Phase: TypeAlias = str | None
Scalar: TypeAlias = bool | int | float
Data: TypeAlias = Scalar | np.ndarray
PlateSizes: TypeAlias = Mapping[Plate, int]


class ValueSupport(Enum):
    POSITIVE_BRANCH = "[0, +inf)"
    NEGATIVE_BRANCH = "(-inf, 0]"
    UNIT_INTERVAL = "[0, 1]"
    REAL = "(-inf, +inf)"


class DataType(IntEnum):
    # we can almost always resolve implicit conversions as
    # max(dtypes), excluding an unfortunate edge case where
    # sometimes we want to merge multiple bools into an int
    BOOL = 0
    INT = 1
    FLOAT = 2


@dataclass(frozen=True, slots=True)
class ValueMeta:
    dtype: DataType
    support: ValueSupport

    @classmethod
    def from_value(cls, value: Data, dtype: DataType) -> "ValueMeta":
        if dtype is DataType.BOOL:
            all_bools = np.all(0 <= value) and np.all(value <= 1)
            if not all_bools:
                raise BackendError("Got BOOL dtype object with non-bool values?")
            return ValueMeta(dtype=DataType.BOOL, support=ValueSupport.UNIT_INTERVAL)

        if np.all(0 <= value):
            if np.all(value <= 1):
                support = ValueSupport.UNIT_INTERVAL
            else:
                support = ValueSupport.POSITIVE_BRANCH
        else:
            if np.all(value <= 0):
                support = ValueSupport.NEGATIVE_BRANCH
            else:
                support = ValueSupport.REAL
        return ValueMeta(dtype=dtype, support=support)


@dataclass(frozen=True)
class PlateLayout:
    # `plates` is always stored in canonical lexicographic order.
    plates: tuple[Plate, ...]

    @staticmethod
    def wrap(plates: Iterable[Plate]) -> "PlateLayout":
        return PlateLayout(tuple(plates))

    @staticmethod
    def check_unique_plates(*plates: Plate) -> tuple[Plate, ...]:
        """Makes sure all plates are unique."""
        unique_plates = set(plates)
        if len(unique_plates) != len(plates):
            raise DuplicatePlateError(f"{plates = } contains duplicates")
        return plates

    def __post_init__(self) -> None:
        if any(not isinstance(plate, str) or not plate for plate in self.plates):
            raise ValueError(
                f"plate names must be non-empty strings, got {self.plates}"
            )
        canonical_plates = PlateLayout.check_unique_plates(*sorted(self.plates))
        object.__setattr__(self, "plates", canonical_plates)

    @cached_property
    def as_set(self) -> frozenset[Plate]:
        return frozenset(self.plates)

    def __len__(self) -> int:
        return len(self.plates)

    def __iter__(self) -> Iterator[Plate]:
        return iter(self.plates)

    @cached_property
    def _plate2axis(self) -> Mapping[Plate, int]:
        return MappingProxyType({plate: idx for idx, plate in enumerate(self.plates)})

    def axis(self, plate: Plate) -> int:
        if plate not in self._plate2axis:
            raise UnknownPlateError(f"unknown plate '{plate}', not in {self.plates}")
        return self._plate2axis[plate]

    def _remove_plates(self, removed: set[Plate] | frozenset[Plate]) -> "PlateLayout":
        extra_plates = removed - self.as_set
        if extra_plates:
            raise UnknownPlateError(
                f"unknown plates {extra_plates}, not in {self.plates}"
            )
        return PlateLayout.wrap(plate for plate in self.plates if plate not in removed)

    def __sub__(self, other: object) -> "PlateLayout":
        # removed plates must all exist
        if not isinstance(other, PlateLayout):
            return NotImplemented
        return self._remove_plates(other.as_set)

    def __rsub__(self, other: object) -> "PlateLayout":
        # removed plates must all exist
        if not isinstance(other, PlateLayout):
            return NotImplemented
        return other._remove_plates(self.as_set)

    def without(self, *plates: Plate) -> "PlateLayout":
        removed_plates = set(PlateLayout.check_unique_plates(*plates))
        return self._remove_plates(removed_plates)

    def __or__(self, other: object) -> "PlateLayout":
        # plates can have overlap
        if not isinstance(other, PlateLayout):
            return NotImplemented
        return PlateLayout.wrap(self.as_set | other.as_set)

    def __ror__(self, other: object) -> "PlateLayout":
        # plates can have overlap
        if not isinstance(other, PlateLayout):
            return NotImplemented
        return PlateLayout.wrap(other.as_set | self.as_set)

    @classmethod
    def union(cls, *layouts: "PlateLayout") -> "PlateLayout":
        return PlateLayout.wrap(
            reduce(
                operator.or_,
                (layout.as_set for layout in layouts),
                frozenset(),
            )
        )

    def __add__(self, other: object) -> "PlateLayout":
        # must be unique plates
        if not isinstance(other, PlateLayout):
            return NotImplemented
        return PlateLayout.wrap(self.plates + other.plates)

    def __radd__(self, other: object) -> "PlateLayout":
        # must be unique plates
        if not isinstance(other, PlateLayout):
            return NotImplemented
        return PlateLayout.wrap(other.plates + self.plates)


EMPTY_PLATE_LAYOUT: Final[PlateLayout] = PlateLayout.wrap(())


@dataclass(frozen=True, slots=True, eq=False)
class ConcreteValue:
    data: np.ndarray
    layout: PlateLayout
    meta: ValueMeta

    @staticmethod
    def wrap(data: Data, layout: PlateLayout, meta: ValueMeta) -> "ConcreteValue":
        return ConcreteValue(np.asarray(data), layout, meta)

    def __post_init__(self) -> None:
        owned = np.array(self.data, copy=True)
        owned.flags.writeable = False
        object.__setattr__(self, "data", owned)
        if self.data.ndim != len(self.layout.plates):
            raise BackendError(
                f"data.ndim = {self.data.ndim} != len(plates) = {len(self.layout.plates)}"
            )

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, ConcreteValue)
            and self.shape == other.shape
            and np.array_equal(self.data, other.data, equal_nan=True)
            and self.layout == other.layout
            and self.meta == other.meta
        )

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape
