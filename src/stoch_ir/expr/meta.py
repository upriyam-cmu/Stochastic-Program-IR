import operator
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from enum import Enum, IntEnum
from functools import cached_property, reduce
from types import MappingProxyType
from typing import Any, Final, TypeAlias

import numpy as np

from ..errors import DuplicatePlateError, UnknownPlateError, ValueValidationError

Plate: TypeAlias = str
Phase: TypeAlias = str | None
Scalar: TypeAlias = bool | int | float
Data: TypeAlias = Scalar | np.generic | np.ndarray
PlateSizes: TypeAlias = Mapping[Plate, int]


class ValueSupport(Enum):
    """Conservative support metadata propagated through expressions."""

    POSITIVE_BRANCH = "[0, +inf)"
    NEGATIVE_BRANCH = "(-inf, 0]"
    UNIT_INTERVAL = "[0, 1]"
    REAL = "(-inf, +inf)"

    def contains(self, value: np.ndarray) -> bool:
        """Return whether every value satisfies this support."""

        match self:
            case ValueSupport.POSITIVE_BRANCH:
                return bool(np.all(0 <= value))
            case ValueSupport.NEGATIVE_BRANCH:
                return bool(np.all(value <= 0))
            case ValueSupport.UNIT_INTERVAL:
                return bool(np.all(0 <= value) and np.all(value <= 1))
            case ValueSupport.REAL:
                return True

        raise AssertionError(f"unreachable support {self!r}")


class DataType(IntEnum):
    """Canonical concrete dtype families supported by v0.1."""

    # we can almost always resolve implicit conversions as
    # max(dtypes), excluding an unfortunate edge case where
    # sometimes we want to merge multiple bools into an int
    BOOL = 0
    INT = 1
    FLOAT = 2

    @property
    def numpy_dtype(self) -> np.dtype[Any]:
        """Return the canonical NumPy dtype for this metadata member."""

        match self:
            case DataType.BOOL:
                return np.dtype(np.bool_)
            case DataType.INT:
                return np.dtype(np.int64)
            case DataType.FLOAT:
                return np.dtype(np.float64)

        raise AssertionError(f"unreachable dtype {self!r}")

    @classmethod
    def infer(cls, value: Data) -> "DataType":
        """Infer a supported dtype family from concrete data."""

        if isinstance(value, (int, np.integer)) and not isinstance(
            value, (bool, np.bool_)
        ):
            return cls.INT
        kind = np.asarray(value).dtype.kind
        if kind == "b":
            return cls.BOOL
        if kind in ("i", "u"):
            return cls.INT
        if kind == "f":
            return cls.FLOAT
        raise ValueValidationError(
            f"unsupported concrete dtype {np.asarray(value).dtype}; "
            "expected boolean, integer, or floating data"
        )

    def coerce(self, value: Data) -> np.ndarray:
        """Coerce concrete data into this dtype's canonical NumPy storage."""

        if self is DataType.INT and isinstance(value, (int, np.integer)):
            integer = int(value)
            bounds = np.iinfo(np.int64)
            if not bounds.min <= integer <= bounds.max:
                raise ValueValidationError(
                    f"integer value {integer} is outside canonical int64 range"
                )
            return np.asarray(integer, dtype=self.numpy_dtype)

        source = np.asarray(value)
        if source.dtype.kind not in ("b", "i", "u", "f"):
            raise ValueValidationError(
                f"cannot coerce unsupported dtype {source.dtype} to {self.name}"
            )
        if self is DataType.BOOL and not bool(np.all((source == 0) | (source == 1))):
            raise ValueValidationError("BOOL values must contain only 0 or 1")
        if self is DataType.INT:
            bounds = np.iinfo(np.int64)
            if source.dtype.kind in ("i", "u") and (
                np.any(source < bounds.min) or np.any(source > bounds.max)
            ):
                raise ValueValidationError(
                    "integer data contains values outside canonical int64 range"
                )
            if source.dtype.kind == "f" and (
                np.any(~np.isfinite(source))
                or np.any(source != np.trunc(source))
                or np.any(source < bounds.min)
                or np.any(source >= bounds.max + 1)
            ):
                raise ValueValidationError(
                    "floating data cannot be losslessly represented as canonical int64"
                )
        return np.asarray(source, dtype=self.numpy_dtype)


@dataclass(frozen=True, slots=True)
class ValueMeta:
    """Canonical dtype and conservative support for an expression value."""

    dtype: DataType
    support: ValueSupport

    @classmethod
    def from_value(cls, value: Data, dtype: DataType) -> "ValueMeta":
        """Derive conservative metadata after canonical dtype coercion."""

        coerced = dtype.coerce(value)
        if dtype is DataType.BOOL:
            return ValueMeta(dtype=dtype, support=ValueSupport.UNIT_INTERVAL)

        if np.all(0 <= coerced):
            if np.all(coerced <= 1):
                support = ValueSupport.UNIT_INTERVAL
            else:
                support = ValueSupport.POSITIVE_BRANCH
        else:
            if np.all(coerced <= 0):
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
        return PlateLayout((plates,) if isinstance(plates, str) else tuple(plates))

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
    """Immutable NumPy value returned by realization.

    Concrete storage is read-only and uses the canonical NumPy dtype declared
    by :attr:`meta`. Named plates follow the same lexicographic order as the
    corresponding array axes.

    Notes
    -----
    Construct concrete values through :func:`stoch_ir.constant` or expression
    realization rather than instantiating this class directly.
    """

    data: np.ndarray
    layout: PlateLayout
    meta: ValueMeta

    @staticmethod
    def wrap(data: Data, layout: PlateLayout, meta: ValueMeta) -> "ConcreteValue":
        return ConcreteValue(np.asarray(data), layout, meta)

    def __post_init__(self) -> None:
        owned = np.array(self.meta.dtype.coerce(self.data), copy=True)
        owned.flags.writeable = False
        object.__setattr__(self, "data", owned)
        if self.data.ndim != len(self.layout.plates):
            raise ValueValidationError(
                f"data.ndim = {self.data.ndim} != len(plates) = {len(self.layout.plates)}"
            )
        if (
            self.meta.support is not ValueSupport.REAL
            and not self.meta.support.contains(self.data)
        ):
            raise ValueValidationError(
                f"concrete value does not satisfy declared support "
                f"{self.meta.support.value}"
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
        """Return the concrete NumPy shape."""

        return self.data.shape

    @property
    def plates(self) -> tuple[Plate, ...]:
        """Return plate names in the canonical axis order."""

        return self.layout.plates

    @property
    def dtype(self) -> DataType:
        """Return the canonical dtype metadata."""

        return self.meta.dtype

    @property
    def support(self) -> ValueSupport:
        """Return the conservative support metadata."""

        return self.meta.support
