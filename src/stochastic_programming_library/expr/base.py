from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from functools import cached_property
from typing import TypeAlias

from typing_extensions import override

from ..errors import MissingPlateSizeError, PlateExpectationError
from ..rng import Seed
from .dtype import Bool, DataType, Float, Int
from .util import Data, Phase, Plate, Reduction, Scalar, is_scalar, normalize_plates


class RandomVariable(ABC):
    @abstractmethod
    def _resolve_plates(self) -> frozenset[Plate]: ...

    @abstractmethod
    def _resolve_pending_phases(self) -> frozenset[Phase]: ...

    @abstractmethod
    def _resolve_has_value(self) -> bool: ...

    @abstractmethod
    def _resolve_dtype(self) -> DataType: ...

    @abstractmethod
    def value(
        self,
        plate_sizes: Mapping[Plate, int] | None = None,
    ) -> Data: ...  # TODO type this function properly

    @cached_property
    def plates(self) -> frozenset[Plate]:
        return self._resolve_plates()

    @cached_property
    def plates_ordered(self) -> tuple[Plate, ...]:
        return tuple(sorted(self.plates))

    @cached_property
    def pending_phases(self) -> frozenset[Phase]:
        return self._resolve_pending_phases()

    @cached_property
    def has_value(self) -> bool:
        return self._resolve_has_value()

    @cached_property
    def dtype(self) -> DataType:
        return self._resolve_dtype()

    def add_plates(
        self,
        *plates: Plate,
        expect: Iterable[Plate] | None = None,
    ) -> "RandomVariable":
        if expect is not None:
            self.check_plates(*expect)
        raise NotImplementedError("TODO make add_plates expr node")

    def check_plates(self, *plates: Plate) -> "RandomVariable":
        checked_plates = frozenset(normalize_plates(*plates))
        if checked_plates != self.plates:
            raise PlateExpectationError(f"expected {checked_plates}, got {self.plates}")
        return self

    def reduce_plates(
        self,
        *plates: Plate,
        reduction: Reduction,
    ) -> "RandomVariable":
        raise NotImplementedError(
            "TODO make reduce_plates expr node"
        )  # reduction node owns plate verification

    def mean(self, *plates: Plate) -> "RandomVariable":
        return self.reduce_plates(*plates, reduction=Reduction.MEAN)

    def sum(self, *plates: Plate) -> "RandomVariable":
        return self.reduce_plates(*plates, reduction=Reduction.SUM)

    def max(self, *plates: Plate) -> "RandomVariable":
        return self.reduce_plates(*plates, reduction=Reduction.MAX)

    def min(self, *plates: Plate) -> "RandomVariable":
        return self.reduce_plates(*plates, reduction=Reduction.MIN)

    def prod(self, *plates: Plate) -> "RandomVariable":
        return self.reduce_plates(*plates, reduction=Reduction.PROD)

    def logsumexp(self, *plates: Plate) -> "RandomVariable":
        return self.reduce_plates(*plates, reduction=Reduction.LOGSUMEXP)

    @abstractmethod
    def materialize(
        self,
        *,
        seed: Seed | None = None,
        plate_sizes: Mapping[Plate, int] | None = None,
        phases: Iterable[Phase] | None = None,
    ) -> "RandomVariable": ...

    @abstractmethod
    def structurally_equal(self, other: "RandomVariable") -> bool: ...

    @abstractmethod
    def stochastically_equal(self, other: "RandomVariable") -> bool: ...

    def __eq__(self, other: object) -> bool:
        return isinstance(other, RandomVariable) and self.structurally_equal(other)

    def __add__(self, other: "ExprInput") -> "RandomVariable":
        from .bin_op import AddOp, BinOpNode

        return BinOpNode(self, _fix_expr(other), AddOp())

    def __radd__(self, other: "ExprInput") -> "RandomVariable":
        from .bin_op import AddOp, BinOpNode

        return BinOpNode(_fix_expr(other), self, AddOp())

    def __sub__(self, other: "ExprInput") -> "RandomVariable": ...
    def __rsub__(self, other: "ExprInput") -> "RandomVariable": ...
    def __mul__(self, other: "ExprInput") -> "RandomVariable": ...
    def __rmul__(self, other: "ExprInput") -> "RandomVariable": ...
    def __truediv__(self, other: "ExprInput") -> "RandomVariable": ...
    def __rtruediv__(self, other: "ExprInput") -> "RandomVariable": ...


class Constant(RandomVariable):
    _value: Data
    _dtype: DataType
    _plates: tuple[Plate, ...]

    def __init__(self, value: Data, dtype: DataType, *plates: Plate) -> None:
        self._value = value
        self._dtype = dtype
        self._plates = normalize_plates(*plates)

    @override
    def _resolve_plates(self) -> frozenset[Plate]:
        return frozenset(self._plates)

    @override
    def _resolve_pending_phases(self) -> frozenset[Phase]:
        return frozenset()

    @override
    def _resolve_has_value(self) -> bool:
        return True

    @override
    def _resolve_dtype(self) -> DataType:
        return self._dtype

    @override
    def value(self, plate_sizes: Mapping[Plate, int] | None = None) -> Data:
        raise NotImplementedError()
        if self._plates:
            if plate_sizes is None:
                raise MissingPlateSizeError(f"plates: {self.plates}")
            output_shape = []
            missing_plates = set()
            for plate in self._plates:
                size = plate_sizes.get(plate)
                if size is not None:
                    output_shape.append(size)
                else:
                    missing_plates.add(plate)
            if missing_plates:
                raise MissingPlateSizeError(f"plates: {missing_plates}")
        else:
            output_shape = ()
        return self._value

    @override
    def materialize(
        self,
        *,
        seed: Seed | None = None,
        plate_sizes: Mapping[Plate, int] | None = None,
        phases: Iterable[Phase] | None = None,
    ) -> "RandomVariable":
        return self

    @override
    def structurally_equal(self, other: "RandomVariable") -> bool:
        return (
            isinstance(other, Constant)
            and self._value == other._value
            and self._dtype == other._dtype
            and self._plates == other._plates
        )

    @override
    def stochastically_equal(self, other: "RandomVariable") -> bool:
        return self.structurally_equal(other)


ExprInput: TypeAlias = RandomVariable | Scalar


def _fix_expr(expr: ExprInput) -> RandomVariable:
    if isinstance(expr, RandomVariable):
        return expr
    assert is_scalar(expr), f"{type(expr) = }"
    if isinstance(expr, bool):
        return Constant(value=expr, dtype=Bool())
    if isinstance(expr, int):
        return Constant(value=expr, dtype=Int(min=expr, max=expr))
    if isinstance(expr, float):
        return Constant(value=expr, dtype=Float(min=expr, max=expr))
    assert False, f"unreachable, {type(expr) = }"
