import operator
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from typing_extensions import override

from ...errors import ValueValidationError
from ..meta import DataType, ValueMeta, ValueSupport
from ._numeric import checked_int64_binary


@dataclass(frozen=True, slots=True)
class BinOpImpl(ABC):
    def _remap_dtype(self, lhs: DataType, rhs: DataType) -> DataType:
        # default: int unless float is present
        return max(lhs, rhs, DataType.INT)

    @abstractmethod
    def _remap_support(self, lhs: ValueSupport, rhs: ValueSupport) -> ValueSupport: ...

    def resolve_meta(self, lhs: ValueMeta, rhs: ValueMeta) -> ValueMeta:
        return ValueMeta(
            dtype=self._remap_dtype(lhs.dtype, rhs.dtype),
            support=self._remap_support(lhs.support, rhs.support),
        )

    @abstractmethod
    def compute_value(self, lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray: ...


def _assert_positive_or_negative(lhs: ValueSupport, rhs: ValueSupport):
    assert {lhs, rhs} <= {ValueSupport.POSITIVE_BRANCH, ValueSupport.NEGATIVE_BRANCH}, (
        f"{(lhs, rhs) = }"
    )


def _integer_operands(lhs: np.ndarray, rhs: np.ndarray) -> bool:
    return lhs.dtype.kind in ("b", "i", "u") and rhs.dtype.kind in ("b", "i", "u")


@dataclass(frozen=True, slots=True)
class AddOp(BinOpImpl):
    @override
    def _remap_support(self, lhs: ValueSupport, rhs: ValueSupport) -> ValueSupport:
        if lhs == ValueSupport.REAL or rhs == ValueSupport.REAL:
            return ValueSupport.REAL

        lhs = lhs if lhs != ValueSupport.UNIT_INTERVAL else ValueSupport.POSITIVE_BRANCH
        rhs = rhs if rhs != ValueSupport.UNIT_INTERVAL else ValueSupport.POSITIVE_BRANCH

        _assert_positive_or_negative(lhs, rhs)
        return lhs if lhs == rhs else ValueSupport.REAL

    @override
    def compute_value(self, lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
        return (
            checked_int64_binary(lhs, rhs, operator.add)
            if _integer_operands(lhs, rhs)
            else lhs + rhs
        )


@dataclass(frozen=True, slots=True)
class SubtractOp(BinOpImpl):
    @override
    def _remap_support(self, lhs: ValueSupport, rhs: ValueSupport) -> ValueSupport:
        if lhs == ValueSupport.REAL or rhs == ValueSupport.REAL:
            return ValueSupport.REAL

        lhs = lhs if lhs != ValueSupport.UNIT_INTERVAL else ValueSupport.POSITIVE_BRANCH
        rhs = rhs if rhs != ValueSupport.UNIT_INTERVAL else ValueSupport.POSITIVE_BRANCH

        _assert_positive_or_negative(lhs, rhs)
        return lhs if lhs != rhs else ValueSupport.REAL

    @override
    def compute_value(self, lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
        return (
            checked_int64_binary(lhs, rhs, operator.sub)
            if _integer_operands(lhs, rhs)
            else lhs - rhs
        )


@dataclass(frozen=True, slots=True)
class MultiplyOp(BinOpImpl):
    @override
    def _remap_support(self, lhs: ValueSupport, rhs: ValueSupport) -> ValueSupport:
        if lhs == ValueSupport.REAL or rhs == ValueSupport.REAL:
            return ValueSupport.REAL

        if lhs == ValueSupport.UNIT_INTERVAL:
            return rhs
        if rhs == ValueSupport.UNIT_INTERVAL:
            return lhs

        _assert_positive_or_negative(lhs, rhs)
        return (
            ValueSupport.POSITIVE_BRANCH if lhs == rhs else ValueSupport.NEGATIVE_BRANCH
        )

    @override
    def compute_value(self, lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
        return (
            checked_int64_binary(lhs, rhs, operator.mul)
            if _integer_operands(lhs, rhs)
            else lhs * rhs
        )


def _division_support_remap(lhs: ValueSupport, rhs: ValueSupport) -> ValueSupport:
    if lhs == ValueSupport.REAL or rhs == ValueSupport.REAL:
        return ValueSupport.REAL

    lhs = lhs if lhs != ValueSupport.UNIT_INTERVAL else ValueSupport.POSITIVE_BRANCH
    rhs = rhs if rhs != ValueSupport.UNIT_INTERVAL else ValueSupport.POSITIVE_BRANCH

    _assert_positive_or_negative(lhs, rhs)
    return ValueSupport.POSITIVE_BRANCH if lhs == rhs else ValueSupport.NEGATIVE_BRANCH


@dataclass(frozen=True, slots=True)
class TrueDivideOp(BinOpImpl):
    @override
    def _remap_dtype(self, lhs: DataType, rhs: DataType) -> DataType:
        return DataType.FLOAT

    @override
    def _remap_support(self, lhs: ValueSupport, rhs: ValueSupport) -> ValueSupport:
        return _division_support_remap(lhs, rhs)

    @override
    def compute_value(self, lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
        return lhs / rhs


@dataclass(frozen=True, slots=True)
class FloorDivideOp(BinOpImpl):
    @override
    def _remap_dtype(self, lhs: DataType, rhs: DataType) -> DataType:
        return max(lhs, rhs, DataType.INT)

    @override
    def _remap_support(self, lhs: ValueSupport, rhs: ValueSupport) -> ValueSupport:
        return _division_support_remap(lhs, rhs)

    @override
    def compute_value(self, lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
        bounds = np.iinfo(np.int64)
        if bool(np.any((lhs == bounds.min) & (rhs == -1))):
            raise ValueValidationError(
                "integer operation result is outside int64 range"
            )
        return lhs // rhs


# TODO implement, max, min, boolean ops, shifts?, power
