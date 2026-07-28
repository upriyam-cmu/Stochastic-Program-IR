from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from typing_extensions import override

from ..meta import DataType, ValueMeta, ValueSupport
from ._numeric import checked_int64_result


@dataclass(frozen=True, slots=True)
class Reduction(ABC):
    """Opaque reduction type used by the public singleton objects."""

    def _remap_dtype(self, child: DataType) -> DataType:
        # default: same as input
        return child

    def _remap_support(self, child: ValueSupport) -> ValueSupport:
        # default: same as input
        return child

    def resolve_meta(self, child: ValueMeta) -> ValueMeta:
        return ValueMeta(
            dtype=self._remap_dtype(child.dtype),
            support=self._remap_support(child.support),
        )

    @abstractmethod
    def compute_value(
        self,
        child: np.ndarray,
        *,
        axes: tuple[int, ...],
    ) -> np.ndarray: ...


@dataclass(frozen=True, slots=True)
class _MeanReduction(Reduction):
    @override
    def _remap_dtype(self, child: DataType) -> DataType:
        return DataType.FLOAT

    @override
    def compute_value(self, child: np.ndarray, *, axes: tuple[int, ...]) -> np.ndarray:
        return np.mean(child, axis=axes)


@dataclass(frozen=True, slots=True)
class _SumReduction(Reduction):
    @override
    def _remap_dtype(self, child: DataType) -> DataType:
        return max(child, DataType.INT)  # bool -> int

    @override
    def _remap_support(self, child: ValueSupport) -> ValueSupport:
        return (
            child
            if child != ValueSupport.UNIT_INTERVAL
            else ValueSupport.POSITIVE_BRANCH
        )

    @override
    def compute_value(self, child: np.ndarray, *, axes: tuple[int, ...]) -> np.ndarray:
        if child.dtype.kind in ("b", "i", "u"):
            return checked_int64_result(np.sum(child.astype(object), axis=axes))
        return np.sum(child, axis=axes)


@dataclass(frozen=True, slots=True)
class _MaxReduction(Reduction):
    @override
    def compute_value(self, child: np.ndarray, *, axes: tuple[int, ...]) -> np.ndarray:
        return np.max(child, axis=axes)


@dataclass(frozen=True, slots=True)
class _MinReduction(Reduction):
    @override
    def compute_value(self, child: np.ndarray, *, axes: tuple[int, ...]) -> np.ndarray:
        return np.min(child, axis=axes)


@dataclass(frozen=True, slots=True)
class _ProductReduction(Reduction):
    @override
    def _remap_dtype(self, child: DataType) -> DataType:
        return max(child, DataType.INT)  # bool -> int

    @override
    def _remap_support(self, child: ValueSupport) -> ValueSupport:
        return child if child != ValueSupport.NEGATIVE_BRANCH else ValueSupport.REAL

    @override
    def compute_value(self, child: np.ndarray, *, axes: tuple[int, ...]) -> np.ndarray:
        if child.dtype.kind in ("b", "i", "u"):
            return checked_int64_result(np.prod(child.astype(object), axis=axes))
        return np.prod(child, axis=axes)


@dataclass(frozen=True, slots=True)
class _LogSumExpReduction(Reduction):
    @override
    def _remap_dtype(self, child: DataType) -> DataType:
        return DataType.FLOAT

    @override
    def _remap_support(self, child: ValueSupport) -> ValueSupport:
        return ValueSupport.REAL

    @override
    def compute_value(self, child: np.ndarray, *, axes: tuple[int, ...]) -> np.ndarray:
        return np.logaddexp.reduce(child, axis=axes)


MEAN = _MeanReduction()
SUM = _SumReduction()
MAX = _MaxReduction()
MIN = _MinReduction()
PROD = _ProductReduction()
LOGSUMEXP = _LogSumExpReduction()
