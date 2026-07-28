import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from typing_extensions import override

from ...errors import InvalidSupportError
from ..meta import DataType, ValueMeta, ValueSupport


@dataclass(frozen=True, slots=True)
class UnaryOpImpl(ABC):
    def _remap_dtype(self, child: DataType) -> DataType:
        # default: return float
        return DataType.FLOAT

    @abstractmethod
    def _remap_support(self, child: ValueSupport) -> ValueSupport: ...

    def resolve_meta(self, child: ValueMeta) -> ValueMeta:
        return ValueMeta(
            dtype=self._remap_dtype(child.dtype),
            support=self._remap_support(child.support),
        )

    @abstractmethod
    def compute_value(self, child: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True, slots=True)
class ExpOp(UnaryOpImpl):
    @override
    def _remap_support(self, child: ValueSupport) -> ValueSupport:
        if child == ValueSupport.NEGATIVE_BRANCH:
            return ValueSupport.UNIT_INTERVAL
        return ValueSupport.POSITIVE_BRANCH

    @override
    def compute_value(self, child: np.ndarray) -> np.ndarray:
        return np.exp(child)


@dataclass(frozen=True, slots=True)
class LogOp(UnaryOpImpl):
    @override
    def _remap_support(self, child: ValueSupport) -> ValueSupport:
        # TODO add specific custom warning types
        # TODO not sure if stacklevel/skip_file_prefixes would help here
        # since, we do lazy computation, so it would be hard to point
        # to the right spot anyways
        match child:
            case ValueSupport.NEGATIVE_BRANCH:
                raise InvalidSupportError(
                    "Log does not support negative inputs -- op will fail"
                )

            case ValueSupport.REAL:
                warnings.warn("Log does not support negative inputs -- op may fail")
                return ValueSupport.REAL

            case ValueSupport.UNIT_INTERVAL:
                return ValueSupport.NEGATIVE_BRANCH

            case ValueSupport.POSITIVE_BRANCH:
                return ValueSupport.REAL

        assert False, f"unreachable: {child}"

    @override
    def compute_value(self, child: np.ndarray) -> np.ndarray:
        return np.log(child)


@dataclass(frozen=True, slots=True)
class SoftplusOp(UnaryOpImpl):
    @override
    def _remap_support(self, child: ValueSupport) -> ValueSupport:
        if child == ValueSupport.NEGATIVE_BRANCH:
            return ValueSupport.UNIT_INTERVAL
        return ValueSupport.POSITIVE_BRANCH

    @override
    def compute_value(self, child: np.ndarray) -> np.ndarray:
        # ln(1 + exp(x))
        return np.logaddexp(child, 0)


@dataclass(frozen=True, slots=True)
class AbsOp(UnaryOpImpl):
    @override
    def _remap_dtype(self, child: DataType) -> DataType:
        return child

    @override
    def _remap_support(self, child: ValueSupport) -> ValueSupport:
        if child == ValueSupport.UNIT_INTERVAL:
            return ValueSupport.UNIT_INTERVAL
        return ValueSupport.POSITIVE_BRANCH

    @override
    def compute_value(self, child: np.ndarray) -> np.ndarray:
        return np.abs(child)


# TODO implement sigmoid, logit, sqrt
