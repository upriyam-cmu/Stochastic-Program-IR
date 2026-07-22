from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from enum import Enum

from typing_extensions import Self, override

from ..rng import Seed
from .base import Constant, RandomVariable
from .dtype import Bool, DataType, Float, Int
from .util import Data, Phase, Plate


class BinOpImpl(ABC):
    @abstractmethod
    def resolve_dtype(self, lhs: DataType, rhs: DataType) -> DataType: ...

    @abstractmethod
    def compute_value(self, lhs: Data, rhs: Data) -> Data: ...

    def __eq__(self, other: object) -> bool:
        return type(self) is type(other)


class BinOpNode(RandomVariable):
    lhs: RandomVariable
    rhs: RandomVariable
    op: BinOpImpl

    def __init__(
        self,
        lhs: RandomVariable,
        rhs: RandomVariable,
        op_impl: BinOpImpl,
    ) -> None:
        self.lhs = lhs
        self.rhs = rhs
        self.op = op_impl

    @override
    def _resolve_plates(self) -> frozenset[Plate]:
        return self.lhs.plates | self.rhs.plates

    @override
    def _resolve_pending_phases(self) -> frozenset[Phase]:
        return self.lhs.pending_phases | self.rhs.pending_phases

    @override
    def _resolve_has_value(self) -> bool:
        return self.lhs.has_value and self.rhs.has_value

    @override
    def _resolve_dtype(self) -> DataType:
        return self.op.resolve_dtype(self.lhs.dtype, self.rhs.dtype)

    @override
    def value(self, plate_sizes: Mapping[Plate, int] | None = None) -> Data: ...

    @override
    def materialize(
        self,
        *,
        seed: Seed | None = None,
        plate_sizes: Mapping[Plate, int] | None = None,
        phases: Iterable[Phase] | None = None,
    ) -> RandomVariable:
        lhs = self.lhs.materialize(seed=seed, plate_sizes=plate_sizes, phases=phases)
        rhs = self.rhs.materialize(seed=seed, plate_sizes=plate_sizes, phases=phases)

        if lhs.has_value and rhs.has_value:
            return Constant(
                self.op.compute_value(
                    lhs.value(plate_sizes),
                    rhs.value(plate_sizes),
                ),
                self.dtype,
                *self.plates,
            )
        if lhs == self.lhs and rhs == self.rhs:
            return self
        return BinOpNode(lhs, rhs, self.op)

    @override
    def structurally_equal(self, other: RandomVariable) -> bool:
        return (
            isinstance(other, BinOpNode)
            and self.lhs.structurally_equal(other.lhs)
            and self.rhs.structurally_equal(other.rhs)
            and self.op == other.op
        )

    @override
    def stochastically_equal(self, other: "RandomVariable") -> bool:
        return (
            isinstance(other, BinOpNode)
            and self.lhs.stochastically_equal(other.lhs)
            and self.rhs.stochastically_equal(other.rhs)
            and self.op == other.op
        )


class AddOp(BinOpImpl):
    @override
    def resolve_dtype(self, lhs: DataType, rhs: DataType) -> DataType: ...

    @override
    def compute_value(self, lhs: Data, rhs: Data) -> Data:
        return lhs + rhs
