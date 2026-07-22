from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Protocol

from ..rng import ResolvedRngKey, RngKey, Seed
from .base import ExprInput, RandomVariable
from .meta import ConcreteValue, Phase, Plate, PlateLayout, PlateSizes, ValueMeta
from .ops import BinOpImpl, Reduction, ReductionImpl, UnaryOpImpl

class DistributionKind(str, Enum):
    NORMAL: str
    UNIFORM: str
    BERNOULLI: str

@dataclass(frozen=True, slots=True)
class DistributionParameter:
    name: str
    value: RandomVariable

class DistributionSampler(Protocol):
    @property
    def kind(self) -> DistributionKind: ...
    def resolve_meta(
        self,
        parameters: Mapping[str, ValueMeta],
    ) -> ValueMeta: ...
    def sample(
        self,
        parameters: Mapping[str, ConcreteValue],
        *,
        seed: Seed,
        output_layout: PlateLayout,
        plate_sizes: PlateSizes,
    ) -> ConcreteValue: ...

class UnaryOpNode(RandomVariable):
    def __init__(self, child: RandomVariable, op: UnaryOpImpl) -> None: ...
    @property
    def child(self) -> RandomVariable: ...
    @property
    def op(self) -> UnaryOpImpl: ...

class BinOpNode(RandomVariable):
    def __init__(
        self,
        lhs: RandomVariable,
        rhs: RandomVariable,
        op: BinOpImpl,
    ) -> None: ...
    @property
    def lhs(self) -> RandomVariable: ...
    @property
    def rhs(self) -> RandomVariable: ...
    @property
    def op(self) -> BinOpImpl: ...

class AddPlatesNode(RandomVariable):
    def __init__(
        self,
        child: RandomVariable,
        *plates: Plate,
    ) -> None: ...
    @property
    def child(self) -> RandomVariable: ...
    @property
    def added_plates(self) -> tuple[Plate, ...]: ...

class ReducePlatesNode(RandomVariable):
    def __init__(
        self,
        child: RandomVariable,
        *plates: Plate,
        reduction: Reduction | ReductionImpl,
    ) -> None: ...
    @property
    def child(self) -> RandomVariable: ...
    @property
    def reduced_plates(self) -> tuple[Plate, ...]: ...
    @property
    def reduction(self) -> ReductionImpl: ...

class DistributionNode(RandomVariable):
    def __init__(
        self,
        sampler: DistributionSampler,
        parameters: tuple[DistributionParameter, ...],
        *,
        phase: Phase = ...,
        rng_key: RngKey | None = ...,
        resolved_rng_key: ResolvedRngKey | None = ...,
    ) -> None: ...
    @property
    def sampler(self) -> DistributionSampler: ...
    @property
    def kind(self) -> DistributionKind: ...
    @property
    def parameters(self) -> tuple[DistributionParameter, ...]: ...
    @property
    def phase(self) -> Phase: ...
    @property
    def rng_key(self) -> RngKey | None: ...
    @property
    def resolved_rng_key(self) -> ResolvedRngKey | None: ...
    def clear_phase(self) -> DistributionNode: ...
    def with_resolved_rng_key(self, key: ResolvedRngKey) -> DistributionNode: ...

def distribution_parameter(name: str, value: ExprInput) -> DistributionParameter: ...
