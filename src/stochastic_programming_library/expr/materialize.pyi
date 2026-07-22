from dataclasses import dataclass
from typing import Iterable

from ..rng import PlateCoordinate, Seed
from .base import RandomVariable
from .hashing import ResolvedGraphHashes
from .meta import ConcreteValue, Phase, Plate, PlateLayout, PlateSizes
from .nodes import DistributionNode

@dataclass(frozen=True, slots=True)
class MaterializeContext:
    seed: Seed
    plate_sizes: PlateSizes
    cleared_phases: frozenset[Phase]
    lifted_layout: PlateLayout
    hashes: ResolvedGraphHashes

    def with_added_plates(self, *plates: Plate) -> MaterializeContext: ...
    def clears(self, phase: Phase) -> bool: ...
    def seed_for(
        self,
        node: DistributionNode,
        coordinates: tuple[PlateCoordinate, ...],
    ) -> Seed: ...

def materialize_node(
    node: RandomVariable,
    context: MaterializeContext,
) -> RandomVariable: ...

def materialize(
    root: RandomVariable,
    *,
    seed: Seed | None = ...,
    plate_sizes: PlateSizes | None = ...,
    phases: Iterable[Phase] | None = ...,
) -> RandomVariable: ...

def realize(
    root: RandomVariable,
    *,
    seed: Seed | None = ...,
    plate_sizes: PlateSizes | None = ...,
) -> ConcreteValue: ...
