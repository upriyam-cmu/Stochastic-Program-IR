from dataclasses import dataclass
from typing import Iterable

from ..rng import Seed
from .meta import ConcreteValue, Phase, PlateSizes
from .nodes import RandomVariable

@dataclass(frozen=True, slots=True)
class SamplingCheckpoint:
    @staticmethod
    def _wrap(
        root: RandomVariable,
        plate_sizes: PlateSizes,
    ) -> SamplingCheckpoint: ...
    @property
    def pending_phases(self) -> frozenset[Phase]: ...
    @property
    def is_fully_materialized(self) -> bool: ...
    def structurally_equal(self, other: SamplingCheckpoint) -> bool: ...
    def stochastically_equal(self, other: SamplingCheckpoint) -> bool: ...
    def materialize(
        self,
        *,
        seed: Seed | None = ...,
        plate_sizes: PlateSizes | None = ...,
        phases: Iterable[Phase] | None = ...,
    ) -> SamplingCheckpoint: ...
    def value(self) -> ConcreteValue: ...
    def realize(
        self,
        *,
        seed: Seed | None = ...,
        plate_sizes: PlateSizes | None = ...,
    ) -> ConcreteValue: ...

def materialize(
    root: RandomVariable,
    *,
    seed: Seed | None = ...,
    plate_sizes: PlateSizes | None = ...,
    phases: Iterable[Phase] | None = ...,
) -> SamplingCheckpoint: ...
def realize(
    root: RandomVariable,
    *,
    seed: Seed | None = ...,
    plate_sizes: PlateSizes | None = ...,
) -> ConcreteValue: ...
