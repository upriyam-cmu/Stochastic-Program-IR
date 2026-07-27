from abc import ABC, abstractmethod

import numpy as np
from typing_extensions import override

from ....errors import UnrealizedGraphError
from ....rng import ResolvedRngKey, RngKey
from ...meta import ConcreteValue, Phase, PlateLayout, PlateSizes
from ..base import RandomVariable, rv_impl


@rv_impl
class RandomDistributionNode(RandomVariable, ABC):
    phase_requirement: Phase | None
    rng_key: RngKey | None
    _resolved_rng_key: ResolvedRngKey | None

    @override
    def _compute_pending_phases(self) -> frozenset[Phase]:
        return (
            frozenset({self.phase_requirement})
            if self.phase_requirement is not None
            else frozenset()
        ).union(*(dep.var.pending_phases for dep in self.dependencies))

    @override
    def _compute_has_value(self) -> bool:
        return False

    @abstractmethod
    def _sample_value(
        self,
        rng: np.random.Generator,
        output_layout: PlateLayout,
        plate_sizes: PlateSizes | None = None,
    ) -> np.ndarray: ...

    @override
    def value(self, plate_sizes: PlateSizes | None = None) -> ConcreteValue:
        # does this just always error? the idea being that you have
        # to materialize into a constant value when the time comes?
        raise UnrealizedGraphError(
            "Cannot get value for a node whose value is not yet realized"
        )
