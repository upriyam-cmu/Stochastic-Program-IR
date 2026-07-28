from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from dataclasses import replace

import numpy as np
from typing_extensions import Self, override

from ....errors import (
    PlateExpectationError,
    RngLabelError,
    UnrealizedGraphError,
    UnresolvedRandomnessError,
)
from ....rng import (
    NodeEntropy,
    RngLabel,
    Seed,
    derive_sampling_seed,
)
from ...meta import ConcreteValue, Phase, Plate, PlateLayout, PlateSizes
from ..base import RandomVariable, rv_impl


@rv_impl
class RandomDistributionNode(RandomVariable, ABC):
    """Base state shared by all stochastic sampling nodes.

    ``phase_requirement`` is the remaining execution barrier. ``rng_label`` is
    immutable user metadata. ``_node_entropy`` is resolved from graph structure
    plus that label when a checkpoint is created. ``_sampling_seed`` is bound
    from one materialization run seed when the phase barrier is lifted.
    """

    phase_requirement: Phase | None
    rng_label: RngLabel | None
    _node_entropy: NodeEntropy | None
    _sampling_seed: Seed | None
    output_layout: PlateLayout

    def __post_init__(self) -> None:
        if self.rng_label is not None and (
            not isinstance(self.rng_label, str) or not self.rng_label
        ):
            raise RngLabelError("rng_label must be None or a non-empty string")
        super().__post_init__()
        for dependency in self._dependency_slots:
            extra = dependency.var.plate_layout.as_set - self.output_layout.as_set
            if extra:
                raise PlateExpectationError(
                    f"{type(self).__name__} parameter {dependency.name!r} has "
                    f"plates {dependency.var.plates}, which are not contained in "
                    f"output plates {self.output_layout.plates}: {sorted(extra)}"
                )

    @override
    def _compute_plate_layout(self) -> PlateLayout:
        return self.output_layout

    @override
    def _compute_pending_phases(self) -> frozenset[str]:
        return (
            frozenset({self.phase_requirement})
            if self.phase_requirement is not None
            else frozenset()
        ).union(*(dep.pending_phases for dep in self.dependencies.values()))

    @override
    def _compute_has_value(self) -> bool:
        return False

    def with_node_entropy(self, node_entropy: NodeEntropy) -> Self:
        if self._node_entropy is not None and self._node_entropy != node_entropy:
            raise UnresolvedRandomnessError(
                "cannot replace an already resolved node entropy"
            )
        return replace(self, _node_entropy=node_entropy)

    def bind_sampling_seed(self, run_seed: Seed) -> Self:
        if self._node_entropy is None:
            raise UnresolvedRandomnessError(
                "cannot bind a sampling seed before resolving node entropy"
            )
        if self._sampling_seed is not None:
            return self
        return replace(
            self,
            phase_requirement=None,
            _sampling_seed=derive_sampling_seed(
                run_seed,
                self._node_entropy,
            ),
        )

    @abstractmethod
    def _sample_value(
        self,
        rng: np.random.Generator,
        dependencies: Mapping[str, ConcreteValue],
        output_layout: PlateLayout,
        plate_sizes: PlateSizes,
    ) -> np.ndarray: ...

    @override
    def _evaluate_concrete(
        self,
        dependencies: Mapping[str, ConcreteValue],
        plate_sizes: PlateSizes,
    ) -> ConcreteValue:
        raise UnrealizedGraphError(
            "a distribution must be sampled rather than deterministically evaluated"
        )


def resolve_output_layout(
    dependencies: Iterable[RandomVariable],
    plates: Iterable[Plate] | None,
) -> PlateLayout:
    """Resolve a distribution's complete output layout."""

    if plates is not None:
        return PlateLayout.wrap(plates)
    return PlateLayout.union(*(dependency.plate_layout for dependency in dependencies))
