from collections.abc import Mapping
from dataclasses import replace

import numpy as np
from typing_extensions import Self, override

from ....errors import InvalidSupportError
from ....phases import current_sampling_phase
from ....rng import RngLabel
from ...meta import (
    ConcreteValue,
    DataType,
    Phase,
    PlateLayout,
    PlateSizes,
    ValueMeta,
    ValueSupport,
)
from ..base import Dependency, ExprInput, RandomVariable, as_random_variable, rv_impl
from .base import RandomDistributionNode


@rv_impl
class BernoulliDistribution(RandomDistributionNode):
    """Elementwise Bernoulli distribution with symbolic probabilities."""

    p: RandomVariable

    @staticmethod
    def wrap(
        p: ExprInput,
        *,
        rng_label: RngLabel | None = None,
        phase_requirement: Phase = None,
    ) -> "BernoulliDistribution":
        return BernoulliDistribution(
            phase_requirement=phase_requirement,
            rng_label=rng_label,
            _node_entropy=None,
            _sampling_seed=None,
            p=as_random_variable(p),
        )

    @override
    def _compute_dependencies(self) -> tuple[Dependency, ...]:
        return Dependency.wrap({"p": self.p})

    @override
    def _rewrite_dependencies(
        self,
        dependencies: Mapping[str, RandomVariable],
    ) -> Self:
        return replace(self, p=dependencies["p"])

    @override
    def _compute_plate_layout(self) -> PlateLayout:
        return self.p.plate_layout

    @override
    def _compute_value_meta(self) -> ValueMeta:
        return ValueMeta(
            dtype=DataType.BOOL,
            support=ValueSupport.UNIT_INTERVAL,
        )

    @override
    def structurally_equal(self, other: RandomVariable) -> bool:
        return isinstance(other, BernoulliDistribution) and self.p.structurally_equal(
            other.p
        )

    @override
    def _sample_value(
        self,
        rng: np.random.Generator,
        dependencies: Mapping[str, ConcreteValue],
        output_layout: PlateLayout,
        plate_sizes: PlateSizes,
    ) -> np.ndarray:
        from ..shape import add_plates

        probability = dependencies["p"]
        probability_data = add_plates(
            probability.data,
            old_layout=probability.layout,
            new_layout=output_layout,
            plate_sizes=plate_sizes,
        )
        if np.any((probability_data < 0) | (probability_data > 1)):
            raise InvalidSupportError("Bernoulli p must satisfy 0 <= p <= 1")
        return np.asarray(rng.binomial(1, probability_data), dtype=np.bool_)


def bernoulli(
    p: ExprInput,
    *,
    rng_label: RngLabel | None = None,
) -> BernoulliDistribution:
    return BernoulliDistribution.wrap(
        p,
        rng_label=rng_label,
        phase_requirement=current_sampling_phase(),
    )


Bernoulli = bernoulli
