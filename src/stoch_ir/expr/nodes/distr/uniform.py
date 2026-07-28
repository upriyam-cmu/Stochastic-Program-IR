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
class UniformDistribution(RandomDistributionNode):
    """Elementwise continuous uniform distribution over symbolic bounds."""

    low: RandomVariable
    high: RandomVariable

    @staticmethod
    def wrap(
        low: ExprInput = 0.0,
        high: ExprInput = 1.0,
        *,
        rng_label: RngLabel | None = None,
        phase_requirement: Phase = None,
    ) -> "UniformDistribution":
        return UniformDistribution(
            phase_requirement=phase_requirement,
            rng_label=rng_label,
            _node_entropy=None,
            _sampling_seed=None,
            low=as_random_variable(low),
            high=as_random_variable(high),
        )

    @override
    def _compute_dependencies(self) -> tuple[Dependency, ...]:
        return Dependency.wrap({"low": self.low, "high": self.high})

    @override
    def _rewrite_dependencies(
        self,
        dependencies: Mapping[str, RandomVariable],
    ) -> Self:
        return replace(
            self,
            low=dependencies["low"],
            high=dependencies["high"],
        )

    @override
    def _compute_plate_layout(self) -> PlateLayout:
        return self.low.plate_layout | self.high.plate_layout

    @override
    def _compute_value_meta(self) -> ValueMeta:
        low_support = self.low.value_meta.support
        high_support = self.high.value_meta.support
        if (
            low_support is ValueSupport.UNIT_INTERVAL
            and high_support is ValueSupport.UNIT_INTERVAL
        ):
            support = ValueSupport.UNIT_INTERVAL
        elif low_support in (
            ValueSupport.UNIT_INTERVAL,
            ValueSupport.POSITIVE_BRANCH,
        ):
            support = ValueSupport.POSITIVE_BRANCH
        elif high_support is ValueSupport.NEGATIVE_BRANCH:
            support = ValueSupport.NEGATIVE_BRANCH
        else:
            support = ValueSupport.REAL
        return ValueMeta(dtype=DataType.FLOAT, support=support)

    @override
    def structurally_equal(self, other: RandomVariable) -> bool:
        return (
            isinstance(other, UniformDistribution)
            and self.low.structurally_equal(other.low)
            and self.high.structurally_equal(other.high)
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

        low = dependencies["low"]
        high = dependencies["high"]
        low_data = add_plates(
            low.data,
            old_layout=low.layout,
            new_layout=output_layout,
            plate_sizes=plate_sizes,
        )
        high_data = add_plates(
            high.data,
            old_layout=high.layout,
            new_layout=output_layout,
            plate_sizes=plate_sizes,
        )
        if np.any(low_data >= high_data):
            raise InvalidSupportError(
                "Uniform requires low to be strictly less than high"
            )
        return np.asarray(rng.uniform(low=low_data, high=high_data))


def uniform(
    low: ExprInput = 0.0,
    high: ExprInput = 1.0,
    *,
    rng_label: RngLabel | None = None,
) -> UniformDistribution:
    return UniformDistribution.wrap(
        low,
        high,
        rng_label=rng_label,
        phase_requirement=current_sampling_phase(),
    )


Uniform = uniform
