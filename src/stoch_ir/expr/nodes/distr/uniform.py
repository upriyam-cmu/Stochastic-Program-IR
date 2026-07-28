from collections.abc import Iterable, Mapping
from dataclasses import replace

import numpy as np
from typing_extensions import Self, override

from ....errors import InvalidSupportError
from ....phases import _current_sampling_phase
from ....rng import RngLabel
from ...meta import (
    ConcreteValue,
    DataType,
    Phase,
    Plate,
    PlateLayout,
    PlateSizes,
    ValueMeta,
    ValueSupport,
)
from ..base import Dependency, ExprInput, RandomVariable, as_random_variable, rv_impl
from .base import (
    RandomDistributionNode,
    align_direct_constants,
    resolve_output_layout,
    warn_possible_invalid_parameter,
)


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
        plates: Iterable[Plate] | None = None,
        rng_label: RngLabel | None = None,
        phase_requirement: Phase = None,
    ) -> "UniformDistribution":
        resolved_low = as_random_variable(low)
        resolved_high = as_random_variable(high)
        aligned_bounds = align_direct_constants(resolved_low, resolved_high)
        if aligned_bounds is not None:
            low_data, high_data = aligned_bounds
            if np.any(
                ~np.isfinite(low_data)
                | ~np.isfinite(high_data)
                | (low_data >= high_data)
            ):
                raise InvalidSupportError(
                    "Uniform requires low to be strictly less than high"
                )
        else:
            low_support = resolved_low.value_meta.support
            high_support = resolved_high.value_meta.support
            if (
                low_support
                in (
                    ValueSupport.UNIT_INTERVAL,
                    ValueSupport.POSITIVE_BRANCH,
                )
                and high_support is ValueSupport.NEGATIVE_BRANCH
            ):
                raise InvalidSupportError(
                    "Uniform bound supports guarantee low >= high"
                )
            if not (
                low_support is ValueSupport.NEGATIVE_BRANCH
                and high_support
                in (
                    ValueSupport.UNIT_INTERVAL,
                    ValueSupport.POSITIVE_BRANCH,
                )
            ):
                warn_possible_invalid_parameter(
                    "Uniform",
                    "low/high",
                    "low < high",
                )
        return UniformDistribution(
            phase_requirement=phase_requirement,
            rng_label=rng_label,
            _node_entropy=None,
            _sampling_seed=None,
            output_layout=resolve_output_layout(
                (resolved_low, resolved_high),
                plates,
            ),
            low=resolved_low,
            high=resolved_high,
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
        if np.any(
            ~np.isfinite(low_data) | ~np.isfinite(high_data) | (low_data >= high_data)
        ):
            raise InvalidSupportError(
                "Uniform requires low to be strictly less than high"
            )
        return np.asarray(rng.uniform(low=low_data, high=high_data))


def Uniform(
    low: RandomVariable | bool | float = 0.0,
    high: RandomVariable | bool | float = 1.0,
    *,
    plates: Iterable[Plate] | None = None,
    rng_label: RngLabel | None = None,
) -> RandomVariable:
    """Create an elementwise continuous uniform random variable.

    Parameters
    ----------
    low
        Symbolic or scalar lower bound.
    high
        Symbolic or scalar upper bound. It must be strictly greater than
        ``low`` when sampled.
    plates
        Complete output plate layout. When omitted, the union of parameter
        plates is used.
    rng_label
        Optional semantic label mixed into graph-derived node entropy.

    Returns
    -------
    RandomVariable
        A symbolic uniform draw with the resolved complete output plates.
    """

    return UniformDistribution.wrap(
        low,
        high,
        plates=plates,
        rng_label=rng_label,
        phase_requirement=_current_sampling_phase(),
    )
