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
    direct_constant_data,
    resolve_output_layout,
    warn_possible_invalid_parameter,
)


@rv_impl
class BernoulliDistribution(RandomDistributionNode):
    """Elementwise Bernoulli distribution with symbolic probabilities."""

    p: RandomVariable

    @staticmethod
    def wrap(
        p: ExprInput,
        *,
        plates: Iterable[Plate] | None = None,
        rng_label: RngLabel | None = None,
        phase_requirement: Phase = None,
    ) -> "BernoulliDistribution":
        resolved_p = as_random_variable(p)
        probability_data = direct_constant_data(resolved_p)
        if probability_data is not None:
            if np.any(
                ~np.isfinite(probability_data)
                | (probability_data < 0)
                | (probability_data > 1)
            ):
                raise InvalidSupportError("Bernoulli p must satisfy 0 <= p <= 1")
        elif resolved_p.value_meta.support is not ValueSupport.UNIT_INTERVAL:
            warn_possible_invalid_parameter(
                "Bernoulli",
                "p",
                "0 <= p <= 1",
            )
        return BernoulliDistribution(
            phase_requirement=phase_requirement,
            rng_label=rng_label,
            _node_entropy=None,
            _sampling_seed=None,
            output_layout=resolve_output_layout((resolved_p,), plates),
            p=resolved_p,
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
    def _compute_value_meta(self) -> ValueMeta:
        return ValueMeta(
            dtype=DataType.BOOL,
            support=ValueSupport.UNIT_INTERVAL,
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
        if np.any(
            ~np.isfinite(probability_data)
            | (probability_data < 0)
            | (probability_data > 1)
        ):
            raise InvalidSupportError("Bernoulli p must satisfy 0 <= p <= 1")
        return np.asarray(rng.binomial(1, probability_data), dtype=np.bool_)


def Bernoulli(
    p: RandomVariable | bool | float,
    *,
    plates: Iterable[Plate] | None = None,
    rng_label: RngLabel | None = None,
) -> RandomVariable:
    """Create an elementwise Bernoulli random variable.

    Parameters
    ----------
    p
        Symbolic or scalar success probability in the closed interval
        ``[0, 1]``.
    plates
        Complete output plate layout. When omitted, the parameter plates are
        used.
    rng_label
        Optional semantic label mixed into graph-derived node entropy.

    Returns
    -------
    RandomVariable
        A symbolic Boolean draw with the resolved complete output plates.
    """

    return BernoulliDistribution.wrap(
        p,
        plates=plates,
        rng_label=rng_label,
        phase_requirement=_current_sampling_phase(),
    )
