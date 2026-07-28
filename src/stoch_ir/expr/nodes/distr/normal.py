from collections.abc import Mapping
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
    PlateLayout,
    PlateSizes,
    ValueMeta,
    ValueSupport,
)
from ..base import (
    Dependency,
    ExprInput,
    RandomVariable,
    as_random_variable,
    rv_impl,
)
from .base import RandomDistributionNode


@rv_impl
class Gaussian(RandomDistributionNode):
    # This represents a univariate Gaussian. Multivariate/event dimensions are
    # intentionally outside the current plate-only shape model.
    mu: RandomVariable
    sigma: RandomVariable

    @staticmethod
    def wrap(
        mu: ExprInput,
        sigma: ExprInput,
        *,
        rng_label: RngLabel | None = None,
        phase_requirement: Phase = None,
    ) -> "Gaussian":
        return Gaussian(
            phase_requirement=phase_requirement,
            rng_label=rng_label,
            _node_entropy=None,
            _sampling_seed=None,
            mu=as_random_variable(mu),
            sigma=as_random_variable(sigma),
        )

    @override
    def _compute_dependencies(self) -> tuple[Dependency, ...]:
        return Dependency.wrap({"mu": self.mu, "sigma": self.sigma})

    @override
    def _rewrite_dependencies(
        self,
        dependencies: Mapping[str, RandomVariable],
    ) -> Self:
        return replace(
            self,
            mu=dependencies["mu"],
            sigma=dependencies["sigma"],
        )

    @override
    def _compute_plate_layout(self) -> PlateLayout:
        return self.mu.plate_layout | self.sigma.plate_layout

    @override
    def _compute_value_meta(self) -> ValueMeta:
        if self.sigma.value_meta.support in (
            ValueSupport.REAL,
            ValueSupport.NEGATIVE_BRANCH,
        ):
            pass  # TODO add warning about negative sigma being bad practice?
        return ValueMeta(dtype=DataType.FLOAT, support=ValueSupport.REAL)

    @override
    def structurally_equal(self, other: RandomVariable) -> bool:
        return (
            isinstance(other, Gaussian)
            and self.mu.structurally_equal(other.mu)
            and self.sigma.structurally_equal(other.sigma)
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

        mu = dependencies["mu"]
        sigma = dependencies["sigma"]
        mu_data = add_plates(
            mu.data,
            old_layout=mu.layout,
            new_layout=output_layout,
            plate_sizes=plate_sizes,
        )
        sigma_data = add_plates(
            sigma.data,
            old_layout=sigma.layout,
            new_layout=output_layout,
            plate_sizes=plate_sizes,
        )
        if np.any(sigma_data <= 0):
            raise InvalidSupportError("Gaussian sigma must be strictly positive")
        return np.asarray(rng.normal(loc=mu_data, scale=sigma_data))


def Normal(
    mu: RandomVariable | bool | int | float,
    sigma: RandomVariable | bool | int | float,
    *,
    rng_label: RngLabel | None = None,
) -> RandomVariable:
    """Create a univariate normal random variable.

    Parameters
    ----------
    mu
        Symbolic or scalar location.
    sigma
        Symbolic or scalar strictly positive scale.
    rng_label
        Optional semantic label mixed into graph-derived node entropy.

    Returns
    -------
    RandomVariable
        A symbolic normal draw with the union of parameter plates.
    """

    return Gaussian.wrap(
        mu,
        sigma,
        rng_label=rng_label,
        phase_requirement=_current_sampling_phase(),
    )
