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
from ..base import (
    Dependency,
    ExprInput,
    RandomVariable,
    as_random_variable,
    rv_impl,
)
from .base import (
    RandomDistributionNode,
    direct_constant_data,
    resolve_output_layout,
    warn_possible_invalid_parameter,
)


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
        plates: Iterable[Plate] | None = None,
        rng_label: RngLabel | None = None,
        phase_requirement: Phase = None,
    ) -> "Gaussian":
        resolved_mu = as_random_variable(mu)
        resolved_sigma = as_random_variable(sigma)
        sigma_data = direct_constant_data(resolved_sigma)
        if sigma_data is not None:
            if np.any(~np.isfinite(sigma_data) | (sigma_data <= 0)):
                raise InvalidSupportError("Gaussian sigma must be strictly positive")
        elif resolved_sigma.value_meta.support is ValueSupport.NEGATIVE_BRANCH:
            raise InvalidSupportError(
                "Gaussian sigma support is guaranteed to be nonpositive"
            )
        elif resolved_sigma.value_meta.support is ValueSupport.REAL:
            warn_possible_invalid_parameter(
                "Gaussian",
                "sigma",
                "strict positivity",
            )
        return Gaussian(
            phase_requirement=phase_requirement,
            rng_label=rng_label,
            _node_entropy=None,
            _sampling_seed=None,
            output_layout=resolve_output_layout(
                (resolved_mu, resolved_sigma),
                plates,
            ),
            mu=resolved_mu,
            sigma=resolved_sigma,
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
    def _compute_value_meta(self) -> ValueMeta:
        return ValueMeta(dtype=DataType.FLOAT, support=ValueSupport.REAL)

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
        if np.any(~np.isfinite(sigma_data) | (sigma_data <= 0)):
            raise InvalidSupportError("Gaussian sigma must be strictly positive")
        return np.asarray(rng.normal(loc=mu_data, scale=sigma_data))


def Normal(
    mu: RandomVariable | bool | float,
    sigma: RandomVariable | bool | float,
    *,
    plates: Iterable[Plate] | None = None,
    rng_label: RngLabel | None = None,
) -> RandomVariable:
    """Create a univariate normal random variable.

    Parameters
    ----------
    mu
        Symbolic or scalar location.
    sigma
        Symbolic or scalar strictly positive scale.
    plates
        Complete output plate layout. When omitted, the union of parameter
        plates is used.
    rng_label
        Optional semantic label mixed into graph-derived node entropy.

    Returns
    -------
    RandomVariable
        A symbolic normal draw with the resolved complete output plates.
    """

    return Gaussian.wrap(
        mu,
        sigma,
        plates=plates,
        rng_label=rng_label,
        phase_requirement=_current_sampling_phase(),
    )
