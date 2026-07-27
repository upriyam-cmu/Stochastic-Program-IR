import numpy as np
from typing_extensions import override

from ....rng import RngKey
from ...meta import DataType, Phase, PlateLayout, PlateSizes, ValueMeta, ValueSupport
from ....phases import current_sampling_phase
from ..base import Dependency, ExprInput, RandomVariable, as_random_variable, rv_impl
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
        rng_key: RngKey | None = None,
        phase_requirement: Phase = None,
    ) -> "Gaussian":
        return Gaussian(
            phase_requirement=phase_requirement,
            rng_key=rng_key,
            _resolved_rng_key=None,
            mu=as_random_variable(mu),
            sigma=as_random_variable(sigma),
        )

    @override
    def _compute_dependencies(self) -> tuple[Dependency, ...]:
        return Dependency.wrap({"mu": self.mu, "sigma": self.sigma})

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
        output_layout: PlateLayout,
        plate_sizes: PlateSizes | None = None,
    ) -> np.ndarray:
        from ..shape import add_plates

        mu = self.mu.value(plate_sizes)
        sigma = self.sigma.value(plate_sizes)
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
            raise ValueError("Gaussian sigma must be strictly positive")
        return np.asarray(rng.normal(loc=mu_data, scale=sigma_data))


def normal(
    mu: ExprInput,
    sigma: ExprInput,
    *,
    rng_key: RngKey | None = None,
) -> Gaussian:
    return Gaussian.wrap(
        mu,
        sigma,
        rng_key=rng_key,
        phase_requirement=current_sampling_phase(),
    )


Normal = normal
