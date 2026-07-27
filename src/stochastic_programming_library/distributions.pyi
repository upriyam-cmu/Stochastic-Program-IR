from .expr import ExprInput
from .expr.nodes import Gaussian as Gaussian
from .rng import RngKey

def normal(
    mu: ExprInput,
    sigma: ExprInput,
    *,
    rng_key: RngKey | None = ...,
) -> Gaussian: ...

Normal = normal
