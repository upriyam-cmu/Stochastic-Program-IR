from .expr import ExprInput
from .expr.nodes import Gaussian as Gaussian
from .rng import RngLabel

def normal(
    mu: ExprInput,
    sigma: ExprInput,
    *,
    rng_label: RngLabel | None = ...,
) -> Gaussian: ...

Normal = normal
