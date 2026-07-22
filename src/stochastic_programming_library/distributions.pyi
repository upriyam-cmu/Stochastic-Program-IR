from .expr import Expr, ExprInput

class Distribution(Expr):
    @property
    def phase(self) -> str | None: ...
    @property
    def rng_name(self) -> str | None: ...

class Normal(Distribution):
    def __init__(
        self,
        mu: ExprInput,
        sigma: ExprInput,
        *,
        rng_name: str | None = ...,
    ) -> None: ...

class Uniform(Distribution):
    def __init__(
        self,
        low: ExprInput,
        high: ExprInput,
        *,
        rng_name: str | None = ...,
    ) -> None: ...

class Bernoulli(Distribution):
    def __init__(
        self,
        p: ExprInput,
        *,
        rng_name: str | None = ...,
    ) -> None: ...
