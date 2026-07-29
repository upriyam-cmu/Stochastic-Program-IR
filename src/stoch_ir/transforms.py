from .expr.nodes.base import RandomVariable, as_random_variable


def exp(expr: RandomVariable | bool | float) -> RandomVariable:
    """Apply the elementwise exponential transform."""

    return as_random_variable(expr).exp()


def log(expr: RandomVariable | bool | float) -> RandomVariable:
    """Apply the elementwise natural logarithm transform."""

    return as_random_variable(expr).log()


def softplus(expr: RandomVariable | bool | float) -> RandomVariable:
    """Apply the elementwise softplus transform."""

    return as_random_variable(expr).softplus()


__all__ = ["exp", "log", "softplus"]
