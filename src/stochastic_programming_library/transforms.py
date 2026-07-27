from .expr.nodes.base import ExprInput, RandomVariable, as_random_variable


def exp(expr: ExprInput) -> RandomVariable:
    return as_random_variable(expr).exp()


def log(expr: ExprInput) -> RandomVariable:
    return as_random_variable(expr).log()


def softplus(expr: ExprInput) -> RandomVariable:
    return as_random_variable(expr).softplus()


__all__ = ["exp", "log", "softplus"]
