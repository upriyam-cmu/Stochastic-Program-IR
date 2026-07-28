from importlib.metadata import version
from types import MappingProxyType

import stoch_ir
from stoch_ir import (
    ConcreteValue,
    DataType,
    Normal,
    ValueMeta,
    ValueSupport,
    errors,
    exp,
    log,
    reductions,
    softplus,
)


def test_runtime_version_uses_distribution_metadata() -> None:
    assert stoch_ir.__version__ == version("stoch-ir")


def test_top_level_api_is_curated() -> None:
    assert stoch_ir.__all__ == [
        "Bernoulli",
        "ConcreteValue",
        "DataType",
        "Normal",
        "RandomVariable",
        "SamplingCheckpoint",
        "Uniform",
        "ValueMeta",
        "ValueSupport",
        "constant",
        "errors",
        "exp",
        "log",
        "reductions",
        "sampling_phase",
        "softplus",
    ]
    for retired in (
        "BernoulliDistribution",
        "Constant",
        "Gaussian",
        "PlateLayout",
        "UniformDistribution",
        "bernoulli",
        "current_sampling_phase",
        "normal",
        "uniform",
    ):
        assert not hasattr(stoch_ir, retired)


def test_public_expression_inspection_is_immutable_and_ordered() -> None:
    source = Normal(0.0, 1.0)
    expr = (source + source).add_plates("row", "col")

    assert expr.plates == ("col", "row")
    assert isinstance(expr.dependencies, MappingProxyType)
    assert tuple(expr.dependencies) == ("arg",)
    assert tuple(expr.dependencies["arg"].dependencies) == ("lhs", "rhs")
    assert expr.dependencies["arg"].dependencies["lhs"] is source
    assert expr.dependencies["arg"].dependencies["rhs"] is source


def test_free_and_fluent_transforms_are_structurally_identical() -> None:
    expr = Normal(0.0, 1.0)

    assert exp(expr) == expr.exp()
    assert log(expr.abs() + 1.0) == (expr.abs() + 1.0).log()
    assert softplus(expr) == expr.softplus()
    assert abs(expr) == expr.abs()


def test_concrete_value_exposes_public_metadata_views() -> None:
    value = stoch_ir.constant(1.0).realize()

    assert isinstance(value, ConcreteValue)
    assert value.plates == ()
    assert value.shape == ()
    assert value.dtype is DataType.FLOAT
    assert value.support is ValueSupport.UNIT_INTERVAL
    assert value.meta == ValueMeta(DataType.FLOAT, ValueSupport.UNIT_INTERVAL)


def test_reduction_singletons_share_the_public_opaque_type() -> None:
    for reduction in (
        reductions.MEAN,
        reductions.SUM,
        reductions.MAX,
        reductions.MIN,
        reductions.PROD,
        reductions.LOGSUMEXP,
    ):
        assert isinstance(reduction, reductions.Reduction)


def test_public_errors_share_one_base_class() -> None:
    public_errors = [
        getattr(errors, name) for name in errors.__all__ if name != "StochIRError"
    ]
    assert all(issubclass(error, errors.StochIRError) for error in public_errors)
