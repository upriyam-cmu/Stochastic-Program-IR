import warnings
from typing import Any, cast

import numpy as np
import pytest

from stoch_ir import (
    Bernoulli,
    DataType,
    Normal,
    Uniform,
    ValueSupport,
    constant,
)
from stoch_ir.errors import (
    InvalidSupportError,
    PlateExpectationError,
    PossibleInvalidSupportWarning,
    RngLabelError,
)


def test_uniform_defaults_and_explicit_unit_bounds_match_metadata() -> None:
    default = Uniform()
    explicit = Uniform(0, 1)

    assert default.value_meta.dtype is DataType.FLOAT
    assert default.value_meta.support is ValueSupport.UNIT_INTERVAL
    assert explicit.value_meta.dtype is DataType.FLOAT
    assert explicit.value_meta.support is ValueSupport.UNIT_INTERVAL


@pytest.mark.parametrize(
    ("low", "high", "support"),
    [
        (2, 3, ValueSupport.POSITIVE_BRANCH),
        (-3, -2, ValueSupport.NEGATIVE_BRANCH),
        (-2, 3, ValueSupport.REAL),
    ],
)
def test_uniform_support_is_conservative(
    low: int,
    high: int,
    support: ValueSupport,
) -> None:
    assert Uniform(low, high).value_meta.support is support


def test_uniform_symbolic_bounds_align_named_plates() -> None:
    low = constant(np.array([0.0, 10.0]), plates=("row",))
    high = constant(np.array([1.0, 11.0]), plates=("row",))
    expr = Uniform(low, high, plates=("row", "col"))

    first = expr.realize(seed=7, plate_sizes={"row": 2, "col": 4})
    second = expr.realize(seed=7, plate_sizes={"row": 2, "col": 4})

    assert first.layout.plates == ("col", "row")
    assert first.data.shape == (4, 2)
    np.testing.assert_array_equal(first.data, second.data)
    assert np.all(first.data[:, 0] >= 0)
    assert np.all(first.data[:, 0] < 1)
    assert np.all(first.data[:, 1] >= 10)
    assert np.all(first.data[:, 1] < 11)


@pytest.mark.parametrize(("low", "high"), [(1, 1), (2, 1)])
def test_uniform_rejects_invalid_elementwise_bounds(low: int, high: int) -> None:
    with pytest.raises(InvalidSupportError, match="strictly less"):
        Uniform(low, high)


def test_uniform_validates_aligned_literal_arrays_at_construction() -> None:
    low = constant(np.array([0.0, 2.0]), plates="row")
    high = constant(np.array([1.0, 3.0]), plates="col")

    with pytest.raises(InvalidSupportError, match="strictly less"):
        Uniform(low, high)


def test_uniform_warns_when_symbolic_ordering_is_unknown() -> None:
    bound = Normal(0, 1)

    with pytest.warns(PossibleInvalidSupportWarning, match="low < high"):
        Uniform(0, bound)


def test_uniform_relaxes_symbolic_support_endpoints() -> None:
    positive = Normal(0, 1).exp()
    negative = positive * -1

    with warnings.catch_warnings():
        warnings.simplefilter("error", PossibleInvalidSupportWarning)
        Uniform(negative, positive)


def test_uniform_rejects_guaranteed_invalid_symbolic_supports() -> None:
    positive = Normal(0, 1).exp()
    negative = positive * -1

    with pytest.raises(InvalidSupportError, match="guarantee"):
        Uniform(positive, negative)


def test_uniform_rewrites_symbolic_dependencies() -> None:
    expr = Uniform(0, 1)
    rewritten = expr._rewrite_dependencies_exact(
        {"high": constant(4), "low": constant(2)}
    )

    assert rewritten.structurally_equal(Uniform(2, 4))


def test_bernoulli_has_boolean_unit_interval_output() -> None:
    expr = Bernoulli(0.5)

    assert expr.value_meta.dtype is DataType.BOOL
    assert expr.value_meta.support is ValueSupport.UNIT_INTERVAL
    value = expr.realize(seed=1)
    repeated = expr.realize(seed=1)
    assert value.data.dtype == np.dtype(np.bool_)
    np.testing.assert_array_equal(value.data, repeated.data)


def test_bernoulli_extreme_probabilities_are_exact() -> None:
    zero = Bernoulli(0, plates="trial").realize(
        seed=1,
        plate_sizes={"trial": 8},
    )
    one = Bernoulli(1, plates="trial").realize(
        seed=1,
        plate_sizes={"trial": 8},
    )

    assert not np.any(zero.data)
    assert np.all(one.data)


def test_bernoulli_plated_probabilities_and_mean_reduction() -> None:
    p = constant(np.array([0.0, 1.0]), plates=("group",))
    expr = Bernoulli(p, plates=("group", "trial"))

    rates = expr.mean("trial").realize(
        seed=5,
        plate_sizes={"group": 2, "trial": 20},
    )

    assert rates.layout.plates == ("group",)
    assert rates.data.dtype == np.dtype(np.float64)
    np.testing.assert_array_equal(rates.data, np.array([0.0, 1.0]))


@pytest.mark.parametrize("p", [-0.1, 1.1])
def test_bernoulli_rejects_invalid_probabilities(p: float) -> None:
    with pytest.raises(InvalidSupportError, match="0 <= p <= 1"):
        Bernoulli(p)


def test_bernoulli_uses_symbolic_support_for_validation() -> None:
    with pytest.warns(PossibleInvalidSupportWarning, match="0 <= p <= 1"):
        Bernoulli(Normal(0, 1))

    unit_interval = Bernoulli(0.5)
    with warnings.catch_warnings():
        warnings.simplefilter("error", PossibleInvalidSupportWarning)
        Bernoulli(unit_interval)


def test_bernoulli_rewrites_symbolic_dependency() -> None:
    expr = Bernoulli(0.2)
    rewritten = expr._rewrite_dependencies_exact({"p": constant(0.8)})

    assert rewritten.structurally_equal(Bernoulli(0.8))


def test_distribution_output_plates_must_contain_parameter_plates() -> None:
    row = constant(np.array([0.0, 1.0]), plates="row")

    assert Normal(row, 1).plates == ("row",)
    assert Normal(row, 1).structurally_equal(Normal(row, 1, plates="row"))
    with pytest.raises(PlateExpectationError, match="parameter 'mu'"):
        Normal(row, 1, plates="col")


def test_explicit_distribution_plates_are_structural() -> None:
    row = Normal(0, 1, plates="row")
    col = Normal(0, 1, plates="col")

    assert not row.structurally_equal(col)


@pytest.mark.parametrize("label", ["", 1])
def test_distribution_rng_label_must_be_nonempty_string(label: object) -> None:
    with pytest.raises(RngLabelError):
        Normal(0, 1, rng_label=cast(Any, label))


def test_gaussian_rejects_nonpositive_sigma() -> None:
    with pytest.raises(InvalidSupportError, match="strictly positive"):
        Normal(0, 0)


def test_gaussian_relaxes_symbolic_support_endpoints() -> None:
    positive = Normal(0, 1).exp()
    with warnings.catch_warnings():
        warnings.simplefilter("error", PossibleInvalidSupportWarning)
        Normal(0, positive)


def test_gaussian_warns_for_symbolic_negative_interior() -> None:
    with pytest.warns(PossibleInvalidSupportWarning, match="strict positivity"):
        Normal(0, Normal(0, 1))


def test_gaussian_rejects_guaranteed_invalid_symbolic_support() -> None:
    negative = Normal(0, 1).exp() * -1

    with pytest.raises(InvalidSupportError, match="guaranteed"):
        Normal(0, negative)


def test_gaussian_checks_nonliteral_parameter_exactly_at_runtime() -> None:
    symbolic_zero = constant(0.0) + constant(0.0)
    expr = Normal(0, symbolic_zero)

    with pytest.raises(InvalidSupportError, match="strictly positive"):
        expr.realize(seed=1)
