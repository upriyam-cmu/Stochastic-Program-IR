import unittest
import warnings
from typing import Any, cast

import numpy as np

from stoch_ir import (
    DataType,
    ValueSupport,
    constant,
    exp,
    log,
    reductions,
    softplus,
)
from stoch_ir.errors import (
    DependencyRewriteError,
    DuplicatePlateError,
    InvalidSupportError,
    MissingPlateSizeError,
    PlateExpectationError,
    PlateSizeMismatchError,
    PossibleInvalidSupportWarning,
    UnknownPlateError,
    ValueValidationError,
)
from stoch_ir.expr.meta import PlateLayout
from stoch_ir.expr.nodes.base import (
    Constant,
    Dependency,
    as_random_variable,
)
from stoch_ir.expr.nodes.ops import (
    BinOpNode,
    ReductionOpNode,
    UnaryOpNode,
)
from stoch_ir.expr.nodes.shape import add_plates
from stoch_ir.expr.ops.binary_op import FloorDivideOp


class PlateAwareOperatorTests(unittest.TestCase):
    def test_dependency_wrapper_can_preserve_insertion_order(self) -> None:
        dependencies = Dependency.wrap(
            {"z": constant(1), "a": constant(2)},
            sort=False,
        )
        self.assertEqual(
            tuple(dependency.name for dependency in dependencies), ("z", "a")
        )

    def test_dependency_rewrite_requires_the_complete_named_mapping(self) -> None:
        expr = Constant.of(1, DataType.INT) + Constant.of(2, DataType.INT)
        rewritten = expr._rewrite_dependencies_exact(
            {
                "lhs": Constant.of(3, DataType.INT),
                "rhs": Constant.of(4, DataType.INT),
            }
        )

        self.assertEqual(rewritten.realize().data, 7)
        with self.assertRaises(DependencyRewriteError):
            expr._rewrite_dependencies_exact({"lhs": Constant.of(3, DataType.INT)})
        with self.assertRaises(DependencyRewriteError):
            expr._rewrite_dependencies_exact(
                cast(
                    Any,
                    {
                        "lhs": Constant.of(3, DataType.INT),
                        "rhs": object(),
                    },
                )
            )

    def test_binary_operator_aligns_by_canonical_plate_layout(self) -> None:
        by_row = Constant.array(
            np.array([1.0, 2.0]),
            DataType.FLOAT,
            PlateLayout.wrap(("row",)),
        )
        by_col = Constant.array(
            np.array([10.0, 20.0, 30.0]),
            DataType.FLOAT,
            PlateLayout.wrap(("col",)),
        )

        result = (by_row + by_col).realize(
            plate_sizes={"row": 2, "col": 3},
        )

        self.assertEqual(result.layout.plates, ("col", "row"))
        np.testing.assert_array_equal(
            result.data,
            np.array(
                [
                    [11.0, 12.0],
                    [21.0, 22.0],
                    [31.0, 32.0],
                ]
            ),
        )

    def test_reduction_removes_named_plate_after_alignment(self) -> None:
        value = Constant.array(
            np.arange(6.0).reshape(3, 2),
            DataType.FLOAT,
            PlateLayout.wrap(("col", "row")),
        )

        result = value.mean("col").realize(
            plate_sizes={"row": 2, "col": 3},
        )

        self.assertEqual(result.layout.plates, ("row",))
        np.testing.assert_array_equal(result.data, np.array([2.0, 3.0]))

    def test_plate_contracts_and_noop_operations(self) -> None:
        scalar = constant(1)
        plated = scalar.add_plates("row", expect=())

        self.assertIs(scalar.add_plates(), scalar)
        self.assertIs(scalar.reduce_plates(reduction=reductions.SUM), scalar)
        self.assertIs(scalar.check_plates(), scalar)
        self.assertEqual(plated.check_plates("row"), plated)
        self.assertEqual(plated.plates, ("row",))
        self.assertEqual(
            plated.add_plates("col", expect="row").plates,
            ("col", "row"),
        )
        with self.assertRaises(PlateExpectationError):
            plated.check_plates("col")
        with self.assertRaises(PlateExpectationError):
            plated.add_plates("col", expect=("batch",))
        with self.assertRaises(DuplicatePlateError):
            plated.add_plates("row")
        with self.assertRaises(UnknownPlateError):
            plated.mean("col")

    def test_constant_plate_sizes_are_validated(self) -> None:
        value = constant(np.array([1, 2]), plates=("row",))

        with self.assertRaisesRegex(MissingPlateSizeError, "missing sizes"):
            value.realize()
        with self.assertRaisesRegex(PlateSizeMismatchError, "data-size"):
            value.realize(plate_sizes={"row": 3})

    def test_low_level_plate_alignment_noop_and_missing_size_paths(self) -> None:
        scalar = np.asarray(1)
        empty = PlateLayout.wrap(())
        row = PlateLayout.wrap(("row",))

        self.assertIs(add_plates(scalar, empty, empty), scalar)
        with self.assertRaisesRegex(MissingPlateSizeError, "Missing sizes"):
            add_plates(scalar, empty, row)
        with self.assertRaisesRegex(MissingPlateSizeError, "Missing size"):
            add_plates(
                scalar,
                empty,
                row,
                added_plates=row,
                plate_sizes={},
            )

    def test_scalar_conversion_and_invalid_expression_input(self) -> None:
        self.assertIs(as_random_variable(True).value_meta.dtype, DataType.BOOL)
        self.assertIs(as_random_variable(1).value_meta.dtype, DataType.INT)
        self.assertIs(as_random_variable(1.0).value_meta.dtype, DataType.FLOAT)
        with self.assertRaises(TypeError):
            as_random_variable(cast(Any, object()))

    def test_all_binary_and_reverse_operators(self) -> None:
        cases = [
            (lambda x: x + 2, 7),
            (lambda x: 2 + x, 7),
            (lambda x: x - 2, 3),
            (lambda x: 2 - x, -3),
            (lambda x: x * 2, 10),
            (lambda x: 2 * x, 10),
            (lambda x: x / 2, 2.5),
            (lambda x: 10 / x, 2.0),
            (lambda x: x // 2, 2),
            (lambda x: 11 // x, 2),
        ]
        for operation, expected in cases:
            with self.subTest(expected=expected):
                expr = operation(constant(5))
                self.assertIsInstance(expr, BinOpNode)
                self.assertEqual(expr.realize().data, expected)

    def test_floor_division_preserves_float_metadata_when_needed(self) -> None:
        int_expr = constant(5) // 2
        float_expr = constant(5.0) // 2

        self.assertIs(int_expr.value_meta.dtype, DataType.INT)
        self.assertIs(float_expr.value_meta.dtype, DataType.FLOAT)
        self.assertEqual(FloorDivideOp().compute_value(np.array(5), np.array(2)), 2)

    def test_unary_transform_surfaces(self) -> None:
        cases = [
            (lambda x: exp(x), np.exp(2.0)),
            (lambda x: x.exp(), np.exp(2.0)),
            (lambda x: softplus(x), np.logaddexp(2.0, 0.0)),
            (lambda x: x.softplus(), np.logaddexp(2.0, 0.0)),
            (lambda x: log(x), np.log(2.0)),
            (lambda x: x.log(), np.log(2.0)),
            (lambda x: x.abs(), 2.0),
            (lambda x: abs(x), 2.0),
        ]
        for factory, expected in cases:
            with self.subTest(expected=expected):
                expr = factory(constant(2.0))
                self.assertIsInstance(expr, UnaryOpNode)
                np.testing.assert_allclose(expr.realize().data, expected)

    def test_unary_support_resolution(self) -> None:
        negative = constant(-2.0)
        unit = constant(0.5)
        positive = constant(2.0)
        real = constant(np.array([-1.0, 1.0]), plates=("row",))

        self.assertIs(negative.exp().value_meta.support, ValueSupport.UNIT_INTERVAL)
        self.assertIs(
            negative.softplus().value_meta.support, ValueSupport.UNIT_INTERVAL
        )
        self.assertIs(unit.abs().value_meta.support, ValueSupport.UNIT_INTERVAL)
        self.assertIs(positive.abs().value_meta.support, ValueSupport.POSITIVE_BRANCH)
        self.assertIs(unit.log().value_meta.support, ValueSupport.NEGATIVE_BRANCH)
        self.assertIs(positive.log().value_meta.support, ValueSupport.REAL)
        with self.assertRaises(InvalidSupportError):
            _ = negative.log()
        with self.assertWarns(UserWarning):
            self.assertIs(real.log().value_meta.support, ValueSupport.REAL)

    def test_log_zero_uses_numpy_extended_value_without_clamping(self) -> None:
        with np.errstate(divide="ignore"):
            result = constant(0.0).log().realize()

        self.assertTrue(np.isneginf(result.data))

    def test_log_possible_invalidity_warns_once(self) -> None:
        real = constant(np.array([-1.0, 1.0]), plates="row")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = real.log().realize(plate_sizes={"row": 2})

        self.assertTrue(np.isnan(result.data[0]))
        self.assertEqual(
            sum(
                issubclass(item.category, PossibleInvalidSupportWarning)
                for item in caught
            ),
            1,
        )
        self.assertFalse(
            any(
                issubclass(item.category, RuntimeWarning)
                and "invalid value" in str(item.message)
                for item in caught
            )
        )

    def test_log_accepts_zero_endpoint_without_warning(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = constant(0).log().realize()

        self.assertTrue(np.isneginf(result.data))
        self.assertEqual(caught, [])

    def test_reduction_singletons_and_convenience_methods(self) -> None:
        cases = [
            ("mean", reductions.MEAN, 2.5),
            ("sum", reductions.SUM, 10),
            ("max", reductions.MAX, 4),
            ("min", reductions.MIN, 1),
            ("prod", reductions.PROD, 24),
            ("logsumexp", reductions.LOGSUMEXP, np.log(np.exp([1, 2, 3, 4]).sum())),
        ]
        for method, implementation, expected in cases:
            with self.subTest(method=method):
                value = constant(np.array([1, 2, 3, 4]), plates=("row",))
                convenience = getattr(value, method)("row")
                generic = value.reduce_plates("row", reduction=implementation)

                self.assertIsInstance(convenience, ReductionOpNode)
                self.assertIs(convenience.op, implementation)
                self.assertTrue(convenience.structurally_equal(generic))
                np.testing.assert_allclose(
                    convenience.realize(plate_sizes={"row": 4}).data,
                    expected,
                )

    def test_reduction_metadata_branches(self) -> None:
        bools = constant(np.array([True, False]), plates=("row",))
        negative = constant(np.array([-2, -3]), plates=("row",))

        self.assertIs(bools.sum("row").value_meta.dtype, DataType.INT)
        self.assertIs(
            bools.sum("row").value_meta.support,
            ValueSupport.POSITIVE_BRANCH,
        )
        self.assertIs(
            negative.prod("row").value_meta.support,
            ValueSupport.REAL,
        )
        self.assertIs(bools.prod("row").value_meta.dtype, DataType.INT)

    def test_logsumexp_handles_extended_values(self) -> None:
        cases = [
            (np.array([-np.inf, -np.inf]), -np.inf),
            (np.array([np.inf, 0.0]), np.inf),
        ]
        for data, expected in cases:
            with self.subTest(data=data):
                value = (
                    constant(data, plates="row")
                    .logsumexp("row")
                    .realize(plate_sizes={"row": 2})
                )
                self.assertEqual(value.data, expected)

    def test_integer_operators_reject_overflow(self) -> None:
        bounds = np.iinfo(np.int64)
        cases = [
            constant(bounds.max) + 1,
            constant(bounds.min) - 1,
            constant(bounds.max) * 2,
            constant(bounds.min) // -1,
            abs(constant(bounds.min)),
            constant(np.array([bounds.max, 1]), plates="row").sum("row"),
            constant(np.array([bounds.max, 2]), plates="row").prod("row"),
        ]
        for expr in cases:
            with (
                self.subTest(expr=expr),
                self.assertRaises(ValueValidationError),
            ):
                expr.realize(plate_sizes={"row": 2})

    def test_binary_support_resolution_branches(self) -> None:
        positive = constant(2)
        negative = constant(-2)
        unit = constant(1)
        real = constant(np.array([-1, 1]), plates=("row",))

        self.assertIs(
            (real - positive).value_meta.support,
            ValueSupport.REAL,
        )
        self.assertIs(
            (positive - negative).value_meta.support,
            ValueSupport.POSITIVE_BRANCH,
        )
        self.assertIs(
            (unit * negative).value_meta.support,
            ValueSupport.NEGATIVE_BRANCH,
        )
        self.assertIs(
            (negative * unit).value_meta.support,
            ValueSupport.NEGATIVE_BRANCH,
        )
        self.assertIs(
            (positive * negative).value_meta.support,
            ValueSupport.NEGATIVE_BRANCH,
        )
        self.assertIs(
            (real / positive).value_meta.support,
            ValueSupport.REAL,
        )


if __name__ == "__main__":
    unittest.main()
