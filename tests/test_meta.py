import unittest
from typing import Any, cast

import numpy as np

from stoch_ir import constant
from stoch_ir.errors import (
    DuplicatePlateError,
    UnknownPlateError,
    ValueValidationError,
)
from stoch_ir.expr.meta import (
    EMPTY_PLATE_LAYOUT,
    ConcreteValue,
    DataType,
    PlateLayout,
    ValueMeta,
    ValueSupport,
)


class PlateLayoutTests(unittest.TestCase):
    def test_wrap_canonicalizes_plate_order(self) -> None:
        layout = PlateLayout.wrap(("row", "batch"))

        self.assertEqual(layout.plates, ("batch", "row"))
        self.assertEqual(layout.axis("batch"), 0)
        self.assertEqual(layout.axis("row"), 1)
        self.assertEqual(PlateLayout.wrap("row").plates, ("row",))

    def test_wrap_rejects_duplicate_plates(self) -> None:
        with self.assertRaises(DuplicatePlateError):
            PlateLayout.wrap(("row", "row"))

    def test_rejects_invalid_plate_names(self) -> None:
        with self.assertRaises(ValueError):
            PlateLayout.wrap(("",))
        with self.assertRaises(ValueError):
            PlateLayout.wrap(cast(Any, (1,)))

    def test_set_like_operations_are_canonical_and_validate_removal(self) -> None:
        row = PlateLayout.wrap(("row",))
        col = PlateLayout.wrap(("col",))

        self.assertEqual((row | col).plates, ("col", "row"))
        self.assertEqual((row + col).plates, ("col", "row"))
        self.assertEqual(row.__or__(object()), NotImplemented)
        self.assertEqual(row.__add__(object()), NotImplemented)
        self.assertEqual(row.__sub__(object()), NotImplemented)
        self.assertEqual(row.__ror__(object()), NotImplemented)
        self.assertEqual(row.__radd__(object()), NotImplemented)
        self.assertEqual(row.__rsub__(object()), NotImplemented)
        self.assertEqual(row.__ror__(col), col | row)
        self.assertEqual(row.__radd__(col), col + row)
        self.assertEqual(row.__rsub__(row | col), col)
        self.assertEqual(PlateLayout.union(row, col), row | col)
        self.assertEqual((row | col).without("col"), row)
        with self.assertRaises(UnknownPlateError):
            row.axis("col")
        with self.assertRaises(UnknownPlateError):
            row.without("col")


class ConcreteValueTests(unittest.TestCase):
    def test_wrap_owns_an_immutable_array(self) -> None:
        source = np.array([1.0, 2.0])
        value = ConcreteValue.wrap(
            source,
            PlateLayout.wrap(("row",)),
            ValueMeta.from_value(source, DataType.FLOAT),
        )

        source[0] = 99.0
        self.assertEqual(value.data[0], 1.0)
        self.assertFalse(value.data.flags.writeable)
        with self.assertRaises(ValueError):
            value.data[0] = 10.0

    def test_wrap_rejects_rank_layout_mismatch(self) -> None:
        source = np.array([1.0, 2.0])

        with self.assertRaises(ValueValidationError):
            ConcreteValue.wrap(
                source,
                PlateLayout.wrap(()),
                ValueMeta.from_value(source, DataType.FLOAT),
            )

    def test_dtype_metadata_controls_canonical_storage(self) -> None:
        cases = [
            (DataType.BOOL, np.array([0, 1]), np.dtype(np.bool_)),
            (DataType.INT, np.array([1, 2], dtype=np.int8), np.dtype(np.int64)),
            (
                DataType.FLOAT,
                np.array([1, 2], dtype=np.float32),
                np.dtype(np.float64),
            ),
        ]
        for dtype, value, numpy_dtype in cases:
            with self.subTest(dtype=dtype):
                concrete = ConcreteValue.wrap(
                    cast(Any, value),
                    PlateLayout.wrap(("row",)),
                    ValueMeta.from_value(cast(Any, value), dtype),
                )
                self.assertEqual(concrete.data.dtype, numpy_dtype)

    def test_dtype_inference(self) -> None:
        cases = [
            (True, DataType.BOOL),
            (np.uint8(2), DataType.INT),
            (1, DataType.INT),
            (np.float32(1.5), DataType.FLOAT),
        ]
        for value, expected in cases:
            with self.subTest(value=value):
                self.assertIs(DataType.infer(cast(Any, value)), expected)

    def test_unsupported_concrete_dtypes_are_rejected(self) -> None:
        values = [
            np.array([1 + 2j]),
            np.array(["one"]),
            np.array([object()], dtype=object),
        ]
        for value in values:
            with self.subTest(dtype=value.dtype):
                with self.assertRaises(ValueValidationError):
                    DataType.infer(value)
                with self.assertRaises(ValueValidationError):
                    DataType.FLOAT.coerce(value)

    def test_boolean_coercion_rejects_values_other_than_zero_or_one(self) -> None:
        with self.assertRaises(ValueValidationError):
            DataType.BOOL.coerce(np.array([0, 2]))

    def test_declared_support_is_checked_after_coercion(self) -> None:
        with self.assertRaises(ValueValidationError):
            ConcreteValue.wrap(
                -1,
                EMPTY_PLATE_LAYOUT,
                ValueMeta(DataType.INT, ValueSupport.POSITIVE_BRANCH),
            )
        value = ConcreteValue.wrap(
            -1,
            EMPTY_PLATE_LAYOUT,
            ValueMeta(DataType.INT, ValueSupport.REAL),
        )
        self.assertEqual(value.data, -1)
        self.assertTrue(ValueSupport.REAL.contains(value.data))

    def test_support_is_derived_from_boundary_value(self) -> None:
        cases = [
            (0, ValueSupport.UNIT_INTERVAL),
            (1, ValueSupport.UNIT_INTERVAL),
            (2, ValueSupport.POSITIVE_BRANCH),
            (-1, ValueSupport.NEGATIVE_BRANCH),
            (np.array([-1, 1]), ValueSupport.REAL),
        ]
        for value, expected in cases:
            with self.subTest(value=value):
                dtype = DataType.infer(value)
                self.assertIs(ValueMeta.from_value(value, dtype).support, expected)

    def test_constant_boundary_infers_dtype_and_named_layout(self) -> None:
        scalar = constant(3)
        vector = constant(np.array([1, 2], dtype=np.int8), plates="row")
        explicit = constant(1, dtype=DataType.FLOAT)

        self.assertIs(scalar.value_meta.dtype, DataType.INT)
        self.assertEqual(vector.plate_layout.plates, ("row",))
        self.assertEqual(vector.realize(plate_sizes={"row": 2}).data.dtype, np.int64)
        self.assertEqual(explicit.realize().data.dtype, np.float64)

    def test_constant_transposes_declared_axes_into_canonical_order(self) -> None:
        source = np.arange(6).reshape(2, 3)
        value = constant(source, plates=("row", "inner")).realize(
            plate_sizes={"row": 2, "inner": 3},
        )

        self.assertEqual(value.plates, ("inner", "row"))
        self.assertEqual(value.shape, (3, 2))
        np.testing.assert_array_equal(value.data, source.T)

    def test_multidimensional_constant_requires_named_plates(self) -> None:
        with self.assertRaises(ValueValidationError):
            constant(np.ones((2, 2)))

    def test_constant_boundary_rejects_unsupported_kinds(self) -> None:
        with self.assertRaises(ValueValidationError):
            constant(np.array(["value"]))

    def test_integer_coercion_rejects_lossy_values(self) -> None:
        for value in (
            np.uint64(2**64 - 1),
            np.array([np.uint64(2**64 - 1)]),
            2**100,
            1.5,
            np.inf,
        ):
            with (
                self.subTest(value=value),
                self.assertRaises(ValueValidationError),
            ):
                constant(cast(Any, value), dtype=DataType.INT)

    def test_concrete_equality_rejects_other_objects_and_detects_metadata(self) -> None:
        left = constant(1).realize()
        same = constant(1).realize()
        floating = constant(1.0).realize()

        self.assertEqual(left, same)
        self.assertNotEqual(left, floating)
        self.assertNotEqual(left, object())


if __name__ == "__main__":
    unittest.main()
