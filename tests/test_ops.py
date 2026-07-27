import unittest

import numpy as np

from stochastic_programming_library import Constant, DataType, PlateLayout
from stochastic_programming_library.errors import DependencyRewriteError


class PlateAwareOperatorTests(unittest.TestCase):
    def test_dependency_rewrite_requires_the_complete_named_mapping(self) -> None:
        expr = Constant.of(1, DataType.INT) + Constant.of(2, DataType.INT)
        rewritten = expr.rewrite_dependencies(
            {
                "lhs": Constant.of(3, DataType.INT),
                "rhs": Constant.of(4, DataType.INT),
            }
        )

        self.assertEqual(rewritten.realize().data, 7)
        with self.assertRaises(DependencyRewriteError):
            expr.rewrite_dependencies({"lhs": Constant.of(3, DataType.INT)})

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


if __name__ == "__main__":
    unittest.main()
