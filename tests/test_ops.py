import unittest

import numpy as np

from stochastic_programming_library import Constant, DataType, PlateLayout


class PlateAwareOperatorTests(unittest.TestCase):
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

        result = (by_row + by_col).value(
            {"row": 2, "col": 3},
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

        result = value.mean("col").value(
            {"row": 2, "col": 3},
        )

        self.assertEqual(result.layout.plates, ("row",))
        np.testing.assert_array_equal(result.data, np.array([2.0, 3.0]))


if __name__ == "__main__":
    unittest.main()
