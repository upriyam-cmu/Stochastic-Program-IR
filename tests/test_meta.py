import unittest

import numpy as np

from stochastic_programming_library.errors import BackendError, DuplicatePlateError
from stochastic_programming_library.expr.meta import (
    ConcreteValue,
    DataType,
    PlateLayout,
    ValueMeta,
)


class PlateLayoutTests(unittest.TestCase):
    def test_wrap_canonicalizes_plate_order(self) -> None:
        layout = PlateLayout.wrap(("row", "batch"))

        self.assertEqual(layout.plates, ("batch", "row"))
        self.assertEqual(layout.axis("batch"), 0)
        self.assertEqual(layout.axis("row"), 1)

    def test_wrap_rejects_duplicate_plates(self) -> None:
        with self.assertRaises(DuplicatePlateError):
            PlateLayout.wrap(("row", "row"))


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

    def test_wrap_rejects_rank_layout_mismatch(self) -> None:
        source = np.array([1.0, 2.0])

        with self.assertRaises(BackendError):
            ConcreteValue.wrap(
                source,
                PlateLayout.wrap(()),
                ValueMeta.from_value(source, DataType.FLOAT),
            )


if __name__ == "__main__":
    unittest.main()
