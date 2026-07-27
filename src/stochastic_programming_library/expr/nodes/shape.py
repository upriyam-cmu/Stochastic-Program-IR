import numpy as np
from typing_extensions import override

from ...errors import MissingPlateSizeError
from ..meta import ConcreteValue, Phase, PlateLayout, PlateSizes, ValueMeta
from .base import Dependency, RandomVariable, rv_impl


def add_plates(
    arr: np.ndarray,
    old_layout: PlateLayout,
    new_layout: PlateLayout,
    added_plates: PlateLayout | None = None,
    plate_sizes: PlateSizes | None = None,
) -> np.ndarray:
    if added_plates is None:
        added_plates = new_layout - old_layout

    if not added_plates:
        return arr

    if plate_sizes is None:
        raise MissingPlateSizeError(f"Missing sizes for {added_plates.plates}")

    new_shape = [1] * len(new_layout)
    for plate, size in zip(old_layout, arr.shape, strict=True):
        idx = new_layout.axis(plate)
        assert new_shape[idx] == 1
        new_shape[idx] = size

    arr = arr.reshape(new_shape)
    for plate in added_plates:
        if plate not in plate_sizes:
            raise MissingPlateSizeError(f"Missing size for {plate=}")
        idx = new_layout.axis(plate)
        assert new_shape[idx] == 1
        new_shape[idx] = plate_sizes[plate]

    return np.broadcast_to(arr, new_shape)


@rv_impl
class AddPlatesNode(RandomVariable):
    arg: RandomVariable
    added_plates: PlateLayout

    def __post_init__(self) -> None:
        assert self.added_plates, (
            "dev-error: Should not create add-plates node with no new plates"
        )
        return super().__post_init__()

    @override
    def _compute_dependencies(self) -> tuple[Dependency, ...]:
        return Dependency.wrap({"arg": self.arg})

    @override
    def _compute_plate_layout(self) -> PlateLayout:
        return self.arg.plate_layout + self.added_plates

    @override
    def _compute_pending_phases(self) -> frozenset[Phase]:
        return self.arg.pending_phases

    @override
    def _compute_has_value(self) -> bool:
        return self.arg.has_value

    @override
    def _compute_value_meta(self) -> ValueMeta:
        return self.arg.value_meta

    @override
    def value(self, plate_sizes: PlateSizes | None = None) -> ConcreteValue:
        arg_val = self.arg.value(plate_sizes)
        if not self.added_plates.plates:
            return arg_val

        return ConcreteValue.wrap(
            data=add_plates(
                arr=arg_val.data,
                old_layout=arg_val.layout,
                new_layout=self.plate_layout,
                added_plates=self.added_plates,
                plate_sizes=plate_sizes,
            ),
            layout=self.plate_layout,
            meta=self.value_meta,
        )

    @override
    def structurally_equal(self, other: "RandomVariable") -> bool:
        return (
            isinstance(other, AddPlatesNode)
            and self.arg.structurally_equal(other.arg)
            and self.added_plates == other.added_plates
        )
