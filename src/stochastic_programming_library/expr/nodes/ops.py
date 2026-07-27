from typing_extensions import override

from ..meta import ConcreteValue, Phase, PlateLayout, PlateSizes, ValueMeta
from ..ops import BinOpImpl, ReductionImpl, UnaryOpImpl
from .base import Dependency, RandomVariable, rv_impl


@rv_impl
class BinOpNode(RandomVariable):
    op: BinOpImpl
    lhs: RandomVariable
    rhs: RandomVariable

    @override
    def _compute_dependencies(self) -> tuple[Dependency, ...]:
        return Dependency.wrap({"lhs": self.lhs, "rhs": self.rhs})

    @override
    def _compute_plate_layout(self) -> PlateLayout:
        return self.lhs.plate_layout | self.rhs.plate_layout

    @override
    def _compute_pending_phases(self) -> frozenset[Phase]:
        return self.lhs.pending_phases | self.rhs.pending_phases

    @override
    def _compute_has_value(self) -> bool:
        return self.lhs.has_value and self.rhs.has_value

    @override
    def _compute_value_meta(self) -> ValueMeta:
        return self.op.resolve_meta(self.lhs.value_meta, self.rhs.value_meta)

    @override
    def value(self, plate_sizes: PlateSizes | None = None) -> ConcreteValue:
        lhs_val = self.lhs.value(plate_sizes)
        rhs_val = self.rhs.value(plate_sizes)

        from .shape import add_plates

        lhs_arr = add_plates(
            arr=lhs_val.data,
            old_layout=lhs_val.layout,
            new_layout=self.plate_layout,
            plate_sizes=plate_sizes,
        )
        rhs_arr = add_plates(
            arr=rhs_val.data,
            old_layout=rhs_val.layout,
            new_layout=self.plate_layout,
            plate_sizes=plate_sizes,
        )
        assert lhs_arr.shape == rhs_arr.shape

        return ConcreteValue.wrap(
            data=self.op.compute_value(lhs_arr, rhs_arr),
            layout=self.plate_layout,
            meta=self.value_meta,
        )

    @override
    def structurally_equal(self, other: "RandomVariable") -> bool:
        return (
            isinstance(other, BinOpNode)
            and self.op == other.op
            and self.lhs.structurally_equal(other.lhs)
            and self.rhs.structurally_equal(other.rhs)
        )


@rv_impl
class UnaryOpNode(RandomVariable):
    op: UnaryOpImpl
    arg: RandomVariable

    @override
    def _compute_dependencies(self) -> tuple[Dependency, ...]:
        return Dependency.wrap({"arg": self.arg})

    @override
    def _compute_plate_layout(self) -> PlateLayout:
        return self.arg.plate_layout

    @override
    def _compute_pending_phases(self) -> frozenset[Phase]:
        return self.arg.pending_phases

    @override
    def _compute_has_value(self) -> bool:
        return self.arg.has_value

    @override
    def _compute_value_meta(self) -> ValueMeta:
        return self.op.resolve_meta(self.arg.value_meta)

    @override
    def value(self, plate_sizes: PlateSizes | None = None) -> ConcreteValue:
        arg_val = self.arg.value(plate_sizes)
        return ConcreteValue.wrap(
            data=self.op.compute_value(arg_val.data),
            layout=self.plate_layout,
            meta=self.value_meta,
        )

    @override
    def structurally_equal(self, other: "RandomVariable") -> bool:
        return (
            isinstance(other, UnaryOpNode)
            and self.op == other.op
            and self.arg.structurally_equal(other.arg)
        )


@rv_impl
class ReductionOpNode(RandomVariable):
    op: ReductionImpl
    arg: RandomVariable
    removed_plates: PlateLayout

    def __post_init__(self) -> None:
        assert self.removed_plates, (
            "dev-error: Should not create remove-plates node with no removed plates"
        )
        return super().__post_init__()

    @override
    def _compute_dependencies(self) -> tuple[Dependency, ...]:
        return Dependency.wrap({"arg": self.arg})

    @override
    def _compute_plate_layout(self) -> PlateLayout:
        return self.arg.plate_layout - self.removed_plates

    @override
    def _compute_pending_phases(self) -> frozenset[Phase]:
        return self.arg.pending_phases

    @override
    def _compute_has_value(self) -> bool:
        return self.arg.has_value

    @override
    def _compute_value_meta(self) -> ValueMeta:
        return self.op.resolve_meta(self.arg.value_meta)

    @override
    def value(self, plate_sizes: PlateSizes | None = None) -> ConcreteValue:
        arg_val = self.arg.value(plate_sizes)
        if not self.removed_plates.plates:
            return arg_val

        return ConcreteValue.wrap(
            data=self.op.compute_value(
                arg_val.data,
                axes=tuple(
                    self.arg.plate_layout.axis(plate)
                    for plate in self.removed_plates.plates
                ),
            ),
            layout=self.plate_layout,
            meta=self.value_meta,
        )

    @override
    def structurally_equal(self, other: "RandomVariable") -> bool:
        return (
            isinstance(other, ReductionOpNode)
            and self.op == other.op
            and self.arg.structurally_equal(other.arg)
            and self.removed_plates == other.removed_plates
        )
