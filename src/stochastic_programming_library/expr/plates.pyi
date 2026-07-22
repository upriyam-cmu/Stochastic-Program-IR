from dataclasses import dataclass

from .meta import ConcreteValue, Plate, PlateLayout, PlateSizes

@dataclass(frozen=True, slots=True)
class PlateAlignment:
    source: PlateLayout
    target: PlateLayout
    permutation: tuple[int, ...]
    inserted_axes: tuple[int, ...]

@dataclass(frozen=True, slots=True)
class BinaryPlateResolution:
    output: PlateLayout
    lhs: PlateAlignment
    rhs: PlateAlignment

def normalize_plates(*plates: Plate) -> tuple[Plate, ...]: ...

def resolve_binary_plates(
    lhs: PlateLayout,
    rhs: PlateLayout,
) -> BinaryPlateResolution: ...

def align_value(
    value: ConcreteValue,
    alignment: PlateAlignment,
    plate_sizes: PlateSizes,
) -> ConcreteValue: ...

def resolve_reduction_axes(
    layout: PlateLayout,
    reduced: tuple[Plate, ...],
) -> tuple[int, ...]: ...
