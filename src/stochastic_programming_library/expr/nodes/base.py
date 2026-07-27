from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cached_property
from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Iterable,
    TypeAlias,
    TypeVar,
    cast,
    overload,
)

import numpy as np
from typing_extensions import Self, dataclass_transform, override

from ...errors import (
    DependencyRewriteError,
    MissingPlateSizeError,
    PlateExpectationError,
    PlateSizeMismatchError,
)
from ...rng import Seed
from ..meta import (
    EMPTY_PLATE_LAYOUT,
    ConcreteValue,
    Data,
    DataType,
    Phase,
    Plate,
    PlateLayout,
    PlateSizes,
    Scalar,
    ValueMeta,
)
from ..ops import BinOpImpl, ReductionImpl, UnaryOpImpl
from ..ops.binary_op import AddOp, FloorDivideOp, MultiplyOp, SubtractOp, TrueDivideOp
from ..ops.reduction import (
    LogSumExpReduction,
    MaxReduction,
    MeanReduction,
    MinReduction,
    ProductReduction,
    SumReduction,
)
from ..ops.unary_op import AbsOp, ExpOp, LogOp, SoftplusOp

if TYPE_CHECKING:
    from ..materialize import SamplingCheckpoint


@dataclass(frozen=True, slots=True)
class Dependency:
    name: str
    var: "RandomVariable"

    @staticmethod
    def wrap(
        var_map: Mapping[str, "RandomVariable"],
        *,
        sort: bool = True,
    ) -> tuple["Dependency", ...]:
        if sort:
            return tuple(Dependency(name, var_map[name]) for name in sorted(var_map))
        else:
            return tuple(Dependency(name, var) for name, var in var_map.items())


_RV = TypeVar("_RV", bound="RandomVariable")


@dataclass_transform(frozen_default=True, eq_default=False)
@overload
def rv_impl(
    cls: None = None,
    /,
) -> Callable[[type[_RV]], type[_RV]]: ...


@dataclass_transform(frozen_default=True, eq_default=False)
@overload
def rv_impl(cls: type[_RV], /) -> type[_RV]: ...


@dataclass_transform(frozen_default=True, eq_default=False)
def rv_impl(
    cls: type[_RV] | None = None,
    /,
) -> Callable[[type[_RV]], type[_RV]] | type[_RV]:
    def decorate(target: type[_RV]) -> type[_RV]:
        dataclass_decorator = dataclass(frozen=True, eq=False)
        return cast(
            type[_RV],
            dataclass_decorator(cast(type[Any], target)),
        )

    return decorate if cls is None else decorate(cls)


@rv_impl
class RandomVariable(ABC):
    def __post_init__(self) -> None:
        # Resolve cached structural metadata eagerly so invalid support,
        # dependency, or plate interactions fail at node construction rather
        # than during a later materialization pass.
        _ = self.dependencies
        _ = self.plate_layout
        _ = self.pending_phases
        _ = self.has_value
        _ = self.value_meta

    @abstractmethod
    def _compute_dependencies(self) -> tuple[Dependency, ...]: ...

    @cached_property
    def dependencies(self) -> tuple[Dependency, ...]:
        """Return the node's complete dependency slots in canonical name order."""

        dependencies = self._compute_dependencies()
        names = tuple(dependency.name for dependency in dependencies)
        if any(not name for name in names):
            raise DependencyRewriteError(
                f"{type(self).__name__} has an empty dependency name"
            )
        if len(names) != len(set(names)):
            raise DependencyRewriteError(
                f"{type(self).__name__} has duplicate dependency names: {names}"
            )
        return tuple(sorted(dependencies, key=lambda dependency: dependency.name))

    @abstractmethod
    def _rewrite_dependencies(
        self,
        dependencies: Mapping[str, "RandomVariable"],
    ) -> Self: ...

    def rewrite_dependencies(
        self,
        dependencies: Mapping[str, "RandomVariable"],
    ) -> Self:
        """Rebuild this node with one replacement for every dependency slot.

        The public wrapper owns the generic rewrite contract; subclasses only
        map the validated dependency names back to their constructor fields.
        """

        expected_names = tuple(dependency.name for dependency in self.dependencies)
        supplied_names = tuple(sorted(dependencies))
        if supplied_names != expected_names:
            raise DependencyRewriteError(
                f"{type(self).__name__} expected dependency names "
                f"{expected_names}, got {supplied_names}"
            )
        if any(
            not isinstance(dependency, RandomVariable)
            for dependency in dependencies.values()
        ):
            raise DependencyRewriteError(
                "rewritten dependencies must all be RandomVariable instances"
            )

        rewritten = self._rewrite_dependencies(MappingProxyType(dict(dependencies)))
        if type(rewritten) is not type(self):
            raise DependencyRewriteError(
                f"{type(self).__name__} dependency rewrite returned "
                f"{type(rewritten).__name__}"
            )
        rewritten_names = tuple(
            dependency.name for dependency in rewritten.dependencies
        )
        if rewritten_names != expected_names:
            raise DependencyRewriteError(
                f"{type(self).__name__} dependency rewrite changed its slots "
                f"from {expected_names} to {rewritten_names}"
            )
        return rewritten

    @abstractmethod
    def _compute_plate_layout(self) -> PlateLayout: ...
    @cached_property
    def plate_layout(self) -> PlateLayout:
        return self._compute_plate_layout()

    @property
    def plates(self) -> frozenset[Plate]:
        return self.plate_layout.as_set

    @abstractmethod
    def _compute_pending_phases(self) -> frozenset[Phase]: ...
    @cached_property
    def pending_phases(self) -> frozenset[Phase]:
        return self._compute_pending_phases()

    @abstractmethod
    def _compute_has_value(self) -> bool: ...
    @cached_property
    def has_value(self) -> bool:
        return self._compute_has_value()

    @abstractmethod
    def _compute_value_meta(self) -> ValueMeta: ...
    @cached_property
    def value_meta(self) -> ValueMeta:
        return self._compute_value_meta()

    @abstractmethod
    def _evaluate_concrete(
        self,
        dependencies: Mapping[str, ConcreteValue],
        plate_sizes: PlateSizes,
    ) -> ConcreteValue:
        """Evaluate this node from already concrete dependency values."""
        ...

    def add_plates(
        self,
        *plates: Plate,
        expect: Iterable[Plate] | None = None,
    ) -> "RandomVariable":
        if expect is not None:
            self.check_plates(*expect)

        from .shape import AddPlatesNode

        return AddPlatesNode(self, PlateLayout.wrap(plates)) if plates else self

    def check_plates(self, *plates: Plate) -> "RandomVariable":
        expected = PlateLayout.wrap(plates)
        if self.plate_layout != expected:
            raise PlateExpectationError(
                f"Expected {expected.plates}, got {self.plate_layout.plates}"
            )
        return self

    def reduce_plates(
        self,
        *plates: Plate,
        reduction: ReductionImpl,
    ) -> "RandomVariable":
        from .ops import ReductionOpNode

        return (
            ReductionOpNode(reduction, self, PlateLayout.wrap(plates))
            if plates
            else self
        )

    def mean(self, *plates: Plate) -> "RandomVariable":
        return self.reduce_plates(*plates, reduction=MeanReduction())

    def sum(self, *plates: Plate) -> "RandomVariable":
        return self.reduce_plates(*plates, reduction=SumReduction())

    def max(self, *plates: Plate) -> "RandomVariable":
        return self.reduce_plates(*plates, reduction=MaxReduction())

    def min(self, *plates: Plate) -> "RandomVariable":
        return self.reduce_plates(*plates, reduction=MinReduction())

    def prod(self, *plates: Plate) -> "RandomVariable":
        return self.reduce_plates(*plates, reduction=ProductReduction())

    def logsumexp(self, *plates: Plate) -> "RandomVariable":
        return self.reduce_plates(*plates, reduction=LogSumExpReduction())

    def apply_unary_op(self, op: UnaryOpImpl) -> "RandomVariable":
        from .ops import UnaryOpNode

        return UnaryOpNode(op, self)

    def exp(self) -> "RandomVariable":
        return self.apply_unary_op(ExpOp())

    def log(self) -> "RandomVariable":
        return self.apply_unary_op(LogOp())

    def softplus(self) -> "RandomVariable":
        return self.apply_unary_op(SoftplusOp())

    def abs(self) -> "RandomVariable":
        return self.apply_unary_op(AbsOp())

    def apply_binary_op(
        self,
        other: "RandomVariable",
        op: BinOpImpl,
    ) -> "RandomVariable":
        from .ops import BinOpNode

        return BinOpNode(op, self, other)

    def __add__(self, other: "ExprInput") -> "RandomVariable":
        return self.apply_binary_op(as_random_variable(other), AddOp())

    def __radd__(self, other: "ExprInput") -> "RandomVariable":
        return as_random_variable(other).apply_binary_op(self, AddOp())

    def __sub__(self, other: "ExprInput") -> "RandomVariable":
        return self.apply_binary_op(as_random_variable(other), SubtractOp())

    def __rsub__(self, other: "ExprInput") -> "RandomVariable":
        return as_random_variable(other).apply_binary_op(self, SubtractOp())

    def __mul__(self, other: "ExprInput") -> "RandomVariable":
        return self.apply_binary_op(as_random_variable(other), MultiplyOp())

    def __rmul__(self, other: "ExprInput") -> "RandomVariable":
        return as_random_variable(other).apply_binary_op(self, MultiplyOp())

    def __truediv__(self, other: "ExprInput") -> "RandomVariable":
        return self.apply_binary_op(as_random_variable(other), TrueDivideOp())

    def __rtruediv__(self, other: "ExprInput") -> "RandomVariable":
        return as_random_variable(other).apply_binary_op(self, TrueDivideOp())

    def __floordiv__(self, other: "ExprInput") -> "RandomVariable":
        return self.apply_binary_op(as_random_variable(other), FloorDivideOp())

    def __rfloordiv__(self, other: "ExprInput") -> "RandomVariable":
        return as_random_variable(other).apply_binary_op(self, FloorDivideOp())

    def materialize(
        self,
        *,
        seed: Seed | None = None,
        plate_sizes: PlateSizes | None = None,
        phases: Iterable[Phase] | None = None,
    ) -> "SamplingCheckpoint":
        from ..materialize import materialize

        return materialize(
            self,
            seed=seed,
            plate_sizes=plate_sizes,
            phases=phases,
        )

    def realize(
        self,
        *,
        seed: Seed | None = None,
        plate_sizes: PlateSizes | None = None,
    ) -> ConcreteValue:
        from ..materialize import realize

        return realize(self, seed=seed, plate_sizes=plate_sizes)

    @abstractmethod
    def structurally_equal(self, other: "RandomVariable") -> bool: ...
    def __eq__(self, other: object) -> bool:
        return isinstance(other, RandomVariable) and self.structurally_equal(other)


@rv_impl
class Constant(RandomVariable):
    val: ConcreteValue

    @staticmethod
    def of(value: Data, dtype: DataType) -> "Constant":
        return Constant(
            ConcreteValue.wrap(
                data=value,
                layout=EMPTY_PLATE_LAYOUT,
                meta=ValueMeta.from_value(value, dtype),
            )
        )

    @staticmethod
    def array(arr: np.ndarray, dtype: DataType, layout: PlateLayout) -> "Constant":
        return Constant(
            ConcreteValue.wrap(
                data=arr,
                layout=layout,
                meta=ValueMeta.from_value(arr, dtype),
            )
        )

    @override
    def _compute_dependencies(self) -> tuple[Dependency, ...]:
        return ()

    @override
    def _rewrite_dependencies(
        self,
        dependencies: Mapping[str, RandomVariable],
    ) -> Self:
        return self

    @override
    def _compute_plate_layout(self) -> PlateLayout:
        return self.val.layout

    @override
    def _compute_pending_phases(self) -> frozenset[Phase]:
        return frozenset()

    @override
    def _compute_has_value(self) -> bool:
        return True

    @override
    def _compute_value_meta(self) -> ValueMeta:
        return self.val.meta

    @override
    def _evaluate_concrete(
        self,
        dependencies: Mapping[str, ConcreteValue],
        plate_sizes: PlateSizes,
    ) -> ConcreteValue:
        if self.val.shape:
            for plate, size in zip(self.val.layout, self.val.shape, strict=True):
                if plate not in plate_sizes:
                    raise MissingPlateSizeError(f"Missing size for {plate=}")
                if plate_sizes[plate] != size:
                    raise PlateSizeMismatchError(
                        f"Expected plate-size={plate_sizes[plate]} for {plate=}, got data-size={size}"
                    )
        return self.val

    @override
    def structurally_equal(self, other: "RandomVariable") -> bool:
        return isinstance(other, Constant) and self.val == other.val


ExprInput: TypeAlias = RandomVariable | Scalar


def as_random_variable(expr: ExprInput) -> RandomVariable:
    if isinstance(expr, RandomVariable):
        return expr
    if isinstance(expr, bool):
        return Constant.of(expr, dtype=DataType.BOOL)
    if isinstance(expr, int):
        return Constant.of(expr, dtype=DataType.INT)
    if isinstance(expr, float):
        return Constant.of(expr, dtype=DataType.FLOAT)
    raise TypeError(f"expected a random variable or scalar, got {type(expr)!r}")
