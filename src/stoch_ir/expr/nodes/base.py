from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from functools import cached_property
from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
    Any,
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
    ValueValidationError,
)
from ...rng import Seed
from ..meta import (
    ConcreteValue,
    Data,
    DataType,
    Plate,
    PlateLayout,
    PlateSizes,
    Scalar,
    ValueMeta,
)
from ..ops import BinOpImpl, Reduction, UnaryOpImpl
from ..ops.binary_op import AddOp, FloorDivideOp, MultiplyOp, SubtractOp, TrueDivideOp
from ..ops.reduction import (
    LOGSUMEXP,
    MAX,
    MEAN,
    MIN,
    PROD,
    SUM,
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
    """Immutable symbolic random-variable expression.

    Random variables form a directed acyclic graph. Deterministic operations
    return new expressions, while distributions remain symbolic until
    :meth:`materialize` or :meth:`realize` is called.

    Notes
    -----
    This class is a public inspection and authoring type, not a supported v0.1
    subclassing interface.
    """

    def __post_init__(self) -> None:
        # Resolve cached structural metadata eagerly so invalid support,
        # dependency, or plate interactions fail at node construction rather
        # than during a later materialization pass.
        _ = self._dependency_slots
        _ = self.plate_layout
        _ = self.pending_phases
        _ = self.has_value
        _ = self.value_meta

    @abstractmethod
    def _compute_dependencies(self) -> tuple[Dependency, ...]: ...

    @cached_property
    def _dependency_slots(self) -> tuple[Dependency, ...]:
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

    @cached_property
    def dependencies(self) -> Mapping[str, "RandomVariable"]:
        """Return an immutable, name-sorted mapping of direct dependencies."""

        return MappingProxyType(
            {dependency.name: dependency.var for dependency in self._dependency_slots}
        )

    @abstractmethod
    def _rewrite_dependencies(
        self,
        dependencies: Mapping[str, "RandomVariable"],
    ) -> Self: ...

    def _rewrite_dependencies_exact(
        self,
        dependencies: Mapping[str, "RandomVariable"],
    ) -> Self:
        """Internal exact reconstruction hook used by immutable graph passes."""

        expected_names = tuple(dependency.name for dependency in self._dependency_slots)
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
            dependency.name for dependency in rewritten._dependency_slots
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
    def plates(self) -> tuple[Plate, ...]:
        """Return named plates in canonical lexicographic order."""

        return self.plate_layout.plates

    @abstractmethod
    def _compute_pending_phases(self) -> frozenset[str]: ...

    @cached_property
    def pending_phases(self) -> frozenset[str]:
        """Return named sampling phases still present in this graph."""

        return self._compute_pending_phases()

    @abstractmethod
    def _compute_has_value(self) -> bool: ...
    @cached_property
    def has_value(self) -> bool:
        """Whether the graph is deterministic and directly realizable."""

        return self._compute_has_value()

    @abstractmethod
    def _compute_value_meta(self) -> ValueMeta: ...
    @cached_property
    def value_meta(self) -> ValueMeta:
        """Return the expression's inferred dtype and support metadata."""

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
        """Broadcast the existing value over new named plates.

        Parameters
        ----------
        *plates
            New plate names. Every name must be unique and absent from the
            expression.
        expect
            Optional exact precondition for the expression's existing plates.

        Returns
        -------
        RandomVariable
            A new expression with the requested plates.
        """

        if expect is not None:
            expected = PlateLayout.wrap(expect)
            if self.plate_layout != expected:
                raise PlateExpectationError(
                    f"Expected {expected.plates}, got {self.plate_layout.plates}"
                )

        from .shape import AddPlatesNode

        return AddPlatesNode(self, PlateLayout.wrap(plates)) if plates else self

    def check_plates(self, *plates: Plate) -> "RandomVariable":
        """Validate the complete current plate set and return this expression."""

        expected = PlateLayout.wrap(plates)
        if self.plate_layout != expected:
            raise PlateExpectationError(
                f"Expected {expected.plates}, got {self.plate_layout.plates}"
            )
        return self

    def reduce_plates(
        self,
        *plates: Plate,
        reduction: Reduction,
    ) -> "RandomVariable":
        """Reduce named plates with a canonical reduction object."""

        from .ops import ReductionOpNode

        return (
            ReductionOpNode(reduction, self, PlateLayout.wrap(plates))
            if plates
            else self
        )

    def mean(self, *plates: Plate) -> "RandomVariable":
        """Return the arithmetic mean over named plates."""

        return self.reduce_plates(*plates, reduction=MEAN)

    def sum(self, *plates: Plate) -> "RandomVariable":
        """Return the sum over named plates."""

        return self.reduce_plates(*plates, reduction=SUM)

    def max(self, *plates: Plate) -> "RandomVariable":
        """Return the maximum over named plates."""

        return self.reduce_plates(*plates, reduction=MAX)

    def min(self, *plates: Plate) -> "RandomVariable":
        """Return the minimum over named plates."""

        return self.reduce_plates(*plates, reduction=MIN)

    def prod(self, *plates: Plate) -> "RandomVariable":
        """Return the product over named plates."""

        return self.reduce_plates(*plates, reduction=PROD)

    def logsumexp(self, *plates: Plate) -> "RandomVariable":
        """Return a stable log-sum-exp over named plates."""

        return self.reduce_plates(*plates, reduction=LOGSUMEXP)

    def _apply_unary_op(self, op: UnaryOpImpl) -> "RandomVariable":
        from .ops import UnaryOpNode

        return UnaryOpNode(op, self)

    def exp(self) -> "RandomVariable":
        """Apply the elementwise exponential transform."""

        return self._apply_unary_op(ExpOp())

    def log(self) -> "RandomVariable":
        """Apply the elementwise natural logarithm transform."""

        return self._apply_unary_op(LogOp())

    def softplus(self) -> "RandomVariable":
        """Apply the elementwise softplus transform."""

        return self._apply_unary_op(SoftplusOp())

    def abs(self) -> "RandomVariable":
        """Apply the elementwise absolute-value transform."""

        return self._apply_unary_op(AbsOp())

    def __abs__(self) -> "RandomVariable":
        """Return ``self.abs()``."""

        return self.abs()

    def _apply_binary_op(
        self,
        other: "RandomVariable",
        op: BinOpImpl,
    ) -> "RandomVariable":
        from .ops import BinOpNode

        return BinOpNode(op, self, other)

    def __add__(self, other: "ExprInput") -> "RandomVariable":
        return self._apply_binary_op(as_random_variable(other), AddOp())

    def __radd__(self, other: "ExprInput") -> "RandomVariable":
        return as_random_variable(other)._apply_binary_op(self, AddOp())

    def __sub__(self, other: "ExprInput") -> "RandomVariable":
        return self._apply_binary_op(as_random_variable(other), SubtractOp())

    def __rsub__(self, other: "ExprInput") -> "RandomVariable":
        return as_random_variable(other)._apply_binary_op(self, SubtractOp())

    def __mul__(self, other: "ExprInput") -> "RandomVariable":
        return self._apply_binary_op(as_random_variable(other), MultiplyOp())

    def __rmul__(self, other: "ExprInput") -> "RandomVariable":
        return as_random_variable(other)._apply_binary_op(self, MultiplyOp())

    def __truediv__(self, other: "ExprInput") -> "RandomVariable":
        return self._apply_binary_op(as_random_variable(other), TrueDivideOp())

    def __rtruediv__(self, other: "ExprInput") -> "RandomVariable":
        return as_random_variable(other)._apply_binary_op(self, TrueDivideOp())

    def __floordiv__(self, other: "ExprInput") -> "RandomVariable":
        return self._apply_binary_op(as_random_variable(other), FloorDivideOp())

    def __rfloordiv__(self, other: "ExprInput") -> "RandomVariable":
        return as_random_variable(other)._apply_binary_op(self, FloorDivideOp())

    def materialize(
        self,
        *,
        seed: Seed | None = None,
        plate_sizes: PlateSizes | None = None,
        phases: Iterable[str] | None = None,
    ) -> "SamplingCheckpoint":
        """Partially materialize selected phases into an immutable checkpoint.

        Parameters
        ----------
        seed
            Run seed mixed with each graph-derived stochastic node entropy.
            ``None`` selects a fresh run seed.
        plate_sizes
            Positive concrete sizes for every named plate in the graph.
        phases
            Phases to enable. ``None`` enables all remaining phases.
        """

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
        """Fully materialize the graph and return its concrete value."""

        from ..materialize import realize

        return realize(self, seed=seed, plate_sizes=plate_sizes)

    @abstractmethod
    def _structurally_equal_shallow(self, other: "RandomVariable") -> bool:
        """Compare node-local structure without descending into dependencies."""

        ...

    def structurally_equal(self, other: "RandomVariable") -> bool:
        """Compare exact computation structure with memoized DAG traversal."""

        if not isinstance(other, RandomVariable):
            return False

        pending: list[tuple[RandomVariable, RandomVariable]] = [(self, other)]
        compared_pairs: set[tuple[int, int]] = set()
        while pending:
            left, right = pending.pop()
            pair = (id(left), id(right))
            if pair in compared_pairs:
                continue
            compared_pairs.add(pair)

            if not left._structurally_equal_shallow(right):
                return False
            left_dependencies = left._dependency_slots
            right_dependencies = right._dependency_slots
            if len(left_dependencies) != len(right_dependencies):
                return False
            for left_dependency, right_dependency in zip(
                left_dependencies,
                right_dependencies,
                strict=True,
            ):
                if left_dependency.name != right_dependency.name:
                    return False
                pending.append((left_dependency.var, right_dependency.var))
        return True

    def __eq__(self, other: object) -> bool:
        return isinstance(other, RandomVariable) and self.structurally_equal(other)


@rv_impl
class Constant(RandomVariable):
    val: ConcreteValue

    @staticmethod
    def of(value: Data, dtype: DataType) -> "Constant":
        return cast(Constant, constant(value, dtype=dtype))

    @staticmethod
    def array(arr: np.ndarray, dtype: DataType, layout: PlateLayout) -> "Constant":
        return cast(Constant, constant(arr, plates=layout, dtype=dtype))

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
    def _compute_pending_phases(self) -> frozenset[str]:
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
    def _structurally_equal_shallow(self, other: "RandomVariable") -> bool:
        return isinstance(other, Constant) and self.val == other.val


ExprInput: TypeAlias = RandomVariable | Scalar


def constant(
    value: bool | float | np.generic | np.ndarray,
    *,
    plates: Iterable[Plate] = (),
    dtype: DataType | None = None,
) -> RandomVariable:
    """Create an immutable concrete expression.

    Parameters
    ----------
    value
        Boolean, integer, floating scalar, NumPy scalar, or NumPy array.
    plates
        Named axes for non-scalar values in input array-axis order. The number
        of plates must equal the array rank. A bare string denotes one plate.
    dtype
        Optional canonical dtype. When omitted, it is inferred from ``value``.

    Returns
    -------
    RandomVariable
        A deterministic expression containing a read-only NumPy value.
    """

    source = np.asarray(value)
    declared_plates = (plates,) if isinstance(plates, str) else tuple(plates)
    layout = PlateLayout.wrap(declared_plates)
    if source.ndim != len(declared_plates):
        raise ValueValidationError(
            f"data.ndim = {source.ndim} != len(plates) = {len(declared_plates)}"
        )
    permutation = tuple(declared_plates.index(plate) for plate in layout)
    canonical_value = np.transpose(source, axes=permutation)
    resolved_dtype = DataType.infer(canonical_value) if dtype is None else dtype
    return Constant(
        ConcreteValue.wrap(
            data=canonical_value,
            layout=layout,
            meta=ValueMeta.from_value(canonical_value, resolved_dtype),
        )
    )


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
