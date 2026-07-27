import random
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType

import numpy as np

from ..errors import (
    MissingPlateSizeError,
    PlateSizeMismatchError,
    UnrealizedGraphError,
    UnresolvedRandomnessError,
)
from ..rng import Seed, derive_seed
from .hashing import resolve_stochastic_hashes, stamp_resolved_rng_keys
from .meta import (
    EMPTY_PLATE_LAYOUT,
    ConcreteValue,
    Phase,
    Plate,
    PlateLayout,
    PlateSizes,
)
from .nodes.base import Constant, RandomVariable
from .nodes.distr.base import RandomDistributionNode
from .nodes.ops import BinOpNode, ReductionOpNode, UnaryOpNode
from .nodes.shape import AddPlatesNode, add_plates


def _all_nodes(root: RandomVariable) -> tuple[RandomVariable, ...]:
    nodes: list[RandomVariable] = []
    seen: set[int] = set()

    def visit(node: RandomVariable) -> None:
        if id(node) in seen:
            return
        seen.add(id(node))
        nodes.append(node)
        for dependency in node.dependencies:
            visit(dependency.var)

    visit(root)
    return tuple(nodes)


def _normalize_plate_sizes(
    root: RandomVariable,
    plate_sizes: PlateSizes | None,
) -> Mapping[Plate, int]:
    supplied = dict(plate_sizes or {})
    required = frozenset(
        plate for node in _all_nodes(root) for plate in node.plate_layout
    )
    missing = required - supplied.keys()
    if missing:
        raise MissingPlateSizeError(f"missing sizes for plates {sorted(missing)}")
    invalid = {
        plate: size
        for plate, size in supplied.items()
        if not isinstance(size, int) or isinstance(size, bool) or size <= 0
    }
    if invalid:
        raise PlateSizeMismatchError(
            f"plate sizes must be positive integers, got {invalid}"
        )
    return MappingProxyType({plate: supplied[plate] for plate in sorted(required)})


def _ensure_resolved(root: RandomVariable) -> None:
    unresolved = [
        node
        for node in _all_nodes(root)
        if isinstance(node, RandomDistributionNode) and node._resolved_rng_key is None
    ]
    if unresolved:
        raise UnresolvedRandomnessError(
            f"{len(unresolved)} distribution nodes do not have resolved RNG keys"
        )


def _missing_layout(
    requested: PlateLayout,
    current: PlateLayout,
) -> PlateLayout:
    return PlateLayout.wrap(requested.as_set - current.as_set)


def _materialize_node(
    node: RandomVariable,
    *,
    seed: Seed,
    plate_sizes: PlateSizes,
    enabled_phases: frozenset[Phase] | None,
    lifted_layout: PlateLayout = EMPTY_PLATE_LAYOUT,
    memo: dict[tuple[int, tuple[Plate, ...]], RandomVariable],
) -> RandomVariable:
    memo_key = (id(node), lifted_layout.plates)
    if memo_key in memo:
        return memo[memo_key]

    if isinstance(node, AddPlatesNode):
        combined_lift = lifted_layout + node.added_plates
        result = _materialize_node(
            node.arg,
            seed=seed,
            plate_sizes=plate_sizes,
            enabled_phases=enabled_phases,
            lifted_layout=combined_lift,
            memo=memo,
        )
        memo[memo_key] = result
        return result

    if isinstance(node, Constant):
        if not lifted_layout:
            result = node
        else:
            target_layout = node.plate_layout | lifted_layout
            result = Constant(
                ConcreteValue.wrap(
                    add_plates(
                        node.val.data,
                        old_layout=node.plate_layout,
                        new_layout=target_layout,
                        plate_sizes=plate_sizes,
                    ),
                    target_layout,
                    node.value_meta,
                )
            )
        memo[memo_key] = result
        return result

    if isinstance(node, RandomDistributionNode):
        dependency_changes = {
            dependency.name: _materialize_node(
                dependency.var,
                seed=seed,
                plate_sizes=plate_sizes,
                enabled_phases=enabled_phases,
                memo=memo,
            )
            for dependency in node.dependencies
        }
        rebuilt = replace(node, **dependency_changes)
        phase_enabled = (
            rebuilt.phase_requirement is None
            or enabled_phases is None
            or rebuilt.phase_requirement in enabled_phases
        )
        if phase_enabled and rebuilt.phase_requirement is not None:
            rebuilt = replace(rebuilt, phase_requirement=None)

        output_layout = rebuilt.plate_layout | lifted_layout
        if phase_enabled and all(
            dependency.var.has_value for dependency in rebuilt.dependencies
        ):
            if rebuilt._resolved_rng_key is None:
                raise UnresolvedRandomnessError(
                    "cannot sample a distribution without a resolved RNG key"
                )
            rng = np.random.default_rng(derive_seed(seed, rebuilt._resolved_rng_key))
            result = Constant(
                ConcreteValue.wrap(
                    rebuilt._sample_value(
                        rng,
                        output_layout=output_layout,
                        plate_sizes=plate_sizes,
                    ),
                    output_layout,
                    rebuilt.value_meta,
                )
            )
        else:
            missing_lift = _missing_layout(lifted_layout, rebuilt.plate_layout)
            result = AddPlatesNode(rebuilt, missing_lift) if missing_lift else rebuilt
        memo[memo_key] = result
        return result

    if isinstance(node, BinOpNode):
        lhs = _materialize_node(
            node.lhs,
            seed=seed,
            plate_sizes=plate_sizes,
            enabled_phases=enabled_phases,
            lifted_layout=lifted_layout,
            memo=memo,
        )
        rhs = _materialize_node(
            node.rhs,
            seed=seed,
            plate_sizes=plate_sizes,
            enabled_phases=enabled_phases,
            lifted_layout=lifted_layout,
            memo=memo,
        )
        rebuilt = replace(node, lhs=lhs, rhs=rhs)
    elif isinstance(node, UnaryOpNode):
        arg = _materialize_node(
            node.arg,
            seed=seed,
            plate_sizes=plate_sizes,
            enabled_phases=enabled_phases,
            lifted_layout=lifted_layout,
            memo=memo,
        )
        rebuilt = replace(node, arg=arg)
    elif isinstance(node, ReductionOpNode):
        if lifted_layout.as_set & node.arg.plate_layout.as_set:
            raise ValueError(
                "cannot lift a plate through a reduction that already uses "
                "the same plate name"
            )
        arg = _materialize_node(
            node.arg,
            seed=seed,
            plate_sizes=plate_sizes,
            enabled_phases=enabled_phases,
            lifted_layout=lifted_layout,
            memo=memo,
        )
        rebuilt = replace(node, arg=arg)
    else:
        raise TypeError(f"unsupported random-variable node {type(node)!r}")

    result = Constant(rebuilt.value(plate_sizes)) if rebuilt.has_value else rebuilt
    memo[memo_key] = result
    return result


def _materialize_resolved(
    root: RandomVariable,
    *,
    seed: Seed | None,
    plate_sizes: PlateSizes,
    phases: Iterable[Phase] | None,
) -> RandomVariable:
    concrete_seed = seed if seed is not None else random.getrandbits(64)
    enabled_phases = None if phases is None else frozenset(phases)
    return _materialize_node(
        root,
        seed=concrete_seed,
        plate_sizes=plate_sizes,
        enabled_phases=enabled_phases,
        memo={},
    )


def _stochastically_equal(
    left: RandomVariable,
    right: RandomVariable,
) -> bool:
    if not left.structurally_equal(right):
        return False
    seen_pairs: set[tuple[int, int]] = set()
    pending = [(left, right)]
    while pending:
        lhs, rhs = pending.pop()
        pair = (id(lhs), id(rhs))
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        if isinstance(lhs, RandomDistributionNode):
            if not isinstance(rhs, RandomDistributionNode):
                return False
            if (
                lhs.phase_requirement != rhs.phase_requirement
                or lhs._resolved_rng_key != rhs._resolved_rng_key
            ):
                return False
        if len(lhs.dependencies) != len(rhs.dependencies):
            return False
        for lhs_dep, rhs_dep in zip(
            lhs.dependencies,
            rhs.dependencies,
            strict=True,
        ):
            if lhs_dep.name != rhs_dep.name:
                return False
            pending.append((lhs_dep.var, rhs_dep.var))
    return True


@dataclass(frozen=True, slots=True, eq=False)
class SamplingCheckpoint:
    _root: RandomVariable
    _plate_sizes: Mapping[Plate, int]

    @staticmethod
    def _wrap(
        root: RandomVariable,
        plate_sizes: PlateSizes,
    ) -> "SamplingCheckpoint":
        checkpoint = SamplingCheckpoint(
            root,
            MappingProxyType(dict(plate_sizes)),
        )
        return checkpoint

    def __post_init__(self) -> None:
        _ensure_resolved(self._root)

    @property
    def pending_phases(self) -> frozenset[Phase]:
        return self._root.pending_phases

    @property
    def is_fully_materialized(self) -> bool:
        return self._root.has_value

    def structurally_equal(self, other: "SamplingCheckpoint") -> bool:
        return isinstance(other, SamplingCheckpoint) and self._root.structurally_equal(
            other._root
        )

    def stochastically_equal(self, other: "SamplingCheckpoint") -> bool:
        return (
            isinstance(other, SamplingCheckpoint)
            and self._plate_sizes == other._plate_sizes
            and _stochastically_equal(self._root, other._root)
        )

    def materialize(
        self,
        *,
        seed: Seed | None = None,
        plate_sizes: PlateSizes | None = None,
        phases: Iterable[Phase] | None = None,
    ) -> "SamplingCheckpoint":
        sizes = (
            self._plate_sizes
            if plate_sizes is None
            else _normalize_plate_sizes(self._root, plate_sizes)
        )
        if dict(sizes) != dict(self._plate_sizes):
            raise PlateSizeMismatchError(
                "a checkpoint cannot change its resolved plate sizes"
            )
        rewritten = _materialize_resolved(
            self._root,
            seed=seed,
            plate_sizes=sizes,
            phases=phases,
        )
        return SamplingCheckpoint._wrap(rewritten, sizes)

    def realize(
        self,
        *,
        seed: Seed | None = None,
        plate_sizes: PlateSizes | None = None,
    ) -> ConcreteValue:
        completed = self.materialize(
            seed=seed,
            plate_sizes=plate_sizes,
            phases=None,
        )
        if not completed._root.has_value:
            raise UnrealizedGraphError(
                "graph still contains unrealized stochastic nodes"
            )
        return completed._root.value(completed._plate_sizes)


def materialize(
    root: RandomVariable,
    *,
    seed: Seed | None = None,
    plate_sizes: PlateSizes | None = None,
    phases: Iterable[Phase] | None = None,
) -> SamplingCheckpoint:
    sizes = _normalize_plate_sizes(root, plate_sizes)
    hashes = resolve_stochastic_hashes(root)
    resolved = stamp_resolved_rng_keys(root, hashes)
    rewritten = _materialize_resolved(
        resolved,
        seed=seed,
        plate_sizes=sizes,
        phases=phases,
    )
    return SamplingCheckpoint._wrap(rewritten, sizes)


def realize(
    root: RandomVariable,
    *,
    seed: Seed | None = None,
    plate_sizes: PlateSizes | None = None,
) -> ConcreteValue:
    return materialize(
        root,
        seed=seed,
        plate_sizes=plate_sizes,
        phases=None,
    ).realize(seed=seed)
