"""Immutable materialization rewrites for stochastic expression graphs.

Materialization has two separate resolution steps:

1. Raw expression graphs are projected to their stochastic structure and each
   distribution is stamped with stable node entropy.
2. Each materialization invocation resolves one run seed. Any distribution
   whose phase is enabled immediately binds that run seed to its node entropy,
   even if unresolved dependencies prevent it from sampling in the same pass.

The second rule makes phase enabling persistent in the returned graph. A later
rewrite may satisfy the blocked dependencies, but the distribution still uses
the sampling seed fixed when its own phase was enabled.

Every rewrite is immutable. Sampled distributions and eagerly computable
deterministic subgraphs collapse to ``Constant`` nodes; unresolved portions
remain ordinary expression nodes with rewritten dependencies.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from ..errors import (
    MissingPlateSizeError,
    PhaseError,
    PlateSizeMismatchError,
    UnrealizedGraphError,
    UnresolvedRandomnessError,
)
from ..rng import Seed, resolve_run_seed
from .hashing import resolve_stochastic_hashes, stamp_node_entropies
from .meta import (
    ConcreteValue,
    Phase,
    Plate,
    PlateSizes,
)
from .nodes.base import Constant, RandomVariable
from .nodes.distr.base import RandomDistributionNode
from .traversal import unique_nodes_postorder, unique_nodes_preorder


def _all_nodes(root: RandomVariable) -> tuple[RandomVariable, ...]:
    return unique_nodes_preorder(root)


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


def _checkpoint_plate_sizes(
    resolved: Mapping[Plate, int],
    supplied: PlateSizes | None,
) -> Mapping[Plate, int]:
    """Validate a checkpoint call against sizes fixed by its source graph."""

    if supplied is None:
        return resolved
    provided = dict(supplied)
    missing = resolved.keys() - provided.keys()
    if missing:
        raise PlateSizeMismatchError(
            f"a checkpoint is missing resolved plate sizes {sorted(missing)}"
        )
    invalid = {
        plate: provided[plate]
        for plate in resolved
        if not isinstance(provided[plate], int)
        or isinstance(provided[plate], bool)
        or provided[plate] <= 0
    }
    if invalid:
        raise PlateSizeMismatchError(
            f"plate sizes must be positive integers, got {invalid}"
        )
    requested = {plate: provided[plate] for plate in resolved}
    if requested != dict(resolved):
        raise PlateSizeMismatchError(
            "a checkpoint cannot change its resolved plate sizes"
        )
    return resolved


def _ensure_resolved(root: RandomVariable) -> None:
    unresolved = [
        node
        for node in _all_nodes(root)
        if isinstance(node, RandomDistributionNode) and node._node_entropy is None
    ]
    if unresolved:
        raise UnresolvedRandomnessError(
            f"{len(unresolved)} distribution nodes do not have resolved node entropy"
        )


def _constant_value(
    node: Constant,
    plate_sizes: PlateSizes,
) -> ConcreteValue:
    return node._evaluate_concrete({}, plate_sizes)


def _concrete_dependencies(
    node: RandomVariable,
    plate_sizes: PlateSizes,
) -> Mapping[str, ConcreteValue] | None:
    if not all(
        isinstance(dependency.var, Constant) for dependency in node._dependency_slots
    ):
        return None
    return MappingProxyType(
        {
            dependency.name: _constant_value(dependency.var, plate_sizes)
            for dependency in node._dependency_slots
            if isinstance(dependency.var, Constant)
        }
    )


def _evaluate_deterministic_tree(
    root: RandomVariable,
    plate_sizes: PlateSizes,
) -> ConcreteValue:
    """Evaluate a value-ready graph without creating a sampling checkpoint."""

    values: dict[int, ConcreteValue] = {}
    for node in unique_nodes_postorder(root):
        if isinstance(node, RandomDistributionNode):
            raise UnrealizedGraphError(
                "a stochastic graph must be materialized before value extraction"
            )
        dependencies = MappingProxyType(
            {
                dependency.name: values[id(dependency.var)]
                for dependency in node._dependency_slots
            }
        )
        values[id(node)] = node._evaluate_concrete(dependencies, plate_sizes)
    return values[id(root)]


def _materialize_node(
    node: RandomVariable,
    *,
    run_seed: Seed,
    plate_sizes: PlateSizes,
    enabled_phases: frozenset[Phase] | None,
    memo: dict[int, RandomVariable],
) -> RandomVariable:
    for current in unique_nodes_postorder(node):
        memo_key = id(current)
        if memo_key in memo:
            continue
        if isinstance(current, Constant):
            memo[memo_key] = current
            continue

        dependency_changes = {
            dependency.name: memo[id(dependency.var)]
            for dependency in current._dependency_slots
        }
        rebuilt = current._rewrite_dependencies_exact(dependency_changes)
        if isinstance(current, RandomDistributionNode):
            if not isinstance(rebuilt, RandomDistributionNode):
                raise TypeError("distribution rewrite changed the node category")

            phase_enabled = (
                rebuilt.phase_requirement is None
                or enabled_phases is None
                or rebuilt.phase_requirement in enabled_phases
            )
            if phase_enabled and rebuilt._sampling_seed is None:
                rebuilt = rebuilt.bind_sampling_seed(run_seed)

            output_layout = rebuilt.plate_layout
            dependencies = _concrete_dependencies(rebuilt, plate_sizes)
            if rebuilt._sampling_seed is not None and dependencies is not None:
                rng = np.random.default_rng(rebuilt._sampling_seed)
                result = Constant(
                    ConcreteValue.wrap(
                        rebuilt._sample_value(
                            rng,
                            dependencies=dependencies,
                            output_layout=output_layout,
                            plate_sizes=plate_sizes,
                        ),
                        output_layout,
                        rebuilt.value_meta,
                    )
                )
            else:
                result = rebuilt
        else:
            dependencies = _concrete_dependencies(rebuilt, plate_sizes)
            result = (
                Constant(rebuilt._evaluate_concrete(dependencies, plate_sizes))
                if dependencies is not None
                else rebuilt
            )
        memo[memo_key] = result
    return memo[id(node)]


def _materialize_resolved(
    root: RandomVariable,
    *,
    seed: Seed | None,
    plate_sizes: PlateSizes,
    phases: Iterable[Phase] | None,
) -> RandomVariable:
    # Exactly one run seed is selected for an invocation. It is immediately
    # bound to every newly enabled distribution encountered in this pass.
    if phases is None:
        enabled_phases = None
    else:
        enabled_phases = (
            frozenset({phases}) if isinstance(phases, str) else frozenset(phases)
        )
        if any(not isinstance(phase, str) or not phase for phase in enabled_phases):
            raise PhaseError("materialization phase names must be non-empty strings")
    run_seed = resolve_run_seed(seed)
    return _materialize_node(
        root,
        run_seed=run_seed,
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
                or lhs._node_entropy != rhs._node_entropy
                or lhs._sampling_seed != rhs._sampling_seed
            ):
                return False
        if len(lhs._dependency_slots) != len(rhs._dependency_slots):
            return False
        for lhs_dep, rhs_dep in zip(
            lhs._dependency_slots,
            rhs._dependency_slots,
            strict=True,
        ):
            if lhs_dep.name != rhs_dep.name:
                return False
            pending.append((lhs_dep.var, rhs_dep.var))
    return True


@dataclass(frozen=True, slots=True, eq=False)
class SamplingCheckpoint:
    """Opaque, immutable partially materialized stochastic graph.

    Checkpoints fix graph-derived node entropy, resolved plate sizes, and any
    values sampled so far. They cannot participate in new expressions; call
    :meth:`materialize`, :meth:`realize`, or :meth:`value` to continue.
    """

    _root: RandomVariable
    _plate_sizes: Mapping[Plate, int]

    @staticmethod
    def _wrap(
        root: RandomVariable,
        plate_sizes: PlateSizes,
    ) -> "SamplingCheckpoint":
        return SamplingCheckpoint(
            root,
            MappingProxyType(dict(plate_sizes)),
        )

    def __post_init__(self) -> None:
        _ensure_resolved(self._root)

    @property
    def pending_phases(self) -> frozenset[str]:
        """Return named sampling phases still present in the checkpoint."""

        return self._root.pending_phases

    @property
    def is_fully_materialized(self) -> bool:
        """Whether the checkpoint has collapsed to one concrete constant."""

        return isinstance(self._root, Constant)

    def structurally_equal(self, other: "SamplingCheckpoint") -> bool:
        """Compare rewritten computation structure, ignoring resolved RNG data."""

        return isinstance(other, SamplingCheckpoint) and self._root.structurally_equal(
            other._root
        )

    def stochastically_equal(self, other: "SamplingCheckpoint") -> bool:
        """Compare structure, stochastic sharing, and resolved RNG state."""

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
        phases: Iterable[str] | None = None,
    ) -> "SamplingCheckpoint":
        """Enable selected phases and return a new immutable checkpoint."""

        sizes = _checkpoint_plate_sizes(self._plate_sizes, plate_sizes)
        rewritten = _materialize_resolved(
            self._root,
            seed=seed,
            plate_sizes=sizes,
            phases=phases,
        )
        return SamplingCheckpoint._wrap(rewritten, sizes)

    def value(self) -> ConcreteValue:
        """Return concrete data or raise if stochastic nodes remain."""

        if not isinstance(self._root, Constant):
            raise UnrealizedGraphError(
                "checkpoint still contains unrealized stochastic nodes"
            )
        return _constant_value(self._root, self._plate_sizes)

    def realize(
        self,
        *,
        seed: Seed | None = None,
        plate_sizes: PlateSizes | None = None,
    ) -> ConcreteValue:
        """Materialize every remaining phase and return concrete data."""

        return self.materialize(
            seed=seed,
            plate_sizes=plate_sizes,
            phases=None,
        ).value()


def materialize(
    root: RandomVariable,
    *,
    seed: Seed | None = None,
    plate_sizes: PlateSizes | None = None,
    phases: Iterable[str] | None = None,
) -> SamplingCheckpoint:
    sizes = _normalize_plate_sizes(root, plate_sizes)
    hashes = resolve_stochastic_hashes(root)
    resolved = stamp_node_entropies(root, hashes)
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
    sizes = _normalize_plate_sizes(root, plate_sizes)
    if root.has_value:
        return _evaluate_deterministic_tree(root, sizes)
    return materialize(
        root,
        seed=seed,
        plate_sizes=sizes,
        phases=None,
    ).value()
