"""Project an expression DAG into stable per-distribution RNG identities.

Only distribution nodes consume randomness, so the hash scheme deliberately
contracts deterministic nodes. The resulting stochastic multigraph retains:

* the nearest upstream distribution nodes for each named input;
* one edge per distinct source at that boundary, in first-occurrence order;
* the multiplicity with which each source is consumed; and
* a synthetic output consumer for the stochastic frontier of the root.

For each distribution ``v`` the resolver computes:

``D(v)``
    A dependency hash from the distribution type, complete output plate layout,
    and every
    ``(input name, multiplicity, D(source))`` edge.

``C(v)``
    A direct-consumer hash from the sorted *multiset* of
    ``(D(consumer), input name, multiplicity)`` edges. Consuming a node twice
    therefore differs from consuming it once. Descendants beyond the direct
    stochastic consumer are excluded.

``E(v)``
    A canonical traversal ordinal among nodes with identical ``(D, C)``.
    Aliases are visited once; separately allocated symmetric nodes therefore
    receive different ordinals.

``G(v) = H(D(v), C(v), E(v))``
    The final graph-structure hash. Distribution output plates participate;
    deterministic operator kinds, constants, phases, and user RNG labels do
    not.

Object identities are used only as in-process lookup keys and are never mixed
into a digest.
"""

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import blake2s
from types import MappingProxyType

from ..errors import GraphCycleError
from ..rng import HashDigest, NodeEntropy, derive_node_entropy
from .nodes.base import RandomVariable
from .nodes.distr.base import RandomDistributionNode
from .traversal import unique_nodes_postorder, unique_nodes_preorder

_HASH_SCHEME = b"spl-v0.1:stochastic-projection:multiplicity"


def _digest(*parts: bytes) -> HashDigest:
    hasher = blake2s(digest_size=16)
    for part in (_HASH_SCHEME, *parts):
        hasher.update(len(part).to_bytes(8, byteorder="little"))
        hasher.update(part)
    return hasher.digest()


@dataclass(frozen=True, slots=True)
class StochasticInputEdge:
    """One distinct source at a stochastic consumer boundary."""

    source: RandomDistributionNode
    consumer: RandomDistributionNode | None
    parameter: str
    multiplicity: int

    @property
    def is_output_edge(self) -> bool:
        return self.consumer is None


@dataclass(frozen=True, slots=True)
class StochasticFrontierEntry:
    """A nearest source and its repeated-use count."""

    source: RandomDistributionNode
    multiplicity: int


def stochastic_frontier(
    expr: RandomVariable,
    *,
    _memo: dict[int, tuple[StochasticFrontierEntry, ...]] | None = None,
) -> tuple[StochasticFrontierEntry, ...]:
    """Return a memoized, multiplicity-compressed stochastic frontier."""

    memo = {} if _memo is None else _memo
    active: set[int] = set()
    pending: list[tuple[RandomVariable, bool]] = [(expr, False)]
    while pending:
        node, expanded = pending.pop()
        identity = id(node)
        if identity in memo:
            continue
        if expanded:
            entries: list[StochasticFrontierEntry] = []
            entry_indices: dict[int, int] = {}
            for dependency in node._dependency_slots:
                for entry in memo[id(dependency.var)]:
                    source_identity = id(entry.source)
                    index = entry_indices.get(source_identity)
                    if index is None:
                        entry_indices[source_identity] = len(entries)
                        entries.append(entry)
                    else:
                        previous = entries[index]
                        entries[index] = StochasticFrontierEntry(
                            source=previous.source,
                            multiplicity=previous.multiplicity + entry.multiplicity,
                        )
            memo[identity] = tuple(entries)
            active.remove(identity)
            continue

        if identity in active:
            raise GraphCycleError("cycle detected while resolving stochastic frontier")
        active.add(identity)
        if isinstance(node, RandomDistributionNode):
            memo[identity] = (StochasticFrontierEntry(node, 1),)
            active.remove(identity)
            continue
        pending.append((node, True))
        pending.extend(
            (dependency.var, False)
            for dependency in reversed(node._dependency_slots)
            if id(dependency.var) not in memo
        )
    return memo[id(expr)]


@dataclass(frozen=True, slots=True)
class StochasticProjection:
    """The stochastic-only multigraph reachable from one expression root."""

    root: RandomVariable
    nodes: tuple[RandomDistributionNode, ...]
    edges: tuple[StochasticInputEdge, ...]

    @classmethod
    def from_root(cls, root: RandomVariable) -> "StochasticProjection":
        nodes = [
            node
            for node in unique_nodes_preorder(root)
            if isinstance(node, RandomDistributionNode)
        ]

        edges: list[StochasticInputEdge] = []
        frontier_memo: dict[int, tuple[StochasticFrontierEntry, ...]] = {}
        for consumer in nodes:
            for dependency in consumer._dependency_slots:
                for entry in stochastic_frontier(
                    dependency.var,
                    _memo=frontier_memo,
                ):
                    edges.append(
                        StochasticInputEdge(
                            source=entry.source,
                            consumer=consumer,
                            parameter=dependency.name,
                            multiplicity=entry.multiplicity,
                        )
                    )
        for entry in stochastic_frontier(root, _memo=frontier_memo):
            edges.append(
                StochasticInputEdge(
                    source=entry.source,
                    consumer=None,
                    parameter="__output__",
                    multiplicity=entry.multiplicity,
                )
            )
        return StochasticProjection(root, tuple(nodes), tuple(edges))

    def dependencies_of(
        self,
        node: RandomVariable,
    ) -> tuple[StochasticInputEdge, ...]:
        return tuple(edge for edge in self.edges if edge.consumer is node)

    def consumers_of(
        self,
        node: RandomVariable,
    ) -> tuple[StochasticInputEdge, ...]:
        return tuple(edge for edge in self.edges if edge.source is node)


@dataclass(frozen=True, slots=True)
class NodeEnumeration:
    """Distinguishes separate nodes with otherwise identical hash context."""

    ordinal: int


@dataclass(frozen=True, slots=True)
class NodeHashParts:
    """Auditable intermediate and final structural hashes for one node."""

    dependency: HashDigest
    consumer: HashDigest
    enumeration: NodeEnumeration
    final: HashDigest


@dataclass(frozen=True, slots=True)
class ResolvedGraphHashes:
    """Identity-indexed structural hashes for one projected expression root."""

    projection: StochasticProjection
    _parts_by_identity: Mapping[int, NodeHashParts]

    def for_node(self, node: RandomVariable) -> NodeHashParts:
        return self._parts_by_identity[id(node)]

    def node_entropy_for(self, node: RandomVariable) -> NodeEntropy:
        if not isinstance(node, RandomDistributionNode):
            raise TypeError("node entropy is defined only for distributions")
        return derive_node_entropy(
            self.for_node(node).final,
            node.rng_label,
        )


def resolve_stochastic_hashes(root: RandomVariable) -> ResolvedGraphHashes:
    projection = StochasticProjection.from_root(root)
    dependency_hashes: dict[int, HashDigest] = {}
    dependencies_by_consumer: defaultdict[int, list[StochasticInputEdge]] = defaultdict(
        list
    )
    consumers_by_source: defaultdict[int, list[StochasticInputEdge]] = defaultdict(list)
    for edge in projection.edges:
        if edge.consumer is not None:
            dependencies_by_consumer[id(edge.consumer)].append(edge)
        consumers_by_source[id(edge.source)].append(edge)

    for candidate in unique_nodes_postorder(root):
        if not isinstance(candidate, RandomDistributionNode):
            continue
        identity = id(candidate)
        edge_parts: list[bytes] = []
        for edge in dependencies_by_consumer[identity]:
            edge_parts.extend(
                (
                    edge.parameter.encode(),
                    _encode_nonnegative_int(edge.multiplicity),
                    dependency_hashes[id(edge.source)],
                )
            )
        result = _digest(
            b"dependency",
            f"{type(candidate).__module__}.{type(candidate).__qualname__}".encode(),
            b"output-plates",
            *(plate.encode() for plate in candidate.output_layout),
            *edge_parts,
        )
        dependency_hashes[identity] = result

    consumer_hashes: dict[int, HashDigest] = {}
    for node in projection.nodes:
        edge_parts = []
        for edge in consumers_by_source[id(node)]:
            consumer_digest = (
                b"__output__"
                if edge.consumer is None
                else dependency_hashes[id(edge.consumer)]
            )
            edge_parts.append(
                _digest(
                    b"consumer-edge",
                    consumer_digest,
                    edge.parameter.encode(),
                    _encode_nonnegative_int(edge.multiplicity),
                )
            )
        consumer_hashes[id(node)] = _digest(
            b"consumer",
            *sorted(edge_parts),
        )

    bucket_counts: defaultdict[tuple[HashDigest, HashDigest], int] = defaultdict(int)
    parts_by_identity: dict[int, NodeHashParts] = {}
    for node in projection.nodes:
        identity = id(node)
        bucket = (dependency_hashes[identity], consumer_hashes[identity])
        enumeration = NodeEnumeration(bucket_counts[bucket])
        bucket_counts[bucket] += 1
        final = _digest(
            b"final",
            bucket[0],
            bucket[1],
            enumeration.ordinal.to_bytes(8, byteorder="little"),
        )
        parts_by_identity[identity] = NodeHashParts(
            dependency=bucket[0],
            consumer=bucket[1],
            enumeration=enumeration,
            final=final,
        )

    return ResolvedGraphHashes(
        projection=projection,
        _parts_by_identity=MappingProxyType(parts_by_identity),
    )


def _encode_nonnegative_int(value: int) -> bytes:
    if value < 0:
        raise ValueError("cannot encode a negative integer")
    size = max(1, (value.bit_length() + 7) // 8)
    return value.to_bytes(size, byteorder="little")


def stamp_node_entropies(
    root: RandomVariable,
    hashes: ResolvedGraphHashes,
) -> RandomVariable:
    """Rewrite a graph with resolved entropy attached to each distribution."""

    rewritten_nodes: dict[int, RandomVariable] = {}
    for node in unique_nodes_postorder(root):
        rewritten_dependencies = {
            dependency.name: rewritten_nodes[id(dependency.var)]
            for dependency in node._dependency_slots
        }
        rewritten = node._rewrite_dependencies_exact(rewritten_dependencies)
        if isinstance(node, RandomDistributionNode):
            if not isinstance(rewritten, RandomDistributionNode):
                raise TypeError("distribution rewrite changed the node category")
            rewritten = rewritten.with_node_entropy(hashes.node_entropy_for(node))
        rewritten_nodes[id(node)] = rewritten
    return rewritten_nodes[id(root)]
