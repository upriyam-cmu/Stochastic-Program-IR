"""Project an expression DAG into stable per-distribution RNG identities.

Only distribution nodes consume randomness, so the hash scheme deliberately
contracts deterministic nodes. The resulting stochastic multigraph retains:

* the nearest upstream distribution nodes for each named input;
* the left-to-right occurrence ordinal within that input;
* repeated edges when one distribution is consumed more than once; and
* a synthetic output consumer for the stochastic frontier of the root.

For each distribution ``v`` the resolver computes:

``D(v)``
    A dependency hash from the distribution type, complete output plate layout,
    and every
    ``(input name, ordinal, D(source))`` edge.

``C(v)``
    A direct-consumer hash from the sorted *multiset* of
    ``(D(consumer), input name, ordinal)`` edges. Duplicate uses remain
    duplicate entries, so consuming a node twice differs from consuming it
    once. Descendants beyond the direct stochastic consumer are excluded.

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

_HASH_SCHEME = b"spl-v0.1:stochastic-projection"


def _digest(*parts: bytes) -> HashDigest:
    hasher = blake2s(digest_size=16)
    for part in (_HASH_SCHEME, *parts):
        hasher.update(len(part).to_bytes(8, byteorder="little"))
        hasher.update(part)
    return hasher.digest()


@dataclass(frozen=True, slots=True)
class StochasticInputEdge:
    """One occurrence of a distribution at a stochastic consumer boundary."""

    source: RandomDistributionNode
    consumer: RandomDistributionNode | None
    parameter: str
    ordinal: int

    @property
    def is_output_edge(self) -> bool:
        return self.consumer is None


def stochastic_frontier(
    expr: RandomVariable,
) -> tuple[RandomDistributionNode, ...]:
    """Return nearest upstream distributions, preserving use multiplicity."""

    if isinstance(expr, RandomDistributionNode):
        return (expr,)
    return tuple(
        node
        for dependency in expr._dependency_slots
        for node in stochastic_frontier(dependency.var)
    )


@dataclass(frozen=True, slots=True)
class StochasticProjection:
    """The stochastic-only multigraph reachable from one expression root."""

    root: RandomVariable
    nodes: tuple[RandomDistributionNode, ...]
    edges: tuple[StochasticInputEdge, ...]

    @classmethod
    def from_root(cls, root: RandomVariable) -> "StochasticProjection":
        nodes: list[RandomDistributionNode] = []
        seen: set[int] = set()
        active: set[int] = set()

        def visit_expression(expr: RandomVariable) -> None:
            identity = id(expr)
            if identity in active:
                raise GraphCycleError(
                    "cycle detected while projecting stochastic graph"
                )
            if isinstance(expr, RandomDistributionNode):
                if identity in seen:
                    return
                seen.add(identity)
                nodes.append(expr)
                active.add(identity)
                for dependency in expr._dependency_slots:
                    visit_expression(dependency.var)
                active.remove(identity)
                return
            active.add(identity)
            for dependency in expr._dependency_slots:
                visit_expression(dependency.var)
            active.remove(identity)

        visit_expression(root)

        edges: list[StochasticInputEdge] = []
        for consumer in nodes:
            for dependency in consumer._dependency_slots:
                for ordinal, source in enumerate(stochastic_frontier(dependency.var)):
                    edges.append(
                        StochasticInputEdge(
                            source=source,
                            consumer=consumer,
                            parameter=dependency.name,
                            ordinal=ordinal,
                        )
                    )
        for ordinal, source in enumerate(stochastic_frontier(root)):
            edges.append(
                StochasticInputEdge(
                    source=source,
                    consumer=None,
                    parameter="__output__",
                    ordinal=ordinal,
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
    active: set[int] = set()

    def dependency_hash(node: RandomDistributionNode) -> HashDigest:
        identity = id(node)
        if identity in dependency_hashes:
            return dependency_hashes[identity]
        if identity in active:
            raise GraphCycleError("cycle detected while hashing stochastic graph")
        active.add(identity)
        edge_parts: list[bytes] = []
        for edge in projection.dependencies_of(node):
            edge_parts.extend(
                (
                    edge.parameter.encode(),
                    edge.ordinal.to_bytes(8, byteorder="little"),
                    dependency_hash(edge.source),
                )
            )
        result = _digest(
            b"dependency",
            f"{type(node).__module__}.{type(node).__qualname__}".encode(),
            b"output-plates",
            *(plate.encode() for plate in node.output_layout),
            *edge_parts,
        )
        active.remove(identity)
        dependency_hashes[identity] = result
        return result

    for node in reversed(projection.nodes):
        dependency_hash(node)

    consumer_hashes: dict[int, HashDigest] = {}
    for node in projection.nodes:
        edge_parts = []
        for edge in projection.consumers_of(node):
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
                    edge.ordinal.to_bytes(8, byteorder="little"),
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


def stamp_node_entropies(
    root: RandomVariable,
    hashes: ResolvedGraphHashes,
) -> RandomVariable:
    """Rewrite a graph with resolved entropy attached to each distribution."""

    memo: dict[int, RandomVariable] = {}

    def rewrite(node: RandomVariable) -> RandomVariable:
        identity = id(node)
        if identity in memo:
            return memo[identity]
        rewritten_dependencies = {
            dependency.name: rewrite(dependency.var)
            for dependency in node._dependency_slots
        }
        rewritten = node._rewrite_dependencies_exact(rewritten_dependencies)
        if isinstance(node, RandomDistributionNode):
            if not isinstance(rewritten, RandomDistributionNode):
                raise TypeError("distribution rewrite changed the node category")
            rewritten = rewritten.with_node_entropy(hashes.node_entropy_for(node))
        memo[identity] = rewritten
        return rewritten

    return rewrite(root)
