from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, replace
from hashlib import blake2s
from types import MappingProxyType

from ..errors import GraphCycleError
from ..rng import (
    HashDigest,
    ResolvedRngKey,
    RngKeyOrigin,
    resolve_explicit_rng_key,
)
from .nodes.base import RandomVariable
from .nodes.distr.base import RandomDistributionNode


def _digest(*parts: bytes) -> HashDigest:
    hasher = blake2s(digest_size=16)
    for part in parts:
        hasher.update(len(part).to_bytes(8, byteorder="little"))
        hasher.update(part)
    return hasher.digest()


@dataclass(frozen=True, slots=True)
class StochasticInputEdge:
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
    if isinstance(expr, RandomDistributionNode):
        return (expr,)
    return tuple(
        node
        for dependency in expr.dependencies
        for node in stochastic_frontier(dependency.var)
    )


@dataclass(frozen=True, slots=True)
class StochasticProjection:
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
                for dependency in expr.dependencies:
                    visit_expression(dependency.var)
                active.remove(identity)
                return
            active.add(identity)
            for dependency in expr.dependencies:
                visit_expression(dependency.var)
            active.remove(identity)

        visit_expression(root)

        edges: list[StochasticInputEdge] = []
        for consumer in nodes:
            for dependency in consumer.dependencies:
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
        node: RandomDistributionNode,
    ) -> tuple[StochasticInputEdge, ...]:
        return tuple(edge for edge in self.edges if edge.consumer is node)

    def consumers_of(
        self,
        node: RandomDistributionNode,
    ) -> tuple[StochasticInputEdge, ...]:
        return tuple(edge for edge in self.edges if edge.source is node)


@dataclass(frozen=True, slots=True)
class NodeEnumeration:
    ordinal: int


@dataclass(frozen=True, slots=True)
class NodeHashParts:
    dependency: HashDigest
    consumer: HashDigest
    enumeration: NodeEnumeration
    final: HashDigest


@dataclass(frozen=True, slots=True)
class ResolvedGraphHashes:
    projection: StochasticProjection
    _parts_by_identity: Mapping[int, NodeHashParts]
    _keys_by_identity: Mapping[int, ResolvedRngKey]

    def for_node(self, node: RandomDistributionNode) -> NodeHashParts:
        return self._parts_by_identity[id(node)]

    def rng_key_for(self, node: RandomDistributionNode) -> ResolvedRngKey:
        return self._keys_by_identity[id(node)]


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
    keys_by_identity: dict[int, ResolvedRngKey] = {}
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
        keys_by_identity[identity] = (
            resolve_explicit_rng_key(node.rng_key)
            if node.rng_key is not None
            else ResolvedRngKey(final, RngKeyOrigin.GRAPH)
        )

    return ResolvedGraphHashes(
        projection=projection,
        _parts_by_identity=MappingProxyType(parts_by_identity),
        _keys_by_identity=MappingProxyType(keys_by_identity),
    )


def stamp_resolved_rng_keys(
    root: RandomVariable,
    hashes: ResolvedGraphHashes,
) -> RandomVariable:
    memo: dict[int, RandomVariable] = {}

    def rewrite(node: RandomVariable) -> RandomVariable:
        identity = id(node)
        if identity in memo:
            return memo[identity]
        dependency_changes = {
            dependency.name: rewrite(dependency.var) for dependency in node.dependencies
        }
        if isinstance(node, RandomDistributionNode):
            rewritten = replace(
                node,
                **dependency_changes,
                _resolved_rng_key=hashes.rng_key_for(node),
            )
        else:
            rewritten = (
                replace(node, **dependency_changes) if dependency_changes else node
            )
        memo[identity] = rewritten
        return rewritten

    return rewrite(root)
