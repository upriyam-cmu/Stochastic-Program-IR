from dataclasses import dataclass
from ..rng import HashDigest, NodeEntropy
from .nodes import RandomDistributionNode, RandomVariable

@dataclass(frozen=True, slots=True)
class StochasticInputEdge:
    # `consumer=None` represents the synthetic output consumer. Ordinals are
    # assigned left-to-right within each named distribution parameter or output.
    source: RandomDistributionNode
    consumer: RandomDistributionNode | None
    parameter: str
    ordinal: int

    @property
    def is_output_edge(self) -> bool: ...

@dataclass(frozen=True, slots=True)
class StochasticProjection:
    root: RandomVariable
    nodes: tuple[RandomDistributionNode, ...]
    edges: tuple[StochasticInputEdge, ...]

    @classmethod
    def from_root(cls, root: RandomVariable) -> StochasticProjection: ...
    def dependencies_of(
        self,
        node: RandomDistributionNode,
    ) -> tuple[StochasticInputEdge, ...]: ...
    def consumers_of(
        self,
        node: RandomDistributionNode,
    ) -> tuple[StochasticInputEdge, ...]: ...

@dataclass(frozen=True, slots=True)
class NodeEnumeration:
    # Assigned by canonical structural traversal of the stochastic projection,
    # not by construction order or object id. Enumeration is local to nodes
    # sharing the same dependency/consumer signature, so unrelated graph edits
    # do not renumber the node. Aliased references to one node reuse one value;
    # separate symmetric nodes necessarily receive different values.
    ordinal: int

@dataclass(frozen=True, slots=True)
class NodeHashParts:
    # The consumer hash combines direct consumer dependency hashes plus the
    # named-parameter/output edge ordinals. It intentionally does not depend on
    # the selected root or on consumer paths above the direct stochastic layer.
    dependency: HashDigest
    consumer: HashDigest
    enumeration: NodeEnumeration
    final: HashDigest

@dataclass(frozen=True, slots=True)
class ResolvedGraphHashes:
    projection: StochasticProjection

    def for_node(self, node: RandomDistributionNode) -> NodeHashParts: ...
    def node_entropy_for(self, node: RandomDistributionNode) -> NodeEntropy: ...

def stochastic_frontier(
    expr: RandomVariable,
) -> tuple[RandomDistributionNode, ...]: ...
def resolve_stochastic_hashes(root: RandomVariable) -> ResolvedGraphHashes: ...
def stamp_node_entropies(
    root: RandomVariable,
    hashes: ResolvedGraphHashes,
) -> RandomVariable: ...
