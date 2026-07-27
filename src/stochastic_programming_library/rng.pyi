from dataclasses import dataclass
from typing import TypeAlias

Seed: TypeAlias = int
RngLabel: TypeAlias = str
HashDigest: TypeAlias = bytes

@dataclass(frozen=True, slots=True)
class NodeEntropy:
    digest: HashDigest

def derive_node_entropy(
    graph_hash: HashDigest,
    rng_label: RngLabel | None,
) -> NodeEntropy: ...
def resolve_run_seed(seed: Seed | None) -> Seed: ...
def derive_sampling_seed(
    run_seed: Seed,
    node_entropy: NodeEntropy,
) -> Seed: ...
