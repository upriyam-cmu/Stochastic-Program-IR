"""Domain-separated entropy derivation for stochastic materialization.

The graph resolver supplies stable per-distribution structure hashes. This
module first mixes an optional user label into that structure to form
``NodeEntropy``, then mixes one materialization invocation's run seed into the
node entropy to form the final NumPy sampling seed.
"""

import secrets
from dataclasses import dataclass
from hashlib import blake2s
from typing import TypeAlias

Seed: TypeAlias = int
RngLabel: TypeAlias = str
HashDigest: TypeAlias = bytes


_NODE_ENTROPY_DOMAIN = b"spl-v0.1:node-entropy"
_SAMPLING_SEED_DOMAIN = b"spl-v0.1:sampling-seed"


@dataclass(frozen=True, slots=True)
class NodeEntropy:
    digest: HashDigest


def _digest(domain: bytes, *parts: bytes, digest_size: int) -> bytes:
    hasher = blake2s(digest_size=digest_size)
    for part in (domain, *parts):
        hasher.update(len(part).to_bytes(8, byteorder="little"))
        hasher.update(part)
    return hasher.digest()


def derive_node_entropy(
    graph_hash: HashDigest,
    rng_label: RngLabel | None,
) -> NodeEntropy:
    """Supplement a mandatory graph hash with an optional semantic label."""

    label = b"label\0" + rng_label.encode() if rng_label is not None else b"no-label"
    return NodeEntropy(
        _digest(
            _NODE_ENTROPY_DOMAIN,
            graph_hash,
            label,
            digest_size=16,
        )
    )


def resolve_run_seed(seed: Seed | None) -> Seed:
    """Select the single run seed used by one materialization invocation."""

    return seed if seed is not None else secrets.randbits(64)


def derive_sampling_seed(
    run_seed: Seed,
    node_entropy: NodeEntropy,
) -> Seed:
    """Bind a materialization run seed to one distribution's node entropy."""

    seed_bytes = (run_seed % (1 << 64)).to_bytes(8, byteorder="little")
    return int.from_bytes(
        _digest(
            _SAMPLING_SEED_DOMAIN,
            seed_bytes,
            node_entropy.digest,
            digest_size=8,
        ),
        byteorder="little",
    )
