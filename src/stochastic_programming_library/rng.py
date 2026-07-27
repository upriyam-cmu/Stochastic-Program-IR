import random
from dataclasses import dataclass
from enum import Enum
from hashlib import blake2s
from typing import Sequence, TypeAlias

Seed: TypeAlias = int
RngKey: TypeAlias = str
HashDigest: TypeAlias = bytes
PlateCoordinate: TypeAlias = tuple[str, int]


class RngKeyOrigin(str, Enum):
    EXPLICIT = "explicit"
    GRAPH = "graph"


@dataclass(frozen=True, slots=True)
class ResolvedRngKey:
    digest: HashDigest
    origin: RngKeyOrigin


def resolve_explicit_rng_key(rng_key: RngKey) -> ResolvedRngKey:
    return ResolvedRngKey(
        digest=blake2s(
            b"explicit\0" + rng_key.encode(),
            digest_size=16,
        ).digest(),
        origin=RngKeyOrigin.EXPLICIT,
    )


def derive_seed(
    seed: Seed | None,
    rng_key: RngKey | ResolvedRngKey,
) -> Seed:
    hash_key_size = 8  # bytes
    seed_mask = 1 << (8 * hash_key_size)
    hash_key = (
        (seed % seed_mask).to_bytes(hash_key_size, byteorder="little")
        if seed is not None
        else random.randbytes(hash_key_size)
    )
    key_bytes = (
        resolve_explicit_rng_key(rng_key).digest
        if isinstance(rng_key, str)
        else rng_key.digest
    )
    return int.from_bytes(
        blake2s(
            key_bytes,
            digest_size=8,
            key=hash_key,
        ).digest(),
        byteorder="little",
    )  # 8-byte/64-bit int output


def derive_draw_seed(
    seed: Seed | None,
    rng_key: RngKey | ResolvedRngKey,
    coordinates: Sequence[PlateCoordinate],
) -> Seed:
    resolved = (
        resolve_explicit_rng_key(rng_key) if isinstance(rng_key, str) else rng_key
    )
    coordinate_bytes = b"\0".join(
        f"{plate}:{index}".encode() for plate, index in coordinates
    )
    coordinate_key = ResolvedRngKey(
        digest=blake2s(
            b"coordinates\0" + resolved.digest + b"\0" + coordinate_bytes,
            digest_size=16,
        ).digest(),
        origin=resolved.origin,
    )
    return derive_seed(seed, coordinate_key)
