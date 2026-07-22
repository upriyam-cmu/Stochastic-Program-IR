import random
from hashlib import blake2s
from typing import TypeAlias

Seed: TypeAlias = int
RngKey: TypeAlias = str


def derive_seed(seed: Seed | None, rng_key: RngKey) -> Seed:
    hash_key_size = 8  # bytes
    seed_mask = 1 << (8 * hash_key_size)
    hash_key = (
        (seed % seed_mask).to_bytes(hash_key_size, byteorder="little")
        if seed is not None
        else random.randbytes(hash_key_size)
    )
    return int.from_bytes(
        blake2s(
            rng_key.encode(),
            digest_size=8,
            key=hash_key,
        ).digest(),
        byteorder="little",
    )  # 8-byte/64-bit int output
