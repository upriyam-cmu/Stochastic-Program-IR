from dataclasses import dataclass
from typing import Sequence, TypeAlias

Seed: TypeAlias = int
RngKey: TypeAlias = str
HashDigest: TypeAlias = bytes
PlateCoordinate: TypeAlias = tuple[str, int]

@dataclass(frozen=True, slots=True)
class ResolvedRngKey:
    digest: HashDigest

def derive_seed(
    seed: Seed | None,
    rng_key: RngKey | ResolvedRngKey,
) -> Seed: ...

def derive_draw_seed(
    seed: Seed | None,
    rng_key: RngKey | ResolvedRngKey,
    coordinates: Sequence[PlateCoordinate],
) -> Seed: ...
