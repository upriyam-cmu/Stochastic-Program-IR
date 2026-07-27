from dataclasses import dataclass
from enum import Enum
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

def resolve_explicit_rng_key(rng_key: RngKey) -> ResolvedRngKey: ...
def derive_seed(
    seed: Seed | None,
    rng_key: RngKey | ResolvedRngKey,
) -> Seed: ...
def derive_draw_seed(
    seed: Seed | None,
    rng_key: RngKey | ResolvedRngKey,
    coordinates: Sequence[PlateCoordinate],
) -> Seed: ...
