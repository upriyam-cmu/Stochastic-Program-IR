# Legacy backend-neutral API prototype, retained for design history.
# NumPy is now the sole internal value/sampling implementation target; backend
# conversion belongs at external API boundaries until a second implementation
# demonstrates that a public backend abstraction is worthwhile.
#
# from dataclasses import dataclass
# from enum import Enum
# from typing import Mapping, Protocol, TypeAlias
#
# Seed: TypeAlias = int | str | bytes
#
# class DistributionKind(str, Enum):
#     NORMAL: str
#     UNIFORM: str
#     BERNOULLI: str
#
# @dataclass(frozen=True)
# class RNGKey:
#     data: bytes
#
# @dataclass(frozen=True)
# class SampleRequest:
#     distribution: DistributionKind
#     parameters: Mapping[str, object]
#     shape: tuple[int, ...]
#
# class SamplingBackend(Protocol):
#     @property
#     def name(self) -> str: ...
#     def sample(self, request: SampleRequest, *, rng_key: RNGKey) -> object: ...
#
# class NumPyBackend:
#     @property
#     def name(self) -> str: ...
#     def sample(self, request: SampleRequest, *, rng_key: RNGKey) -> object: ...
