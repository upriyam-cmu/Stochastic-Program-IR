from dataclasses import dataclass
from typing import TypeAlias
from enum import Enum

# For these support ranges, I'm thinking of instead having
# separate ranges for the positive/negative branches, so
# we can handle multiplication and division easily, not
# to mention things like log/exp etc or split-interval supports
# better than just min/max ranges. the tradeoff is a lot more
# complexity on the support intervals though, so I'm not sure
# if it's worth it. Also, arguably we don't even need support
# tracking -- I figured it would be nice for things like validating
# logs and other input/argument validation, but it's not strictly
# necessary.

class Infinity:
    POS_INF = '+inf'
    NEG_INF = '-inf'
    INVALID = 'nan'

    def __add__(self, other: "ExprInput") -> "RandomVariable": ...
    def __radd__(self, other: "ExprInput") -> "RandomVariable": ...
    def __sub__(self, other: "ExprInput") -> "RandomVariable": ...
    def __rsub__(self, other: "ExprInput") -> "RandomVariable": ...
    def __mul__(self, other: "ExprInput") -> "RandomVariable": ...
    def __rmul__(self, other: "ExprInput") -> "RandomVariable": ...
    def __truediv__(self, other: "ExprInput") -> "RandomVariable": ...
    def __rtruediv__(self, other: "ExprInput") -> "RandomVariable": ...

@dataclass(frozen=True, slots=True)
class Bool:
    @property
    def min(self) -> int:
        return 0

    @property
    def max(self) -> int:
        return 1


@dataclass(frozen=True, slots=True)
class Int:
    # TODO add support for infinity as bounds
    min: int
    max: int

    def __post_init__(self) -> None:
        assert self.min <= self.max


@dataclass(frozen=True, slots=True)
class Float:
    min: float = float("-inf")
    max: float = float("inf")

    def __post_init__(self) -> None:
        assert self.min <= self.max


DataType: TypeAlias = Bool | Int | Float

def resolve_dtype(min: int | float)
