from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator

from .expr.meta import Phase

_CURRENT_SAMPLING_PHASE: ContextVar[Phase] = ContextVar(
    "current_sampling_phase",
    default=None,
)


@contextmanager
def sampling_phase(phase: str) -> Iterator[None]:
    if not phase:
        raise ValueError("sampling phase names must be non-empty strings")
    token = _CURRENT_SAMPLING_PHASE.set(phase)
    try:
        yield
    finally:
        _CURRENT_SAMPLING_PHASE.reset(token)


def current_sampling_phase() -> Phase:
    return _CURRENT_SAMPLING_PHASE.get()
