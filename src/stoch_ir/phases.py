from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

from .errors import PhaseError
from .expr.meta import Phase

_CURRENT_SAMPLING_PHASE: ContextVar[Phase] = ContextVar(
    "current_sampling_phase",
    default=None,
)


@contextmanager
def sampling_phase(phase: str) -> Iterator[None]:
    """Assign a sampling phase to distributions created in the context.

    Parameters
    ----------
    phase
        Non-empty phase name. Phase names are unordered; graph dependencies
        determine when enabled distributions become ready.

    Yields
    ------
    None
        Control to the distribution-authoring block.
    """

    if not isinstance(phase, str) or not phase:
        raise PhaseError("sampling phase names must be non-empty strings")
    token = _CURRENT_SAMPLING_PHASE.set(phase)
    try:
        yield
    finally:
        _CURRENT_SAMPLING_PHASE.reset(token)


def _current_sampling_phase() -> Phase:
    return _CURRENT_SAMPLING_PHASE.get()


__all__ = ["sampling_phase"]
