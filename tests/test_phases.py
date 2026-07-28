from typing import Any, cast

import pytest

from stoch_ir import (
    Normal,
    sampling_phase,
)
from stoch_ir.errors import PhaseError
from stoch_ir.phases import _current_sampling_phase


def test_sampling_phase_nests_and_restores() -> None:
    assert _current_sampling_phase() is None
    with sampling_phase("outer"):
        assert _current_sampling_phase() == "outer"
        outer = Normal(0, 1)
        with sampling_phase("inner"):
            assert _current_sampling_phase() == "inner"
            inner = Normal(0, 1)
        assert _current_sampling_phase() == "outer"

    assert _current_sampling_phase() is None
    assert outer.pending_phases == frozenset({"outer"})
    assert inner.pending_phases == frozenset({"inner"})


def test_sampling_phase_restores_after_exception() -> None:
    with pytest.raises(RuntimeError), sampling_phase("temporary"):
        raise RuntimeError("stop")
    assert _current_sampling_phase() is None


def test_distribution_outside_phase_is_unphased() -> None:
    assert Normal(0, 1).pending_phases == frozenset()


@pytest.mark.parametrize("phase", ["", None, 1])
def test_sampling_phase_rejects_invalid_names(phase: object) -> None:
    with pytest.raises(PhaseError), sampling_phase(cast(Any, phase)):
        pass


@pytest.mark.parametrize("phase", ["", None, 1])
def test_materialization_rejects_invalid_phase_names(phase: object) -> None:
    with pytest.raises(PhaseError):
        Normal(0, 1).materialize(phases=cast(Any, (phase,)))
