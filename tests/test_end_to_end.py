import numpy as np
import runpy
import sys

from stochastic_programming_library import (
    Bernoulli,
    Normal,
    Uniform,
    constant,
    sampling_phase,
)
from stochastic_programming_library.examples.bernoulli_trials import (
    build_model as build_bernoulli_model,
)
from stochastic_programming_library.examples.bernoulli_trials import (
    run as run_bernoulli,
)
from stochastic_programming_library.examples.hierarchical_gaussian import (
    build_model as build_gaussian_model,
)
from stochastic_programming_library.examples.hierarchical_gaussian import (
    run as run_gaussian,
)


def test_hierarchical_row_column_gaussian_example() -> None:
    model = build_gaussian_model()
    value = run_gaussian(seed=10)

    assert model.plate_layout.plates == ("row",)
    assert value.layout.plates == ("row",)
    assert value.data.shape == (4,)
    assert np.all(np.isfinite(value.data))


def test_fixed_latent_can_feed_independently_resampled_observations() -> None:
    with sampling_phase("latent"):
        latent = Normal(0, 1).add_plates("row")
    with sampling_phase("observation"):
        observations = Normal(latent, 1).add_plates(
            "replicate",
            expect=("row",),
        )

    fixed = observations.materialize(
        seed=10,
        phases=("latent",),
        plate_sizes={"row": 4, "replicate": 32},
    )
    first = fixed.realize(seed=20)
    second = fixed.realize(seed=21)
    repeated = fixed.realize(seed=20)

    assert fixed.pending_phases == frozenset({"observation"})
    assert not np.array_equal(first.data, second.data)
    np.testing.assert_array_equal(first.data, repeated.data)


def test_grouped_uniform_bernoulli_example() -> None:
    model = build_bernoulli_model()
    rates = run_bernoulli(seed=4)

    assert model.plate_layout.plates == ("group",)
    assert rates.data.shape == (4,)
    assert np.all((rates.data >= 0) & (rates.data <= 1))


def test_independent_graph_allocations_and_aliasing_equality() -> None:
    expected = Normal(0, 1) + Normal(0, 1)
    generated = Normal(0, 1) + Normal(0, 1)
    source = Normal(0, 1)
    shared = source + source

    expected_checkpoint = expected.materialize(seed=8, phases=())
    generated_checkpoint = generated.materialize(seed=8, phases=())
    shared_checkpoint = shared.materialize(seed=8, phases=())

    assert expected.structurally_equal(generated)
    assert expected_checkpoint.stochastically_equal(generated_checkpoint)
    assert shared.structurally_equal(expected)
    assert not shared_checkpoint.stochastically_equal(expected_checkpoint)


def test_named_numpy_constants_feed_stochastic_graph() -> None:
    low = constant(np.array([0.1, 0.4]), plates=("group",))
    high = constant(np.array([0.2, 0.9]), plates=("group",))
    probability = Uniform(low, high)
    trials = Bernoulli(probability).add_plates(
        "trial",
        expect=("group",),
    )

    value = trials.realize(
        seed=9,
        plate_sizes={"group": 2, "trial": 8},
    )

    assert value.layout.plates == ("group", "trial")
    assert value.data.shape == (2, 8)


def test_example_modules_are_directly_executable(capsys) -> None:
    sys.modules.pop(
        "stochastic_programming_library.examples.hierarchical_gaussian",
        None,
    )
    sys.modules.pop(
        "stochastic_programming_library.examples.bernoulli_trials",
        None,
    )
    runpy.run_module(
        "stochastic_programming_library.examples.hierarchical_gaussian",
        run_name="__main__",
    )
    runpy.run_module(
        "stochastic_programming_library.examples.bernoulli_trials",
        run_name="__main__",
    )

    assert capsys.readouterr().out
