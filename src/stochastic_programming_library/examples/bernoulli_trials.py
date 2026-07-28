"""Grouped Uniform probabilities feeding replicated Bernoulli trials."""

from stochastic_programming_library import Bernoulli, Uniform, sampling_phase


def build_model():
    with sampling_phase("probability"):
        probability = Uniform().add_plates("group")

    with sampling_phase("trial"):
        trial = Bernoulli(probability).add_plates(
            "trial",
            expect=("group",),
        )

    return trial.mean("trial").check_plates("group")


def run(seed: int = 1):
    return build_model().realize(
        seed=seed,
        plate_sizes={"group": 4, "trial": 100},
    )


if __name__ == "__main__":
    print(run().data)
