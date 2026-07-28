"""A row/column Gaussian model with staged sampling and named reduction."""

from stoch_ir import Normal, sampling_phase, softplus


def build_model():
    with sampling_phase("latent"):
        row_effect = Normal(0.0, 1.0, rng_label="row-effect").add_plates("row")
        col_scale = Normal(0.0, 1.0, rng_label="col-scale").add_plates("col")

    with sampling_phase("observation"):
        observation = Normal(
            mu=row_effect,
            sigma=softplus(col_scale) + 0.1,
            rng_label="observation",
        )

    return observation.mean("col").check_plates("row")


def run(seed: int = 1):
    return build_model().realize(
        seed=seed,
        plate_sizes={"row": 4, "col": 3},
    )


if __name__ == "__main__":
    print(run().data)
