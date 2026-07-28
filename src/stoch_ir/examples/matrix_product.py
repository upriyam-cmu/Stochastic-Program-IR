"""An explicit named-plate matrix contraction."""

from stoch_ir import Normal


def build_model():
    """Build a stochastic matrix product with an explicit contraction."""

    left = Normal(0.0, 1.0, plates=("row", "inner"))
    right = Normal(0.0, 1.0, plates=("inner", "col"))
    return (left * right).sum("inner").check_plates("row", "col")


def run(seed: int = 1):
    """Realize the example matrix product."""

    return build_model().realize(
        seed=seed,
        plate_sizes={"row": 2, "inner": 3, "col": 4},
    )


if __name__ == "__main__":
    print(run().data)
