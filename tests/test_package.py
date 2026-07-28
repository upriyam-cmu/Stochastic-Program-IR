from importlib.metadata import version

import stoch_ir


def test_runtime_version_uses_distribution_metadata() -> None:
    assert stoch_ir.__version__ == version("stoch-ir")
