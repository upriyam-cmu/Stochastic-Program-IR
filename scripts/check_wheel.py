"""Validate the wheel's inline-typing and public-import contract."""

from pathlib import Path
import sys
from zipfile import ZipFile


def main(wheel: Path) -> None:
    with ZipFile(wheel) as archive:
        names = set(archive.namelist())

    stubs = sorted(name for name in names if name.endswith(".pyi"))
    if stubs:
        raise AssertionError(f"wheel unexpectedly contains stubs: {stubs}")
    if "stochastic_programming_library/py.typed" not in names:
        raise AssertionError("wheel does not contain py.typed")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: check_wheel.py PATH_TO_WHEEL")
    main(Path(sys.argv[1]))
