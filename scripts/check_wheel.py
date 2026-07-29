"""Validate the wheel's inline-typing and public-import contract."""

import sys
from pathlib import Path
from zipfile import ZipFile


def main(wheel: Path) -> None:
    with ZipFile(wheel) as archive:
        names = set(archive.namelist())
        metadata_names = sorted(
            name for name in names if name.endswith(".dist-info/METADATA")
        )
        license_names = sorted(
            name for name in names if name.endswith(".dist-info/licenses/LICENSE")
        )
        if len(metadata_names) != 1:
            raise AssertionError(
                f"wheel must contain exactly one METADATA file: {metadata_names}"
            )
        metadata = archive.read(metadata_names[0]).decode()

    stubs = sorted(name for name in names if name.endswith(".pyi"))
    if stubs:
        raise AssertionError(f"wheel unexpectedly contains stubs: {stubs}")
    if "stoch_ir/py.typed" not in names:
        raise AssertionError("wheel does not contain py.typed")
    if license_names != [metadata_names[0].replace("METADATA", "licenses/LICENSE")]:
        raise AssertionError(
            f"wheel does not contain its license file: {license_names}"
        )
    if "License-Expression: BSD-3-Clause\n" not in metadata:
        raise AssertionError("wheel metadata does not declare BSD-3-Clause")
    if "License-File: LICENSE\n" not in metadata:
        raise AssertionError("wheel metadata does not reference LICENSE")
    expected_urls = (
        "Project-URL: Homepage, https://github.com/upriyam-cmu/Stochastic-Program-IR\n",
        (
            "Project-URL: Documentation, "
            "https://upriyam-cmu.github.io/Stochastic-Program-IR/\n"
        ),
    )
    missing_urls = [url for url in expected_urls if url not in metadata]
    if missing_urls:
        raise AssertionError(f"wheel metadata is missing project URLs: {missing_urls}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: check_wheel.py PATH_TO_WHEEL")
    main(Path(sys.argv[1]))
