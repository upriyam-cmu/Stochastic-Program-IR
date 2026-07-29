"""Reject API-reference builds that contain unparsed autodoc directives."""

import sys
from pathlib import Path


def main(html_root: Path) -> None:
    api_reference = (html_root / "api-reference.html").read_text()
    releasing = (html_root / "releasing.html").read_text()
    raw_directives = (".. py:function::", ".. py:class::", ".. py:method::")
    present = [marker for marker in raw_directives if marker in api_reference]
    if present:
        raise AssertionError(
            f"API reference contains unparsed autodoc directives: {present}"
        )
    if "stoch_ir.Normal" not in api_reference:
        raise AssertionError("API reference does not contain the Normal constructor")
    if "Trusted Publishing" not in releasing:
        raise AssertionError(
            "release documentation does not describe Trusted Publishing"
        )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: check_docs.py HTML_ROOT")
    main(Path(sys.argv[1]))
