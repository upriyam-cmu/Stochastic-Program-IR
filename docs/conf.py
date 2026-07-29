"""Sphinx configuration for the public Stochastic Program IR documentation."""

from importlib.metadata import version as distribution_version

project = "Stochastic Program IR"
author = "Utkarsh Priyam"
copyright = "2026, Utkarsh Priyam"
release = distribution_version("stoch-ir")
version = release

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
]

autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_typehints = "description"
napoleon_numpy_docstring = True
napoleon_google_docstring = False

myst_enable_extensions = ["colon_fence"]

exclude_patterns = ["_build"]
html_theme = "furo"
html_title = f"{project} {release}"
