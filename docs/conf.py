"""Sphinx config for the SmokeSight docs.

Build locally with::

    pip install -e ".[dev]"
    sphinx-build -b html docs docs/_build/html
"""

from __future__ import annotations

import os
import sys
from datetime import datetime

# Make the package importable for autodoc without needing it installed.
sys.path.insert(0, os.path.abspath(".."))

from smokesight import __version__  # noqa: E402

project = "SmokeSight"
author = "SmokeSight Contributors"
copyright = f"{datetime.now().year}, {author}"
version = __version__
release = __version__

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "nbsphinx",
]

# napoleon settings -- we use NumPy-style docstrings throughout
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_use_rtype = False

# autodoc settings
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
    "member-order": "bysource",
}
autodoc_typehints = "description"
autodoc_mock_imports = ["py6s", "pymodtran", "sklearn"]

# nbsphinx: don't execute notebooks at build time. The notebooks ship
# pre-executed so docs builds stay deterministic and don't need FFmpeg
# / numpy compatibility to match the docs CI environment.
nbsphinx_execute = "never"

# Cross-project linking
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "xarray": ("https://docs.xarray.dev/en/stable/", None),
}

# HTML output
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
templates_path = ["_templates"]
exclude_patterns = ["_build", "**.ipynb_checkpoints"]
