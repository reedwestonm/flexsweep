import os
import sys

os.environ.setdefault("FLEXSWEEP_BUILD_DOCS", "1")

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

project = "Flex-sweep"
copyright = "2025, Jesús Murga-Moreno"
author = "Jesús Murga-Moreno, David Enard"

version = "2.0"
release = "2.0"


sys.path.insert(0, os.path.abspath(".."))

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration


extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.githubpages",
    "sphinx.ext.viewcode",
]


napoleon_numpy_docstring = True
napoleon_google_docstring = True
napoleon_use_param = False
napoleon_use_rtype = False

# Only third-party heavy deps belong here. Mocking `flexsweep.cnn` itself would
# turn every autodoc directive pointing into it (e.g. `flexsweep.cnn.CNN` in
# api.rst) into an empty stub.
autodoc_mock_imports = ["tensorflow"]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "cms.rst"]

language = "en"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_show_sourcelink = True
html_static_path = []
