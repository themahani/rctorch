"""Sphinx configuration for the rcTorch documentation."""

import os
import sys
import tomllib
from datetime import date

# sphinx-multiversion builds every version with this file, but from a checkout of that version, and sets
# SPHINX_MULTIVERSION_SOURCEDIR to the checkout's docs directory. Import rctorch from that checkout so each
# version documents its own code instead of the installed package.
docs_dir = os.path.abspath(os.environ.get("SPHINX_MULTIVERSION_SOURCEDIR", os.path.dirname(__file__)))
root_dir = os.path.dirname(docs_dir)
sys.path.insert(0, root_dir)

with open(os.path.join(root_dir, "pyproject.toml"), "rb") as f:
    release = tomllib.load(f)["project"]["version"]
version = release

project = "rcTorch"
author = "Ali Mahani"
copyright = f"{date.today().year}, {author}"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "numpydoc",
    "sphinx_multiversion",
]

templates_path = ["_templates"]
exclude_patterns = ["_build"]

autodoc_member_order = "bysource"
# The class docstrings document the constructor parameters
autoclass_content = "class"
# Autodoc already lists the members, so a numpydoc summary table would repeat them
numpydoc_show_class_members = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "torch": ("https://docs.pytorch.org/docs/stable", None),
}

html_theme = "furo"
html_title = "rcTorch"
# Each version lives in its own directory, such as /rctorch/main/, so the canonical links include it
html_baseurl = f"https://themahani.github.io/rctorch/{os.environ.get('SPHINX_MULTIVERSION_NAME', 'main')}/"
html_sidebars = {
    "**": [
        "sidebar/brand.html",
        "sidebar/search.html",
        "sidebar/scroll-start.html",
        "sidebar/navigation.html",
        "sidebar/versions.html",
        "sidebar/scroll-end.html",
    ]
}

# sphinx-multiversion builds the main and dev branches and every release tag such as v0.2.0
smv_branch_whitelist = r"^(main|dev)$"
smv_tag_whitelist = r"^v\d+\.\d+\.\d+$"
smv_remote_whitelist = r"^origin$"
smv_prefer_remote_refs = True
smv_released_pattern = r"^refs/tags/.*$"
smv_latest_version = "main"
