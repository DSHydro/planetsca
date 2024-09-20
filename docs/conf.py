# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

from planetsca import __version__

sys.path.insert(0, os.path.abspath("../src/planetsca"))

autosummary_generate = True

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "PlanetSCA"
copyright = "2024, DSHydro"
author = "DSHydro"
version = __version__
release = __version__
# the page title will default to "<project> <relsease> documentation" e.g. PlanetSCA 1.0.0 documentation

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "myst_parser",
    "nbsphinx",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# prevent execution of jupyter notebooks when building docs
nbsphinx_execute = "never"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = [
    "_static"
]  # this may generate a warning: WARNING: html_static_path entry '_static' does not exist, if we don't have anything in _static
