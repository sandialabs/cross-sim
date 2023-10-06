# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
sys.path.insert(0, os.path.abspath('../../..')) # cross-sim/sphinx/source

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'CrossSim'
copyright = '2023, Sandia National Laboratories'
author = 'Sandia National Laboratories'
release = '2.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.duration', 
    'sphinx.ext.autodoc', 
    'sphinx.ext.napoleon',
    'sphinx.ext.inheritance_diagram',
    'sphinx.ext.graphviz',
    'sphinx.ext.viewcode',
    'sphinx.ext.todo',
    'myst_parser',
]

templates_path = ['_templates']
exclude_patterns = []

autoclass_content = 'both'
autodoc_typehints = 'description'

modindex_common_prefix = ['simulator.']

todo_include_todos = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_theme_options = {
    'navigation_depth': 2
}

