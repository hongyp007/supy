import os
import sys
sys.path.insert(0, os.path.abspath('../'))  #

project = 'supy'

extensions = ['sphinx.ext.autodoc', 'sphinx_rtd_theme', 'nbsphinx']
html_theme = "sphinx_rtd_theme"
autosummary_generate = True
