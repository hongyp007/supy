import os
import sys
sys.path.insert(0, os.path.abspath('../'))  #

project = 'supy'

extensions = [
	'nbsphinx',
	'sphinx_rtd_theme', 
	'sphinx.ext.autodoc', 
	'sphinx.ext.napoleon',  # If using Google or NumPy style docstrings
    'sphinx.ext.autosummary',  # Optional: Generates summary tables
    ]
html_theme = "sphinx_rtd_theme"
autosummary_generate = True
