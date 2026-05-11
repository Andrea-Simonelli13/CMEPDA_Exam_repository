import os
import sys

sys.path.insert(0, os.path.abspath("../src"))

# Import the version string
from CMEPDA_Exam_repository._version import __version__

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'CMEPDA_Exam_repository'
copyright = '2026, Andrea Simonelli'
author = 'Andrea Simonelli'
release = __version__

rst_prolog = """
.. |Python| replace:: `Python <https://www.python.org/>`__
.. |Sphinx| replace:: `Sphinx <https://www.sphinx-doc.org/en/master/>`__
.. |numpy| replace:: `NumPy <https://numpy.org/>`__
.. |GitHub| replace:: `GitHub <https://github.com/>`__
.. |INSPIRE-HEP| replace:: `INSPIRE-HEP <https://inspirehep.net/>`__
.. |arXiv| replace:: `arXiv <https://arxiv.org/archive/hep-ph>`__
.. |tensorflow| replace:: `tensorflow <https://www.tensorflow.org/>`__
.. |Keras| replace:: `Keras <https://keras.io/>`__
.. |Dataset| replace:: `Dataset <https://andrea-simonelli13.github.io/CMEPDA_Exam_repository/dataset.html>`__
.. |Exploratory Data Analysis| replace:: `Exploratory Data Analysis <https://andrea-simonelli13.github.io/CMEPDA_Exam_repository/EDA.html>`__
.. |Modelli e Ottimizzazione| replace:: `Modelli e Ottimizzazione <https://andrea-simonelli13.github.io/CMEPDA_Exam_repository/models.html>`__
"""

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.todo',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'undoc-members': True,
    'private-members': True
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinxawesome_theme'#'alabaster'
html_permalinks_icon = '<span>#</span>'
pygments_style = 'default'
pygments_dark_style = 'default'
html_theme_options = {
    'awesome_external_links': True,
}
html_static_path = ['_static']