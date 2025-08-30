"""
Configuration file for the Sphinx documentation builder.

For the full list of built-in configuration values, see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html
"""

import sys
from pathlib import Path

# -- Path setup --------------------------------------------------------------
# Add the project root to the Python path so we can import cloakpivot
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Also add docpivot if it exists in the same directory structure
docpivot_path = project_root / "docpivot"
if docpivot_path.exists():
    sys.path.insert(0, str(docpivot_path))

# -- Project information -----------------------------------------------------
project = "CloakPivot"
copyright = "2023, CloakPivot Team"
author = "CloakPivot Team"
release = "0.1.0"
version = "0.1.0"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.githubpages",
    "myst_parser",
    # "nbsphinx",  # Temporarily disabled due to template issues
    "sphinx_click",
    "sphinx_rtd_theme",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

# -- MyST Parser configuration -----------------------------------------------
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "html_admonition",
    "html_image",
    "linkify",
    "replacements",
    "smartquotes",
    "substitution",
    "tasklist",
]

# -- Napoleon configuration --------------------------------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_type_aliases = None
napoleon_attr_annotations = True

# -- Autodoc configuration ---------------------------------------------------
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}
autodoc_class_signature = "mixed"
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented"

# -- Autosummary configuration -----------------------------------------------
autosummary_generate = True
autosummary_generate_overwrite = True

# -- Todo configuration ------------------------------------------------------
todo_include_todos = True

# -- nbsphinx configuration --------------------------------------------------
nbsphinx_execute = "never"  # Don't execute notebooks during build
nbsphinx_allow_errors = True
nbsphinx_requirejs_path = ""

# -- Intersphinx mapping -----------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "presidio": ("https://microsoft.github.io/presidio/", None),
}

# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "canonical_url": "",
    "analytics_id": "",
    "logo_only": False,
    "display_version": True,
    "prev_next_buttons_location": "bottom",
    "style_external_links": False,
    "vcs_pageview_mode": "",
    "style_nav_header_background": "#2980B9",
    "collapse_navigation": True,
    "sticky_navigation": True,
    "navigation_depth": 4,
    "includehidden": True,
    "titles_only": False,
}

html_context = {
    "display_github": True,
    "github_user": "your-org",
    "github_repo": "cloakpivot",
    "github_version": "main",
    "conf_py_path": "/docs/",
}

html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_js_files = ["custom.js"]

# -- Options for LaTeX output ------------------------------------------------
latex_elements = {
    "papersize": "letterpaper",
    "pointsize": "10pt",
    "preamble": "",
    "fncychap": "\\usepackage[Bjornstrup]{fncychap}",
    "printindex": "\\footnotesize\\raggedright\\printindex",
}

latex_documents = [
    (
        "index",
        "cloakpivot.tex",
        "CloakPivot Documentation",
        "CloakPivot Team",
        "manual",
    ),
]

# -- Options for manual page output ------------------------------------------
man_pages = [
    ("index", "cloakpivot", "CloakPivot Documentation", [author], 1)
]

# -- Options for Texinfo output ----------------------------------------------
texinfo_documents = [
    (
        "index",
        "cloakpivot",
        "CloakPivot Documentation",
        author,
        "cloakpivot",
        "PII masking/unmasking on top of DocPivot and Presidio.",
        "Miscellaneous",
    ),
]

# -- Sphinx-Click configuration ----------------------------------------------


def setup(app):
    """Sphinx setup function."""
    app.setup_extension("sphinx_click")


# Mock imports for dependencies that might not be available during docs build
autodoc_mock_imports = [
    "docling_core",
    "docpivot",
    "presidio_analyzer",
    "presidio_anonymizer",
    "cryptography",
    "yaml",
    "pydantic",
]

# Suppress specific warnings
suppress_warnings = [
    "autodoc.import_object",
]
