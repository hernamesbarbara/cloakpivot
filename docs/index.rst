CloakPivot Documentation
========================

.. image:: https://img.shields.io/pypi/v/cloakpivot.svg
    :target: https://pypi.python.org/pypi/cloakpivot
    :alt: PyPI version

.. image:: https://img.shields.io/pypi/pyversions/cloakpivot.svg
    :target: https://pypi.python.org/pypi/cloakpivot
    :alt: Python versions

.. image:: https://github.com/your-org/cloakpivot/workflows/CI/badge.svg
    :target: https://github.com/your-org/cloakpivot/actions
    :alt: CI status

CloakPivot is a Python package that enables reversible document masking while preserving structure and formatting. It leverages DocPivot for robust document processing and Presidio for PII detection and anonymization.

Key Features
------------

* **Reversible Masking**: Mask PII while maintaining the ability to restore original content
* **Structure Preservation**: Maintain document layout, formatting, and hierarchy during masking
* **Policy-Driven**: Configurable masking strategies per entity type with comprehensive policy system
* **Format Support**: Works with multiple document formats through DocPivot integration
* **Security**: Optional encryption and integrity verification for CloakMaps
* **CLI & API**: Both command-line interface and programmatic Python API

Quick Start
-----------

.. code-block:: bash

   # Install CloakPivot
   pip install cloakpivot

   # Mask a document
   cloakpivot mask document.pdf --out masked.json --cloakmap map.json

   # Unmask later
   cloakpivot unmask masked.json --cloakmap map.json --out restored.json

.. code-block:: python

   # Python API usage
   from cloakpivot import mask_document, unmask_document
   
   # Mask a document
   result = mask_document("document.pdf", policy="my-policy.yaml")
   
   # Unmask later
   restored = unmask_document(result.masked_path, result.cloakmap_path)

Documentation Contents
----------------------

.. toctree::
   :maxdepth: 2
   :caption: User Guide
   
   getting_started
   installation
   quick_start
   basic_concepts

.. toctree::
   :maxdepth: 2
   :caption: CLI Reference
   
   cli/overview
   cli/mask_command
   cli/unmask_command
   cli/policy_commands
   cli/diagnostics
   cli/workflows

.. toctree::
   :maxdepth: 2
   :caption: Policy Development
   
   policies/overview
   policies/creating_policies
   policies/entity_strategies
   policies/context_rules
   policies/validation
   policies/best_practices

.. toctree::
   :maxdepth: 2
   :caption: API Reference
   
   api/core
   api/masking
   api/unmasking
   api/policies
   api/document
   api/formats
   api/diagnostics

.. toctree::
   :maxdepth: 2
   :caption: Examples & Tutorials
   
   examples/basic_usage
   examples/policy_examples
   examples/industry_scenarios
   examples/advanced_features
   notebooks/index

.. toctree::
   :maxdepth: 2
   :caption: Developer Guide
   
   development/setup
   development/architecture
   development/plugins
   development/testing
   development/contributing

.. toctree::
   :maxdepth: 1
   :caption: Reference
   
   reference/glossary
   reference/faq
   reference/troubleshooting
   reference/changelog

Indices and Search
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`