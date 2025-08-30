CLI Overview
============

CloakPivot provides a comprehensive command-line interface for document masking, unmasking, and policy management. The CLI is designed to handle both simple single-document operations and complex batch processing workflows.

.. contents::
   :local:
   :depth: 2

Installation and Setup
----------------------

Once CloakPivot is installed, the ``cloakpivot`` command becomes available:

.. code-block:: bash

    $ cloakpivot --help

Main Commands
-------------

The CloakPivot CLI is organized into the following main command groups:

Core Operations
~~~~~~~~~~~~~~~

* **mask** - Mask PII in documents while preserving structure
* **unmask** - Restore original content using CloakMaps
* **diff** - Compare documents and analyze masking differences

Policy Management
~~~~~~~~~~~~~~~~~

* **policy sample** - Generate sample policy files
* **policy validate** - Validate policy file syntax and rules
* **policy template** - Generate policies from built-in templates
* **policy test** - Test policies against sample text
* **policy create** - Interactive policy creation wizard
* **policy info** - Show detailed policy information

Format Operations
~~~~~~~~~~~~~~~~~

* **format convert** - Convert between document formats
* **format detect** - Detect document format
* **format list** - List supported formats

Diagnostics and Analysis
~~~~~~~~~~~~~~~~~~~~~~~~

* **diagnostics analyze** - Generate comprehensive analysis reports
* **diagnostics summary** - Quick masking statistics overview

Batch Operations
~~~~~~~~~~~~~~~~

* **batch mask** - Batch masking of multiple documents
* **batch unmask** - Batch unmasking operations
* **batch validate** - Validate multiple CloakMaps

Common Usage Patterns
---------------------

Basic Masking Workflow
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    # 1. Create or obtain a policy
    $ cloakpivot policy template balanced > my-policy.yaml
    
    # 2. Mask a document
    $ cloakpivot mask document.pdf \
        --policy my-policy.yaml \
        --out masked.json \
        --cloakmap document.cloakmap.json
    
    # 3. Later, unmask when needed
    $ cloakpivot unmask masked.json \
        --cloakmap document.cloakmap.json \
        --out restored.json

Policy Development Workflow
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    # 1. Start with interactive creation
    $ cloakpivot policy create --output healthcare-policy.yaml
    
    # 2. Validate the policy
    $ cloakpivot policy validate healthcare-policy.yaml
    
    # 3. Test with sample text
    $ cloakpivot policy test healthcare-policy.yaml \
        --text "Patient John Doe, DOB: 1980-01-01, SSN: 123-45-6789"
    
    # 4. Apply to real documents
    $ cloakpivot mask patient-records.pdf \
        --policy healthcare-policy.yaml \
        --verbose

Analysis and Diagnostics
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    # Quick overview of masking results
    $ cloakpivot diagnostics summary document.cloakmap.json
    
    # Comprehensive analysis report
    $ cloakpivot diagnostics analyze masked.json document.cloakmap.json \
        --format html --output analysis-report.html
    
    # Compare different masking approaches
    $ cloakpivot diff document1.json document2.json \
        --cloakmap1 map1.json --cloakmap2 map2.json \
        --format html --output comparison.html

Global Options
--------------

All CloakPivot commands support these global options:

.. option:: --verbose, -v

   Enable verbose output for detailed operation information

.. option:: --quiet, -q

   Suppress all non-error output

.. option:: --config <path>

   Specify a YAML configuration file for default settings

.. option:: --help

   Show command help and exit

Configuration File
------------------

You can create a configuration file to set default options:

.. code-block:: yaml

    # ~/.cloakpivot/config.yaml
    verbose: false
    default_policy: "~/.cloakpivot/policies/default.yaml"
    output_format: "lexical"
    
    # Default masking options
    masking:
      min_score: 0.7
      language: "en"
      resolve_conflicts: true
    
    # Default paths
    paths:
      policies_dir: "~/.cloakpivot/policies"
      output_dir: "./masked_documents"
    
    # Security settings
    security:
      encrypt_cloakmaps: false
      verify_integrity: true

Environment Variables
--------------------

CloakPivot recognizes these environment variables:

.. envvar:: CLOAKPIVOT_CONFIG

   Path to the default configuration file

.. envvar:: CLOAKPIVOT_POLICIES_DIR

   Default directory for policy files

.. envvar:: CLOAKPIVOT_VERBOSE

   Set to ``1`` to enable verbose output by default

.. envvar:: PRESIDIO_ANALYZER_NLP_ENGINE

   Specify the NLP engine for Presidio (e.g., ``spacy``, ``stanza``)

Shell Completion
----------------

CloakPivot supports shell completion for bash, zsh, and fish shells.

Bash Completion
~~~~~~~~~~~~~~~

.. code-block:: bash

    # Add to ~/.bashrc
    _CLOAKPIVOT_COMPLETE=bash_source cloakpivot > ~/.cloakpivot-complete.bash
    source ~/.cloakpivot-complete.bash

Zsh Completion
~~~~~~~~~~~~~~

.. code-block:: bash

    # Add to ~/.zshrc
    _CLOAKPIVOT_COMPLETE=zsh_source cloakpivot > ~/.cloakpivot-complete.zsh
    source ~/.cloakpivot-complete.zsh

Fish Completion
~~~~~~~~~~~~~~~

.. code-block:: bash

    # Install completion
    _CLOAKPIVOT_COMPLETE=fish_source cloakpivot > ~/.config/fish/completions/cloakpivot.fish

Exit Codes
----------

CloakPivot uses standard exit codes:

* **0** - Success
* **1** - General error
* **2** - Misuse of shell command (invalid arguments)
* **64** - Command line usage error
* **65** - Data format error
* **66** - Cannot open input file
* **73** - Cannot create output file
* **74** - I/O error
* **75** - Temporary failure

Getting Help
------------

Each command and subcommand provides detailed help:

.. code-block:: bash

    # General help
    $ cloakpivot --help
    
    # Command-specific help
    $ cloakpivot mask --help
    $ cloakpivot policy --help
    $ cloakpivot policy create --help
    
    # List all available commands
    $ cloakpivot --help | grep Commands -A 20

For more detailed information, refer to the specific command documentation pages.