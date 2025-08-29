Mask Command
============

The ``mask`` command is the core operation for applying PII masking to documents while preserving their structure and format.

.. contents::
   :local:
   :depth: 2

Synopsis
--------

.. code-block:: bash

    cloakpivot mask [OPTIONS] INPUT_PATH

Description
-----------

The ``mask`` command analyzes a document for PII entities using Presidio, applies configurable masking strategies based on a policy, and generates:

1. **Masked Document**: A version with PII replaced according to the policy
2. **CloakMap**: A secure mapping file that enables exact restoration of the original content

The masking process preserves document structure, formatting, and layout while ensuring that sensitive information is properly protected.

Arguments
---------

.. option:: INPUT_PATH

    Path to the input document to be masked. Supported formats include PDF files (converted via docpivot) and structured JSON formats (docling.json, lexical.json).

    **Examples:**
    
    * ``document.pdf``
    * ``report.docling.json``
    * ``article.lexical.json``

Options
-------

Output Options
~~~~~~~~~~~~~~

.. option:: --out, -o <path>

    Output path for the masked document. If not specified, defaults to ``INPUT_PATH.masked.FORMAT.json`` where FORMAT matches the specified format.

    **Examples:**
    
    .. code-block:: bash
    
        # Explicit output path
        cloakpivot mask document.pdf --out masked_document.json
        
        # Auto-generated: document.pdf.masked.lexical.json
        cloakpivot mask document.pdf

.. option:: --cloakmap <path>

    Path to save the CloakMap file. If not specified, defaults to ``INPUT_PATH.cloakmap.json``.

    .. code-block:: bash
    
        cloakpivot mask document.pdf --cloakmap secure_map.json

.. option:: --format <format>

    Output format for the masked document. 

    **Choices:** ``lexical``, ``docling``, ``markdown``, ``md``, ``html``, ``doctags``
    
    **Default:** ``lexical``

    .. code-block:: bash
    
        # Output as markdown
        cloakpivot mask document.pdf --format markdown --out masked.md
        
        # Output as HTML for web display
        cloakpivot mask document.pdf --format html --out masked.html

Policy Options
~~~~~~~~~~~~~~

.. option:: --policy <path>

    Path to a YAML policy file that defines masking strategies, thresholds, and rules.

    .. code-block:: bash
    
        cloakpivot mask document.pdf --policy healthcare-policy.yaml

    If not specified, uses the default policy with redaction strategy.

Detection Options
~~~~~~~~~~~~~~~~~

.. option:: --lang <code>

    Language code for PII analysis. Affects which recognizers are used and how entities are detected.

    **Default:** ``en`` (English)

    .. code-block:: bash
    
        # Spanish document analysis
        cloakpivot mask documento.pdf --lang es
        
        # French document analysis  
        cloakpivot mask document.pdf --lang fr

.. option:: --min-score <float>

    Minimum confidence score (0.0-1.0) for PII detection. Entities with lower confidence scores will be ignored.

    **Default:** ``0.5``

    .. code-block:: bash
    
        # High confidence only
        cloakpivot mask document.pdf --min-score 0.8
        
        # More sensitive detection
        cloakpivot mask document.pdf --min-score 0.3

Security Options
~~~~~~~~~~~~~~~~

.. option:: --encrypt

    Encrypt the CloakMap file for additional security. Requires external key management.

    .. code-block:: bash
    
        cloakpivot mask document.pdf --encrypt --key-id prod-key-2023

.. option:: --key-id <id>

    Specify the key ID for CloakMap encryption. Used with ``--encrypt``.

Examples
--------

Basic Usage
~~~~~~~~~~~

Mask a document with default settings:

.. code-block:: bash

    $ cloakpivot mask important-document.pdf
    ✅ Masking completed successfully!
       Masked document: important-document.pdf.masked.lexical.json
       CloakMap: important-document.pdf.cloakmap.json
       Entities processed: 15

Custom Policy and Output
~~~~~~~~~~~~~~~~~~~~~~~~

Use a custom policy and specify output paths:

.. code-block:: bash

    $ cloakpivot mask patient-records.pdf \
        --policy healthcare-policy.yaml \
        --out masked-records.json \
        --cloakmap records-map.json \
        --verbose
    📋 Loading policy: healthcare-policy.yaml
    ✓ Enhanced policy loaded successfully
       (with inheritance and validation support)
    🔍 Loading document: patient-records.pdf
    ✓ Loaded document: Patient Medical Records
      Text items: 245
      Tables: 12
    🔍 Detecting PII entities (min_score: 0.5, lang: en)
    ✓ Detected 42 entities
      PERSON: 8
      EMAIL_ADDRESS: 3
      PHONE_NUMBER: 5
      US_SSN: 2
      MEDICAL_LICENSE: 4
      DATE_TIME: 20
    📝 Extracted 156 text segments
    ✓ Masked 42 entities
    💾 Saving masked document: masked-records.json
    🗺️  Saving CloakMap: records-map.json
    ✅ Masking completed successfully!
       Masked document: masked-records.json
       CloakMap: records-map.json
       Entities processed: 42

High Security Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Mask with strict settings and encryption:

.. code-block:: bash

    $ cloakpivot mask confidential.pdf \
        --policy conservative-policy.yaml \
        --min-score 0.8 \
        --encrypt \
        --key-id security-key-2023 \
        --format docling
    🔍 Loading document: confidential.pdf
    📋 Loading policy: conservative-policy.yaml
    🔍 Detecting PII entities (min_score: 0.8, lang: en)
    ✓ Detected 23 entities
    ✓ Masked 23 entities
    🔒 Encrypting CloakMap with key: security-key-2023
    ✅ Masking completed successfully!
       Masked document: confidential.pdf.masked.docling.json
       CloakMap: confidential.pdf.cloakmap.json (encrypted)
       Entities processed: 23

Multi-language Document
~~~~~~~~~~~~~~~~~~~~~~~

Process a Spanish document:

.. code-block:: bash

    $ cloakpivot mask documento-personal.pdf \
        --lang es \
        --policy spanish-policy.yaml \
        --format markdown \
        --out documento-enmascarado.md
    🔍 Detecting PII entities (min_score: 0.5, lang: es)
    ✓ Detected 18 entities
      PERSONA: 5
      CORREO_ELECTRONICO: 2
      TELEFONO: 3
      DNI_ES: 4
      FECHA: 4
    ✓ Masked 18 entities
    ✅ Masking completed successfully!

Batch Processing
~~~~~~~~~~~~~~~~

Process multiple documents with consistent settings:

.. code-block:: bash

    # Process all PDFs in a directory
    $ find documents/ -name "*.pdf" -exec cloakpivot mask {} \
        --policy standard-policy.yaml \
        --format lexical \;

    # Or use xargs for better performance
    $ find documents/ -name "*.pdf" | xargs -I {} -P 4 cloakpivot mask {} \
        --policy standard-policy.yaml

Output Details
--------------

Masked Document Structure
~~~~~~~~~~~~~~~~~~~~~~~~

The masked document maintains the original structure while replacing PII entities:

.. code-block:: json

    {
      "name": "Important Document",
      "texts": [
        {
          "text": "Contact [PERSON_8A3F2E] at [EMAIL_REDACTED] for more information.",
          "type": "text",
          "node_id": "text_001"
        }
      ],
      "metadata": {
        "masking_applied": true,
        "masked_at": "2023-12-07T10:30:00Z",
        "policy_name": "healthcare-policy"
      }
    }

CloakMap Structure
~~~~~~~~~~~~~~~~~

The CloakMap contains the information needed for unmasking:

.. code-block:: json

    {
      "version": "1.0",
      "doc_id": "doc_123456",
      "doc_hash": "sha256:abc123...",
      "created_at": "2023-12-07T10:30:00Z",
      "anchors": [
        {
          "node_id": "text_001",
          "start": 8,
          "end": 21,
          "entity_type": "PERSON",
          "masked_value": "[PERSON_8A3F2E]",
          "strategy_used": "template",
          "original_checksum": "sha256:def456..."
        }
      ],
      "policy_snapshot": {
        "name": "healthcare-policy",
        "version": "1.2"
      }
    }

Error Handling
--------------

Common Issues and Solutions
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**No PII Detected**

.. code-block:: bash

    $ cloakpivot mask document.pdf
    ℹ️  No PII entities detected in document
    Continue with masking anyway? [y/N]:

**Policy Loading Errors**

.. code-block:: bash

    $ cloakpivot mask document.pdf --policy invalid.yaml
    ⚠️  Failed to load policy file: YAML parsing error at line 15
       Using default policy

**Format Detection Issues**

.. code-block:: bash

    $ cloakpivot mask unknown.json --verbose
    ⚠️  Could not detect input format for unknown.json
    📋 Detected input format: unknown
    # Processing continues with best-effort format detection

**Insufficient Permissions**

.. code-block:: bash

    $ cloakpivot mask restricted.pdf --out /protected/output.json
    ❌ Masking failed: Permission denied writing to /protected/output.json

Performance Considerations
--------------------------

For optimal performance when masking large documents:

1. **Use appropriate confidence thresholds**: Higher ``--min-score`` values reduce false positives and processing time
2. **Choose efficient output formats**: ``lexical`` format is generally fastest for round-trip operations
3. **Consider batch processing**: Process multiple documents in parallel using shell tools
4. **Monitor memory usage**: Very large documents may require chunking (handled automatically)

Integration with Other Tools
----------------------------

The mask command integrates well with other command-line tools:

.. code-block:: bash

    # Pipeline with find and parallel processing
    find . -name "*.pdf" | parallel cloakpivot mask {} --policy policy.yaml
    
    # Integration with git hooks
    git diff --name-only --cached | grep '\.pdf$' | xargs cloakpivot mask
    
    # Automated processing with make
    %.masked.json: %.pdf policy.yaml
        cloakpivot mask $< --policy policy.yaml --out $@

See Also
--------

* :doc:`unmask_command` - Restore original content from masked documents
* :doc:`policy_commands` - Create and manage masking policies
* :doc:`../policies/creating_policies` - Policy development guide
* :doc:`../examples/basic_usage` - Complete usage examples