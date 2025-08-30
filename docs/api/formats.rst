Format Handling API Reference
=============================

The format handling API provides serialization and conversion capabilities for different document formats.

.. currentmodule:: cloakpivot.formats

Serialization
-------------

.. automodule:: cloakpivot.formats.serialization
   :members:
   :undoc-members:
   :show-inheritance:

Format Registry
---------------

.. automodule:: cloakpivot.formats.registry
   :members:
   :undoc-members:
   :show-inheritance:

Supported Formats
-----------------

CloakPivot supports the following document formats through DocPivot integration:

Input Formats
~~~~~~~~~~~~~

* **docling.json**: DocPivot's native JSON format with full structure preservation
* **lexical.json**: Lexical editor format for rich text documents
* **markdown**: Markdown text format (*.md, *.markdown)
* **html**: HTML format with structural elements

Output Formats
~~~~~~~~~~~~~~

* **docling**: DocPivot JSON format (default for round-trip fidelity)
* **lexical**: Lexical JSON format for editor integration
* **markdown**: Markdown format for readable text output
* **html**: HTML format for web display
* **doctags**: Tagged format showing document structure

Format Detection
----------------

CloakPivot automatically detects document formats based on:

1. **File Extension**: .json, .md, .html extensions
2. **Naming Convention**: Files with .docling.json or .lexical.json patterns
3. **Content Analysis**: JSON structure inspection for format identification

Example Format Usage
--------------------

.. code-block:: python

    from cloakpivot.formats.serialization import CloakPivotSerializer
    
    serializer = CloakPivotSerializer()
    
    # Detect format
    format_type = serializer.detect_format("document.lexical.json")
    print(f"Detected format: {format_type}")
    
    # Convert between formats
    result = serializer.convert_format(
        input_path="document.docling.json",
        output_format="markdown",
        output_path="document.md"
    )
    
    # Get format information
    format_info = serializer.get_format_info("docling")
    print(f"Supported: {format_info['supported']}")
    print(f"Extensions: {format_info['extensions']}")

Format Conversion Examples
--------------------------

Converting from DocPivot JSON to Markdown:

.. code-block:: python

    from cloakpivot.formats.serialization import CloakPivotSerializer
    
    serializer = CloakPivotSerializer()
    result = serializer.convert_format(
        input_path="document.docling.json",
        output_format="markdown"
    )
    
    print(f"Converted to: {result.output_path}")
    print(f"Size: {result.size_kb:.1f} KB")

Converting from Markdown to HTML:

.. code-block:: python

    result = serializer.convert_format(
        input_path="document.md",
        output_format="html",
        output_path="document.html"
    )
    
    # Access conversion metadata
    metadata = result.metadata
    print(f"Document name: {metadata.get('document_name')}")
    print(f"Text items: {metadata.get('document_texts')}")

Round-trip Conversion:

.. code-block:: python

    # Original format is preserved in CloakMap metadata
    from cloakpivot import mask_document
    
    # Mask with format preservation
    result = mask_document(
        "document.lexical.json",
        output_format="docling",  # Temporary format
        policy="policy.yaml"
    )
    
    # Unmask back to original format (lexical)
    restored = unmask_document(
        result.masked_path,
        result.cloakmap_path,
        output_format="auto"  # Detects original format
    )