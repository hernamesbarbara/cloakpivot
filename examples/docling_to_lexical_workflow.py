#!/usr/bin/env python3
"""
Example: PDF to Docling JSON to Lexical JSON Workflow

This example demonstrates:
1. Using Docling to extract content from a PDF and save as Docling JSON
2. Using docpivot to load the Docling JSON and convert it to Lexical JSON

This shows how docpivot can seamlessly work with Docling's output format.
"""

import json
from pathlib import Path
from docling.document_converter import DocumentConverter
from docling.datamodel.document import ConversionResult
from docling_core.types import DoclingDocument
from cloakpivot.compat import load_document, to_lexical

def main():
    # Setup paths - use existing test PDFs
    pdf_file = Path("data/pdf/email.pdf")  # Using existing test PDF
    output_dir = Path("./output/")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("PDF → Docling JSON → Lexical JSON Workflow")
    print("=" * 60)

    # Step 1: Convert PDF to Docling JSON using Docling
    print("\n📄 Step 1: Converting PDF to Docling JSON")
    print(f"  Input: {pdf_file}")

    converter = DocumentConverter()
    conv_result: ConversionResult = converter.convert(pdf_file)

    # Get the document and metadata
    dl_doc: DoclingDocument = conv_result.document
    doc_filename = conv_result.input.file.stem

    # Save as Docling JSON
    docling_json_path = output_dir / f"{doc_filename}.docling.json"
    docling_dict = dl_doc.export_to_dict()

    with open(docling_json_path, "w", encoding="utf-8") as f:
        json.dump(docling_dict, f, indent=2)

    print(f"  ✓ Saved Docling JSON: {docling_json_path}")
    print(f"  - Document name: {dl_doc.name}")
    print(f"  - Number of pages: {len(dl_doc.pages) if dl_doc.pages else 'N/A'}")

    # Step 2: Load Docling JSON using docpivot
    print("\n🔄 Step 2: Loading Docling JSON with docpivot")
    print(f"  Input: {docling_json_path}")

    # Use docpivot to load the Docling JSON
    loaded_doc: DoclingDocument = load_document(docling_json_path)

    print(f"  ✓ Successfully loaded document")
    print(f"  - Document type: {type(loaded_doc).__name__}")
    print(f"  - Has content: {bool(loaded_doc.texts)}")

    # Step 3: Convert to Lexical JSON using docpivot
    print("\n🎯 Step 3: Converting to Lexical JSON format")

    # Convert the Docling document to Lexical format
    lexical_doc = to_lexical(loaded_doc)

    # Save as Lexical JSON
    lexical_json_path = output_dir / f"{doc_filename}.lexical.json"

    with open(lexical_json_path, "w", encoding="utf-8") as f:
        json.dump(lexical_doc, f, indent=2)

    print(f"  ✓ Saved Lexical JSON: {lexical_json_path}")

    # Display summary of Lexical structure
    if "root" in lexical_doc:
        root = lexical_doc["root"]
        if "children" in root:
            print(f"  - Root children: {len(root['children'])}")

            # Count node types
            node_types = {}
            for child in root["children"]:
                node_type = child.get("type", "unknown")
                node_types[node_type] = node_types.get(node_type, 0) + 1

            print("  - Node types:")
            for node_type, count in sorted(node_types.items()):
                print(f"    • {node_type}: {count}")

    # Step 4: Verify the conversion
    print("\n✅ Workflow Complete!")
    print("\nFiles created:")
    print(f"  1. Docling JSON: {docling_json_path}")
    print(f"  2. Lexical JSON: {lexical_json_path}")

    # Show a sample of the Lexical structure
    print("\n📋 Sample Lexical JSON structure:")
    sample = {
        "root": {
            "type": root.get("type"),
            "format": root.get("format"),
            "children_count": len(root.get("children", [])),
            "first_child": root["children"][0] if root.get("children") else None
        }
    }
    print(json.dumps(sample, indent=2)[:500] + "...")

    return docling_json_path, lexical_json_path

if __name__ == "__main__":
    try:
        docling_path, lexical_path = main()
        print(f"\n🎉 Success! You can now inspect the generated files:")
        print(f"   - {docling_path}")
        print(f"   - {lexical_path}")
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("Please ensure the PDF file exists at the specified path.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Please check that all required packages are installed:")
        print("  pip install docling docpivot")