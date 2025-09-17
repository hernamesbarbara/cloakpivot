#!/usr/bin/env python3
"""Complete PDF workflow example: Convert, Mask, and Unmask.

This example demonstrates a real-world workflow:
1. Convert a PDF to DoclingDocument format
2. Mask PII entities using CloakEngine
3. Save masked document and CloakMap
4. Load and unmask the document later

Required: Place a PDF file in data/pdf/ directory or modify the path below.
"""

import json
import html
from pathlib import Path
from docling.document_converter import DocumentConverter
from docling_core.types import DoclingDocument
from cloakpivot import CloakEngine
from cloakpivot.core.cloakmap import CloakMap


def convert_pdf_to_docling(pdf_path: Path, output_dir: Path):
    """Step 1: Convert PDF to DoclingDocument."""
    print("📄 Converting PDF to DoclingDocument...")

    converter = DocumentConverter()
    result = converter.convert(pdf_path)
    doc = result.document

    # Save DoclingDocument as JSON for later use
    doc_filename = pdf_path.stem
    docling_path = output_dir / f"{doc_filename}.docling.json"

    with open(docling_path, "w", encoding="utf-8") as f:
        json.dump(doc.export_to_dict(), f, indent=2)

    # Also save as markdown for human reading
    md_path = output_dir / f"{doc_filename}.original.md"
    original_markdown = html.unescape(doc.export_to_markdown())
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(original_markdown)

    print(f"  ✓ Saved DoclingDocument: {docling_path}")
    print(f"  ✓ Saved original markdown: {md_path}")

    # Validate original markdown is not empty
    if not original_markdown:
        raise ValueError("Original markdown export is empty - document may have structural issues")
    print(f"    (Original markdown: {len(original_markdown)} characters)")

    return doc, docling_path


def mask_document(doc: DoclingDocument, output_dir: Path):
    """Step 2: Mask PII in the document."""
    print("\n🔒 Masking PII entities...")

    # Initialize CloakEngine with default settings
    engine = CloakEngine()

    # Mask the document
    result = engine.mask_document(doc)

    print(f"  ✓ Found {result.entities_found} PII entities")
    print(f"  ✓ Masked {result.entities_masked} entities")

    # Save masked document
    doc_name = doc.name if hasattr(doc, 'name') else 'document'
    doc_filename = Path(doc_name).stem

    masked_json_path = output_dir / f"{doc_filename}.masked.json"
    with open(masked_json_path, "w", encoding="utf-8") as f:
        json.dump(result.document.export_to_dict(), f, indent=2)

    # Save masked markdown for viewing
    masked_md_path = output_dir / f"{doc_filename}.masked.md"
    masked_markdown = html.unescape(result.document.export_to_markdown())
    with open(masked_md_path, "w", encoding="utf-8") as f:
        f.write(masked_markdown)

    # Save CloakMap for unmasking
    cloakmap_path = output_dir / f"{doc_filename}.cloakmap.json"
    result.cloakmap.save_to_file(cloakmap_path)

    print(f"  ✓ Saved masked document: {masked_json_path}")
    print(f"  ✓ Saved masked markdown: {masked_md_path}")
    print(f"  ✓ Saved CloakMap: {cloakmap_path}")

    # Validate masked markdown is not empty
    if not masked_markdown:
        raise ValueError("❌ ERROR: Masked markdown export is empty - document structure was lost during masking")
    print(f"    (Masked markdown: {len(masked_markdown)} characters)")

    return result, cloakmap_path


def unmask_document(masked_doc_path: Path, cloakmap_path: Path, output_dir: Path):
    """Step 3: Unmask the document using CloakMap."""
    print("\n🔓 Unmasking document...")

    # Load masked document
    with open(masked_doc_path, "r", encoding="utf-8") as f:
        masked_dict = json.load(f)
    masked_doc = DoclingDocument(**masked_dict)

    # Load CloakMap
    cloakmap = CloakMap.load_from_file(cloakmap_path)
    print(f"  ✓ Loaded CloakMap with {len(cloakmap.anchors)} anchors")

    # Unmask using CloakEngine
    engine = CloakEngine()
    unmasked_doc = engine.unmask_document(masked_doc, cloakmap)

    # Save unmasked document
    doc_filename = masked_doc_path.stem.replace('.masked', '')
    unmasked_md_path = output_dir / f"{doc_filename}.unmasked.md"
    unmasked_markdown = html.unescape(unmasked_doc.export_to_markdown())
    with open(unmasked_md_path, "w", encoding="utf-8") as f:
        f.write(unmasked_markdown)

    print(f"  ✓ Saved unmasked document: {unmasked_md_path}")

    # Validate unmasked markdown is not empty
    if not unmasked_markdown:
        raise ValueError("❌ ERROR: Unmasked markdown export is empty - unmasking failed to restore document structure")
    print(f"    (Unmasked markdown: {len(unmasked_markdown)} characters)")

    return unmasked_doc


def main():
    """Run the complete PDF workflow."""
    print("=" * 60)
    print("CloakPivot PDF Workflow Example")
    print("=" * 60)

    # Setup paths
    pdf_path = Path("data/pdf/email.pdf")  # Using existing test PDF
    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if PDF exists, create a sample message if not
    if not pdf_path.exists():
        print(f"\n⚠️  PDF not found: {pdf_path}")
        print("\nTo run this example:")
        print("1. Create the directory: mkdir -p data/pdf/")
        print("2. Add a PDF file: cp your_document.pdf data/pdf/sample.pdf")
        print("3. Run this script again")
        print("\nAlternatively, modify the pdf_path variable to point to your PDF.")
        return

    # Step 1: Convert PDF
    doc, docling_path = convert_pdf_to_docling(pdf_path, output_dir)

    # Step 2: Mask PII
    mask_result, cloakmap_path = mask_document(doc, output_dir)

    # Step 3: Demonstrate loading and unmasking
    masked_json_path = output_dir / f"{pdf_path.stem}.masked.json"
    unmasked_doc = unmask_document(masked_json_path, cloakmap_path, output_dir)

    # Verify round-trip
    print("\n✅ Verification:")
    original_text = doc.export_to_markdown()
    masked_text = mask_result.document.export_to_markdown()
    restored_text = unmasked_doc.export_to_markdown()

    # Check if masking actually worked
    if original_text == masked_text:
        print("  ❌ ERROR: Masked text is identical to original - masking didn't work!")
    else:
        print(f"  ✓ Masking worked: Content was successfully masked")
        print(f"    Original: {len(original_text)} chars, Masked: {len(masked_text)} chars")

    if original_text == restored_text:
        print("  ✓ Perfect round-trip: Original content fully restored!")
    else:
        print("  ⚠ Content differs (check for formatting issues)")
        # Show a snippet of the difference for debugging
        import difflib
        diff = list(difflib.unified_diff(
            original_text[:200].splitlines(keepends=True),
            restored_text[:200].splitlines(keepends=True),
            fromfile='original',
            tofile='restored',
            n=1
        ))
        if diff:
            print("\n  First differences (showing first 200 chars):")
            print(''.join(diff[:10]))

    print("\n" + "=" * 60)
    print("Workflow complete! Check the output/ directory for:")
    print("  • Original document (markdown and JSON)")
    print("  • Masked document (markdown and JSON)")
    print("  • CloakMap (for unmasking)")
    print("  • Unmasked document (markdown)")
    print("=" * 60)


if __name__ == "__main__":
    main()