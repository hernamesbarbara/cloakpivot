#!/usr/bin/env python3
"""Simple usage example of CloakEngine for PII masking.

This example demonstrates the simplest way to use CloakPivot:
1. Convert a document using Docling
2. Mask PII with one line using CloakEngine
3. Save the masked document and CloakMap
4. Unmask to recover the original
"""

from pathlib import Path
from docling.document_converter import DocumentConverter
from cloakpivot import CloakEngine


CLOAKPIVOT_ROOT = Path(__file__).parent.parent
DATA_DIR = CLOAKPIVOT_ROOT / "data"


def main():
    """Demonstrate simple PII masking workflow."""
    print("=" * 60)
    print("CloakPivot Simple Usage Example")
    print("=" * 60)
    
    # Step 1: Convert document using Docling
    print("\n1. Converting document...")
    converter = DocumentConverter()
    
    # You can use any supported document format
    # For testing, you could create a simple markdown file:
    test_doc = Path("test_document.md")
    if not test_doc.exists():
        test_doc.write_text("""
# Employee Information

John Smith works as our lead developer. You can reach him at:
- Email: john.smith@company.com
- Phone: 555-123-4567
- Employee ID: EMP-2023-001

For payroll, his SSN is 123-45-6789.

Emergency contact: Jane Doe (555-987-6543)
        """)
        print(f"  Created test document: {test_doc}")
    
    result = converter.convert(str(test_doc))
    doc = result.document
    print(f"  ✓ Converted document: {doc.name if hasattr(doc, 'name') else 'document'}")
    print(f"  ✓ Text items: {len(doc.texts)}")
    
    # Step 2: Mask PII with CloakEngine (one line!)
    print("\n2. Masking PII entities...")
    engine = CloakEngine()  # Uses smart defaults
    masked_result = engine.mask_document(doc)
    
    print(f"  ✓ Found {masked_result.entities_found} PII entities")
    print(f"  ✓ Masked {masked_result.entities_masked} entities")
    
    # Show masked content preview
    masked_text = masked_result.document.export_to_markdown()
    print("\n3. Masked content preview:")
    print("-" * 40)
    print(masked_text[:500] + "..." if len(masked_text) > 500 else masked_text)
    print("-" * 40)
    
    # Step 3: Save masked document and CloakMap
    print("\n4. Saving results...")
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # Save masked document
    masked_path = output_dir / "masked_document.md"
    masked_path.write_text(masked_result.document.export_to_markdown())
    print(f"  ✓ Saved masked document: {masked_path}")
    
    # Save CloakMap for unmasking later
    cloakmap_path = output_dir / "document.cloakmap.json"
    masked_result.cloakmap.save_to_file(cloakmap_path)
    print(f"  ✓ Saved CloakMap: {cloakmap_path}")
    print(f"    (Contains {len(masked_result.cloakmap.anchors)} anchors for unmasking)")
    
    # Step 4: Demonstrate unmasking
    print("\n5. Unmasking document...")
    unmasked_doc = engine.unmask_document(
        masked_result.document, 
        masked_result.cloakmap
    )
    
    # Verify original content restored
    original_text = doc.export_to_markdown()
    recovered_text = unmasked_doc.export_to_markdown()
    
    if original_text == recovered_text:
        print("  ✓ Successfully restored original document!")
    else:
        print("  ⚠ Warning: Unmasked document differs from original")
    
    # Cleanup temporary file
    if test_doc.exists():
        test_doc.unlink()

    print("\n" + "=" * 60)
    print("✓ Example completed successfully!")
    print("=" * 60)
    print("\nKey takeaways:")
    print("  • CloakEngine provides one-line masking: engine.mask_document(doc)")
    print("  • Smart defaults handle common PII types automatically")
    print("  • CloakMap enables perfect round-trip masking/unmasking")
    print("  • Works with any Docling-supported document format")


if __name__ == "__main__":
    main()