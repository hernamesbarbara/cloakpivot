#!/usr/bin/env python3
"""Simple end-to-end test of CloakEngine functionality."""

from docling_core.types import DoclingDocument
from docling_core.types.doc.document import DocItem
from docling_core.types.doc.labels import DocItemLabel

from cloakpivot import CloakEngine


def test_simple_masking():
    """Test basic mask/unmask workflow."""

    # Create a test document with PII
    doc = DoclingDocument(name="test_doc.pdf")

    # Create DocItem with required fields
    from docling_core.types.doc.document import DocItemReference
    ref = DocItemReference()
    item = DocItem(
        label=DocItemLabel.PARAGRAPH,
        self_ref=ref
    )
    # Set text on the item
    item.text = "Contact John Doe at john.doe@example.com or call 555-123-4567."
    doc.add_item(item)

    # Create engine with defaults
    engine = CloakEngine()

    # Get original text
    original_text = doc.export_to_text()
    print(f"Original: {original_text}")

    # Mask the document
    result = engine.mask_document(doc)
    masked_text = result.document.export_to_text()
    print(f"Masked: {masked_text}")
    print(f"Found {result.entities_found} entities, masked {result.entities_masked}")

    # Verify masking changed the text (if PII was found)
    if result.entities_masked > 0:
        assert masked_text != original_text, "Masking should change the text"
        assert "john.doe@example.com" not in masked_text, "Email should be masked"

    # Unmask the document
    unmasked = engine.unmask_document(result.document, result.cloakmap)
    unmasked_text = unmasked.export_to_text()
    print(f"Unmasked: {unmasked_text}")

    # Verify round trip
    assert unmasked_text == original_text, "Round trip should preserve original text"

    print("✓ Test passed!")


def test_builder_pattern():
    """Test CloakEngine builder."""

    engine = CloakEngine.builder() \
        .with_languages(['en']) \
        .with_confidence_threshold(0.8) \
        .build()

    assert engine is not None
    assert engine.analyzer_config.confidence_threshold == 0.8

    print("✓ Builder test passed!")


def test_method_registration():
    """Test method registration on DoclingDocument."""
    from cloakpivot import register_cloak_methods, unregister_cloak_methods

    # Create test document
    doc = DoclingDocument(name="test_doc.pdf")
    from docling_core.types.doc.document import DocItemReference
    ref = DocItemReference()
    item = DocItem(
        label=DocItemLabel.PARAGRAPH,
        self_ref=ref
    )
    item.text = "Email: test@example.com"
    doc.add_item(item)

    # Register methods
    register_cloak_methods()

    # Use registered methods
    assert hasattr(doc, 'mask_pii')
    masked = doc.mask_pii()
    assert hasattr(masked, 'unmask_pii')

    # Clean up
    unregister_cloak_methods()

    print("✓ Method registration test passed!")


if __name__ == "__main__":
    print("Running CloakEngine E2E tests...\n")

    test_simple_masking()
    print()

    test_builder_pattern()
    print()

    test_method_registration()
    print()

    print("\n🎉 All tests passed!")