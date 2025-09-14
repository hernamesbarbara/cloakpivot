#!/usr/bin/env python3
"""Test CloakEngine integration with real document masking."""

from cloakpivot.engine import CloakEngine
from docling_core.types import DoclingDocument
from docling_core.types.doc.document import TextItem

def test_basic_masking():
    """Test basic masking functionality."""
    # Create a proper test document
    doc = DoclingDocument(name="test_doc.txt")

    # Add text with PII
    text_item = TextItem(
        text="My email is john.doe@example.com and my phone is 555-123-4567.",
        self_ref="#/texts/0",
        label="text",
        orig="My email is john.doe@example.com and my phone is 555-123-4567."
    )
    doc.texts = [text_item]

    # Test CloakEngine
    engine = CloakEngine()
    print("✓ CloakEngine created successfully")

    # Try masking
    result = engine.mask_document(doc)
    print(f"✓ Masking completed. Found {result.entities_found} entities, masked {result.entities_masked}")

    # Check the masked document
    masked_text = result.document.texts[0].text
    print(f"Original: {text_item.text}")
    print(f"Masked:   {masked_text}")

    # Verify masking worked
    assert "john.doe@example.com" not in masked_text
    assert "555-123-4567" not in masked_text
    assert "[EMAIL]" in masked_text or "EMAIL" in masked_text
    assert "[PHONE]" in masked_text or "PHONE" in masked_text

    # Test unmasking
    unmasked_doc = engine.unmask_document(result.document, result.cloakmap)
    unmasked_text = unmasked_doc.texts[0].text
    print(f"Unmasked: {unmasked_text}")

    # Verify unmasking restored original
    assert unmasked_text == text_item.text
    print("✓ Unmasking successfully restored original text")

    return result

def test_specific_entities():
    """Test masking specific entity types only."""
    doc = DoclingDocument(name="test_doc.txt")

    text_item = TextItem(
        text="John Smith works at 123 Main St. Email: john@example.com",
        self_ref="#/texts/0",
        label="text",
        orig="John Smith works at 123 Main St. Email: john@example.com"
    )
    doc.texts = [text_item]

    engine = CloakEngine()

    # Mask only email addresses
    result = engine.mask_document(doc, entities=['EMAIL_ADDRESS'])
    masked_text = result.document.texts[0].text

    print(f"\nEmail-only masking:")
    print(f"Original: {text_item.text}")
    print(f"Masked:   {masked_text}")

    # Email should be masked, but name and address should not
    assert "john@example.com" not in masked_text
    assert "John Smith" in masked_text  # Name not masked
    assert "123 Main St" in masked_text  # Address not masked
    print("✓ Selective entity masking works correctly")

def test_custom_policy():
    """Test using a custom masking policy."""
    from cloakpivot.defaults import get_conservative_policy

    doc = DoclingDocument(name="test_doc.txt")

    text_item = TextItem(
        text="Contact john@example.com or call 555-123-4567",
        self_ref="#/texts/0",
        label="text",
        orig="Contact john@example.com or call 555-123-4567"
    )
    doc.texts = [text_item]

    engine = CloakEngine()
    conservative_policy = get_conservative_policy()

    # Mask with conservative policy
    result = engine.mask_document(doc, policy=conservative_policy)
    masked_text = result.document.texts[0].text

    print(f"\nConservative policy masking:")
    print(f"Original: {text_item.text}")
    print(f"Masked:   {masked_text}")

    # With conservative policy, entities should be completely removed
    assert "john@example.com" not in masked_text
    assert "555-123-4567" not in masked_text
    # The entities should be masked (either with asterisks or [REMOVED])
    print("✓ Custom policy masking works correctly")

if __name__ == "__main__":
    print("Testing CloakEngine Integration")
    print("=" * 50)

    try:
        # Run basic test
        test_basic_masking()

        # Run specific entities test
        test_specific_entities()

        # Run custom policy test
        test_custom_policy()

        print("\n" + "=" * 50)
        print("✅ All CloakEngine integration tests passed!")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()