#!/usr/bin/env python3
"""Test examples from the specification."""

from cloakpivot.engine import CloakEngine
from docling_core.types import DoclingDocument
from docling_core.types.doc.document import TextItem

def test_simple_example():
    """Test the simple usage example from the spec."""
    # Create a document
    doc = DoclingDocument(name="test.txt")
    text_item = TextItem(
        text="Contact John Doe at john.doe@example.com or 555-1234",
        self_ref="#/texts/0",
        label="text",
        orig="Contact John Doe at john.doe@example.com or 555-1234"
    )
    doc.texts = [text_item]

    # Simple usage with defaults
    engine = CloakEngine()
    result = engine.mask_document(doc)

    print("Simple example:")
    print(f"  Original: {text_item.text}")
    print(f"  Masked:   {result.document.texts[0].text}")
    print(f"  Entities found: {result.entities_found}")

    # Unmask
    original = engine.unmask_document(result.document, result.cloakmap)
    print(f"  Unmasked: {original.texts[0].text}")

    assert original.texts[0].text == text_item.text
    print("✓ Simple example works")

def test_builder_example():
    """Test the builder pattern example from the spec."""
    # Create test document (using English for now due to Presidio language limitations)
    doc = DoclingDocument(name="test.txt")
    text_item = TextItem(
        text="My email is john@example.com with high confidence",
        self_ref="#/texts/0",
        label="text",
        orig="My email is john@example.com with high confidence"
    )
    doc.texts = [text_item]

    # Advanced configuration with builder (using higher confidence threshold)
    engine = CloakEngine.builder()\
        .with_confidence_threshold(0.9)\
        .build()

    result = engine.mask_document(doc)

    print("\nBuilder example:")
    print(f"  Original: {text_item.text}")
    print(f"  Masked:   {result.document.texts[0].text}")
    print(f"  Entities found: {result.entities_found}")

    # Email should be detected with high confidence
    assert "john@example.com" not in result.document.texts[0].text
    print("✓ Builder pattern works")

def test_specific_entities_example():
    """Test masking specific entities only."""
    doc = DoclingDocument(name="test.txt")
    text_item = TextItem(
        text="John Smith's email is john@example.com and phone is 555-1234",
        self_ref="#/texts/0",
        label="text",
        orig="John Smith's email is john@example.com and phone is 555-1234"
    )
    doc.texts = [text_item]

    engine = CloakEngine()

    # Detect specific entities only
    result = engine.mask_document(doc, entities=['EMAIL_ADDRESS'])

    print("\nSpecific entities example:")
    print(f"  Original: {text_item.text}")
    print(f"  Masked:   {result.document.texts[0].text}")

    # Only email should be masked
    assert "john@example.com" not in result.document.texts[0].text
    assert "John Smith" in result.document.texts[0].text  # Name not masked
    assert "555-1234" in result.document.texts[0].text  # Phone not masked
    print("✓ Specific entity masking works")

if __name__ == "__main__":
    print("Testing Specification Examples")
    print("=" * 50)

    try:
        test_simple_example()
        test_builder_example()
        test_specific_entities_example()

        print("\n" + "=" * 50)
        print("✅ All specification examples work correctly!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()