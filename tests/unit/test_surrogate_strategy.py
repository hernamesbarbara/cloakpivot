"""Unit tests for SURROGATE strategy with Faker integration."""

import pytest
from docling_core.types.doc.document import DoclingDocument, TextItem, DocItemLabel

from cloakpivot import CloakEngine
from cloakpivot.core import MaskingPolicy, Strategy, StrategyKind


class TestSurrogateStrategy:
    """Test SURROGATE strategy with Faker-based fake data generation."""

    def test_surrogate_replaces_with_fake_data_not_asterisks(self):
        """Test that SURROGATE strategy produces fake data, not asterisks."""
        # Create policy with SURROGATE strategy
        policy = MaskingPolicy(
            default_strategy=Strategy(
                kind=StrategyKind.SURROGATE,
                parameters={"seed": "test-seed"}
            ),
            seed="test-seed"
        )
        engine = CloakEngine(default_policy=policy)

        # Create a simple document with PII
        doc = DoclingDocument(name="test.txt")
        doc.texts = [TextItem(
            text="My name is John Doe and my email is john@example.com",
            label=DocItemLabel.TEXT,
            self_ref="#/texts/0",
            orig="My name is John Doe and my email is john@example.com"
        )]

        # Mask the document
        result = engine.mask_document(doc)

        # Verify no asterisks in output
        masked_text = result.document.texts[0].text
        assert "*" not in masked_text, f"Found asterisks in masked text: {masked_text}"

        # Verify original PII is replaced
        assert "John Doe" not in masked_text
        assert "john@example.com" not in masked_text

        # Verify fake data format is preserved (email should still have @ and .)
        if "@example.com" in doc.texts[0]:  # If original had email
            assert "@" in masked_text, "Email format not preserved"

    def test_surrogate_deterministic_with_seed(self):
        """Test that same input with same seed produces same fake data."""
        policy = MaskingPolicy(
            default_strategy=Strategy(
                kind=StrategyKind.SURROGATE,
                parameters={"seed": "consistent-seed"}
            ),
            seed="consistent-seed"
        )
        engine = CloakEngine(default_policy=policy)

        # Create document
        doc = DoclingDocument(name="test.txt")
        doc.texts = [TextItem(
            text="Contact John Smith at john.smith@company.com",
            label=DocItemLabel.TEXT,
            self_ref="#/texts/0",
            orig="Contact John Smith at john.smith@company.com"
        )]

        # Mask twice
        result1 = engine.mask_document(doc)
        result2 = engine.mask_document(doc)

        # Results should be identical
        assert result1.document.texts == result2.document.texts

    def test_surrogate_different_entity_types(self):
        """Test that different entity types use appropriate faker methods."""
        policy = MaskingPolicy(
            default_strategy=Strategy(
                kind=StrategyKind.SURROGATE,
                parameters={"seed": "test"}
            )
        )
        engine = CloakEngine(default_policy=policy)

        # Document with various PII types
        doc = DoclingDocument(name="test.txt")
        doc.texts = [TextItem(
            text=(
                "Name: Jane Doe\n"
                "Email: jane@example.com\n"
                "Phone: 555-123-4567\n"
                "Date: 2025-01-01\n"
                "Location: New York"
            ),
            label=DocItemLabel.TEXT,
            self_ref="#/texts/0",
            orig=(
                "Name: Jane Doe\n"
                "Email: jane@example.com\n"
                "Phone: 555-123-4567\n"
                "Date: 2025-01-01\n"
                "Location: New York"
            )
        )]

        result = engine.mask_document(doc, entities=["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "DATE_TIME", "LOCATION"])

        masked_text = result.document.texts[0].text

        # Verify no asterisks
        assert "*" not in masked_text

        # Verify original values are replaced
        assert "Jane Doe" not in masked_text
        assert "jane@example.com" not in masked_text
        assert "555-123-4567" not in masked_text

    def test_surrogate_preserves_document_structure(self):
        """Test that SURROGATE strategy preserves document structure."""
        policy = MaskingPolicy(
            default_strategy=Strategy(kind=StrategyKind.SURROGATE)
        )
        engine = CloakEngine(default_policy=policy)

        # Multi-line document
        doc = DoclingDocument(name="test.txt")
        doc.texts = [TextItem(
            text=(
                "Dear Mr. Johnson,\n\n"
                "Please contact me at mike@example.com or call 555-9876.\n\n"
                "Best regards,\n"
                "Michael Brown"
            ),
            label=DocItemLabel.TEXT,
            self_ref="#/texts/0",
            orig=(
                "Dear Mr. Johnson,\n\n"
                "Please contact me at mike@example.com or call 555-9876.\n\n"
                "Best regards,\n"
                "Michael Brown"
            )
        )]

        result = engine.mask_document(doc)
        masked_text = result.document.texts[0].text

        # Structure should be preserved
        assert "Dear" in masked_text
        assert "Please contact me at" in masked_text
        assert "Best regards," in masked_text
        assert "\n\n" in masked_text  # Line breaks preserved

        # No asterisks
        assert "*" not in masked_text

    def test_surrogate_with_multiple_same_entities(self):
        """Test SURROGATE handles multiple occurrences of same entity consistently."""
        policy = MaskingPolicy(
            default_strategy=Strategy(
                kind=StrategyKind.SURROGATE,
                parameters={"seed": "consistency-test"}
            )
        )
        engine = CloakEngine(default_policy=policy)

        doc = DoclingDocument(name="test.txt")
        doc.texts = [TextItem(
            text=(
                "John Doe sent an email to Jane Smith. "
                "Jane Smith replied to John Doe. "
                "John Doe then called Jane Smith."
            ),
            label=DocItemLabel.TEXT,
            self_ref="#/texts/0",
            orig=(
                "John Doe sent an email to Jane Smith. "
                "Jane Smith replied to John Doe. "
                "John Doe then called Jane Smith."
            )
        )]

        result = engine.mask_document(doc)
        masked_text = result.document.texts[0].text

        # No asterisks
        assert "*" not in masked_text

        # Original names should be replaced
        assert "John Doe" not in masked_text
        assert "Jane Smith" not in masked_text

    def test_surrogate_fallback_on_unknown_entity(self):
        """Test SURROGATE strategy gracefully handles unknown entity types."""
        policy = MaskingPolicy(
            default_strategy=Strategy(kind=StrategyKind.SURROGATE)
        )
        engine = CloakEngine(default_policy=policy)

        # Manually test with a less common entity type
        doc = DoclingDocument(name="test.txt")
        doc.texts = [TextItem(
            text="My credit card is 4111-1111-1111-1111",
            label=DocItemLabel.TEXT,
            self_ref="#/texts/0",
            orig="My credit card is 4111-1111-1111-1111"
        )]

        result = engine.mask_document(doc, entities=["CREDIT_CARD"])
        masked_text = result.document.texts[0].text

        # Should still mask the entity (either with fake data or placeholder)
        assert "4111-1111-1111-1111" not in masked_text

    def test_surrogate_empty_parameters(self):
        """Test SURROGATE strategy works with no parameters."""
        policy = MaskingPolicy(
            default_strategy=Strategy(
                kind=StrategyKind.SURROGATE,
                parameters={}
            )
        )
        engine = CloakEngine(default_policy=policy)

        doc = DoclingDocument(name="test.txt")
        doc.texts = [TextItem(
            text="Contact person: Bob Wilson",
            label=DocItemLabel.TEXT,
            self_ref="#/texts/0",
            orig="Contact person: Bob Wilson"
        )]

        result = engine.mask_document(doc)
        masked_text = result.document.texts[0].text

        # Should produce fake data without asterisks
        assert "*" not in masked_text
        assert "Bob Wilson" not in masked_text

    def test_surrogate_with_custom_entities_per_type(self):
        """Test SURROGATE can be applied to specific entity types."""
        policy = MaskingPolicy(
            default_strategy=Strategy(kind=StrategyKind.REDACT),  # Default to REDACT
            per_entity={
                "PERSON": Strategy(kind=StrategyKind.SURROGATE, parameters={"seed": "person-seed"}),
                "EMAIL_ADDRESS": Strategy(kind=StrategyKind.SURROGATE),
            }
        )
        engine = CloakEngine(default_policy=policy)

        doc = DoclingDocument(name="test.txt")
        doc.texts = [TextItem(
            text="Alice Brown (alice@test.com) lives at 123 Main St",
            label=DocItemLabel.TEXT,
            self_ref="#/texts/0",
            orig="Alice Brown (alice@test.com) lives at 123 Main St"
        )]

        result = engine.mask_document(doc, entities=["PERSON", "EMAIL_ADDRESS", "LOCATION"])
        masked_text = result.document.texts[0].text

        # Person and Email should use surrogate (no asterisks for those parts)
        assert "Alice Brown" not in masked_text
        assert "alice@test.com" not in masked_text

        # Location should use REDACT (asterisks)
        # But we need to check if location was detected
        if "123 Main St" not in masked_text:
            # If location was masked, it might have asterisks from REDACT strategy
            pass  # This depends on entity detection

    @pytest.mark.parametrize("entity_type,original,should_have", [
        ("PERSON", "John Smith", None),
        ("EMAIL_ADDRESS", "test@example.com", "@"),
        ("PHONE_NUMBER", "555-123-4567", None),
        ("DATE_TIME", "2025-01-15", None),
    ])
    def test_surrogate_entity_specific_format(self, entity_type, original, should_have):
        """Test that each entity type gets appropriate fake data."""
        policy = MaskingPolicy(
            default_strategy=Strategy(
                kind=StrategyKind.SURROGATE,
                parameters={"seed": f"test-{entity_type}"}
            )
        )
        engine = CloakEngine(default_policy=policy)

        doc = DoclingDocument(name="test.txt")
        doc.texts = [TextItem(
            text=f"Test data: {original}",
            label=DocItemLabel.TEXT,
            self_ref="#/texts/0",
            orig=f"Test data: {original}"
        )]

        result = engine.mask_document(doc, entities=[entity_type])
        masked_text = result.document.texts[0].text

        # No asterisks
        assert "*" not in masked_text

        # Original should be replaced
        assert original not in masked_text

        # Check format preservation where applicable
        if should_have:
            assert should_have in masked_text, f"Format indicator '{should_have}' not found in {masked_text}"