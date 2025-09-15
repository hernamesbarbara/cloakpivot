"""Round-trip masking/unmasking tests to ensure data integrity."""

import pytest
from docling_core.types import DoclingDocument

from cloakpivot.core.policies import MaskingPolicy, PrivacyLevel, Strategy, StrategyKind
from cloakpivot.engine import CloakEngine


class TestRoundTrip:
    """Test complete masking and unmasking cycles."""

    @pytest.mark.parametrize(
        "text",
        [
            "Simple email: test@example.com",
            "Phone: (555) 123-4567 and SSN: 123-45-6789",
            "Mixed: John Doe (john@email.com) lives at 123 Main St",
            "Multiple emails: alice@test.com, bob@test.com, charlie@test.com",
            "International phone: +44 20 7123 4567",
            "Credit card: 4111-1111-1111-1111 expires 12/25",
        ],
    )
    def test_simple_roundtrips(self, text: str):
        """Test roundtrip with various PII patterns."""
        doc = DoclingDocument(version="1.0.0", name="test", text=text)

        engine = CloakEngine()
        masked = engine.mask_document(doc)
        restored = engine.unmask_document(masked.document, masked.cloakmap)

        assert restored.text == text

    def test_empty_and_edge_cases(self):
        """Test edge cases like empty strings and special characters."""
        test_cases = [
            "",  # Empty
            " " * 10,  # Just spaces
            "\n\n\n",  # Just newlines
            "!@#$%^&*()",  # Special characters
            "🎉 Unicode 🦄 emojis 🔥",  # Unicode
            "Mixed\ttabs\nand\r\nnewlines",  # Mixed whitespace
        ]

        engine = CloakEngine()

        for text in test_cases:
            doc = DoclingDocument(version="1.0.0", name="test", text=text)
            masked = engine.mask_document(doc)
            restored = engine.unmask_document(masked.document, masked.cloakmap)
            assert restored.text == text, f"Failed for: {repr(text)}"

    @pytest.mark.parametrize(
        "strategy_kind",
        [
            StrategyKind.REDACT,
            StrategyKind.TEMPLATE,
            StrategyKind.PARTIAL,
            StrategyKind.HASH,
            StrategyKind.SURROGATE,
        ],
    )
    def test_different_strategies(self, strategy_kind: StrategyKind, sample_pii_text: str):
        """Test roundtrip with different masking strategies."""
        doc = DoclingDocument(version="1.0.0", name="test", text=sample_pii_text)

        # Create policy with specific strategy
        policy = MaskingPolicy(
            name="test_policy",
            default_strategy=Strategy(kind=strategy_kind),
        )

        engine = CloakEngine(default_policy=policy)
        masked = engine.mask_document(doc)
        restored = engine.unmask_document(masked.document, masked.cloakmap)

        assert restored.text == sample_pii_text

    def test_multiple_rounds(self, simple_text_document: DoclingDocument):
        """Test multiple rounds of masking and unmasking."""
        engine = CloakEngine()
        doc = simple_text_document

        # Multiple rounds should always return to original
        for round_num in range(3):
            masked = engine.mask_document(doc)
            doc = engine.unmask_document(masked.document, masked.cloakmap)

        assert doc.text == simple_text_document.text

    def test_overlapping_entities(self):
        """Test handling of overlapping entity detection."""
        # This can happen when multiple recognizers identify overlapping text
        text = "Contact me at john.smith@example.com or john.smith@company.org"
        doc = DoclingDocument(version="1.0.0", name="test", text=text)

        engine = CloakEngine()
        masked = engine.mask_document(doc)
        restored = engine.unmask_document(masked.document, masked.cloakmap)

        assert restored.text == text

    def test_repeated_pii(self):
        """Test documents where the same PII appears multiple times."""
        text = """
        Primary: john@example.com
        Secondary: jane@example.com
        CC: john@example.com  # Same as primary
        Reply-to: john@example.com  # Same again
        """
        doc = DoclingDocument(version="1.0.0", name="test", text=text)

        engine = CloakEngine()
        masked = engine.mask_document(doc)

        # All instances should be masked
        assert "john@example.com" not in masked.document.text
        assert "jane@example.com" not in masked.document.text

        # Roundtrip should work
        restored = engine.unmask_document(masked.document, masked.cloakmap)
        assert restored.text == text

    def test_document_structure_preservation(self):
        """Test that document structure (paragraphs, lists, etc.) is preserved."""
        text = """
        # Header

        Paragraph with email@test.com in it.

        - List item 1 with phone: 555-1234
        - List item 2 with SSN: 123-45-6789

        | Table | With | Data |
        |-------|------|------|
        | Cell  | test@example.com | Value |

        Final paragraph.
        """
        doc = DoclingDocument(version="1.0.0", name="test", text=text)

        engine = CloakEngine()
        masked = engine.mask_document(doc)
        restored = engine.unmask_document(masked.document, masked.cloakmap)

        assert restored.text == text
        # Structure markers should be preserved
        assert "# Header" in restored.text
        assert "- List item" in restored.text
        assert "| Table |" in restored.text

    def test_partial_masking_roundtrip(self, sample_pii_text: str):
        """Test that partial masking (e.g., showing last 4 digits) still roundtrips."""
        doc = DoclingDocument(version="1.0.0", name="test", text=sample_pii_text)

        policy = MaskingPolicy(
            name="partial_policy",
            per_entity={
                "CREDIT_CARD": Strategy(
                    kind=StrategyKind.PARTIAL,
                    parameters={"show_last": 4}
                ),
                "PHONE_NUMBER": Strategy(
                    kind=StrategyKind.PARTIAL,
                    parameters={"show_last": 4}
                ),
            },
        )

        engine = CloakEngine(default_policy=policy)
        masked = engine.mask_document(doc)

        # Even with partial masking, roundtrip should work
        restored = engine.unmask_document(masked.document, masked.cloakmap)
        assert restored.text == sample_pii_text

    def test_large_document_roundtrip(self):
        """Test roundtrip with a large document."""
        # Create a large document with repeated patterns
        base_text = """
        Name: Person {num}
        Email: person{num}@example.com
        Phone: 555-{num:04d}
        Address: {num} Main Street, City, State 12345

        """

        parts = [base_text.format(num=i) for i in range(100)]
        large_text = "".join(parts)

        doc = DoclingDocument(version="1.0.0", name="large", text=large_text)

        engine = CloakEngine()
        masked = engine.mask_document(doc)
        restored = engine.unmask_document(masked.document, masked.cloakmap)

        assert restored.text == large_text
        assert masked.entities_found > 0  # Should find many entities