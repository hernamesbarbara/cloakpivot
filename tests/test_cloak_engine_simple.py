"""Test simplified CloakEngine API."""

import pytest
from docling_core.types import DoclingDocument
from docling_core.types.doc.document import TextItem

from cloakpivot.engine import CloakEngine, MaskResult
from cloakpivot.core.policies import MaskingPolicy
from cloakpivot.core.strategies import Strategy, StrategyKind


class TestCloakEngineBasic:
    """Basic CloakEngine functionality tests."""

    @pytest.fixture
    def simple_document(self):
        """Create a simple document with PII."""
        doc = DoclingDocument(name="test_doc.txt")
        text_item = TextItem(
            text="Contact John Doe at john.doe@example.com or call 555-123-4567.",
            self_ref="#/texts/0",
            label="text",
            orig="Contact John Doe at john.doe@example.com or call 555-123-4567."
        )
        doc.texts = [text_item]
        return doc

    @pytest.fixture
    def multi_section_document(self):
        """Create a document with multiple text sections."""
        doc = DoclingDocument(name="multi_doc.txt")

        sections = [
            "Our CEO Jane Smith can be reached at jane@company.com",
            "For support, contact support@company.com or 1-800-555-1234",
            "Office address: 123 Main St, New York, NY 10001"
        ]

        doc.texts = []
        for i, text in enumerate(sections):
            text_item = TextItem(
                text=text,
                self_ref=f"#/texts/{i}",
                label="text",
                orig=text
            )
            doc.texts.append(text_item)

        return doc

    def test_default_initialization(self):
        """Test CloakEngine initializes with sensible defaults."""
        engine = CloakEngine()
        assert engine is not None
        assert engine.default_policy is not None
        assert engine.analyzer_config is not None

    def test_one_line_masking(self, simple_document):
        """Test simplest use case - one line masking."""
        engine = CloakEngine()
        result = engine.mask_document(simple_document)

        assert isinstance(result, MaskResult)
        assert result.entities_found > 0
        assert result.entities_masked > 0
        assert result.document is not None
        assert result.cloakmap is not None

    def test_masked_content_differs(self, simple_document):
        """Test that masking actually changes the content."""
        engine = CloakEngine()
        result = engine.mask_document(simple_document)

        original_text = simple_document.texts[0].text
        masked_text = result.document.texts[0].text

        assert masked_text != original_text
        assert "john.doe@example.com" not in masked_text
        assert "555-123-4567" not in masked_text

    def test_unmask_roundtrip(self, simple_document):
        """Test masking/unmasking preserves original content."""
        engine = CloakEngine()

        # Mask the document
        result = engine.mask_document(simple_document)

        # Unmask it
        unmasked = engine.unmask_document(result.document, result.cloakmap)

        # Verify content is restored
        assert unmasked.texts[0].text == simple_document.texts[0].text

    def test_selective_entity_masking(self, simple_document):
        """Test masking only specific entity types."""
        engine = CloakEngine()

        # Mask only email addresses
        result = engine.mask_document(simple_document, entities=['EMAIL_ADDRESS'])
        masked_text = result.document.texts[0].text

        # Email should be masked
        assert "john.doe@example.com" not in masked_text

        # Phone might not be masked (depends on entity list)
        # Name might not be masked
        assert result.entities_found >= 1

    def test_empty_document(self):
        """Test handling of empty document."""
        doc = DoclingDocument(name="empty.txt")
        doc.texts = []

        engine = CloakEngine()
        result = engine.mask_document(doc)

        assert result.entities_found == 0
        assert result.entities_masked == 0
        assert len(result.cloakmap.anchors) == 0

    def test_no_pii_document(self):
        """Test document with no PII."""
        doc = DoclingDocument(name="no_pii.txt")
        text_item = TextItem(
            text="This is a simple text with no personal information.",
            self_ref="#/texts/0",
            label="text",
            orig="This is a simple text with no personal information."
        )
        doc.texts = [text_item]

        engine = CloakEngine()
        result = engine.mask_document(doc)

        # Should handle gracefully with no entities found
        assert result.entities_found == 0
        assert result.entities_masked == 0
        assert result.document.texts[0].text == doc.texts[0].text

    def test_multi_section_masking(self, multi_section_document):
        """Test masking across multiple document sections."""
        engine = CloakEngine()
        result = engine.mask_document(multi_section_document)

        # Should find entities across all sections
        assert result.entities_found > 0

        # Check each section was processed
        for i, text_item in enumerate(result.document.texts):
            original = multi_section_document.texts[i].text
            masked = text_item.text

            # At least some sections should be different
            if "@" in original or "555" in original or "800" in original:
                assert masked != original

    def test_custom_policy(self, simple_document):
        """Test using a custom masking policy."""
        # Create a custom policy that uses REDACT for everything
        custom_policy = MaskingPolicy(
            per_entity={},
            default_strategy=Strategy(
                kind=StrategyKind.REDACT,
                parameters={"replacement": "[REMOVED]"}
            )
        )

        engine = CloakEngine(default_policy=custom_policy)
        result = engine.mask_document(simple_document)

        masked_text = result.document.texts[0].text

        # Should have masked something
        assert result.entities_found > 0
        # Original PII should be gone
        assert "john.doe@example.com" not in masked_text

    def test_confidence_threshold_config(self, simple_document):
        """Test analyzer configuration with confidence threshold."""
        # High confidence threshold might find fewer entities
        high_conf_config = {
            'confidence_threshold': 0.95
        }

        engine = CloakEngine(analyzer_config=high_conf_config)
        result = engine.mask_document(simple_document)

        # Should still work, might find fewer entities
        assert result is not None
        assert result.document is not None

    def test_entity_types_list(self, simple_document):
        """Test with different entity type combinations."""
        engine = CloakEngine()

        # Test with multiple entity types
        entity_types = ['EMAIL_ADDRESS', 'PHONE_NUMBER', 'PERSON']
        result = engine.mask_document(simple_document, entities=entity_types)

        assert result.entities_found > 0
        masked_text = result.document.texts[0].text

        # Known entities should be masked
        assert "john.doe@example.com" not in masked_text

    def test_result_structure(self, simple_document):
        """Test the MaskResult structure contains expected fields."""
        engine = CloakEngine()
        result = engine.mask_document(simple_document)

        # Check MaskResult attributes
        assert hasattr(result, 'document')
        assert hasattr(result, 'cloakmap')
        assert hasattr(result, 'entities_found')
        assert hasattr(result, 'entities_masked')

        # Check types
        assert isinstance(result.document, DoclingDocument)
        assert result.entities_found >= 0
        assert result.entities_masked >= 0
        assert result.entities_masked <= result.entities_found

    def test_preserve_document_structure(self, multi_section_document):
        """Test that document structure is preserved during masking."""
        engine = CloakEngine()
        result = engine.mask_document(multi_section_document)

        # Same number of text sections
        assert len(result.document.texts) == len(multi_section_document.texts)

        # Same document name
        assert result.document.name == multi_section_document.name

        # Text items maintain their references
        for i in range(len(result.document.texts)):
            assert result.document.texts[i].self_ref == multi_section_document.texts[i].self_ref

    def test_cloakmap_contains_anchors(self, simple_document):
        """Test that CloakMap contains proper anchor entries."""
        engine = CloakEngine()
        result = engine.mask_document(simple_document)

        if result.entities_found > 0:
            assert len(result.cloakmap.anchors) > 0

            # Check anchor has required fields
            anchor = result.cloakmap.anchors[0]
            assert hasattr(anchor, 'node_id')
            assert hasattr(anchor, 'start')
            assert hasattr(anchor, 'end')
            assert hasattr(anchor, 'entity_type')
            assert hasattr(anchor, 'masked_value')

    def test_unmask_with_wrong_cloakmap(self, simple_document):
        """Test unmasking with mismatched CloakMap."""
        engine = CloakEngine()

        # Mask two different documents
        result1 = engine.mask_document(simple_document)

        doc2 = DoclingDocument(name="other.txt")
        doc2.texts = [TextItem(
            text="Different content here",
            self_ref="#/texts/0",
            label="text",
            orig="Different content here"
        )]
        result2 = engine.mask_document(doc2)

        # Try to unmask with wrong cloakmap - should handle gracefully
        # This might restore wrong content or handle the mismatch
        unmasked = engine.unmask_document(result1.document, result2.cloakmap)
        assert unmasked is not None  # Should not crash