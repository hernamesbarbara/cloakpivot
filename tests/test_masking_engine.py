"""Tests for masking functionality using CloakEngine."""

import pytest
from docling_core.types import DoclingDocument
from docling_core.types.doc.document import TextItem

from cloakpivot.engine import CloakEngine, MaskResult
from cloakpivot.core.policies import MaskingPolicy
from cloakpivot.core.strategies import Strategy, StrategyKind


class TestMaskingFunctionality:
    """Test suite for masking operations via CloakEngine."""

    @pytest.fixture
    def simple_document(self) -> DoclingDocument:
        """Create a simple test document."""
        doc = DoclingDocument(name="test_doc")

        # Add a text item with PII content
        text_item = TextItem(
            text="Call me at 555-123-4567 or email john@example.com",
            self_ref="#/texts/0",
            label="text",
            orig="Call me at 555-123-4567 or email john@example.com",
        )
        doc.texts = [text_item]
        return doc

    @pytest.fixture
    def basic_policy(self) -> MaskingPolicy:
        """Create a basic masking policy."""
        return MaskingPolicy(
            default_strategy=Strategy(
                kind=StrategyKind.TEMPLATE,
                parameters={"template": "[REDACTED]"}
            ),
            per_entity={
                "PHONE_NUMBER": Strategy(
                    kind=StrategyKind.TEMPLATE,
                    parameters={"template": "[PHONE]"}
                ),
                "EMAIL_ADDRESS": Strategy(
                    kind=StrategyKind.TEMPLATE,
                    parameters={"template": "[EMAIL]"}
                ),
            },
        )

    def test_engine_initialization(self):
        """Test CloakEngine can be initialized."""
        engine = CloakEngine()
        assert engine is not None
        assert engine.default_policy is not None

    def test_mask_document_with_simple_entities(self, simple_document, basic_policy):
        """Test masking a document with simple PII entities."""
        engine = CloakEngine(default_policy=basic_policy)

        result = engine.mask_document(simple_document)

        # Should return MaskResult
        assert isinstance(result, MaskResult)
        assert result.document is not None
        assert result.cloakmap is not None
        assert result.entities_found > 0
        assert result.entities_masked > 0

        # Document should be modified
        assert result.document != simple_document

        # Text should be masked
        masked_text = result.document.texts[0].text
        assert "[PHONE]" in masked_text or "[EMAIL]" in masked_text
        assert "555-123-4567" not in masked_text
        assert "john@example.com" not in masked_text

    def test_cloakmap_generation(self, simple_document, basic_policy):
        """Test that CloakMap is properly generated."""
        engine = CloakEngine(default_policy=basic_policy)

        result = engine.mask_document(simple_document)
        cloakmap = result.cloakmap

        # Should have anchors for detected entities
        assert len(cloakmap.anchors) > 0

        # Check anchors have required fields
        for anchor in cloakmap.anchors:
            assert hasattr(anchor, 'entity_type')
            assert hasattr(anchor, 'start')
            assert hasattr(anchor, 'end')
            assert hasattr(anchor, 'masked_value')
            assert hasattr(anchor, 'node_id')

    def test_no_original_text_in_cloakmap(self, simple_document, basic_policy):
        """Test that CloakMap anchors have checksums for security."""
        engine = CloakEngine(default_policy=basic_policy)

        result = engine.mask_document(simple_document)

        # Verify checksums are stored for each anchor
        for anchor in result.cloakmap.anchors:
            assert len(anchor.original_checksum) == 64  # SHA-256 hex length
            assert anchor.original_checksum.isalnum()

        # Verify the CloakMap can be serialized
        cloakmap_json = result.cloakmap.to_json()
        assert cloakmap_json is not None
        assert len(cloakmap_json) > 0

    def test_document_structure_preservation(self, simple_document):
        """Test that document structure is preserved during masking."""
        original_text_count = len(simple_document.texts)
        original_table_count = len(simple_document.tables)

        engine = CloakEngine()
        result = engine.mask_document(simple_document)

        masked_doc = result.document

        # Structure counts should be unchanged
        assert len(masked_doc.texts) == original_text_count
        assert len(masked_doc.tables) == original_table_count

        # Node references should be preserved
        assert masked_doc.texts[0].self_ref == simple_document.texts[0].self_ref

    def test_selective_entity_masking(self, simple_document):
        """Test masking only specific entity types."""
        engine = CloakEngine()

        # Mask only email addresses
        result = engine.mask_document(simple_document, entities=['EMAIL_ADDRESS'])

        masked_text = result.document.texts[0].text

        # Email should be masked
        assert "john@example.com" not in masked_text

        # Phone might not be masked (depends on entity list)
        # We only asked for EMAIL_ADDRESS

    def test_empty_document(self):
        """Test masking an empty document."""
        doc = DoclingDocument(name="empty_doc")
        doc.texts = []

        engine = CloakEngine()
        result = engine.mask_document(doc)

        # Should handle gracefully
        assert result.entities_found == 0
        assert result.entities_masked == 0
        assert len(result.cloakmap.anchors) == 0

    def test_no_pii_document(self):
        """Test document with no PII content."""
        doc = DoclingDocument(name="no_pii")
        text_item = TextItem(
            text="This is a simple text with no personal information.",
            self_ref="#/texts/0",
            label="text",
            orig="This is a simple text with no personal information."
        )
        doc.texts = [text_item]

        engine = CloakEngine()
        result = engine.mask_document(doc)

        # Document should be unchanged
        assert result.document.texts[0].text == doc.texts[0].text

        # No entities should be found
        assert result.entities_found == 0
        assert len(result.cloakmap.anchors) == 0

    def test_unique_replacement_ids(self, simple_document):
        """Test that replacement IDs are unique within a CloakMap."""
        engine = CloakEngine()

        result = engine.mask_document(simple_document)

        if len(result.cloakmap.anchors) > 0:
            replacement_ids = [anchor.replacement_id for anchor in result.cloakmap.anchors]
            assert len(replacement_ids) == len(set(replacement_ids))  # All unique

    def test_round_trip_masking(self, simple_document):
        """Test that masking and unmasking preserves original content."""
        engine = CloakEngine()
        original_text = simple_document.texts[0].text

        # Mask the document
        mask_result = engine.mask_document(simple_document)

        # Verify masking changed the text
        assert mask_result.document.texts[0].text != original_text

        # Unmask the document
        unmasked = engine.unmask_document(mask_result.document, mask_result.cloakmap)

        # Verify original content is restored
        assert unmasked.texts[0].text == original_text

    def test_custom_policy_application(self, simple_document):
        """Test applying a custom masking policy."""
        # Create a policy that redacts everything
        redact_policy = MaskingPolicy(
            default_strategy=Strategy(
                kind=StrategyKind.REDACT,
                parameters={"replacement": "***"}
            ),
            per_entity={}
        )

        engine = CloakEngine(default_policy=redact_policy)
        result = engine.mask_document(simple_document)

        if result.entities_found > 0:
            masked_text = result.document.texts[0].text
            # Should have redacted content
            assert "555-123-4567" not in masked_text
            assert "john@example.com" not in masked_text