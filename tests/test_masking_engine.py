"""Tests for the MaskingEngine core functionality."""

from typing import List

import pytest
from docling_core.types import DoclingDocument
from presidio_analyzer import RecognizerResult

from cloakpivot.core.policies import MaskingPolicy
from cloakpivot.core.strategies import PHONE_TEMPLATE, Strategy, StrategyKind
from cloakpivot.document.extractor import TextSegment
from cloakpivot.masking.engine import MaskingEngine, MaskingResult


class TestMaskingEngine:
    """Test suite for the MaskingEngine class."""

    @pytest.fixture
    def simple_document(self) -> DoclingDocument:
        """Create a simple test document."""
        from docling_core.types.doc.document import TextItem

        doc = DoclingDocument(name="test_doc")

        # Add a text item with PII content - include all required fields
        text_item = TextItem(
            text="Call me at 555-123-4567 or email john@example.com",
            self_ref="#/texts/0",
            label="text",
            orig="Call me at 555-123-4567 or email john@example.com"
        )
        doc.texts = [text_item]

        return doc

    @pytest.fixture
    def detected_entities(self) -> List[RecognizerResult]:
        """Create sample detected PII entities."""
        return [
            RecognizerResult(
                entity_type="PHONE_NUMBER",
                start=11,
                end=23,
                score=0.95
            ),
            RecognizerResult(
                entity_type="EMAIL_ADDRESS",
                start=33,
                end=49,
                score=0.88
            )
        ]

    @pytest.fixture
    def basic_policy(self) -> MaskingPolicy:
        """Create a basic masking policy."""
        return MaskingPolicy(
            default_strategy=Strategy(StrategyKind.TEMPLATE, {"template": "[REDACTED]"}),
            per_entity={
                "PHONE_NUMBER": PHONE_TEMPLATE,
                "EMAIL_ADDRESS": Strategy(StrategyKind.TEMPLATE, {"template": "[EMAIL]"})
            }
        )

    @pytest.fixture
    def text_segments(self) -> List[TextSegment]:
        """Create text segments for testing."""
        return [
            TextSegment(
                node_id="#/texts/0",
                text="Call me at 555-123-4567 or email john@example.com",
                start_offset=0,
                end_offset=49,
                node_type="TextItem"
            )
        ]

    def test_masking_engine_initialization(self):
        """Test MaskingEngine can be initialized."""
        engine = MaskingEngine()
        assert engine is not None

    def test_mask_document_with_simple_entities(
        self, simple_document, detected_entities, basic_policy, text_segments
    ):
        """Test masking a document with simple PII entities."""
        engine = MaskingEngine()

        result = engine.mask_document(
            document=simple_document,
            entities=detected_entities,
            policy=basic_policy,
            text_segments=text_segments
        )

        # Should return MaskingResult
        assert isinstance(result, MaskingResult)
        assert result.masked_document is not None
        assert result.cloakmap is not None

        # Document should be modified
        assert result.masked_document != simple_document

        # Text should be masked
        masked_text = result.masked_document.texts[0].text
        assert "[PHONE]" in masked_text
        assert "[EMAIL]" in masked_text
        assert "555-123-4567" not in masked_text
        assert "john@example.com" not in masked_text

    def test_cloakmap_generation(
        self, simple_document, detected_entities, basic_policy, text_segments
    ):
        """Test that CloakMap is properly generated."""
        engine = MaskingEngine()

        result = engine.mask_document(
            document=simple_document,
            entities=detected_entities,
            policy=basic_policy,
            text_segments=text_segments
        )

        cloakmap = result.cloakmap

        # Should have anchors for each entity
        assert len(cloakmap.anchors) == 2

        # Check phone number anchor
        phone_anchor = next(a for a in cloakmap.anchors if a.entity_type == "PHONE_NUMBER")
        assert phone_anchor.start == 11
        assert phone_anchor.end == 23
        assert phone_anchor.masked_value == "[PHONE]"
        assert phone_anchor.node_id == "#/texts/0"

        # Check email anchor
        email_anchor = next(a for a in cloakmap.anchors if a.entity_type == "EMAIL_ADDRESS")
        assert email_anchor.start == 33
        assert email_anchor.end == 49
        assert email_anchor.masked_value == "[EMAIL]"
        assert email_anchor.node_id == "#/texts/0"

    def test_no_original_text_in_cloakmap(
        self, simple_document, detected_entities, basic_policy, text_segments
    ):
        """Test that original PII text is not stored in CloakMap."""
        engine = MaskingEngine()

        result = engine.mask_document(
            document=simple_document,
            entities=detected_entities,
            policy=basic_policy,
            text_segments=text_segments
        )

        cloakmap_json = result.cloakmap.to_json()

        # Original PII should not appear anywhere in the CloakMap
        assert "555-123-4567" not in cloakmap_json
        assert "john@example.com" not in cloakmap_json

        # Only checksums should be stored
        for anchor in result.cloakmap.anchors:
            assert len(anchor.original_checksum) == 64  # SHA-256 hex length
            assert anchor.original_checksum.isalnum()

    def test_document_structure_preservation(
        self, simple_document, detected_entities, basic_policy, text_segments
    ):
        """Test that document structure is preserved during masking."""
        original_text_count = len(simple_document.texts)
        original_table_count = len(simple_document.tables)

        engine = MaskingEngine()

        result = engine.mask_document(
            document=simple_document,
            entities=detected_entities,
            policy=basic_policy,
            text_segments=text_segments
        )

        masked_doc = result.masked_document

        # Structure counts should be unchanged
        assert len(masked_doc.texts) == original_text_count
        assert len(masked_doc.tables) == original_table_count

        # Node references should be preserved
        assert masked_doc.texts[0].self_ref == simple_document.texts[0].self_ref

    def test_empty_entities_list(
        self, simple_document, basic_policy, text_segments
    ):
        """Test masking with no detected entities."""
        engine = MaskingEngine()

        result = engine.mask_document(
            document=simple_document,
            entities=[],
            policy=basic_policy,
            text_segments=text_segments
        )

        # Document should be unchanged
        assert result.masked_document.texts[0].text == simple_document.texts[0].text

        # CloakMap should be empty
        assert len(result.cloakmap.anchors) == 0

    def test_overlapping_entities_handling(
        self, simple_document, basic_policy, text_segments
    ):
        """Test handling of overlapping entity detections."""
        overlapping_entities = [
            RecognizerResult(entity_type="PHONE_NUMBER", start=11, end=23, score=0.95),
            RecognizerResult(entity_type="US_DRIVER_LICENSE", start=15, end=25, score=0.70)  # Overlaps
        ]

        engine = MaskingEngine()

        # Should either handle gracefully or raise clear error
        with pytest.raises(ValueError, match="entities detected"):
            engine.mask_document(
                document=simple_document,
                entities=overlapping_entities,
                policy=basic_policy,
                text_segments=text_segments
            )

    def test_unique_replacement_ids(
        self, simple_document, detected_entities, basic_policy, text_segments
    ):
        """Test that replacement IDs are unique within a CloakMap."""
        engine = MaskingEngine()

        result = engine.mask_document(
            document=simple_document,
            entities=detected_entities,
            policy=basic_policy,
            text_segments=text_segments
        )

        replacement_ids = [anchor.replacement_id for anchor in result.cloakmap.anchors]
        assert len(replacement_ids) == len(set(replacement_ids))  # All unique
