"""Unit tests for CloakEngine - the main API for PII masking/unmasking."""

import pytest
from docling_core.types import DoclingDocument

from cloakpivot.engine import CloakEngine, MaskResult


class TestCloakEngine:
    """Test the CloakEngine main API."""

    def test_engine_initialization(self):
        """Test that CloakEngine initializes with default configuration."""
        engine = CloakEngine()

        assert engine is not None
        assert engine.default_policy is not None
        assert engine.analyzer_config is not None
        assert engine.analyzer_config.language == "en"
        assert engine.analyzer_config.min_confidence == 0.7

    def test_engine_with_custom_config(self):
        """Test CloakEngine initialization with custom configuration."""
        engine = CloakEngine(
            analyzer_config={"language": "es", "confidence_threshold": 0.9}
        )

        assert engine.analyzer_config.language == "es"
        assert engine.analyzer_config.min_confidence == 0.9

    def test_mask_simple_document(self, email_docling_document: DoclingDocument):
        """Test masking a simple document with known PII."""
        engine = CloakEngine()
        result = engine.mask_document(email_docling_document)

        assert isinstance(result, MaskResult)
        assert result.document is not None
        assert result.cloakmap is not None
        assert result.entities_found > 0
        assert result.entities_masked == result.entities_found

        # Check that some text was processed
        # (Can't check specific PII since we're using a real document now)

    def test_mask_specific_entities(self, simple_text_document: DoclingDocument):
        """Test masking only specific entity types."""
        engine = CloakEngine()
        result = engine.mask_document(
            simple_text_document,
            entities=["EMAIL_ADDRESS"]
        )

        # Email should be masked but phone might not be
        assert "john.doe@example.com" not in result.document.text

    def test_unmask_document(self, simple_text_document: DoclingDocument):
        """Test that unmasking restores original content."""
        engine = CloakEngine()

        # First mask the document
        mask_result = engine.mask_document(simple_text_document)

        # Then unmask it
        restored = engine.unmask_document(
            mask_result.document,
            mask_result.cloakmap
        )

        # Should match the original
        assert restored.text == simple_text_document.text

    def test_roundtrip_integrity(self, sample_pii_text: str):
        """Test that mask->unmask roundtrip preserves content exactly."""
        doc = DoclingDocument(
            version="1.0.0",
            name="test",
            text=sample_pii_text
        )

        engine = CloakEngine()
        masked = engine.mask_document(doc)
        restored = engine.unmask_document(masked.document, masked.cloakmap)

        assert restored.text == doc.text

    def test_builder_pattern(self):
        """Test the builder pattern for engine configuration."""
        engine = (
            CloakEngine.builder()
            .with_language("en")
            .with_confidence_threshold(0.85)
            .build()
        )

        assert engine.analyzer_config.language == "en"
        assert engine.analyzer_config.min_confidence == 0.85

    def test_mask_result_metadata(self, simple_text_document: DoclingDocument):
        """Test that MaskResult contains proper metadata."""
        engine = CloakEngine()
        result = engine.mask_document(simple_text_document)

        # Check metadata
        assert result.entities_found >= 0
        assert result.entities_masked >= 0
        assert result.entities_masked <= result.entities_found

        # CloakMap should have entries for each masked entity
        assert len(result.cloakmap.entries) == result.entities_masked
        assert len(result.cloakmap.anchors) == result.entities_masked

    def test_empty_document(self):
        """Test handling of empty document."""
        doc = DoclingDocument(
            version="1.0.0",
            name="empty",
            text=""
        )

        engine = CloakEngine()
        result = engine.mask_document(doc)

        assert result.entities_found == 0
        assert result.entities_masked == 0
        assert result.document.text == ""

    def test_no_pii_document(self):
        """Test document with no PII to mask."""
        doc = DoclingDocument(
            version="1.0.0",
            name="no_pii",
            text="This is a simple text with no personal information."
        )

        engine = CloakEngine()
        result = engine.mask_document(doc)

        assert result.entities_found == 0
        assert result.entities_masked == 0
        assert result.document.text == doc.text  # Text unchanged