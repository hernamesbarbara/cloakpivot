"""Unit tests for CloakEngine."""

import pytest
from docling_core.types import DoclingDocument

from cloakpivot import CloakEngine


class TestCloakEngine:
    """Test CloakEngine core functionality."""

    def test_engine_creation_default(self):
        """Test creating engine with default settings."""
        engine = CloakEngine()
        assert engine is not None

    def test_engine_creation_with_analyzer_config(self):
        """Test creating engine with custom analyzer config."""
        config = {"confidence_threshold": 0.8}
        engine = CloakEngine(analyzer_config=config)
        assert engine is not None

    def test_engine_creation_with_default_policy(self):
        """Test creating engine with custom default policy."""
        from cloakpivot.core import MaskingPolicy, Strategy, StrategyKind

        policy = MaskingPolicy(
            default_strategy=Strategy(StrategyKind.REDACT)
        )
        engine = CloakEngine(default_policy=policy)
        assert engine is not None

    def test_engine_creation_with_conflict_resolution(self):
        """Test creating engine with conflict resolution config."""
        config = {"merge_threshold_chars": 10}
        engine = CloakEngine(conflict_resolution_config=config)
        assert engine is not None

    def test_engine_creation_empty(self):
        """Test that engine can be created with no arguments."""
        engine = CloakEngine()
        assert engine is not None

    def test_mask_document_with_entities_param(self, email_docling_document: DoclingDocument):
        """Test masking with specific entities."""
        engine = CloakEngine()
        result = engine.mask_document(
            email_docling_document, entities=["EMAIL_ADDRESS", "PERSON"]
        )
        assert result.cloakmap is not None

    def test_mask_document_returns_mask_result(
        self, email_docling_document: DoclingDocument
    ):
        """Test that mask_document returns a MaskResult."""
        engine = CloakEngine()
        result = engine.mask_document(email_docling_document)

        # Check the result has expected attributes
        assert result.document is not None
        assert result.cloakmap is not None
        assert result.entities_found >= 0
        assert result.entities_masked >= 0

    def test_mask_and_unmask_roundtrip(
        self, email_docling_document: DoclingDocument
    ):
        """Test that masking and unmasking returns to original."""
        engine = CloakEngine()

        # Mask the document
        mask_result = engine.mask_document(email_docling_document)
        masked_doc = mask_result.document
        cloakmap = mask_result.cloakmap

        # Unmask the document
        restored_doc = engine.unmask_document(masked_doc, cloakmap)

        # Check that we get back a document
        assert isinstance(restored_doc, DoclingDocument)

        # Original and restored should have same structure
        assert len(restored_doc.texts) == len(email_docling_document.texts)

    def test_builder_method(self):
        """Test that builder() class method returns a builder."""
        builder = CloakEngine.builder()
        assert builder is not None

        # Build an engine from the builder
        engine = builder.with_confidence_threshold(0.8).build()
        assert isinstance(engine, CloakEngine)

    def test_engine_with_multiple_documents(
        self,
        email_docling_document: DoclingDocument,
        pdf_styles_docling_document: DoclingDocument,
    ):
        """Test engine can handle multiple documents."""
        engine = CloakEngine()

        # Mask first document
        result1 = engine.mask_document(email_docling_document)
        assert result1.cloakmap is not None

        # Mask second document
        result2 = engine.mask_document(pdf_styles_docling_document)
        assert result2.cloakmap is not None

        # Each should have unique doc_id
        assert result1.cloakmap.doc_id != result2.cloakmap.doc_id

    def test_engine_entities_tracking(self, email_docling_document: DoclingDocument):
        """Test that engine tracks entities properly."""
        engine = CloakEngine()
        result = engine.mask_document(email_docling_document)

        assert result.entities_found >= 0
        assert result.entities_masked >= 0
        assert result.entities_masked <= result.entities_found

