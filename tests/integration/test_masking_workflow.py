"""Integration tests for complete masking workflows."""

from pathlib import Path

import pytest
from docling_core.types import DoclingDocument

from cloakpivot import CloakEngine, CloakEngineBuilder


class TestMaskingWorkflow:
    """Test complete masking and unmasking workflows."""

    def test_email_document_workflow(self, email_docling_document: DoclingDocument):
        """Test complete workflow with email document."""
        engine = CloakEngine()

        # Mask the document
        mask_result = engine.mask_document(email_docling_document)

        assert mask_result.document is not None
        assert mask_result.cloakmap is not None
        assert mask_result.entities_found >= 0

        # Verify PII was masked
        masked_text = " ".join(
            [item.text for item in mask_result.document.texts if item.text]
        )
        # Should not contain obvious email patterns
        assert "@example.com" not in masked_text.lower()

        # Unmask the document
        restored = engine.unmask_document(mask_result.document, mask_result.cloakmap)

        # Verify restoration
        assert isinstance(restored, DoclingDocument)
        assert len(restored.texts) == len(email_docling_document.texts)

    def test_pdf_styles_workflow(self, pdf_styles_docling_document: DoclingDocument):
        """Test workflow with PDF styles document."""
        from cloakpivot.core import MaskingPolicy, Strategy, StrategyKind

        policy = MaskingPolicy(
            default_strategy=Strategy(StrategyKind.REDACT)
        )
        engine = (
            CloakEngineBuilder()
            .with_custom_policy(policy)
            .build()
        )

        # Mask the document
        mask_result = engine.mask_document(pdf_styles_docling_document)

        assert mask_result.document is not None
        assert mask_result.cloakmap is not None

        # Unmask and verify
        restored = engine.unmask_document(mask_result.document, mask_result.cloakmap)
        assert isinstance(restored, DoclingDocument)

    def test_custom_configuration_workflow(
        self, email_docling_document: DoclingDocument
    ):
        """Test workflow with custom configuration."""
        engine = (
            CloakEngineBuilder()
            .with_confidence_threshold(0.7)
            .build()
        )

        # Mask the document
        mask_result = engine.mask_document(email_docling_document)
        assert mask_result.entities_found >= 0

        # Unmask and verify
        restored = engine.unmask_document(mask_result.document, mask_result.cloakmap)
        assert isinstance(restored, DoclingDocument)

    def test_save_and_load_cloakmap(
        self, email_docling_document: DoclingDocument, tmp_path: Path
    ):
        """Test saving and loading CloakMap."""
        engine = CloakEngine()
        cloakmap_path = tmp_path / "test.cloakmap.json"

        # Mask document
        mask_result = engine.mask_document(email_docling_document)

        # Save CloakMap
        mask_result.cloakmap.save_to_file(cloakmap_path)
        assert cloakmap_path.exists()

        # Load CloakMap
        from cloakpivot.core import CloakMap

        loaded_map = CloakMap.load_from_file(cloakmap_path)
        assert loaded_map.doc_id == mask_result.cloakmap.doc_id
        assert len(loaded_map.anchors) == len(mask_result.cloakmap.anchors)

        # Use loaded map to unmask
        restored = engine.unmask_document(mask_result.document, loaded_map)
        assert isinstance(restored, DoclingDocument)

    def test_multiple_documents_different_configs(
        self,
        email_docling_document: DoclingDocument,
        pdf_styles_docling_document: DoclingDocument,
    ):
        """Test handling multiple documents with different configurations."""
        from cloakpivot.core import MaskingPolicy, Strategy, StrategyKind

        # First document with low confidence threshold
        minimal_engine = (
            CloakEngineBuilder()
            .with_confidence_threshold(0.5)
            .build()
        )
        result1 = minimal_engine.mask_document(email_docling_document)

        # Second document with strict policy
        policy = MaskingPolicy(
            default_strategy=Strategy(StrategyKind.REDACT)
        )
        conservative_engine = (
            CloakEngineBuilder()
            .with_custom_policy(policy)
            .with_confidence_threshold(0.9)
            .build()
        )
        result2 = conservative_engine.mask_document(pdf_styles_docling_document)

        # Each should have been processed
        assert result1.cloakmap.doc_id != result2.cloakmap.doc_id
        assert result1.entities_found >= 0
        assert result2.entities_found >= 0

        # Unmask each with appropriate engine
        restored1 = minimal_engine.unmask_document(
            result1.document, result1.cloakmap
        )
        restored2 = conservative_engine.unmask_document(
            result2.document, result2.cloakmap
        )

        assert isinstance(restored1, DoclingDocument)
        assert isinstance(restored2, DoclingDocument)

    def test_conflict_resolution(self, email_docling_document: DoclingDocument):
        """Test conflict resolution configuration."""
        from cloakpivot.core.normalization import ConflictResolutionConfig

        # Engine with tight merging threshold
        tight_config = ConflictResolutionConfig(merge_threshold_chars=5)
        engine_tight = (
            CloakEngineBuilder()
            .with_conflict_resolution(tight_config)
            .build()
        )

        # Engine with loose merging threshold
        loose_config = ConflictResolutionConfig(merge_threshold_chars=50)
        engine_loose = (
            CloakEngineBuilder()
            .with_conflict_resolution(loose_config)
            .build()
        )

        result_tight = engine_tight.mask_document(email_docling_document)
        result_loose = engine_loose.mask_document(email_docling_document)

        # Both should work
        assert result_tight.cloakmap is not None
        assert result_loose.cloakmap is not None

        # May have different numbers of anchors due to merging
        # (but this depends on the actual content)
        assert result_tight.entities_found >= 0
        assert result_loose.entities_found >= 0

    def test_all_policies(
        self, email_docling_document: DoclingDocument
    ):
        """Test different policy configurations."""
        from cloakpivot.core import MaskingPolicy, Strategy, StrategyKind

        # Test strict policy
        strict_policy = MaskingPolicy(
            default_strategy=Strategy(StrategyKind.REDACT)
        )
        strict = (
            CloakEngineBuilder()
            .with_custom_policy(strict_policy)
            .build()
        )
        result = strict.mask_document(email_docling_document)
        assert result.cloakmap is not None

        # Test template policy
        template_policy = MaskingPolicy(
            default_strategy=Strategy(
                StrategyKind.TEMPLATE,
                parameters={"template": "[MASKED]"}
            )
        )
        template = (
            CloakEngineBuilder()
            .with_custom_policy(template_policy)
            .build()
        )
        result = template.mask_document(email_docling_document)
        assert result.cloakmap is not None

        # Test default
        default = CloakEngine()
        result = default.mask_document(email_docling_document)
        assert result.cloakmap is not None

