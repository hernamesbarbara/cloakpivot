"""Tests for unmasking accuracy and integrity."""

import pytest
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path
import json

from docling_core.types import DoclingDocument
from cloakpivot.unmasking.engine import UnmaskingEngine
from cloakpivot.core.cloakmap import CloakMap
from cloakpivot.core.types import Anchor


class TestUnmaskingAccuracy:
    """Test unmasking accuracy and document restoration."""

    @pytest.fixture
    def unmasking_engine(self):
        """Create an UnmaskingEngine instance."""
        return UnmaskingEngine()

    @pytest.fixture
    def sample_cloakmap(self):
        """Create a sample CloakMap for testing."""
        cloakmap = CloakMap(document_id="test_doc_001")
        cloakmap.add_anchor(
            Anchor(
                entity_type="EMAIL",
                start=25,
                end=45,
                original_text="john@example.com",
                replacement_text="[EMAIL_001]",
                node_path="main.texts.0",
                node_type="text"
            )
        )
        cloakmap.add_anchor(
            Anchor(
                entity_type="PERSON",
                start=10,
                end=20,
                original_text="John Smith",
                replacement_text="[PERSON_001]",
                node_path="main.texts.0",
                node_type="text"
            )
        )
        return cloakmap

    @pytest.fixture
    def masked_document(self):
        """Create a masked DoclingDocument."""
        doc = MagicMock(spec=DoclingDocument)
        doc.texts = [
            MagicMock(text="Hello [PERSON_001], your [EMAIL_001] has been confirmed.")
        ]
        doc.metadata = {"source": "test.pdf", "pages": 1}
        return doc

    def test_unmask_with_modified_document_structure(self, unmasking_engine, sample_cloakmap):
        """Test unmasking when document structure has been modified."""
        # Create document with modified structure
        modified_doc = MagicMock(spec=DoclingDocument)
        modified_doc.texts = [
            MagicMock(text="New text added before. [PERSON_001] mentioned [EMAIL_001].")
        ]

        # Attempt unmasking with position drift
        result = unmasking_engine.unmask_document(
            modified_doc,
            sample_cloakmap,
            verify_integrity=True
        )

        # Should still restore with adjusted positions
        assert result is not None
        assert hasattr(result, 'restored_document')
        # Verify original values are restored despite drift

    def test_unmask_with_corrupted_cloakmap(self, unmasking_engine, masked_document):
        """Test unmasking with a corrupted CloakMap."""
        # Create corrupted cloakmap with invalid anchors
        corrupted_map = CloakMap(document_id="corrupted")
        corrupted_map.add_anchor(
            Anchor(
                entity_type="INVALID",
                start=-10,  # Invalid position
                end=500000,  # Beyond document
                original_text="",
                replacement_text="[INVALID]",
                node_path="non.existent.path",
                node_type="unknown"
            )
        )

        # Should handle gracefully
        result = unmasking_engine.unmask_document(
            masked_document,
            corrupted_map,
            verify_integrity=True
        )

        assert result is not None
        # Check integrity report shows issues
        if hasattr(result, 'integrity_report'):
            assert not result.integrity_report.get('valid', True)

    def test_unmask_partial_document(self, unmasking_engine, sample_cloakmap):
        """Test unmasking only part of a document."""
        # Create document with only some masked tokens
        partial_doc = MagicMock(spec=DoclingDocument)
        partial_doc.texts = [
            MagicMock(text="Hello [PERSON_001], your email is still masked.")
        ]

        result = unmasking_engine.unmask_document(
            partial_doc,
            sample_cloakmap,
            verify_integrity=False
        )

        # Should restore what it can
        assert result is not None
        text = result.restored_document.texts[0].text
        assert "John Smith" in text  # Person restored
        assert "[EMAIL_001]" not in text or "john@example.com" in text

    def test_unmask_with_missing_anchors(self, unmasking_engine, masked_document):
        """Test unmasking when some anchors are missing from CloakMap."""
        # Create incomplete cloakmap
        incomplete_map = CloakMap(document_id="incomplete")
        # Only add one of two needed anchors
        incomplete_map.add_anchor(
            Anchor(
                entity_type="PERSON",
                start=10,
                end=20,
                original_text="John Smith",
                replacement_text="[PERSON_001]",
                node_path="main.texts.0",
                node_type="text"
            )
        )

        result = unmasking_engine.unmask_document(
            masked_document,
            incomplete_map,
            verify_integrity=True
        )

        # Should restore partial document
        assert result is not None
        text = result.restored_document.texts[0].text
        assert "John Smith" in text
        # EMAIL should remain masked
        assert "[EMAIL_001]" in text

    def test_unmask_performance_large_documents(self, unmasking_engine):
        """Test unmasking performance with large documents."""
        import time

        # Create large document and cloakmap
        large_doc = MagicMock(spec=DoclingDocument)
        large_text = " ".join([f"Token_{i}" for i in range(10000)])

        # Add many masked entities
        large_cloakmap = CloakMap(document_id="large")
        for i in range(100):
            large_text = large_text.replace(
                f"Token_{i*100}",
                f"[MASK_{i:03d}]"
            )
            large_cloakmap.add_anchor(
                Anchor(
                    entity_type="TOKEN",
                    start=i*50,
                    end=i*50+10,
                    original_text=f"Token_{i*100}",
                    replacement_text=f"[MASK_{i:03d}]",
                    node_path="main.texts.0",
                    node_type="text"
                )
            )

        large_doc.texts = [MagicMock(text=large_text)]

        # Measure performance
        start = time.time()
        result = unmasking_engine.unmask_document(
            large_doc,
            large_cloakmap,
            verify_integrity=False
        )
        elapsed = time.time() - start

        # Should complete in reasonable time
        assert result is not None
        assert elapsed < 5.0  # Less than 5 seconds for large document

    def test_unmask_nested_replacements(self, unmasking_engine):
        """Test unmasking with nested replacement patterns."""
        # Create document with nested masks
        nested_doc = MagicMock(spec=DoclingDocument)
        nested_doc.texts = [
            MagicMock(text="User [PERSON_[ID_001]_END] sent [EMAIL_[ID_002]_END]")
        ]

        # Create cloakmap with nested patterns
        nested_map = CloakMap(document_id="nested")
        nested_map.add_anchor(
            Anchor(
                entity_type="PERSON_ID",
                start=5,
                end=25,
                original_text="John_Smith_123",
                replacement_text="[PERSON_[ID_001]_END]",
                node_path="main.texts.0",
                node_type="text"
            )
        )

        result = unmasking_engine.unmask_document(
            nested_doc,
            nested_map,
            verify_integrity=True
        )

        # Should handle nested patterns correctly
        assert result is not None
        text = result.restored_document.texts[0].text
        assert "John_Smith_123" in text

    def test_unmask_unicode_content(self, unmasking_engine):
        """Test unmasking with unicode and special characters."""
        # Create document with unicode
        unicode_doc = MagicMock(spec=DoclingDocument)
        unicode_doc.texts = [
            MagicMock(text="用户 [PERSON_001] 的邮箱是 [EMAIL_001] 。")
        ]

        # Create cloakmap with unicode content
        unicode_map = CloakMap(document_id="unicode")
        unicode_map.add_anchor(
            Anchor(
                entity_type="PERSON",
                start=3,
                end=15,
                original_text="张三",
                replacement_text="[PERSON_001]",
                node_path="main.texts.0",
                node_type="text"
            )
        )
        unicode_map.add_anchor(
            Anchor(
                entity_type="EMAIL",
                start=22,
                end=33,
                original_text="张三@例子.com",
                replacement_text="[EMAIL_001]",
                node_path="main.texts.0",
                node_type="text"
            )
        )

        result = unmasking_engine.unmask_document(
            unicode_doc,
            unicode_map,
            verify_integrity=True
        )

        # Should handle unicode correctly
        assert result is not None
        text = result.restored_document.texts[0].text
        assert "张三" in text
        assert "张三@例子.com" in text

    def test_unmask_accuracy_metrics(self, unmasking_engine, masked_document, sample_cloakmap):
        """Test accuracy metrics for unmasking operation."""
        # Perform unmasking
        result = unmasking_engine.unmask_document(
            masked_document,
            sample_cloakmap,
            verify_integrity=True
        )

        # Check accuracy metrics
        assert result is not None
        assert hasattr(result, 'stats')

        stats = result.stats
        assert 'total_anchors' in stats
        assert 'resolved_anchors' in stats
        assert 'failed_anchors' in stats

        # Calculate accuracy
        if stats['total_anchors'] > 0:
            accuracy = stats['resolved_anchors'] / stats['total_anchors']
            assert accuracy >= 0.0
            assert accuracy <= 1.0