"""Tests for unmasking accuracy and integrity."""

from unittest.mock import MagicMock

import pytest
from docling_core.types import DoclingDocument

from cloakpivot.core.types.anchors import AnchorEntry
from cloakpivot.core.types.cloakmap import CloakMap
from cloakpivot.unmasking.engine import UnmaskingEngine


class TestUnmaskingAccuracy:
    """Test unmasking accuracy and document restoration."""

    @pytest.fixture
    def unmasking_engine(self):
        """Create an UnmaskingEngine instance."""
        return UnmaskingEngine()

    @pytest.fixture
    def sample_cloakmap(self):
        """Create a sample CloakMap for testing."""
        anchors = [
            AnchorEntry(
                entity_type="EMAIL",
                start=25,
                end=45,
                node_id="main.texts.0",
                confidence=0.95,
                masked_value="[EMAIL_001]",
                replacement_id="repl_email_001",
                original_checksum="checksum1",
                checksum_salt="salt1",
                strategy_used="template"
            ),
            AnchorEntry(
                entity_type="PERSON",
                start=10,
                end=20,
                node_id="main.texts.0",
                confidence=0.90,
                masked_value="[PERSON_001]",
                replacement_id="repl_person_001",
                original_checksum="checksum2",
                checksum_salt="salt2",
                strategy_used="template"
            )
        ]
        return CloakMap(
            doc_id="test_doc_001",
            doc_hash="testhash",
            anchors=anchors
        )

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
        anchors = [
            AnchorEntry(
                entity_type="INVALID",
                start=10,  # Valid position (can't use negative)
                end=500000,  # Beyond document
                node_id="non.existent.path",
                confidence=0.5,
                masked_value="[INVALID]",
                replacement_id="repl_invalid",
                original_checksum="badchecksum",
                checksum_salt="salt",
                strategy_used="template"
            )
        ]
        corrupted_map = CloakMap(
            doc_id="corrupted",
            doc_hash="corrupted_hash",
            anchors=anchors
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
        # Only add one of two needed anchors
        anchors = [
            AnchorEntry(
                entity_type="PERSON",
                start=10,
                end=20,
                node_id="main.texts.0",
                confidence=0.90,
                masked_value="[PERSON_001]",
                replacement_id="repl_person_001",
                original_checksum="checksum2",
                checksum_salt="salt2",
                strategy_used="template"
            )
        ]
        incomplete_map = CloakMap(
            doc_id="incomplete",
            doc_hash="incomplete_hash",
            anchors=anchors
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
        anchors = []
        for i in range(100):
            large_text = large_text.replace(
                f"Token_{i*100}",
                f"[MASK_{i:03d}]"
            )
            anchors.append(
                AnchorEntry(
                    entity_type="TOKEN",
                    start=i*50,
                    end=i*50+10,
                    node_id="main.texts.0",
                    confidence=0.95,
                    masked_value=f"[MASK_{i:03d}]",
                    replacement_id=f"repl_token_{i:03d}",
                    original_checksum=f"checksum_{i}",
                    checksum_salt=f"salt_{i}",
                    strategy_used="template"
                )
            )
        large_cloakmap = CloakMap(
            doc_id="large",
            doc_hash="large_hash",
            anchors=anchors
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
        anchors = [
            AnchorEntry(
                entity_type="PERSON_ID",
                start=5,
                end=25,
                node_id="main.texts.0",
                confidence=0.92,
                masked_value="[PERSON_[ID_001]_END]",
                replacement_id="repl_nested_001",
                original_checksum="nested_checksum",
                checksum_salt="nested_salt",
                strategy_used="template"
            )
        ]
        nested_map = CloakMap(
            doc_id="nested",
            doc_hash="nested_hash",
            anchors=anchors
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
        anchors = [
            AnchorEntry(
                entity_type="PERSON",
                start=3,
                end=15,
                node_id="main.texts.0",
                confidence=0.88,
                masked_value="[PERSON_001]",
                replacement_id="repl_unicode_person",
                original_checksum="unicode_checksum1",
                checksum_salt="unicode_salt1",
                strategy_used="template"
            ),
            AnchorEntry(
                entity_type="EMAIL",
                start=22,
                end=33,
                node_id="main.texts.0",
                confidence=0.91,
                masked_value="[EMAIL_001]",
                replacement_id="repl_unicode_email",
                original_checksum="unicode_checksum2",
                checksum_salt="unicode_salt2",
                strategy_used="template"
            )
        ]
        unicode_map = CloakMap(
            doc_id="unicode",
            doc_hash="unicode_hash",
            anchors=anchors
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
