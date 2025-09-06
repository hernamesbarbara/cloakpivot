"""Tests for PresidioUnmaskingAdapter."""

from datetime import datetime
from unittest.mock import patch

from docling_core.types import DoclingDocument
from docling_core.types.doc.document import TextItem
from presidio_anonymizer import DeanonymizeEngine

from cloakpivot.core.anchors import AnchorEntry
from cloakpivot.core.cloakmap import CloakMap
from cloakpivot.core.cloakmap_enhancer import CloakMapEnhancer
from cloakpivot.unmasking.presidio_adapter import PresidioUnmaskingAdapter


class TestPresidioUnmaskingAdapter:

    def _get_document_text(self, document: DoclingDocument) -> str:
        """Helper to get text from document, handling both formats."""
        if hasattr(document, '_main_text'):
            return document._main_text
        elif document.texts:
            return document.texts[0].text
        return ""

    def _set_document_text(self, document: DoclingDocument, text: str) -> None:
        """Helper to set text in document, handling both formats."""
        from docling_core.types.doc.document import TextItem
        # Create proper TextItem
        text_item = TextItem(
            text=text,
            self_ref="#/texts/0",
            label="text",
            orig=text
        )
        document.texts = [text_item]
        # Also set _main_text for backward compatibility
        document._main_text = text

    """Test suite for PresidioUnmaskingAdapter."""

    def test_initialization(self):
        """Test adapter initialization."""
        adapter = PresidioUnmaskingAdapter()
        assert adapter.deanonymizer is not None
        assert isinstance(adapter.deanonymizer, DeanonymizeEngine)
        assert adapter.cloakmap_enhancer is not None
        assert isinstance(adapter.cloakmap_enhancer, CloakMapEnhancer)

    def test_unmask_document_with_presidio_metadata(self):
        """Test unmasking with Presidio metadata (v2.0 CloakMap)."""
        adapter = PresidioUnmaskingAdapter()

        # Create masked document
        masked_doc = DoclingDocument(name="test_doc")
        self._set_document_text(masked_doc, "My phone is [PHONE] and email is [EMAIL].")

        # Create operator results for reversible operations
        operator_results = [
            {
                "entity_type": "PHONE_NUMBER",
                "start": 12,
                "end": 19,
                "operator": "replace",
                "text": "[PHONE]",
                "original_text": "555-1234"
            },
            {
                "entity_type": "EMAIL_ADDRESS",
                "start": 33,
                "end": 40,
                "operator": "replace",
                "text": "[EMAIL]",
                "original_text": "test@example.com"
            }
        ]

        # Create v2.0 CloakMap with Presidio metadata
        cloakmap = CloakMap(
            version="2.0",
            doc_id="test_doc",
            doc_hash="abc123",
            anchors=[],  # Not used for reversible operations
            presidio_metadata={
                "operator_results": operator_results,
                "reversible_operators": ["replace"],
                "engine_version": "2.2.x"
            }
        )

        # Unmask document
        result = adapter.unmask_document(masked_doc, cloakmap)

        # Verify restoration
        assert self._get_document_text(result.restored_document) == "My phone is 555-1234 and email is test@example.com."
        assert result.stats["presidio_restored"] == 2
        assert result.stats["anchor_restored"] == 0

    def test_unmask_document_without_presidio_metadata(self):
        """Test unmasking with v1.0 CloakMap (anchor-based)."""
        adapter = PresidioUnmaskingAdapter()

        # Create masked document
        masked_doc = DoclingDocument(name="test_doc")
        self._set_document_text(masked_doc, "My SSN is [SSN_TOKEN_123].")

        # Create anchor entries for non-reversible operations
        anchor = AnchorEntry(
            node_id="node_1",
            start=10,
            end=25,
            entity_type="SSN",
            confidence=0.95,
            masked_value="[SSN_TOKEN_123]",
            replacement_id="SSN_TOKEN_123",
            original_checksum="a" * 64,  # 64-character SHA-256 hex string
            checksum_salt="dGVzdA==",  # base64 encoded
            strategy_used="template",
            timestamp=datetime.now()
        )

        # Create v1.0 CloakMap without Presidio metadata
        cloakmap = CloakMap(
            version="1.0",
            doc_id="test_doc",
            doc_hash="abc123",
            anchors=[anchor]
        )

        # Mock anchor restoration since we don't have original values
        with patch.object(adapter, '_anchor_based_restoration') as mock_restore:
            from cloakpivot.unmasking.engine import UnmaskingResult

            # Create a mock result with restored document
            mock_doc = DoclingDocument(name="test_doc")
            self._set_document_text(mock_doc, "My SSN is 123-45-6789.")

            mock_result = UnmaskingResult(
                restored_document=mock_doc,
                cloakmap=cloakmap,
                stats={"method": "anchor_based", "anchor_restored": 1}
            )
            mock_restore.return_value = mock_result

            result = adapter.unmask_document(masked_doc, cloakmap)

            # Verify anchor-based restoration was called
            mock_restore.assert_called_once()
            assert self._get_document_text(result.restored_document) == "My SSN is 123-45-6789."

    def test_hybrid_restoration(self):
        """Test hybrid restoration with both Presidio and anchor operations."""
        adapter = PresidioUnmaskingAdapter()

        # Create masked document with mixed operations
        masked_doc = DoclingDocument(name="test_doc")
        self._set_document_text(masked_doc, "Phone: [PHONE], SSN: [REDACTED]")

        # Reversible operation (can be restored via Presidio)
        operator_results = [
            {
                "entity_type": "PHONE_NUMBER",
                "start": 7,
                "end": 14,
                "operator": "replace",
                "text": "[PHONE]",
                "original_text": "555-1234"
            }
        ]

        # Non-reversible operation (needs anchor)
        anchor = AnchorEntry(
            node_id="node_1",
            start=21,
            end= 31,
            entity_type="SSN",
            confidence=0.95,
            masked_value="[REDACTED]",
            replacement_id="REDACTED",
            original_checksum="b" * 64,  # 64-character SHA-256 hex string
            checksum_salt="dGVzdDI=",  # base64 encoded
            strategy_used="redact",
            timestamp=datetime.now()
        )

        # Create v2.0 CloakMap with both types
        cloakmap = CloakMap(
            version="2.0",
            doc_id="test_doc",
            doc_hash="abc123",
            anchors=[anchor],
            presidio_metadata={
                "operator_results": operator_results,
                "reversible_operators": ["replace"],
                "engine_version": "2.2.x"
            }
        )

        # Perform hybrid restoration
        result = adapter.unmask_document(masked_doc, cloakmap)

        # Verify mixed restoration
        assert "[PHONE]" not in self._get_document_text(result.restored_document)
        assert "555-1234" in self._get_document_text(result.restored_document)
        # SSN remains redacted (non-reversible)
        assert "[REDACTED]" in self._get_document_text(result.restored_document)

    def test_restore_content_with_encryption(self):
        """Test content restoration for encrypted values."""
        adapter = PresidioUnmaskingAdapter()

        masked_text = "Credit card: ENCRYPTED_VALUE_XYZ"

        # Use dict format which our implementation handles
        operator_results = [
            {
                "entity_type": "CREDIT_CARD",
                "start": 13,
                "end": 32,
                "text": "ENCRYPTED_VALUE_XYZ",
                "operator": "encrypt",
                "key": "test_key_123"
            }
        ]

        with patch('cloakpivot.unmasking.presidio_adapter.logger') as mock_logger:
            restored = adapter.restore_content(masked_text, operator_results)

            # Should log warning about encryption not being implemented
            mock_logger.warning.assert_called_with(
                "Encryption reversal not yet implemented"
            )
            # Text remains unchanged since encryption reversal isn't implemented
            assert restored == masked_text

    def test_error_handling_fallback(self):
        """Test error handling with fallback to anchor restoration."""
        adapter = PresidioUnmaskingAdapter()

        masked_doc = DoclingDocument(name="test_doc")
        self._set_document_text(masked_doc, "Test [MASKED] content")

        # Create v1.0 CloakMap to trigger fallback to anchor-based restoration
        cloakmap = CloakMap(
            version="1.0",
            doc_id="test_doc",
            doc_hash="abc123",
            anchors=[]  # Empty anchors for simplicity
        )

        # Should use anchor-based restoration for v1.0
        result = adapter.unmask_document(masked_doc, cloakmap)

        # Verify we got a result and it used anchor-based method
        assert result is not None
        assert result.stats["method"] == "anchor_based"

        # Test with invalid Presidio metadata that causes ValueError
        with patch.object(adapter.cloakmap_enhancer, 'extract_operator_results') as mock_extract:
            mock_extract.side_effect = ValueError("Invalid metadata")

            cloakmap_v2 = CloakMap(
                version="2.0",
                doc_id="test_doc",
                doc_hash="abc123",
                anchors=[],
                presidio_metadata={"operator_results": [], "reversible_operators": []}
            )

            # Should fall back to anchor-based when extraction fails
            result = adapter.unmask_document(masked_doc, cloakmap_v2)
            assert result.stats["method"] == "anchor_based"

    def test_backward_compatibility(self):
        """Test backward compatibility with v1.0 CloakMaps."""
        adapter = PresidioUnmaskingAdapter()

        # Create v1.0 CloakMap
        cloakmap_v1 = CloakMap(
            version="1.0",
            doc_id="legacy_doc",
            doc_hash="xyz789",
            anchors=[
                AnchorEntry(
                    node_id="node_1",
                    start=0,
                    end=10,
                    entity_type="NAME",
                    confidence=0.9,
                    masked_value="[NAME]",
                    replacement_id="NAME_1",
                    original_checksum="c" * 64,  # 64-character SHA-256 hex string
                    checksum_salt="dGVzdDM=",  # base64 encoded
                    strategy_used="template",
                    timestamp=datetime.now()
                )
            ]
        )

        masked_doc = DoclingDocument(name="legacy_doc")
        self._set_document_text(masked_doc, "[NAME] is here")

        # Should handle v1.0 gracefully
        with patch.object(adapter, '_anchor_based_restoration') as mock_restore:
            from cloakpivot.unmasking.engine import UnmaskingResult

            mock_result = UnmaskingResult(
                restored_document=masked_doc,
                cloakmap=cloakmap_v1,
                stats={"version": "1.0", "method": "anchor_based", "anchor_restored": 0}
            )
            mock_restore.return_value = mock_result

            result = adapter.unmask_document(masked_doc, cloakmap_v1)

            # Should use anchor-based restoration for v1.0
            mock_restore.assert_called_once()
            assert result.stats["version"] == "1.0"
            assert result.stats["method"] == "anchor_based"

    def test_empty_operator_results(self):
        """Test handling of empty operator results."""
        adapter = PresidioUnmaskingAdapter()

        masked_doc = DoclingDocument(name="test_doc")
        self._set_document_text(masked_doc, "No PII here")

        cloakmap = CloakMap(
            version="2.0",
            doc_id="test_doc",
            doc_hash="abc123",
            anchors=[],
            presidio_metadata={
                "operator_results": [],
                "reversible_operators": []
            }
        )

        result = adapter.unmask_document(masked_doc, cloakmap)

        # Document should remain unchanged
        assert self._get_document_text(result.restored_document) == "No PII here"
        assert result.stats["presidio_restored"] == 0
        assert result.stats["anchor_restored"] == 0

    def test_partial_restoration_success(self):
        """Test partial restoration when some operations fail."""
        adapter = PresidioUnmaskingAdapter()

        masked_doc = DoclingDocument(name="test_doc")
        self._set_document_text(masked_doc, "Phone: [PHONE], Email: [EMAIL]")

        # One valid, one invalid operator result
        operator_results = [
            {
                "entity_type": "PHONE_NUMBER",
                "start": 7,
                "end": 14,
                "operator": "replace",
                "text": "[PHONE]",
                "original_text": "555-1234"
            },
            {
                "entity_type": "EMAIL_ADDRESS",
                "start": 23,
                "end": 30,
                "operator": "custom",
                # Missing required data for custom operator
                "text": "[EMAIL]"
            }
        ]

        cloakmap = CloakMap(
            version="2.0",
            doc_id="test_doc",
            doc_hash="abc123",
            anchors=[],
            presidio_metadata={
                "operator_results": operator_results,
                "reversible_operators": ["replace", "custom"]
            }
        )

        result = adapter.unmask_document(masked_doc, cloakmap)

        # Phone should be restored, email should remain masked
        assert "555-1234" in self._get_document_text(result.restored_document)
        assert "[EMAIL]" in self._get_document_text(result.restored_document)
        assert result.stats["presidio_restored"] == 1
        assert result.stats["presidio_failed"] == 1
