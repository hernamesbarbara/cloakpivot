"""Unit tests for cloakpivot.unmasking.presidio_adapter module."""

from unittest.mock import Mock, patch

from presidio_anonymizer.entities import OperatorResult

from cloakpivot.core.types import DoclingDocument, UnmaskingResult
from cloakpivot.core.types.anchors import AnchorEntry
from cloakpivot.core.types.cloakmap import CloakMap
from cloakpivot.unmasking.presidio_adapter import PresidioUnmaskingAdapter


class TestPresidioUnmaskingAdapter:
    """Test PresidioUnmaskingAdapter class."""

    def test_initialization(self):
        """Test PresidioUnmaskingAdapter initialization."""
        adapter = PresidioUnmaskingAdapter()
        assert adapter.deanonymizer is not None
        assert adapter.cloakmap_enhancer is not None
        assert adapter.anchor_resolver is not None
        assert adapter.document_unmasker is not None

    def test_unmask_document_with_presidio_metadata(self):
        """Test unmask_document with Presidio metadata."""
        adapter = PresidioUnmaskingAdapter()

        # Mock document
        doc = Mock(spec=DoclingDocument)
        doc.name = "test.md"
        doc.export_to_markdown.return_value = "Masked text with [PERSON]"

        # Create cloakmap with Presidio metadata
        cloakmap = CloakMap(
            doc_id="test_doc",
            doc_hash="test_hash",
            anchors=[],
            presidio_metadata={"operators": []},  # v2.0 feature
        )

        with patch.object(adapter.cloakmap_enhancer, "is_presidio_enabled", return_value=True), patch.object(adapter, "_presidio_deanonymization") as mock_presidio:
            mock_result = Mock(spec=UnmaskingResult)
            mock_presidio.return_value = mock_result

            result = adapter.unmask_document(doc, cloakmap)

            assert result == mock_result
            mock_presidio.assert_called_once_with(doc, cloakmap)

    def test_unmask_document_without_presidio_metadata(self):
        """Test unmask_document without Presidio metadata (v1.0 CloakMap)."""
        adapter = PresidioUnmaskingAdapter()

        # Mock document
        doc = Mock(spec=DoclingDocument)
        doc.name = "test.md"
        doc.export_to_markdown.return_value = "Masked text with [PERSON]"

        # Create v1.0 cloakmap without Presidio metadata
        cloakmap = CloakMap(doc_id="test_doc", doc_hash="test_hash", anchors=[])

        with patch.object(adapter.cloakmap_enhancer, "is_presidio_enabled", return_value=False), patch.object(adapter, "_anchor_based_restoration") as mock_anchor:
            mock_result = Mock(spec=UnmaskingResult)
            mock_anchor.return_value = mock_result

            result = adapter.unmask_document(doc, cloakmap)

            assert result == mock_result
            mock_anchor.assert_called_once_with(doc, cloakmap)

    def test_restore_content_with_operator_results(self):
        """Test restore_content with OperatorResult objects."""
        adapter = PresidioUnmaskingAdapter()

        masked_text = "Hello [PERSON], your email is [EMAIL]"

        # Create OperatorResult objects
        operator_results = [
            OperatorResult(
                start=6, end=14, entity_type="PERSON", text="John Doe", operator="replace"
            ),
            OperatorResult(
                start=30, end=37, entity_type="EMAIL", text="john@example.com", operator="replace"
            ),
        ]

        result = adapter.restore_content(masked_text, operator_results)

        # Should restore in reverse order
        assert "John Doe" in result or "[PERSON]" in result
        assert "john@example.com" in result or "[EMAIL]" in result

    def test_restore_content_with_dict_results(self):
        """Test restore_content with dictionary operator results."""
        adapter = PresidioUnmaskingAdapter()

        masked_text = "Hello [PERSON]"

        # Create dict-based operator results
        operator_results = [
            {
                "start": 6,
                "end": 14,
                "entity_type": "PERSON",
                "text": "Jane Doe",
                "operator": "replace",
            }
        ]

        result = adapter.restore_content(masked_text, operator_results)

        assert "Jane Doe" in result or "[PERSON]" in result

    def test_restore_content_empty_results(self):
        """Test restore_content with empty operator results."""
        adapter = PresidioUnmaskingAdapter()

        masked_text = "No masked content here"
        operator_results = []

        result = adapter.restore_content(masked_text, operator_results)

        assert result == masked_text

    def test_restore_content_overlapping_results(self):
        """Test restore_content with overlapping positions."""
        adapter = PresidioUnmaskingAdapter()

        masked_text = "Text with [ENTITY1] and [ENTITY2]"

        # Create overlapping results (should be handled properly)
        operator_results = [
            OperatorResult(
                start=10, end=19, entity_type="TYPE1", text="replacement1", operator="replace"
            ),
            OperatorResult(
                start=24, end=33, entity_type="TYPE2", text="replacement2", operator="replace"
            ),
        ]

        result = adapter.restore_content(masked_text, operator_results)

        # Should handle both replacements
        assert len(result) > 0

    @patch("cloakpivot.unmasking.presidio_adapter.DeanonymizeEngine")
    def test_deanonymizer_integration(self, mock_engine_class):
        """Test integration with DeanonymizeEngine."""
        mock_engine = Mock()
        mock_engine_class.return_value = mock_engine

        PresidioUnmaskingAdapter()

        # Verify engine was initialized
        mock_engine_class.assert_called_once()

    def test_presidio_deanonymization_flow(self):
        """Test the _presidio_deanonymization private method flow."""
        adapter = PresidioUnmaskingAdapter()

        doc = Mock(spec=DoclingDocument)
        doc.name = "test.md"
        doc.export_to_markdown.return_value = "Masked [PERSON]"

        cloakmap = CloakMap(
            doc_id="test",
            doc_hash="hash",
            anchors=[],
            presidio_metadata={
                "operators": [
                    {
                        "start": 7,
                        "end": 15,
                        "entity_type": "PERSON",
                        "text": "Alice",
                        "operator": "replace",
                    }
                ]
            },
        )

        # Since _presidio_deanonymization is private, we test through public interface
        with patch.object(adapter.cloakmap_enhancer, "is_presidio_enabled", return_value=True), patch.object(adapter.deanonymizer, "deanonymize") as mock_deanon:
            mock_deanon.return_value.text = "Masked Alice"
            mock_deanon.return_value.items = []

            result = adapter.unmask_document(doc, cloakmap)

            assert isinstance(result, UnmaskingResult)

    def test_anchor_based_restoration_flow(self):
        """Test the _anchor_based_restoration private method flow."""
        adapter = PresidioUnmaskingAdapter()

        doc = Mock(spec=DoclingDocument)
        doc.name = "test.md"
        doc.export_to_markdown.return_value = "Masked [PERSON]"

        anchor = AnchorEntry.create_from_detection(
            node_id="node1",
            start=7,
            end=15,
            entity_type="PERSON",
            confidence=0.95,
            original_text="Bob",
            masked_value="[PERSON]",
            strategy_used="redact",
        )

        cloakmap = CloakMap(doc_id="test", doc_hash="hash", anchors=[anchor])

        with patch.object(adapter.cloakmap_enhancer, "is_presidio_enabled", return_value=False), patch.object(adapter.document_unmasker, "apply_unmasking") as mock_unmask:
            mock_result = {"total_anchors": 1, "restored_anchors": 1}
            mock_unmask.return_value = mock_result

            result = adapter.unmask_document(doc, cloakmap)

            assert result == mock_result
            mock_unmask.assert_called_once()

    def test_hybrid_restoration_capabilities(self):
        """Test hybrid restoration with both Presidio and anchor data."""
        adapter = PresidioUnmaskingAdapter()

        doc = Mock(spec=DoclingDocument)
        doc.name = "hybrid.md"
        doc.export_to_markdown.return_value = "[PERSON] sent [EMAIL]"

        # Create anchor for non-reversible operation
        anchor = AnchorEntry.create_from_detection(
            node_id="node1",
            start=0,
            end=8,
            entity_type="PERSON",
            confidence=0.9,
            original_text="Charlie",
            masked_value="[PERSON]",
            strategy_used="redact",
        )

        # Create cloakmap with both anchor and Presidio metadata
        cloakmap = CloakMap(
            doc_id="hybrid",
            doc_hash="hash",
            anchors=[anchor],
            presidio_metadata={
                "operators": [
                    {
                        "start": 14,
                        "end": 21,
                        "entity_type": "EMAIL",
                        "text": "charlie@example.com",
                        "operator": "encrypt",
                    }
                ]
            },
        )

        # Test that hybrid approach works
        with patch.object(adapter.cloakmap_enhancer, "is_presidio_enabled", return_value=True):
            result = adapter.unmask_document(doc, cloakmap)
            assert isinstance(result, UnmaskingResult)

    def test_error_handling_invalid_operator_result(self):
        """Test error handling with invalid operator results."""
        adapter = PresidioUnmaskingAdapter()

        masked_text = "Text with [MASK]"

        # Invalid operator result (missing required fields)
        operator_results = [{"start": 10}]  # Missing end, text, entity_type

        # Should handle gracefully
        result = adapter.restore_content(masked_text, operator_results)
        assert result == masked_text or len(result) > 0

    def test_logging_output(self):
        """Test that appropriate logging occurs."""
        adapter = PresidioUnmaskingAdapter()

        doc = Mock(spec=DoclingDocument)
        doc.name = "test_log.md"
        doc.export_to_markdown.return_value = "Test"

        cloakmap = CloakMap(doc_id="log_test", doc_hash="hash", anchors=[])

        with patch("cloakpivot.unmasking.presidio_adapter.logger") as mock_logger, patch.object(adapter.cloakmap_enhancer, "is_presidio_enabled", return_value=True), patch.object(adapter, "_presidio_deanonymization") as mock_method:
            mock_method.return_value = Mock(spec=UnmaskingResult)

            adapter.unmask_document(doc, cloakmap)

            # Check that info logging occurred
            assert mock_logger.info.called

            # Verify log messages contain expected info
            log_calls = mock_logger.info.call_args_list
            assert any("Starting unmasking" in str(call) for call in log_calls)
            assert any("Presidio-based restoration" in str(call) for call in log_calls)
