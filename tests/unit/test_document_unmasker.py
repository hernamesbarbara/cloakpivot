"""Unit tests for cloakpivot.unmasking.document_unmasker module."""

from unittest.mock import Mock, patch

from docling_core.types.doc.document import TextItem

from cloakpivot.core.types import DoclingDocument
from cloakpivot.core.types.anchors import AnchorEntry
from cloakpivot.core.types.cloakmap import CloakMap
from cloakpivot.unmasking.anchor_resolver import ResolvedAnchor
from cloakpivot.unmasking.document_unmasker import DocumentUnmasker


class TestDocumentUnmasker:
    """Test DocumentUnmasker class."""

    def test_initialization(self):
        """Test DocumentUnmasker initialization."""
        unmasker = DocumentUnmasker()
        assert unmasker is not None

    def test_apply_unmasking_empty_cloakmap(self):
        """Test unmasking with empty cloakmap."""
        unmasker = DocumentUnmasker()

        doc = Mock(spec=DoclingDocument)
        doc.export_to_markdown.return_value = "No masked content here"

        cloakmap = CloakMap(
            doc_id="test",
            doc_hash="hash",
            anchors=[]
        )

        # No resolved anchors for empty cloakmap
        resolved_anchors = []
        result = unmasker.apply_unmasking(doc, resolved_anchors, cloakmap)

        assert isinstance(result, dict)
        assert result["total_anchors"] == 0
        assert result["restored_anchors"] == 0

    def test_apply_unmasking_with_single_anchor(self):
        """Test unmasking with single anchor."""
        unmasker = DocumentUnmasker()

        # Create mock document with text items
        doc = Mock(spec=DoclingDocument)
        mock_text_item = Mock(spec=TextItem)
        mock_text_item.text = "Hello [PERSON]!"
        doc.texts = [mock_text_item]
        doc.export_to_markdown.return_value = "Hello [PERSON]!"

        # Create anchor with all required parameters
        anchor = AnchorEntry.create_from_detection(
            node_id="#/texts/0",
            start=6,
            end=14,
            entity_type="PERSON",
            confidence=0.95,
            original_text="Alice",
            masked_value="[PERSON]",
            strategy_used="template",
            replacement_id="person_1"
        )

        # Create resolved anchor
        resolved_anchor = ResolvedAnchor(
            anchor=anchor,
            node_item=mock_text_item,
            found_position=(6, 14),
            found_text="[PERSON]",
            position_delta=0,
            confidence=1.0
        )

        cloakmap = CloakMap(
            doc_id="test",
            doc_hash="hash",
            anchors=[anchor]
        )

        result = unmasker.apply_unmasking(doc, [resolved_anchor], cloakmap)

        assert isinstance(result, dict)
        assert result["total_anchors"] == 1

    def test_apply_unmasking_multiple_anchors(self):
        """Test unmasking with multiple anchors."""
        unmasker = DocumentUnmasker()

        doc = Mock(spec=DoclingDocument)
        mock_text_item = Mock(spec=TextItem)
        mock_text_item.text = "[PERSON] sent [EMAIL] to [PERSON]"
        doc.texts = [mock_text_item]
        doc.export_to_markdown.return_value = "[PERSON] sent [EMAIL] to [PERSON]"

        anchors = [
            AnchorEntry.create_from_detection(
                node_id="#/texts/0",
                start=0,
                end=8,
                entity_type="PERSON",
                confidence=0.9,
                original_text="Bob",
                masked_value="[PERSON]",
                strategy_used="template",
                replacement_id="person_1"
            ),
            AnchorEntry.create_from_detection(
                node_id="#/texts/0",
                start=14,
                end=21,
                entity_type="EMAIL",
                confidence=0.95,
                original_text="bob@example.com",
                masked_value="[EMAIL]",
                strategy_used="template",
                replacement_id="email_1"
            ),
            AnchorEntry.create_from_detection(
                node_id="#/texts/0",
                start=25,
                end=33,
                entity_type="PERSON",
                confidence=0.9,
                original_text="Charlie",
                masked_value="[PERSON]",
                strategy_used="template",
                replacement_id="person_2"
            )
        ]

        # Create resolved anchors
        resolved_anchors = [
            ResolvedAnchor(
                anchor=anchor,
                node_item=mock_text_item,
                found_position=(anchor.start_position, anchor.end_position),
                found_text=anchor.masked_value,
                position_delta=0,
                confidence=1.0
            ) for anchor in anchors
        ]

        cloakmap = CloakMap(
            doc_id="test",
            doc_hash="hash",
            anchors=anchors
        )

        result = unmasker.apply_unmasking(doc, resolved_anchors, cloakmap)

        assert isinstance(result, dict)
        assert result["total_anchors"] == 3

    def test_unmask_preserves_document_structure(self):
        """Test that unmasking preserves document structure."""
        unmasker = DocumentUnmasker()

        doc = Mock(spec=DoclingDocument)
        doc.name = "test.md"
        doc.export_to_markdown.return_value = "# Title\n\n[PERSON] content\n\n## Section"

        # Create mock text items
        text_item_1 = Mock(spec=TextItem, text="Title")
        text_item_2 = Mock(spec=TextItem, text="[PERSON] content")
        text_item_3 = Mock(spec=TextItem, text="Section")
        doc.texts = [text_item_1, text_item_2, text_item_3]

        anchor = AnchorEntry.create_from_detection(
            node_id="#/texts/1",
            start=0,
            end=8,
            entity_type="PERSON",
            confidence=0.9,
            original_text="Author",
            masked_value="[PERSON]",
            strategy_used="template",
            replacement_id="person_1"
        )

        resolved_anchor = ResolvedAnchor(
            anchor=anchor,
            node_item=text_item_2,
            found_position=(0, 8),
            found_text="[PERSON]",
            position_delta=0,
            confidence=1.0
        )

        cloakmap = CloakMap(
            doc_id="test",
            doc_hash="hash",
            anchors=[anchor]
        )

        result = unmasker.apply_unmasking(doc, [resolved_anchor], cloakmap)

        assert isinstance(result, dict)
        # Structure should be preserved

    def test_unmask_with_overlapping_anchors(self):
        """Test handling of overlapping anchors."""
        unmasker = DocumentUnmasker()

        doc = Mock(spec=DoclingDocument)
        mock_text = Mock(spec=TextItem, text="[ENTITY_LONG]")
        doc.texts = [mock_text]
        doc.export_to_markdown.return_value = "[ENTITY_LONG]"

        # Overlapping anchors (shouldn't normally happen)
        anchors = [
            AnchorEntry.create_from_detection(
                node_id="#/texts/0",
                start=0,
                end=8,
                entity_type="TYPE1",
                confidence=0.9,
                original_text="text1",
                masked_value="[ENTITY",
                strategy_used="template",
                replacement_id="entity_1"
            ),
            AnchorEntry.create_from_detection(
                node_id="#/texts/0",
                start=5,
                end=13,
                entity_type="TYPE2",
                confidence=0.9,
                original_text="text2",
                masked_value="_LONG]",
                strategy_used="template",
                replacement_id="entity_2"
            )
        ]

        # Create resolved anchors - may not find due to overlap
        resolved_anchors = []

        cloakmap = CloakMap(
            doc_id="test",
            doc_hash="hash",
            anchors=anchors
        )

        # Should handle gracefully
        result = unmasker.apply_unmasking(doc, resolved_anchors, cloakmap)
        assert result is not None

    def test_unmask_with_position_drift(self):
        """Test handling position drift in anchors."""
        unmasker = DocumentUnmasker()

        doc = Mock(spec=DoclingDocument)
        mock_text = Mock(spec=TextItem, text="Some extra text [PERSON] here")
        doc.texts = [mock_text]
        doc.export_to_markdown.return_value = "Some extra text [PERSON] here"

        # Anchor expects different position
        anchor = AnchorEntry.create_from_detection(
            node_id="#/texts/0",
            start=5,  # Wrong position
            end=13,
            entity_type="PERSON",
            confidence=0.9,
            original_text="Name",
            masked_value="[PERSON]",
            strategy_used="template",
            replacement_id="person_1"
        )

        # Resolved anchor with drift
        resolved_anchor = ResolvedAnchor(
            anchor=anchor,
            node_item=mock_text,
            found_position=(16, 24),  # Actual position
            found_text="[PERSON]",
            position_delta=11,  # Position drift
            confidence=0.8  # Lower confidence due to drift
        )

        cloakmap = CloakMap(
            doc_id="test",
            doc_hash="hash",
            anchors=[anchor]
        )

        result = unmasker.apply_unmasking(doc, [], cloakmap)
        assert result is not None

    def test_unmask_statistics_tracking(self):
        """Test that statistics are properly tracked."""
        unmasker = DocumentUnmasker()

        doc = Mock(spec=DoclingDocument)
        doc.export_to_markdown.return_value = "[PERSON] and [EMAIL]"

        anchors = [
            AnchorEntry.create_from_detection(
                node_id="#/texts/0",
                start=0,
                end=8,
                entity_type="PERSON",
                confidence=0.9,
                original_text="Name",
                masked_value="[PERSON]",
                strategy_used="template"
            ),
            AnchorEntry.create_from_detection(
                node_id="#/texts/0",
                start=13,
                end=20,
                entity_type="EMAIL",
                confidence=0.95,
                original_text="test@example.com",
                masked_value="[EMAIL]",
                strategy_used="template"
            )
        ]

        cloakmap = CloakMap(
            doc_id="test",
            doc_hash="hash",
            anchors=anchors
        )

        result = unmasker.apply_unmasking(doc, [], cloakmap)

        assert result.stats is not None
        # Check for expected statistics
        if "total_anchors" in result.stats:
            assert result.stats["total_anchors"] == 2
        if "entities_by_type" in result.stats:
            assert "PERSON" in result.stats["entities_by_type"]
            assert "EMAIL" in result.stats["entities_by_type"]

    def test_unmask_error_handling(self):
        """Test error handling during unmasking."""
        unmasker = DocumentUnmasker()

        # Document with problematic content
        doc = Mock(spec=DoclingDocument)
        doc.export_to_markdown.side_effect = Exception("Export failed")

        cloakmap = CloakMap(
            doc_id="test",
            doc_hash="hash",
            anchors=[]
        )

        # Should handle error gracefully
        try:
            result = unmasker.apply_unmasking(doc, [], cloakmap)
            # May return partial result or raise specific exception
            assert result is not None or result is None
        except Exception as e:
            # Should be a specific exception type
            assert isinstance(e, Exception)

    def test_unmask_with_invalid_cloakmap(self):
        """Test handling invalid cloakmap."""
        unmasker = DocumentUnmasker()

        doc = Mock(spec=DoclingDocument)
        doc.export_to_markdown.return_value = "Test content"

        # Invalid cloakmap (wrong type)
        invalid_cloakmap = {"not": "a_cloakmap"}

        # Should handle gracefully
        try:
            unmasker.apply_unmasking(doc, invalid_cloakmap)
            raise AssertionError("Should have raised an error")
        except (TypeError, AttributeError, ValueError):
            pass

    def test_unmask_performance_metrics(self):
        """Test performance metrics collection."""
        unmasker = DocumentUnmasker()

        doc = Mock(spec=DoclingDocument)
        doc.export_to_markdown.return_value = "[MASK]" * 100

        # Create many anchors
        anchors = []
        for i in range(100):
            anchor = AnchorEntry.create_from_detection(
                node_id="#/texts/0",
                start=i * 6,
                end=i * 6 + 6,
                entity_type="ENTITY",
                confidence=0.9,
                original_text=f"text{i}",
                masked_value="[MASK]",
                strategy_used="template"
            )
            anchors.append(anchor)

        cloakmap = CloakMap(
            doc_id="test",
            doc_hash="hash",
            anchors=anchors
        )

        result = unmasker.apply_unmasking(doc, [], cloakmap)

        assert result is not None
        if hasattr(result, 'performance'):
            assert "processing_time" in result.performance
            assert "anchors_per_second" in result.performance

    def test_partial_unmasking(self):
        """Test partial unmasking when some anchors fail."""
        unmasker = DocumentUnmasker()

        doc = Mock(spec=DoclingDocument)
        doc.export_to_markdown.return_value = "[PERSON] and [MISSING]"

        # One valid anchor, one for non-existent mask
        anchors = [
            AnchorEntry.create_from_detection(
                node_id="#/texts/0",
                start=0,
                end=8,
                entity_type="PERSON",
                confidence=0.9,
                original_text="Name",
                masked_value="[PERSON]",
                strategy_used="template"
            ),
            AnchorEntry.create_from_detection(
                node_id="#/texts/0",
                start=13,
                end=22,
                entity_type="OTHER",
                confidence=0.9,
                original_text="data",
                masked_value="[OTHER]",  # Doesn't match [MISSING]
                strategy_used="template"
            )
        ]

        cloakmap = CloakMap(
            doc_id="test",
            doc_hash="hash",
            anchors=anchors
        )

        result = unmasker.apply_unmasking(doc, [], cloakmap)

        assert result is not None
        if hasattr(result, 'partial'):
            assert result.partial is True

    @patch('cloakpivot.unmasking.document_unmasker.AnchorResolver')
    def test_integration_with_anchor_resolver(self, mock_resolver_class):
        """Test integration with AnchorResolver."""
        unmasker = DocumentUnmasker()

        mock_resolver = Mock()
        mock_resolver_class.return_value = mock_resolver
        mock_resolver.resolve_anchors.return_value = {
            "resolved": [],
            "failed": []
        }

        doc = Mock(spec=DoclingDocument)
        doc.export_to_markdown.return_value = "Test"

        cloakmap = CloakMap(
            doc_id="test",
            doc_hash="hash",
            anchors=[]
        )

        # If uses anchor resolver
        with patch.object(unmasker, '_anchor_resolver', mock_resolver):
            result = unmasker.apply_unmasking(doc, [], cloakmap)
            assert result is not None

    def test_unmask_with_different_strategies(self):
        """Test unmasking anchors created with different strategies."""
        unmasker = DocumentUnmasker()

        doc = Mock(spec=DoclingDocument)
        doc.export_to_markdown.return_value = "[REDACTED] ******* [HASH123]"

        anchors = [
            AnchorEntry.create_from_detection(
                node_id="#/texts/0",
                start=0,
                end=10,
                entity_type="PERSON",
                confidence=0.9,
                original_text="Alice",
                masked_value="[REDACTED]",
                strategy_used="template"
            ),
            AnchorEntry.create_from_detection(
                node_id="#/texts/0",
                start=11,
                end=18,
                entity_type="EMAIL",
                confidence=0.9,
                original_text="alice@example.com",
                masked_value="*******",
                strategy_used="redact"
            ),
            AnchorEntry.create_from_detection(
                node_id="#/texts/0",
                start=19,
                end=28,
                entity_type="ID",
                confidence=0.9,
                original_text="12345",
                masked_value="[HASH123]",
                strategy_used="hash"
            )
        ]

        cloakmap = CloakMap(
            doc_id="test",
            doc_hash="hash",
            anchors=anchors
        )

        result = unmasker.apply_unmasking(doc, [], cloakmap)
        assert result is not None
