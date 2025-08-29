"""Comprehensive tests for document integration functionality."""

from unittest.mock import Mock, patch

import pytest
from docling_core.types import DoclingDocument
from docling_core.types.doc.document import (
    TextItem,
)
from presidio_analyzer import RecognizerResult

from cloakpivot.core.anchors import AnchorEntry
from cloakpivot.document.extractor import TextExtractor, TextSegment
from cloakpivot.document.mapper import AnchorMapper, NodeReference
from cloakpivot.document.processor import DocumentProcessor


class TestDocumentProcessor:
    """Test cases for DocumentProcessor class."""

    def test_init(self):
        """Test processor initialization."""
        processor = DocumentProcessor()
        assert processor is not None
        stats = processor.get_processing_stats()
        assert stats.files_processed == 0
        assert stats.errors_encountered == 0

    @patch('cloakpivot.document.processor.load_document')
    def test_load_document_success(self, mock_load_document):
        """Test successful document loading."""
        # Setup mock
        mock_doc = Mock(spec=DoclingDocument)
        mock_doc.name = "test_document"
        mock_doc.texts = []
        mock_doc.tables = []
        mock_doc.pictures = []
        mock_doc.key_value_items = []
        mock_doc.form_items = []
        mock_load_document.return_value = mock_doc

        processor = DocumentProcessor()

        # Test loading
        result = processor.load_document("test.docling.json")

        assert result == mock_doc
        assert processor.get_processing_stats().files_processed == 1
        mock_load_document.assert_called_once_with("test.docling.json")

    @patch('cloakpivot.document.processor.load_document')
    def test_load_document_with_validation_disabled(self, mock_load_document):
        """Test document loading with validation disabled."""
        mock_doc = Mock(spec=DoclingDocument)
        mock_doc.name = "test_document"
        mock_doc.texts = []
        mock_doc.tables = []
        mock_doc.pictures = []
        mock_doc.key_value_items = []
        mock_doc.form_items = []
        mock_load_document.return_value = mock_doc

        processor = DocumentProcessor()
        result = processor.load_document("test.docling.json", validate=False)

        assert result == mock_doc
        mock_load_document.assert_called_once_with("test.docling.json")

    @patch('cloakpivot.document.processor.load_document')
    def test_load_document_file_not_found(self, mock_load_document):
        """Test handling of file not found error."""
        from docpivot.io.readers.exceptions import FileAccessError

        mock_load_document.side_effect = FileAccessError(
            "File not found",
            file_path="nonexistent.json",
            operation="load_document"
        )

        processor = DocumentProcessor()

        with pytest.raises(FileAccessError):
            processor.load_document("nonexistent.json")

        assert processor.get_processing_stats().errors_encountered == 1

    def test_supports_format(self):
        """Test format support detection."""
        processor = DocumentProcessor()

        # Test supported formats
        assert processor.supports_format("test.docling.json")
        assert processor.supports_format("test.lexical.json")
        assert processor.supports_format("document.json")  # Generic JSON

        # Test unsupported formats
        assert not processor.supports_format("test.txt")
        assert not processor.supports_format("test.pdf")
        assert not processor.supports_format("test.docx")

    def test_reset_stats(self):
        """Test statistics reset functionality."""
        processor = DocumentProcessor()

        # Modify stats
        processor._stats.files_processed = 5
        processor._stats.errors_encountered = 2

        # Reset
        processor.reset_stats()

        # Verify reset
        stats = processor.get_processing_stats()
        assert stats.files_processed == 0
        assert stats.errors_encountered == 0


class TestTextExtractor:
    """Test cases for TextExtractor class."""

    def create_mock_text_item(self, text: str, self_ref: str = None) -> Mock:
        """Create a mock TextItem."""
        item = Mock(spec=TextItem)
        item.text = text
        item.self_ref = self_ref or f"#/texts/{hash(text) % 1000}"
        return item

    def create_mock_document(self, text_items: list[Mock] = None) -> Mock:
        """Create a mock DoclingDocument."""
        doc = Mock(spec=DoclingDocument)
        doc.name = "test_document"
        doc.texts = text_items or []
        doc.tables = []
        doc.key_value_items = []
        doc.pictures = []
        doc.form_items = []
        return doc

    def test_init(self):
        """Test extractor initialization."""
        extractor = TextExtractor()
        assert extractor.normalize_whitespace is False

        extractor_no_norm = TextExtractor(normalize_whitespace=False)
        assert extractor_no_norm.normalize_whitespace is False

    def test_extract_text_segments_single_item(self):
        """Test extracting segments from a document with one text item."""
        text_item = self.create_mock_text_item("Hello world", "#/texts/0")
        document = self.create_mock_document([text_item])

        extractor = TextExtractor()
        segments = extractor.extract_text_segments(document)

        assert len(segments) == 1
        segment = segments[0]
        assert segment.text == "Hello world"
        assert segment.node_id == "#/texts/0"
        assert segment.start_offset == 0
        assert segment.end_offset == 11
        assert segment.node_type == "Mock"  # Mock class name

    def test_extract_text_segments_multiple_items(self):
        """Test extracting segments from multiple text items."""
        text_items = [
            self.create_mock_text_item("First paragraph", "#/texts/0"),
            self.create_mock_text_item("Second paragraph", "#/texts/1"),
        ]
        document = self.create_mock_document(text_items)

        extractor = TextExtractor()
        segments = extractor.extract_text_segments(document)

        assert len(segments) == 2

        # Check first segment
        assert segments[0].text == "First paragraph"
        assert segments[0].start_offset == 0
        assert segments[0].end_offset == 15

        # Check second segment (should account for separator)
        assert segments[1].text == "Second paragraph"
        assert segments[1].start_offset == 17  # 15 + len("\n\n")
        assert segments[1].end_offset == 33

    def test_extract_full_text(self):
        """Test extracting full text as a single string."""
        text_items = [
            self.create_mock_text_item("First paragraph", "#/texts/0"),
            self.create_mock_text_item("Second paragraph", "#/texts/1"),
        ]
        document = self.create_mock_document(text_items)

        extractor = TextExtractor()
        full_text = extractor.extract_full_text(document)

        assert full_text == "First paragraph\n\nSecond paragraph"

    def test_find_segment_containing_offset(self):
        """Test finding segments by global offset."""
        segments = [
            TextSegment("#/texts/0", "Hello", 0, 5, "TextItem"),
            TextSegment("#/texts/1", "world", 7, 12, "TextItem"),
        ]

        extractor = TextExtractor()

        # Test finding segments
        assert extractor.find_segment_containing_offset(segments, 0) == segments[0]
        assert extractor.find_segment_containing_offset(segments, 4) == segments[0]
        assert extractor.find_segment_containing_offset(segments, 7) == segments[1]
        assert extractor.find_segment_containing_offset(segments, 11) == segments[1]

        # Test out of bounds
        assert extractor.find_segment_containing_offset(segments, 15) is None

    def test_get_extraction_stats(self):
        """Test extraction statistics generation."""
        text_items = [
            self.create_mock_text_item("Hello", "#/texts/0"),
            self.create_mock_text_item("World", "#/texts/1"),
        ]
        document = self.create_mock_document(text_items)

        extractor = TextExtractor()
        stats = extractor.get_extraction_stats(document)

        assert stats["total_text_items"] == 2
        assert stats["total_tables"] == 0
        assert stats["extractable_segments"] == 2
        assert stats["total_extractable_chars"] == 10  # "Hello" + "World"

    def test_normalize_whitespace(self):
        """Test whitespace normalization."""
        extractor = TextExtractor()

        # Test multiple spaces (4 spaces -> 2 spaces, conservative normalization)
        result = extractor._normalize_whitespace("Hello    world")
        assert result == "Hello  world"

        # Test line breaks (4 newlines -> 3 newlines, conservative normalization) 
        result = extractor._normalize_whitespace("Hello\n\n\n\nworld")
        assert result == "Hello\n\n\nworld"

        # Test mixed whitespace (conservative normalization preserves most formatting)
        result = extractor._normalize_whitespace("  Hello \t world  \n\n  ")
        assert result == "  Hello \t world  \n\n  "

    def test_extract_empty_document(self):
        """Test extraction from empty document."""
        document = self.create_mock_document([])

        extractor = TextExtractor()
        segments = extractor.extract_text_segments(document)

        assert len(segments) == 0

    def test_extract_with_empty_text_items(self):
        """Test extraction with empty or None text items."""
        text_items = [
            self.create_mock_text_item("Valid text", "#/texts/0"),
            Mock(spec=TextItem, text="", self_ref="#/texts/1"),  # Empty text
            Mock(spec=TextItem, text=None, self_ref="#/texts/2"),  # None text
        ]
        document = self.create_mock_document(text_items)

        extractor = TextExtractor()
        segments = extractor.extract_text_segments(document)

        # Should only extract the valid text item
        assert len(segments) == 1
        assert segments[0].text == "Valid text"


class TestTextSegment:
    """Test cases for TextSegment class."""

    def test_valid_segment_creation(self):
        """Test creating a valid text segment."""
        segment = TextSegment(
            node_id="#/texts/0",
            text="Hello world",
            start_offset=0,
            end_offset=11,
            node_type="TextItem"
        )

        assert segment.node_id == "#/texts/0"
        assert segment.text == "Hello world"
        assert segment.length == 11
        assert segment.contains_offset(5)
        assert not segment.contains_offset(15)

    def test_invalid_segment_creation(self):
        """Test validation errors in segment creation."""
        # Test invalid offsets
        with pytest.raises(ValueError, match="end_offset must be greater than start_offset"):
            TextSegment("#/texts/0", "Hello", 5, 5, "TextItem")

        # Test mismatched text length
        with pytest.raises(ValueError, match="text length must match offset difference"):
            TextSegment("#/texts/0", "Hello", 0, 10, "TextItem")  # Text is 5 chars, offset diff is 10

        # Test empty node_id
        with pytest.raises(ValueError, match="node_id cannot be empty"):
            TextSegment("", "Hello", 0, 5, "TextItem")

    def test_relative_offset(self):
        """Test relative offset calculation."""
        segment = TextSegment("#/texts/0", "Hello world", 10, 21, "TextItem")

        # Test valid relative offsets
        assert segment.relative_offset(10) == 0  # Start of segment
        assert segment.relative_offset(15) == 5  # Middle of segment
        assert segment.relative_offset(20) == 10  # End of segment

        # Test invalid offset
        with pytest.raises(ValueError):
            segment.relative_offset(25)  # Outside segment


class TestAnchorMapper:
    """Test cases for AnchorMapper class."""

    def create_mock_recognizer_result(
        self,
        start: int,
        end: int,
        entity_type: str = "PHONE_NUMBER",
        score: float = 0.9
    ) -> RecognizerResult:
        """Create a mock RecognizerResult."""
        result = RecognizerResult(
            entity_type=entity_type,
            start=start,
            end=end,
            score=score
        )
        return result

    def test_init(self):
        """Test mapper initialization."""
        mapper = AnchorMapper()
        assert mapper is not None

    def test_map_global_to_node_position(self):
        """Test mapping global positions to node positions."""
        segments = [
            TextSegment("#/texts/0", "Hello world", 0, 11, "TextItem"),
            TextSegment("#/texts/1", "How are you?", 13, 25, "TextItem"),
        ]

        mapper = AnchorMapper()

        # Test mapping within first segment
        node_ref = mapper.map_global_to_node_position(0, 5, segments)
        assert node_ref is not None
        assert node_ref.node_id == "#/texts/0"
        assert node_ref.start_pos == 0
        assert node_ref.end_pos == 5
        assert node_ref.global_start == 0
        assert node_ref.global_end == 5

        # Test mapping within second segment
        node_ref = mapper.map_global_to_node_position(13, 16, segments)
        assert node_ref is not None
        assert node_ref.node_id == "#/texts/1"
        assert node_ref.start_pos == 0
        assert node_ref.end_pos == 3
        assert node_ref.global_start == 13
        assert node_ref.global_end == 16

    def test_map_node_to_global_position(self):
        """Test mapping node positions to global positions."""
        segments = [
            TextSegment("#/texts/0", "Hello world", 0, 11, "TextItem"),
            TextSegment("#/texts/1", "How are you?", 13, 25, "TextItem"),
        ]

        mapper = AnchorMapper()

        # Test mapping from first segment
        global_pos = mapper.map_node_to_global_position("#/texts/0", 0, 5, segments)
        assert global_pos == (0, 5)

        # Test mapping from second segment
        global_pos = mapper.map_node_to_global_position("#/texts/1", 0, 3, segments)
        assert global_pos == (13, 16)

        # Test non-existent node
        global_pos = mapper.map_node_to_global_position("#/texts/999", 0, 3, segments)
        assert global_pos is None

    def test_create_anchors_from_detections(self):
        """Test creating anchors from Presidio detections."""
        # Setup test data
        segments = [
            TextSegment("#/texts/0", "Call me at 555-1234", 0, 19, "TextItem"),
        ]

        detections = [
            self.create_mock_recognizer_result(11, 19, "PHONE_NUMBER", 0.95)
        ]

        original_texts = {
            "#/texts/0": "Call me at 555-1234"
        }

        mapper = AnchorMapper()
        anchors = mapper.create_anchors_from_detections(
            detections, segments, original_texts, "redact"
        )

        assert len(anchors) == 1
        anchor = anchors[0]
        assert anchor.node_id == "#/texts/0"
        assert anchor.start == 11
        assert anchor.end == 19
        assert anchor.entity_type == "PHONE_NUMBER"
        assert anchor.confidence == 0.95
        assert anchor.strategy_used == "redact"

    def test_resolve_anchor_conflicts_no_conflicts(self):
        """Test conflict resolution with no overlapping anchors."""
        anchors = [
            AnchorEntry.create_from_detection(
                "#/texts/0", 0, 5, "PERSON", 0.9, "Alice", "[PERSON]", "redact"
            ),
            AnchorEntry.create_from_detection(
                "#/texts/0", 10, 15, "PHONE_NUMBER", 0.95, "12345", "[PHONE]", "redact"
            ),
        ]

        mapper = AnchorMapper()
        resolved = mapper.resolve_anchor_conflicts(anchors)

        assert len(resolved) == 2
        assert all(anchor in resolved for anchor in anchors)

    def test_resolve_anchor_conflicts_with_overlaps(self):
        """Test conflict resolution with overlapping anchors."""
        # Create overlapping anchors (same position range)
        anchor1 = AnchorEntry.create_from_detection(
            "#/texts/0", 0, 10, "PERSON", 0.8, "John Smith", "[PERSON]", "redact"
        )
        anchor2 = AnchorEntry.create_from_detection(
            "#/texts/0", 5, 15, "PHONE_NUMBER", 0.9, "555-1234", "[PHONE]", "redact"
        )

        anchors = [anchor1, anchor2]

        mapper = AnchorMapper()
        resolved = mapper.resolve_anchor_conflicts(anchors)

        # Should resolve to one anchor (the one with higher confidence)
        assert len(resolved) == 1
        assert resolved[0].entity_type == "PHONE_NUMBER"  # Higher confidence
        assert resolved[0].confidence == 0.9


class TestNodeReference:
    """Test cases for NodeReference class."""

    def test_valid_node_reference_creation(self):
        """Test creating a valid node reference."""
        ref = NodeReference(
            node_id="#/texts/0",
            start_pos=5,
            end_pos=10,
            global_start=15,
            global_end=20,
            segment_index=0
        )

        assert ref.node_id == "#/texts/0"
        assert ref.local_length == 5
        assert ref.global_length == 5

    def test_invalid_node_reference_creation(self):
        """Test validation errors in node reference creation."""
        # Test invalid local positions
        with pytest.raises(ValueError, match="end_pos must be greater than start_pos"):
            NodeReference("#/texts/0", 10, 10, 15, 20, 0)

        # Test invalid global positions
        with pytest.raises(ValueError, match="global_end must be greater than global_start"):
            NodeReference("#/texts/0", 5, 10, 20, 20, 0)

        # Test invalid segment index
        with pytest.raises(ValueError, match="segment_index must be non-negative"):
            NodeReference("#/texts/0", 5, 10, 15, 20, -1)


class TestDocumentIntegrationEndToEnd:
    """End-to-end integration tests."""

    @patch('cloakpivot.document.processor.load_document')
    def test_complete_document_processing_workflow(self, mock_load_document):
        """Test the complete document processing workflow."""
        # Create a mock document with realistic structure
        mock_text_item = Mock(spec=TextItem)
        mock_text_item.text = "Contact John Smith at 555-123-4567 for more information."
        mock_text_item.self_ref = "#/texts/0"

        mock_doc = Mock(spec=DoclingDocument)
        mock_doc.name = "test_document"
        mock_doc.texts = [mock_text_item]
        mock_doc.tables = []
        mock_doc.key_value_items = []
        mock_doc.pictures = []
        mock_doc.form_items = []

        mock_load_document.return_value = mock_doc

        # Step 1: Load document
        processor = DocumentProcessor()
        document = processor.load_document("test.docling.json")
        assert document == mock_doc

        # Step 2: Extract text segments
        extractor = TextExtractor()
        segments = extractor.extract_text_segments(document)
        assert len(segments) == 1
        assert segments[0].text == "Contact John Smith at 555-123-4567 for more information."

        # Step 3: Create mock detections
        detections = [
            RecognizerResult(entity_type="PERSON", start=8, end=18, score=0.9),
            RecognizerResult(entity_type="PHONE_NUMBER", start=22, end=34, score=0.95),
        ]

        # Step 4: Create anchors
        mapper = AnchorMapper()
        original_texts = {"#/texts/0": mock_text_item.text}
        anchors = mapper.create_anchors_from_detections(
            detections, segments, original_texts
        )

        assert len(anchors) == 2

        # Verify person anchor
        person_anchor = next(a for a in anchors if a.entity_type == "PERSON")
        assert person_anchor.node_id == "#/texts/0"
        assert person_anchor.start == 8
        assert person_anchor.end == 18

        # Verify phone anchor
        phone_anchor = next(a for a in anchors if a.entity_type == "PHONE_NUMBER")
        assert phone_anchor.node_id == "#/texts/0"
        assert phone_anchor.start == 22
        assert phone_anchor.end == 34

    def test_empty_document_workflow(self):
        """Test workflow with empty document."""
        # Create empty document
        mock_doc = Mock(spec=DoclingDocument)
        mock_doc.name = "empty_document"
        mock_doc.texts = []
        mock_doc.tables = []
        mock_doc.key_value_items = []
        mock_doc.pictures = []
        mock_doc.form_items = []

        # Extract text (should be empty)
        extractor = TextExtractor()
        segments = extractor.extract_text_segments(mock_doc)
        assert len(segments) == 0

        # Create anchors from empty detections
        mapper = AnchorMapper()
        anchors = mapper.create_anchors_from_detections([], segments, {})
        assert len(anchors) == 0
