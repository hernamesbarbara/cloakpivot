"""Extended unit tests for document processor to increase coverage."""

import json
import logging
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from cloakpivot.document.processor import DocumentProcessor
from cloakpivot.type_imports import DoclingDocument


class TestDocumentProcessorExtended:
    """Extended tests for DocumentProcessor functionality."""

    def test_validate_document_structure_valid(self):
        """Test validating a valid document structure."""
        processor = DocumentProcessor(enable_chunked_processing=False)

        # Create mock document
        mock_doc = Mock(spec=DoclingDocument)
        mock_doc.name = "test_doc"
        mock_doc.texts = [Mock(self_ref="#/texts/0")]
        mock_doc.tables = []
        mock_doc.key_value_items = []

        # Should not raise any exception
        processor._validate_document_structure(mock_doc)

    def test_validate_document_structure_no_name(self, caplog):
        """Test validating document without a name."""
        processor = DocumentProcessor(enable_chunked_processing=False)

        # Create mock document without name
        mock_doc = Mock(spec=DoclingDocument)
        mock_doc.name = None
        mock_doc.texts = [Mock(self_ref="#/texts/0")]
        mock_doc.tables = []
        mock_doc.key_value_items = []

        with caplog.at_level(logging.WARNING):
            processor._validate_document_structure(mock_doc)

        assert "Document has no name" in caplog.text

    def test_validate_document_structure_no_content(self, caplog):
        """Test validating document without text content."""
        processor = DocumentProcessor(enable_chunked_processing=False)

        # Create mock document without content
        mock_doc = Mock(spec=DoclingDocument)
        mock_doc.name = "empty_doc"
        mock_doc.texts = []
        mock_doc.tables = []
        mock_doc.key_value_items = []

        with caplog.at_level(logging.WARNING):
            processor._validate_document_structure(mock_doc)

        assert "no text content" in caplog.text

    def test_validate_document_structure_invalid_type(self):
        """Test validating non-DoclingDocument type."""
        processor = DocumentProcessor(enable_chunked_processing=False)

        # Create invalid document type
        mock_doc = {"not": "a DoclingDocument"}

        with pytest.raises(ValueError, match="Expected DoclingDocument"):
            processor._validate_document_structure(mock_doc)

    def test_validate_node_references_missing_self_ref(self, caplog):
        """Test validating nodes with missing self_ref."""
        processor = DocumentProcessor(enable_chunked_processing=False)

        # Create mock document with missing self_refs
        mock_doc = Mock(spec=DoclingDocument)
        mock_doc.name = "test_doc"

        # Text item without self_ref
        text_item1 = Mock()
        text_item1.self_ref = None
        text_item2 = Mock()
        text_item2.self_ref = "#/texts/1"
        mock_doc.texts = [text_item1, text_item2]

        # Table item without self_ref
        table_item = Mock()
        table_item.self_ref = None
        mock_doc.tables = [table_item]

        mock_doc.key_value_items = []

        with caplog.at_level(logging.WARNING):
            processor._validate_node_references(mock_doc)

        assert "Text item 0 missing self_ref" in caplog.text
        assert "Table item 0 missing self_ref" in caplog.text

    def test_get_processing_stats(self):
        """Test getting processing statistics."""
        processor = DocumentProcessor(enable_chunked_processing=False)
        processor._stats.files_processed = 5
        processor._stats.errors_encountered = 2

        stats = processor.get_processing_stats()

        assert stats.files_processed == 5
        assert stats.errors_encountered == 2

    def test_supports_format_json(self):
        """Test format support for JSON files."""
        processor = DocumentProcessor(enable_chunked_processing=False)

        assert processor.supports_format("test.json") is True
        assert processor.supports_format(Path("test.JSON")) is True

    def test_supports_format_docling(self):
        """Test format support for docling files."""
        processor = DocumentProcessor(enable_chunked_processing=False)

        assert processor.supports_format("test.docling.json") is True
        assert processor.supports_format("test_docling.txt") is True

    def test_supports_format_lexical(self):
        """Test format support for lexical files."""
        processor = DocumentProcessor(enable_chunked_processing=False)

        assert processor.supports_format("test.lexical.json") is True
        assert processor.supports_format("lexical_output.txt") is True

    def test_supports_format_unsupported(self):
        """Test format support for unsupported files."""
        processor = DocumentProcessor(enable_chunked_processing=False)

        assert processor.supports_format("test.pdf") is False
        assert processor.supports_format("test.txt") is False

    @patch("cloakpivot.document.processor.Path.open")
    @patch("cloakpivot.document.processor.json.load")
    def test_load_document_version_1_7_0(self, mock_json_load, mock_open, caplog):
        """Test loading document with version 1.7.0."""
        # Setup
        test_data = {"test": "data", "version": "1.7.0"}
        mock_json_load.return_value = test_data

        with patch("cloakpivot.document.processor.DoclingDocument.model_validate") as mock_validate:
            mock_doc = Mock(spec=DoclingDocument)
            mock_doc.name = "test_doc"
            mock_doc.version = "1.7.0"
            mock_doc.texts = []
            mock_doc.tables = []
            mock_doc.pictures = []
            mock_doc.key_value_items = []
            mock_validate.return_value = mock_doc

            processor = DocumentProcessor(enable_chunked_processing=False)

            with caplog.at_level(logging.DEBUG):
                result = processor.load_document("test.json")

            assert result == mock_doc
            assert "v1.7.0+ uses segment-local charspans" in caplog.text

    @patch("cloakpivot.document.processor.Path.open")
    @patch("cloakpivot.document.processor.json.load")
    def test_load_document_old_version(self, mock_json_load, mock_open, caplog):
        """Test loading document with older version."""
        # Setup
        test_data = {"test": "data", "version": "1.2.0"}
        mock_json_load.return_value = test_data

        with patch("cloakpivot.document.processor.DoclingDocument.model_validate") as mock_validate:
            mock_doc = Mock(spec=DoclingDocument)
            mock_doc.name = "test_doc"
            mock_doc.version = "1.2.0"
            mock_doc.texts = []
            mock_doc.tables = []
            mock_doc.pictures = []
            mock_doc.key_value_items = []
            mock_validate.return_value = mock_doc

            processor = DocumentProcessor(enable_chunked_processing=False)

            with caplog.at_level(logging.DEBUG):
                result = processor.load_document("test.json")

            assert result == mock_doc
            # Should not log v1.7.0 message
            assert "v1.7.0+ uses segment-local charspans" not in caplog.text

    @patch(
        "cloakpivot.document.processor.Path.open", side_effect=FileNotFoundError("File not found")
    )
    def test_load_document_raises_file_not_found(self, mock_open):
        """Test load_document raises FileNotFoundError."""
        processor = DocumentProcessor(enable_chunked_processing=False)

        with pytest.raises(FileNotFoundError, match="File not found"):
            processor.load_document("missing.json")

        assert processor._stats.errors_encountered == 1

    @patch("cloakpivot.document.processor.Path.open")
    @patch(
        "cloakpivot.document.processor.json.load",
        side_effect=json.JSONDecodeError("Invalid JSON", "doc", 0),
    )
    def test_load_document_raises_value_error(self, mock_json_load, mock_open):
        """Test load_document raises ValueError on invalid JSON."""
        processor = DocumentProcessor(enable_chunked_processing=False)

        with pytest.raises(ValueError, match="Invalid JSON"):
            processor.load_document("invalid.json")

        assert processor._stats.errors_encountered == 1

    @patch("cloakpivot.document.processor.Path.open")
    @patch("cloakpivot.document.processor.json.load")
    @patch(
        "cloakpivot.document.processor.DoclingDocument.model_validate",
        side_effect=Exception("Unexpected error"),
    )
    def test_load_document_raises_runtime_error(self, mock_validate, mock_json_load, mock_open):
        """Test load_document raises RuntimeError on unexpected exceptions."""
        mock_json_load.return_value = {"test": "data"}
        processor = DocumentProcessor(enable_chunked_processing=False)

        with pytest.raises(RuntimeError, match="Unexpected error"):
            processor.load_document("test.json")

        assert processor._stats.errors_encountered == 1

    @patch("cloakpivot.document.processor.Path.open")
    @patch("cloakpivot.document.processor.json.load")
    def test_load_document_with_validation_disabled(self, mock_json_load, mock_open):
        """Test loading document with validation disabled."""
        # Setup
        test_data = {"test": "data", "version": "1.6.0"}
        mock_json_load.return_value = test_data

        with patch("cloakpivot.document.processor.DoclingDocument.model_validate") as mock_validate:
            mock_doc = Mock(spec=DoclingDocument)
            mock_doc.name = "test_doc"
            mock_doc.version = "1.6.0"
            mock_doc.texts = []
            mock_doc.tables = []
            mock_doc.pictures = []
            mock_doc.key_value_items = []
            mock_validate.return_value = mock_doc

            processor = DocumentProcessor(enable_chunked_processing=False)

            # Mock _validate_document_structure to verify it's not called
            with patch.object(processor, "_validate_document_structure") as mock_validate_structure:
                result = processor.load_document("test.json", validate=False)

                assert result == mock_doc
                mock_validate_structure.assert_not_called()

    def test_init_with_debug_logging(self, caplog):
        """Test processor initialization with debug logging."""
        with caplog.at_level(logging.DEBUG):
            DocumentProcessor(enable_chunked_processing=True)

        assert "DocumentProcessor initialized" in caplog.text
        assert "chunked_processing=True" in caplog.text

    def test_reset_stats_with_logging(self, caplog):
        """Test reset_stats with debug logging."""
        processor = DocumentProcessor(enable_chunked_processing=False)
        processor._stats.files_processed = 10

        with caplog.at_level(logging.DEBUG):
            processor.reset_stats()

        assert processor._stats.files_processed == 0
        assert "Processing statistics reset" in caplog.text
