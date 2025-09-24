"""Comprehensive unit tests for document processor module."""

import json
from pathlib import Path
from unittest.mock import Mock, patch

from cloakpivot.document.processor import DocumentProcessingStats, DocumentProcessor
from cloakpivot.type_imports import DoclingDocument


class TestDocumentProcessingStats:
    """Test DocumentProcessingStats class."""

    def test_reset_stats(self):
        """Test resetting statistics."""
        stats = DocumentProcessingStats(files_processed=10, errors_encountered=3)
        stats.reset()
        assert stats.files_processed == 0
        assert stats.errors_encountered == 0


class TestDocumentProcessor:
    """Test DocumentProcessor class."""

    def test_init_with_chunked_processing_enabled(self):
        """Test initialization with chunked processing enabled."""
        with patch("cloakpivot.document.processor.ChunkedDocumentProcessor") as mock_chunked:
            mock_chunked_instance = Mock()
            mock_chunked.return_value = mock_chunked_instance

            processor = DocumentProcessor(enable_chunked_processing=True)

            assert processor._enable_chunked_processing is True
            assert processor._chunked_processor == mock_chunked_instance
            mock_chunked.assert_called_once()

    def test_init_with_chunked_processing_disabled(self):
        """Test initialization with chunked processing disabled."""
        processor = DocumentProcessor(enable_chunked_processing=False)

        assert processor._enable_chunked_processing is False
        assert processor._chunked_processor is None

    def test_init_with_chunked_processor_import_error(self):
        """Test initialization when ChunkedDocumentProcessor import fails."""
        # ChunkedDocumentProcessor has been removed, so this test is no longer relevant
        processor = DocumentProcessor(enable_chunked_processing=True)

        assert processor._enable_chunked_processing is True
        assert processor._chunked_processor is None

    @patch("cloakpivot.document.processor.Path.open")
    @patch("cloakpivot.document.processor.json.load")
    @patch("cloakpivot.document.processor.DoclingDocument.model_validate")
    def test_load_document_success(self, mock_validate, mock_json_load, mock_path_open):
        """Test successful document loading."""
        # Setup
        test_data = {"test": "data", "version": "1.6.0"}
        mock_json_load.return_value = test_data
        mock_doc = Mock(spec=DoclingDocument)
        mock_validate.return_value = mock_doc

        processor = DocumentProcessor(enable_chunked_processing=False)

        # Execute
        result = processor.load_document("test.json")

        # Verify
        assert result == mock_doc
        assert processor._stats.files_processed == 1
        assert processor._stats.errors_encountered == 0
        mock_validate.assert_called_once_with(test_data)

    @patch("cloakpivot.document.processor.Path.open")
    @patch("cloakpivot.document.processor.json.load")
    @patch("cloakpivot.document.processor.DoclingDocument.model_validate")
    @patch("cloakpivot.document.processor.DoclingDocument.model_validate_json")
    def test_load_document_with_validation(self, mock_validate_json, mock_validate, mock_json_load, mock_path_open):
        """Test document loading with validation."""
        # Setup
        test_data = {"test": "data", "version": "1.6.0"}
        mock_json_load.return_value = test_data
        mock_doc = Mock(spec=DoclingDocument)
        mock_validate.return_value = mock_doc
        mock_validate_json.return_value = mock_doc

        processor = DocumentProcessor(enable_chunked_processing=False)

        # Execute
        result = processor.load_document("test.json", validate=True)

        # Verify
        assert result == mock_doc
        assert processor._stats.files_processed == 1

    @patch("cloakpivot.document.processor.Path")
    def test_load_document_file_not_found(self, mock_path_class):
        """Test loading non-existent document."""
        mock_path = Mock()
        mock_path.exists.return_value = False
        mock_path_class.return_value = mock_path

        processor = DocumentProcessor(enable_chunked_processing=False)

        # Execute
        result = processor.load_document("nonexistent.json")

        # Verify
        assert result is None
        assert processor._stats.errors_encountered == 1

    @patch("cloakpivot.document.processor.Path.open", side_effect=PermissionError("Access denied"))
    def test_load_document_permission_error(self, mock_path_open):
        """Test loading document with permission error."""
        processor = DocumentProcessor(enable_chunked_processing=False)

        # Execute
        result = processor.load_document("protected.json")

        # Verify
        assert result is None
        assert processor._stats.errors_encountered == 1

    @patch("cloakpivot.document.processor.Path.open")
    @patch("cloakpivot.document.processor.json.load",
           side_effect=json.JSONDecodeError("Invalid JSON", "doc", 0))
    def test_load_document_invalid_json(self, mock_json_load, mock_path_open):
        """Test loading document with invalid JSON."""
        processor = DocumentProcessor(enable_chunked_processing=False)

        # Execute
        result = processor.load_document("invalid.json")

        # Verify
        assert result is None
        assert processor._stats.errors_encountered == 1

    @patch("cloakpivot.document.processor.Path.open")
    @patch("cloakpivot.document.processor.json.load")
    @patch("cloakpivot.document.processor.DoclingDocument.model_validate",
           side_effect=ValueError("Invalid document structure"))
    def test_load_document_validation_error(self, mock_validate, mock_json_load, mock_path_open):
        """Test loading document with validation error."""
        mock_json_load.return_value = {"invalid": "data"}

        processor = DocumentProcessor(enable_chunked_processing=False)

        # Execute
        result = processor.load_document("test.json")

        # Verify
        assert result is None
        assert processor._stats.errors_encountered == 1

    # Note: load_multiple method was removed from DocumentProcessor
    # These tests are commented out as the functionality no longer exists

    # def test_load_multiple_documents(self):
    #     """Test loading multiple documents."""
    #     pass  # Method removed

    # def test_load_multiple_documents_batch(self):
    #     """Test batch loading multiple documents."""
    #     pass  # Method removed

    # def test_load_multiple_with_errors(self):
    #     """Test loading multiple documents with some errors."""
    #     pass  # Method removed

    def test_get_stats(self):
        """Test getting processing statistics."""
        processor = DocumentProcessor(enable_chunked_processing=False)
        processor._stats.files_processed = 5
        processor._stats.errors_encountered = 2

        stats = processor.get_stats()
        assert stats.files_processed == 5
        assert stats.errors_encountered == 2

    def test_reset_stats(self):
        """Test resetting processor statistics."""
        processor = DocumentProcessor(enable_chunked_processing=False)
        processor._stats.files_processed = 10
        processor._stats.errors_encountered = 3

        processor.reset_stats()

        assert processor._stats.files_processed == 0
        assert processor._stats.errors_encountered == 0

    def test_process_chunk_with_chunked_processor(self):
        """Test processing a chunk when chunked processor is available."""
        mock_chunked = Mock()
        mock_chunked.process_chunk.return_value = {"processed": True}

        processor = DocumentProcessor(enable_chunked_processing=False)
        processor._chunked_processor = mock_chunked

        result = processor.process_chunk({"chunk": "data"}, chunk_size=100)

        assert result == {"processed": True}
        mock_chunked.process_chunk.assert_called_once_with({"chunk": "data"}, 100)

    def test_process_chunk_without_chunked_processor(self):
        """Test processing a chunk when chunked processor is not available."""
        processor = DocumentProcessor(enable_chunked_processing=False)
        processor._chunked_processor = None

        result = processor.process_chunk({"chunk": "data"}, chunk_size=100)

        assert result is None

    @patch("cloakpivot.document.processor.Path.open")
    @patch("cloakpivot.document.processor.json.load")
    def test_load_document_with_path_object(self, mock_json_load, mock_path_open):
        """Test loading document using Path object."""
        test_data = {"test": "data", "version": "1.6.0"}
        mock_json_load.return_value = test_data

        with patch("cloakpivot.document.processor.DoclingDocument.model_validate") as mock_validate:
            mock_doc = Mock(spec=DoclingDocument)
            mock_validate.return_value = mock_doc

            processor = DocumentProcessor(enable_chunked_processing=False)
            result = processor.load_document(Path("test.json"))

            assert result == mock_doc

    def test_repr(self):
        """Test string representation of processor."""
        processor = DocumentProcessor(enable_chunked_processing=True)
        processor._stats.files_processed = 3
        processor._stats.errors_encountered = 1

        repr_str = repr(processor)
        assert "DocumentProcessor" in repr_str
        assert "files_processed=3" in repr_str
        assert "errors_encountered=1" in repr_str
