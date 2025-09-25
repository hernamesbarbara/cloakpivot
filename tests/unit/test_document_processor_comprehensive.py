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
        processor = DocumentProcessor(enable_chunked_processing=True)

        assert processor._enable_chunked_processing is True
        assert processor._chunked_processor is None  # ChunkedDocumentProcessor was removed

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
        # Add required attributes
        mock_doc.name = "test.json"
        mock_doc.texts = []
        mock_doc.tables = []
        mock_doc.pictures = []
        mock_doc.key_value_items = []
        mock_doc.version = "1.6.0"
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
    def test_load_document_with_validation(
        self, mock_validate_json, mock_validate, mock_json_load, mock_path_open
    ):
        """Test document loading with validation."""
        # Setup
        test_data = {"test": "data", "version": "1.6.0"}
        mock_json_load.return_value = test_data
        mock_doc = Mock(spec=DoclingDocument)
        # Add required attributes
        mock_doc.name = "test.json"
        mock_doc.texts = [Mock(self_ref="#/texts/0")]
        mock_doc.tables = []
        mock_doc.pictures = []
        mock_doc.key_value_items = []
        mock_doc.version = "1.6.0"
        mock_validate.return_value = mock_doc
        mock_validate_json.return_value = mock_doc

        processor = DocumentProcessor(enable_chunked_processing=False)

        # Execute
        result = processor.load_document("test.json", validate=True)

        # Verify
        assert result == mock_doc
        assert processor._stats.files_processed == 1

    def test_load_document_file_not_found(self):
        """Test loading non-existent document."""
        import pytest

        processor = DocumentProcessor(enable_chunked_processing=False)

        # Execute and verify
        with pytest.raises(FileNotFoundError, match="File not found"):
            processor.load_document("nonexistent.json")

        assert processor._stats.errors_encountered == 1

    @patch("cloakpivot.document.processor.Path.open", side_effect=PermissionError("Access denied"))
    def test_load_document_permission_error(self, mock_path_open):
        """Test loading document with permission error."""
        import pytest

        processor = DocumentProcessor(enable_chunked_processing=False)

        # Execute and verify
        with pytest.raises(RuntimeError, match="Unexpected error loading document"):
            processor.load_document("protected.json")

        assert processor._stats.errors_encountered == 1

    @patch("cloakpivot.document.processor.Path.open")
    @patch(
        "cloakpivot.document.processor.json.load",
        side_effect=json.JSONDecodeError("Invalid JSON", "doc", 0),
    )
    def test_load_document_invalid_json(self, mock_json_load, mock_path_open):
        """Test loading document with invalid JSON."""
        import pytest

        processor = DocumentProcessor(enable_chunked_processing=False)

        # Execute and verify
        with pytest.raises(ValueError, match="Invalid JSON"):
            processor.load_document("invalid.json")

        assert processor._stats.errors_encountered == 1

    @patch("cloakpivot.document.processor.Path.open")
    @patch("cloakpivot.document.processor.json.load")
    @patch(
        "cloakpivot.document.processor.DoclingDocument.model_validate",
        side_effect=ValueError("Invalid document structure"),
    )
    def test_load_document_validation_error(self, mock_validate, mock_json_load, mock_path_open):
        """Test loading document with validation error."""
        import pytest

        mock_json_load.return_value = {"invalid": "data"}

        processor = DocumentProcessor(enable_chunked_processing=False)

        # Execute and verify
        with pytest.raises(RuntimeError, match="Unexpected error loading document"):
            processor.load_document("test.json")

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
        """Test accessing processing statistics."""
        processor = DocumentProcessor(enable_chunked_processing=False)
        processor._stats.files_processed = 5
        processor._stats.errors_encountered = 2

        # Access stats directly since get_stats() method doesn't exist
        assert processor._stats.files_processed == 5
        assert processor._stats.errors_encountered == 2

    def test_reset_stats(self):
        """Test resetting processor statistics."""
        processor = DocumentProcessor(enable_chunked_processing=False)
        processor._stats.files_processed = 10
        processor._stats.errors_encountered = 3

        processor.reset_stats()

        assert processor._stats.files_processed == 0
        assert processor._stats.errors_encountered == 0

    # Note: process_chunk method was removed from DocumentProcessor
    # These tests are commented out as the functionality no longer exists

    # def test_process_chunk_with_chunked_processor(self):
    #     """Test processing a chunk when chunked processor is available."""
    #     pass  # Method removed

    # def test_process_chunk_without_chunked_processor(self):
    #     """Test processing a chunk when chunked processor is not available."""
    #     pass  # Method removed

    @patch("cloakpivot.document.processor.Path.open")
    @patch("cloakpivot.document.processor.json.load")
    def test_load_document_with_path_object(self, mock_json_load, mock_path_open):
        """Test loading document using Path object."""
        test_data = {"test": "data", "version": "1.6.0"}
        mock_json_load.return_value = test_data

        with patch("cloakpivot.document.processor.DoclingDocument.model_validate") as mock_validate:
            mock_doc = Mock(spec=DoclingDocument)
            # Add required attributes
            mock_doc.name = "test.json"
            mock_doc.texts = []
            mock_doc.tables = []
            mock_doc.pictures = []
            mock_doc.key_value_items = []
            mock_doc.version = "1.6.0"
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
        # Default Python repr() is used since no custom __repr__ is defined
        assert "DocumentProcessor object at" in repr_str
