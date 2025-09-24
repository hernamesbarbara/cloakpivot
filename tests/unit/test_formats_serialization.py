"""Comprehensive unit tests for cloakpivot.formats.serialization module.

This test module provides full coverage of the serialization system,
including SerializationResult, CloakPivotSerializer, and error handling.
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from cloakpivot.formats.serialization import (
    CloakPivotSerializer,
    SerializationError,
    SerializationResult,
)


class TestSerializationError:
    """Test the SerializationError exception class."""

    def test_basic_error(self):
        """Test basic SerializationError initialization."""
        error = SerializationError(
            message="Serialization failed",
            format_name="json",
        )
        assert str(error) == "Serialization failed"
        assert error.format_name == "json"
        assert error.context == {}

    def test_error_with_context(self):
        """Test SerializationError with context."""
        context = {"document": "test.md", "error": "Invalid format"}
        error = SerializationError(
            message="Failed to serialize",
            format_name="yaml",
            context=context,
        )
        assert error.format_name == "yaml"
        assert error.context == context
        assert error.context["document"] == "test.md"


class TestSerializationResult:
    """Test the SerializationResult dataclass."""

    def test_basic_initialization(self):
        """Test basic SerializationResult initialization."""
        result = SerializationResult(
            content="Test content",
            format_name="markdown",
            size_bytes=12,
            metadata={},
        )
        assert result.content == "Test content"
        assert result.format_name == "markdown"
        assert result.size_bytes == 12
        assert result.metadata == {}

    def test_size_kb_property(self):
        """Test size_kb property calculation."""
        result = SerializationResult(
            content="x" * 1024,
            format_name="text",
            size_bytes=1024,
            metadata={},
        )
        assert result.size_kb == 1.0

        result2 = SerializationResult(
            content="x" * 2048,
            format_name="text",
            size_bytes=2048,
            metadata={},
        )
        assert result2.size_kb == 2.0

    def test_save_to_file_string_path(self):
        """Test save_to_file with string path."""
        result = SerializationResult(
            content="Test content to save",
            format_name="markdown",
            size_bytes=20,
            metadata={"key": "value"},
        )

        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            file_path = f.name

        try:
            result.save_to_file(file_path)

            # Verify file was created with correct content
            with open(file_path) as f:
                saved_content = f.read()
            assert saved_content == "Test content to save"
        finally:
            Path(file_path).unlink(missing_ok=True)

    def test_save_to_file_path_object(self):
        """Test save_to_file with Path object."""
        result = SerializationResult(
            content="Path object test",
            format_name="json",
            size_bytes=16,
            metadata={},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test.json"
            result.save_to_file(file_path)

            assert file_path.exists()
            assert file_path.read_text() == "Path object test"

    def test_save_to_file_creates_parent_dirs(self):
        """Test that save_to_file creates parent directories."""
        result = SerializationResult(
            content="Nested dir test",
            format_name="yaml",
            size_bytes=15,
            metadata={},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "nested" / "dirs" / "test.yaml"
            assert not file_path.parent.exists()

            result.save_to_file(file_path)

            assert file_path.parent.exists()
            assert file_path.exists()
            assert file_path.read_text() == "Nested dir test"

    @patch("cloakpivot.formats.serialization.logger")
    def test_save_to_file_logging(self, mock_logger):
        """Test that save_to_file logs information."""
        result = SerializationResult(
            content="Log test",
            format_name="text",
            size_bytes=8,
            metadata={},
        )

        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            file_path = f.name

        try:
            result.save_to_file(file_path)
            mock_logger.info.assert_called_once()
            log_message = str(mock_logger.info.call_args)
            assert "saved to" in log_message
            assert "KB" in log_message
        finally:
            Path(file_path).unlink(missing_ok=True)


class TestCloakPivotSerializer:
    """Test the CloakPivotSerializer class."""

    def test_initialization(self):
        """Test CloakPivotSerializer initialization."""
        with patch("cloakpivot.formats.serialization.FormatRegistry") as mock_registry:
            serializer = CloakPivotSerializer()
            assert serializer._registry is not None
            mock_registry.assert_called_once()

    def test_supported_formats_property(self):
        """Test supported_formats property."""
        with patch("cloakpivot.formats.serialization.FormatRegistry") as mock_registry:
            mock_instance = Mock()
            mock_instance.list_supported_formats.return_value = ["json", "yaml", "markdown"]
            mock_registry.return_value = mock_instance

            serializer = CloakPivotSerializer()
            formats = serializer.supported_formats

            assert formats == ["json", "yaml", "markdown"]
            mock_instance.list_supported_formats.assert_called_once()

    def test_serialize_document_unsupported_format(self):
        """Test serialize_document with unsupported format."""
        with patch("cloakpivot.formats.serialization.FormatRegistry") as mock_registry:
            mock_instance = Mock()
            mock_instance.is_format_supported.return_value = False
            mock_instance.list_supported_formats.return_value = ["json", "yaml"]
            mock_registry.return_value = mock_instance

            serializer = CloakPivotSerializer()
            mock_doc = Mock()
            mock_doc.name = "test.md"

            with pytest.raises(SerializationError) as exc_info:
                serializer.serialize_document(mock_doc, "unsupported")

            assert exc_info.value.format_name == "unsupported"
            assert "Unsupported format" in str(exc_info.value)

    def test_serialize_document_success(self):
        """Test successful document serialization."""
        with patch("cloakpivot.formats.serialization.FormatRegistry") as mock_registry:
            # Set up registry mock
            mock_registry_instance = Mock()
            mock_registry_instance.is_format_supported.return_value = True

            # Set up serializer mock
            mock_serializer = Mock()
            mock_serializer.serialize.return_value = Mock(
                text="Serialized content", metadata={"key": "value"}
            )
            mock_registry_instance.get_serializer.return_value = mock_serializer

            mock_registry.return_value = mock_registry_instance

            # Set up document mock
            mock_doc = Mock()
            mock_doc.name = "test.md"
            mock_doc.texts = ["text1", "text2"]
            mock_doc.tables = ["table1"]

            serializer = CloakPivotSerializer()
            result = serializer.serialize_document(mock_doc, "markdown")

            assert isinstance(result, SerializationResult)
            assert result.format_name == "markdown"
            assert result.content == "Serialized content"
            assert result.size_bytes == len(b"Serialized content")
            assert result.metadata["document_name"] == "test.md"
            assert result.metadata["document_texts"] == 2
            assert result.metadata["document_tables"] == 1

    def test_serialize_document_with_string_return(self):
        """Test serialization when serializer returns string."""
        with patch("cloakpivot.formats.serialization.FormatRegistry") as mock_registry:
            mock_registry_instance = Mock()
            mock_registry_instance.is_format_supported.return_value = True

            # Serializer returns plain string
            mock_serializer = Mock()
            mock_serializer.serialize.return_value = "Plain string content"
            mock_registry_instance.get_serializer.return_value = mock_serializer

            mock_registry.return_value = mock_registry_instance

            mock_doc = Mock()
            mock_doc.name = "test.txt"
            mock_doc.texts = []
            mock_doc.tables = []

            serializer = CloakPivotSerializer()
            result = serializer.serialize_document(mock_doc, "text")

            assert result.content == "Plain string content"
            assert result.metadata["serializer_metadata"] == {}

    def test_serialize_document_with_kwargs(self):
        """Test serialize_document with additional kwargs."""
        with patch("cloakpivot.formats.serialization.FormatRegistry") as mock_registry:
            mock_registry_instance = Mock()
            mock_registry_instance.is_format_supported.return_value = True

            mock_serializer = Mock()
            mock_serializer.serialize.return_value = "Content"
            mock_registry_instance.get_serializer.return_value = mock_serializer

            mock_registry.return_value = mock_registry_instance

            mock_doc = Mock()
            mock_doc.name = "test.json"
            mock_doc.texts = []
            mock_doc.tables = []

            serializer = CloakPivotSerializer()
            result = serializer.serialize_document(
                mock_doc,
                "json",
                custom_param="value",
                another_param=42,
            )

            assert result.metadata["custom_param"] == "value"
            assert result.metadata["another_param"] == 42

    def test_serialize_document_exception_handling(self):
        """Test exception handling during serialization."""
        with patch("cloakpivot.formats.serialization.FormatRegistry") as mock_registry:
            mock_registry_instance = Mock()
            mock_registry_instance.is_format_supported.return_value = True

            # Serializer raises exception
            mock_serializer = Mock()
            mock_serializer.serialize.side_effect = RuntimeError("Serialization failed")
            mock_registry_instance.get_serializer.return_value = mock_serializer

            mock_registry.return_value = mock_registry_instance

            mock_doc = Mock()
            mock_doc.name = "test.yaml"
            mock_doc.texts = []
            mock_doc.tables = []

            serializer = CloakPivotSerializer()

            with pytest.raises(SerializationError) as exc_info:
                serializer.serialize_document(mock_doc, "yaml")

            assert exc_info.value.format_name == "yaml"
            assert "Failed to serialize document" in str(exc_info.value)
            assert exc_info.value.context["document_name"] == "test.yaml"

    @patch("cloakpivot.formats.serialization.logger")
    def test_serialize_document_logging(self, mock_logger):
        """Test logging during serialization."""
        with patch("cloakpivot.formats.serialization.FormatRegistry") as mock_registry:
            mock_registry_instance = Mock()
            mock_registry_instance.is_format_supported.return_value = True

            mock_serializer = Mock()
            mock_serializer.serialize.return_value = "Content"
            mock_registry_instance.get_serializer.return_value = mock_serializer

            mock_registry.return_value = mock_registry_instance

            mock_doc = Mock()
            mock_doc.name = "test.md"
            mock_doc.texts = []
            mock_doc.tables = []

            serializer = CloakPivotSerializer()
            result = serializer.serialize_document(mock_doc, "markdown")

            # Check debug and info logs were called
            assert mock_logger.debug.call_count >= 1
            assert mock_logger.info.call_count >= 1

    def test_apply_format_specific_processing(self):
        """Test _apply_format_specific_processing method."""
        serializer = CloakPivotSerializer()

        # Test that method exists and returns content unchanged by default
        content = "Test content"
        processed = serializer._apply_format_specific_processing(content, "json")
        assert processed == content

        processed2 = serializer._apply_format_specific_processing(content, "markdown")
        assert processed2 == content
