"""Tests for CloakPivot format support and serialization."""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from docling_core.types import DoclingDocument

from cloakpivot.formats import (
    CloakPivotSerializer,
    FormatRegistry,
    SerializationError,
    SupportedFormat,
)


class TestSupportedFormat:
    """Test SupportedFormat enum functionality."""

    def test_from_string_valid_formats(self):
        """Test conversion from string to SupportedFormat enum."""
        assert SupportedFormat.from_string("lexical") == SupportedFormat.LEXICAL
        assert SupportedFormat.from_string("docling") == SupportedFormat.DOCLING
        assert SupportedFormat.from_string("markdown") == SupportedFormat.MARKDOWN
        assert SupportedFormat.from_string("md") == SupportedFormat.MARKDOWN
        assert SupportedFormat.from_string("html") == SupportedFormat.HTML

    def test_from_string_case_insensitive(self):
        """Test case insensitive format conversion."""
        assert SupportedFormat.from_string("LEXICAL") == SupportedFormat.LEXICAL
        assert SupportedFormat.from_string("Markdown") == SupportedFormat.MARKDOWN
        assert SupportedFormat.from_string("  HTML  ") == SupportedFormat.HTML

    def test_from_string_invalid_format(self):
        """Test invalid format raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported format"):
            SupportedFormat.from_string("invalid")

        with pytest.raises(ValueError, match="Unsupported format"):
            SupportedFormat.from_string("")

    def test_list_formats(self):
        """Test listing all supported formats."""
        formats = SupportedFormat.list_formats()
        expected = ["lexical", "docling", "markdown", "md", "html", "doctags"]
        assert all(fmt in formats for fmt in expected)
        assert len(formats) == 6


class TestFormatRegistry:
    """Test FormatRegistry functionality."""

    def test_initialization(self):
        """Test FormatRegistry initializes correctly."""
        registry = FormatRegistry()
        assert registry is not None

        # Should have some supported formats
        formats = registry.list_supported_formats()
        assert len(formats) > 0
        assert "lexical" in formats

    def test_is_format_supported(self):
        """Test format support checking."""
        registry = FormatRegistry()

        # These should be supported
        assert registry.is_format_supported("lexical")
        assert registry.is_format_supported("html")
        assert registry.is_format_supported("markdown")

        # These should not be supported
        assert not registry.is_format_supported("invalid")
        assert not registry.is_format_supported("")

    def test_detect_format_from_path(self):
        """Test format detection from file paths."""
        registry = FormatRegistry()

        # Test specific format indicators
        assert (
            registry.detect_format_from_path("doc.lexical.json")
            == SupportedFormat.LEXICAL
        )
        assert (
            registry.detect_format_from_path("doc.docling.json")
            == SupportedFormat.DOCLING
        )

        # Test extension mappings
        assert (
            registry.detect_format_from_path("document.md") == SupportedFormat.MARKDOWN
        )
        assert registry.detect_format_from_path("document.html") == SupportedFormat.HTML

        # Test unknown formats
        assert registry.detect_format_from_path("document.txt") is None

    def test_detect_format_from_content(self):
        """Test format detection from content."""
        registry = FormatRegistry()

        # Test JSON formats
        docling_content = '{"texts": [], "tables": [], "name": "test"}'
        assert (
            registry.detect_format_from_content(docling_content)
            == SupportedFormat.DOCLING
        )

        lexical_content = '{"root": {"children": []}, "version": "1.0"}'
        assert (
            registry.detect_format_from_content(lexical_content)
            == SupportedFormat.LEXICAL
        )

        # Test HTML format
        html_content = "<html><body><p>Test</p></body></html>"
        assert registry.detect_format_from_content(html_content) == SupportedFormat.HTML

        # Test Markdown format
        markdown_content = "# Title\n\n- List item\n\n**Bold text**"
        assert (
            registry.detect_format_from_content(markdown_content)
            == SupportedFormat.MARKDOWN
        )

        # Test unknown content
        unknown_content = "Plain text without clear format indicators"
        assert registry.detect_format_from_content(unknown_content) is None

    def test_get_serializer(self):
        """Test getting serializer instances."""
        registry = FormatRegistry()

        # Should return serializer for supported formats
        serializer = registry.get_serializer("lexical")
        assert serializer is not None

        # Should raise error for unsupported formats
        with pytest.raises(ValueError, match="Unsupported format"):
            registry.get_serializer("invalid")

    def test_get_format_extensions(self):
        """Test getting file extensions for formats."""
        registry = FormatRegistry()

        lexical_extensions = registry.get_format_extensions("lexical")
        assert ".lexical.json" in lexical_extensions
        assert ".json" in lexical_extensions

        markdown_extensions = registry.get_format_extensions("markdown")
        assert ".md" in markdown_extensions
        assert ".markdown" in markdown_extensions

    def test_suggest_output_extension(self):
        """Test output extension suggestions."""
        registry = FormatRegistry()

        assert registry.suggest_output_extension("lexical") == ".lexical.json"
        assert registry.suggest_output_extension("docling") == ".docling.json"
        assert registry.suggest_output_extension("markdown") == ".md"
        assert registry.suggest_output_extension("html") == ".html"
        assert registry.suggest_output_extension("invalid") == ".txt"

    def test_validate_format_compatibility(self):
        """Test format compatibility validation."""
        registry = FormatRegistry()

        # Supported formats should be compatible
        assert registry.validate_format_compatibility("lexical", "markdown")
        assert registry.validate_format_compatibility("html", "docling")
        assert registry.validate_format_compatibility("markdown", "html")

        # Unsupported formats should not be compatible
        assert not registry.validate_format_compatibility("invalid", "lexical")
        assert not registry.validate_format_compatibility("lexical", "invalid")


class TestCloakPivotSerializer:
    """Test CloakPivotSerializer functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.serializer = CloakPivotSerializer()

        # Create mock document
        self.mock_document = Mock(spec=DoclingDocument)
        self.mock_document.name = "test_document"
        self.mock_document.texts = []
        self.mock_document.tables = []
        self.mock_document.pictures = []

    def test_initialization(self):
        """Test CloakPivotSerializer initializes correctly."""
        serializer = CloakPivotSerializer()
        assert serializer is not None
        assert len(serializer.supported_formats) > 0

    @patch("cloakpivot.formats.serialization.FormatRegistry")
    def test_serialize_document_success(self, mock_registry_class):
        """Test successful document serialization."""
        # Setup mocks
        mock_registry = Mock()
        mock_registry_class.return_value = mock_registry
        mock_registry.is_format_supported.return_value = True

        mock_serializer = Mock()
        mock_serializer.serialize.return_value = Mock(
            text="serialized content", metadata={}
        )
        mock_registry.get_serializer.return_value = mock_serializer

        serializer = CloakPivotSerializer()
        result = serializer.serialize_document(self.mock_document, "markdown")

        assert result.content == "serialized content"
        assert result.format_name == "markdown"
        assert result.size_bytes > 0

    @patch("cloakpivot.formats.serialization.FormatRegistry")
    def test_serialize_document_unsupported_format(self, mock_registry_class):
        """Test serialization with unsupported format."""
        mock_registry = Mock()
        mock_registry_class.return_value = mock_registry
        mock_registry.is_format_supported.return_value = False

        serializer = CloakPivotSerializer()

        with pytest.raises(SerializationError, match="Unsupported format"):
            serializer.serialize_document(self.mock_document, "invalid")

    def test_process_markdown_content(self):
        """Test markdown-specific content processing."""
        serializer = CloakPivotSerializer()

        # Test markdown escaping
        content = "Some text with [***] and [###] tokens"
        processed = serializer._process_markdown_content(content)

        assert "[\\*\\*\\*]" in processed
        assert "[\\#\\#\\#]" in processed

    def test_process_html_content(self):
        """Test HTML-specific content processing."""
        serializer = CloakPivotSerializer()

        # Test HTML masking styles
        content = "Some text with [REDACTED] and [MASKED] content"
        processed = serializer._process_html_content(content)

        assert '<span class="cloak-redacted">' in processed
        assert '<span class="cloak-masked">' in processed
        assert "<style>" in processed

    def test_detect_format(self):
        """Test format detection."""
        serializer = CloakPivotSerializer()

        # Test detection
        assert serializer.detect_format("doc.lexical.json") == "lexical"
        assert serializer.detect_format("doc.md") == "markdown"
        assert serializer.detect_format("doc.html") == "html"

    def test_get_format_info(self):
        """Test getting format information."""
        serializer = CloakPivotSerializer()

        # Test valid format
        info = serializer.get_format_info("markdown")
        assert info["supported"] is True
        assert info["is_text_format"] is True
        assert info["suggested_extension"] == ".md"

        # Test invalid format
        info = serializer.get_format_info("invalid")
        assert info["supported"] is False
        assert "error" in info


class TestSerializationResult:
    """Test SerializationResult functionality."""

    def test_initialization(self):
        """Test SerializationResult initialization."""
        from cloakpivot.formats.serialization import SerializationResult

        result = SerializationResult(
            content="test content",
            format_name="markdown",
            size_bytes=100,
            metadata={"test": "data"},
        )

        assert result.content == "test content"
        assert result.format_name == "markdown"
        assert result.size_bytes == 100
        assert result.metadata == {"test": "data"}
        assert result.size_kb == 100 / 1024

    def test_save_to_file(self):
        """Test saving serialized content to file."""
        from cloakpivot.formats.serialization import SerializationResult

        result = SerializationResult(
            content="# Test Content\n\nThis is test content.",
            format_name="markdown",
            size_bytes=35,
            metadata={},
        )

        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".md"
        ) as tmp_file:
            tmp_path = Path(tmp_file.name)

        try:
            result.save_to_file(tmp_path)

            # Verify file was created and contains correct content
            assert tmp_path.exists()
            content = tmp_path.read_text(encoding="utf-8")
            assert content == result.content

        finally:
            tmp_path.unlink(missing_ok=True)


class TestIntegrationScenarios:
    """Test integration scenarios for format support."""

    def test_format_chain_conversion(self):
        """Test conversion chain between multiple formats."""
        # This would test: lexical -> markdown -> html -> docling
        # Implementation depends on having actual test documents
        pass

    def test_masked_content_preservation(self):
        """Test that masked content is preserved across format conversions."""
        # This would test format conversion with masked tokens
        pass

    def test_round_trip_fidelity(self):
        """Test round-trip conversion fidelity."""
        # This would test: format A -> format B -> format A
        pass


if __name__ == "__main__":
    pytest.main([__file__])
