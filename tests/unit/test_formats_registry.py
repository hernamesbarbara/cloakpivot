"""Comprehensive unit tests for cloakpivot.formats.registry module.

This test module provides full coverage of the format registry system,
including SupportedFormat enum and FormatRegistry class.
"""

from pathlib import Path
from unittest.mock import Mock

from cloakpivot.formats.registry import FormatRegistry, SupportedFormat


class TestSupportedFormat:
    """Test the SupportedFormat enum."""

    def test_enum_values(self):
        """Test that all enum values are defined correctly."""
        assert SupportedFormat.JSON.value == "json"
        assert SupportedFormat.MARKDOWN.value == "markdown"
        assert SupportedFormat.MD.value == "markdown"  # Alias
        assert SupportedFormat.HTML.value == "html"
        assert SupportedFormat.TEXT.value == "text"
        assert SupportedFormat.XML.value == "xml"

    def test_from_string_valid(self):
        """Test from_string with valid values."""
        assert SupportedFormat.from_string("json") == SupportedFormat.JSON
        assert SupportedFormat.from_string("markdown") == SupportedFormat.MARKDOWN
        assert SupportedFormat.from_string("html") == SupportedFormat.HTML
        assert SupportedFormat.from_string("text") == SupportedFormat.TEXT
        assert SupportedFormat.from_string("xml") == SupportedFormat.XML

    def test_from_string_invalid(self):
        """Test from_string with invalid values."""
        assert SupportedFormat.from_string("invalid") is None
        assert SupportedFormat.from_string("") is None
        assert SupportedFormat.from_string("JSON") is None  # Case sensitive
        assert SupportedFormat.from_string("yaml") is None

    def test_markdown_alias(self):
        """Test that MD is an alias for MARKDOWN."""
        # Both should resolve to "markdown" value
        assert SupportedFormat.MARKDOWN.value == "markdown"
        assert SupportedFormat.MD.value == "markdown"
        # from_string should return the first matching enum
        assert SupportedFormat.from_string("markdown") in [
            SupportedFormat.MARKDOWN,
            SupportedFormat.MD,
        ]


class TestFormatRegistry:
    """Test the FormatRegistry class."""

    def test_initialization(self):
        """Test FormatRegistry initialization."""
        registry = FormatRegistry()
        assert registry._formats is not None
        assert len(registry._formats) > 0

    def test_list_supported_formats(self):
        """Test list_supported_formats method."""
        registry = FormatRegistry()
        formats = registry.list_supported_formats()

        assert isinstance(formats, list)
        assert "json" in formats
        assert "markdown" in formats
        assert "html" in formats
        assert "text" in formats
        assert "xml" in formats

    def test_is_supported(self):
        """Test is_supported method."""
        registry = FormatRegistry()

        # Valid formats
        assert registry.is_supported("json") is True
        assert registry.is_supported("markdown") is True
        assert registry.is_supported("html") is True
        assert registry.is_supported("text") is True
        assert registry.is_supported("xml") is True

        # Invalid formats
        assert registry.is_supported("yaml") is False
        assert registry.is_supported("invalid") is False
        assert registry.is_supported("") is False

    def test_is_format_supported_alias(self):
        """Test is_format_supported (alias for is_supported)."""
        registry = FormatRegistry()

        assert registry.is_format_supported("json") is True
        assert registry.is_format_supported("invalid") is False

        # Should behave identically to is_supported
        for format_name in ["json", "markdown", "html", "text", "xml"]:
            assert registry.is_format_supported(format_name) == registry.is_supported(format_name)

    def test_get_format(self):
        """Test get_format method."""
        registry = FormatRegistry()

        # Valid formats
        assert registry.get_format("json") == SupportedFormat.JSON
        assert registry.get_format("markdown") in [SupportedFormat.MARKDOWN, SupportedFormat.MD]
        assert registry.get_format("html") == SupportedFormat.HTML
        assert registry.get_format("text") == SupportedFormat.TEXT
        assert registry.get_format("xml") == SupportedFormat.XML

        # Invalid formats
        assert registry.get_format("invalid") is None
        assert registry.get_format("") is None

    def test_get_serializer(self):
        """Test get_serializer method (placeholder)."""
        registry = FormatRegistry()
        mock_doc = Mock()

        # Currently returns None as it's a placeholder
        serializer = registry.get_serializer("json", mock_doc)
        assert serializer is None

    def test_validate_format_compatibility(self):
        """Test validate_format_compatibility method."""
        registry = FormatRegistry()

        # Valid combinations
        assert registry.validate_format_compatibility("json", "markdown") is True
        assert registry.validate_format_compatibility("html", "text") is True
        assert registry.validate_format_compatibility("xml", "json") is True

        # Invalid input format
        assert registry.validate_format_compatibility("invalid", "json") is False

        # Invalid output format
        assert registry.validate_format_compatibility("json", "invalid") is False

        # Both invalid
        assert registry.validate_format_compatibility("invalid1", "invalid2") is False

    def test_suggest_output_extension(self):
        """Test suggest_output_extension method."""
        registry = FormatRegistry()

        assert registry.suggest_output_extension("json") == ".json"
        assert registry.suggest_output_extension("markdown") == ".markdown"
        assert registry.suggest_output_extension("html") == ".html"
        assert registry.suggest_output_extension("text") == ".text"
        assert registry.suggest_output_extension("xml") == ".xml"

        # Invalid format returns default
        assert registry.suggest_output_extension("invalid") == ".txt"
        assert registry.suggest_output_extension("") == ".txt"

    def test_get_format_extensions(self):
        """Test get_format_extensions method."""
        registry = FormatRegistry()

        # JSON
        assert registry.get_format_extensions("json") == [".json"]

        # Markdown has multiple extensions
        md_exts = registry.get_format_extensions("markdown")
        assert ".md" in md_exts
        assert ".markdown" in md_exts

        # HTML
        html_exts = registry.get_format_extensions("html")
        assert ".html" in html_exts
        assert ".htm" in html_exts

        # XML
        assert registry.get_format_extensions("xml") == [".xml"]

        # Text
        text_exts = registry.get_format_extensions("text")
        assert ".txt" in text_exts
        assert ".text" in text_exts

        # Invalid format
        assert registry.get_format_extensions("invalid") == []
        assert registry.get_format_extensions("") == []

    def test_detect_format_from_content_json(self):
        """Test detect_format_from_content for JSON."""
        registry = FormatRegistry()

        # Valid JSON content
        json_content = '{"key": "value", "number": 42}'
        path = Path("test.json")
        assert registry.detect_format_from_content(json_content, path) == SupportedFormat.JSON

        # Invalid JSON but .json extension
        invalid_json = "not json content"
        assert registry.detect_format_from_content(invalid_json, path) is None

    def test_detect_format_from_content_html(self):
        """Test detect_format_from_content for HTML."""
        registry = FormatRegistry()
        path = Path("test.html")

        # HTML with doctype
        html1 = "<!DOCTYPE html>\n<html><body>Test</body></html>"
        assert registry.detect_format_from_content(html1, path) == SupportedFormat.HTML

        # HTML without doctype but with html tag
        html2 = "<html><head></head><body>Test</body></html>"
        assert registry.detect_format_from_content(html2, path) == SupportedFormat.HTML

        # Case insensitive
        html3 = "<HTML>Test</HTML>"
        assert registry.detect_format_from_content(html3, path) == SupportedFormat.HTML

    def test_detect_format_from_content_xml(self):
        """Test detect_format_from_content for XML."""
        registry = FormatRegistry()
        path = Path("test.xml")

        # Standard XML declaration
        xml_content = '<?xml version="1.0"?>\n<root><item>Test</item></root>'
        assert registry.detect_format_from_content(xml_content, path) == SupportedFormat.XML

        # With whitespace
        xml_with_space = '  <?xml version="1.0"?>\n<root></root>'
        assert registry.detect_format_from_content(xml_with_space, path) == SupportedFormat.XML

    def test_detect_format_from_content_markdown(self):
        """Test detect_format_from_content for Markdown."""
        registry = FormatRegistry()
        path = Path("test.md")

        # Heading
        md1 = "# Title\n\nSome content"
        assert registry.detect_format_from_content(md1, path) == SupportedFormat.MARKDOWN

        # Subheading
        md2 = "## Subtitle\n\nContent"
        assert registry.detect_format_from_content(md2, path) == SupportedFormat.MARKDOWN

        # Code block
        md3 = "```python\ncode here\n```"
        assert registry.detect_format_from_content(md3, path) == SupportedFormat.MARKDOWN

        # Link
        md4 = "Check this [link](http://example.com)"
        assert registry.detect_format_from_content(md4, path) == SupportedFormat.MARKDOWN

    def test_detect_format_from_content_unknown(self):
        """Test detect_format_from_content with unknown content."""
        registry = FormatRegistry()
        path = Path("test.txt")

        # Plain text without markers
        plain = "Just plain text without any format markers"
        assert registry.detect_format_from_content(plain, path) is None

    def test_detect_format_from_path(self):
        """Test detect_format_from_path method."""
        registry = FormatRegistry()

        # JSON
        assert registry.detect_format_from_path(Path("file.json")) == SupportedFormat.JSON
        assert registry.detect_format_from_path(Path("FILE.JSON")) == SupportedFormat.JSON

        # Markdown
        assert registry.detect_format_from_path(Path("doc.md")) == SupportedFormat.MARKDOWN
        assert registry.detect_format_from_path(Path("README.markdown")) == SupportedFormat.MARKDOWN

        # HTML
        assert registry.detect_format_from_path(Path("index.html")) == SupportedFormat.HTML
        assert registry.detect_format_from_path(Path("page.htm")) == SupportedFormat.HTML

        # XML
        assert registry.detect_format_from_path(Path("config.xml")) == SupportedFormat.XML

        # Text
        assert registry.detect_format_from_path(Path("notes.txt")) == SupportedFormat.TEXT
        assert registry.detect_format_from_path(Path("file.text")) == SupportedFormat.TEXT

        # Unknown extensions
        assert registry.detect_format_from_path(Path("file.yaml")) is None
        assert registry.detect_format_from_path(Path("file.pdf")) is None
        assert registry.detect_format_from_path(Path("file")) is None  # No extension

    def test_detect_format_from_path_case_insensitive(self):
        """Test that extension detection is case insensitive."""
        registry = FormatRegistry()

        # Various case combinations
        assert registry.detect_format_from_path(Path("file.JSON")) == SupportedFormat.JSON
        assert registry.detect_format_from_path(Path("file.Md")) == SupportedFormat.MARKDOWN
        assert registry.detect_format_from_path(Path("file.HTML")) == SupportedFormat.HTML
        assert registry.detect_format_from_path(Path("file.TXT")) == SupportedFormat.TEXT
