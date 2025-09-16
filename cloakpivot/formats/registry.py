"""Format registry for managing supported serialization formats."""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Any


class SupportedFormat(Enum):
    """Supported serialization formats."""

    JSON = "json"
    MARKDOWN = "markdown"
    MD = "markdown"  # Alias for MARKDOWN
    HTML = "html"
    TEXT = "text"
    XML = "xml"

    @classmethod
    def from_string(cls, value: str) -> SupportedFormat | None:
        """Create SupportedFormat from string value."""
        for fmt in cls:
            if fmt.value == value:
                return fmt
        return None


class FormatRegistry:
    """Registry for managing supported formats."""

    def __init__(self) -> None:
        """Initialize the format registry."""
        self._formats = {
            SupportedFormat.JSON.value: SupportedFormat.JSON,
            SupportedFormat.MARKDOWN.value: SupportedFormat.MARKDOWN,
            SupportedFormat.MD.value: SupportedFormat.MD,
            SupportedFormat.HTML.value: SupportedFormat.HTML,
            SupportedFormat.TEXT.value: SupportedFormat.TEXT,
            SupportedFormat.XML.value: SupportedFormat.XML,
        }

    def list_supported_formats(self) -> list[str]:
        """Get list of supported format names."""
        return list(self._formats.keys())

    def is_supported(self, format_name: str) -> bool:
        """Check if a format is supported."""
        return format_name in self._formats

    def is_format_supported(self, format_name: str) -> bool:
        """Check if a format is supported (alias for is_supported)."""
        return self.is_supported(format_name)

    def get_format(self, format_name: str) -> SupportedFormat | None:
        """Get a format enum by name."""
        return self._formats.get(format_name)

    def get_serializer(self, format_name: str, document: Any) -> Any:
        """Get a serializer for the format (placeholder)."""
        # This is a placeholder - actual implementation would return a serializer
        return None

    def validate_format_compatibility(self, input_format: str, output_format: str) -> bool:
        """Validate format compatibility."""
        return self.is_supported(input_format) and self.is_supported(output_format)

    def suggest_output_extension(self, format_name: str) -> str:
        """Suggest output extension for format."""
        fmt = self.get_format(format_name)
        if fmt:
            return f".{fmt.value}"
        return ".txt"

    def get_format_extensions(self, format_name: str) -> list[str]:
        """Get file extensions for a format."""
        if format_name == "json":
            return [".json"]
        if format_name == "markdown":
            return [".md", ".markdown"]
        if format_name == "html":
            return [".html", ".htm"]
        if format_name == "xml":
            return [".xml"]
        if format_name == "text":
            return [".txt", ".text"]
        return []

    def detect_format_from_content(self, content: str, path: Path) -> SupportedFormat | None:
        """Detect format from file content."""
        # Try to parse as JSON
        if path.suffix == ".json":
            try:
                json.loads(content)
                return SupportedFormat.JSON
            except json.JSONDecodeError:
                pass

        # Check for HTML markers
        if "<html" in content.lower() or "<!doctype html" in content.lower():
            return SupportedFormat.HTML

        # Check for XML markers
        if content.strip().startswith("<?xml"):
            return SupportedFormat.XML

        # Check for markdown markers
        if any(marker in content for marker in ["# ", "## ", "```", "[", "]("]):
            return SupportedFormat.MARKDOWN

        return None

    def detect_format_from_path(self, path: Path) -> SupportedFormat | None:
        """Detect format from file path/extension."""
        ext = path.suffix.lower()

        if ext == ".json":
            return SupportedFormat.JSON
        if ext in [".md", ".markdown"]:
            return SupportedFormat.MARKDOWN
        if ext in [".html", ".htm"]:
            return SupportedFormat.HTML
        if ext == ".xml":
            return SupportedFormat.XML
        if ext in [".txt", ".text"]:
            return SupportedFormat.TEXT

        return None
