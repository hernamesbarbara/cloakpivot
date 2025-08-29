"""Format registry system for CloakPivot serialization."""

import logging
from enum import Enum
from pathlib import Path
from typing import Optional, Union

from docpivot import SerializerProvider

logger = logging.getLogger(__name__)


class SupportedFormat(Enum):
    """Enumeration of formats supported by CloakPivot."""

    LEXICAL = "lexical"
    DOCLING = "docling"
    MARKDOWN = "markdown"
    MD = "md"
    HTML = "html"
    DOCTAGS = "doctags"

    @classmethod
    def from_string(cls, format_str: str) -> "SupportedFormat":
        """Convert string to SupportedFormat enum."""
        format_str = format_str.lower().strip()

        # Handle aliases
        if format_str in ("md", "markdown"):
            return cls.MARKDOWN

        for fmt in cls:
            if fmt.value == format_str:
                return fmt

        raise ValueError(f"Unsupported format: {format_str}")

    @classmethod
    def list_formats(cls) -> list[str]:
        """List all supported format strings."""
        return [fmt.value for fmt in cls]


class FormatRegistry:
    """Registry for managing document formats and their serializers.

    This class provides a high-level interface to docpivot's SerializerProvider
    with CloakPivot-specific enhancements like format detection and validation.
    """

    # File extension mappings
    EXTENSION_MAP = {
        ".json": {SupportedFormat.LEXICAL, SupportedFormat.DOCLING},
        ".lexical.json": {SupportedFormat.LEXICAL},
        ".docling.json": {SupportedFormat.DOCLING},
        ".md": {SupportedFormat.MARKDOWN},
        ".markdown": {SupportedFormat.MARKDOWN},
        ".html": {SupportedFormat.HTML},
        ".htm": {SupportedFormat.HTML},
    }

    def __init__(self):
        """Initialize the format registry."""
        self._provider = SerializerProvider()
        self._custom_serializers: dict[str, type] = {}

        # Enable registry integration for extensibility
        self._provider.enable_registry_integration()

        logger.debug(f"FormatRegistry initialized with formats: {self.list_supported_formats()}")

    def is_format_supported(self, format_name: str) -> bool:
        """Check if a format is supported.

        Args:
            format_name: Name of the format to check

        Returns:
            True if the format is supported, False otherwise
        """
        try:
            SupportedFormat.from_string(format_name)
            # Handle special case formats
            if format_name == "docling":
                return True  # We handle docling format specially
            return self._provider.is_format_supported(format_name)
        except ValueError:
            return False

    def list_supported_formats(self) -> list[str]:
        """List all supported formats.

        Returns:
            List of supported format names
        """
        formats = self._provider.list_formats().copy()
        # Add our special case formats
        if "docling" not in formats:
            formats.append("docling")
        return formats

    def detect_format_from_path(self, file_path: Union[str, Path]) -> Optional[SupportedFormat]:
        """Detect format from file path based on extension and naming conventions.

        Args:
            file_path: Path to the file

        Returns:
            Detected format or None if unable to determine

        Examples:
            >>> registry = FormatRegistry()
            >>> registry.detect_format_from_path("doc.lexical.json")
            SupportedFormat.LEXICAL
            >>> registry.detect_format_from_path("doc.md")
            SupportedFormat.MARKDOWN
        """
        path = Path(file_path)

        # Check for specific format indicators in filename
        if ".lexical.json" in path.name:
            return SupportedFormat.LEXICAL
        elif ".docling.json" in path.name:
            return SupportedFormat.DOCLING

        # Check extension mappings
        extension = path.suffix.lower()
        if extension in self.EXTENSION_MAP:
            formats = self.EXTENSION_MAP[extension]
            # If multiple formats possible for extension, return the first one
            # Could be enhanced with content-based detection
            return next(iter(formats))

        return None

    def detect_format_from_content(self, content: str, file_path: Optional[Path] = None) -> Optional[SupportedFormat]:
        """Detect format from file content.

        Args:
            content: File content to analyze
            file_path: Optional file path for additional context

        Returns:
            Detected format or None if unable to determine
        """
        content = content.strip()

        # JSON formats - check these first before markdown
        if content.startswith('{') and content.endswith('}'):
            try:
                import json
                data = json.loads(content)

                # Check for docling-specific fields
                if isinstance(data, dict):
                    # DoclingDocument format has schema_name field and other specific structure
                    if "schema_name" in data and data.get("schema_name") == "DoclingDocument":
                        return SupportedFormat.DOCLING
                    elif "texts" in data and "tables" in data and "schema_name" not in data:
                        # Legacy or variant DoclingDocument format
                        return SupportedFormat.DOCLING
                    elif "root" in data and "children" in data:
                        return SupportedFormat.LEXICAL
                    # If it's JSON but not recognizable, check if it might be lexical with children deeply nested
                    elif "root" in data:
                        # Deeper check for lexical format
                        root = data.get("root", {})
                        if isinstance(root, dict) and "children" in root:
                            return SupportedFormat.LEXICAL

            except json.JSONDecodeError:
                pass

        # HTML format
        if any(tag in content.lower() for tag in ["<html", "<body", "<div", "<p>"]):
            return SupportedFormat.HTML

        # Markdown format - look for markdown syntax
        markdown_indicators = ["# ", "## ", "- ", "* ", "**", "__", "[", "]("]
        if any(indicator in content for indicator in markdown_indicators):
            return SupportedFormat.MARKDOWN

        # Fall back to file path detection
        if file_path:
            return self.detect_format_from_path(file_path)

        return None

    def get_serializer(self, format_name: str, document=None):
        """Get a serializer instance for the specified format.

        Args:
            format_name: Name of the format
            document: DoclingDocument needed for docpivot serializers (optional for compatibility)

        Returns:
            Serializer instance

        Raises:
            ValueError: If the format is not supported
        """
        # Validate format
        try:
            SupportedFormat.from_string(format_name)
        except ValueError as e:
            raise ValueError(f"Unsupported format: {format_name}") from e

        # Handle special case formats that aren't supported by docpivot SerializerProvider
        if format_name == "docling":
            # For docling format, we use the DoclingDocument's native export_to_dict method
            class DoclingSerializer:
                def __init__(self, document):
                    self.document = document

                def serialize(self):
                    import json
                    # Use model_dump() instead of export_to_dict() for complete serialization
                    result_dict = self.document.model_dump()
                    return json.dumps(result_dict, indent=2)

            if document is None:
                return lambda doc: DoclingSerializer(doc)
            else:
                return DoclingSerializer(document)

        # Check if format is supported by docpivot
        if not self._provider.is_format_supported(format_name):
            raise ValueError(f"Format '{format_name}' not available in docpivot SerializerProvider")

        # For docpivot, we need a document to get the serializer
        if document is None:
            # Return a factory function that creates the serializer when given a document
            return lambda doc: self._provider.get_serializer(format_name, doc)
        else:
            return self._provider.get_serializer(format_name, document)

    def register_custom_serializer(self, format_name: str, serializer_class: type) -> None:
        """Register a custom serializer for a format.

        Args:
            format_name: Name of the format
            serializer_class: Serializer class to register
        """
        self._custom_serializers[format_name] = serializer_class
        self._provider.register_serializer(format_name, serializer_class)
        logger.info(f"Registered custom serializer for format: {format_name}")

    def get_format_extensions(self, format_name: str) -> set[str]:
        """Get file extensions associated with a format.

        Args:
            format_name: Name of the format

        Returns:
            Set of file extensions (including the dot)
        """
        try:
            fmt = SupportedFormat.from_string(format_name)
        except ValueError:
            return set()

        extensions = set()
        for ext, formats in self.EXTENSION_MAP.items():
            if fmt in formats:
                extensions.add(ext)

        return extensions

    def suggest_output_extension(self, format_name: str) -> str:
        """Suggest an appropriate file extension for a format.

        Args:
            format_name: Name of the format

        Returns:
            Suggested file extension (including the dot)
        """
        try:
            fmt = SupportedFormat.from_string(format_name)
        except ValueError:
            return ".txt"

        # Format-specific extension suggestions
        extension_map = {
            SupportedFormat.LEXICAL: ".lexical.json",
            SupportedFormat.DOCLING: ".docling.json",
            SupportedFormat.MARKDOWN: ".md",
            SupportedFormat.MD: ".md",
            SupportedFormat.HTML: ".html",
            SupportedFormat.DOCTAGS: ".doctags.json"
        }

        return extension_map.get(fmt, ".txt")

    def validate_format_compatibility(self, input_format: str, output_format: str) -> bool:
        """Check if conversion between two formats is supported.

        Args:
            input_format: Source format
            output_format: Target format

        Returns:
            True if conversion is supported
        """
        # All supported formats should be inter-convertible through DoclingDocument
        return (self.is_format_supported(input_format) and
                self.is_format_supported(output_format))
