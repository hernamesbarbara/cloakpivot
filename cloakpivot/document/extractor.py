"""Text extraction engine for DoclingDocument objects."""

import logging
from dataclasses import dataclass
from typing import Any, Optional, Union

from docling_core.types.doc.document import (
    CodeItem,
    FormulaItem,
    KeyValueItem,
    ListItem,
    NodeItem,
    SectionHeaderItem,
    TableItem,
    TextItem,
    TitleItem,
)
from packaging import version

from cloakpivot.core.types import DoclingDocument

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TextSegment:
    """
    Represents a segment of extracted text with its structural context.

    This class maintains the mapping between plain text and the original
    document structure, enabling round-trip operations and anchor mapping.

    Attributes:
        node_id: Unique identifier for the source document node
        text: The extracted plain text content
        start_offset: Starting character position in the full extracted text
        end_offset: Ending character position in the full extracted text
        node_type: Type of the source node (e.g., 'TextItem', 'TitleItem')
        metadata: Additional context information about the segment

    Examples:
        >>> segment = TextSegment(
        ...     node_id="#/texts/0",
        ...     text="Hello world",
        ...     start_offset=0,
        ...     end_offset=11,
        ...     node_type="TextItem"
        ... )
        >>> print(f"Text: '{segment.text}' from {segment.node_type}")
    """

    node_id: str
    text: str
    start_offset: int
    end_offset: int
    node_type: str
    metadata: Optional[dict[str, Any]] = None

    def __post_init__(self) -> None:
        """Validate segment data after initialization."""
        if self.end_offset <= self.start_offset:
            raise ValueError("end_offset must be greater than start_offset")

        if len(self.text) != (self.end_offset - self.start_offset):
            raise ValueError("text length must match offset difference")

        if not self.node_id.strip():
            raise ValueError("node_id cannot be empty")

    @property
    def length(self) -> int:
        """Get the length of the text segment."""
        return self.end_offset - self.start_offset

    def contains_offset(self, offset: int) -> bool:
        """Check if an offset falls within this segment."""
        return self.start_offset <= offset < self.end_offset

    def relative_offset(self, global_offset: int) -> int:
        """Convert a global offset to a relative offset within this segment."""
        if not self.contains_offset(global_offset):
            raise ValueError(
                f"Offset {global_offset} not in segment range [{self.start_offset}, {self.end_offset})"
            )
        return global_offset - self.start_offset


class TextExtractor:
    """
    Text extraction engine that traverses DoclingDocument structure.

    This class implements document traversal to find text-bearing nodes,
    extract text content while preserving structural boundaries, and
    generate text segments suitable for PII analysis.

    The extractor maintains the mapping between extracted plain text and
    the original document structure to enable round-trip operations.

    Examples:
        >>> extractor = TextExtractor()
        >>> segments = extractor.extract_text_segments(document)
        >>> full_text = extractor.extract_full_text(document)
        >>> print(f"Extracted {len(segments)} segments, {len(full_text)} characters")
    """

    def __init__(self, normalize_whitespace: bool = False) -> None:
        """
        Initialize the text extractor.

        Args:
            normalize_whitespace: Whether to normalize whitespace in extracted text
                                 Default is False to preserve round-trip fidelity
        """
        self.normalize_whitespace = normalize_whitespace
        self._segment_separator = "\n\n"  # Separator between text segments
        logger.debug(
            f"TextExtractor initialized with normalize_whitespace={normalize_whitespace}"
        )

    def extract_text_segments(self, document: DoclingDocument) -> list[TextSegment]:
        """
        Extract text segments from a DoclingDocument with structural anchors.

        This method traverses the document structure and extracts all text-bearing
        content as individual segments, maintaining the mapping to original nodes.

        Note: DoclingDocument v1.7.0+ uses segment-local charspans in the prov field,
        but this extractor builds its own segments with global offsets for consistent
        masking operations across all document versions.

        Args:
            document: The DoclingDocument to extract text from

        Returns:
            List[TextSegment]: List of text segments with structural anchors

        Examples:
            >>> extractor = TextExtractor()
            >>> segments = extractor.extract_text_segments(doc)
            >>> for segment in segments:
            ...     print(f"{segment.node_type}: {segment.text[:50]}...")
        """
        # Check document version for logging purposes
        doc_version = getattr(document, 'version', '1.2.0')
        logger.info(f"Extracting text segments from document: {document.name} (version: {doc_version})")

        # Log version-specific information
        if version.parse(str(doc_version)) >= version.parse('1.7.0'):
            logger.debug(
                "Document is v1.7.0+: prov charspans are segment-local. "
                "TextExtractor builds independent segments with global offsets."
            )

        segments: list[TextSegment] = []
        current_offset = 0

        # Extract from text items (paragraphs, headings, etc.)
        for text_item in document.texts:
            segment = self._extract_from_text_item(text_item, current_offset)
            if segment:
                segments.append(segment)
                current_offset = segment.end_offset + len(self._segment_separator)

        # Extract from tables
        for table_item in document.tables:
            table_segments = self._extract_from_table_item(table_item, current_offset)
            segments.extend(table_segments)
            if table_segments:
                current_offset = table_segments[-1].end_offset + len(
                    self._segment_separator
                )

        # Extract from key-value items
        for kv_item in document.key_value_items:
            kv_segments = self._extract_from_key_value_item(kv_item, current_offset)
            segments.extend(kv_segments)
            if kv_segments:
                current_offset = kv_segments[-1].end_offset + len(
                    self._segment_separator
                )

        logger.info(
            f"Extracted {len(segments)} text segments, {current_offset} total characters"
        )
        return segments

    def extract_full_text(self, document: DoclingDocument) -> str:
        """
        Extract all text content as a single string.

        Args:
            document: The DoclingDocument to extract text from

        Returns:
            str: The complete extracted text with segment separators
        """
        segments = self.extract_text_segments(document)
        return self._segment_separator.join(segment.text for segment in segments)

    def find_segment_containing_offset(
        self, segments: list[TextSegment], offset: int
    ) -> Optional[TextSegment]:
        """
        Find the text segment that contains the given global offset.

        Args:
            segments: List of text segments to search
            offset: Global character offset to find

        Returns:
            Optional[TextSegment]: The segment containing the offset, or None
        """
        for segment in segments:
            if segment.contains_offset(offset):
                return segment
        return None

    def get_extraction_stats(self, document: DoclingDocument) -> dict[str, Any]:
        """
        Get statistics about extractable content in the document.

        Args:
            document: The DoclingDocument to analyze

        Returns:
            Dict[str, Any]: Statistics about the document content
        """
        stats: dict[str, Any] = {
            "total_text_items": len(document.texts),
            "total_tables": len(document.tables),
            "total_key_value_items": len(document.key_value_items),
            "total_pictures": len(document.pictures),
            "total_form_items": len(document.form_items),
        }

        # Analyze text item types
        text_types: dict[str, int] = {}
        for text_item in document.texts:
            item_type = type(text_item).__name__
            text_types[item_type] = text_types.get(item_type, 0) + 1
        stats["text_item_types"] = text_types

        # Estimate extractable text length
        segments = self.extract_text_segments(document)
        stats["extractable_segments"] = len(segments)
        stats["total_extractable_chars"] = sum(len(seg.text) for seg in segments)

        return stats

    def _extract_from_text_item(
        self,
        text_item: Union[
            TextItem, TitleItem, SectionHeaderItem, ListItem, CodeItem, FormulaItem
        ],
        start_offset: int,
    ) -> Optional[TextSegment]:
        """Extract text from a text-bearing item."""
        if not hasattr(text_item, "text") or not text_item.text:
            return None

        text = text_item.text
        if self.normalize_whitespace:
            text = self._normalize_whitespace(text)

        if not text.strip():
            return None

        node_id = self._get_node_id(text_item)
        node_type = type(text_item).__name__
        end_offset = start_offset + len(text)

        metadata = self._extract_text_item_metadata(text_item)

        return TextSegment(
            node_id=node_id,
            text=text,
            start_offset=start_offset,
            end_offset=end_offset,
            node_type=node_type,
            metadata=metadata,
        )

    def _extract_from_table_item(
        self, table_item: TableItem, start_offset: int
    ) -> list[TextSegment]:
        """Extract text from a table item, creating segments for each cell."""
        segments: list[TextSegment] = []
        current_offset = start_offset

        if not hasattr(table_item, "data") or not table_item.data:
            return segments

        base_node_id = self._get_node_id(table_item)

        # Extract text from table cells
        table_data = table_item.data
        if hasattr(table_data, "table_cells"):
            for row_idx, row in enumerate(table_data.table_cells):
                for col_idx, cell in enumerate(row):
                    # mypy thinks cell is a tuple, but it's actually the cell object
                    if hasattr(cell, "text") and cell.text:  # type: ignore
                        text = cell.text  # type: ignore
                        if self.normalize_whitespace:
                            text = self._normalize_whitespace(text)

                        if text.strip():
                            cell_node_id = f"{base_node_id}/cell_{row_idx}_{col_idx}"
                            end_offset = current_offset + len(text)

                            segment = TextSegment(
                                node_id=cell_node_id,
                                text=text,
                                start_offset=current_offset,
                                end_offset=end_offset,
                                node_type="TableCell",
                                metadata={
                                    "table_node_id": base_node_id,
                                    "row": row_idx,
                                    "column": col_idx,
                                },
                            )
                            segments.append(segment)
                            current_offset = end_offset + len(self._segment_separator)

        return segments

    def _extract_from_key_value_item(
        self, kv_item: KeyValueItem, start_offset: int
    ) -> list[TextSegment]:
        """Extract text from a key-value item."""
        segments: list[TextSegment] = []
        current_offset = start_offset
        base_node_id = self._get_node_id(kv_item)

        # Extract key text
        if hasattr(kv_item, "key") and kv_item.key and hasattr(kv_item.key, "text"):
            key_text = kv_item.key.text
            if self.normalize_whitespace:
                key_text = self._normalize_whitespace(key_text)

            if key_text.strip():
                key_segment = TextSegment(
                    node_id=f"{base_node_id}/key",
                    text=key_text,
                    start_offset=current_offset,
                    end_offset=current_offset + len(key_text),
                    node_type="KeyValueKey",
                    metadata={"kv_node_id": base_node_id, "part": "key"},
                )
                segments.append(key_segment)
                current_offset = key_segment.end_offset + len(self._segment_separator)

        # Extract value text
        if (
            hasattr(kv_item, "value")
            and kv_item.value
            and hasattr(kv_item.value, "text")
        ):
            value_text = kv_item.value.text
            if self.normalize_whitespace:
                value_text = self._normalize_whitespace(value_text)

            if value_text.strip():
                value_segment = TextSegment(
                    node_id=f"{base_node_id}/value",
                    text=value_text,
                    start_offset=current_offset,
                    end_offset=current_offset + len(value_text),
                    node_type="KeyValueValue",
                    metadata={"kv_node_id": base_node_id, "part": "value"},
                )
                segments.append(value_segment)

        return segments

    def _get_node_id(self, node_item: NodeItem) -> str:
        """
        Get or generate a stable node ID for the given node item.

        Args:
            node_item: The node item to get an ID for

        Returns:
            str: A stable node identifier
        """
        # Try to use existing self_ref
        if hasattr(node_item, "self_ref") and node_item.self_ref:
            return node_item.self_ref

        # Generate a deterministic ID based on node properties
        node_type = type(node_item).__name__

        # Try to use text content for ID generation
        if hasattr(node_item, "text") and node_item.text:
            text_hash = hash(node_item.text[:50])  # Use first 50 chars for hash
            return f"#{node_type.lower()}_{abs(text_hash)}"

        # Fall back to object id (less stable but better than nothing)
        return f"#{node_type.lower()}_{id(node_item)}"

    def _extract_text_item_metadata(
        self,
        text_item: Union[
            TextItem, TitleItem, SectionHeaderItem, ListItem, CodeItem, FormulaItem
        ],
    ) -> dict[str, Any]:
        """Extract metadata from a text item."""
        metadata = {"item_type": type(text_item).__name__}

        # Add level for headings
        if hasattr(text_item, "level"):
            metadata["level"] = text_item.level

        # Add formatting information
        if hasattr(text_item, "formatting") and text_item.formatting:
            metadata["formatting"] = str(text_item.formatting)

        # Add language for code items
        if hasattr(text_item, "code_language") and text_item.code_language:
            metadata["code_language"] = str(text_item.code_language)

        return metadata

    def _normalize_whitespace(self, text: str) -> str:
        """
        Normalize whitespace in text while preserving essential formatting.

        This is a conservative normalization that preserves round-trip fidelity
        while still handling some common whitespace issues.

        Args:
            text: Input text to normalize

        Returns:
            str: Text with conservatively normalized whitespace
        """
        import re

        # Only do minimal normalization to preserve round-trip fidelity
        # Convert Windows line endings to Unix, but preserve standalone \r
        text = re.sub(r"\r\n", "\n", text)

        # Only collapse excessive consecutive spaces (3 or more), not all multiple spaces
        text = re.sub(r"   +", "  ", text)  # Reduce 3+ spaces to 2 spaces

        # Only collapse excessive line breaks (4 or more), preserve structure
        text = re.sub(r"\n\n\n\n+", "\n\n\n", text)  # Reduce 4+ newlines to 3

        # Do NOT strip leading/trailing whitespace as it may be significant
        return text
