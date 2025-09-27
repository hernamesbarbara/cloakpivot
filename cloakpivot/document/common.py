"""Common utilities for document processing modules.

This module consolidates shared functionality between mapper.py, processor.py,
and extractor.py to reduce code duplication and improve maintainability.
"""

import logging
import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from docling_core.types.doc.document import NodeItem

    from .extractor import TextSegment

logger = logging.getLogger(__name__)


class DocumentValidator:
    """Common validation utilities for document processing."""

    @staticmethod
    def validate_offsets(start_offset: int, end_offset: int, text: str | None = None) -> None:
        """Validate that offsets are valid.

        Args:
            start_offset: Starting offset
            end_offset: Ending offset
            text: Optional text to validate length against

        Raises:
            ValueError: If offsets are invalid
        """
        if end_offset <= start_offset:
            raise ValueError("end_offset must be greater than start_offset")

        if text is not None and len(text) != (end_offset - start_offset):
            raise ValueError("text length must match offset difference")

    @staticmethod
    def validate_node_id(node_id: str) -> None:
        """Validate that a node ID is valid.

        Args:
            node_id: Node ID to validate

        Raises:
            ValueError: If node ID is invalid
        """
        if not node_id or not node_id.strip():
            raise ValueError("node_id cannot be empty")


class SegmentFinder:
    """Utilities for finding and working with text segments."""

    @staticmethod
    def find_segment_containing_offset(
        segments: list["TextSegment"], offset: int
    ) -> "TextSegment | None":
        """Find the text segment that contains the given offset.

        Args:
            segments: List of text segments to search
            offset: Character offset to find

        Returns:
            The segment containing the offset, or None
        """
        for segment in segments:
            if segment.contains_offset(offset):
                return segment
        return None

    @staticmethod
    def find_segment_for_global_position(
        segments: list["TextSegment"], global_position: int
    ) -> "TextSegment | None":
        """Find the text segment that contains a global position.

        Args:
            segments: List of text segments
            global_position: Global character position

        Returns:
            The segment containing the position, or None
        """
        for segment in segments:
            if segment.contains_offset(global_position):
                return segment
        return None


class NodeIdGenerator:
    """Utilities for generating and working with node IDs."""

    @staticmethod
    def get_node_id(node_item: "NodeItem") -> str:
        """Get or generate a stable node ID for the given node item.

        Args:
            node_item: The node item to get an ID for

        Returns:
            A stable node identifier
        """
        # Try to use existing self_ref
        if hasattr(node_item, "self_ref") and node_item.self_ref:
            return str(node_item.self_ref)

        # Generate a deterministic ID based on node properties
        node_type = type(node_item).__name__

        # Try to use text content for ID generation
        if hasattr(node_item, "text") and node_item.text:
            text_hash = hash(node_item.text[:50])  # Use first 50 chars for hash
            return f"#{node_type.lower()}_{abs(text_hash)}"

        # Fall back to object id (less stable but better than nothing)
        return f"#{node_type.lower()}_{id(node_item)}"

    @staticmethod
    def create_cell_node_id(base_node_id: str, row_idx: int, col_idx: int) -> str:
        """Create a node ID for a table cell.

        Args:
            base_node_id: Base table node ID
            row_idx: Row index
            col_idx: Column index

        Returns:
            Cell node ID
        """
        return f"{base_node_id}/cell_{row_idx}_{col_idx}"

    @staticmethod
    def create_kv_node_id(base_node_id: str, part: str) -> str:
        """Create a node ID for a key-value part.

        Args:
            base_node_id: Base key-value node ID
            part: Part name ('key' or 'value')

        Returns:
            Part node ID
        """
        return f"{base_node_id}/{part}"


class TextNormalizer:
    """Utilities for text normalization."""

    @staticmethod
    def normalize_whitespace(text: str, conservative: bool = True) -> str:
        """Normalize whitespace in text.

        Args:
            text: Input text to normalize
            conservative: If True, preserve round-trip fidelity

        Returns:
            Text with normalized whitespace
        """
        if conservative:
            # Conservative normalization for round-trip fidelity
            # Convert Windows line endings to Unix
            text = re.sub(r"\r\n", "\n", text)

            # Only collapse excessive consecutive spaces (3 or more)
            text = re.sub(r"   +", "  ", text)  # Reduce 3+ spaces to 2 spaces

            # Only collapse excessive line breaks (4 or more)
            return re.sub(r"\n\n\n\n+", "\n\n\n", text)  # Reduce 4+ newlines to 3
        # Aggressive normalization
        # Normalize all whitespace sequences to single space
        text = re.sub(r"\s+", " ", text)

        # Strip leading/trailing whitespace
        return text.strip()


class MetadataExtractor:
    """Utilities for extracting metadata from document items."""

    @staticmethod
    def extract_text_item_metadata(text_item: Any) -> dict[str, Any]:
        """Extract metadata from a text item.

        Args:
            text_item: Text item to extract metadata from

        Returns:
            Dictionary of metadata
        """
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
