"""Text processing utilities for the PresidioMaskingAdapter.

This module contains text manipulation and processing logic extracted from
the main PresidioMaskingAdapter to improve maintainability.
"""

import bisect
import logging
from typing import Any

from cloakpivot.document.extractor import TextSegment
from cloakpivot.masking.protocols import SegmentBoundary

logger = logging.getLogger(__name__)

# Constants
SEGMENT_SEPARATOR = "\n\n"


class TextProcessor:
    """Process text operations for masking."""

    def __init__(self) -> None:
        """Initialize the text processor."""
        self._segment_starts: list[int] | None = None

    def build_full_text_and_boundaries(
        self, text_segments: list[TextSegment]
    ) -> tuple[str, list[SegmentBoundary]]:
        """Build the full document text and segment boundaries efficiently.

        Args:
            text_segments: List of text segments from document

        Returns:
            Tuple of (full_text, segment_boundaries)
        """
        if not text_segments:
            return "", []

        # Cache segment starts for later binary search
        self._segment_starts = [s.start_offset for s in text_segments]

        # Pre-allocate lists for better performance
        segment_boundaries: list[SegmentBoundary] = []
        text_parts: list[str] = []

        # Build boundaries and collect text parts in single pass
        cursor = 0
        for i, segment in enumerate(text_segments):
            start = cursor
            end = start + len(segment.text)

            segment_boundaries.append(
                SegmentBoundary(
                    segment_index=i,
                    start=start,
                    end=end,
                    node_id=segment.node_id,
                )
            )

            text_parts.append(segment.text)
            cursor = end

            if i < len(text_segments) - 1:
                text_parts.append(SEGMENT_SEPARATOR)
                cursor += len(SEGMENT_SEPARATOR)

        # Join all parts at once - much faster than repeated concatenation
        document_text = "".join(text_parts)
        return document_text, segment_boundaries

    def apply_spans(self, text: str, spans: list[tuple[int, int, str]]) -> str:
        """Apply replacement spans to text efficiently using O(n) algorithm.

        This method applies multiple replacements to a text string efficiently
        using a single pass through the text with list building instead of
        repeated string slicing which is O(n²).

        Args:
            text: Original text to modify
            spans: List of (start, end, replacement) tuples to apply

        Returns:
            Text with all spans applied
        """
        if not spans:
            return text

        # Sort spans by start position for sequential processing
        sorted_spans = sorted(spans, key=lambda x: x[0])

        # Build result in O(n) time using list concatenation
        result = []
        cursor = 0

        for start, end, replacement in sorted_spans:
            # Validate bounds
            if start < 0 or end > len(text) or start >= end:
                logger.warning(f"Invalid span [{start}:{end}] for text of length {len(text)}")
                continue

            # Add text between last replacement and this one
            if cursor < start:
                result.append(text[cursor:start])

            # Add the replacement
            result.append(replacement)
            cursor = end

        # Add any remaining text after last replacement
        if cursor < len(text):
            result.append(text[cursor:])

        return "".join(result)

    def apply_masks_to_text(
        self,
        text: str,
        operator_results: list[Any],  # OperatorResultLike
        entities: list[Any],  # RecognizerResult
    ) -> str:
        """Apply masks to the text based on operator results.

        Args:
            text: Original text
            operator_results: Results from Presidio operators
            entities: Original entities that were detected

        Returns:
            Text with masks applied
        """
        # Build replacement spans from operator results
        spans = []

        # Map operator results by position for matching
        op_results_by_pos = {(r.start, r.end): r for r in operator_results}

        for entity in entities:
            key = (entity.start, entity.end)
            op_result = op_results_by_pos.get(key)

            if op_result:
                spans.append((op_result.start, op_result.end, op_result.text))
            else:
                # Fallback if no matching result
                logger.warning(f"No operator result for entity at {entity.start}-{entity.end}")

        # Apply all spans efficiently
        return self.apply_spans(text, spans)

    def find_segment_for_position(self, position: int, segments: list[TextSegment]) -> str | None:
        """Find the segment containing a given position.

        Uses binary search for efficiency with large documents.

        Args:
            position: Character position in the full text
            segments: List of text segments

        Returns:
            Node ID of the containing segment, or None
        """
        if not segments:
            return None

        if self._segment_starts is None:
            self._segment_starts = [s.start_offset for s in segments]

        # Use binary search to find the segment
        idx = bisect.bisect_right(self._segment_starts, position) - 1

        if 0 <= idx < len(segments):
            segment = segments[idx]
            # Verify position is actually within this segment
            if segment.start_offset <= position < segment.end_offset:
                return segment.node_id

        # Fallback: linear search if binary search fails
        for segment in segments:
            if segment.start_offset <= position < segment.end_offset:
                return segment.node_id

        # Default to first segment if position is before all segments
        if position < segments[0].start_offset:
            return segments[0].node_id

        # Default to last segment if position is after all segments
        if position >= segments[-1].end_offset:
            return segments[-1].node_id

        return None

    def extract_segment_entities(
        self,
        segment: TextSegment,
        anchor_entries: list[Any],  # AnchorEntry
        segment_text: str,
    ) -> list[dict[str, Any]]:
        """Extract entities that affect a specific segment.

        Args:
            segment: The text segment
            anchor_entries: All anchor entries
            segment_text: The segment's text

        Returns:
            List of entity info dictionaries for this segment
        """
        segment_entities: list[dict[str, Any]] = []

        for anchor in anchor_entries:
            # Check if anchor overlaps with this segment
            if (
                anchor.metadata
                and "original_text" in anchor.metadata
                and segment.start_offset <= anchor.start < segment.end_offset
            ):
                local_start = anchor.start - segment.start_offset
                local_end = min(
                    local_start + len(anchor.metadata["original_text"]),
                    len(segment_text),
                )
                if local_start < local_end:
                    segment_entities.append(
                        {
                            "anchor": anchor,
                            "local_start": local_start,
                            "local_end": local_end,
                        }
                    )

        return segment_entities
