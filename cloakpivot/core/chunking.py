"""Document chunking for efficient processing of large documents."""

import logging
import os
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Optional

from cloakpivot.core.types import DoclingDocument

from ..document.extractor import TextExtractor, TextSegment

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChunkedTextSegment(TextSegment):
    """
    Extended TextSegment with chunk boundary information.

    Attributes:
        chunk_id: Unique identifier for the chunk containing this segment
        chunk_start: Start offset of the chunk in the full document
        chunk_end: End offset of the chunk in the full document
        is_chunk_boundary: True if this segment spans across chunk boundaries
        original_segment_id: Reference to the original segment before chunking
    """

    chunk_id: str = ""
    chunk_start: int = 0
    chunk_end: int = 0
    is_chunk_boundary: bool = False
    original_segment_id: Optional[str] = None

    @classmethod
    def from_text_segment(
        cls,
        segment: TextSegment,
        chunk_id: str,
        chunk_start: int,
        chunk_end: int,
        is_chunk_boundary: bool = False,
    ) -> "ChunkedTextSegment":
        """Create ChunkedTextSegment from existing TextSegment."""
        return cls(
            node_id=segment.node_id,
            text=segment.text,
            start_offset=segment.start_offset,
            end_offset=segment.end_offset,
            node_type=segment.node_type,
            metadata=segment.metadata,
            chunk_id=chunk_id,
            chunk_start=chunk_start,
            chunk_end=chunk_end,
            is_chunk_boundary=is_chunk_boundary,
            original_segment_id=segment.node_id,
        )


@dataclass
class ChunkBoundary:
    """Information about chunk boundaries and cross-chunk references."""

    chunk_id: str
    start_offset: int
    end_offset: int
    segments: list[ChunkedTextSegment]
    cross_chunk_segments: list[str]  # Node IDs that span this boundary

    @property
    def size(self) -> int:
        """Get the size of this chunk in characters."""
        return self.end_offset - self.start_offset


class ChunkedDocumentProcessor:
    """
    Processor for breaking large documents into manageable chunks.

    This processor segments documents into configurable chunks while maintaining
    structural boundaries and cross-chunk reference tracking for anchor mapping.
    """

    DEFAULT_CHUNK_SIZE = 100 * 1024  # 100KB default chunk size

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        overlap_size: int = 1024,
        preserve_segment_boundaries: bool = True,
    ) -> None:
        """
        Initialize chunked document processor.

        Args:
            chunk_size: Target size per chunk in characters (None for env default)
            overlap_size: Overlap between chunks to handle boundary cases
            preserve_segment_boundaries: Whether to avoid splitting segments across chunks
        """
        self.chunk_size = chunk_size or self._get_chunk_size_from_env()
        self.overlap_size = overlap_size
        self.preserve_segment_boundaries = preserve_segment_boundaries
        self.text_extractor = TextExtractor()

        logger.info(
            f"ChunkedDocumentProcessor initialized: chunk_size={self.chunk_size}, "
            f"overlap_size={self.overlap_size}, preserve_boundaries={self.preserve_segment_boundaries}"
        )

    def _get_chunk_size_from_env(self) -> int:
        """Get chunk size from environment variable or use default."""
        env_chunk_size = os.environ.get("CLOAKPIVOT_CHUNK_SIZE")
        if env_chunk_size:
            try:
                chunk_size = int(env_chunk_size)
                if chunk_size <= 0:
                    logger.warning(
                        f"Invalid CLOAKPIVOT_CHUNK_SIZE: {chunk_size}, using default"
                    )
                    return self.DEFAULT_CHUNK_SIZE
                return chunk_size
            except ValueError:
                logger.warning(
                    f"Invalid CLOAKPIVOT_CHUNK_SIZE format: {env_chunk_size}, using default"
                )

        return self.DEFAULT_CHUNK_SIZE

    def chunk_document(self, document: DoclingDocument) -> list[ChunkBoundary]:
        """
        Break document into chunks with boundary tracking.

        Args:
            document: DoclingDocument to chunk

        Returns:
            List of ChunkBoundary objects representing the chunks
        """
        logger.info(
            f"Chunking document {document.name} with chunk size {self.chunk_size}"
        )

        # Extract all text segments first
        segments = self.text_extractor.extract_text_segments(document)

        if not segments:
            logger.warning(f"No text segments found in document {document.name}")
            return []

        # Calculate total text length
        total_length = segments[-1].end_offset if segments else 0
        logger.debug(f"Total document text length: {total_length} characters")

        # If document is smaller than chunk size, return single chunk
        if total_length <= self.chunk_size:
            logger.debug("Document fits in single chunk")
            return [self._create_single_chunk(segments)]

        # Create chunks
        chunks = []
        current_offset = 0
        chunk_counter = 0

        while current_offset < total_length:
            chunk_counter += 1
            chunk_id = f"chunk_{chunk_counter:03d}"

            chunk_end = min(current_offset + self.chunk_size, total_length)

            # Find segments that fall within this chunk
            chunk_segments = self._get_segments_in_range(
                segments, current_offset, chunk_end
            )

            # Adjust chunk end to preserve segment boundaries if enabled
            if self.preserve_segment_boundaries and chunk_segments:
                chunk_end = self._adjust_chunk_end_for_boundaries(
                    segments, chunk_end, current_offset + self.chunk_size
                )
                # Re-get segments with adjusted boundary
                chunk_segments = self._get_segments_in_range(
                    segments, current_offset, chunk_end
                )

            # Create chunked segments
            chunked_segments = []
            cross_chunk_segments = []

            for segment in chunk_segments:
                is_boundary = (
                    segment.start_offset < current_offset
                    or segment.end_offset > chunk_end
                )

                if is_boundary:
                    cross_chunk_segments.append(segment.node_id)

                chunked_segment = ChunkedTextSegment.from_text_segment(
                    segment, chunk_id, current_offset, chunk_end, is_boundary
                )
                chunked_segments.append(chunked_segment)

            chunk = ChunkBoundary(
                chunk_id=chunk_id,
                start_offset=current_offset,
                end_offset=chunk_end,
                segments=chunked_segments,
                cross_chunk_segments=cross_chunk_segments,
            )
            chunks.append(chunk)

            logger.debug(
                f"Created {chunk_id}: offset {current_offset}-{chunk_end} "
                f"({chunk.size} chars, {len(chunked_segments)} segments)"
            )

            # Move to next chunk with overlap
            current_offset = max(chunk_end - self.overlap_size, current_offset + 1)

        logger.info(f"Created {len(chunks)} chunks for document {document.name}")
        return chunks

    def _create_single_chunk(self, segments: list[TextSegment]) -> ChunkBoundary:
        """Create a single chunk containing all segments."""
        total_length = segments[-1].end_offset if segments else 0

        chunked_segments = [
            ChunkedTextSegment.from_text_segment(
                segment, "chunk_001", 0, total_length, False
            )
            for segment in segments
        ]

        return ChunkBoundary(
            chunk_id="chunk_001",
            start_offset=0,
            end_offset=total_length,
            segments=chunked_segments,
            cross_chunk_segments=[],
        )

    def _get_segments_in_range(
        self, segments: list[TextSegment], start: int, end: int
    ) -> list[TextSegment]:
        """Get all segments that intersect with the given range."""
        result = []
        for segment in segments:
            # Check if segment intersects with chunk range
            if segment.start_offset < end and segment.end_offset > start:
                result.append(segment)
        return result

    def _adjust_chunk_end_for_boundaries(
        self, segments: list[TextSegment], target_end: int, max_end: int
    ) -> int:
        """Adjust chunk end to avoid splitting segments when possible."""
        # Find the segment that would be split by target_end
        for segment in segments:
            if segment.start_offset <= target_end < segment.end_offset:
                # Try to include the entire segment if it fits within reasonable bounds
                if segment.end_offset <= max_end + self.overlap_size:
                    return segment.end_offset
                # Otherwise, end before this segment starts
                else:
                    return segment.start_offset

        # No problematic segment found, use target end
        return target_end

    def get_chunk_statistics(self, chunks: list[ChunkBoundary]) -> dict[str, Any]:
        """Get statistics about the chunking operation."""
        if not chunks:
            return {"total_chunks": 0, "total_size": 0}

        total_size = sum(chunk.size for chunk in chunks)
        total_segments = sum(len(chunk.segments) for chunk in chunks)
        total_cross_chunk = sum(len(chunk.cross_chunk_segments) for chunk in chunks)

        chunk_sizes = [chunk.size for chunk in chunks]

        return {
            "total_chunks": len(chunks),
            "total_size": total_size,
            "total_segments": total_segments,
            "cross_chunk_segments": total_cross_chunk,
            "average_chunk_size": total_size // len(chunks),
            "min_chunk_size": min(chunk_sizes),
            "max_chunk_size": max(chunk_sizes),
            "target_chunk_size": self.chunk_size,
        }

    def iterate_chunks(self, document: DoclingDocument) -> Iterator[ChunkBoundary]:
        """
        Iterate through document chunks for streaming processing.

        Args:
            document: DoclingDocument to process

        Yields:
            ChunkBoundary: Each chunk in sequence
        """
        chunks = self.chunk_document(document)
        yield from chunks

    def extract_chunk_text(self, chunk: ChunkBoundary) -> str:
        """
        Extract the full text content for a chunk.

        Args:
            chunk: ChunkBoundary to extract text from

        Returns:
            Combined text content of all segments in the chunk
        """
        chunk_texts = []
        for segment in chunk.segments:
            chunk_texts.append(segment.text)

        return "\n\n".join(chunk_texts)
