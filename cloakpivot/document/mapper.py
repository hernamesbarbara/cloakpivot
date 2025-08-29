"""Anchor mapping system for linking text positions to document structure."""

import logging
from dataclasses import dataclass
from typing import Optional

from presidio_analyzer import RecognizerResult

from ..core.anchors import AnchorEntry
from .extractor import TextSegment

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class NodeReference:
    """
    Represents a stable reference to a position in the document structure.

    This class provides a stable way to reference positions within document
    nodes that survives serialization/deserialization cycles and document
    modifications.

    Attributes:
        node_id: Unique identifier for the document node
        start_pos: Starting character position within the node's text
        end_pos: Ending character position within the node's text
        global_start: Starting position in the full extracted text
        global_end: Ending position in the full extracted text
        segment_index: Index of the text segment containing this reference

    Examples:
        >>> ref = NodeReference(
        ...     node_id="#/texts/0",
        ...     start_pos=10,
        ...     end_pos=20,
        ...     global_start=45,
        ...     global_end=55,
        ...     segment_index=0
        ... )
        >>> print(f"Reference to {ref.node_id}[{ref.start_pos}:{ref.end_pos}]")
    """

    node_id: str
    start_pos: int
    end_pos: int
    global_start: int
    global_end: int
    segment_index: int

    def __post_init__(self) -> None:
        """Validate reference data after initialization."""
        if self.end_pos <= self.start_pos:
            raise ValueError("end_pos must be greater than start_pos")

        if self.global_end <= self.global_start:
            raise ValueError("global_end must be greater than global_start")

        if self.segment_index < 0:
            raise ValueError("segment_index must be non-negative")

    @property
    def local_length(self) -> int:
        """Get the length within the node."""
        return self.end_pos - self.start_pos

    @property
    def global_length(self) -> int:
        """Get the length in the global text."""
        return self.global_end - self.global_start


class AnchorMapper:
    """
    Anchor mapping system for linking text positions to document structure.

    This class manages the complex mapping between character positions in
    extracted plain text and their corresponding positions in the original
    DoclingDocument structure. It provides stable position references that
    enable round-trip operations for masking and unmasking.

    The mapper works with TextSegment objects to maintain the relationship
    between extracted text and document nodes, and creates AnchorEntry
    objects for detected PII entities.

    Examples:
        >>> mapper = AnchorMapper()
        >>> anchors = mapper.create_anchors_from_detections(
        ...     detections, segments, original_texts
        ... )
        >>> print(f"Created {len(anchors)} anchor entries")
    """

    def __init__(self) -> None:
        """Initialize the anchor mapper."""
        self._logger = logger
        logger.debug("AnchorMapper initialized")

    def create_anchors_from_detections(
        self,
        detections: list[RecognizerResult],
        segments: list[TextSegment],
        original_texts: dict[str, str],
        strategy_used: str = "redact",
    ) -> list[AnchorEntry]:
        """
        Create anchor entries from Presidio detection results.

        This method maps Presidio's RecognizerResult objects to CloakPivot's
        AnchorEntry objects, maintaining the connection between detected entities
        and their positions in the document structure.

        Args:
            detections: List of RecognizerResult objects from Presidio
            segments: List of TextSegment objects from text extraction
            original_texts: Mapping of node_id to original text content
            strategy_used: The masking strategy that will be applied

        Returns:
            List[AnchorEntry]: List of anchor entries for the detected entities

        Examples:
            >>> detections = analyzer.analyze(text, entities=["PHONE_NUMBER"])
            >>> anchors = mapper.create_anchors_from_detections(
            ...     detections, segments, {"#/texts/0": "Call me at 555-1234"}
            ... )
        """
        logger.info(
            f"Creating anchors from {len(detections)} detections across {len(segments)} segments"
        )

        anchors = []

        for detection in detections:
            # Find which segment contains this detection
            segment = self._find_segment_for_global_position(segments, detection.start)
            if not segment:
                logger.warning(
                    f"No segment found for detection at global position {detection.start}"
                )
                continue

            # Create node reference
            node_ref = self._create_node_reference(detection, segment, segments)
            if not node_ref:
                logger.warning(
                    f"Could not create node reference for detection {detection}"
                )
                continue

            # Get original text for this detection
            original_text = self._extract_original_text(
                node_ref, original_texts, segments
            )
            if not original_text:
                logger.warning(
                    f"Could not extract original text for detection {detection}"
                )
                continue

            # Create anchor entry
            anchor = AnchorEntry.create_from_detection(
                node_id=node_ref.node_id,
                start=node_ref.start_pos,
                end=node_ref.end_pos,
                entity_type=detection.entity_type,
                confidence=detection.score,
                original_text=original_text,
                masked_value=f"[{detection.entity_type}]",  # Default template
                strategy_used=strategy_used,
                metadata={
                    "global_start": node_ref.global_start,
                    "global_end": node_ref.global_end,
                    "segment_index": node_ref.segment_index,
                    "presidio_analysis_explanation": (
                        detection.analysis_explanation.__dict__
                        if detection.analysis_explanation
                        else None
                    ),
                },
            )

            anchors.append(anchor)
            logger.debug(
                f"Created anchor for {detection.entity_type} at {node_ref.node_id}[{node_ref.start_pos}:{node_ref.end_pos}]"
            )

        logger.info(f"Successfully created {len(anchors)} anchor entries")
        return anchors

    def map_global_to_node_position(
        self, global_start: int, global_end: int, segments: list[TextSegment]
    ) -> Optional[NodeReference]:
        """
        Map global text positions to node-specific positions.

        Args:
            global_start: Starting position in the full extracted text
            global_end: Ending position in the full extracted text
            segments: List of text segments

        Returns:
            Optional[NodeReference]: Node reference or None if mapping fails
        """
        # Find the segment containing the start position
        containing_segment = self._find_segment_for_global_position(
            segments, global_start
        )
        if not containing_segment:
            return None

        # Check if the entire span is within this segment
        if global_end > containing_segment.end_offset:
            logger.warning(
                f"Position span [{global_start}, {global_end}] crosses segment boundaries"
            )
            return None

        # Convert to local positions within the segment
        local_start = containing_segment.relative_offset(global_start)
        local_end = (
            containing_segment.relative_offset(global_end - 1) + 1
        )  # Inclusive end

        # Find segment index
        segment_index = next(
            (i for i, seg in enumerate(segments) if seg == containing_segment), -1
        )

        return NodeReference(
            node_id=containing_segment.node_id,
            start_pos=local_start,
            end_pos=local_end,
            global_start=global_start,
            global_end=global_end,
            segment_index=segment_index,
        )

    def map_node_to_global_position(
        self, node_id: str, start_pos: int, end_pos: int, segments: list[TextSegment]
    ) -> Optional[tuple[int, int]]:
        """
        Map node-specific positions to global text positions.

        Args:
            node_id: The node identifier
            start_pos: Starting position within the node
            end_pos: Ending position within the node
            segments: List of text segments

        Returns:
            Optional[Tuple[int, int]]: Global (start, end) positions or None
        """
        # Find the segment with matching node_id
        matching_segment = next(
            (seg for seg in segments if seg.node_id == node_id), None
        )

        if not matching_segment:
            logger.warning(f"No segment found for node_id: {node_id}")
            return None

        # Validate positions are within segment
        if start_pos < 0 or end_pos > len(matching_segment.text):
            logger.warning(
                f"Positions [{start_pos}, {end_pos}] outside segment bounds for {node_id}"
            )
            return None

        # Convert to global positions
        global_start = matching_segment.start_offset + start_pos
        global_end = matching_segment.start_offset + end_pos

        return (global_start, global_end)

    def resolve_anchor_conflicts(self, anchors: list[AnchorEntry]) -> list[AnchorEntry]:
        """
        Resolve conflicts between overlapping anchor entries.

        This method handles cases where multiple PII detections overlap
        in the same text region by applying resolution rules to determine
        which anchor should take precedence.

        Args:
            anchors: List of potentially conflicting anchor entries

        Returns:
            List[AnchorEntry]: List of non-conflicting anchor entries
        """
        if len(anchors) <= 1:
            return anchors

        logger.info(f"Resolving conflicts among {len(anchors)} anchors")

        # Group anchors by node_id for conflict detection
        by_node: dict[str, list[AnchorEntry]] = {}
        for anchor in anchors:
            if anchor.node_id not in by_node:
                by_node[anchor.node_id] = []
            by_node[anchor.node_id].append(anchor)

        resolved_anchors = []

        for _node_id, node_anchors in by_node.items():
            if len(node_anchors) == 1:
                resolved_anchors.extend(node_anchors)
                continue

            # Sort by start position
            node_anchors.sort(key=lambda a: a.start)

            # Resolve overlaps within this node
            non_overlapping: list[AnchorEntry] = []
            for anchor in node_anchors:
                # Check for overlaps with already accepted anchors
                overlaps = any(
                    existing.overlaps_with(anchor) for existing in non_overlapping
                )

                if not overlaps:
                    non_overlapping.append(anchor)
                else:
                    # Apply conflict resolution rules
                    resolved_anchor = self._resolve_overlap_conflict(
                        anchor, non_overlapping
                    )
                    if resolved_anchor:
                        # Remove conflicting anchors and add the resolved one
                        non_overlapping = [
                            a
                            for a in non_overlapping
                            if not a.overlaps_with(resolved_anchor)
                        ]
                        non_overlapping.append(resolved_anchor)

            resolved_anchors.extend(non_overlapping)

        logger.info(f"Resolved to {len(resolved_anchors)} non-conflicting anchors")
        return resolved_anchors

    def _find_segment_for_global_position(
        self, segments: list[TextSegment], global_position: int
    ) -> Optional[TextSegment]:
        """Find the text segment that contains a global position."""
        for segment in segments:
            if segment.contains_offset(global_position):
                return segment
        return None

    def _create_node_reference(
        self,
        detection: RecognizerResult,
        segment: TextSegment,
        all_segments: list[TextSegment],
    ) -> Optional[NodeReference]:
        """Create a node reference from a detection and its containing segment."""
        try:
            # Convert global positions to local positions within the segment
            local_start = segment.relative_offset(detection.start)
            local_end = segment.relative_offset(detection.end - 1) + 1  # Inclusive end

            # Find segment index
            segment_index = next(
                (i for i, seg in enumerate(all_segments) if seg == segment), -1
            )

            if segment_index == -1:
                logger.error(
                    f"Could not find segment index for segment {segment.node_id}"
                )
                return None

            return NodeReference(
                node_id=segment.node_id,
                start_pos=local_start,
                end_pos=local_end,
                global_start=detection.start,
                global_end=detection.end,
                segment_index=segment_index,
            )

        except (ValueError, IndexError) as e:
            logger.error(f"Error creating node reference: {e}")
            return None

    def _extract_original_text(
        self,
        node_ref: NodeReference,
        original_texts: dict[str, str],
        segments: list[TextSegment],
    ) -> Optional[str]:
        """Extract the original text for a node reference."""
        # Try to get from original_texts mapping first
        if node_ref.node_id in original_texts:
            original_text = original_texts[node_ref.node_id]
            if node_ref.start_pos < len(original_text) and node_ref.end_pos <= len(
                original_text
            ):
                return original_text[node_ref.start_pos : node_ref.end_pos]

        # Fall back to segment text
        if 0 <= node_ref.segment_index < len(segments):
            segment = segments[node_ref.segment_index]
            if segment.node_id == node_ref.node_id:
                segment_text = segment.text
                if node_ref.start_pos < len(segment_text) and node_ref.end_pos <= len(
                    segment_text
                ):
                    return segment_text[node_ref.start_pos : node_ref.end_pos]

        logger.error(f"Could not extract original text for {node_ref}")
        return None

    def _resolve_overlap_conflict(
        self, new_anchor: AnchorEntry, existing_anchors: list[AnchorEntry]
    ) -> Optional[AnchorEntry]:
        """
        Resolve overlap conflicts by choosing the best anchor.

        Resolution rules (in order of priority):
        1. Higher confidence score
        2. Longer text span (more specific detection)
        3. Earlier position in document
        """
        conflicting_anchors = [
            a for a in existing_anchors if a.overlaps_with(new_anchor)
        ]

        if not conflicting_anchors:
            return new_anchor

        # Include the new anchor in the comparison
        all_candidates = conflicting_anchors + [new_anchor]

        # Sort by resolution criteria
        def resolution_key(anchor: AnchorEntry) -> tuple[float, int, int]:
            return (
                -anchor.confidence,  # Higher confidence first (negative for descending)
                -anchor.span_length,  # Longer span first (negative for descending)
                anchor.start,  # Earlier position first
            )

        all_candidates.sort(key=resolution_key)
        best_anchor = all_candidates[0]

        logger.debug(
            f"Resolved conflict: chose {best_anchor.entity_type} "
            f"(confidence={best_anchor.confidence}, length={best_anchor.span_length})"
        )

        return best_anchor
