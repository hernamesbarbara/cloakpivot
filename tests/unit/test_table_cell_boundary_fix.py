"""Test for table cell boundary validation fix.

This test verifies that entities spanning across segment boundaries
(e.g., table cells) are properly handled and don't cause concatenation artifacts.
"""

from presidio_analyzer import RecognizerResult

from cloakpivot.masking.presidio_adapter import PresidioMaskingAdapter
from cloakpivot.masking.protocols import SegmentBoundary


class TestTableCellBoundaryFix:
    """Test suite for table cell boundary validation."""

    def test_entity_spanning_segments_is_truncated(self):
        """Test that entities spanning segment boundaries are truncated."""
        adapter = PresidioMaskingAdapter()

        # Create document text with separator between segments
        document_text = "John Smith was born on 1958-04-21\n\nBorden\n\nAshley was born in 1944"
        # Segments:
        # [0:34] "John Smith was born on 1958-04-21"
        # [36:42] "Borden"
        # [44:79] "Ashley was born in 1944"

        segment_boundaries = [
            SegmentBoundary(segment_index=0, start=0, end=34, node_id="#/texts/0"),
            SegmentBoundary(segment_index=1, start=36, end=42, node_id="#/texts/1"),
            SegmentBoundary(segment_index=2, start=44, end=79, node_id="#/texts/2"),
        ]

        # Create an entity that incorrectly spans across segments
        # This simulates the bug where Presidio detects a date spanning into next cells
        entities = [
            RecognizerResult(
                entity_type="DATE_TIME",
                start=23,  # Start of "1958-04-21"
                end=50,  # Incorrectly extends past separator into "Borden"
                score=0.95,
            )
        ]

        # Apply boundary validation
        validated_entities = adapter._validate_entities_against_boundaries(
            entities, document_text, segment_boundaries
        )

        # The entity should be truncated at the segment boundary
        assert len(validated_entities) == 1
        validated = validated_entities[0]
        assert validated.start == 23
        # Truncated to just before the separator
        assert (
            validated.end == 33 or validated.end == 34
        )  # May be 33 or 34 depending on implementation
        # The important thing is it doesn't extend past the separator
        assert "\n\n" not in document_text[validated.start : validated.end]
        assert "1958-04-21" in document_text[validated.start : validated.end]

    def test_entity_fully_within_segment_unchanged(self):
        """Test that entities fully within a segment are not modified."""
        adapter = PresidioMaskingAdapter()

        document_text = "John Smith\n\nBorden Ashley\n\n1958-04-21"
        segment_boundaries = [
            SegmentBoundary(segment_index=0, start=0, end=10, node_id="#/texts/0"),
            SegmentBoundary(segment_index=1, start=12, end=25, node_id="#/texts/1"),
            SegmentBoundary(segment_index=2, start=27, end=37, node_id="#/texts/2"),
        ]

        # Entity fully within second segment
        entities = [
            RecognizerResult(entity_type="PERSON", start=12, end=25, score=0.95)  # "Borden Ashley"
        ]

        validated_entities = adapter._validate_entities_against_boundaries(
            entities, document_text, segment_boundaries
        )

        # Entity should remain unchanged
        assert len(validated_entities) == 1
        validated = validated_entities[0]
        assert validated.start == 12
        assert validated.end == 25
        assert document_text[validated.start : validated.end] == "Borden Ashley"

    def test_multiple_entities_with_mixed_boundaries(self):
        """Test handling of multiple entities with different boundary conditions."""
        adapter = PresidioMaskingAdapter()

        document_text = "Name: John\n\nDate: 1958-04-21\n\nLocation: NYC"
        segment_boundaries = [
            SegmentBoundary(segment_index=0, start=0, end=10, node_id="#/table/0/cell_0_0"),
            SegmentBoundary(segment_index=1, start=12, end=28, node_id="#/table/0/cell_0_1"),
            SegmentBoundary(segment_index=2, start=30, end=43, node_id="#/table/0/cell_0_2"),
        ]

        entities = [
            # Entity within first segment
            RecognizerResult(entity_type="PERSON", start=6, end=10, score=0.95),
            # Entity within second segment
            RecognizerResult(entity_type="DATE_TIME", start=18, end=28, score=0.95),
            # Entity spanning segments (should be truncated)
            RecognizerResult(
                entity_type="LOCATION", start=40, end=45, score=0.95
            ),  # "NYC" extends beyond
        ]

        validated_entities = adapter._validate_entities_against_boundaries(
            entities, document_text, segment_boundaries
        )

        assert len(validated_entities) == 3

        # First entity unchanged
        assert validated_entities[0].start == 6
        assert validated_entities[0].end == 10

        # Second entity unchanged
        assert validated_entities[1].start == 18
        assert validated_entities[1].end == 28

        # Third entity truncated
        assert validated_entities[2].start == 40
        assert validated_entities[2].end == 43

    def test_entity_with_separator_in_text(self):
        """Test that entities containing the segment separator are properly handled."""
        adapter = PresidioMaskingAdapter()

        # Document text where an entity incorrectly spans the separator
        document_text = "Date is 1958-04-21\n\nNext cell"
        segment_boundaries = [
            SegmentBoundary(segment_index=0, start=0, end=18, node_id="#/texts/0"),
            SegmentBoundary(segment_index=1, start=20, end=29, node_id="#/texts/1"),
        ]

        # Entity that spans across the separator
        entities = [
            RecognizerResult(
                entity_type="DATE_TIME",
                start=8,  # Start of "1958-04-21"
                end=25,  # Incorrectly extends into "Next c"
                score=0.95,
            )
        ]

        validated_entities = adapter._validate_entities_against_boundaries(
            entities, document_text, segment_boundaries
        )

        # Entity should be truncated at the separator
        assert len(validated_entities) == 1
        validated = validated_entities[0]
        assert validated.start == 8
        assert validated.end == 18  # Truncated at segment boundary
        assert document_text[validated.start : validated.end] == "1958-04-21"

    def test_empty_entity_after_truncation_is_skipped(self):
        """Test that entities that become empty after truncation are skipped."""
        adapter = PresidioMaskingAdapter()

        document_text = "Text\n\n   \n\nMore text"
        segment_boundaries = [
            SegmentBoundary(segment_index=0, start=0, end=4, node_id="#/texts/0"),
            SegmentBoundary(segment_index=1, start=6, end=9, node_id="#/texts/1"),
            SegmentBoundary(segment_index=2, start=11, end=20, node_id="#/texts/2"),
        ]

        # Entity that would be empty after truncation (only whitespace)
        entities = [
            RecognizerResult(
                entity_type="MISC",
                start=6,  # Whitespace segment
                end=15,  # Spans into next segment
                score=0.95,
            )
        ]

        validated_entities = adapter._validate_entities_against_boundaries(
            entities, document_text, segment_boundaries
        )

        # Entity should be skipped because truncation results in whitespace only
        assert len(validated_entities) == 0

    def test_empty_segment_boundaries_list(self):
        """Test behavior with empty segment boundaries list."""
        adapter = PresidioMaskingAdapter()

        document_text = "John Smith was born on 1958-04-21"
        segment_boundaries = []  # Empty list

        entities = [
            RecognizerResult(
                entity_type="PERSON",
                start=0,
                end=10,
                score=0.95,
            ),
            RecognizerResult(
                entity_type="DATE_TIME",
                start=23,
                end=34,
                score=0.95,
            ),
        ]

        # When no boundaries, entities are skipped as they cannot be validated
        validated_entities = adapter._validate_entities_against_boundaries(
            entities, document_text, segment_boundaries
        )

        # With empty boundaries, entities cannot be contained in any segment, so they're skipped
        assert len(validated_entities) == 0

    def test_entity_without_separator_passes_through(self):
        """Test that entities without separator in text pass through unchanged."""
        adapter = PresidioMaskingAdapter()

        document_text = "John Smith\n\nBorden Ashley\n\n1958-04-21"
        segment_boundaries = [
            SegmentBoundary(segment_index=0, start=0, end=10, node_id="#/texts/0"),
            SegmentBoundary(segment_index=1, start=12, end=25, node_id="#/texts/1"),
            SegmentBoundary(segment_index=2, start=27, end=37, node_id="#/texts/2"),
        ]

        # Entities that don't span boundaries (no separator in entity text)
        entities = [
            RecognizerResult(entity_type="PERSON", start=0, end=10, score=0.95),  # "John Smith"
            RecognizerResult(entity_type="PERSON", start=12, end=18, score=0.95),  # "Borden"
            RecognizerResult(entity_type="DATE_TIME", start=27, end=37, score=0.95),  # "1958-04-21"
        ]

        validated_entities = adapter._validate_entities_against_boundaries(
            entities, document_text, segment_boundaries
        )

        # All entities should pass through unchanged
        assert len(validated_entities) == 3
        for original, validated in zip(entities, validated_entities, strict=False):
            assert validated.start == original.start
            assert validated.end == original.end
            assert validated.entity_type == original.entity_type

    def test_entity_at_segment_edge(self):
        """Test entity that ends exactly at segment boundary."""
        adapter = PresidioMaskingAdapter()

        document_text = "John Smith\n\nNext segment"
        segment_boundaries = [
            SegmentBoundary(segment_index=0, start=0, end=10, node_id="#/texts/0"),
            SegmentBoundary(segment_index=1, start=12, end=24, node_id="#/texts/1"),
        ]

        # Entity ends exactly at segment boundary
        entities = [
            RecognizerResult(
                entity_type="PERSON",
                start=0,
                end=10,  # Exactly at boundary
                score=0.95,
            )
        ]

        validated_entities = adapter._validate_entities_against_boundaries(
            entities, document_text, segment_boundaries
        )

        # Should pass through unchanged
        assert len(validated_entities) == 1
        assert validated_entities[0].start == 0
        assert validated_entities[0].end == 10

    def test_multiple_separators_in_entity(self):
        """Test entity containing multiple separators gets truncated at first."""
        adapter = PresidioMaskingAdapter()

        document_text = "Start\n\nMiddle\n\nEnd"
        segment_boundaries = [
            SegmentBoundary(segment_index=0, start=0, end=5, node_id="#/texts/0"),
            SegmentBoundary(segment_index=1, start=7, end=13, node_id="#/texts/1"),
            SegmentBoundary(segment_index=2, start=15, end=18, node_id="#/texts/2"),
        ]

        # Entity spans multiple segments
        entities = [
            RecognizerResult(
                entity_type="MISC",
                start=0,
                end=18,  # Spans all three segments
                score=0.95,
            )
        ]

        validated_entities = adapter._validate_entities_against_boundaries(
            entities, document_text, segment_boundaries
        )

        # Should truncate at first separator
        assert len(validated_entities) == 1
        assert validated_entities[0].start == 0
        assert validated_entities[0].end == 5  # Truncated at first separator
