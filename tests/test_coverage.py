"""Tests for coverage analysis functionality."""

from unittest.mock import MagicMock

import pytest

from cloakpivot.core.anchors import AnchorEntry
from cloakpivot.diagnostics.coverage import (
    CoverageAnalyzer,
    CoverageMetrics,
    DocumentSection,
)
from cloakpivot.document.extractor import TextSegment


@pytest.fixture
def sample_text_segments():
    """Sample text segments representing different document sections."""
    text1 = "Introduction to Privacy Policy"
    text2 = "We collect personal information including names, email addresses, and phone numbers."
    text3 = "Contact: John Doe at john.doe@company.com or 555-123-4567"
    text4 = "Data Protection Officer: Jane Smith (jane@company.com)"

    return [
        TextSegment(
            text=text1,
            node_id="heading_1",
            start_offset=0,
            end_offset=len(text1),
            node_type="heading",
        ),
        TextSegment(
            text=text2,
            node_id="paragraph_1",
            start_offset=len(text1) + 1,
            end_offset=len(text1) + 1 + len(text2),
            node_type="paragraph",
        ),
        TextSegment(
            text=text3,
            node_id="paragraph_2",
            start_offset=len(text1) + 1 + len(text2) + 1,
            end_offset=len(text1) + 1 + len(text2) + 1 + len(text3),
            node_type="paragraph",
        ),
        TextSegment(
            text=text4,
            node_id="footer_1",
            start_offset=len(text1) + 1 + len(text2) + 1 + len(text3) + 1,
            end_offset=len(text1) + 1 + len(text2) + 1 + len(text3) + 1 + len(text4),
            node_type="footer",
        ),
    ]


@pytest.fixture
def sample_anchor_entries():
    """Sample anchor entries for coverage analysis."""
    return [
        AnchorEntry.create_from_detection(
            node_id="paragraph_2",
            start=0,
            end=8,
            entity_type="PERSON",
            confidence=0.95,
            original_text="John Doe",
            masked_value="[PERSON]",
            strategy_used="template",
            replacement_id="repl_1",
        ),
        AnchorEntry.create_from_detection(
            node_id="paragraph_2",
            start=12,
            end=35,
            entity_type="EMAIL_ADDRESS",
            confidence=0.90,
            original_text="john.doe@company.com",
            masked_value="[EMAIL]",
            strategy_used="template",
            replacement_id="repl_2",
        ),
        AnchorEntry.create_from_detection(
            node_id="paragraph_2",
            start=39,
            end=51,
            entity_type="PHONE_NUMBER",
            confidence=0.85,
            original_text="555-123-4567",
            masked_value="XXX-XXX-4567",
            strategy_used="partial",
            replacement_id="repl_3",
        ),
        AnchorEntry.create_from_detection(
            node_id="footer_1",
            start=25,
            end=35,
            entity_type="PERSON",
            confidence=0.88,
            original_text="Jane Smith",
            masked_value="[PERSON]",
            strategy_used="template",
            replacement_id="repl_4",
        ),
        AnchorEntry.create_from_detection(
            node_id="footer_1",
            start=37,
            end=54,
            entity_type="EMAIL_ADDRESS",
            confidence=0.92,
            original_text="jane@company.com",
            masked_value="[EMAIL]",
            strategy_used="template",
            replacement_id="repl_5",
        ),
    ]


class TestCoverageAnalyzer:
    """Test the CoverageAnalyzer class."""

    def test_analyze_document_coverage_basic(
        self, sample_text_segments, sample_anchor_entries
    ):
        """Test basic document coverage analysis."""
        analyzer = CoverageAnalyzer()

        metrics = analyzer.analyze_document_coverage(
            text_segments=sample_text_segments, anchor_entries=sample_anchor_entries
        )

        assert isinstance(metrics, CoverageMetrics)
        assert metrics.total_segments == 4
        assert (
            metrics.segments_with_entities == 2
        )  # paragraph_2 and footer_1 have entities
        assert metrics.overall_coverage_rate == 0.5  # 2/4 segments covered

    def test_analyze_section_breakdown(
        self, sample_text_segments, sample_anchor_entries
    ):
        """Test section-by-section coverage breakdown."""
        analyzer = CoverageAnalyzer()

        metrics = analyzer.analyze_document_coverage(
            text_segments=sample_text_segments, anchor_entries=sample_anchor_entries
        )

        # Check section coverage details
        assert len(metrics.section_coverage) == 3  # heading, paragraph, footer

        paragraph_coverage = next(
            (s for s in metrics.section_coverage if s.section_type == "paragraph"), None
        )
        assert paragraph_coverage is not None
        assert paragraph_coverage.total_segments == 2
        assert paragraph_coverage.segments_with_entities == 1
        assert paragraph_coverage.coverage_rate == 0.5

    def test_analyze_coverage_with_invalid_anchor_entries(self, sample_text_segments):
        """Test coverage analysis with invalid anchor entries."""
        analyzer = CoverageAnalyzer()

        # Test with anchor entry missing required attributes
        invalid_anchor = MagicMock()
        invalid_anchor.node_id = None  # Invalid node_id

        with pytest.raises(ValueError, match="references invalid node_id 'None'"):
            analyzer.analyze_document_coverage(
                text_segments=sample_text_segments, anchor_entries=[invalid_anchor]
            )

    def test_analyze_coverage_with_invalid_node_id(self, sample_text_segments):
        """Test coverage analysis with anchor referencing non-existent node_id."""
        analyzer = CoverageAnalyzer()

        # Create anchor entry with invalid node_id
        invalid_anchor = AnchorEntry.create_from_detection(
            node_id="non_existent_node",  # This node_id doesn't exist in text_segments
            start=0,
            end=5,
            entity_type="PERSON",
            confidence=0.95,
            original_text="Test",
            masked_value="[PERSON]",
            strategy_used="template",
            replacement_id="repl_1",
        )

        with pytest.raises(
            ValueError, match="references invalid node_id 'non_existent_node'"
        ):
            analyzer.analyze_document_coverage(
                text_segments=sample_text_segments, anchor_entries=[invalid_anchor]
            )

    def test_analyze_coverage_with_invalid_positions(self, sample_text_segments):
        """Test coverage analysis with invalid start/end positions."""
        CoverageAnalyzer()

        # Test that creating anchor entry with negative positions raises error
        with pytest.raises(
            ValueError, match="start position must be a non-negative integer"
        ):
            AnchorEntry.create_from_detection(
                node_id="paragraph_1",
                start=-1,  # Invalid negative position
                end=5,
                entity_type="PERSON",
                confidence=0.95,
                original_text="Test",
                masked_value="[PERSON]",
                strategy_used="template",
                replacement_id="repl_1",
            )

    def test_analyze_coverage_with_invalid_range(self, sample_text_segments):
        """Test coverage analysis with invalid start >= end range."""
        CoverageAnalyzer()

        # Test that creating anchor entry with start >= end raises error
        with pytest.raises(
            ValueError, match="end position must be greater than start position"
        ):
            AnchorEntry.create_from_detection(
                node_id="paragraph_1",
                start=10,
                end=5,  # end < start is invalid
                entity_type="PERSON",
                confidence=0.95,
                original_text="Test",
                masked_value="[PERSON]",
                strategy_used="template",
                replacement_id="repl_1",
            )

    def test_analyze_coverage_with_empty_entity_type(self, sample_text_segments):
        """Test coverage analysis with empty entity_type."""
        CoverageAnalyzer()

        # Test that creating anchor entry with empty entity_type raises error
        with pytest.raises(ValueError, match="entity_type must be a non-empty string"):
            AnchorEntry.create_from_detection(
                node_id="paragraph_1",
                start=0,
                end=5,
                entity_type="",  # Empty entity_type is invalid
                confidence=0.95,
                original_text="Test",
                masked_value="[PERSON]",
                strategy_used="template",
                replacement_id="repl_1",
            )

    def test_analyze_entity_distribution(
        self, sample_text_segments, sample_anchor_entries
    ):
        """Test entity type distribution analysis."""
        analyzer = CoverageAnalyzer()

        metrics = analyzer.analyze_document_coverage(
            text_segments=sample_text_segments, anchor_entries=sample_anchor_entries
        )

        # Check entity distribution
        assert metrics.entity_distribution["PERSON"] == 2
        assert metrics.entity_distribution["EMAIL_ADDRESS"] == 2
        assert metrics.entity_distribution["PHONE_NUMBER"] == 1
        assert metrics.total_entities == 5

    def test_calculate_entity_density(
        self, sample_text_segments, sample_anchor_entries
    ):
        """Test entity density calculation."""
        analyzer = CoverageAnalyzer()

        metrics = analyzer.analyze_document_coverage(
            text_segments=sample_text_segments, anchor_entries=sample_anchor_entries
        )

        # Should calculate entities per segment
        expected_density = 5 / 4  # 5 entities across 4 segments
        assert abs(metrics.entity_density - expected_density) < 0.01

    def test_identify_coverage_gaps(self, sample_text_segments, sample_anchor_entries):
        """Test identification of coverage gaps."""
        analyzer = CoverageAnalyzer()

        metrics = analyzer.analyze_document_coverage(
            text_segments=sample_text_segments, anchor_entries=sample_anchor_entries
        )

        # Should identify segments without entities as gaps
        gaps = metrics.coverage_gaps
        assert len(gaps) == 2  # heading_1 and paragraph_1 have no entities
        gap_node_ids = [gap["node_id"] for gap in gaps]
        assert "heading_1" in gap_node_ids
        assert "paragraph_1" in gap_node_ids

    def test_generate_coverage_recommendations(
        self, sample_text_segments, sample_anchor_entries
    ):
        """Test generation of coverage improvement recommendations."""
        analyzer = CoverageAnalyzer()

        metrics = analyzer.analyze_document_coverage(
            text_segments=sample_text_segments, anchor_entries=sample_anchor_entries
        )

        recommendations = analyzer.generate_recommendations(metrics)

        assert isinstance(recommendations, list)
        assert len(recommendations) > 0

        # Should recommend reviewing segments without entities
        gap_recommendation = any(
            "uncovered segments" in rec.lower() or "gap" in rec.lower()
            for rec in recommendations
        )
        assert gap_recommendation

    def test_empty_document_coverage(self):
        """Test coverage analysis with empty document."""
        analyzer = CoverageAnalyzer()

        metrics = analyzer.analyze_document_coverage(
            text_segments=[], anchor_entries=[]
        )

        assert metrics.total_segments == 0
        assert metrics.segments_with_entities == 0
        assert metrics.overall_coverage_rate == 1.0  # 100% for empty document
        assert len(metrics.section_coverage) == 0

    def test_no_entities_coverage(self, sample_text_segments):
        """Test coverage analysis with no entities found."""
        analyzer = CoverageAnalyzer()

        metrics = analyzer.analyze_document_coverage(
            text_segments=sample_text_segments, anchor_entries=[]
        )

        assert metrics.total_segments == 4
        assert metrics.segments_with_entities == 0
        assert metrics.overall_coverage_rate == 0.0
        assert metrics.total_entities == 0


class TestCoverageMetrics:
    """Test the CoverageMetrics data class."""

    def test_coverage_metrics_creation(self):
        """Test basic CoverageMetrics creation."""
        section = DocumentSection(
            section_type="paragraph",
            total_segments=3,
            segments_with_entities=2,
            entity_count=5,
        )

        metrics = CoverageMetrics(
            total_segments=10,
            segments_with_entities=8,
            overall_coverage_rate=0.8,
            section_coverage=[section],
            entity_distribution={"PERSON": 3, "EMAIL": 2},
            entity_density=0.5,
            coverage_gaps=[{"node_id": "gap1", "type": "paragraph"}],
        )

        assert metrics.total_segments == 10
        assert metrics.overall_coverage_rate == 0.8
        assert metrics.total_entities == 5  # 3 + 2
        assert len(metrics.section_coverage) == 1

    def test_coverage_metrics_to_dict(self):
        """Test conversion to dictionary."""
        section = DocumentSection(
            section_type="heading",
            total_segments=2,
            segments_with_entities=1,
            entity_count=1,
        )

        metrics = CoverageMetrics(
            total_segments=5,
            segments_with_entities=3,
            overall_coverage_rate=0.6,
            section_coverage=[section],
            entity_distribution={"PERSON": 1},
            entity_density=0.2,
            coverage_gaps=[],
        )

        result = metrics.to_dict()

        assert result["total_segments"] == 5
        assert result["overall_coverage_rate"] == 0.6
        assert result["entity_distribution"]["PERSON"] == 1
        assert len(result["section_coverage"]) == 1
        assert result["section_coverage"][0]["section_type"] == "heading"


class TestDocumentSection:
    """Test the DocumentSection data class."""

    def test_document_section_coverage_rate(self):
        """Test coverage rate calculation."""
        section = DocumentSection(
            section_type="paragraph",
            total_segments=10,
            segments_with_entities=7,
            entity_count=15,
        )

        assert section.coverage_rate == 0.7  # 7/10
        assert section.average_entities_per_segment == 1.5  # 15/10

    def test_document_section_zero_segments(self):
        """Test handling of zero segments."""
        section = DocumentSection(
            section_type="table",
            total_segments=0,
            segments_with_entities=0,
            entity_count=0,
        )

        assert section.coverage_rate == 1.0  # Default to 100% for empty sections
        assert section.average_entities_per_segment == 0.0

    def test_document_section_to_dict(self):
        """Test conversion to dictionary."""
        section = DocumentSection(
            section_type="footer",
            total_segments=3,
            segments_with_entities=2,
            entity_count=4,
        )

        result = section.to_dict()

        assert result["section_type"] == "footer"
        assert result["total_segments"] == 3
        assert result["coverage_rate"] == pytest.approx(0.667, abs=0.01)
        assert result["average_entities_per_segment"] == pytest.approx(1.333, abs=0.01)
