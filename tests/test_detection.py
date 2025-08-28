"""Tests for entity detection pipeline."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import List

from cloakpivot.core.detection import (
    EntityDetectionPipeline,
    SegmentAnalysisResult,
    DocumentAnalysisResult
)
from cloakpivot.core.analyzer import EntityDetectionResult
from cloakpivot.core.policies import MaskingPolicy
from cloakpivot.document.extractor import TextSegment


@pytest.fixture
def mock_analyzer():
    """Create a mock analyzer for testing."""
    analyzer = Mock()
    analyzer.analyze_text.return_value = []
    return analyzer


@pytest.fixture
def sample_text_segments():
    """Create sample text segments for testing."""
    return [
        TextSegment(
            node_id="#/texts/0",
            text="Contact John Doe at john.doe@email.com or 555-123-4567",
            start_offset=0,
            end_offset=54,
            node_type="TextItem"
        ),
        TextSegment(
            node_id="#/texts/1", 
            text="His SSN is 123-45-6789",
            start_offset=55,
            end_offset=77,
            node_type="TextItem"
        ),
        TextSegment(
            node_id="#/titles/0",
            text="Personal Information",
            start_offset=78,
            end_offset=98,
            node_type="TitleItem"
        )
    ]


@pytest.fixture
def sample_entities():
    """Create sample detected entities."""
    return [
        EntityDetectionResult("PERSON", 8, 16, 0.9, "John Doe"),
        EntityDetectionResult("EMAIL_ADDRESS", 20, 37, 0.95, "john.doe@email.com"),
        EntityDetectionResult("PHONE_NUMBER", 41, 53, 0.85, "555-123-4567"),
        EntityDetectionResult("US_SSN", 11, 23, 0.98, "123-45-6789")
    ]


class TestEntityDetectionPipeline:
    """Test the main entity detection pipeline."""
    
    def test_pipeline_initialization(self):
        """Test pipeline initialization with default analyzer."""
        pipeline = EntityDetectionPipeline()
        
        assert pipeline.analyzer is not None
        assert pipeline.text_extractor is not None
    
    def test_pipeline_from_policy(self):
        """Test pipeline creation from masking policy."""
        policy = MaskingPolicy(locale="es", thresholds={"PERSON": 0.8})
        
        pipeline = EntityDetectionPipeline.from_policy(policy)
        
        assert pipeline.analyzer.config.language == "es"
    
    @patch('cloakpivot.core.detection.TextExtractor')
    def test_analyze_document(self, mock_extractor_class, mock_analyzer, sample_text_segments, sample_entities):
        """Test document analysis."""
        # Setup mocks
        mock_extractor = Mock()
        mock_extractor.extract_text_segments.return_value = sample_text_segments
        mock_extractor_class.return_value = mock_extractor
        
        mock_analyzer.analyze_text.side_effect = [
            [sample_entities[0], sample_entities[1], sample_entities[2]],  # First segment
            [sample_entities[3]],  # Second segment
            []  # Third segment (title)
        ]
        
        # Create pipeline
        pipeline = EntityDetectionPipeline(mock_analyzer)
        pipeline.text_extractor = mock_extractor
        
        # Create mock document
        mock_doc = Mock()
        mock_doc.name = "test_document.docling.json"
        
        # Run analysis
        result = pipeline.analyze_document(mock_doc)
        
        # Verify results
        assert result.document_name == "test_document.docling.json"
        assert result.segments_analyzed == 3
        assert result.total_entities == 4
        assert result.success_rate == 1.0
        assert "PERSON" in result.entity_breakdown
        assert "EMAIL_ADDRESS" in result.entity_breakdown
        assert "PHONE_NUMBER" in result.entity_breakdown
        assert "US_SSN" in result.entity_breakdown
    
    def test_analyze_text_segments(self, mock_analyzer, sample_text_segments, sample_entities):
        """Test analyzing text segments directly."""
        mock_analyzer.analyze_text.side_effect = [
            [sample_entities[0], sample_entities[1]],  # First segment
            [sample_entities[2]],  # Second segment
            []  # Third segment
        ]
        
        pipeline = EntityDetectionPipeline(mock_analyzer)
        
        results = pipeline.analyze_text_segments(sample_text_segments)
        
        assert len(results) == 3
        assert results[0].entity_count == 2
        assert results[1].entity_count == 1
        assert results[2].entity_count == 0
        assert all(result.success for result in results)
    
    def test_policy_filtering(self, mock_analyzer, sample_text_segments):
        """Test entity filtering based on policy."""
        # Create entities with different confidence levels
        entities = [
            EntityDetectionResult("PERSON", 0, 8, 0.9, "John Doe"),  # Above threshold
            EntityDetectionResult("PERSON", 10, 18, 0.6, "Jane Smith"),  # Below threshold
        ]
        
        mock_analyzer.analyze_text.return_value = entities
        
        # Create policy with high threshold for PERSON
        policy = MaskingPolicy(thresholds={"PERSON": 0.8})
        
        pipeline = EntityDetectionPipeline(mock_analyzer)
        result = pipeline._analyze_segment(sample_text_segments[0], policy)
        
        # Should only keep the high-confidence entity
        assert len(result.entities) == 1
        assert result.entities[0].confidence == 0.9
    
    def test_context_extraction(self, mock_analyzer):
        """Test context extraction from segments."""
        pipeline = EntityDetectionPipeline(mock_analyzer)
        
        # Test different node types
        title_segment = TextSegment("#/titles/0", "Title", 0, 5, "TitleItem")
        text_segment = TextSegment("#/texts/0", "Text", 0, 4, "TextItem")
        table_segment = TextSegment("#/tables/0", "Data", 0, 4, "TableItem")
        
        assert pipeline._extract_context_from_segment(title_segment) == "heading"
        assert pipeline._extract_context_from_segment(text_segment) == "paragraph"
        assert pipeline._extract_context_from_segment(table_segment) == "table"
    
    def test_error_handling(self, mock_analyzer, sample_text_segments):
        """Test error handling during segment analysis."""
        mock_analyzer.analyze_text.side_effect = Exception("Analysis failed")
        
        pipeline = EntityDetectionPipeline(mock_analyzer)
        result = pipeline._analyze_segment(sample_text_segments[0])
        
        assert not result.success
        assert "Analysis failed" in result.error
        assert result.entity_count == 0
    
    def test_empty_segment_handling(self, mock_analyzer):
        """Test handling of empty or very short segments."""
        # Create segments that satisfy validation: text length == offset difference
        empty_segment = TextSegment("#/texts/0", " ", 0, 1, "TextItem")  # Single space
        short_segment = TextSegment("#/texts/1", "x", 0, 1, "TextItem")
        
        pipeline = EntityDetectionPipeline(mock_analyzer)
        
        empty_result = pipeline._analyze_segment(empty_segment)
        short_result = pipeline._analyze_segment(short_segment)
        
        assert empty_result.success
        assert empty_result.entity_count == 0
        assert short_result.success
        assert short_result.entity_count == 0
        
        # Analyzer should not be called for empty/short segments (whitespace and single char)
        mock_analyzer.analyze_text.assert_not_called()


class TestSegmentAnalysisResult:
    """Test segment analysis result functionality."""
    
    def test_result_properties(self, sample_text_segments, sample_entities):
        """Test result properties and statistics."""
        segment = sample_text_segments[0]
        
        # Successful result
        success_result = SegmentAnalysisResult(
            segment=segment,
            entities=sample_entities[:2],
            processing_time_ms=123.45
        )
        
        assert success_result.success
        assert success_result.entity_count == 2
        
        # Failed result
        error_result = SegmentAnalysisResult(
            segment=segment,
            error="Processing failed"
        )
        
        assert not error_result.success
        assert error_result.entity_count == 0


class TestDocumentAnalysisResult:
    """Test document analysis result functionality."""
    
    def test_result_aggregation(self, sample_text_segments, sample_entities):
        """Test result aggregation and statistics."""
        result = DocumentAnalysisResult("test_document")
        
        # Add successful segment results
        success_result1 = SegmentAnalysisResult(
            segment=sample_text_segments[0],
            entities=sample_entities[:2],
            processing_time_ms=100.0
        )
        
        success_result2 = SegmentAnalysisResult(
            segment=sample_text_segments[1], 
            entities=sample_entities[2:],
            processing_time_ms=150.0
        )
        
        # Add failed segment result
        error_result = SegmentAnalysisResult(
            segment=sample_text_segments[2],
            error="Analysis failed",
            processing_time_ms=50.0
        )
        
        result.add_segment_result(success_result1)
        result.add_segment_result(success_result2)
        result.add_segment_result(error_result)
        
        # Verify aggregation
        assert result.segments_analyzed == 3
        assert result.total_entities == 4
        assert result.total_processing_time_ms == 300.0
        assert len(result.errors) == 1
        assert result.success_rate == 2/3  # 2 successful out of 3 segments
        
        # Check entity breakdown
        assert result.entity_breakdown["PERSON"] == 1
        assert result.entity_breakdown["EMAIL_ADDRESS"] == 1
        assert result.entity_breakdown["PHONE_NUMBER"] == 1
        assert result.entity_breakdown["US_SSN"] == 1
    
    def test_get_all_entities(self, sample_text_segments, sample_entities):
        """Test getting all entities with their segments."""
        result = DocumentAnalysisResult("test_document")
        
        success_result = SegmentAnalysisResult(
            segment=sample_text_segments[0],
            entities=sample_entities[:2]
        )
        
        result.add_segment_result(success_result)
        
        all_entities = result.get_all_entities()
        
        assert len(all_entities) == 2
        for entity, segment in all_entities:
            assert isinstance(entity, EntityDetectionResult)
            assert segment == sample_text_segments[0]


class TestAnchorMapping:
    """Test entity to anchor mapping functionality."""
    
    def test_map_entities_to_anchors(self, mock_analyzer, sample_text_segments, sample_entities):
        """Test mapping entities to document anchors."""
        # Create analysis result
        analysis_result = DocumentAnalysisResult("test_document")
        
        segment_result = SegmentAnalysisResult(
            segment=sample_text_segments[0],  # start_offset=0, end_offset=54
            entities=[
                EntityDetectionResult("PERSON", 8, 16, 0.9, "John Doe"),
                EntityDetectionResult("EMAIL_ADDRESS", 20, 37, 0.95, "john.doe@email.com")
            ]
        )
        
        analysis_result.add_segment_result(segment_result)
        
        pipeline = EntityDetectionPipeline(mock_analyzer)
        anchors = pipeline.map_entities_to_anchors(analysis_result)
        
        assert len(anchors) == 2
        
        # Check first anchor (PERSON)
        person_anchor = anchors[0]
        assert person_anchor.entity_type == "PERSON"
        assert person_anchor.node_id == "#/texts/0"
        assert person_anchor.start == 8  # Relative to segment
        assert person_anchor.end == 16   # Relative to segment
        assert person_anchor.metadata["global_start"] == 8  # Global position (segment_start + entity_start)
        assert person_anchor.metadata["global_end"] == 16   # Global position (segment_start + entity_end)
        assert person_anchor.metadata["original_text"] == "John Doe"
        assert person_anchor.confidence == 0.9
        
        # Check second anchor (EMAIL)
        email_anchor = anchors[1]
        assert email_anchor.entity_type == "EMAIL_ADDRESS"
        assert email_anchor.metadata["global_start"] == 20
        assert email_anchor.metadata["global_end"] == 37
    
    def test_anchor_mapping_with_multiple_segments(self, mock_analyzer, sample_text_segments):
        """Test anchor mapping across multiple segments."""
        analysis_result = DocumentAnalysisResult("test_document")
        
        # First segment: start_offset=0, end_offset=54
        segment1_result = SegmentAnalysisResult(
            segment=sample_text_segments[0],
            entities=[EntityDetectionResult("PERSON", 8, 16, 0.9, "John Doe")]
        )
        
        # Second segment: start_offset=55, end_offset=77  
        segment2_result = SegmentAnalysisResult(
            segment=sample_text_segments[1],
            entities=[EntityDetectionResult("US_SSN", 11, 23, 0.98, "123-45-6789")]
        )
        
        analysis_result.add_segment_result(segment1_result)
        analysis_result.add_segment_result(segment2_result)
        
        pipeline = EntityDetectionPipeline(mock_analyzer)
        anchors = pipeline.map_entities_to_anchors(analysis_result)
        
        assert len(anchors) == 2
        
        # First anchor should have global position based on first segment
        assert anchors[0].metadata["global_start"] == 8  # 0 + 8
        assert anchors[0].metadata["global_end"] == 16   # 0 + 16
        
        # Second anchor should have global position based on second segment
        assert anchors[1].metadata["global_start"] == 66  # 55 + 11
        assert anchors[1].metadata["global_end"] == 78    # 55 + 23