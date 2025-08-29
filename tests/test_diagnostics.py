"""Tests for diagnostics and reporting functionality."""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from presidio_analyzer import RecognizerResult

from cloakpivot.core.anchors import AnchorEntry
from cloakpivot.core.cloakmap import CloakMap
from cloakpivot.core.results import MaskResult, OperationStatus, PerformanceMetrics, ProcessingStats
from cloakpivot.diagnostics.collector import DiagnosticsCollector, MaskingStatistics


@pytest.fixture
def sample_anchor_entries():
    """Sample anchor entries for testing."""
    return [
        AnchorEntry.create_from_detection(
            node_id="text_1",
            start=0,
            end=8,
            entity_type="PERSON",
            confidence=0.95,
            original_text="John Doe",
            masked_value="[PERSON]",
            strategy_used="template",
            replacement_id="repl_1"
        ),
        AnchorEntry.create_from_detection(
            node_id="text_1", 
            start=20,
            end=35,
            entity_type="EMAIL_ADDRESS",
            confidence=0.85,
            original_text="john@example.com",
            masked_value="[EMAIL]",
            strategy_used="template", 
            replacement_id="repl_2"
        ),
        AnchorEntry.create_from_detection(
            node_id="text_2",
            start=5,
            end=17,
            entity_type="PHONE_NUMBER", 
            confidence=0.75,
            original_text="555-123-4567",
            masked_value="XXX-XXX-4567",
            strategy_used="partial",
            replacement_id="repl_3"
        )
    ]


@pytest.fixture
def sample_entities():
    """Sample RecognizerResult entities for testing."""
    return [
        RecognizerResult(entity_type="PERSON", start=0, end=8, score=0.95),
        RecognizerResult(entity_type="EMAIL_ADDRESS", start=20, end=35, score=0.85),
        RecognizerResult(entity_type="PHONE_NUMBER", start=50, end=62, score=0.75)
    ]


@pytest.fixture
def sample_mask_result(sample_anchor_entries):
    """Sample MaskResult for testing with realistic data."""
    # Create a realistic CloakMap instead of mock
    cloakmap = CloakMap(
        doc_id="test_document_001",
        doc_hash="abc123def456",
        version="1.0",
        anchors=sample_anchor_entries,
        created_at=datetime.now(timezone.utc),
        policy_snapshot=None
    )
    
    # Create a realistic masked document structure
    masked_document = {
        "metadata": {
            "document_id": "test_document_001",
            "processed_at": datetime.now(timezone.utc).isoformat(),
            "format": "json"
        },
        "content": {
            "text": "Hello [PERSON_1] your email [EMAIL_1] and phone [PHONE_1] are masked.",
            "segments": [
                {
                    "node_id": "text_0", 
                    "content": "Hello [PERSON_1] your email",
                    "node_type": "paragraph"
                },
                {
                    "node_id": "text_1",
                    "content": "[EMAIL_1] and phone", 
                    "node_type": "paragraph"
                },
                {
                    "node_id": "text_2",
                    "content": "[PHONE_1] are masked.",
                    "node_type": "paragraph"
                }
            ]
        }
    }
    
    return MaskResult(
        status=OperationStatus.SUCCESS,
        masked_document=masked_document,
        cloakmap=cloakmap,
        stats=ProcessingStats(
            total_entities_found=3,
            entities_masked=3,
            entities_skipped=0,
            entities_failed=0
        ),
        performance=PerformanceMetrics(
            total_time=timedelta(seconds=2.5),
            detection_time=timedelta(seconds=1.0),
            masking_time=timedelta(seconds=1.2),
            serialization_time=timedelta(seconds=0.3)
        )
    )


class TestDiagnosticsCollector:
    """Test the DiagnosticsCollector class."""

    def test_collect_masking_statistics_basic(self, sample_mask_result, sample_entities):
        """Test basic statistics collection from masking results."""
        collector = DiagnosticsCollector()
        
        stats = collector.collect_masking_statistics(
            mask_result=sample_mask_result,
            original_entities=sample_entities
        )
        
        assert isinstance(stats, MaskingStatistics)
        assert stats.total_entities_detected == 3
        assert stats.total_entities_masked == 3
        assert stats.masking_success_rate == 1.0
        assert stats.entity_counts_by_type == {"PERSON": 1, "EMAIL_ADDRESS": 1, "PHONE_NUMBER": 1}

    def test_collect_masking_statistics_with_confidence_distribution(self, sample_mask_result, sample_entities):
        """Test confidence distribution calculation."""
        collector = DiagnosticsCollector()
        
        stats = collector.collect_masking_statistics(
            mask_result=sample_mask_result,
            original_entities=sample_entities
        )
        
        assert "confidence_distribution" in stats.detailed_metrics
        confidence_dist = stats.detailed_metrics["confidence_distribution"]
        
        # Should have entries for high, medium confidence ranges
        assert "high" in confidence_dist  # >= 0.8
        assert "medium" in confidence_dist  # 0.5 - 0.8
        assert confidence_dist["high"] == 2  # PERSON (0.95) and EMAIL (0.85)
        assert confidence_dist["medium"] == 1  # PHONE (0.75)

    def test_collect_masking_statistics_strategy_breakdown(self, sample_mask_result, sample_entities):
        """Test strategy usage breakdown."""
        collector = DiagnosticsCollector()
        
        stats = collector.collect_masking_statistics(
            mask_result=sample_mask_result,
            original_entities=sample_entities
        )
        
        assert stats.strategy_usage == {"template": 2, "partial": 1}

    def test_collect_masking_statistics_with_failures(self, sample_entities):
        """Test statistics collection with some failed entities."""
        # Create result with failures
        cloakmap = MagicMock(spec=CloakMap)
        cloakmap.anchors = []  # No successful anchors
        cloakmap.entity_count_by_type = {}
        cloakmap.anchor_count = 0
        
        mask_result = MaskResult(
            status=OperationStatus.PARTIAL,
            masked_document=MagicMock(),
            cloakmap=cloakmap,
            stats=ProcessingStats(
                total_entities_found=3,
                entities_masked=0,
                entities_skipped=1,
                entities_failed=2
            )
        )
        
        collector = DiagnosticsCollector()
        stats = collector.collect_masking_statistics(
            mask_result=mask_result,
            original_entities=sample_entities
        )
        
        assert stats.total_entities_detected == 3
        assert stats.total_entities_masked == 0
        assert stats.masking_success_rate == 0.0
        assert stats.entities_skipped == 1
        assert stats.entities_failed == 2

    def test_collect_performance_metrics(self, sample_mask_result):
        """Test performance metrics collection."""
        collector = DiagnosticsCollector()
        
        perf_metrics = collector.collect_performance_metrics(sample_mask_result)
        
        assert perf_metrics["total_time_seconds"] == 2.5
        assert perf_metrics["detection_time_seconds"] == 1.0
        assert perf_metrics["masking_time_seconds"] == 1.2
        assert perf_metrics["serialization_time_seconds"] == 0.3
        assert "throughput_entities_per_second" in perf_metrics

    def test_collect_processing_diagnostics(self):
        """Test diagnostic information collection."""
        # Create new diagnostics object with some issues
        from cloakpivot.core.results import DiagnosticInfo, MaskResult, OperationStatus
        
        diagnostics_with_issues = DiagnosticInfo(
            warnings=["Warning 1", "Warning 2"],
            errors=["Error 1"]
        )
        
        mask_result = MaskResult(
            status=OperationStatus.SUCCESS,
            masked_document=MagicMock(),
            cloakmap=MagicMock(),
            diagnostics=diagnostics_with_issues
        )
        
        collector = DiagnosticsCollector()
        
        diagnostics = collector.collect_processing_diagnostics(mask_result)
        
        assert diagnostics["warning_count"] == 2
        assert diagnostics["error_count"] == 1
        assert diagnostics["warnings"] == ["Warning 1", "Warning 2"]
        assert diagnostics["errors"] == ["Error 1"]

    def test_generate_comprehensive_report(self, sample_mask_result, sample_entities):
        """Test comprehensive report generation."""
        collector = DiagnosticsCollector()
        
        report = collector.generate_comprehensive_report(
            mask_result=sample_mask_result,
            original_entities=sample_entities,
            document_metadata={"name": "test_doc.pdf", "size_bytes": 1024}
        )
        
        # Check report structure
        assert "statistics" in report
        assert "performance" in report  
        assert "diagnostics" in report

    def test_collect_masking_statistics_with_null_stats(self):
        """Test statistics collection when MaskResult has null/missing stats."""
        collector = DiagnosticsCollector()
        
        # Create mask result with no stats object
        mask_result_no_stats = MaskResult(
            status=OperationStatus.SUCCESS,
            masked_document={},
            cloakmap=MagicMock(),
            stats=None,  # Missing stats
            performance=None
        )
        
        stats = collector.collect_masking_statistics(mask_result_no_stats)
        
        # Should return default values when stats is None
        assert stats.total_entities_detected == 0
        assert stats.total_entities_masked == 0
        assert stats.entities_skipped == 0
        assert stats.entities_failed == 0
        assert stats.entity_counts_by_type == {}

    def test_collect_performance_metrics_with_null_performance(self):
        """Test performance metrics collection when MaskResult has null performance."""
        collector = DiagnosticsCollector()
        
        # Create mask result with no performance object
        mask_result_no_perf = MaskResult(
            status=OperationStatus.SUCCESS,
            masked_document={},
            cloakmap=MagicMock(),
            stats=MagicMock(),
            performance=None  # Missing performance
        )
        
        perf_metrics = collector.collect_performance_metrics(mask_result_no_perf)
        
        # Should return default values when performance is None
        assert perf_metrics["total_time_seconds"] == 0.0
        assert perf_metrics["detection_time_seconds"] == 0.0
        assert perf_metrics["masking_time_seconds"] == 0.0
        assert perf_metrics["serialization_time_seconds"] == 0.0
        assert perf_metrics["throughput_entities_per_second"] == 0.0

    def test_collect_performance_metrics_with_missing_fields(self):
        """Test performance metrics collection with missing optional fields."""
        collector = DiagnosticsCollector()
        
        # Create performance object with missing optional fields
        incomplete_performance = MagicMock()
        incomplete_performance.total_time_seconds = None  # Missing field
        incomplete_performance.detection_time = None
        incomplete_performance.masking_time = None
        incomplete_performance.serialization_time = None
        
        mask_result = MaskResult(
            status=OperationStatus.SUCCESS,
            masked_document={},
            cloakmap=MagicMock(),
            stats=MagicMock(),
            performance=incomplete_performance
        )
        
        perf_metrics = collector.collect_performance_metrics(mask_result)
        
        # Should handle missing fields gracefully
        assert perf_metrics["total_time_seconds"] == 0.0
        assert perf_metrics["detection_time_seconds"] == 0.0
        assert perf_metrics["masking_time_seconds"] == 0.0
        assert perf_metrics["serialization_time_seconds"] == 0.0


class TestMaskingStatistics:
    """Test the MaskingStatistics data class."""

    def test_masking_statistics_creation(self):
        """Test basic MaskingStatistics creation."""
        stats = MaskingStatistics(
            total_entities_detected=10,
            total_entities_masked=8,
            entities_skipped=1,
            entities_failed=1,
            entity_counts_by_type={"PERSON": 3, "EMAIL": 5},
            strategy_usage={"redact": 6, "template": 2}
        )
        
        assert stats.total_entities_detected == 10
        assert stats.total_entities_masked == 8
        assert stats.masking_success_rate == 0.8
        assert stats.entity_counts_by_type["PERSON"] == 3

    def test_masking_statistics_coverage_calculation(self):
        """Test coverage percentage calculation."""
        stats = MaskingStatistics(
            total_entities_detected=20,
            total_entities_masked=18,
            entity_counts_by_type={"PERSON": 8, "EMAIL": 10}
        )
        
        coverage = stats.calculate_coverage_percentage()
        assert coverage == 90.0  # 18/20 * 100

    def test_masking_statistics_to_dict(self):
        """Test conversion to dictionary."""
        stats = MaskingStatistics(
            total_entities_detected=5,
            total_entities_masked=4,
            entity_counts_by_type={"PERSON": 2, "EMAIL": 2},
            strategy_usage={"redact": 4}
        )
        
        result = stats.to_dict()
        
        assert result["total_entities_detected"] == 5
        assert result["masking_success_rate"] == 0.8
        assert result["entity_counts_by_type"]["PERSON"] == 2
        assert result["strategy_usage"]["redact"] == 4
        assert "coverage_percentage" in result


class TestDiagnosticsCollectorIntegration:
    """Integration tests for DiagnosticsCollector."""
    
    def test_end_to_end_statistics_collection(self, tmp_path):
        """Test end-to-end statistics collection with file I/O."""
        # This would test with actual document processing
        # For now, placeholder for future integration test
        pass