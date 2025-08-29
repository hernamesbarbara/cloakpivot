"""Tests for diagnostic reporting functionality."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from presidio_analyzer import RecognizerResult

from cloakpivot.core.results import MaskResult, OperationStatus, PerformanceMetrics, ProcessingStats
from cloakpivot.diagnostics.collector import DiagnosticsCollector, MaskingStatistics
from cloakpivot.diagnostics.coverage import CoverageMetrics, DocumentSection
from cloakpivot.diagnostics.reporting import DiagnosticReporter, ReportFormat, ReportData


@pytest.fixture
def sample_report_data():
    """Sample comprehensive report data for testing."""
    statistics = MaskingStatistics(
        total_entities_detected=10,
        total_entities_masked=8,
        entities_skipped=1,
        entities_failed=1,
        entity_counts_by_type={"PERSON": 3, "EMAIL_ADDRESS": 3, "PHONE_NUMBER": 2},
        strategy_usage={"template": 6, "partial": 2},
        confidence_statistics={"min_confidence": 0.6, "max_confidence": 0.95, "mean_confidence": 0.82}
    )
    
    coverage = CoverageMetrics(
        total_segments=5,
        segments_with_entities=3,
        overall_coverage_rate=0.6,
        section_coverage=[
            DocumentSection("paragraph", 3, 2, 6),
            DocumentSection("heading", 2, 1, 2)
        ],
        entity_distribution={"PERSON": 3, "EMAIL_ADDRESS": 3, "PHONE_NUMBER": 2},
        entity_density=1.6,
        coverage_gaps=[{"node_id": "gap1", "type": "paragraph"}]
    )
    
    performance = {
        "total_time_seconds": 3.5,
        "detection_time_seconds": 1.2,
        "masking_time_seconds": 1.8,
        "serialization_time_seconds": 0.5,
        "throughput_entities_per_second": 2.29
    }
    
    diagnostics = {
        "warning_count": 2,
        "error_count": 1,
        "warnings": ["Warning about low confidence entity", "Performance warning"],
        "errors": ["Failed to mask complex entity"],
        "has_issues": True
    }
    
    return ReportData(
        statistics=statistics,
        coverage=coverage,
        performance=performance,
        diagnostics=diagnostics,
        document_metadata={"name": "test_doc.pdf", "size_bytes": 2048},
        recommendations=["Review low confidence entities", "Optimize policy for better coverage"]
    )


class TestDiagnosticReporter:
    """Test the DiagnosticReporter class."""

    def test_reporter_initialization(self):
        """Test basic reporter initialization."""
        reporter = DiagnosticReporter()
        assert reporter is not None

    def test_generate_json_report(self, sample_report_data):
        """Test JSON report generation."""
        reporter = DiagnosticReporter()
        
        json_report = reporter.generate_report(
            data=sample_report_data,
            format=ReportFormat.JSON
        )
        
        # Should be valid JSON
        parsed = json.loads(json_report)
        
        # Check structure
        assert "statistics" in parsed
        assert "coverage" in parsed
        assert "performance" in parsed
        assert "diagnostics" in parsed
        assert "document_metadata" in parsed
        assert "recommendations" in parsed
        assert "timestamp" in parsed
        
        # Check specific values
        assert parsed["statistics"]["total_entities_detected"] == 10
        assert parsed["coverage"]["overall_coverage_rate"] == 0.6
        assert parsed["performance"]["total_time_seconds"] == 3.5

    def test_generate_html_report(self, sample_report_data):
        """Test HTML report generation."""
        reporter = DiagnosticReporter()
        
        html_report = reporter.generate_report(
            data=sample_report_data,
            format=ReportFormat.HTML
        )
        
        # Should contain HTML structure
        assert "<!DOCTYPE html>" in html_report
        assert "<html" in html_report  # Could be <html lang="en">
        assert "<head>" in html_report
        assert "<body>" in html_report
        
        # Should contain key information
        assert "CloakPivot Diagnostic Report" in html_report
        assert "10" in html_report  # total entities
        assert "60.0%" in html_report or "0.6" in html_report  # coverage rate (flexible format)
        assert "3.5" in html_report  # total time

    def test_generate_markdown_report(self, sample_report_data):
        """Test Markdown report generation."""
        reporter = DiagnosticReporter()
        
        md_report = reporter.generate_report(
            data=sample_report_data,
            format=ReportFormat.MARKDOWN
        )
        
        # Should contain Markdown structure
        assert "# CloakPivot Diagnostic Report" in md_report
        assert "## Statistics" in md_report
        assert "## Coverage Analysis" in md_report
        assert "## Performance" in md_report
        
        # Should contain data
        assert "10 entities detected" in md_report
        assert "60.0%" in md_report  # coverage
        assert "3.50 seconds" in md_report  # timing

    def test_save_report_to_file(self, sample_report_data, tmp_path):
        """Test saving report to file."""
        reporter = DiagnosticReporter()
        
        output_file = tmp_path / "test_report.json"
        
        reporter.save_report(
            data=sample_report_data,
            output_path=output_file,
            format=ReportFormat.JSON
        )
        
        # File should exist
        assert output_file.exists()
        
        # Should contain valid JSON
        with open(output_file) as f:
            data = json.load(f)
            assert data["statistics"]["total_entities_detected"] == 10

    def test_generate_summary_report(self, sample_report_data):
        """Test generation of summary report."""
        reporter = DiagnosticReporter()
        
        summary = reporter.generate_summary(sample_report_data)
        
        assert "document" in summary
        assert "entities" in summary
        assert "coverage" in summary
        assert "performance" in summary
        assert "issues" in summary
        
        # Check specific summary values
        assert summary["entities"]["detected"] == 10
        assert summary["entities"]["masked"] == 8
        assert summary["coverage"]["rate"] == 0.6
        assert summary["issues"]["has_errors"] is True

    def test_generate_recommendations(self, sample_report_data):
        """Test recommendation generation."""
        reporter = DiagnosticReporter()
        
        recommendations = reporter._generate_recommendations(sample_report_data)
        
        assert isinstance(recommendations, list)
        assert len(recommendations) >= 2  # Should have the provided recommendations
        assert "Review low confidence entities" in recommendations
        assert "Optimize policy for better coverage" in recommendations

    def test_html_report_includes_charts(self, sample_report_data):
        """Test that HTML reports include chart visualizations."""
        reporter = DiagnosticReporter()
        
        html_report = reporter.generate_report(
            data=sample_report_data,
            format=ReportFormat.HTML
        )
        
        # Should include Chart.js for visualizations
        assert "Chart.js" in html_report or "chart" in html_report.lower()
        
        # Should have chart containers
        assert "entity-distribution-chart" in html_report or "chart" in html_report

    def test_report_with_empty_data(self):
        """Test report generation with minimal data."""
        empty_data = ReportData(
            statistics=MaskingStatistics(),
            coverage=CoverageMetrics(),
            performance={},
            diagnostics={},
            document_metadata={},
            recommendations=[]
        )
        
        reporter = DiagnosticReporter()
        
        # Should not crash with empty data
        json_report = reporter.generate_report(empty_data, ReportFormat.JSON)
        parsed = json.loads(json_report)
        
        assert "statistics" in parsed
        assert parsed["statistics"]["total_entities_detected"] == 0


class TestReportData:
    """Test the ReportData class."""

    def test_report_data_creation(self):
        """Test basic ReportData creation."""
        data = ReportData(
            statistics=MaskingStatistics(total_entities_detected=5),
            coverage=CoverageMetrics(total_segments=3),
            performance={"total_time_seconds": 2.0},
            diagnostics={"warning_count": 1},
            document_metadata={"name": "test.pdf"},
            recommendations=["Improve detection"]
        )
        
        assert data.statistics.total_entities_detected == 5
        assert data.coverage.total_segments == 3
        assert data.performance["total_time_seconds"] == 2.0

    def test_report_data_to_dict(self):
        """Test conversion to dictionary."""
        data = ReportData(
            statistics=MaskingStatistics(total_entities_detected=3),
            coverage=CoverageMetrics(total_segments=2),
            performance={"time": 1.5},
            diagnostics={"errors": []},
            document_metadata={"size": 1024},
            recommendations=["Test rec"]
        )
        
        result = data.to_dict()
        
        assert "statistics" in result
        assert "coverage" in result
        assert "performance" in result
        assert result["document_metadata"]["size"] == 1024


class TestReportFormats:
    """Test different report formats."""

    def test_json_format_enum(self):
        """Test JSON format enum."""
        assert ReportFormat.JSON.value == "json"

    def test_html_format_enum(self):
        """Test HTML format enum."""
        assert ReportFormat.HTML.value == "html"

    def test_markdown_format_enum(self):
        """Test Markdown format enum."""
        assert ReportFormat.MARKDOWN.value == "markdown"


class TestReportingIntegration:
    """Integration tests for reporting with other components."""
    
    def test_end_to_end_report_generation(self, tmp_path):
        """Test complete report generation from mask result."""
        # Create mock mask result
        mask_result = MaskResult(
            status=OperationStatus.SUCCESS,
            masked_document=MagicMock(),
            cloakmap=MagicMock(),
            stats=ProcessingStats(total_entities_found=5, entities_masked=4),
            performance=PerformanceMetrics()
        )
        mask_result.cloakmap.anchors = []
        mask_result.cloakmap.entity_count_by_type = {"PERSON": 2, "EMAIL": 2}
        
        # Use diagnostics collector to gather data
        collector = DiagnosticsCollector()
        comprehensive_report = collector.generate_comprehensive_report(
            mask_result=mask_result,
            original_entities=[
                RecognizerResult(entity_type="PERSON", start=0, end=5, score=0.9),
                RecognizerResult(entity_type="EMAIL", start=10, end=25, score=0.8)
            ],
            document_metadata={"name": "integration_test.pdf"}
        )
        
        # Convert to ReportData format and generate reports
        report_data = ReportData(
            statistics=collector.collect_masking_statistics(mask_result),
            coverage=CoverageMetrics(),  # Would normally be calculated
            performance=comprehensive_report["performance"],
            diagnostics=comprehensive_report["diagnostics"],
            document_metadata=comprehensive_report["document_metadata"],
            recommendations=[]
        )
        
        reporter = DiagnosticReporter()
        
        # Generate different format reports
        json_report = reporter.generate_report(report_data, ReportFormat.JSON)
        html_report = reporter.generate_report(report_data, ReportFormat.HTML)
        
        assert json_report is not None
        assert html_report is not None
        
        # Verify JSON content
        parsed_json = json.loads(json_report)
        assert parsed_json["statistics"]["total_entities_detected"] == 5