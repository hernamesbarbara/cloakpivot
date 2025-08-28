"""Comprehensive tests for Results functionality."""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List

import pytest

from cloakpivot.core.results import (
    OperationStatus,
    ProcessingStats,
    PerformanceMetrics,
    DiagnosticInfo,
    MaskResult,
    UnmaskResult,
    BatchResult,
    create_performance_metrics,
    create_processing_stats
)
from cloakpivot.core.cloakmap import CloakMap


class TestOperationStatus:
    """Test OperationStatus enum."""
    
    def test_status_values(self):
        """Test all status values are defined."""
        assert OperationStatus.SUCCESS.value == "success"
        assert OperationStatus.PARTIAL.value == "partial"
        assert OperationStatus.FAILED.value == "failed"
        assert OperationStatus.CANCELLED.value == "cancelled"


class TestProcessingStats:
    """Test ProcessingStats functionality."""
    
    def test_create_empty_stats(self):
        """Test creating empty ProcessingStats."""
        stats = ProcessingStats()
        
        assert stats.total_entities_found == 0
        assert stats.entities_masked == 0
        assert stats.entities_skipped == 0
        assert stats.entities_failed == 0
    
    def test_create_full_stats(self):
        """Test creating full ProcessingStats."""
        stats = ProcessingStats(
            total_entities_found=20,
            entities_masked=15,
            entities_skipped=3,
            entities_failed=2
        )
        
        assert stats.total_entities_found == 20
        assert stats.entities_masked == 15
        assert stats.entities_skipped == 3
        assert stats.entities_failed == 2
    
    def test_success_rate_property(self):
        """Test success_rate computed property."""
        # Perfect success
        stats1 = ProcessingStats(
            total_entities_found=10,
            entities_masked=10
        )
        assert stats1.success_rate == 1.0
        
        # Partial success
        stats2 = ProcessingStats(
            total_entities_found=10,
            entities_masked=8
        )
        assert stats2.success_rate == 0.8
        
        # No entities found
        stats3 = ProcessingStats(
            total_entities_found=0,
            entities_masked=0
        )
        assert stats3.success_rate == 1.0  # 100% success when no entities to process


class TestPerformanceMetrics:
    """Test PerformanceMetrics functionality."""
    
    def test_create_empty_metrics(self):
        """Test creating empty PerformanceMetrics."""
        metrics = PerformanceMetrics()
        
        assert metrics.total_time == timedelta(seconds=0)
        assert metrics.detection_time == timedelta(seconds=0)
        assert metrics.masking_time == timedelta(seconds=0)
        assert metrics.serialization_time == timedelta(seconds=0)
        assert metrics.memory_peak_mb == 0.0
        assert metrics.throughput_mb_per_sec == 0.0
    
    def test_create_full_metrics(self):
        """Test creating full PerformanceMetrics."""
        metrics = PerformanceMetrics(
            total_time=timedelta(seconds=10),
            detection_time=timedelta(seconds=5),
            masking_time=timedelta(seconds=3),
            serialization_time=timedelta(seconds=2),
            memory_peak_mb=256.5,
            throughput_mb_per_sec=10.2
        )
        
        assert metrics.total_time == timedelta(seconds=10)
        assert metrics.detection_time == timedelta(seconds=5)
        assert metrics.masking_time == timedelta(seconds=3)
        assert metrics.serialization_time == timedelta(seconds=2)
        assert metrics.memory_peak_mb == 256.5
        assert metrics.throughput_mb_per_sec == 10.2
    
    def test_total_time_seconds_property(self):
        """Test total_time_seconds computed property."""
        metrics = PerformanceMetrics(
            total_time=timedelta(seconds=10, milliseconds=500)
        )
        
        assert metrics.total_time_seconds == 10.5
    
    def test_efficiency_ratio_property(self):
        """Test efficiency_ratio computed property."""
        # All time spent on core operations
        metrics1 = PerformanceMetrics(
            total_time=timedelta(seconds=10),
            detection_time=timedelta(seconds=5),
            masking_time=timedelta(seconds=3),
            serialization_time=timedelta(seconds=2)
        )
        assert metrics1.efficiency_ratio == 1.0  # (5+3+2)/10
        
        # Some overhead
        metrics2 = PerformanceMetrics(
            total_time=timedelta(seconds=10),
            detection_time=timedelta(seconds=3),
            masking_time=timedelta(seconds=2),
            serialization_time=timedelta(seconds=1)
        )
        assert metrics2.efficiency_ratio == 0.6  # (3+2+1)/10
        
        # Zero total time
        metrics3 = PerformanceMetrics()
        assert metrics3.efficiency_ratio == 1.0  # 100% when no time spent


class TestDiagnosticInfo:
    """Test DiagnosticInfo functionality."""
    
    def test_create_empty_diagnostics(self):
        """Test creating empty DiagnosticInfo."""
        diag = DiagnosticInfo()
        
        assert diag.warnings == []
        assert diag.errors == []
        assert diag.debug_info == {}
    
    def test_create_full_diagnostics(self):
        """Test creating full DiagnosticInfo."""
        warnings = ["Low confidence entity detected"]
        errors = ["Failed to process entity"]
        debug_info = {"memory_usage": "256MB", "cpu_time": "5.2s"}
        
        diag = DiagnosticInfo(
            warnings=warnings,
            errors=errors,
            debug_info=debug_info
        )
        
        assert diag.warnings == warnings
        assert diag.errors == errors
        assert diag.debug_info == debug_info
    
    def test_has_issues_property(self):
        """Test has_issues computed property."""
        # No issues
        diag1 = DiagnosticInfo()
        assert diag1.has_issues is False
        
        # Only warnings
        diag2 = DiagnosticInfo(warnings=["Warning"])
        assert diag2.has_issues is True
        
        # Only errors
        diag3 = DiagnosticInfo(errors=["Error"])
        assert diag3.has_issues is True
        
        # Both warnings and errors
        diag4 = DiagnosticInfo(warnings=["Warning"], errors=["Error"])
        assert diag4.has_issues is True
    
    def test_issue_count_property(self):
        """Test issue_count computed property."""
        diag = DiagnosticInfo(
            warnings=["Warning1", "Warning2"],
            errors=["Error1"]
        )
        
        assert diag.issue_count == 3  # 2 warnings + 1 error


class TestMaskResult:
    """Test MaskResult functionality."""
    
    def test_create_minimal_mask_result(self):
        """Test creating minimal MaskResult."""
        stats = ProcessingStats(
            total_entities_found=5,
            entities_masked=5
        )
        
        metrics = PerformanceMetrics(
            total_time=timedelta(seconds=5),
            detection_time=timedelta(seconds=2),
            masking_time=timedelta(seconds=2),
            serialization_time=timedelta(seconds=1),
            memory_peak_mb=128.0,
            throughput_mb_per_sec=200.0
        )
        
        cloakmap = CloakMap("1.0", "doc1", "hash1", [], {})
        
        result = MaskResult(
            status=OperationStatus.SUCCESS,
            masked_document="masked content",
            cloakmap=cloakmap,
            input_file_path=Path("input.txt"),
            output_file_path=Path("masked.txt"),
            cloakmap_file_path=Path("mapping.json"),
            stats=stats,
            performance=metrics
        )
        
        assert result.status == OperationStatus.SUCCESS
        assert result.input_file_path == Path("input.txt")
        assert result.output_file_path == Path("masked.txt")
        assert result.cloakmap_file_path == Path("mapping.json")
        assert result.stats == stats
        assert result.performance == metrics
    
    def test_create_full_mask_result(self):
        """Test creating full MaskResult with all fields."""
        stats = ProcessingStats(
            total_entities_found=10,
            entities_masked=8,
            entities_skipped=2
        )
        
        metrics = PerformanceMetrics(
            total_time=timedelta(seconds=10),
            detection_time=timedelta(seconds=4),
            masking_time=timedelta(seconds=4),
            serialization_time=timedelta(seconds=2),
            memory_peak_mb=256.0,
            throughput_mb_per_sec=204.8
        )
        
        diag = DiagnosticInfo(
            warnings=["Low confidence entity"],
            debug_info={"model_version": "1.0"}
        )
        
        cloakmap = CloakMap("1.0", "doc1", "hash1", [], {})
        
        result = MaskResult(
            status=OperationStatus.PARTIAL,
            masked_document="masked content",
            cloakmap=cloakmap,
            input_file_path=Path("input.txt"),
            output_file_path=Path("masked.txt"),
            cloakmap_file_path=Path("mapping.json"),
            stats=stats,
            performance=metrics,
            diagnostics=diag,
            metadata={"policy": "strict"}
        )
        
        assert result.status == OperationStatus.PARTIAL
        assert result.diagnostics.warnings == ["Low confidence entity"]
        assert result.metadata["policy"] == "strict"


class TestUnmaskResult:
    """Test UnmaskResult functionality."""
    
    def test_create_minimal_unmask_result(self):
        """Test creating minimal UnmaskResult."""
        stats = ProcessingStats(
            total_entities_found=5,
            entities_masked=5  # In unmasking context, this represents restored entities
        )
        
        metrics = PerformanceMetrics(
            total_time=timedelta(seconds=3),
            masking_time=timedelta(seconds=2),    # Time to restore entities
            serialization_time=timedelta(seconds=1),
            memory_peak_mb=64.0,
            throughput_mb_per_sec=333.3
        )
        
        cloakmap = CloakMap("1.0", "doc1", "hash1", [], {})
        
        result = UnmaskResult(
            status=OperationStatus.SUCCESS,
            unmasked_document="restored content",
            cloakmap=cloakmap,
            masked_file_path=Path("masked.txt"),
            output_file_path=Path("restored.txt"),
            cloakmap_file_path=Path("mapping.json"),
            restored_stats=stats,
            performance=metrics
        )
        
        assert result.status == OperationStatus.SUCCESS
        assert result.masked_file_path == Path("masked.txt")
        assert result.output_file_path == Path("restored.txt")
        assert result.cloakmap_file_path == Path("mapping.json")
        assert result.restored_stats == stats
        assert result.performance == metrics
    
    def test_entities_restored_property(self):
        """Test entities_restored computed property."""
        stats = ProcessingStats(
            total_entities_found=10,
            entities_masked=8  # This represents restored entities in unmask context
        )
        
        cloakmap = CloakMap("1.0", "doc1", "hash1", [], {})
        
        result = UnmaskResult(
            status=OperationStatus.PARTIAL,
            unmasked_document="restored content",
            cloakmap=cloakmap,
            restored_stats=stats
        )
        
        assert result.entities_restored == 8


class TestBatchResult:
    """Test BatchResult functionality."""
    
    def test_create_empty_batch_result(self):
        """Test creating empty BatchResult."""
        result = BatchResult(
            operation_type="mask",
            status=OperationStatus.SUCCESS,
            individual_results=[],
            failed_files=[],
            total_processing_time=timedelta(seconds=0)
        )
        
        assert result.status == OperationStatus.SUCCESS
        assert result.individual_results == []
        assert result.total_files == 0
        assert result.successful_files == 0
        assert result.failed_file_count == 0
        assert result.total_processing_time == timedelta(seconds=0)
    
    def test_success_rate_property(self):
        """Test success_rate computed property."""
        # Create mock successful results
        cloakmap = CloakMap("1.0", "doc1", "hash1", [], {})
        successful_results = [
            MaskResult(
                status=OperationStatus.SUCCESS,
                masked_document="content",
                cloakmap=cloakmap
            ) for _ in range(5)
        ]
        
        # All successful
        result1 = BatchResult(
            operation_type="mask",
            status=OperationStatus.SUCCESS,
            individual_results=successful_results,
            failed_files=[],
            total_processing_time=timedelta(seconds=10)
        )
        assert result1.success_rate == 1.0
        
        # Create mixed results - 8 successful, 2 failed
        partial_successful_results = [
            MaskResult(
                status=OperationStatus.SUCCESS,
                masked_document="content",
                cloakmap=cloakmap
            ) for _ in range(8)
        ]
        
        # Partial success
        result2 = BatchResult(
            operation_type="mask",
            status=OperationStatus.PARTIAL,
            individual_results=partial_successful_results,
            failed_files=["file1.txt", "file2.txt"],
            total_processing_time=timedelta(seconds=20)
        )
        assert result2.success_rate == 0.8
        
        # No files
        result3 = BatchResult(
            operation_type="mask",
            status=OperationStatus.SUCCESS,
            individual_results=[],
            failed_files=[],
            total_processing_time=timedelta(seconds=0)
        )
        assert result3.success_rate == 1.0  # 100% when no files to process


class TestUtilityFunctions:
    """Test utility functions for creating metrics and stats."""
    
    def test_create_performance_metrics(self):
        """Test create_performance_metrics utility function."""
        start_time = datetime.now()
        end_time = start_time + timedelta(seconds=5)
        
        metrics = create_performance_metrics(
            start_time=start_time,
            end_time=end_time,
            entity_detection=timedelta(seconds=2.0),
            masking=timedelta(seconds=2.5),
            serialization=timedelta(seconds=0.5)
        )
        
        assert metrics.total_time == timedelta(seconds=5)
        assert metrics.detection_time == timedelta(seconds=2.0)
        assert metrics.masking_time == timedelta(seconds=2.5)
        assert metrics.serialization_time == timedelta(seconds=0.5)
    
    def test_create_processing_stats(self):
        """Test create_processing_stats utility function."""
        entity_confidences = [0.85, 0.92, 0.78, 0.96, 0.83]
        
        stats = create_processing_stats(
            entities_found=5,
            entities_masked=4,
            entities_skipped=1,
            bytes_processed=2048,
            entity_confidences=entity_confidences
        )
        
        assert stats.total_entities_found == 5
        assert stats.entities_masked == 4
        assert stats.entities_skipped == 1
        assert stats.success_rate == 0.8  # 4/5
    
    def test_create_processing_stats_empty_confidences(self):
        """Test create_processing_stats with empty confidences."""
        stats = create_processing_stats(
            entities_found=0,
            entities_masked=0,
            entities_skipped=0,
            bytes_processed=0,
            entity_confidences=[]
        )
        
        assert stats.total_entities_found == 0
        assert stats.entities_masked == 0
        assert stats.entities_skipped == 0
        assert stats.success_rate == 1.0