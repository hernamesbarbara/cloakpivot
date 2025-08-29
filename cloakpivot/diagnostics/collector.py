"""Statistics collection and diagnostic data gathering for masking operations."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from presidio_analyzer import RecognizerResult

from ..core.results import MaskResult


@dataclass
class MaskingStatistics:
    """
    Comprehensive statistics about a masking operation.
    
    Provides detailed metrics about entity detection, masking effectiveness,
    strategy usage, and processing outcomes for analysis and reporting.
    """
    
    total_entities_detected: int = 0
    total_entities_masked: int = 0
    entities_skipped: int = 0
    entities_failed: int = 0
    entity_counts_by_type: dict[str, int] = field(default_factory=dict)
    strategy_usage: dict[str, int] = field(default_factory=dict)
    confidence_statistics: dict[str, float] = field(default_factory=dict)
    detailed_metrics: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    @property
    def masking_success_rate(self) -> float:
        """Calculate the success rate of entity masking."""
        if self.total_entities_detected == 0:
            return 1.0
        return self.total_entities_masked / self.total_entities_detected
    
    def calculate_coverage_percentage(self) -> float:
        """Calculate coverage percentage as masked/detected * 100."""
        if self.total_entities_detected == 0:
            return 100.0
        return (self.total_entities_masked / self.total_entities_detected) * 100.0
    
    def to_dict(self) -> dict[str, Any]:
        """Convert statistics to dictionary for serialization."""
        return {
            "total_entities_detected": self.total_entities_detected,
            "total_entities_masked": self.total_entities_masked,
            "entities_skipped": self.entities_skipped,
            "entities_failed": self.entities_failed,
            "masking_success_rate": self.masking_success_rate,
            "coverage_percentage": self.calculate_coverage_percentage(),
            "entity_counts_by_type": self.entity_counts_by_type,
            "strategy_usage": self.strategy_usage,
            "confidence_statistics": self.confidence_statistics,
            "detailed_metrics": self.detailed_metrics,
            "timestamp": self.timestamp
        }


class DiagnosticsCollector:
    """
    Collects comprehensive diagnostic information from masking operations.
    
    Gathers statistics, performance metrics, and diagnostic data to provide
    insights into masking effectiveness, policy performance, and areas for
    optimization.
    """
    
    def __init__(self) -> None:
        """Initialize the diagnostics collector."""
        pass
    
    def collect_masking_statistics(
        self,
        mask_result: MaskResult,
        original_entities: Optional[list[RecognizerResult]] = None
    ) -> MaskingStatistics:
        """
        Collect comprehensive masking statistics from a MaskResult.
        
        Args:
            mask_result: The result of a masking operation
            original_entities: Original entities detected (for additional analysis)
            
        Returns:
            MaskingStatistics with collected data
        """
        # Get basic counts from the mask result
        stats_obj = mask_result.stats
        
        statistics = MaskingStatistics(
            total_entities_detected=stats_obj.total_entities_found,
            total_entities_masked=stats_obj.entities_masked,
            entities_skipped=stats_obj.entities_skipped,
            entities_failed=stats_obj.entities_failed,
            entity_counts_by_type=mask_result.entities_by_type.copy(),
        )
        
        # Calculate strategy usage from anchor entries
        strategy_counts: dict[str, int] = {}
        for anchor in mask_result.cloakmap.anchors:
            strategy = anchor.strategy_used
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
        statistics.strategy_usage = strategy_counts
        
        # Calculate confidence statistics if original entities provided
        if original_entities:
            confidence_stats = self._calculate_confidence_statistics(original_entities)
            statistics.confidence_statistics = confidence_stats
            
            # Add confidence distribution to detailed metrics
            conf_distribution = self._calculate_confidence_distribution(original_entities)
            statistics.detailed_metrics["confidence_distribution"] = conf_distribution
        
        return statistics
    
    def collect_performance_metrics(self, mask_result: MaskResult) -> dict[str, Any]:
        """
        Extract performance metrics from a MaskResult.
        
        Args:
            mask_result: The result of a masking operation
            
        Returns:
            Dictionary of performance metrics
        """
        perf = mask_result.performance
        
        metrics = {
            "total_time_seconds": perf.total_time_seconds,
            "detection_time_seconds": perf.detection_time.total_seconds() if perf.detection_time else 0.0,
            "masking_time_seconds": perf.masking_time.total_seconds() if perf.masking_time else 0.0,
            "serialization_time_seconds": perf.serialization_time.total_seconds() if perf.serialization_time else 0.0,
        }
        
        # Calculate throughput if we have timing data
        total_time = metrics["total_time_seconds"]
        if total_time > 0:
            entities_processed = mask_result.stats.entities_masked
            metrics["throughput_entities_per_second"] = entities_processed / total_time
        else:
            metrics["throughput_entities_per_second"] = 0.0
            
        return metrics
    
    def collect_processing_diagnostics(self, mask_result: MaskResult) -> dict[str, Any]:
        """
        Extract diagnostic information from a MaskResult.
        
        Args:
            mask_result: The result of a masking operation
            
        Returns:
            Dictionary of diagnostic information
        """
        diagnostics = mask_result.diagnostics
        
        return {
            "warning_count": len(diagnostics.warnings),
            "error_count": len(diagnostics.errors),
            "warnings": diagnostics.warnings.copy(),
            "errors": diagnostics.errors.copy(),
            "entity_conflicts": diagnostics.entity_conflicts.copy(),
            "policy_violations": diagnostics.policy_violations.copy(),
            "anchor_issues": diagnostics.anchor_issues.copy(),
            "has_issues": diagnostics.has_issues,
            "total_issues": diagnostics.total_issues
        }
    
    def generate_comprehensive_report(
        self,
        mask_result: MaskResult,
        original_entities: Optional[list[RecognizerResult]] = None,
        document_metadata: Optional[dict[str, Any]] = None
    ) -> dict[str, Any]:
        """
        Generate a comprehensive diagnostic report.
        
        Args:
            mask_result: The result of a masking operation
            original_entities: Original entities detected
            document_metadata: Additional document metadata
            
        Returns:
            Comprehensive diagnostic report
        """
        statistics = self.collect_masking_statistics(mask_result, original_entities)
        performance = self.collect_performance_metrics(mask_result)
        diagnostics = self.collect_processing_diagnostics(mask_result)
        
        report = {
            "statistics": statistics.to_dict(),
            "performance": performance,
            "diagnostics": diagnostics,
            "document_metadata": document_metadata or {},
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        return report
    
    def _calculate_confidence_statistics(self, entities: list[RecognizerResult]) -> dict[str, float]:
        """Calculate confidence score statistics."""
        if not entities:
            return {}
            
        scores = [entity.score for entity in entities]
        
        return {
            "min_confidence": min(scores),
            "max_confidence": max(scores),
            "mean_confidence": sum(scores) / len(scores),
            "median_confidence": sorted(scores)[len(scores) // 2]
        }
    
    def _calculate_confidence_distribution(self, entities: list[RecognizerResult]) -> dict[str, int]:
        """Calculate distribution of confidence scores across ranges."""
        if not entities:
            return {}
            
        distribution = {
            "high": 0,    # >= 0.8
            "medium": 0,  # 0.5 - 0.8
            "low": 0      # < 0.5
        }
        
        for entity in entities:
            score = entity.score
            if score >= 0.8:
                distribution["high"] += 1
            elif score >= 0.5:
                distribution["medium"] += 1
            else:
                distribution["low"] += 1
                
        return distribution