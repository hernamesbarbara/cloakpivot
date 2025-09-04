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
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

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
            "timestamp": self.timestamp,
        }


class DiagnosticsCollector:
    """
    Collects comprehensive diagnostic information from masking operations.

    Gathers statistics, performance metrics, and diagnostic data to provide
    insights into masking effectiveness, policy performance, and areas for
    optimization.

    The DiagnosticsCollector extracts detailed metrics from MaskResult objects
    including entity detection rates, masking success rates, performance timing,
    and error diagnostics. This information is essential for policy tuning,
    performance optimization, and compliance reporting.

    Examples:
        Basic usage for collecting masking statistics:

        >>> from cloakpivot.diagnostics import DiagnosticsCollector
        >>> from cloakpivot.core.results import MaskResult, ProcessingStats
        >>>
        >>> collector = DiagnosticsCollector()
        >>>
        >>> # Collect statistics from a masking operation
        >>> stats = collector.collect_masking_statistics(mask_result)
        >>> print(f"Detected {stats.total_entities_detected} entities")
        >>> print(f"Masked {stats.total_entities_masked} entities")
        >>> print(f"Success rate: {stats.masking_success_rate:.1%}")

        Collecting performance metrics:

        >>> # Get performance metrics with timing data
        >>> perf_metrics = collector.collect_performance_metrics(mask_result)
        >>> print(f"Total time: {perf_metrics['total_time_seconds']:.2f}s")
        >>> print(f"Throughput: {perf_metrics['throughput_entities_per_second']:.1f} entities/sec")

        Comprehensive report generation:

        >>> # Generate a complete diagnostic report
        >>> report = collector.generate_comprehensive_report(
        ...     mask_result=mask_result,
        ...     original_entities=detected_entities,
        ...     document_metadata={"name": "document.pdf", "size_bytes": 1024}
        ... )
        >>> print(f"Report contains {len(report)} sections")

        Analyzing entity confidence distribution:

        >>> from presidio_analyzer import RecognizerResult
        >>>
        >>> original_entities = [
        ...     RecognizerResult(entity_type="PERSON", start=0, end=8, score=0.95),
        ...     RecognizerResult(entity_type="EMAIL", start=10, end=25, score=0.85)
        ... ]
        >>>
        >>> stats = collector.collect_masking_statistics(
        ...     mask_result=mask_result,
        ...     original_entities=original_entities
        ... )
        >>>
        >>> # Access confidence distribution
        >>> conf_dist = stats.detailed_metrics.get("confidence_distribution", {})
        >>> print(f"High confidence entities: {conf_dist.get('high', 0)}")
        >>> print(f"Medium confidence entities: {conf_dist.get('medium', 0)}")
        >>> print(f"Low confidence entities: {conf_dist.get('low', 0)}")

    Attributes:
        None - This class is stateless and processes data from provided MaskResult objects.

    Note:
        All methods are designed to handle missing or None values gracefully,
        returning sensible defaults when data is unavailable.
    """

    def __init__(self) -> None:
        """Initialize the diagnostics collector."""
        pass

    def collect_masking_statistics(
        self,
        mask_result: MaskResult,
        original_entities: Optional[list[RecognizerResult]] = None,
    ) -> MaskingStatistics:
        """
        Collect comprehensive masking statistics from a MaskResult.

        Args:
            mask_result: The result of a masking operation
            original_entities: Original entities detected (for additional analysis)

        Returns:
            MaskingStatistics with collected data
        """
        # Handle missing or None stats object
        if not hasattr(mask_result, "stats") or mask_result.stats is None:
            return MaskingStatistics(
                total_entities_detected=0,
                total_entities_masked=0,
                entities_skipped=0,
                entities_failed=0,
                entity_counts_by_type={},
                strategy_usage={},
                confidence_statistics={},
                detailed_metrics={},
            )

        # Get basic counts from the mask result with null checking
        stats_obj = mask_result.stats

        # Safely extract entity counts with defaults
        total_entities_found = getattr(stats_obj, "total_entities_found", 0)
        entities_masked = getattr(stats_obj, "entities_masked", 0)
        entities_skipped = getattr(stats_obj, "entities_skipped", 0)
        entities_failed = getattr(stats_obj, "entities_failed", 0)

        # Safely get entities_by_type with null check
        entities_by_type = {}
        if (
            hasattr(mask_result, "entities_by_type")
            and mask_result.entities_by_type is not None
        ):
            entities_by_type = mask_result.entities_by_type.copy()

        statistics = MaskingStatistics(
            total_entities_detected=total_entities_found,
            total_entities_masked=entities_masked,
            entities_skipped=entities_skipped,
            entities_failed=entities_failed,
            entity_counts_by_type=entities_by_type,
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
            conf_distribution = self._calculate_confidence_distribution(
                original_entities
            )
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
        # Handle missing or None performance object
        if not hasattr(mask_result, "performance") or mask_result.performance is None:
            return {
                "total_time_seconds": 0.0,
                "detection_time_seconds": 0.0,
                "masking_time_seconds": 0.0,
                "serialization_time_seconds": 0.0,
                "throughput_entities_per_second": 0.0,
            }

        perf = mask_result.performance

        # Handle missing or None total_time_seconds
        total_time = getattr(perf, "total_time_seconds", None)
        if total_time is None or not isinstance(total_time, (int, float)):
            total_time = 0.0

        metrics = {
            "total_time_seconds": total_time,
            "detection_time_seconds": (
                perf.detection_time.total_seconds()
                if hasattr(perf, "detection_time") and perf.detection_time is not None
                else 0.0
            ),
            "masking_time_seconds": (
                perf.masking_time.total_seconds()
                if hasattr(perf, "masking_time") and perf.masking_time is not None
                else 0.0
            ),
            "serialization_time_seconds": (
                perf.serialization_time.total_seconds()
                if hasattr(perf, "serialization_time")
                and perf.serialization_time is not None
                else 0.0
            ),
        }

        # Calculate throughput if we have timing data and stats
        if (
            total_time > 0
            and hasattr(mask_result, "stats")
            and mask_result.stats is not None
        ):
            entities_processed = getattr(mask_result.stats, "entities_masked", 0)
            if isinstance(entities_processed, (int, float)) and entities_processed >= 0:
                metrics["throughput_entities_per_second"] = (
                    entities_processed / total_time
                )
            else:
                metrics["throughput_entities_per_second"] = 0.0
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
            "total_issues": diagnostics.total_issues,
        }

    def generate_comprehensive_report(
        self,
        mask_result: MaskResult,
        original_entities: Optional[list[RecognizerResult]] = None,
        document_metadata: Optional[dict[str, Any]] = None,
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
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        return report

    def _calculate_confidence_statistics(
        self, entities: list[RecognizerResult]
    ) -> dict[str, float]:
        """Calculate confidence score statistics."""
        if not entities:
            return {}

        scores = [entity.score for entity in entities]

        return {
            "min_confidence": min(scores),
            "max_confidence": max(scores),
            "mean_confidence": sum(scores) / len(scores),
            "median_confidence": sorted(scores)[len(scores) // 2],
        }

    def _calculate_confidence_distribution(
        self, entities: list[RecognizerResult]
    ) -> dict[str, int]:
        """Calculate distribution of confidence scores across ranges."""
        if not entities:
            return {}

        distribution = {
            "high": 0,  # >= 0.8
            "medium": 0,  # 0.5 - 0.8
            "low": 0,  # < 0.5
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
