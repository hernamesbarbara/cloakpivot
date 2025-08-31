"""Result data structures for masking and unmasking operations."""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Union

from .cloakmap import CloakMap


class OperationStatus(Enum):
    """Status of a masking or unmasking operation."""

    SUCCESS = "success"
    PARTIAL = "partial"  # Some entities processed, some failed
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class ProcessingStats:
    """Statistics about entity processing during masking/unmasking."""

    total_entities_found: int = 0
    entities_masked: int = 0
    entities_skipped: int = 0
    entities_failed: int = 0
    confidence_threshold_rejections: int = 0
    allow_list_skips: int = 0
    deny_list_forced: int = 0

    @property
    def success_rate(self) -> float:
        """Calculate the success rate of entity processing."""
        if self.total_entities_found == 0:
            return 1.0
        return self.entities_masked / self.total_entities_found

    @property
    def entities_by_outcome(self) -> dict[str, int]:
        """Get a breakdown of entities by processing outcome."""
        return {
            "masked": self.entities_masked,
            "skipped": self.entities_skipped,
            "failed": self.entities_failed,
            "threshold_rejected": self.confidence_threshold_rejections,
            "allow_list_skipped": self.allow_list_skips,
            "deny_list_forced": self.deny_list_forced,
        }


@dataclass(frozen=True)
class PerformanceMetrics:
    """Performance metrics for masking/unmasking operations."""

    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    total_time: timedelta = field(default_factory=lambda: timedelta(seconds=0))
    document_load_time: Optional[timedelta] = None
    entity_detection_time: Optional[timedelta] = None
    detection_time: Optional[timedelta] = field(
        default_factory=lambda: timedelta(seconds=0)
    )
    masking_time: Optional[timedelta] = field(
        default_factory=lambda: timedelta(seconds=0)
    )
    serialization_time: Optional[timedelta] = field(
        default_factory=lambda: timedelta(seconds=0)
    )
    cloakmap_creation_time: Optional[timedelta] = None
    memory_peak_mb: float = 0.0
    throughput_mb_per_sec: float = 0.0

    @property
    def total_duration(self) -> Optional[timedelta]:
        """Get the total duration of the operation."""
        if self.end_time:
            return self.end_time - self.start_time
        return None

    @property
    def total_time_seconds(self) -> float:
        """Get the total_time in seconds."""
        return self.total_time.total_seconds()

    @property
    def efficiency_ratio(self) -> float:
        """Calculate the ratio of core operation time to total time."""
        total_seconds = self.total_time.total_seconds()

        if total_seconds == 0:
            return 1.0

        # Sum core operation times
        core_time_seconds = 0.0

        if self.detection_time:
            core_time_seconds += self.detection_time.total_seconds()
        if self.masking_time:
            core_time_seconds += self.masking_time.total_seconds()
        if self.serialization_time:
            core_time_seconds += self.serialization_time.total_seconds()

        return core_time_seconds / total_seconds

    @property
    def duration_seconds(self) -> Optional[float]:
        """Get the total duration in seconds."""
        duration = self.total_duration
        return duration.total_seconds() if duration else None

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "total_duration_seconds": self.duration_seconds,
            "document_load_seconds": (
                self.document_load_time.total_seconds()
                if self.document_load_time
                else None
            ),
            "entity_detection_seconds": (
                self.entity_detection_time.total_seconds()
                if self.entity_detection_time
                else None
            ),
            "masking_seconds": (
                self.masking_time.total_seconds() if self.masking_time else None
            ),
            "serialization_seconds": (
                self.serialization_time.total_seconds()
                if self.serialization_time
                else None
            ),
            "cloakmap_creation_seconds": (
                self.cloakmap_creation_time.total_seconds()
                if self.cloakmap_creation_time
                else None
            ),
        }


@dataclass(frozen=True)
class DiagnosticInfo:
    """Diagnostic information about processing issues and warnings."""

    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    entity_conflicts: list[dict[str, Any]] = field(default_factory=list)
    policy_violations: list[str] = field(default_factory=list)
    anchor_issues: list[str] = field(default_factory=list)
    debug_info: dict[str, Any] = field(default_factory=dict)

    @property
    def has_warnings(self) -> bool:
        """Check if there are any warnings."""
        return bool(self.warnings)

    @property
    def has_errors(self) -> bool:
        """Check if there are any errors."""
        return bool(self.errors)

    @property
    def total_issues(self) -> int:
        """Get total count of all issues."""
        return (
            len(self.warnings)
            + len(self.errors)
            + len(self.entity_conflicts)
            + len(self.policy_violations)
            + len(self.anchor_issues)
        )

    @property
    def has_issues(self) -> bool:
        """Check if there are any issues (warnings or errors)."""
        return self.has_warnings or self.has_errors

    @property
    def issue_count(self) -> int:
        """Get total count of warnings and errors only."""
        return len(self.warnings) + len(self.errors)

    def to_dict(self) -> dict[str, Any]:
        """Convert diagnostics to dictionary."""
        return {
            "warnings": self.warnings,
            "errors": self.errors,
            "entity_conflicts": self.entity_conflicts,
            "policy_violations": self.policy_violations,
            "anchor_issues": self.anchor_issues,
            "summary": {
                "has_warnings": self.has_warnings,
                "has_errors": self.has_errors,
                "total_issues": self.total_issues,
            },
        }


@dataclass(frozen=True)
class MaskResult:
    """
    Result of a document masking operation.

    Contains the masked document, associated CloakMap, processing statistics,
    performance metrics, and diagnostic information.

    Attributes:
        status: Overall status of the masking operation
        masked_document: The masked document content or structure
        cloakmap: CloakMap containing reversible mapping information
        input_file_path: Path to the original input file
        output_file_path: Path where masked content was saved (if applicable)
        cloakmap_file_path: Path where CloakMap was saved (if applicable)
        stats: Statistics about entity processing
        performance: Performance metrics for the operation
        diagnostics: Diagnostic information about issues encountered
        metadata: Additional operation metadata

    Examples:
        >>> # Successful masking result
        >>> result = MaskResult(
        ...     status=OperationStatus.SUCCESS,
        ...     masked_document=masked_doc,
        ...     cloakmap=cloakmap,
        ...     input_file_path="/path/to/input.json",
        ...     stats=ProcessingStats(total_entities_found=5, entities_masked=5)
        ... )
        >>>
        >>> print(f"Masked {result.stats.entities_masked} entities")
        >>> print(f"Success rate: {result.stats.success_rate:.2%}")
    """

    status: OperationStatus
    masked_document: Any  # Could be DoclingDocument, string, bytes, etc.
    cloakmap: CloakMap
    input_file_path: Optional[Union[str, Path]] = None
    output_file_path: Optional[Union[str, Path]] = None
    cloakmap_file_path: Optional[Union[str, Path]] = None
    stats: ProcessingStats = field(default_factory=ProcessingStats)
    performance: PerformanceMetrics = field(default_factory=PerformanceMetrics)
    diagnostics: DiagnosticInfo = field(default_factory=DiagnosticInfo)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_successful(self) -> bool:
        """Check if the masking operation was successful."""
        return self.status == OperationStatus.SUCCESS

    @property
    def is_partial(self) -> bool:
        """Check if the masking operation was partially successful."""
        return self.status == OperationStatus.PARTIAL

    @property
    def entities_by_type(self) -> dict[str, int]:
        """Get count of masked entities by type."""
        return self.cloakmap.entity_count_by_type

    @property
    def total_anchors(self) -> int:
        """Get total number of anchors in the CloakMap."""
        return self.cloakmap.anchor_count

    def get_summary(self) -> dict[str, Any]:
        """Get a summary of the masking operation."""
        return {
            "status": self.status.value,
            "input_file": str(self.input_file_path) if self.input_file_path else None,
            "output_file": (
                str(self.output_file_path) if self.output_file_path else None
            ),
            "cloakmap_file": (
                str(self.cloakmap_file_path) if self.cloakmap_file_path else None
            ),
            "entities_found": self.stats.total_entities_found,
            "entities_masked": self.stats.entities_masked,
            "success_rate": f"{self.stats.success_rate:.2%}",
            "duration_seconds": self.performance.duration_seconds,
            "entities_by_type": self.entities_by_type,
            "has_warnings": self.diagnostics.has_warnings,
            "has_errors": self.diagnostics.has_errors,
            "cloakmap_stats": self.cloakmap.get_stats(),
        }

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            "status": self.status.value,
            "input_file_path": (
                str(self.input_file_path) if self.input_file_path else None
            ),
            "output_file_path": (
                str(self.output_file_path) if self.output_file_path else None
            ),
            "cloakmap_file_path": (
                str(self.cloakmap_file_path) if self.cloakmap_file_path else None
            ),
            "stats": {
                "total_entities_found": self.stats.total_entities_found,
                "entities_masked": self.stats.entities_masked,
                "entities_skipped": self.stats.entities_skipped,
                "entities_failed": self.stats.entities_failed,
                "success_rate": self.stats.success_rate,
                "entities_by_outcome": self.stats.entities_by_outcome,
            },
            "performance": self.performance.to_dict(),
            "diagnostics": self.diagnostics.to_dict(),
            "cloakmap_stats": self.cloakmap.get_stats(),
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class UnmaskResult:
    """
    Result of a document unmasking operation.

    Contains the restored original document, validation information,
    processing statistics, and diagnostic data.

    Attributes:
        status: Overall status of the unmasking operation
        unmasked_document: The restored original document content
        cloakmap: CloakMap used for restoration
        masked_file_path: Path to the masked input file
        output_file_path: Path where unmasked content was saved (if applicable)
        cloakmap_file_path: Path to the CloakMap file used
        restored_stats: Statistics about entity restoration
        validation_results: Results of integrity validation
        performance: Performance metrics for the operation
        diagnostics: Diagnostic information about issues encountered
        metadata: Additional operation metadata

    Examples:
        >>> # Successful unmasking result
        >>> result = UnmaskResult(
        ...     status=OperationStatus.SUCCESS,
        ...     unmasked_document=original_doc,
        ...     cloakmap=cloakmap,
        ...     masked_file_path="/path/to/masked.json",
        ...     restored_stats=ProcessingStats(entities_masked=5)
        ... )
        >>>
        >>> print(f"Restored {result.restored_stats.entities_masked} entities")
        >>> print(f"Validation passed: {result.validation_passed}")
    """

    status: OperationStatus
    unmasked_document: Any  # Could be DoclingDocument, string, bytes, etc.
    cloakmap: CloakMap
    masked_file_path: Optional[Union[str, Path]] = None
    output_file_path: Optional[Union[str, Path]] = None
    cloakmap_file_path: Optional[Union[str, Path]] = None
    restored_stats: ProcessingStats = field(default_factory=ProcessingStats)
    validation_results: dict[str, Any] = field(default_factory=dict)
    performance: PerformanceMetrics = field(default_factory=PerformanceMetrics)
    diagnostics: DiagnosticInfo = field(default_factory=DiagnosticInfo)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_successful(self) -> bool:
        """Check if the unmasking operation was successful."""
        return self.status == OperationStatus.SUCCESS

    @property
    def is_partial(self) -> bool:
        """Check if the unmasking operation was partially successful."""
        return self.status == OperationStatus.PARTIAL

    @property
    def validation_passed(self) -> bool:
        """Check if validation checks passed."""
        valid = self.validation_results.get("valid", False)
        return bool(valid)

    @property
    def entities_restored(self) -> int:
        """Get the number of entities successfully restored."""
        return (
            self.restored_stats.entities_masked
        )  # In unmasking, this represents restored entities

    @property
    def restoration_rate(self) -> float:
        """Calculate the success rate of entity restoration."""
        total_anchors = self.cloakmap.anchor_count
        if total_anchors == 0:
            return 1.0
        return self.entities_restored / total_anchors

    def get_summary(self) -> dict[str, Any]:
        """Get a summary of the unmasking operation."""
        return {
            "status": self.status.value,
            "masked_file": (
                str(self.masked_file_path) if self.masked_file_path else None
            ),
            "output_file": (
                str(self.output_file_path) if self.output_file_path else None
            ),
            "cloakmap_file": (
                str(self.cloakmap_file_path) if self.cloakmap_file_path else None
            ),
            "entities_restored": self.entities_restored,
            "total_anchors": self.cloakmap.anchor_count,
            "restoration_rate": f"{self.restoration_rate:.2%}",
            "validation_passed": self.validation_passed,
            "duration_seconds": self.performance.duration_seconds,
            "has_warnings": self.diagnostics.has_warnings,
            "has_errors": self.diagnostics.has_errors,
            "cloakmap_integrity": self.validation_results,
        }

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            "status": self.status.value,
            "masked_file_path": (
                str(self.masked_file_path) if self.masked_file_path else None
            ),
            "output_file_path": (
                str(self.output_file_path) if self.output_file_path else None
            ),
            "cloakmap_file_path": (
                str(self.cloakmap_file_path) if self.cloakmap_file_path else None
            ),
            "restored_stats": {
                "entities_restored": self.entities_restored,
                "total_anchors": self.cloakmap.anchor_count,
                "restoration_rate": self.restoration_rate,
            },
            "validation_results": self.validation_results,
            "performance": self.performance.to_dict(),
            "diagnostics": self.diagnostics.to_dict(),
            "cloakmap_stats": self.cloakmap.get_stats(),
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class BatchResult:
    """
    Result of a batch masking or unmasking operation.

    Contains results from multiple file operations along with overall statistics.
    """

    operation_type: str  # "mask" or "unmask"
    status: OperationStatus = OperationStatus.SUCCESS
    individual_results: list[Union[MaskResult, UnmaskResult]] = field(
        default_factory=list
    )
    failed_files: list[str] = field(default_factory=list)
    total_processing_time: timedelta = field(
        default_factory=lambda: timedelta(seconds=0)
    )
    batch_stats: dict[str, Any] = field(default_factory=dict)
    overall_performance: PerformanceMetrics = field(default_factory=PerformanceMetrics)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def total_files(self) -> int:
        """Get total number of files processed."""
        return len(self.individual_results) + len(self.failed_files)

    @property
    def successful_files(self) -> int:
        """Get number of successfully processed files."""
        return sum(1 for result in self.individual_results if result.is_successful)

    @property
    def partial_files(self) -> int:
        """Get number of partially processed files."""
        return sum(1 for result in self.individual_results if result.is_partial)

    @property
    def failed_file_count(self) -> int:
        """Get number of completely failed files."""
        return len(self.failed_files)

    @property
    def success_rate(self) -> float:
        """Calculate overall success rate."""
        if self.total_files == 0:
            return 1.0
        return self.successful_files / self.total_files

    def get_summary(self) -> dict[str, Any]:
        """Get a summary of the batch operation."""
        total_entities = 0
        total_processed = 0

        for result in self.individual_results:
            if isinstance(result, MaskResult):
                total_entities += result.stats.total_entities_found
                total_processed += result.stats.entities_masked
            elif isinstance(result, UnmaskResult):
                total_entities += result.cloakmap.anchor_count
                total_processed += result.entities_restored

        return {
            "operation_type": self.operation_type,
            "total_files": self.total_files,
            "successful_files": self.successful_files,
            "partial_files": self.partial_files,
            "failed_files": self.failed_file_count,
            "success_rate": f"{self.success_rate:.2%}",
            "total_entities": total_entities,
            "entities_processed": total_processed,
            "overall_duration_seconds": self.overall_performance.duration_seconds,
            "failed_file_list": self.failed_files,
        }


# Utility functions for result operations


def create_performance_metrics(
    start_time: datetime, end_time: datetime, **durations: timedelta
) -> PerformanceMetrics:
    """
    Create performance metrics with timing information.

    Args:
        start_time: Operation start time
        end_time: Operation end time
        **durations: Named duration parameters

    Returns:
        PerformanceMetrics instance
    """
    total_time = end_time - start_time
    return PerformanceMetrics(
        start_time=start_time,
        end_time=end_time,
        total_time=total_time,
        document_load_time=durations.get("document_load"),
        entity_detection_time=durations.get("entity_detection"),
        detection_time=durations.get("entity_detection", timedelta(seconds=0)),
        masking_time=durations.get("masking", timedelta(seconds=0)),
        serialization_time=durations.get("serialization", timedelta(seconds=0)),
        cloakmap_creation_time=durations.get("cloakmap_creation"),
    )


def create_processing_stats(
    entities_found: int = 0,
    entities_masked: int = 0,
    entities_skipped: int = 0,
    **kwargs,
) -> ProcessingStats:
    """
    Create processing statistics with entity counts.

    Args:
        entities_found: Total entities found
        entities_masked: Entities successfully masked
        entities_skipped: Entities skipped during processing
        **kwargs: Additional parameters (ignored for compatibility)

    Returns:
        ProcessingStats instance
    """
    return ProcessingStats(
        total_entities_found=entities_found,
        entities_masked=entities_masked,
        entities_skipped=entities_skipped,
        entities_failed=kwargs.get("entities_failed", 0),
        confidence_threshold_rejections=kwargs.get("threshold_rejected", 0),
        allow_list_skips=kwargs.get("allow_list_skipped", 0),
        deny_list_forced=kwargs.get("deny_list_forced", 0),
    )


def create_diagnostics(
    warnings: Optional[list[str]] = None,
    errors: Optional[list[str]] = None,
    **issue_lists: list[Any],
) -> DiagnosticInfo:
    """
    Create diagnostic information with issue lists.

    Args:
        warnings: List of warning messages
        errors: List of error messages
        **issue_lists: Named lists of specific issue types

    Returns:
        DiagnosticInfo instance
    """
    return DiagnosticInfo(
        warnings=warnings or [],
        errors=errors or [],
        entity_conflicts=issue_lists.get("entity_conflicts", []),
        policy_violations=issue_lists.get("policy_violations", []),
        anchor_issues=issue_lists.get("anchor_issues", []),
    )
