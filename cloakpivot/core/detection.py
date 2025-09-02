"""Entity detection pipeline for analyzing text segments and mapping results to document anchors."""

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Optional

# Temporarily mock docling_core import to allow tests to run
from cloakpivot.core.types import DoclingDocument

from ..document.extractor import TextExtractor, TextSegment
from .analyzer import AnalyzerEngineWrapper, EntityDetectionResult
from .anchors import AnchorEntry
from .performance import profile_method
from .policies import MaskingPolicy

logger = logging.getLogger(__name__)


@dataclass
class SegmentAnalysisResult:
    """Result of analyzing a single text segment.

    Attributes:
        segment: The original text segment that was analyzed
        entities: List of entities detected in this segment
        processing_time_ms: Time taken to process this segment in milliseconds
        error: Optional error message if analysis failed
    """

    segment: TextSegment
    entities: list[EntityDetectionResult] = field(default_factory=list)
    processing_time_ms: float = 0.0
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        """Whether the analysis completed successfully."""
        return self.error is None

    @property
    def entity_count(self) -> int:
        """Number of entities detected in this segment."""
        return len(self.entities)


@dataclass
class DocumentAnalysisResult:
    """Result of analyzing an entire document.

    Attributes:
        document_name: Name of the analyzed document
        segments_analyzed: Number of segments that were analyzed
        total_entities: Total number of entities detected
        entity_breakdown: Count of entities by type
        segment_results: Individual segment analysis results
        total_processing_time_ms: Total time for document analysis
        errors: List of error messages from failed segments
    """

    document_name: str
    segments_analyzed: int = 0
    total_entities: int = 0
    entity_breakdown: dict[str, int] = field(default_factory=dict)
    segment_results: list[SegmentAnalysisResult] = field(default_factory=list)
    total_processing_time_ms: float = 0.0
    errors: list[str] = field(default_factory=list)

    def add_segment_result(self, result: SegmentAnalysisResult) -> None:
        """Add a segment analysis result and update statistics."""
        self.segment_results.append(result)
        self.segments_analyzed += 1
        self.total_processing_time_ms += result.processing_time_ms

        if result.success:
            self.total_entities += result.entity_count

            # Update entity breakdown
            for entity in result.entities:
                self.entity_breakdown[entity.entity_type] = (
                    self.entity_breakdown.get(entity.entity_type, 0) + 1
                )
        else:
            if result.error:
                self.errors.append(result.error)

    @property
    def success_rate(self) -> float:
        """Calculate the success rate of segment analysis."""
        if self.segments_analyzed == 0:
            return 1.0

        successful = sum(1 for result in self.segment_results if result.success)
        return successful / self.segments_analyzed

    def get_all_entities(self) -> list[tuple[Any, TextSegment]]:
        """Get all detected entities with their source segments."""
        entities_with_segments = []

        for result in self.segment_results:
            if result.success:
                for entity in result.entities:
                    entities_with_segments.append((entity, result.segment))

        return entities_with_segments


class EntityDetectionPipeline:
    """Pipeline for detecting PII entities in document text segments."""

    def __init__(
        self,
        analyzer: Optional[AnalyzerEngineWrapper] = None,
        use_shared_analyzer: Optional[bool] = None,
    ):
        """Initialize the detection pipeline.

        Args:
            analyzer: Pre-configured analyzer wrapper (creates default if None)
            use_shared_analyzer: Whether to use shared analyzer instance (defaults to environment variable or True)
        """
        if analyzer is not None:
            # Use provided analyzer directly
            self.analyzer = analyzer
            self._used_shared_analyzer = False
        else:
            # Determine shared analyzer usage from parameter, environment, or default
            default_use_shared = (
                os.getenv("CLOAKPIVOT_USE_SINGLETON", "true").lower() == "true"
            )
            use_shared = (
                use_shared_analyzer
                if use_shared_analyzer is not None
                else default_use_shared
            )

            if use_shared:
                self.analyzer = AnalyzerEngineWrapper.create_shared()
                self._used_shared_analyzer = True
            else:
                self.analyzer = AnalyzerEngineWrapper()
                self._used_shared_analyzer = False

        self.text_extractor = TextExtractor(normalize_whitespace=True)

        shared_status = "shared" if self._used_shared_analyzer else "direct"
        logger.info(
            f"EntityDetectionPipeline initialized with {shared_status} analyzer"
        )

    @classmethod
    def from_policy(
        cls, policy: MaskingPolicy, use_shared_analyzer: Optional[bool] = None
    ) -> "EntityDetectionPipeline":
        """Create detection pipeline from masking policy.

        Args:
            policy: MaskingPolicy to configure the analyzer
            use_shared_analyzer: Whether to use shared analyzer instance (defaults to environment variable or True)

        Returns:
            Configured EntityDetectionPipeline instance
        """
        from .analyzer import AnalyzerConfig

        # Determine shared analyzer usage from parameter, environment, or default
        default_use_shared = (
            os.getenv("CLOAKPIVOT_USE_SINGLETON", "true").lower() == "true"
        )
        use_shared = (
            use_shared_analyzer
            if use_shared_analyzer is not None
            else default_use_shared
        )

        if use_shared:
            # Use shared analyzer with policy-derived config
            config = AnalyzerConfig.from_policy(policy)
            analyzer = AnalyzerEngineWrapper.create_shared(config)
        else:
            # Use direct analyzer creation
            analyzer = AnalyzerEngineWrapper.from_policy(policy)

        return cls(analyzer)

    @profile_method("document_analysis")
    def analyze_document(
        self, document: DoclingDocument, policy: Optional[MaskingPolicy] = None
    ) -> DocumentAnalysisResult:
        """Analyze a complete document for PII entities with performance tracking.

        Args:
            document: DoclingDocument to analyze
            policy: Optional masking policy for filtering/configuration

        Returns:
            DocumentAnalysisResult with detected entities and statistics
        """
        logger.info(f"Starting document analysis for: {document.name}")

        result = DocumentAnalysisResult(document_name=document.name or "unknown")

        try:
            # Extract text segments from document
            segments = self.text_extractor.extract_text_segments(document)
            logger.debug(f"Extracted {len(segments)} text segments")

            # Analyze each segment
            for segment in segments:
                segment_result = self._analyze_segment(segment, policy)
                result.add_segment_result(segment_result)

            logger.info(
                f"Document analysis completed: {result.total_entities} entities found "
                f"in {result.segments_analyzed} segments"
            )

        except Exception as e:
            error_msg = f"Failed to analyze document {document.name}: {e}"
            logger.error(error_msg)
            result.errors.append(error_msg)

        return result

    @profile_method("segment_batch_analysis")
    def analyze_text_segments(
        self, segments: list[TextSegment], policy: Optional[MaskingPolicy] = None
    ) -> list[SegmentAnalysisResult]:
        """Analyze a list of text segments for PII entities with performance tracking.

        Args:
            segments: List of TextSegment objects to analyze
            policy: Optional masking policy for filtering/configuration

        Returns:
            List of SegmentAnalysisResult objects
        """
        logger.info(f"Analyzing {len(segments)} text segments")

        results = []
        for segment in segments:
            result = self._analyze_segment(segment, policy)
            results.append(result)

        return results

    def _analyze_segment(
        self, segment: TextSegment, policy: Optional[MaskingPolicy] = None
    ) -> SegmentAnalysisResult:
        """Analyze a single text segment for PII entities.

        Args:
            segment: TextSegment to analyze
            policy: Optional masking policy for filtering

        Returns:
            SegmentAnalysisResult with detected entities
        """
        import time

        start_time = time.perf_counter()

        result = SegmentAnalysisResult(segment=segment)

        try:
            # Skip empty or very short segments
            if not segment.text.strip() or len(segment.text.strip()) < 2:
                logger.debug(f"Skipping empty/short segment: {segment.node_id}")
                return result

            # Run entity detection
            raw_entities = self.analyzer.analyze_text(segment.text)
            logger.debug(
                f"Found {len(raw_entities)} raw entities in segment {segment.node_id}"
            )

            # Filter entities based on policy if provided
            if policy:
                filtered_entities = []
                for entity in raw_entities:
                    context = self._extract_context_from_segment(segment)

                    if policy.should_mask_entity(
                        entity.text, entity.entity_type, entity.confidence, context
                    ):
                        filtered_entities.append(entity)
                    else:
                        logger.debug(
                            f"Filtered out entity {entity.entity_type} "
                            f"(confidence={entity.confidence:.3f}) due to policy"
                        )

                result.entities = filtered_entities
            else:
                result.entities = raw_entities

            logger.debug(
                f"Segment {segment.node_id}: {len(result.entities)} entities after filtering"
            )

        except Exception as e:
            error_msg = f"Error analyzing segment {segment.node_id}: {e}"
            logger.error(error_msg)
            result.error = error_msg

        finally:
            end_time = time.perf_counter()
            result.processing_time_ms = (end_time - start_time) * 1000

        return result

    def _extract_context_from_segment(self, segment: TextSegment) -> Optional[str]:
        """Extract context information from a text segment.

        Args:
            segment: TextSegment to extract context from

        Returns:
            Context string or None if no specific context
        """
        # Map node types to context strings
        node_type_mapping = {
            "TitleItem": "heading",
            "SectionHeaderItem": "heading",
            "ListItem": "list",
            "TableItem": "table",
            "TextItem": "paragraph",
            "KeyValueItem": "table",  # Treat key-value as table-like
            "CodeItem": "code",
            "FormulaItem": "formula",
        }

        return node_type_mapping.get(segment.node_type)

    @profile_method("entity_anchor_mapping")
    def map_entities_to_anchors(
        self, analysis_result: DocumentAnalysisResult
    ) -> list[AnchorEntry]:
        """Map detected entities to document anchor positions with performance tracking.

        Args:
            analysis_result: Result of document analysis with detected entities

        Returns:
            List of AnchorEntry objects mapping entities to document positions
        """
        logger.info("Mapping entities to document anchors")

        anchor_entries = []
        anchor_id = 1

        for entity, segment in analysis_result.get_all_entities():
            try:
                # Calculate global positions relative to the segment
                global_start = segment.start_offset + entity.start
                global_end = segment.start_offset + entity.end

                # Create anchor entry using factory method with salted checksum
                replacement_id = f"repl_{anchor_id:06d}"

                anchor_entry = AnchorEntry.create_from_detection(
                    node_id=segment.node_id,
                    start=entity.start,  # Relative to segment
                    end=entity.end,  # Relative to segment
                    entity_type=entity.entity_type,
                    confidence=entity.confidence,
                    original_text=entity.text,
                    masked_value="[DETECTED]",  # Placeholder masked value
                    strategy_used="detection",  # Strategy used for detection phase
                    replacement_id=replacement_id,
                    metadata={
                        "segment_type": segment.node_type,
                        "segment_length": segment.length,
                        "global_start": global_start,  # Store global position in metadata
                        "global_end": global_end,
                        "original_text": entity.text,  # Store original text in metadata
                        "detection_timestamp": None,  # Will be set by caller
                    },
                )

                anchor_entries.append(anchor_entry)
                anchor_id += 1

                logger.debug(
                    f"Created anchor {anchor_entry.replacement_id} for {entity.entity_type} "
                    f"at {segment.node_id}[{entity.start}:{entity.end}]"
                )

            except Exception as e:
                logger.error(
                    f"Failed to create anchor for entity {entity.entity_type}: {e}"
                )

        logger.info(f"Created {len(anchor_entries)} anchor entries")
        return anchor_entries

    def get_analyzer_diagnostics(self) -> dict[str, Any]:
        """Get diagnostic information about the analyzer configuration.

        Returns:
            Dictionary with analyzer diagnostics
        """
        return self.analyzer.validate_configuration()

    def get_supported_entities(self) -> list[str]:
        """Get list of supported entity types.

        Returns:
            List of entity type names supported by the analyzer
        """
        return self.analyzer.get_supported_entities()
