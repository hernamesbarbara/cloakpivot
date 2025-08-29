"""Coverage analysis for PII detection and masking effectiveness."""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from ..core.anchors import AnchorEntry
from ..document.extractor import TextSegment


@dataclass
class DocumentSection:
    """
    Coverage metrics for a specific document section type.
    
    Tracks PII detection and masking coverage within specific
    document sections like headings, paragraphs, tables, etc.
    """
    
    section_type: str
    total_segments: int
    segments_with_entities: int
    entity_count: int
    
    @property
    def coverage_rate(self) -> float:
        """Calculate coverage rate as segments with entities / total segments."""
        if self.total_segments == 0:
            return 1.0  # Consider empty sections as 100% covered
        return self.segments_with_entities / self.total_segments
    
    @property
    def average_entities_per_segment(self) -> float:
        """Calculate average entities per segment in this section type."""
        if self.total_segments == 0:
            return 0.0
        return self.entity_count / self.total_segments
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "section_type": self.section_type,
            "total_segments": self.total_segments,
            "segments_with_entities": self.segments_with_entities,
            "entity_count": self.entity_count,
            "coverage_rate": self.coverage_rate,
            "average_entities_per_segment": self.average_entities_per_segment
        }


@dataclass
class CoverageMetrics:
    """
    Comprehensive coverage metrics for document PII detection and masking.
    
    Provides detailed analysis of how effectively PII was detected and masked
    across different document sections and entity types.
    """
    
    total_segments: int = 0
    segments_with_entities: int = 0
    overall_coverage_rate: float = 0.0
    section_coverage: list[DocumentSection] = field(default_factory=list)
    entity_distribution: dict[str, int] = field(default_factory=dict)
    entity_density: float = 0.0
    coverage_gaps: list[dict[str, Any]] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: "")
    
    @property
    def total_entities(self) -> int:
        """Calculate total entities from distribution."""
        return sum(self.entity_distribution.values())
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "total_segments": self.total_segments,
            "segments_with_entities": self.segments_with_entities,
            "overall_coverage_rate": self.overall_coverage_rate,
            "total_entities": self.total_entities,
            "entity_density": self.entity_density,
            "entity_distribution": self.entity_distribution,
            "section_coverage": [section.to_dict() for section in self.section_coverage],
            "coverage_gaps": self.coverage_gaps,
            "timestamp": self.timestamp
        }


class CoverageAnalyzer:
    """
    Analyzes PII detection and masking coverage across documents.
    
    Provides comprehensive analysis of how effectively PII entities
    are detected and masked across different document sections,
    identifying gaps and providing optimization recommendations.
    """
    
    def __init__(self) -> None:
        """Initialize the coverage analyzer."""
        pass
    
    def analyze_document_coverage(
        self,
        text_segments: list[TextSegment],
        anchor_entries: list[AnchorEntry]
    ) -> CoverageMetrics:
        """
        Analyze PII coverage across a document's text segments.
        
        Args:
            text_segments: List of text segments from the document
            anchor_entries: List of anchor entries from masking operation
            
        Returns:
            CoverageMetrics with comprehensive coverage analysis
        """
        if not text_segments:
            return CoverageMetrics(overall_coverage_rate=1.0)
        
        # Group anchor entries by node_id for quick lookup
        entities_by_node: dict[str, list[AnchorEntry]] = defaultdict(list)
        for anchor in anchor_entries:
            entities_by_node[anchor.node_id].append(anchor)
        
        # Calculate basic coverage metrics
        segments_with_entities = sum(
            1 for segment in text_segments 
            if segment.node_id in entities_by_node
        )
        
        overall_coverage_rate = segments_with_entities / len(text_segments)
        
        # Analyze section-by-section coverage
        section_coverage = self._analyze_section_coverage(text_segments, entities_by_node)
        
        # Calculate entity distribution
        entity_distribution = self._calculate_entity_distribution(anchor_entries)
        
        # Calculate entity density
        total_entities = len(anchor_entries)
        entity_density = total_entities / len(text_segments)
        
        # Identify coverage gaps
        coverage_gaps = self._identify_coverage_gaps(text_segments, entities_by_node)
        
        return CoverageMetrics(
            total_segments=len(text_segments),
            segments_with_entities=segments_with_entities,
            overall_coverage_rate=overall_coverage_rate,
            section_coverage=section_coverage,
            entity_distribution=entity_distribution,
            entity_density=entity_density,
            coverage_gaps=coverage_gaps
        )
    
    def generate_recommendations(self, metrics: CoverageMetrics) -> list[str]:
        """
        Generate optimization recommendations based on coverage analysis.
        
        Args:
            metrics: Coverage metrics to analyze
            
        Returns:
            List of actionable recommendations
        """
        recommendations = []
        
        # Check overall coverage
        if metrics.overall_coverage_rate < 0.3:
            recommendations.append(
                "Consider reviewing PII detection configuration - low overall coverage detected"
            )
        
        # Check for coverage gaps
        if len(metrics.coverage_gaps) > 0:
            gap_count = len(metrics.coverage_gaps)
            recommendations.append(
                f"Review {gap_count} uncovered segments that might contain missed PII entities"
            )
        
        # Check section-specific issues
        for section in metrics.section_coverage:
            if section.coverage_rate < 0.1 and section.total_segments > 1:
                recommendations.append(
                    f"Low PII detection in {section.section_type} sections - "
                    f"consider tuning policies for this content type"
                )
        
        # Check entity distribution
        if metrics.total_entities > 0:
            dominant_entity = max(metrics.entity_distribution.items(), key=lambda x: x[1])
            if dominant_entity[1] > metrics.total_entities * 0.8:
                recommendations.append(
                    f"High concentration of {dominant_entity[0]} entities - "
                    f"ensure detection coverage for other entity types"
                )
        
        # Check entity density
        if metrics.entity_density < 0.1 and metrics.total_segments > 5:
            recommendations.append(
                "Low entity density detected - document may require more comprehensive PII detection"
            )
        
        return recommendations
    
    def _analyze_section_coverage(
        self,
        text_segments: list[TextSegment],
        entities_by_node: dict[str, list[AnchorEntry]]
    ) -> list[DocumentSection]:
        """Analyze coverage by document section type."""
        section_stats: dict[str, dict[str, int]] = defaultdict(lambda: {
            "total_segments": 0,
            "segments_with_entities": 0,
            "entity_count": 0
        })
        
        for segment in text_segments:
            section_type = getattr(segment, 'node_type', 'unknown')
            stats = section_stats[section_type]
            
            stats["total_segments"] += 1
            
            if segment.node_id in entities_by_node:
                stats["segments_with_entities"] += 1
                stats["entity_count"] += len(entities_by_node[segment.node_id])
        
        return [
            DocumentSection(
                section_type=section_type,
                total_segments=stats["total_segments"],
                segments_with_entities=stats["segments_with_entities"],
                entity_count=stats["entity_count"]
            )
            for section_type, stats in section_stats.items()
        ]
    
    def _calculate_entity_distribution(self, anchor_entries: list[AnchorEntry]) -> dict[str, int]:
        """Calculate distribution of entity types."""
        distribution: dict[str, int] = defaultdict(int)
        
        for anchor in anchor_entries:
            distribution[anchor.entity_type] += 1
            
        return dict(distribution)
    
    def _identify_coverage_gaps(
        self,
        text_segments: list[TextSegment],
        entities_by_node: dict[str, list[AnchorEntry]]
    ) -> list[dict[str, Any]]:
        """Identify segments without any detected entities."""
        gaps = []
        
        for segment in text_segments:
            if segment.node_id not in entities_by_node:
                gaps.append({
                    "node_id": segment.node_id,
                    "type": getattr(segment, 'node_type', 'unknown'),
                    "text_preview": segment.text[:100] + "..." if len(segment.text) > 100 else segment.text,
                    "start_offset": segment.start_offset,
                    "end_offset": segment.end_offset
                })
        
        return gaps