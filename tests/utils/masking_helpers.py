"""Helper utilities for masking tests."""

from typing import List

from docling_core.types import DoclingDocument
from presidio_analyzer import AnalyzerEngine, RecognizerResult

from cloakpivot.core.policies import MaskingPolicy
from cloakpivot.document.extractor import TextExtractor, TextSegment
from cloakpivot.masking.engine import MaskingEngine, MaskingResult


def mask_document_with_detection(
    document: DoclingDocument, 
    policy: MaskingPolicy,
    analyzer: AnalyzerEngine = None
) -> MaskingResult:
    """
    Convenience function that performs entity detection and masking in one step.
    
    This is used by tests that expect a simpler API for masking documents.
    
    Args:
        document: The DoclingDocument to mask
        policy: Masking policy defining strategies per entity type
        analyzer: Optional Presidio AnalyzerEngine (creates default if None)
        
    Returns:
        MaskingResult containing the masked document and CloakMap
    """
    if analyzer is None:
        analyzer = AnalyzerEngine()
    
    # Extract text segments from document
    extractor = TextExtractor()
    text_segments = extractor.extract_text_segments(document)
    
    # Detect entities in each text segment and adjust positions to global coordinates
    all_entities = []
    for segment in text_segments:
        segment_entities = analyzer.analyze(
            text=segment.text,
            language="en"
        )
        
        # Adjust entity positions from segment-relative to global coordinates
        for entity in segment_entities:
            # Entity positions from analyzer are relative to segment.text
            # We need to adjust them to global document positions
            adjusted_entity = RecognizerResult(
                entity_type=entity.entity_type,
                start=entity.start + segment.start_offset,
                end=entity.end + segment.start_offset, 
                score=entity.score,
                analysis_explanation=entity.analysis_explanation
            )
            all_entities.append(adjusted_entity)
    
    # Apply masking with conflict resolution enabled
    engine = MaskingEngine(resolve_conflicts=True)
    return engine.mask_document(
        document=document,
        entities=all_entities,
        policy=policy,
        text_segments=text_segments
    )


def create_text_segments_from_document(document: DoclingDocument) -> List[TextSegment]:
    """
    Create text segments from a DoclingDocument.
    
    This is a simplified version for testing purposes.
    """
    segments = []
    
    for i, text_item in enumerate(document.texts):
        segment = TextSegment(
            node_id=text_item.self_ref or f"#/texts/{i}",
            text=text_item.text,
            start_offset=0,
            end_offset=len(text_item.text),
            node_type="TextItem"
        )
        segments.append(segment)
    
    return segments


def detect_entities_in_text_segments(
    text_segments: List[TextSegment],
    analyzer: AnalyzerEngine = None
) -> List[RecognizerResult]:
    """
    Detect entities in text segments.
    
    Args:
        text_segments: List of text segments to analyze
        analyzer: Optional Presidio AnalyzerEngine
        
    Returns:
        List of detected entities across all segments
    """
    if analyzer is None:
        analyzer = AnalyzerEngine()
    
    all_entities = []
    for segment in text_segments:
        segment_entities = analyzer.analyze(
            text=segment.text,
            language="en"
        )
        all_entities.extend(segment_entities)
    
    return all_entities