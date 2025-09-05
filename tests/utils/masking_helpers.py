"""Helper utilities for masking tests."""

import re
from typing import Optional

from docling_core.types import DoclingDocument
from docling_core.types.doc.document import TextItem
from presidio_analyzer import AnalyzerEngine, RecognizerResult

from cloakpivot.core.policies import MaskingPolicy
from cloakpivot.document.extractor import TextExtractor, TextSegment
from cloakpivot.masking.engine import MaskingEngine, MaskingResult

# Global variable to hold shared analyzer instance for test reuse
_test_shared_analyzer: Optional[AnalyzerEngine] = None


def set_test_shared_analyzer(analyzer: AnalyzerEngine) -> None:
    """Set the shared analyzer instance for tests to reuse."""
    global _test_shared_analyzer
    _test_shared_analyzer = analyzer


def get_test_shared_analyzer() -> Optional[AnalyzerEngine]:
    """Get the shared analyzer instance if available."""
    return _test_shared_analyzer


def clear_test_shared_analyzer() -> None:
    """Clear the shared analyzer instance."""
    global _test_shared_analyzer
    _test_shared_analyzer = None


def mask_document_with_detection(
    document: DoclingDocument,
    policy: MaskingPolicy,
    analyzer: AnalyzerEngine = None,
    resolve_conflicts: bool = True,
    timing_log: bool = False,
    force_new_analyzer: bool = False,
) -> MaskingResult:
    """
    Convenience function that performs entity detection and masking in one step.

    This is used by tests that expect a simpler API for masking documents.

    Args:
        document: The DoclingDocument to mask
        policy: Masking policy defining strategies per entity type
        analyzer: Optional Presidio AnalyzerEngine (creates default if None)
        resolve_conflicts: Whether to enable conflict resolution (default: True)
        timing_log: Whether to enable timing logs (default: False)
        force_new_analyzer: Force creation of new analyzer even if shared is available

    Returns:
        MaskingResult containing the masked document and CloakMap
    """
    if analyzer is None:
        if force_new_analyzer:
            # Create new analyzer instance (used for performance benchmarks)
            analyzer = AnalyzerEngine()
        else:
            # Try to use shared analyzer first to avoid creating multiple expensive instances
            shared_analyzer = get_test_shared_analyzer()
            if shared_analyzer is not None:
                analyzer = shared_analyzer
            else:
                analyzer = AnalyzerEngine()

    # Extract text segments from document
    extractor = TextExtractor()
    text_segments = extractor.extract_text_segments(document)

    # Set _main_text if not already set (needed for Presidio adapter)
    if not hasattr(document, '_main_text'):
        if text_segments:
            document._main_text = ''.join(seg.text for seg in text_segments)

    # Detect entities in each text segment and adjust positions to global coordinates
    all_entities = []
    for segment in text_segments:
        segment_entities = analyzer.analyze(text=segment.text, language="en")

        # Adjust entity positions from segment-relative to global coordinates
        for entity in segment_entities:
            # Entity positions from analyzer are relative to segment.text
            # We need to adjust them to global document positions
            adjusted_entity = RecognizerResult(
                entity_type=entity.entity_type,
                start=entity.start + segment.start_offset,
                end=entity.end + segment.start_offset,
                score=entity.score,
                analysis_explanation=entity.analysis_explanation,
            )
            all_entities.append(adjusted_entity)

    # Apply masking with conflict resolution enabled
    engine = MaskingEngine(resolve_conflicts=resolve_conflicts)
    return engine.mask_document(
        document=document,
        entities=all_entities,
        policy=policy,
        text_segments=text_segments,
    )


def create_text_segments_from_document(document: DoclingDocument) -> list[TextSegment]:
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
            node_type="TextItem",
        )
        segments.append(segment)

    return segments


def detect_entities_in_text_segments(
    text_segments: list[TextSegment], analyzer: AnalyzerEngine = None
) -> list[RecognizerResult]:
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
        segment_entities = analyzer.analyze(text=segment.text, language="en")
        all_entities.extend(segment_entities)

    return all_entities


class FastRegexDetector:
    """Fast regex-based detector for predictable unit testing.

    This detector uses simple regex patterns to identify PII in text
    for fast, deterministic testing without the overhead of full Presidio analysis.
    """

    def __init__(self):
        """Initialize the fast regex detector with common PII patterns."""
        self.patterns = {
            "PHONE_NUMBER": [
                re.compile(r"\b\d{3}-\d{3}-\d{4}\b"),  # 555-123-4567
                re.compile(r"\b\d{3}\.\d{3}\.\d{4}\b"),  # 555.123.4567
            ],
            "EMAIL_ADDRESS": [
                re.compile(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b"),
            ],
            "SSN": [
                re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),  # 123-45-6789
            ],
        }

    def analyze_text(self, text: str) -> list[RecognizerResult]:
        """Analyze text and return detected PII entities.

        Args:
            text: Text to analyze

        Returns:
            List of RecognizerResult objects for detected entities
        """
        entities = []

        for entity_type, patterns in self.patterns.items():
            for pattern in patterns:
                for match in pattern.finditer(text):
                    entity = RecognizerResult(
                        entity_type=entity_type,
                        start=match.start(),
                        end=match.end(),
                        score=0.95,  # High confidence for regex matches
                    )
                    entities.append(entity)

        return entities


def create_simple_document(text: str) -> DoclingDocument:
    """Create a simple DoclingDocument with a single text section.

    Args:
        text: Text content for the document

    Returns:
        DoclingDocument with single TextItem
    """
    doc = DoclingDocument(name="test_doc")

    text_item = TextItem(text=text, self_ref="#/texts/0", label="text", orig=text)
    doc.texts = [text_item]

    return doc


def create_multi_section_document(sections: list[str]) -> DoclingDocument:
    """Create a DoclingDocument with multiple text sections.

    Args:
        sections: List of text strings, each becomes a separate TextItem

    Returns:
        DoclingDocument with multiple TextItems
    """
    doc = DoclingDocument(name="multi_section_doc")

    text_items = []
    for i, section_text in enumerate(sections):
        text_item = TextItem(
            text=section_text, self_ref=f"#/texts/{i}", label="text", orig=section_text
        )
        text_items.append(text_item)

    doc.texts = text_items

    return doc
