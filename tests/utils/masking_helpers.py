"""Helper functions for property-based masking tests."""

import time
from typing import Optional

from docling_core.types import DoclingDocument
from docling_core.types.doc.document import TextItem
from presidio_analyzer import AnalyzerEngine, RecognizerResult

from cloakpivot.core.policies import MaskingPolicy
from cloakpivot.document.extractor import TextExtractor
from cloakpivot.masking.engine import MaskingEngine, MaskingResult


def mask_document_with_detection(
    document: DoclingDocument,
    policy: MaskingPolicy,
    analyzer: Optional[AnalyzerEngine] = None,
    resolve_conflicts: bool = False,
    timing_log: bool = False
) -> MaskingResult:
    """Mask a document with entity detection and policy application.

    This helper is optimized for property-based testing by:
    1. Accepting a reusable analyzer to avoid per-example initialization
    2. Defaulting resolve_conflicts to False to skip expensive normalization
    3. Optional timing instrumentation for performance monitoring

    Args:
        document: Document to mask
        policy: Masking policy to apply
        analyzer: Optional reusable analyzer (creates new one if None)
        resolve_conflicts: Whether to resolve overlapping entities (default False)
        timing_log: Whether to log timing information for slow examples

    Returns:
        MaskingResult with masked document and cloakmap
    """
    start_time = time.perf_counter() if timing_log else 0

    # Use provided analyzer or create new one (fallback for compatibility)
    analyzer = analyzer or AnalyzerEngine()

    # Extract text segments
    extractor = TextExtractor()
    segments = extractor.extract_text_segments(document)

    if timing_log:
        extraction_time = time.perf_counter() - start_time
        if extraction_time > 0.1:  # Log if extraction takes > 100ms
            print(f"Slow text extraction: {extraction_time:.3f}s for {len(segments)} segments")

    # Detect entities in all segments
    all_entities = []
    detection_start = time.perf_counter() if timing_log else 0

    for segment in segments:
        if len(segment.text.strip()) > 1:  # Skip empty/single char segments
            entity_results = analyzer.analyze(segment.text, language="en")

            # Convert to our internal format and adjust offsets
            for result in entity_results:
                # Apply confidence threshold from policy
                threshold = policy.thresholds.get(result.entity_type, 0.5)
                if result.score >= threshold:
                    # Adjust offsets to be relative to segment
                    adjusted_result = result
                    adjusted_result.start = segment.start_offset + result.start
                    adjusted_result.end = segment.start_offset + result.end
                    all_entities.append(adjusted_result)

    if timing_log:
        detection_time = time.perf_counter() - detection_start
        if detection_time > 0.5:  # Log if detection takes > 500ms
            print(f"Slow entity detection: {detection_time:.3f}s for {len(all_entities)} entities")

    # Create masking engine with controlled conflict resolution
    engine = MaskingEngine(resolve_conflicts=resolve_conflicts)

    # Perform masking
    masking_start = time.perf_counter() if timing_log else 0
    result = engine.mask_document(
        document=document,
        entities=all_entities,
        policy=policy,
        text_segments=segments
    )

    if timing_log:
        masking_time = time.perf_counter() - masking_start
        total_time = time.perf_counter() - start_time
        if total_time > 1.0:  # Log if total time > 1 second
            print(f"Slow masking example: total={total_time:.3f}s, "
                  f"masking={masking_time:.3f}s, entities={len(all_entities)}")

    return result


def create_simple_document(text: str, doc_name: str = "test_doc") -> DoclingDocument:
    """Create a simple document with a single text item.

    Optimized for property testing with minimal overhead.
    """
    doc = DoclingDocument(name=doc_name)

    if text.strip():  # Only add non-empty text
        text_item = TextItem(
            text=text,
            self_ref="#/texts/0",
            label="text",
            orig=text
        )
        doc.texts = [text_item]

    return doc


def create_multi_section_document(
    sections: list[str],
    doc_name: str = "multi_section_doc"
) -> DoclingDocument:
    """Create a document with multiple text sections.

    Args:
        sections: List of text sections (empty sections are skipped)
        doc_name: Document name

    Returns:
        Document with multiple TextItems
    """
    doc = DoclingDocument(name=doc_name)

    text_items = []
    for i, section in enumerate(sections):
        if section.strip():  # Skip empty sections
            text_item = TextItem(
                text=section,
                self_ref=f"#/texts/{i}",
                label="text",
                orig=section
            )
            text_items.append(text_item)

    doc.texts = text_items
    return doc


# Minimal regex-based detector for unit-level properties (no Presidio overhead)
class FastRegexDetector:
    """Minimal regex-based entity detector for unit tests.

    This provides deterministic entity detection without Presidio overhead
    for testing masking logic in isolation.
    """

    import re

    PATTERNS = {
        "PHONE_NUMBER": re.compile(r'\b\d{3}-\d{3}-\d{4}\b'),
        "EMAIL_ADDRESS": re.compile(r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b'),
        "US_SSN": re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
    }

    def analyze_text(self, text: str, language: str = "en") -> list[RecognizerResult]:
        """Analyze text and return mock RecognizerResults."""
        results = []
        for entity_type, pattern in self.PATTERNS.items():
            for match in pattern.finditer(text):
                result = RecognizerResult(
                    entity_type=entity_type,
                    start=match.start(),
                    end=match.end(),
                    score=0.9  # Fixed high confidence
                )
                results.append(result)

        return results
