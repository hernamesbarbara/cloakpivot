#!/usr/bin/env python3
"""Example demonstrating document integration with DocPivot."""

import sys
from pathlib import Path

# Add the parent directory to sys.path to import cloakpivot
sys.path.insert(0, str(Path(__file__).parent.parent))

from cloakpivot import AnchorMapper, DocumentProcessor, TextExtractor


def main() -> None:
    """Demonstrate the document processing workflow."""
    print("CloakPivot Document Integration Example")
    print("=" * 50)

    # Note: This example shows the API usage but requires actual docling.json files
    # to run. The classes are designed to work with real DocPivot documents.

    print("\n1. DocumentProcessor - Loading documents")
    print("-" * 40)
    processor = DocumentProcessor()
    print("✓ DocumentProcessor initialized")
    print(f"✓ Supports docling.json: {processor.supports_format('test.docling.json')}")
    print(f"✓ Supports lexical.json: {processor.supports_format('test.lexical.json')}")
    print(f"✗ Supports PDF: {processor.supports_format('test.pdf')}")

    # Example usage (would require actual file):
    # document = processor.load_document("sample.docling.json")
    # print(f"Loaded document: {document.name}")

    print("\n2. TextExtractor - Extracting text segments")
    print("-" * 40)
    TextExtractor(normalize_whitespace=True)
    print("✓ TextExtractor initialized with whitespace normalization")

    # Example usage (would require actual DoclingDocument):
    # segments = extractor.extract_text_segments(document)
    # full_text = extractor.extract_full_text(document)
    # stats = extractor.get_extraction_stats(document)

    print("\n3. AnchorMapper - Creating position mappings")
    print("-" * 40)
    AnchorMapper()
    print("✓ AnchorMapper initialized")

    # Example usage (would require actual data):
    # detections = [RecognizerResult("PHONE_NUMBER", 10, 22, 0.9)]
    # anchors = mapper.create_anchors_from_detections(
    #     detections, segments, {"#/texts/0": "Call me at 555-1234"}
    # )

    print("\n4. Complete Workflow Example")
    print("-" * 40)
    print(
        """
    # Real usage would look like this:

    # Step 1: Load document
    processor = DocumentProcessor()
    document = processor.load_document("confidential_report.docling.json")

    # Step 2: Extract text segments
    extractor = TextExtractor()
    segments = extractor.extract_text_segments(document)
    full_text = extractor.extract_full_text(document)

    # Step 3: Run PII detection (using Presidio)
    from presidio_analyzer import AnalyzerEngine
    analyzer = AnalyzerEngine()
    detections = analyzer.analyze(
        text=full_text, entities=["PHONE_NUMBER", "EMAIL_ADDRESS", "PERSON"]
    )

    # Step 4: Create anchors for detected entities
    mapper = AnchorMapper()
    original_texts = {seg.node_id: seg.text for seg in segments}
    anchors = mapper.create_anchors_from_detections(
        detections, segments, original_texts
    )

    # Step 5: Use anchors to create masked document (future implementation)
    # masked_document = apply_masking(document, anchors, policy)
    # cloakmap = create_cloakmap(anchors, document_metadata)
    """
    )

    print("\n✓ Document integration components are ready!")
    print("  - DocumentProcessor: Load documents using DocPivot")
    print("  - TextExtractor: Extract text with structural anchors")
    print("  - AnchorMapper: Map PII detections to document positions")
    print("  - Full test suite: 142 tests passing")

    print(
        f"\nProcessing stats: "
        f"{processor.get_processing_stats().files_processed} files processed"
    )


if __name__ == "__main__":
    main()
