#!/usr/bin/env python
# tmp_pdf_to_masked_json.py
"""
Read a PDF with some PII data. Use Docling to convert it into Docling Json. Finally, use CloakPivot to mask the PII.
"""
import os
import sys
import html
import json
from pathlib import Path
from docling.document_converter import DocumentConverter
from docling_core.types import DoclingDocument
from presidio_analyzer import AnalyzerEngine

# Import CloakPivot components
from cloakpivot import (
    MaskingEngine,
    TextExtractor,
    MaskingPolicy,
    CONSERVATIVE_POLICY
)
from cloakpivot.core.strategies import Strategy, StrategyKind

DATA_DIR = Path("data")
PDF_DIR  = DATA_DIR / Path("pdf")
JSON_DIR = DATA_DIR / Path("json")

# Ensure directories exist
PDF_DIR.mkdir(parents=True, exist_ok=True)
JSON_DIR.mkdir(parents=True, exist_ok=True)

PDF_UNMASKED = PDF_DIR / "email_unmasked.pdf"

print(f"Reading PDF: {PDF_UNMASKED}\n")
converter = DocumentConverter()
dl_doc = converter.convert(PDF_UNMASKED).document

print(f"PDF Contents: \n")
print(html.unescape(dl_doc.export_to_markdown())+"\n")

OUTPUT_DOCLING_JSON_UNMASKED = JSON_DIR / "email_unmasked.docling.json"
OUTPUT_DOCLING_JSON_MASKED = JSON_DIR / "email_masked.docling.json"
OUTPUT_PDF_MASKED = PDF_DIR / "email_masked.pdf"

# Step 1: Save unmasked docling JSON
print(f"Saving unmasked docling JSON to: {OUTPUT_DOCLING_JSON_UNMASKED}")
with open(OUTPUT_DOCLING_JSON_UNMASKED, "w") as f:
    json.dump(dl_doc.model_dump(mode='json', by_alias=True), f, indent=2)
print(f"✓ Saved unmasked docling JSON\n")

# Step 2: Detect PII entities using Presidio
print("Detecting PII entities...")
analyzer = AnalyzerEngine()
extractor = TextExtractor()

# Extract text segments from document
text_segments = extractor.extract_text_segments(dl_doc)
print(f"  Extracted {len(text_segments)} text segments")

# Detect entities in each text segment and adjust positions to global coordinates
all_entities = []
entity_types_found = set()

for segment in text_segments:
    segment_entities = analyzer.analyze(text=segment.text, language="en")
    
    # Adjust entity positions from segment-relative to global coordinates
    for entity in segment_entities:
        from presidio_analyzer import RecognizerResult
        adjusted_entity = RecognizerResult(
            entity_type=entity.entity_type,
            start=entity.start + segment.start_offset,
            end=entity.end + segment.start_offset,
            score=entity.score,
            analysis_explanation=entity.analysis_explanation,
        )
        all_entities.append(adjusted_entity)
        entity_types_found.add(entity.entity_type)

print(f"  Found {len(all_entities)} PII entities")
if entity_types_found:
    print(f"  Entity types: {', '.join(sorted(entity_types_found))}")
print()

# Step 3: Create masking policy
print("Creating masking policy...")
# Use conservative policy as base but customize for specific entity types
policy = MaskingPolicy(
    strategies={
        "EMAIL_ADDRESS": Strategy(kind=StrategyKind.REDACT, value="[EMAIL]"),
        "PERSON": Strategy(kind=StrategyKind.REDACT, value="[NAME]"),
        "PHONE_NUMBER": Strategy(kind=StrategyKind.REDACT, value="[PHONE]"),
        "LOCATION": Strategy(kind=StrategyKind.REDACT, value="[LOCATION]"),
        "DATE_TIME": Strategy(kind=StrategyKind.REDACT, value="[DATE]"),
        "URL": Strategy(kind=StrategyKind.REDACT, value="[URL]"),
        # Add more entity types as needed
    },
    default_strategy=Strategy(kind=StrategyKind.REDACT, value="[REDACTED]")
)
print(f"✓ Policy created with {len(policy.strategies)} specific entity strategies\n")

# Step 4: Mask the document
print("Masking document...")
masking_engine = MaskingEngine(resolve_conflicts=True)
mask_result = masking_engine.mask_document(
    document=dl_doc,
    entities=all_entities,
    policy=policy,
    text_segments=text_segments
)

print(f"✓ Masking complete")
print(f"  Entries in CloakMap: {len(mask_result.cloakmap.entries)}")
print(f"  Processing time: {mask_result.performance_metrics.total_time_ms:.2f}ms\n")

# Step 5: Save masked docling JSON
print(f"Saving masked docling JSON to: {OUTPUT_DOCLING_JSON_MASKED}")
with open(OUTPUT_DOCLING_JSON_MASKED, "w") as f:
    json.dump(mask_result.masked_document.model_dump(mode='json', by_alias=True), f, indent=2)
print(f"✓ Saved masked docling JSON\n")

# Step 6: Convert masked docling JSON back to PDF
print(f"Converting masked document to PDF...")
# Docling doesn't have direct JSON to PDF conversion, so we'll export to markdown
# and then use a markdown to PDF converter if available, or just save the markdown
masked_markdown = mask_result.masked_document.export_to_markdown()

# For now, let's save the masked content as a text file since direct PDF conversion
# would require additional libraries like reportlab or weasyprint
OUTPUT_MASKED_TEXT = PDF_DIR / "email_masked.txt"
with open(OUTPUT_MASKED_TEXT, "w") as f:
    f.write(html.unescape(masked_markdown))
print(f"✓ Saved masked content as text: {OUTPUT_MASKED_TEXT}")

# Note: To convert to PDF, you would need a library like:
# - reportlab: for programmatic PDF creation
# - weasyprint: for HTML/CSS to PDF
# - pdfkit: wrapper for wkhtmltopdf
# Since these aren't in the current dependencies, we save as text for now

print("\n" + "="*50)
print("Summary:")
print(f"  ✓ Input PDF: {PDF_UNMASKED}")
print(f"  ✓ Unmasked JSON: {OUTPUT_DOCLING_JSON_UNMASKED}")
print(f"  ✓ Masked JSON: {OUTPUT_DOCLING_JSON_MASKED}")
print(f"  ✓ Masked text: {OUTPUT_MASKED_TEXT}")
print(f"  Total PII entities masked: {len(all_entities)}")

# Also save the CloakMap for potential unmasking later
OUTPUT_CLOAKMAP = JSON_DIR / "email_cloakmap.json"
with open(OUTPUT_CLOAKMAP, "w") as f:
    json.dump(mask_result.cloakmap.model_dump(mode='json'), f, indent=2)
print(f"  ✓ CloakMap saved: {OUTPUT_CLOAKMAP} (for reversible unmasking)")
print("="*50)
