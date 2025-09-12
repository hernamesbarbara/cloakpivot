# BUG: Document Structure Corrupted When Masking DoclingDocument v1.7.0

## Summary
When masking a DoclingDocument v1.7.0 file, CloakPivot corrupts the document structure by collapsing all text segments into a single malformed segment with overlapping mask placeholders. This occurs despite the DOCLING_V1.7.0_MIGRATION.md claiming "No code changes required in downstream applications."

## Severity: CRITICAL
Data corruption, loss of document structure, unusable output

## Environment
- **cloakpivot**: 0.1.0 (latest from main branch)
- **docpivot**: 1.0.0
- **docling**: 2.52.0 (generates v1.7.0 documents)
- **Python**: 3.x
- **OS**: macOS

## Reproduction Steps

### 1. Generate a DoclingDocument v1.7.0
```python
# pdf2docling.py
from docling.document_converter import DocumentConverter
from pathlib import Path
import json

converter = DocumentConverter()
conv_result = converter.convert("any.pdf")
doc = conv_result.document
output_json = doc.export_to_dict()

with open("output/doc.docling.json", "w") as f:
    f.write(json.dumps(output_json))
```

### 2. Apply PII Masking
```python
# docling2cloaked.py (minimal version)
from docpivot import load_document
from cloakpivot import TextExtractor, MaskingEngine, MaskingPolicy, Strategy, StrategyKind
from presidio_analyzer import AnalyzerEngine

# Load v1.7.0 document
doc = load_document("output/doc.docling.json")

# Extract segments and analyze
extractor = TextExtractor()
segments = extractor.extract_text_segments(doc)  # Returns 22 segments
full_text = extractor.extract_full_text(doc)     # Returns 574 chars

# Find PII entities
analyzer = AnalyzerEngine()
entities = analyzer.analyze(text=full_text, language="en")  # Finds 13 entities

# Create masking policy
policy = MaskingPolicy(
    per_entity={
        "EMAIL_ADDRESS": Strategy(StrategyKind.TEMPLATE, {"template": "[EMAIL]"}),
        "PERSON": Strategy(StrategyKind.TEMPLATE, {"template": "[NAME]"}),
        "DATE_TIME": Strategy(StrategyKind.TEMPLATE, {"template": "[DATE]"}),
    },
    default_strategy=Strategy(StrategyKind.TEMPLATE, {"template": "[REDACTED]"})
)

# Apply masking
engine = MaskingEngine()
result = engine.mask_document(
    document=doc,
    entities=entities,
    policy=policy,
    text_segments=segments
)

# Save corrupted output
masked_json = result.masked_document.export_to_dict()
```

## Expected Behavior

### Input Document (v1.7.0)
```json
{
  "version": "1.7.0",
  "texts": [
    {"text": "---------- Forwarded message ----------", ...},
    {"text": "From: Cameron MacIntyre <cameron@example.com>", ...},
    {"text": "Date: Tuesday, September 2 2025 at 12:19 PM EDT", ...},
    // ... 19 more segments (total 22)
  ]
}
```

### Expected Masked Output
```json
{
  "version": "1.7.0",
  "texts": [
    {"text": "---------- Forwarded message ----------", ...},
    {"text": "From: [NAME] <[EMAIL]>", ...},
    {"text": "Date: [DATE]", ...},
    // ... 19 more properly masked segments (total 22 preserved)
  ]
}
```

## Actual Behavior

### Corrupted Output
```json
{
  "version": "1.7.0",
  "texts": [
    {"text": "---------- Forwarded message ----------[DATE][D[NAME]NA[EMAIL]TE][DATE]", ...}
    // Only 1 segment instead of 22!
  ]
}
```

### Corruption Details
1. **Structure Collapse**: 22 segments → 1 segment
2. **Mask Corruption**: Overlapping placeholders like `[D[NAME]NA[EMAIL]TE]`
3. **Text Truncation**: 574 characters → 73 characters
4. **Data Loss**: Provenance info (bboxes, page numbers) stripped
5. **File Size**: 17KB → 9KB (47% reduction, abnormal for masking)

## Error Logs
```
ERROR - cloakpivot.masking.presidio_adapter - Batch processing failed: 
Invalid analyzer result, start: 564 and end: 574, while text length is only 39.
```

This error suggests the masking engine is trying to apply global offsets to individual segments.

## Root Cause Analysis

### DoclingDocument v1.7.0 Changes
- **Charspan offsets**: Now segment-local (each starts at 0), not global
- **Pages structure**: Object with string keys `{"1": {...}}` vs array

### CloakPivot Issues
1. **TextExtractor** correctly extracts 22 segments with global offsets
2. **MaskingEngine** fails to map global entity positions back to segments
3. **Document reconstruction** collapses all segments into one

The issue appears to be in the document reconstruction phase where masked segments are not properly mapped back to the original document structure.

## Impact
- Cannot mask v1.7.0 documents without data corruption
- Production pipelines broken for users of latest docling
- Migration guide claims "no changes needed" but this is false

## Workaround
None available. Downgrading docling to generate v1.4.0 documents would work but defeats the purpose of updating.

## Suggested Fix

The masking engine needs to:
1. Preserve the original document's segment structure
2. Apply masks to each segment individually using segment-local coordinates
3. Reconstruct the document maintaining all original metadata

Key code areas to investigate:
- `MaskingEngine.mask_document()` - Document reconstruction logic
- `DocumentMasker` or equivalent - Segment handling
- Position mapping between global entities and segment-local text

## Test Case for Fix Validation

```python
def test_v17_segment_preservation():
    # Load v1.7.0 document with 22 segments
    doc = load_document("test_v17.docling.json")
    segments = extractor.extract_text_segments(doc)
    
    # Apply masking
    result = engine.mask_document(doc, entities, policy, segments)
    
    # Verify structure preserved
    assert len(result.masked_document['texts']) == 22  # Not 1!
    assert "[NAME]" in result.masked_document['texts'][1]['text']
    assert "[EMAIL]" in result.masked_document['texts'][1]['text']
    assert "[DATE]" in result.masked_document['texts'][2]['text']
    
    # Verify no corruption
    for text_segment in result.masked_document['texts']:
        assert "[D[NAME]" not in text_segment['text']  # No overlapping masks
```

## Additional Context

The DOCLING_V1.7.0_MIGRATION.md states:
> "CloakPivot's TextExtractor does not rely on DoclingDocument's charspan data"
> "No code changes required in downstream applications"

This is demonstrably false as shown by this bug. The TextExtractor may not rely on charspans, but the document reconstruction clearly fails with v1.7.0 structure.

## Files Attached
- `docling2cloaked.py` - Minimal reproducer script
- `output/email.docling.json` - Sample v1.7.0 input (17KB, 22 segments)
- `output/email.masked.docling.json` - Corrupted output (9KB, 1 segment)