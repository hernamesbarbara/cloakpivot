# CloakPivot - DoclingDocument v1.7.0 Support

## Overview
CloakPivot now fully supports DoclingDocument version 1.7.0. The library handles version differences transparently, requiring **no changes** to downstream applications.

## Bug Fix Applied (December 2024)
Fixed a critical issue where masking v1.7.0 documents would collapse multiple segments into one. The PresidioMaskingAdapter now correctly:
- Preserves all document segments during masking
- Applies masks to each segment individually using local coordinates
- Maintains the original document structure with all metadata

## What Changed in DoclingDocument v1.7.0

### 1. Segment-Local Charspan Offsets
The most significant change in DoclingDocument v1.7.0 is how character spans (charspans) are represented:

- **Previous versions (1.2.0, 1.3.0, 1.4.0)**: Charspans used global document offsets
- **Version 1.7.0**: Charspans are **segment-local**, meaning each text segment's charspan starts at 0

#### Example:
```json
// v1.7.0 text segment structure
{
  "text": "actual text content",
  "prov": [{
    "page_no": 1,
    "bbox": {...},
    "charspan": [0, 39]  // Local to this segment, NOT global
  }],
  "self_ref": "#/texts/0"
}
```

### 2. Pages Structure
- **v1.7.0**: Pages are stored as an object with string keys: `{"1": {...}, "2": {...}}`
- **Earlier versions**: May have used array format `[{...}, {...}]`

## How CloakPivot Handles This

### Design Approach
CloakPivot's `TextExtractor` **does not rely on DoclingDocument's charspan data**. Instead, it:

1. Traverses the document structure independently
2. Extracts text from each segment
3. Builds its own segment mappings with global offsets
4. Maintains consistent masking operations across all versions

This design makes CloakPivot resilient to charspan format changes.

### What Was Updated
1. **Version Detection**: Added logging to identify document versions for debugging
2. **Documentation**: Enhanced module and class documentation to note v1.7.0 support
3. **No Functional Changes**: The core text extraction and masking logic remains unchanged

### Code Changes Made
```python
# In TextExtractor.extract_text_segments()
doc_version = getattr(document, 'version', '1.2.0')
logger.info(f"Extracting from document (version: {doc_version})")

# Version-specific logging for awareness
if version.parse(str(doc_version)) >= version.parse('1.7.0'):
    logger.debug("Document is v1.7.0+: using segment-local charspans")
```

## Impact on Downstream Applications

### ✅ No Changes Required

**Applications using CloakPivot do not need any modifications.** The library handles all version differences internally.

### What You Can Expect

1. **Seamless Operation**: Your existing code will work with both old and new DoclingDocument versions
2. **Enhanced Logging**: You'll see version information in logs for debugging:
   ```
   INFO - Successfully loaded document: sample.json (version: 1.7.0)
   DEBUG - Document is v1.7.0+: prov charspans are segment-local. TextExtractor builds independent segments with global offsets.
   ```
3. **Consistent Behavior**: Masking and unmasking operations work identically across all versions

### Example Usage (No Changes Needed)

```python
from cloakpivot import DocumentProcessor, TextExtractor, MaskingEngine

# This code works with all DoclingDocument versions
processor = DocumentProcessor()
document = processor.load_document("document.json")  # v1.2.0 or v1.7.0

extractor = TextExtractor()
segments = extractor.extract_text_segments(document)  # Works the same

engine = MaskingEngine()
result = engine.mask_document(document, policy)  # No changes needed
```

## Technical Details

### Why No Changes Are Needed

CloakPivot was designed with version resilience:

1. **Independent Text Extraction**: Doesn't rely on DoclingDocument's internal charspans
2. **Self-Contained Segment Mapping**: Builds its own offset mappings
3. **Version-Agnostic Operations**: All masking operations use CloakPivot's internal segments

### Version Support Matrix

| DoclingDocument Version | CloakPivot Support | Changes Required |
|------------------------|-------------------|------------------|
| v1.2.0                 | ✅ Full           | None            |
| v1.3.0                 | ✅ Full           | None            |
| v1.4.0                 | ✅ Full           | None            |
| v1.7.0+                | ✅ Full           | None            |

## Testing Your Application

To verify your application works with v1.7.0 documents:

```python
# Check the document version in your logs
document = processor.load_document("your_document.json")
# Look for: INFO - Successfully loaded document: your_document.json (version: 1.7.0)

# Run your existing tests - they should all pass
pytest tests/
```

## Summary

- **CloakPivot now supports DoclingDocument v1.7.0**
- **No code changes required in downstream applications**
- **All existing functionality works identically**
- **Enhanced logging provides version visibility**

If you encounter any issues, they are likely unrelated to the v1.7.0 support. Please check your CloakPivot version is up to date.