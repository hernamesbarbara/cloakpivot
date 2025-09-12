# DoclingDocument v1.7.0 Migration Guide

## Overview
DocPivot now supports DoclingDocument version 1.7.0. This version introduces structural changes that may affect downstream consumers of the library, particularly those performing text extraction or masking operations.

## Key Changes in v1.7.0

### 1. Segment-Local Charspan Offsets
**CRITICAL CHANGE**: The most significant change in v1.7.0 is how character spans (charspans) are represented in text segments.

- **Previous versions (1.2.0, 1.3.0, 1.4.0)**: Charspans may have used global document offsets
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

### 2. Pages Structure (Already Handled)
- **v1.7.0**: Pages are stored as an object with string keys: `{"1": {...}, "2": {...}}`
- **Earlier versions**: May have used array format `[{...}, {...}]`
- **Note**: DocPivot handles this transparently through the DoclingDocument model

## Impact on Downstream Consumers

### For Text Extraction/Masking Libraries (e.g., cloakpivot)
If your library performs text extraction or masking operations, you'll need to update your offset calculation logic:

#### Before (Global Offsets):
```python
def extract_text_segments(document):
    segments = []
    for text_item in document.texts:
        # Assuming charspan provides global offsets
        start = text_item['prov'][0]['charspan'][0]
        end = text_item['prov'][0]['charspan'][1]
        segments.append(TextSegment(start, end, text_item['text']))
    return segments
```

#### After (v1.7.0 with Segment-Local Offsets):
```python
def extract_text_segments_v17(document):
    segments = []
    global_offset = 0
    
    for idx, text_item in enumerate(document.texts):
        segment_text = text_item.get('text', '')
        
        # Create segment with GLOBAL offsets
        segment = TextSegment(
            start_offset=global_offset,  # Global position
            end_offset=global_offset + len(segment_text),
            text=segment_text
        )
        segments.append(segment)
        
        # Update global offset for next segment
        global_offset += len(segment_text)
        # Add separator if segments are meant to be joined
        if idx < len(document.texts) - 1:
            global_offset += 1  # Account for space/newline between segments
    
    return segments
```

### For Masking Position Mapping:
When applying masks based on global entity positions (e.g., from Presidio), you need to map them back to segment-local coordinates:

```python
def apply_masks_v17(document, entities, segments):
    # Build segment offset map
    segment_map = []
    current_pos = 0
    for seg in segments:
        segment_map.append({
            'start': current_pos,
            'end': current_pos + len(seg.text),
            'segment': seg
        })
        current_pos = segment_map[-1]['end'] + 1  # +1 for separator
    
    # Group entities by segment
    for entity in entities:
        segment_idx = find_segment_for_position(entity.start, segment_map)
        if segment_idx is not None:
            # Convert global position to segment-local position
            local_start = entity.start - segment_map[segment_idx]['start']
            local_end = entity.end - segment_map[segment_idx]['start']
            apply_mask_to_segment(segment_idx, local_start, local_end, mask_text)
```

## Version Detection

DocPivot logs the document version during loading. You can use this information to apply version-specific handling:

```python
# In DocPivot logs, you'll see:
# INFO - Loaded DoclingDocument v1.7.0 from file.json. Note: This version uses segment-local charspans (each segment starts at 0).
```

## Backward Compatibility

- DocPivot maintains full backward compatibility with versions 1.2.0, 1.3.0, and 1.4.0
- No changes are required for applications that don't perform text offset calculations
- The DoclingDocument model handles structural differences transparently

## Recommendations

1. **Check Document Version**: Always check the document version before applying offset calculations
2. **Test with v1.7.0 Documents**: Ensure your text extraction/masking logic works with both old and new formats
3. **Add Version-Specific Handlers**: Implement separate handlers for different document versions if needed
4. **Preserve Document Structure**: When reconstructing documents after processing, maintain the original segment structure

## Support

If you encounter issues with v1.7.0 documents, please:
1. Check that you're handling segment-local offsets correctly
2. Verify that text segment boundaries are preserved
3. Ensure provenance information (bounding boxes, page positions) is maintained

For additional support, please open an issue in the DocPivot repository with:
- The document version you're processing
- A description of the issue
- Sample code demonstrating the problem