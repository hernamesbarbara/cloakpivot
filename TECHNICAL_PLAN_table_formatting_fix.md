# Technical Plan: Fix DoclingDocument export_to_markdown() After Masking

## Executive Summary

After thorough investigation, the core bug reported in `BUG_table_formatting_lost_when_masking.md` appears to be **partially resolved**. The `export_to_markdown()` method now returns content after masking, and table formatting is preserved. However, there are still issues that need addressing for a complete fix.

## Current State Analysis

### What's Working
1. **Document Structure Preservation**: The fix applied in `examples/ISSUE_MASKING_BUG.md` successfully preserves document structure by serializing/deserializing the document:
   ```python
   doc_dict = json.loads(document.model_dump_json())
   masked_document = DoclingDocument.model_validate(doc_dict)
   ```

2. **Table Masking**: Tables ARE being masked via the `_update_table_cells` method in `presidio_adapter.py`

3. **Markdown Export**: `export_to_markdown()` returns non-empty content with proper formatting

4. **Body Children Preservation**: The `body.children` array is maintained, keeping document hierarchy intact

### Remaining Issues

1. **Table Cell Masking Anomalies**: When masking dates in table cells, extra text appears to be appended:
   - Original: `1958-04-21`
   - Masked: `1826-13-78  Frzylh  Bwmpni` (should be just the date replacement)

2. **Fallback Rendering in pii-cloak**: The script still uses a fallback mechanism that may not be necessary

3. **Deprecation Warning**: `TableItem.export_to_markdown()` shows deprecation warning when called without `doc` argument

## Root Cause Analysis

### Primary Issue (Resolved)
The original issue was that creating a new DoclingDocument with partial fields lost internal state needed for `export_to_markdown()`. This has been fixed by:
1. Serializing the complete document to JSON
2. Deserializing to create a full copy with all internal state
3. Updating texts in-place to maintain references

### Secondary Issue (Active)
The table cell masking appears to have an edge case where:
1. Multiple entities might be detected in the same cell
2. The masking replacement logic might be applying replacements incorrectly
3. Text segments from tables might be processed multiple times

### Current Implementation Analysis

Our code in `presidio_adapter.py` currently:
1. **Does NOT use `DoclingDocument.add_table_cell()`** - We directly modify existing cells via `cell.text = masked_value`
2. **Creates new TableCell objects when needed** - Lines 1125-1137 import and instantiate `TableCell` directly
3. **Appends cells directly to `table_data.table_cells`** - Line 1134: `table_data.table_cells.append(new_cell)`

### Should We Use Docling's `add_table_cell` API?

**No, we should NOT use `add_table_cell()` for our use case. Here's why:**

1. **Purpose Mismatch**: `add_table_cell()` is designed for building documents from scratch or adding new cells. We're modifying existing cells' text content.

2. **Validation Overhead**: The method includes parent validation logic (checking `RichTableCell` references) that isn't relevant for text masking.

3. **Direct Modification Works**: Our current approach of directly setting `cell.text = masked_value` is appropriate for in-place text updates.

4. **Preserve Structure**: We need to maintain the exact same table structure, just with masked text. Creating new cells risks disrupting the table layout.

## Technical Solution Plan

### Phase 1: Diagnose Table Cell Masking Issue (Priority: CRITICAL)

**Objective**: Understand why extra text appears in masked table cells

**Root Cause Identified**: The artifact `"1826-13-78  Frzylh  Bwmpni"` suggests that:
- The date `1958-04-21` is being masked to `1826-13-78`
- But additional text `"Frzylh  Bwmpni"` is being appended
- This looks like person names from adjacent cells being concatenated

**Hypothesis**: Table cells might be getting their text from multiple text segments, causing concatenation

**Tasks**:
1. Add detailed logging to `_update_table_cells` method
2. Trace entity detection in table cells
3. Verify that table cell text segments are processed only once
4. Check if multiple entities overlap in table cells
5. Debug the anchor_entries to see if multiple anchors map to the same cell

**Implementation Location**: `cloakpivot/masking/presidio_adapter.py`

### Phase 2: Fix Table Cell Masking Logic

**Objective**: Ensure clean, accurate masking of table cell contents

**Proposed Fix**:
```python
def _update_table_cells(self, masked_document, text_segments, anchor_entries):
    """Update table cells with masked values based on anchors."""

    # Track which cells have been updated to prevent double-masking
    updated_cells = set()

    for table_item in masked_document.tables:
        # ... existing logic ...

        for cell in table_data.table_cells:
            cell_key = (table_id, row_idx, col_idx)
            if cell_key not in updated_cells:
                # Apply masking only once per cell
                cell.text = masked_text
                updated_cells.add(cell_key)
```

### Phase 3: Optimize Text Segment Processing

**Objective**: Ensure text segments from tables are handled correctly

**Tasks**:
1. Verify that table cell segments have correct offsets
2. Ensure entity positions align with segment boundaries
3. Prevent overlapping replacements in the same text region

**Implementation**:
- Add validation in `_apply_masks_to_text` to detect overlapping replacements
- Implement conflict resolution for overlapping entities in table cells

### Phase 4: Update pii-cloak Script

**Objective**: Remove unnecessary fallback logic now that export works

**Changes**:
```python
def render_markdown(doc, *, title: str, entities_found: int, entities_masked: int) -> str:
    """Return markdown from DoclingDocument."""
    md = doc.export_to_markdown()

    # Add metadata header if needed
    header = f"# {title}\n\n**Entities Found:** {entities_found}\n**Entities Masked:** {entities_masked}\n\n"

    return header + md if md else header + "# Document could not be rendered"
```

### Phase 5: Add Comprehensive Tests

**Objective**: Ensure robustness of the fix

**Test Cases**:

1. **Test Table Export Preservation**:
```python
def test_table_export_after_masking():
    """Verify tables export correctly after masking."""
    doc = load_test_document_with_table()
    masked_doc = mask_document(doc)

    assert masked_doc.export_to_markdown()
    assert "|" in masked_doc.export_to_markdown()
    assert len(masked_doc.tables) == len(doc.tables)
```

2. **Test Table Cell Masking Accuracy**:
```python
def test_table_cell_masking_clean():
    """Ensure table cells are masked without artifacts."""
    doc = create_doc_with_date_table()
    masked_doc = mask_with_surrogate(doc)

    for table in masked_doc.tables:
        for cell in table.data.table_cells:
            # No cell should have double-spaced replacements
            assert "  " not in cell.text
```

3. **Test Complex Table Structures**:
```python
def test_nested_table_preservation():
    """Test tables with complex structures."""
    # Test with merged cells, nested tables, etc.
```

## Implementation Priority

### Critical (Immediate)
1. Fix table cell masking anomalies (Phase 2)
2. Add validation to prevent double-masking

### High Priority (Next Sprint)
1. Optimize text segment processing (Phase 3)
2. Add comprehensive test suite (Phase 5)

### Medium Priority
1. Update pii-cloak script (Phase 4)
2. Handle deprecation warnings

## Risk Mitigation

### Potential Risks
1. **Breaking Changes**: Modifying masking logic could affect existing functionality
   - **Mitigation**: Extensive testing with existing test suite

2. **Performance Impact**: Additional validation might slow processing
   - **Mitigation**: Use efficient data structures (sets for tracking)

3. **DoclingDocument Version Changes**: Future versions might break compatibility
   - **Mitigation**: Pin DoclingDocument version, add version checks

## Success Criteria

1. ✅ `export_to_markdown()` returns valid markdown after masking
2. ✅ Table structure and formatting preserved
3. ✅ Table cells masked cleanly without artifacts
4. ✅ All existing tests pass
5. ✅ New table-specific tests pass
6. ✅ Performance impact < 5% on typical documents

## Testing Strategy

### Unit Tests
- Test individual methods: `_update_table_cells`, `_apply_masks_to_text`
- Test edge cases: empty tables, single-cell tables, nested tables

### Integration Tests
- Full pipeline: PDF → DoclingDocument → Mask → Export
- Various document types: legal, medical, financial
- Different masking strategies: REDACT, SURROGATE, TEMPLATE

### Regression Tests
- Ensure existing functionality unchanged
- Verify backward compatibility

## Monitoring and Validation

### Metrics to Track
1. Markdown export success rate
2. Table cell masking accuracy
3. Processing time per document
4. Memory usage during masking

### Validation Steps
1. Manual review of masked documents
2. Automated comparison of table structures
3. Character-level diff analysis of exports

## Conclusion

The core issue of `export_to_markdown()` returning empty has been resolved through document serialization/deserialization. However, table cell masking needs refinement to eliminate artifacts.

### Key Findings:

1. **Docling's `add_table_cell()` API is NOT needed** - Our direct modification approach is correct for masking use cases
2. **The main bug is RESOLVED** - Documents export to markdown successfully after masking
3. **Minor artifact issue remains** - Table cells show concatenated text that needs investigation

### The proposed solution focuses on:

1. Preventing double-masking of table cells
2. Ensuring clean text replacement without concatenation
3. Maintaining document structure integrity
4. Continuing to use direct cell.text modification (not add_table_cell)

With these fixes, CloakPivot will provide reliable, clean masking of DoclingDocuments while preserving all formatting and export capabilities.