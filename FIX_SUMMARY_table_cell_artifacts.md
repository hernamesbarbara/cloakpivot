# Fix Summary: Table Cell Masking Artifacts

## Issue Resolved
Fixed table cell masking artifacts where dates in table cells were being incorrectly concatenated with text from adjacent cells during PII masking.

## Root Cause
When text segments (including table cells) were joined with separators (`\n\n`) to form the full document text, Presidio's entity detection would sometimes identify entities that spanned across these segment boundaries. This caused the masked value to include text from multiple table cells.

### Example of the Bug:
- Original cell: `1958-04-21`
- Masked result: `1826-13-78  Frzylh  Bwmpni` (included names from adjacent cells)

## Solution Implemented

### 1. Added Boundary Validation Method
Created `_validate_entities_against_boundaries()` in `presidio_adapter.py` that:
- Detects when entities span across segment boundaries
- Truncates entities to fit within their originating segment
- Prevents text from multiple segments being concatenated

### 2. Integration into Masking Pipeline
Updated the masking flow to call the boundary validation after standard entity validation (Step 4b in the pipeline).

### 3. Smart Truncation Logic
Instead of discarding entities that cross boundaries entirely, the fix:
- Truncates entities at segment boundaries
- Preserves the valid portion of the entity
- Ensures all legitimate PII is still masked

## Files Modified
1. `cloakpivot/masking/presidio_adapter.py`
   - Added `_validate_entities_against_boundaries()` method
   - Updated `mask_document_with_presidio()` to call the new validation

## Testing
1. Created debug script `examples/debug_table_artifacts.py` to reproduce the issue
2. Added comprehensive unit tests in `tests/unit/test_table_cell_boundary_fix.py`
3. Verified fix with `email.pdf` test document
4. All existing tests pass (80 tests)

## Results
- Table cells now mask cleanly without artifacts
- `export_to_markdown()` continues to work properly
- Table formatting preserved in markdown export
- No regression in existing functionality

## Before and After

### Before Fix:
```markdown
| surname   | first name   | birthdate                  |
|-----------|--------------|----------------------------|
| White     | Johnson      | 7970-36-76  Zfkcei  Hyewck |
```

### After Fix:
```markdown
| surname   | first name   | birthdate   |
|-----------|--------------|-------------|
| White     | Johnson      | 6856-13-12  |
```

## Technical Details
The fix works by:
1. Checking if entity text contains segment separator (`\n\n`)
2. Finding the separator position and truncating the entity
3. Validating entities are fully contained within single segments
4. Adjusting entity boundaries when they cross segments

This ensures that each table cell is masked independently without interference from adjacent cells.