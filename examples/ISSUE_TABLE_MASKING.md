# Issue: Table Data Not Being Masked in Markdown Output

## Problem Description
When masking a document containing tables, the PII in table cells is NOT properly masked in the final markdown output, even though some dates in the table text items are masked in the JSON. Names and dates in the table remain visible in their original form.

### Example
**Original table in markdown:**
```markdown
| surname   | first name   | birthdate   |
|-----------|--------------|-------------|
| White     | Johnson      | 1958-04-21  |
| Borden    | Ashley       | 1944-12-22  |
| Green     | Marjorie     | 1958-04-21  |
```

**Expected masked output:**
```markdown
| surname   | first name   | birthdate   |
|-----------|--------------|-------------|
| [NAME]    | [NAME]       | [DATE]      |
| [NAME]    | [NAME]       | [DATE]      |
| [NAME]    | [NAME]       | [DATE]      |
```

**Actual masked output:**
```markdown
| surname   | first name   | birthdate   |
|-----------|--------------|-------------|
| White     | Johnson      | 1958-04-21  |  ← NOT MASKED!
| Borden    | Ashley       | 1944-12-22  |  ← NOT MASKED!
| Green     | Marjorie     | 1958-04-21  |  ← NOT MASKED!
```

## Root Causes

### 1. Dual Storage of Table Data
Table content is stored in TWO places in DoclingDocument:
- `document.texts[]` array - Individual text items (dates ARE masked here)
- `document.tables[].data.grid[][]` - Table grid structure (NOT masked)

When exporting to markdown, DoclingDocument uses the `grid` data, not the masked `texts`.

### 2. Mismatched Node IDs
The masking creates anchors with node IDs like:
- `#/texts/15` (for text items)

But `_update_table_cells()` expects:
- `#/tables/0/cell_0_0` (for grid cells)

This mismatch means table cells are never updated with masked values.

### 3. Poor Entity Detection in Tables
- Single-word names like "Johnson", "Ashley", "Marjorie" are NOT detected as PERSON entities
- Multi-line entity detection incorrectly captures: "1958-04-21\n\nBorden\n\nAshley" as a single DATE entity

## Impact
- Sensitive PII in tables remains exposed in masked documents
- Tables are a common place for structured PII (employee lists, customer data, etc.)
- This defeats the purpose of masking for documents with tabular data

## TODOs to Fix

### TODO 1: Fix Table Cell Masking in Grid Structure
- [ ] Update `presidio_adapter.py` to properly mask `tables[].data.grid[row][col].text`
- [ ] Ensure both `table_cells` and `grid` structures are updated
- [ ] Test with various table structures (nested tables, merged cells, etc.)

### TODO 2: Generate Correct Node IDs for Table Cells
- [ ] When extracting table segments, generate node IDs like `#/tables/0/cell_row_col`
- [ ] Ensure anchors use these table-specific node IDs
- [ ] Update `_update_table_cells()` to correctly map anchors to grid cells

### TODO 3: Improve Entity Detection for Table Content
- [ ] Add context-aware detection for single-word names in table columns
- [ ] Consider column headers ("first name", "surname") as hints for entity types
- [ ] Fix multi-line detection to not span across table cells
- [ ] Add specific handling for common table patterns (employee lists, contact lists)

### TODO 4: Add Table-Specific Masking Tests
- [ ] Create test cases with various table formats
- [ ] Test that both JSON and markdown outputs have masked table data
- [ ] Verify round-trip masking/unmasking for tables
- [ ] Test edge cases: empty cells, merged cells, nested tables

### TODO 5: Consider Table Structure Preservation
- [ ] Decide if table headers should be masked
- [ ] Handle special table elements (footnotes, captions)
- [ ] Preserve table formatting while masking content

## Temporary Workaround
Currently, there is no workaround. Tables in documents will not be properly masked, leaving PII exposed.

## Code References
- Table extraction: `/cloakpivot/document/extractor.py:164-169`
- Table cell update attempt: `/cloakpivot/masking/presidio_adapter.py:851-901`
- Anchor generation: `/cloakpivot/masking/presidio_adapter.py:281-301`

## Priority
**HIGH** - Tables often contain the most structured and sensitive PII in documents (employee records, financial data, contact lists). This issue makes the masking feature unreliable for many business documents.