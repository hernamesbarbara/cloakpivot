# Issue: Masked Documents Lose Structure, Cannot Export to Markdown [RESOLVED]

## Problem Description
The `pdf_workflow.py` example appears to run successfully but silently fails to produce valid masked markdown output. Both `email.masked.md` and `email.unmasked.md` files are created but remain empty (0 bytes).

## Root Cause
The masking process strips out the `body.children` array from the DoclingDocument structure. This array is critical for `export_to_markdown()` to work properly.

### Evidence
- Original document: `body.children` has 8 items (references to groups)
- Masked document: `body.children` is empty array
- Result: `export_to_markdown()` returns empty string

## Impact
1. Users cannot view masked documents in markdown format
2. The workflow appears successful but produces unusable output
3. Unmasking fails with "Failed to resolve anchor" warnings for all anchors

## TODOs to Fix

### TODO 1: Debug the Masking Process
- [ ] Trace through `CloakEngine.mask_document()` to find where `body.children` is lost
- [ ] Check if the issue is in document serialization/deserialization
- [ ] Verify if the problem occurs during the masking operation itself

### TODO 2: Preserve Document Structure During Masking
- [ ] Ensure `body.children` array is preserved when creating masked document
- [ ] Maintain all document structure references (`$ref` entries)
- [ ] Keep the relationship between body, groups, and texts intact

### TODO 3: Fix the Unmasking Process
- [ ] Ensure anchors can be properly resolved
- [ ] Verify the CloakMap contains correct position information
- [ ] Test that unmasking restores the original structure

### TODO 4: Add Validation to pdf_workflow.py
- [ ] Check that markdown files are not empty after creation
- [ ] Verify masked document maintains structure
- [ ] Add explicit validation that masking worked correctly
- [ ] Fail loudly if any step produces invalid output

### TODO 5: Add Tests
- [ ] Create unit test for document structure preservation
- [ ] Add integration test for full PDF workflow
- [ ] Test with various document types to ensure robustness

## Expected Behavior
1. Read `data/pdf/email.pdf`
2. Convert to DoclingDocument
3. Export original to markdown (✓ currently works)
4. Mask the document while preserving structure
5. Export masked to markdown (should produce valid markdown with masked content)
6. Unmask and verify round-trip integrity

## Current Behavior
Steps 1-3 work correctly, but steps 4-6 fail silently:
- Masking appears to work but loses document structure
- Masked markdown export produces empty file
- Unmasking fails to restore original content

## RESOLUTION

### Fix Applied
Modified `/cloakpivot/masking/presidio_adapter.py` to preserve document structure:
1. Instead of creating a new DoclingDocument with empty fields, serialize the original document to preserve all structure
2. Update text items in-place rather than replacing the texts array
3. This maintains all references between body, groups, pictures, and texts

### Code Changes
```python
# Before: Created document with partial fields
masked_document = DoclingDocument(
    name=document.name,
    texts=[],  # Empty, loses references
    tables=...,
    # Missing body, groups, etc.
)

# After: Preserve full structure
doc_dict = json.loads(document.model_dump_json())
masked_document = DoclingDocument.model_validate(doc_dict)
# Then update texts in-place to maintain references
```

### Results
✓ Original markdown exports correctly (689 characters)
✓ Masked markdown exports correctly (613 characters)
✓ PII is properly masked (names → [NAME], emails → [EMAIL], dates → [DATE])
✓ Document structure preserved (body.children maintained)
✓ Files are no longer empty

### Remaining Issue
Unmasking still doesn't restore original values - this is a separate issue related to anchor resolution in the unmasking process.