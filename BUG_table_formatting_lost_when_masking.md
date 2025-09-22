# BUG: DoclingDocument's export_to_markdown() Returns Empty After Masking

## Summary

When CloakEngine masks a DoclingDocument, the resulting masked document loses its ability to export to markdown via the `export_to_markdown()` method. This method returns an empty string instead of properly formatted markdown, forcing downstream consumers to implement workarounds that inevitably lose document formatting fidelity (especially for tables).

## Impact

### Severity: High
- Breaks a core DoclingDocument API method
- Forces downstream projects to implement complex workarounds
- Results in loss of document formatting (tables, lists, nested structures)
- Affects any project using cloakpivot with DoclingDocument for markdown output

### Affected Components
- `cloakpivot.masking.presidio_adapter.PresidioMaskingAdapter`
- `cloakpivot.engine.CloakEngine.mask_document()`
- DoclingDocument integration (v1.7.0+)

## Reproduction

```python
from docling.document_converter import DocumentConverter
from cloakpivot import CloakEngine
from cloakpivot.core.policies import MaskingPolicy
from cloakpivot.core.strategies import Strategy, StrategyKind

# Convert PDF to DoclingDocument
converter = DocumentConverter()
result = converter.convert("document_with_table.pdf")
original_doc = result.document

# Original document exports markdown correctly
original_md = original_doc.export_to_markdown()
print(len(original_md))  # Output: 701 (contains proper table formatting)

# Apply masking
policy = MaskingPolicy(
    default_strategy=Strategy(
        kind=StrategyKind.SURROGATE,
        parameters={"seed": "test"},
    )
)
cloak_engine = CloakEngine(default_policy=policy)
mask_result = cloak_engine.mask_document(original_doc)

# Masked document's export_to_markdown() is broken
masked_md = mask_result.document.export_to_markdown()
print(len(masked_md))  # Output: 0 (empty string!)
```

## Downstream Behavior

### Current Workaround in pii-cloak

Projects using cloakpivot are forced to implement fallback rendering:

```python
def render_markdown(doc, *, title: str, entities_found: int, entities_masked: int) -> str:
    if hasattr(doc, "export_to_markdown"):
        md = doc.export_to_markdown()
        if md:
            return md  # Never reaches here after masking!

    # Fallback: manually reconstruct markdown
    parts = [f"# {title}", ""]
    if getattr(doc, "texts", None):
        for t in doc.texts:
            parts.append(getattr(t, "text", ""))
    # Tables are lost - just become individual text lines!
    return "\n".join(parts)
```

### Why the Workaround Fails

1. **Table Structure Lost**: Tables are decomposed into individual text elements, losing pipe delimiters and row/column structure
2. **Formatting Lost**: Headers, lists, code blocks, and other markdown elements are not preserved
3. **Document Hierarchy Lost**: Nested structures and relationships between elements are flattened
4. **Maintenance Burden**: Each downstream project must implement complex document reconstruction logic

### Example of Data Loss

**Before masking (via export_to_markdown())**:
```markdown
| surname   | first name   | birthdate   |
|-----------|--------------|-------------|
| White     | Johnson      | 1958-04-21  |
| Borden    | Ashley       | 1944-12-22  |
```

**After masking (fallback rendering)**:
```
surname
first name
birthdate
White
Johnson
1620-73-81
Lvrsoj
Jgvzcy
...
```

## Root Cause Analysis

### The Problem

In `cloakpivot/masking/presidio_adapter.py` (lines 310-320), a new DoclingDocument is created:

```python
masked_document = DoclingDocument(
    name=document.name,
    texts=[],  # Populated with masked TextItems
    tables=copy.deepcopy(document.tables),
    key_value_items=copy.deepcopy(document.key_value_items),
    origin=document.origin,
)
```

However, this approach misses critical internal state that DoclingDocument needs for `export_to_markdown()` to function:

1. **Missing Internal Metadata**: DoclingDocument likely maintains internal state beyond the visible properties
2. **Document Structure Graph**: The relationships between elements may not be preserved
3. **Rendering Context**: Internal rendering configuration or converters may be lost
4. **Version-Specific State**: DoclingDocument v1.7.0+ may have additional requirements

### Evidence

- Original document: `export_to_markdown()` returns 701 characters of properly formatted markdown
- Masked document: `export_to_markdown()` returns empty string (0 characters)
- Both documents have identical visible structure (texts, tables, key_value_items)
- The `tables` property exists but individual table objects lack `to_markdown()` method

## Proposed Solutions

### Solution 1: Deep Clone with State Preservation (Recommended)

Instead of creating a new DoclingDocument, deep clone the original and modify in place:

```python
def mask_document_with_presidio(self, document, entities, policy, text_segments):
    # Deep clone the entire document to preserve all internal state
    masked_document = copy.deepcopy(document)

    # Clear and repopulate texts with masked versions
    masked_document.texts.clear()

    for original_item in document.texts:
        # Apply masking to text
        masked_text = self._apply_masking_to_text(original_item.text, entities)

        # Create new TextItem preserving all metadata
        masked_item = copy.deepcopy(original_item)
        masked_item.text = masked_text
        masked_document.texts.append(masked_item)

    # Update table cells in place
    if hasattr(masked_document, 'tables'):
        for table in masked_document.tables:
            if hasattr(table, 'data') and hasattr(table.data, 'table_cells'):
                for cell in table.data.table_cells:
                    cell.text = self._apply_masking_to_text(cell.text, entities)

    return MaskingResult(document=masked_document, cloakmap=cloakmap)
```

### Solution 2: Use DoclingDocument's API Methods

If DoclingDocument provides methods for modifying content, use them instead of direct manipulation:

```python
def mask_document_with_api(self, document, entities, policy):
    # Start with a deep copy
    masked_document = copy.deepcopy(document)

    # Use DoclingDocument's own methods to update content
    for i, text_item in enumerate(masked_document.texts):
        masked_text = self._apply_masking_to_text(text_item.text, entities)
        # Look for an update method like:
        masked_document.update_text(i, masked_text)
        # or
        text_item.update_content(masked_text)

    return masked_document
```

### Solution 3: Preserve Export Capability

Investigate what specifically `export_to_markdown()` requires and ensure it's preserved:

```python
# In PresidioMaskingAdapter.mask_document_with_presidio()

# Check what makes export_to_markdown work
required_attrs = ['_markdown_exporter', '_render_config', '_document_graph']
for attr in required_attrs:
    if hasattr(document, attr):
        setattr(masked_document, attr, getattr(document, attr))
```

## Critical Files for Debugging

### In cloakpivot:
1. **`masking/presidio_adapter.py`** (lines 305-400)
   - The `mask_document_with_presidio()` method that creates the broken document
   - Specifically line 310 where new DoclingDocument is instantiated

2. **`engine.py`** (lines 132-195)
   - The `mask_document()` method that orchestrates the masking

3. **`document/extractor.py`**
   - Text extraction logic that might reveal document structure dependencies

4. **`core/types.py`**
   - Type definitions and imports for DoclingDocument

### Testing Files Needed:
- Test PDF with tables, lists, and complex formatting
- Unit tests verifying `export_to_markdown()` works after masking
- Integration tests with real DoclingDocument instances

## Recommended Fix Priority

1. **Immediate**: Implement Solution 1 (deep clone approach) as it's most likely to preserve all state
2. **Test**: Verify `export_to_markdown()` works with various document types
3. **Investigate**: Work with docling team to understand the proper way to create modified documents
4. **Long-term**: Consider adding a document adapter pattern to handle different document types

## Testing Requirements

```python
def test_export_to_markdown_preserved():
    """Verify export_to_markdown() works after masking."""
    # Load document with tables
    original = load_test_document("table_document.pdf")

    # Verify original exports markdown
    original_md = original.export_to_markdown()
    assert len(original_md) > 0
    assert "|" in original_md  # Has table formatting

    # Apply masking
    engine = CloakEngine()
    result = engine.mask_document(original)

    # Verify masked document exports markdown
    masked_md = result.document.export_to_markdown()
    assert len(masked_md) > 0
    assert "|" in masked_md  # Table formatting preserved
    assert "REDACTED" in masked_md or "MASKED" in masked_md  # Content is masked
```

## Additional Context

- DoclingDocument version: 1.7.0
- Cloakpivot version: Current
- The issue only occurs with certain masking strategies (confirmed with SURROGATE strategy)
- The `tables` property is correctly deep copied but internal rendering state is lost
- This affects all document types with complex formatting, not just tables

## Expected Resolution

After fixing this bug, `mask_result.document.export_to_markdown()` should:
1. Return non-empty markdown string
2. Preserve all original formatting (tables, lists, headers)
3. Contain masked values in place of PII
4. Maintain document structure and hierarchy