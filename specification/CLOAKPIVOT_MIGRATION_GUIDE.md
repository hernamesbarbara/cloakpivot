# CloakPivot Migration Guide to DocPivot v2.0.1

## Quick Summary

DocPivot v2.0.1 replaces multiple functions with a single `DocPivotEngine` class. Since CloakPivot mainly loads Docling JSON files and converts to Lexical format, most changes are straightforward.

## Core Changes Required

### 1. Update `examples/docling_to_lexical_workflow.py`

**Before:**
```python
from docpivot import load_document, to_lexical

loaded_doc: DoclingDocument = load_document(docling_json_path)
lexical_doc = to_lexical(loaded_doc)
```

**After:**
```python
from docpivot import DocPivotEngine
import json
from docling_core.types import DoclingDocument

# Create engine once
engine = DocPivotEngine()

# For Docling JSON files, you can load directly:
with open(docling_json_path, 'r') as f:
    doc_dict = json.load(f)
loaded_doc = DoclingDocument.model_validate(doc_dict)

# Convert to Lexical
result = engine.convert_to_lexical(loaded_doc)
lexical_doc = json.loads(result.content)
```

### 2. Update `cloakpivot/document/processor.py`

Since you're mainly loading Docling JSON files, you don't need DocPivot for loading at all:

**Before:**
```python
from docpivot import load_document
from docpivot.io.readers.exceptions import FileAccessError, TransformationError

def load_document(self, file_path: Union[str, Path], validate: bool = None, **kwargs) -> DoclingDocument:
    document = cast(DoclingDocument, load_document(file_path, **kwargs))
```

**After:**
```python
import json
from pathlib import Path
from docling_core.types import DoclingDocument

def load_document(self, file_path: Union[str, Path], validate: bool = None, **kwargs) -> DoclingDocument:
    """Load a Docling JSON document directly."""
    try:
        with open(file_path, 'r') as f:
            doc_dict = json.load(f)
        document = DoclingDocument.model_validate(doc_dict)

        if validate:
            self._validate_document(document)

        return document
    except FileNotFoundError:
        raise FileAccessError(f"File not found: {file_path}")
    except json.JSONDecodeError as e:
        raise TransformationError(f"Invalid JSON: {e}")
```

### 3. Update `cloakpivot/formats/registry.py`

**Before:**
```python
from docpivot import SerializerProvider

class FormatRegistry:
    def __init__(self):
        self._provider = SerializerProvider()
```

**After:**
```python
from docpivot import DocPivotEngine

class FormatRegistry:
    def __init__(self):
        self._engine = DocPivotEngine()

    def serialize_to_lexical(self, document: DoclingDocument) -> str:
        """Serialize document to Lexical JSON."""
        result = self._engine.convert_to_lexical(document)
        return result.content  # Returns JSON string
```

### 4. Update `cloakpivot/core/batch.py`

**Before:**
```python
from docpivot import LexicalDocSerializer

serializer = LexicalDocSerializer(masking_result.masked_document)
serialized_content = serializer.serialize()
```

**After:**
```python
from docpivot import DocPivotEngine

# Create engine once at class level
class BatchProcessor:
    def __init__(self):
        self._engine = DocPivotEngine()

    def process_document(self, masking_result):
        # Convert to Lexical
        result = self._engine.convert_to_lexical(masking_result.masked_document)
        serialized_content = result.content  # JSON string
```

## Complete Example Migration

Here's a complete example showing the new pattern:

```python
#!/usr/bin/env python3
"""Example using DocPivot v2.0.1"""

import json
from pathlib import Path
from docling_core.types import DoclingDocument
from docpivot import DocPivotEngine

def process_document(docling_json_path: Path):
    # Step 1: Load Docling JSON directly (no DocPivot needed)
    with open(docling_json_path, 'r') as f:
        doc_dict = json.load(f)
    document = DoclingDocument.model_validate(doc_dict)

    # Step 2: Create engine for conversions
    engine = DocPivotEngine()

    # Step 3: Convert to Lexical
    result = engine.convert_to_lexical(document, pretty=True)

    # Step 4: Get the JSON content
    lexical_json = result.content  # This is a JSON string

    # Step 5: Parse if you need a dict
    lexical_dict = json.loads(lexical_json)

    return lexical_dict
```

## Key Points

1. **No more `load_document()`** - Just load JSON files directly
2. **No more `to_lexical()`** - Use `engine.convert_to_lexical()`
3. **No more `SerializerProvider`** - Use `DocPivotEngine` directly
4. **No more `LexicalDocSerializer`** - Use `engine.convert_to_lexical()`

## Testing Updates

Update test mocks from:
```python
@patch("docpivot.load_document")
```

To:
```python
@patch("builtins.open", new_callable=mock_open, read_data='{"valid": "json"}')
@patch("docling_core.types.DoclingDocument.model_validate")
```

## Benefits After Migration

- Simpler code (no unnecessary DocPivot calls for JSON loading)
- Better performance (direct JSON loading)
- Cleaner API (one engine instance for all conversions)
- Less mocking in tests