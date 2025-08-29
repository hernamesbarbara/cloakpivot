#!/usr/bin/env python3
"""Debug the unmask result issue."""

import sys
sys.path.insert(0, '.')

try:
    from cloakpivot.unmasking.engine import UnmaskingResult
    print(f"UnmaskingResult attributes: {UnmaskingResult.__annotations__}")
    
    # Try to create an instance
    from cloakpivot.core.cloakmap import CloakMap
    from docling_core.types import DoclingDocument
    from docling_core.types.doc.document import TextItem
    
    # Create a minimal document
    doc = DoclingDocument(name="test")
    text_item = TextItem(text="test content", self_ref="#/texts/0", label="text", orig="test content")
    doc.texts = [text_item]
    
    # Create a minimal cloakmap
    cloakmap = CloakMap(document_hash="test", anchors=[])
    
    # Create UnmaskingResult
    result = UnmaskingResult(restored_document=doc, cloakmap=cloakmap)
    print(f"Created UnmaskingResult with attributes: {dir(result)}")
    print(f"Has restored_document: {hasattr(result, 'restored_document')}")
    print(f"Has unmasked_document: {hasattr(result, 'unmasked_document')}")
    
except Exception as e:
    import traceback
    print(f"Error: {e}")
    traceback.print_exc()

print("Debug completed")