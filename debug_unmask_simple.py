#!/usr/bin/env python3
"""Simple debug of unmask result issue."""

import sys
sys.path.insert(0, '.')

try:
    from cloakpivot.unmasking.engine import UnmaskingResult, UnmaskingEngine
    from cloakpivot.core.cloakmap import CloakMap
    from docling_core.types import DoclingDocument
    from docling_core.types.doc.document import TextItem
    
    print("Testing UnmaskingResult creation...")
    
    # Create a minimal document
    doc = DoclingDocument(name="test")
    text_item = TextItem(text="test content", self_ref="#/texts/0", label="text", orig="test content")
    doc.texts = [text_item]
    
    # Create a minimal cloakmap - using proper parameters
    cloakmap = CloakMap(doc_hash="testhash", doc_id="testdoc")
    
    # Create UnmaskingResult
    result = UnmaskingResult(restored_document=doc, cloakmap=cloakmap)
    print(f"✅ Created UnmaskingResult successfully")
    print(f"   Has restored_document: {hasattr(result, 'restored_document')}")
    print(f"   Has unmasked_document: {hasattr(result, 'unmasked_document')}")
    print(f"   Restored document name: {result.restored_document.name}")
    
    # Now let's test the actual unmask_document method 
    engine = UnmaskingEngine()
    print(f"✅ Created UnmaskingEngine successfully")
    
    try:
        # This will likely fail due to missing entities, but let's see what error we get
        result2 = engine.unmask_document(doc, cloakmap)
        print(f"✅ unmask_document returned: {type(result2)}")
        print(f"   Has restored_document: {hasattr(result2, 'restored_document')}")
    except Exception as e:
        print(f"⚠️  unmask_document failed (expected): {e}")
        # This is expected - we just want to see if the method signature is correct
    
except Exception as e:
    import traceback
    print(f"❌ Error: {e}")
    traceback.print_exc()

print("Debug completed")