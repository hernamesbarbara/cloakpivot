#!/usr/bin/env python3
"""Debug large document masking/unmasking issues."""

import sys
sys.path.insert(0, '.')

try:
    from docling_core.types import DoclingDocument
    from docling_core.types.doc.document import TextItem
    from cloakpivot.core.policies import MaskingPolicy  
    from cloakpivot.core.strategies import Strategy, StrategyKind
    from cloakpivot.masking.engine import MaskingEngine
    from cloakpivot.unmasking.engine import UnmaskingEngine
    from tests.utils.masking_helpers import mask_document_with_detection
    
    print("Creating large test document with repeated PII...")
    
    # Create the same sample text that appears in the test fixture
    sample_text = (
        "Contact John Doe at 555-123-4567 or john.doe@example.com. "
        "His SSN is 123-45-6789 and credit card is 4532-1234-5678-9012. "
        "Address: 123 Main St, New York, NY 10001. "
        "License: DL123456789 expires 12/31/2025."
    )
    
    # Create large document with same structure as the test fixture
    doc = DoclingDocument(name="large_test_document")
    text_items = []
    for i in range(5):  # Use smaller number for debugging
        text_item = TextItem(
            text=f"Section {i}: {sample_text}",
            self_ref=f"#/texts/{i}",
            label="text",
            orig=f"Section {i}: {sample_text}"
        )
        text_items.append(text_item)
    doc.texts = text_items
    
    # Create comprehensive masking policy with SURROGATE strategies for reversibility
    policy = MaskingPolicy(
        locale="en",
        per_entity={
            "EMAIL_ADDRESS": Strategy(kind=StrategyKind.SURROGATE, parameters={"format_type": "email"}),
            "DATE_TIME": Strategy(kind=StrategyKind.SURROGATE, parameters={"format_type": "custom"}),
            "LOCATION": Strategy(kind=StrategyKind.SURROGATE, parameters={"format_type": "custom"}),
        },
        thresholds={
            "EMAIL_ADDRESS": 0.5,
            "DATE_TIME": 0.5,
            "LOCATION": 0.5,
        }
    )
    
    print("Extracting text segments...")
    from cloakpivot.document.extractor import TextExtractor
    extractor = TextExtractor()
    text_segments = extractor.extract_text_segments(doc)
    
    print(f"Found {len(text_segments)} text segments:")
    for i, segment in enumerate(text_segments):
        print(f"  {i}: {segment.node_id} - '{segment.text[:50]}...'")
        print(f"      Offsets: {segment.start_offset}-{segment.end_offset}")
    print()
    
    print("Masking document...")
    mask_result = mask_document_with_detection(doc, policy)
    
    print(f"CloakMap has {len(mask_result.cloakmap.anchors)} anchors:")
    print()
    
    # Group anchors by original text to see duplication patterns
    original_texts = {}
    for i, anchor in enumerate(mask_result.cloakmap.anchors):
        if anchor.metadata and "original_text" in anchor.metadata:
            original_text = anchor.metadata["original_text"]
            if original_text not in original_texts:
                original_texts[original_text] = []
            original_texts[original_text].append((i, anchor))
    
    for original_text, anchor_list in original_texts.items():
        print(f"Original text: '{original_text}' appears {len(anchor_list)} times:")
        for i, anchor in anchor_list:
            print(f"  {i+1}. {anchor.entity_type} at {anchor.start}-{anchor.end} in {anchor.node_id}")
            print(f"      Masked value: '{anchor.masked_value}'")
        print()
    
    print("Sample masked sections:")
    for i in range(min(3, len(mask_result.masked_document.texts))):
        text_item = mask_result.masked_document.texts[i]
        print(f"Section {i}: {text_item.text}")
    print()
    
    print("Attempting unmasking...")
    unmasking_engine = UnmaskingEngine()
    unmask_result = unmasking_engine.unmask_document(
        mask_result.masked_document,
        mask_result.cloakmap
    )
    
    print("Sample unmasked sections:")
    for i in range(min(3, len(unmask_result.restored_document.texts))):
        text_item = unmask_result.restored_document.texts[i]
        print(f"Section {i}: {text_item.text}")
    print()
    
    print("✅ Debug completed successfully")
    
except Exception as e:
    import traceback
    print(f"❌ Error: {e}")
    traceback.print_exc()