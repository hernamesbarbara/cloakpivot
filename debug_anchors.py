#!/usr/bin/env python3
"""Debug anchor resolution issues."""

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
    
    print("Creating test document with PII...")
    
    # Create sample document
    sample_text = (
        "Contact John Doe at 555-123-4567 or john.doe@example.com. "
        "His SSN is 123-45-6789 and credit card is 4532-1234-5678-9012. "
        "Address: 123 Main St, New York, NY 10001. "
        "License: DL123456789 expires 12/31/2025."
    )
    
    doc = DoclingDocument(name="test_document")
    text_item = TextItem(
        text=sample_text,
        self_ref="#/texts/0", 
        label="text",
        orig=sample_text
    )
    doc.texts = [text_item]
    
    # Create comprehensive masking policy with SURROGATE strategies for reversibility
    policy = MaskingPolicy(
        locale="en",
        per_entity={
            "PHONE_NUMBER": Strategy(kind=StrategyKind.SURROGATE, parameters={"format_type": "phone"}),
            "EMAIL_ADDRESS": Strategy(kind=StrategyKind.SURROGATE, parameters={"format_type": "email"}),
            "US_SSN": Strategy(kind=StrategyKind.SURROGATE, parameters={"format_type": "ssn"}),
            "CREDIT_CARD": Strategy(kind=StrategyKind.SURROGATE, parameters={"format_type": "credit_card"}),
            "DATE_TIME": Strategy(kind=StrategyKind.SURROGATE, parameters={"format_type": "custom"}),
            "LOCATION": Strategy(kind=StrategyKind.SURROGATE, parameters={"format_type": "custom"}),
            "PERSON": Strategy(kind=StrategyKind.SURROGATE, parameters={"format_type": "name"}),
        },
        thresholds={
            "PHONE_NUMBER": 0.5,
            "EMAIL_ADDRESS": 0.5,
            "US_SSN": 0.5,
            "CREDIT_CARD": 0.5,
            "DATE_TIME": 0.5,
            "LOCATION": 0.5,
            "PERSON": 0.5,
        }
    )
    
    print("Masking document...")
    mask_result = mask_document_with_detection(doc, policy)
    
    print(f"Original text: {sample_text}")
    print(f"Masked text: {mask_result.masked_document.texts[0].text}")
    print(f"CloakMap has {len(mask_result.cloakmap.anchors)} anchors:")
    
    for i, anchor in enumerate(mask_result.cloakmap.anchors):
        print(f"  {i+1}. {anchor.entity_type} at {anchor.start}-{anchor.end}")
        print(f"      Replacement ID: '{anchor.replacement_id}'")
        print(f"      Masked value: '{anchor.masked_value}'")
        print(f"      Node ID: {anchor.node_id}")
        print(f"      Strategy used: {anchor.strategy_used}")
        if anchor.metadata:
            print(f"      Metadata: {anchor.metadata}")
        else:
            print(f"      Metadata: None")
        print()
    
    print("Attempting unmasking...")
    unmasking_engine = UnmaskingEngine()
    unmask_result = unmasking_engine.unmask_document(
        mask_result.masked_document,
        mask_result.cloakmap
    )
    
    print(f"Unmasked text: {unmask_result.restored_document.texts[0].text}")
    
    print("✅ Debug completed successfully")
    
except Exception as e:
    import traceback
    print(f"❌ Error: {e}")
    traceback.print_exc()