#!/usr/bin/env python3
"""Debug script to isolate the round-trip SSN issue."""

import logging
from docling_core.types import DoclingDocument
from docling_core.types.doc.document import TextItem
from presidio_analyzer import RecognizerResult

from cloakpivot.core.policies import MaskingPolicy
from cloakpivot.document.extractor import TextExtractor
from cloakpivot.masking.engine import MaskingEngine
from cloakpivot.unmasking.engine import UnmaskingEngine
from tests.utils.masking_helpers import mask_document_with_detection

# Set up logging to see detailed output
logging.basicConfig(level=logging.WARNING)  # Reduce noise
logger = logging.getLogger(__name__)

def create_test_document_with_ssn():
    """Create a simple test document with an SSN."""
    text_content = "Contact: 555-123-4567 Email: alice.smith@company.org SSN: 987-65-4321"
    
    # Create a text item with required fields (following conftest.py pattern)
    text_item = TextItem(
        text=text_content,
        self_ref="#/texts/0",
        label="text",
        orig=text_content
    )
    
    # Create document
    document = DoclingDocument(
        name="test_document",
        texts=[text_item]
    )
    
    return document

def create_ssn_entity(text_content: str):
    """Create a RecognizerResult for the SSN in the text."""
    ssn_start = text_content.find("987-65-4321")
    ssn_end = ssn_start + len("987-65-4321")
    
    return RecognizerResult(
        entity_type="US_SSN",
        start=ssn_start,
        end=ssn_end,
        score=0.95
    )

def main():
    print("🔍 Debugging SSN round-trip issue")
    print("=" * 50)
    
    # Create test document
    document = create_test_document_with_ssn()
    text_content = document.texts[0].text
    print(f"📝 Original text: '{text_content}'")
    
    # Create SSN entity
    ssn_entity = create_ssn_entity(text_content)
    entities = [ssn_entity]
    print(f"🎯 SSN Entity: {ssn_entity.entity_type} at {ssn_entity.start}-{ssn_entity.end}")
    
    # Create masking policy
    policy = MaskingPolicy()
    print(f"📋 Policy: {policy.default_strategy.kind}")
    
    # Extract text segments
    extractor = TextExtractor()
    text_segments = extractor.extract_text_segments(document)
    print(f"📄 Text segments: {len(text_segments)}")
    for i, segment in enumerate(text_segments):
        print(f"  Segment {i}: node_id='{segment.node_id}', text='{segment.text}', range={segment.start_offset}-{segment.end_offset}")
    
    # Mask the document using the SAME method as the failing test
    print("\n🎭 MASKING PHASE (using test helper)")
    print("-" * 30)
    
    try:
        mask_result = mask_document_with_detection(document, policy)
        
        masked_text = mask_result.masked_document.texts[0].text
        print(f"✅ Masked text: '{masked_text}'")
        print(f"📍 CloakMap anchors: {len(mask_result.cloakmap.anchors)}")
        
        for i, anchor in enumerate(mask_result.cloakmap.anchors):
            print(f"  Anchor {i}:")
            print(f"    ID: {anchor.replacement_id}")
            print(f"    Node: {anchor.node_id}")
            print(f"    Position: {anchor.start}-{anchor.end}")
            print(f"    Entity: {anchor.entity_type}")
            print(f"    Masked value: '{anchor.masked_value}'")
            if hasattr(anchor, 'metadata') and anchor.metadata and 'original_text' in anchor.metadata:
                print(f"    Original: '{anchor.metadata['original_text']}'")
        
    except Exception as e:
        print(f"❌ Masking failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Unmask the document
    print("\n🔓 UNMASKING PHASE")
    print("-" * 30)
    unmasking_engine = UnmaskingEngine()
    
    try:
        unmask_result = unmasking_engine.unmask_document(
            masked_document=mask_result.masked_document,
            cloakmap=mask_result.cloakmap,
            verify_integrity=True
        )
        
        unmasked_text = unmask_result.restored_document.texts[0].text
        print(f"🔄 Unmasked text: '{unmasked_text}'")
        
        if unmask_result.stats:
            print(f"📊 Stats: {unmask_result.stats}")
        
        if unmask_result.integrity_report:
            print(f"🔍 Integrity: {unmask_result.integrity_report}")
        
        # Check round-trip fidelity
        print("\n✅ ROUND-TRIP CHECK")
        print("-" * 30)
        original_ssn = text_content[ssn_entity.start:ssn_entity.end]
        unmasked_ssn = unmasked_text[ssn_entity.start:ssn_entity.end]
        
        print(f"Original SSN: '{original_ssn}'")
        print(f"Unmasked SSN: '{unmasked_ssn}'")
        
        if original_ssn == unmasked_ssn:
            print("🎉 Round-trip SUCCESSFUL!")
        else:
            print("💥 Round-trip FAILED!")
            print(f"   Length diff: {len(unmasked_ssn) - len(original_ssn)}")
        
    except Exception as e:
        print(f"❌ Unmasking failed: {e}")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    main()