#!/usr/bin/env python3
"""Debug script to replicate the exact failing test case."""

import logging
from docling_core.types import DoclingDocument
from docling_core.types.doc.document import TextItem

from cloakpivot.core.policies import MaskingPolicy
from cloakpivot.unmasking.engine import UnmaskingEngine
from tests.utils.masking_helpers import mask_document_with_detection
from tests.utils.assertions import assert_round_trip_fidelity

# Set up logging to see warnings
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

def create_failing_document():
    """Create the exact document that's failing in the test."""
    # This is the text from the failing test output
    text_content = "0000000000 Contact: 555-123-4567 Email: alice.smith@company.org SSN: 987-65-4321"
    
    # Create document matching the test
    doc = DoclingDocument(name="0")  # Notice: name="0" like in test
    text_item = TextItem(
        text=text_content,
        self_ref="#/texts/0",
        label="text",
        orig=text_content
    )
    doc.texts = [text_item]
    return doc

def create_failing_policy():
    """Create the policy from the failing test."""
    # From the test output, it's using default REDACT strategy
    return MaskingPolicy()

def main():
    print("🔍 Debugging SPECIFIC test failure")
    print("=" * 50)
    
    # Create the exact same document and policy as the failing test
    document = create_failing_document()
    policy = create_failing_policy()
    
    print(f"📝 Document name: '{document.name}'")
    print(f"📝 Text: '{document.texts[0].text}'")
    print(f"📋 Policy: {policy.default_strategy.kind}")
    
    # Mask using the same helper as the test
    print("\n🎭 MASKING (using test helper)")
    print("-" * 40)
    try:
        mask_result = mask_document_with_detection(document, policy)
        
        masked_text = mask_result.masked_document.texts[0].text
        print(f"✅ Masked: '{masked_text}'")
        print(f"📍 Anchors: {len(mask_result.cloakmap.anchors)}")
        
        for i, anchor in enumerate(mask_result.cloakmap.anchors):
            print(f"  Anchor {i}: {anchor.entity_type} '{anchor.masked_value}' at {anchor.start}-{anchor.end}")
        
    except Exception as e:
        print(f"❌ Masking failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Unmask using the same engine as the test
    print("\n🔓 UNMASKING")
    print("-" * 40)
    try:
        unmasking_engine = UnmaskingEngine()
        unmask_result = unmasking_engine.unmask_document(
            mask_result.masked_document,
            mask_result.cloakmap
        )
        
        unmasked_text = unmask_result.unmasked_document.texts[0].text
        print(f"🔄 Unmasked: '{unmasked_text}'")
        
        if unmask_result.stats:
            resolved = unmask_result.stats.get('resolved_anchors', 0)
            failed = unmask_result.stats.get('failed_anchors', 0)
            print(f"📊 Resolution: {resolved} resolved, {failed} failed")
        
    except Exception as e:
        print(f"❌ Unmasking failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Apply the same assertion as the test
    print("\n✅ ASSERTION TEST")
    print("-" * 40)
    try:
        assert_round_trip_fidelity(
            document,
            mask_result.masked_document,
            unmask_result.unmasked_document,
            mask_result.cloakmap
        )
        print("🎉 Round-trip assertion PASSED!")
        
    except AssertionError as e:
        print(f"💥 Round-trip assertion FAILED:")
        print(f"   {e}")
        
        # Detailed comparison
        original_text = document.texts[0].text
        unmasked_text = unmask_result.unmasked_document.texts[0].text
        
        print("\n🔍 DETAILED COMPARISON")
        print(f"Original : '{original_text}'")
        print(f"Unmasked : '{unmasked_text}'")
        print(f"Length   : {len(original_text)} vs {len(unmasked_text)}")
        print(f"Match    : {original_text == unmasked_text}")
        
        if len(original_text) == len(unmasked_text):
            for i, (c1, c2) in enumerate(zip(original_text, unmasked_text)):
                if c1 != c2:
                    print(f"First diff at position {i}: '{c1}' vs '{c2}'")
                    print(f"Context: ...{original_text[max(0,i-10):i+10]}...")
                    break
    
    except Exception as e:
        print(f"❌ Assertion test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()