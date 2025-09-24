#!/usr/bin/env python3
"""Advanced usage example with CloakEngine builder pattern and custom policies.

This example demonstrates advanced CloakEngine features:
1. Using the builder pattern for configuration
2. Custom masking policies
3. Different confidence thresholds
4. Loading CloakMaps from files
5. Using different policy presets
"""
import sys
from pathlib import Path
from docling.document_converter import DocumentConverter
from cloakpivot import (
    CloakEngine,
    get_default_policy,
    get_conservative_policy,
    get_permissive_policy,
    MaskingPolicy,
    Strategy,
    StrategyKind,
    CloakMap,
)

CLOAKPIVOT_ROOT = Path(__file__).parent.parent
DATA_DIR = CLOAKPIVOT_ROOT / "data"

def example_builder_pattern():
    """Demonstrate using the builder pattern for configuration."""
    print("\n" + "=" * 60)
    print("Example 1: Builder Pattern Configuration")
    print("=" * 60)
    
    # Create a custom policy
    custom_policy = MaskingPolicy(
        per_entity={
            "EMAIL_ADDRESS": Strategy(StrategyKind.PARTIAL, {
                "visible_chars": 3,
                "position": "start"
            }),
            "PERSON": Strategy(StrategyKind.TEMPLATE, {"template": "[PERSON]"}),
            "PHONE_NUMBER": Strategy(StrategyKind.REDACT, {}),
            "US_SSN": Strategy(StrategyKind.PARTIAL, {
                "visible_chars": 4,
                "position": "end"
            }),
        },
        default_strategy=Strategy(StrategyKind.TEMPLATE, {"template": "[PII]"})
    )
    
    # Build engine with custom configuration
    engine = CloakEngine.builder() \
        .with_confidence_threshold(0.85) \
        .with_custom_policy(custom_policy) \
        .with_languages(['en']) \
        .build()
    
    print("  ✓ Built CloakEngine with:")
    print("    • Confidence threshold: 0.85")
    print("    • Custom policy with 4 entity strategies")
    print("    • Language: English")
    
    # Create test document
    test_doc = Path("test_advanced.md")
    test_doc.write_text("""
# Confidential Report

Employee: Michael Johnson (ID: EMP-2024-789)
Email: m.johnson@techcorp.com
Direct Line: 555-444-3333
SSN: 987-65-4321

Performance Review Date: January 15, 2024
Manager: Sarah Williams (s.williams@techcorp.com)
    """)
    
    # Convert and mask
    converter = DocumentConverter()
    doc = converter.convert(str(test_doc)).document
    result = engine.mask_document(doc)
    
    print(f"\n  Results:")
    print(f"    • Entities found: {result.entities_found}")
    print(f"    • Entities masked: {result.entities_masked}")
    print("\n  Masked content preview:")
    print("  " + "-" * 40)
    masked = result.document.export_to_markdown()
    for line in masked.split('\n')[:10]:
        if line.strip():
            print(f"  {line}")
    
    # Cleanup
    test_doc.unlink()
    return result


def example_policy_presets():
    """Demonstrate different policy presets."""
    print("\n" + "=" * 60)
    print("Example 2: Policy Presets Comparison")
    print("=" * 60)
    
    # Create test document with various PII
    test_doc = Path("test_policies.md")
    test_doc.write_text("""
Contact Information:
- Name: Alice Cooper
- Email: alice@example.com
- Phone: 555-111-2222
- Credit Card: 4532-1234-5678-9012
- Date of Birth: 01/15/1990
    """)
    
    converter = DocumentConverter()
    doc = converter.convert(str(test_doc)).document
    
    # Test different policies
    policies = [
        ("Default", get_default_policy()),
        ("Conservative", get_conservative_policy()),
        ("Permissive", get_permissive_policy()),
    ]
    
    for policy_name, policy in policies:
        engine = CloakEngine(default_policy=policy)
        result = engine.mask_document(doc)
        
        print(f"\n  {policy_name} Policy:")
        print(f"    • Entities masked: {result.entities_masked}")
        print(f"    • Sample masking:")
        
        # Show a sample of the masked content
        masked_text = result.document.export_to_markdown()
        for line in masked_text.split('\n'):
            if 'Email:' in line or 'Phone:' in line:
                print(f"      {line.strip()}")
    
    # Cleanup
    test_doc.unlink()


def example_confidence_thresholds():
    """Demonstrate the effect of different confidence thresholds."""
    print("\n" + "=" * 60)
    print("Example 3: Confidence Threshold Effects")
    print("=" * 60)
    
    # Create ambiguous content
    test_doc = Path("test_confidence.md")
    test_doc.write_text("""
Meeting Notes:

Discussed with Jordan about the May project timeline.
Contact: jordan@sales or extension 1234.
Reference: Order #555-ABC-7890
    """)
    
    converter = DocumentConverter()
    doc = converter.convert(str(test_doc)).document
    
    # Test different confidence thresholds
    thresholds = [0.5, 0.7, 0.9]
    
    for threshold in thresholds:
        engine = CloakEngine.builder() \
            .with_confidence_threshold(threshold) \
            .build()
        
        result = engine.mask_document(doc)
        
        print(f"\n  Threshold {threshold}:")
        print(f"    • Entities found: {result.entities_found}")
        print(f"    • Entities masked: {result.entities_masked}")
        print(f"    • Detection sensitivity: {'High' if threshold < 0.6 else 'Medium' if threshold < 0.8 else 'Low'}")
    
    # Cleanup
    test_doc.unlink()


def example_file_based_unmasking():
    """Demonstrate loading a CloakMap from file for unmasking."""
    print("\n" + "=" * 60)
    print("Example 4: File-Based CloakMap Loading")
    print("=" * 60)
    
    # Create and mask a document
    test_doc = Path("test_unmask.md")
    test_doc.write_text("""
Customer Record:
- Name: Robert Smith
- Account: robert.smith@email.com
- Phone: 555-999-8888
    """)
    
    converter = DocumentConverter()
    original_doc = converter.convert(str(test_doc)).document
    
    engine = CloakEngine()
    mask_result = engine.mask_document(original_doc)
    
    # Save masked document and CloakMap separately
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    masked_path = output_dir / "masked.json"
    masked_dict = mask_result.document.export_to_dict()
    
    import json
    masked_path.write_text(json.dumps(masked_dict, indent=2))
    
    cloakmap_path = output_dir / "document.cloakmap.json"
    mask_result.cloakmap.save_to_file(cloakmap_path)
    
    print(f"  ✓ Saved masked document: {masked_path}")
    print(f"  ✓ Saved CloakMap: {cloakmap_path}")
    
    # Simulate loading from files later
    print("\n  Loading from files...")
    
    # Load masked document
    from docling_core.types import DoclingDocument
    masked_data = json.loads(masked_path.read_text())
    masked_doc = DoclingDocument.model_validate(masked_data)
    
    # Load CloakMap
    loaded_cloakmap = CloakMap.load_from_file(cloakmap_path)
    
    print(f"  ✓ Loaded masked document")
    print(f"  ✓ Loaded CloakMap with {len(loaded_cloakmap.anchors)} anchors")
    
    # Unmask using loaded data
    unmasked_doc = engine.unmask_document(masked_doc, loaded_cloakmap)
    
    # Verify
    original_text = original_doc.export_to_markdown()
    recovered_text = unmasked_doc.export_to_markdown()
    
    if original_text == recovered_text:
        print("  ✓ Successfully unmasked from files!")
    else:
        print("  ⚠ Unmask verification failed")
    
    # Cleanup
    test_doc.unlink()
    masked_path.unlink()
    cloakmap_path.unlink()


def main():
    """Run all advanced examples."""
    print("="*60)
    print("CloakPivot Advanced Usage Examples")
    print("="*60)
    print("\nThese examples demonstrate advanced CloakEngine features.")
    
    try:
        # Run examples
        example_builder_pattern()
        example_policy_presets()
        example_confidence_thresholds()
        example_file_based_unmasking()
        
        print("\n" + "="*60)
        print("✓ All examples completed successfully!")
        print("="*60)
        print("\nKey advanced features demonstrated:")
        print("  • Builder pattern for flexible configuration")
        print("  • Custom policies with per-entity strategies")
        print("  • Policy presets (default, conservative, permissive)")
        print("  • Confidence threshold tuning")
        print("  • File-based CloakMap persistence")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
