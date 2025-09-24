#!/usr/bin/env python3
"""Advanced CloakEngine builder features demonstration.

This example demonstrates advanced CloakEngine features not shown in other examples:
1. ConflictResolutionConfig - Control entity grouping/merging behavior
2. .with_conflict_resolution() builder method
3. .with_presidio_engine() explicit Presidio configuration
4. Direct DocPivotEngine usage for format conversion
"""
import sys
import json
from pathlib import Path
from docling.document_converter import DocumentConverter
from docpivot import DocPivotEngine, DoclingJsonReader
from cloakpivot import CloakEngine, MaskingPolicy, Strategy, StrategyKind
from cloakpivot.core.processing.normalization import ConflictResolutionConfig, ConflictResolutionStrategy

CLOAKPIVOT_ROOT = Path(__file__).parent.parent
DATA_DIR = CLOAKPIVOT_ROOT / "data"

def demonstrate_conflict_resolution():
    """Show how ConflictResolutionConfig controls entity grouping."""
    print("\n" + "=" * 60)
    print("Example 1: ConflictResolutionConfig - Entity Grouping Control")
    print("=" * 60)

    # Create test document with adjacent entities
    test_doc = Path("test_conflict.md")
    test_doc.write_text("""
Customer Details:
John Smith called from 555-1234 at john.smith@email.com
SSN: 123-45-6789, Credit Card: 4532-1111-2222-3333

Meeting with Sarah Johnson (sarah@company.com) tomorrow.
Her direct line: 555-5678.
    """)

    # Convert to DoclingDocument
    converter = DocumentConverter()
    doc = converter.convert(str(test_doc)).document

    # Create custom policy
    policy = MaskingPolicy(
        per_entity={
            "PERSON": Strategy(StrategyKind.TEMPLATE, {"template": "[NAME]"}),
            "EMAIL_ADDRESS": Strategy(StrategyKind.TEMPLATE, {"template": "[EMAIL]"}),
            "PHONE_NUMBER": Strategy(StrategyKind.TEMPLATE, {"template": "[PHONE]"}),
            "US_SSN": Strategy(StrategyKind.PARTIAL, {"visible_chars": 4, "position": "end"}),
            "CREDIT_CARD": Strategy(StrategyKind.REDACT, {}),
        },
        default_strategy=Strategy(StrategyKind.TEMPLATE, {"template": "[PII]"})
    )

    print("\n  Testing different merge thresholds:")
    print("  " + "-" * 40)

    # Test 1: No merging (threshold = 0)
    conflict_config = ConflictResolutionConfig(
        merge_threshold_chars=0  # Don't group adjacent entities
    )

    engine = (CloakEngine.builder()
        .with_custom_policy(policy)
        .with_conflict_resolution(conflict_config)
        .with_presidio_engine(True)  # Explicitly enable Presidio
        .build())

    result = engine.mask_document(doc)

    print(f"\n  Merge threshold = 0 (No grouping):")
    print(f"    • Entities found: {result.entities_found}")
    print(f"    • Entities masked: {result.entities_masked}")
    print(f"    • Anchors created: {len(result.cloakmap.anchors)}")

    # Test 2: Moderate merging (threshold = 10)
    conflict_config = ConflictResolutionConfig(
        merge_threshold_chars=10  # Group entities within 10 chars
    )

    engine = (CloakEngine.builder()
        .with_custom_policy(policy)
        .with_conflict_resolution(conflict_config)
        .with_presidio_engine(True)
        .build())

    result = engine.mask_document(doc)

    print(f"\n  Merge threshold = 10 (Moderate grouping):")
    print(f"    • Entities found: {result.entities_found}")
    print(f"    • Entities masked: {result.entities_masked}")
    print(f"    • Anchors created: {len(result.cloakmap.anchors)}")

    # Test 3: Aggressive merging (threshold = 50)
    conflict_config = ConflictResolutionConfig(
        merge_threshold_chars=50  # Group entities within 50 chars
    )

    engine = (CloakEngine.builder()
        .with_custom_policy(policy)
        .with_conflict_resolution(conflict_config)
        .with_presidio_engine(True)
        .build())

    result = engine.mask_document(doc)

    print(f"\n  Merge threshold = 50 (Aggressive grouping):")
    print(f"    • Entities found: {result.entities_found}")
    print(f"    • Entities masked: {result.entities_masked}")
    print(f"    • Anchors created: {len(result.cloakmap.anchors)}")

    # Show sample of masked content
    print("\n  Sample masked content (threshold=0):")
    print("  " + "-" * 40)
    masked_text = result.document.export_to_markdown()
    for line in masked_text.split('\n')[:5]:
        if line.strip():
            print(f"  {line}")

    # Cleanup
    test_doc.unlink()
    return result


def demonstrate_presidio_configuration():
    """Show explicit Presidio engine configuration options."""
    print("\n" + "=" * 60)
    print("Example 2: Presidio Engine Configuration")
    print("=" * 60)

    # Create test document
    test_doc = Path("test_presidio.md")
    test_doc.write_text("""
International Office Contacts:

UK Office: +44 20 7123 4567
- Manager: Elizabeth Windsor
- Email: e.windsor@royalcorp.co.uk

US Office: (555) 123-4567
- Manager: George Washington
- Email: g.washington@uscorp.com
- SSN: 987-65-4321

France Office: +33 1 42 68 53 00
- Manager: Marie Curie
- Email: m.curie@sciencecorp.fr
    """)

    converter = DocumentConverter()
    doc = converter.convert(str(test_doc)).document

    print("\n  Comparing Presidio on/off:")
    print("  " + "-" * 40)

    # Test with Presidio explicitly enabled
    engine_with_presidio = (CloakEngine.builder()
        .with_presidio_engine(True)
        .with_confidence_threshold(0.7)
        .with_languages(['en', 'fr'])  # Multi-language support
        .build())

    result_with = engine_with_presidio.mask_document(doc)

    print(f"\n  With Presidio (multi-language):")
    print(f"    • Entities found: {result_with.entities_found}")
    print(f"    • Entities masked: {result_with.entities_masked}")

    # Analyze entity types found
    entity_types = set()
    for anchor in result_with.cloakmap.anchors:
        if hasattr(anchor, 'entity_type'):
            entity_types.add(anchor.entity_type)
    print(f"    • Entity types: {', '.join(sorted(entity_types))}")

    # Test with Presidio disabled (would use fallback if implemented)
    engine_without_presidio = (CloakEngine.builder()
        .with_presidio_engine(False)
        .build())

    try:
        result_without = engine_without_presidio.mask_document(doc)
        print(f"\n  Without Presidio (fallback mode):")
        print(f"    • Entities found: {result_without.entities_found}")
        print(f"    • Entities masked: {result_without.entities_masked}")
    except Exception as e:
        print(f"\n  Without Presidio: Not implemented (expected)")

    # Cleanup
    test_doc.unlink()
    return result_with


def demonstrate_docpivot_conversion():
    """Show direct DocPivotEngine usage for format conversion."""
    print("\n" + "=" * 60)
    print("Example 3: Direct DocPivotEngine Format Conversion")
    print("=" * 60)

    # Create test document
    test_doc = Path("test_conversion.md")
    test_doc.write_text("""
# Quarterly Report

**Author:** Jane Doe (jane.doe@company.com)
**Date:** January 15, 2024

## Executive Summary

Our Q4 results show strong growth with revenue reaching $2.5M.
Key contact: Bob Smith (555-1234, bob@sales.com).

## Financial Details

- Revenue: $2,500,000
- Expenses: $1,800,000
- Net Profit: $700,000

Account details: 4532-1234-5678-9012
    """)

    print("\n  Step 1: Convert PDF/Markdown to DoclingDocument")
    converter = DocumentConverter()
    conv_result = converter.convert(str(test_doc))
    docling_doc = conv_result.document

    # Save as DoclingDocument JSON
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    docling_path = output_dir / "test_conversion.docling.json"
    docling_dict = docling_doc.export_to_dict()
    docling_path.write_text(json.dumps(docling_dict, indent=2))
    print(f"  ✓ Saved DoclingDocument: {docling_path}")

    print("\n  Step 2: Use DocPivotEngine for Lexical conversion")
    try:
        # Initialize DocPivot engine
        docpivot_engine = DocPivotEngine()

        # Method 1: Convert from file path
        lexical_result = docpivot_engine.convert_document(
            str(test_doc),
            output_format='lexical'
        )

        if lexical_result and hasattr(lexical_result, 'content'):
            lexical_path = output_dir / "test_conversion.lexical.json"

            # Handle both string and dict content
            if isinstance(lexical_result.content, str):
                lexical_data = json.loads(lexical_result.content)
            else:
                lexical_data = lexical_result.content

            lexical_path.write_text(json.dumps(lexical_data, indent=2))
            print(f"  ✓ Converted to Lexical format: {lexical_path}")

            # Show structure
            if isinstance(lexical_data, dict):
                print(f"  ✓ Lexical structure: {list(lexical_data.keys())[:5]}")

        # Method 2: Convert DoclingDocument directly
        print("\n  Step 3: Convert DoclingDocument to Lexical directly")
        lexical_doc = docpivot_engine.convert_to_lexical(docling_doc)

        if lexical_doc:
            print(f"  ✓ Direct conversion successful")
            if hasattr(lexical_doc, 'root'):
                print(f"  ✓ Root type: {type(lexical_doc.root).__name__}")

    except Exception as e:
        print(f"  ⚠ DocPivot conversion: {e}")
        print(f"  Note: This is optional - masking works without it")

    print("\n  Step 4: Mask the DoclingDocument")

    # Create engine with all advanced features
    engine = (CloakEngine.builder()
        .with_presidio_engine(True)
        .with_confidence_threshold(0.8)
        .with_conflict_resolution(ConflictResolutionConfig(merge_threshold_chars=5))
        .build())

    mask_result = engine.mask_document(docling_doc)

    print(f"  ✓ Masking complete:")
    print(f"    • Entities found: {mask_result.entities_found}")
    print(f"    • Entities masked: {mask_result.entities_masked}")

    # Save masked document
    masked_path = output_dir / "test_conversion.masked.json"
    masked_dict = mask_result.document.export_to_dict()
    masked_path.write_text(json.dumps(masked_dict, indent=2))
    print(f"  ✓ Saved masked document: {masked_path}")

    # Cleanup
    test_doc.unlink()
    for file in [docling_path, masked_path]:
        if file.exists():
            file.unlink()

    # Try to cleanup lexical file if it exists
    lexical_path = output_dir / "test_conversion.lexical.json"
    if lexical_path.exists():
        lexical_path.unlink()

    return mask_result


def demonstrate_combined_features():
    """Combine all advanced features in one comprehensive example."""
    print("\n" + "=" * 60)
    print("Example 4: Combined Advanced Features")
    print("=" * 60)

    # Use existing test data
    json_dir = Path("data/json")
    docling_files = list(json_dir.glob("*.docling.json"))

    if not docling_files:
        print("  ⚠ No test data found in data/json/")
        return None

    # Use first available file
    test_file = docling_files[0]
    print(f"\n  Using test file: {test_file.name}")

    # Load DoclingDocument
    reader = DoclingJsonReader()
    doc = reader.load_data(test_file)

    # Create comprehensive configuration
    print("\n  Building engine with all advanced features:")

    # Custom conflict resolution
    conflict_config = ConflictResolutionConfig(
        merge_threshold_chars=15,  # Moderate grouping
        strategy=ConflictResolutionStrategy.MOST_SPECIFIC
    )

    # Custom policy
    policy = MaskingPolicy(
        per_entity={
            "EMAIL_ADDRESS": Strategy(
                StrategyKind.PARTIAL,
                {"visible_chars": 0, "position": "end", "mask_char": "*"}
            ),
            "PERSON": Strategy(StrategyKind.TEMPLATE, {"template": "[NAME]"}),
            "PHONE_NUMBER": Strategy(StrategyKind.REDACT, {}),
            "CREDIT_CARD": Strategy(StrategyKind.REDACT, {}),
            "US_SSN": Strategy(
                StrategyKind.PARTIAL,
                {"visible_chars": 4, "position": "end"}
            ),
        },
        default_strategy=Strategy(StrategyKind.TEMPLATE, {"template": "[MASKED]"})
    )

    # Build engine with all features
    engine = (CloakEngine.builder()
        .with_custom_policy(policy)
        .with_presidio_engine(True)
        .with_conflict_resolution(conflict_config)
        .with_confidence_threshold(0.75)
        .with_languages(['en'])
        .build())

    print("  ✓ Engine configured with:")
    print("    • Custom masking policy (5 entity types)")
    print("    • Presidio engine enabled")
    print("    • Conflict resolution (15 char threshold)")
    print("    • Confidence threshold: 0.75")
    print("    • Language: English")

    # Mask document
    result = engine.mask_document(doc)

    print(f"\n  Masking results:")
    print(f"    • Entities found: {result.entities_found}")
    print(f"    • Entities masked: {result.entities_masked}")
    print(f"    • CloakMap anchors: {len(result.cloakmap.anchors)}")

    # Analyze entity distribution
    entity_counts = {}
    for anchor in result.cloakmap.anchors:
        if hasattr(anchor, 'entity_type'):
            entity_type = anchor.entity_type
            entity_counts[entity_type] = entity_counts.get(entity_type, 0) + 1

    if entity_counts:
        print(f"\n  Entity distribution:")
        for entity_type, count in sorted(entity_counts.items()):
            print(f"    • {entity_type}: {count}")

    # Save results
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    output_name = test_file.stem.replace('.docling', '')
    masked_path = output_dir / f"{output_name}.advanced.masked.json"
    cloakmap_path = output_dir / f"{output_name}.advanced.cloakmap.json"

    # Save masked document
    masked_dict = result.document.export_to_dict()
    masked_path.write_text(json.dumps(masked_dict, indent=2))

    # Save CloakMap
    result.cloakmap.save_to_file(cloakmap_path)

    print(f"\n  Output saved:")
    print(f"    • Masked: {masked_path.name}")
    print(f"    • CloakMap: {cloakmap_path.name}")

    # Verify round-trip
    print(f"\n  Verifying round-trip unmask...")
    unmasked = engine.unmask_document(result.document, result.cloakmap)

    original_text = doc.export_to_markdown()
    recovered_text = unmasked.export_to_markdown()

    if original_text == recovered_text:
        print("  ✓ Perfect round-trip restoration!")
    else:
        print("  ⚠ Round-trip content differs")

    # Cleanup output files
    masked_path.unlink()
    cloakmap_path.unlink()

    return result


def main():
    """Run all advanced builder feature examples."""
    print("=" * 60)
    print("CloakPivot Advanced Builder Features")
    print("=" * 60)
    print("\nDemonstrating features not shown in other examples:")
    print("• ConflictResolutionConfig for entity grouping control")
    print("• .with_conflict_resolution() builder method")
    print("• .with_presidio_engine() explicit configuration")
    print("• Direct DocPivotEngine usage for format conversion")

    try:
        # Run demonstrations
        demonstrate_conflict_resolution()
        demonstrate_presidio_configuration()
        demonstrate_docpivot_conversion()
        demonstrate_combined_features()

        print("\n" + "=" * 60)
        print("✓ All advanced features demonstrated successfully!")
        print("=" * 60)

        print("\nKey takeaways:")
        print("• merge_threshold_chars controls how adjacent entities are grouped")
        print("• .with_presidio_engine(True) explicitly enables Presidio")
        print("• .with_conflict_resolution() customizes entity handling")
        print("• DocPivotEngine enables format conversion (optional)")
        print("• Builder pattern allows combining all features flexibly")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
