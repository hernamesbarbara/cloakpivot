#!/usr/bin/env python
"""
Demonstrates the SURROGATE strategy with Faker integration for realistic fake data generation.

The SURROGATE strategy replaces PII with high-quality fake data that looks realistic
but is completely synthetic. This is useful when you need to:
- Maintain data realism for testing or demos
- Preserve format and structure while removing sensitive information
- Generate deterministic fake data with seed parameters
- Create realistic-looking documents for training or examples

Key features demonstrated:
1. Basic SURROGATE strategy usage
2. Deterministic generation with seed parameter
3. Comparison with other masking strategies
4. Entity-specific fake data generation
5. Format preservation in fake data
"""

import json
from pathlib import Path

from docling_core.types.doc.document import DocItemLabel, DoclingDocument, TextItem

from cloakpivot import CloakEngine, CloakEngineBuilder
from cloakpivot.core import MaskingPolicy, Strategy, StrategyKind



CLOAKPIVOT_ROOT = Path(__file__).parent.parent
DATA_DIR = CLOAKPIVOT_ROOT / "data"


def create_sample_document() -> DoclingDocument:
    """Create a sample document with various PII types."""
    doc = DoclingDocument(name="employee_records.txt")

    doc.texts = [
        TextItem(
            text=(
                "Employee Information\n"
                "====================\n\n"
                "Name: John Michael Doe\n"
                "Email: john.doe@techcorp.com\n"
                "Phone: (555) 123-4567\n"
                "SSN: 123-45-6789\n"
                "Date of Birth: 1985-03-15\n"
                "Address: 742 Evergreen Terrace, Springfield, IL 62704\n\n"
                "Emergency Contact: Jane Smith (555) 987-6543\n"
                "Direct Manager: Robert Johnson (robert.j@techcorp.com)\n"
            ),
            label=DocItemLabel.TEXT,
            self_ref="#/texts/0",
            orig=(
                "Employee Information\n"
                "====================\n\n"
                "Name: John Michael Doe\n"
                "Email: john.doe@techcorp.com\n"
                "Phone: (555) 123-4567\n"
                "SSN: 123-45-6789\n"
                "Date of Birth: 1985-03-15\n"
                "Address: 742 Evergreen Terrace, Springfield, IL 62704\n\n"
                "Emergency Contact: Jane Smith (555) 987-6543\n"
                "Direct Manager: Robert Johnson (robert.j@techcorp.com)\n"
            ),
        ),
    ]

    return doc


def demonstrate_basic_surrogate():
    """Demonstrate basic SURROGATE strategy usage."""
    print("\n=== Basic SURROGATE Strategy ===\n")

    # Create policy with SURROGATE strategy
    policy = MaskingPolicy(
        default_strategy=Strategy(kind=StrategyKind.SURROGATE)
    )

    engine = CloakEngine(default_policy=policy)
    doc = create_sample_document()

    # Mask the document
    result = engine.mask_document(doc)

    print("Original text:")
    print("-" * 40)
    print(doc.texts[0].text)

    print("\nMasked with SURROGATE (fake data):")
    print("-" * 40)
    print(result.document.texts[0].text)

    print(f"\nStatistics:")
    print(f"  Entities found: {result.entities_found}")
    print(f"  Entities masked: {result.entities_masked}")

    # Verify no asterisks in output
    masked_text = result.document.texts[0].text
    if "*" not in masked_text:
        print("  ✓ No asterisks - using high-quality fake data")

    return result


def demonstrate_deterministic_generation():
    """Demonstrate deterministic generation with seed parameter."""
    print("\n=== Deterministic Generation with Seed ===\n")

    # Create policy with seed for deterministic generation
    seed = "company-demo-2024"
    policy = MaskingPolicy(
        default_strategy=Strategy(
            kind=StrategyKind.SURROGATE,
            parameters={"seed": seed}
        )
    )

    engine = CloakEngine(default_policy=policy)
    doc = create_sample_document()

    # Mask the same document multiple times
    result1 = engine.mask_document(doc)
    result2 = engine.mask_document(doc)
    result3 = engine.mask_document(doc)

    print(f"Using seed: '{seed}'")
    print("-" * 40)

    print("\nFirst run:")
    print(result1.document.texts[0].text[:200] + "...")

    print("\nSecond run:")
    print(result2.document.texts[0].text[:200] + "...")

    print("\nThird run:")
    print(result3.document.texts[0].text[:200] + "...")

    # Verify deterministic generation
    if (result1.document.texts[0].text == result2.document.texts[0].text ==
        result3.document.texts[0].text):
        print("\n✓ All three runs produced identical results (deterministic)")
    else:
        print("\n✗ Results differ (should be deterministic with seed)")

    return result1


def compare_strategies():
    """Compare SURROGATE with other masking strategies."""
    print("\n=== Strategy Comparison ===\n")

    # Create a simple document for comparison
    doc = DoclingDocument(name="comparison.txt")
    doc.texts = [
        TextItem(
            text="Contact John Doe at john.doe@example.com or call 555-1234",
            label=DocItemLabel.TEXT,
            self_ref="#/texts/0",
            orig="Contact John Doe at john.doe@example.com or call 555-1234",
        )
    ]

    strategies = [
        ("REDACT", Strategy(kind=StrategyKind.REDACT)),
        ("TEMPLATE", Strategy(kind=StrategyKind.TEMPLATE,
                            parameters={"template": "[MASKED]"})),
        ("HASH", Strategy(kind=StrategyKind.HASH)),
        ("PARTIAL", Strategy(kind=StrategyKind.PARTIAL,
                           parameters={"visible_chars": 3, "position": "end"})),
        ("SURROGATE", Strategy(kind=StrategyKind.SURROGATE,
                             parameters={"seed": "demo"})),
    ]

    print("Original text:")
    print(f"  {doc.texts[0].text}\n")

    print("Masked with different strategies:")
    print("-" * 60)

    for name, strategy in strategies:
        policy = MaskingPolicy(default_strategy=strategy)
        engine = CloakEngine(default_policy=policy)
        result = engine.mask_document(doc)

        masked = result.document.texts[0].text
        print(f"{name:12} → {masked}")

    print("\nNote: SURROGATE produces realistic fake data while other strategies")
    print("use symbols, templates, or partial masking.")


def demonstrate_entity_specific_surrogates():
    """Show how different entity types get appropriate fake data."""
    print("\n=== Entity-Specific Fake Data ===\n")

    # Create document with various entity types
    doc = DoclingDocument(name="entities.txt")
    doc.texts = [
        TextItem(
            text=(
                "Personal: Alice Johnson\n"
                "Email: alice@company.org\n"
                "Phone: 555-0123\n"
                "Credit Card: 4111-1111-1111-1111\n"
                "Date: 2024-01-15\n"
                "Location: New York, NY\n"
            ),
            label=DocItemLabel.TEXT,
            self_ref="#/texts/0",
            orig=(
                "Personal: Alice Johnson\n"
                "Email: alice@company.org\n"
                "Phone: 555-0123\n"
                "Credit Card: 4111-1111-1111-1111\n"
                "Date: 2024-01-15\n"
                "Location: New York, NY\n"
            ),
        )
    ]

    # Use SURROGATE with seed for consistent results
    policy = MaskingPolicy(
        default_strategy=Strategy(
            kind=StrategyKind.SURROGATE,
            parameters={"seed": "entity-demo"}
        )
    )

    engine = CloakEngine(default_policy=policy)
    result = engine.mask_document(doc,
                                 entities=["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER",
                                         "CREDIT_CARD", "DATE_TIME", "LOCATION"])

    print("Original:")
    print("-" * 40)
    print(doc.texts[0].text)

    print("Masked with entity-appropriate fake data:")
    print("-" * 40)
    print(result.document.texts[0].text)

    print("\nObservations:")
    print("- Names are replaced with realistic fake names")
    print("- Emails maintain valid email format")
    print("- Phone numbers look like real phone numbers")
    print("- Credit cards use test card prefixes")
    print("- Dates remain in date format")
    print("- Locations are replaced with fake locations")


def demonstrate_mixed_strategies():
    """Show using SURROGATE for some entities and other strategies for others."""
    print("\n=== Mixed Strategies (SURROGATE + Others) ===\n")

    doc = create_sample_document()

    # Use SURROGATE for names/emails, REDACT for sensitive numbers
    policy = MaskingPolicy(
        default_strategy=Strategy(kind=StrategyKind.REDACT),
        per_entity={
            "PERSON": Strategy(kind=StrategyKind.SURROGATE,
                             parameters={"seed": "person-seed"}),
            "EMAIL_ADDRESS": Strategy(kind=StrategyKind.SURROGATE,
                                    parameters={"seed": "email-seed"}),
            "US_SSN": Strategy(kind=StrategyKind.TEMPLATE,
                             parameters={"template": "[SSN-REDACTED]"}),
            "PHONE_NUMBER": Strategy(kind=StrategyKind.PARTIAL,
                                   parameters={"visible_chars": 4, "position": "end"}),
        }
    )

    engine = CloakEngine(default_policy=policy)
    result = engine.mask_document(doc)

    print("Using mixed strategies:")
    print("-" * 40)
    print("- PERSON → SURROGATE (fake names)")
    print("- EMAIL_ADDRESS → SURROGATE (fake emails)")
    print("- US_SSN → TEMPLATE")
    print("- PHONE_NUMBER → PARTIAL")
    print("- Others → REDACT (default)")

    print("\nResult:")
    print("-" * 40)
    print(result.document.texts[0].text)


def save_results(result):
    """Save masked document and cloakmap for inspection."""
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    # Save masked document
    masked_path = output_dir / "surrogate_masked.json"
    with open(masked_path, "w") as f:
        json.dump(result.document.model_dump(), f, indent=2, default=str)

    # Save cloakmap
    cloakmap_path = output_dir / "surrogate.cloakmap"
    result.cloakmap.save_to_file(str(cloakmap_path))

    print(f"\n✓ Saved masked document to: {masked_path}")
    print(f"✓ Saved cloakmap to: {cloakmap_path}")

    return masked_path, cloakmap_path


def main():
    """Run all SURROGATE strategy demonstrations."""
    print("=" * 70)
    print("SURROGATE Strategy with Faker Integration")
    print("=" * 70)

    # Run demonstrations
    result = demonstrate_basic_surrogate()
    demonstrate_deterministic_generation()
    compare_strategies()
    demonstrate_entity_specific_surrogates()
    demonstrate_mixed_strategies()

    # Save results
    print("\n=== Saving Results ===")
    save_results(result)

    print("\n" + "=" * 70)
    print("Summary:")
    print("-" * 70)
    print("The SURROGATE strategy provides high-quality fake data generation:")
    print("• Realistic-looking replacements instead of symbols/asterisks")
    print("• Deterministic generation with seed parameter")
    print("• Format-preserving replacements")
    print("• Entity-specific fake data (names, emails, phones, etc.)")
    print("• Can be mixed with other strategies for flexible masking")
    print("=" * 70)


if __name__ == "__main__":
    main()