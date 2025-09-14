#!/usr/bin/env python3
"""Integration with existing DoclingDocument files.

This example shows how to work with pre-converted DoclingDocument JSON files,
which is useful when you have a pipeline that already uses Docling for document
processing and you want to add PII masking as a downstream step.

This demonstrates:
1. Loading DoclingDocument from JSON
2. Masking with custom policies
3. Saving and loading CloakMaps
4. Batch processing multiple documents
"""

import json
from pathlib import Path
from typing import List, Tuple
from docling_core.types import DoclingDocument
from cloakpivot import CloakEngine
from cloakpivot.core.policies import MaskingPolicy
from cloakpivot.core.strategies import Strategy, StrategyKind
from cloakpivot.core.cloakmap import CloakMap


def load_docling_document(json_path: Path) -> DoclingDocument:
    """Load a DoclingDocument from a JSON file."""
    with open(json_path, "r", encoding="utf-8") as f:
        doc_dict = json.load(f)
    return DoclingDocument(**doc_dict)


def create_custom_policy() -> MaskingPolicy:
    """Create a custom masking policy for sensitive documents."""
    return MaskingPolicy(
        per_entity={
            # Emails: Show domain only
            "EMAIL_ADDRESS": Strategy(
                StrategyKind.PARTIAL,
                {"visible_chars": 0, "position": "end", "mask_char": "*"}
            ),
            # Names: Use consistent replacement
            "PERSON": Strategy(
                StrategyKind.TEMPLATE,
                {"template": "[PERSON]"}
            ),
            # Phone: Redact completely
            "PHONE_NUMBER": Strategy(
                StrategyKind.REDACT,
                {}
            ),
            # SSN: Show last 4 only
            "US_SSN": Strategy(
                StrategyKind.PARTIAL,
                {"visible_chars": 4, "position": "end"}
            ),
            # Credit cards: Redact
            "CREDIT_CARD": Strategy(
                StrategyKind.REDACT,
                {}
            ),
            # Locations: Keep for context
            "LOCATION": Strategy(
                StrategyKind.KEEP,
                {}
            ),
            # Dates: Keep for context
            "DATE_TIME": Strategy(
                StrategyKind.KEEP,
                {}
            ),
        },
        default_strategy=Strategy(
            StrategyKind.TEMPLATE,
            {"template": "[CONFIDENTIAL]"}
        )
    )


def process_document(
    doc_path: Path,
    engine: CloakEngine,
    output_dir: Path
) -> Tuple[Path, Path]:
    """Process a single DoclingDocument file."""
    print(f"\n📄 Processing: {doc_path.name}")

    # Load document
    doc = load_docling_document(doc_path)
    doc_name = doc_path.stem

    # Mask PII
    result = engine.mask_document(doc)
    print(f"  ✓ Found {result.entities_found} entities")
    print(f"  ✓ Masked {result.entities_masked} entities")

    # Collect entity types
    entity_types = set()
    for anchor in result.cloakmap.anchors:
        if hasattr(anchor, 'entity_type'):
            entity_types.add(anchor.entity_type)
    if entity_types:
        print(f"  ✓ Types: {', '.join(sorted(entity_types))}")

    # Save masked document
    masked_path = output_dir / f"{doc_name}.masked.json"
    with open(masked_path, "w", encoding="utf-8") as f:
        json.dump(result.document.export_to_dict(), f, indent=2)

    # Save CloakMap
    cloakmap_path = output_dir / f"{doc_name}.cloakmap.json"
    result.cloakmap.save_to_file(cloakmap_path)

    print(f"  ✓ Saved: {masked_path.name}")
    print(f"  ✓ CloakMap: {cloakmap_path.name}")

    return masked_path, cloakmap_path


def batch_process_documents(
    input_dir: Path,
    output_dir: Path,
    pattern: str = "*.docling.json"
) -> List[Tuple[Path, Path]]:
    """Process all DoclingDocument files in a directory."""
    print(f"\n🔍 Searching for {pattern} in {input_dir}")

    # Find all matching files
    doc_files = list(input_dir.glob(pattern))
    if not doc_files:
        print(f"  ⚠️ No files found matching {pattern}")
        return []

    print(f"  ✓ Found {len(doc_files)} documents to process")

    # Create engine with custom policy
    engine = CloakEngine.builder() \
        .with_custom_policy(create_custom_policy()) \
        .with_confidence_threshold(0.7) \
        .build()

    # Process each document
    results = []
    for doc_path in doc_files:
        masked_path, cloakmap_path = process_document(
            doc_path, engine, output_dir
        )
        results.append((masked_path, cloakmap_path))

    return results


def verify_unmasking(
    masked_path: Path,
    cloakmap_path: Path,
    original_path: Path
) -> bool:
    """Verify that unmasking restores the original document."""
    print(f"\n🔓 Verifying: {masked_path.name}")

    # Load original
    original_doc = load_docling_document(original_path)
    original_text = original_doc.export_to_markdown()

    # Load masked and CloakMap
    masked_doc = load_docling_document(masked_path)
    cloakmap = CloakMap.load_from_file(cloakmap_path)

    # Unmask
    engine = CloakEngine()
    unmasked_doc = engine.unmask_document(masked_doc, cloakmap)
    unmasked_text = unmasked_doc.export_to_markdown()

    # Compare
    if original_text == unmasked_text:
        print("  ✓ Perfect restoration!")
        return True
    else:
        print("  ⚠️ Content differs")
        return False


def main():
    """Demonstrate DoclingDocument integration."""
    print("=" * 60)
    print("CloakPivot DoclingDocument Integration Example")
    print("=" * 60)

    # Setup paths
    input_dir = Path("data/docling")
    output_dir = Path("output/masked")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create sample DoclingDocument if needed
    sample_path = input_dir / "sample.docling.json"
    if not sample_path.exists():
        print(f"\n📝 Creating sample DoclingDocument...")
        input_dir.mkdir(parents=True, exist_ok=True)

        from docling_core.types import DoclingDocument
        from docling_core.types.doc.document import TextItem

        doc = DoclingDocument(name="sample.pdf")
        doc.texts = [
            TextItem(
                text="Contact John Smith at john.smith@example.com or 555-123-4567.",
                self_ref="#/texts/0",
                label="text",
                orig="Contact John Smith at john.smith@example.com or 555-123-4567."
            ),
            TextItem(
                text="Payment card: 4111-1111-1111-1111, SSN: 123-45-6789",
                self_ref="#/texts/1",
                label="text",
                orig="Payment card: 4111-1111-1111-1111, SSN: 123-45-6789"
            )
        ]

        with open(sample_path, "w") as f:
            json.dump(doc.export_to_dict(), f, indent=2)
        print(f"  ✓ Created: {sample_path}")

    # Process documents
    print("\n🚀 Processing documents...")
    results = batch_process_documents(input_dir, output_dir)

    if results:
        print(f"\n✅ Processed {len(results)} documents")

        # Verify first document
        masked_path, cloakmap_path = results[0]
        original_path = input_dir / masked_path.name.replace('.masked', '')
        verify_unmasking(masked_path, cloakmap_path, original_path)

    print("\n" + "=" * 60)
    print("Integration example complete!")
    print("Check output/masked/ for results")
    print("=" * 60)


if __name__ == "__main__":
    main()