#!/usr/bin/env python3
"""
Example demonstrating Presidio integration for PII detection and normalization.

This example shows how to:
1. Configure Presidio analyzer with custom settings
2. Detect PII entities in text segments
3. Normalize and resolve conflicts between overlapping entities
4. Map entities back to document anchors
5. Apply masking policies based on detected entities

Run this example with:
    python examples/presidio_integration_example.py
"""

import logging

from cloakpivot.core.analyzer import AnalyzerEngineWrapper, EntityDetectionResult
from cloakpivot.core.detection import DocumentAnalysisResult, EntityDetectionPipeline
from cloakpivot.core.normalization import (
    ConflictResolutionConfig,
    ConflictResolutionStrategy,
    EntityNormalizer,
)
from cloakpivot.core.policies import MaskingPolicy, Strategy, StrategyKind
from cloakpivot.document.extractor import TextSegment

# Configure logging to see detailed information
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def create_sample_text_segments() -> list[TextSegment]:
    """Create sample text segments for demonstration."""
    # Create segments with correct length calculations
    text1 = "John Smith is our lead developer. You can reach him at john.smith@company.com or call 555-123-4567."
    text2 = "For security purposes, his employee ID is EMP-2023-001 and SSN is 123-45-6789."
    text3 = "Personal Information"
    text4 = "Jane Doe works in HR. Contact her at jane.doe@company.com. Phone: 555-987-6543 or 555-987-6544."

    segments = [
        TextSegment(
            node_id="#/texts/0",
            text=text1,
            start_offset=0,
            end_offset=len(text1),
            node_type="TextItem",
            metadata={"section": "employee_info"}
        ),
        TextSegment(
            node_id="#/texts/1",
            text=text2,
            start_offset=len(text1) + 1,  # +1 for separator
            end_offset=len(text1) + 1 + len(text2),
            node_type="TextItem",
            metadata={"section": "confidential"}
        ),
        TextSegment(
            node_id="#/headers/0",
            text=text3,
            start_offset=len(text1) + 1 + len(text2) + 1,
            end_offset=len(text1) + 1 + len(text2) + 1 + len(text3),
            node_type="TitleItem",
            metadata={"level": 1}
        ),
        TextSegment(
            node_id="#/texts/2",
            text=text4,
            start_offset=len(text1) + 1 + len(text2) + 1 + len(text3) + 1,
            end_offset=len(text1) + 1 + len(text2) + 1 + len(text3) + 1 + len(text4),
            node_type="TextItem",
            metadata={"section": "employee_info"}
        ),
    ]
    return segments


def demonstrate_basic_analysis():
    """Demonstrate basic PII entity detection."""
    print("\n" + "="*60)
    print("1. BASIC PII ENTITY DETECTION")
    print("="*60)

    # Create analyzer with default configuration
    analyzer = AnalyzerEngineWrapper()
    print(f"Created analyzer with language: {analyzer.config.language}")

    # Analyze a simple text
    sample_text = "Contact John Doe at john.doe@email.com or 555-123-4567"
    print(f"\nAnalyzing text: '{sample_text}'")

    # Note: This would normally use Presidio, but we'll simulate results
    # since Presidio may not be fully configured in test environment
    mock_entities = [
        EntityDetectionResult("PERSON", 8, 16, 0.9, "John Doe"),
        EntityDetectionResult("EMAIL_ADDRESS", 20, 37, 0.95, "john.doe@email.com"),
        EntityDetectionResult("PHONE_NUMBER", 41, 53, 0.85, "555-123-4567")
    ]

    print(f"Detected {len(mock_entities)} entities:")
    for entity in mock_entities:
        print(f"  - {entity.entity_type}: '{entity.text}' (confidence: {entity.confidence:.2f})")


def demonstrate_policy_configuration():
    """Demonstrate policy-based configuration."""
    print("\n" + "="*60)
    print("2. POLICY-BASED CONFIGURATION")
    print("="*60)

    # Create a masking policy with specific requirements
    policy = MaskingPolicy(
        locale="en",
        thresholds={
            "PERSON": 0.8,
            "EMAIL_ADDRESS": 0.9,
            "PHONE_NUMBER": 0.7,
            "US_SSN": 0.95
        },
        per_entity={
            "PERSON": Strategy(StrategyKind.TEMPLATE, {"template": "[NAME]"}),
            "EMAIL_ADDRESS": Strategy(StrategyKind.PARTIAL, {"visible_chars": 3, "position": "start"}),
            "PHONE_NUMBER": Strategy(StrategyKind.TEMPLATE, {"template": "[PHONE]"}),
            "US_SSN": Strategy(StrategyKind.REDACT, {"redact_char": "X"})
        },
        context_rules={
            "heading": {"enabled": False},  # Don't mask PII in headings
            "table": {"threshold": 0.9}     # Higher threshold for table content
        }
    )

    print("Created masking policy with:")
    print(f"  - Locale: {policy.locale}")
    print(f"  - Entity thresholds: {policy.thresholds}")
    print(f"  - Context rules: {len(policy.context_rules)} rules")

    # Create analyzer from policy
    analyzer = AnalyzerEngineWrapper.from_policy(policy)
    print(f"  - Analyzer configured for language: {analyzer.config.language}")

    # Demonstrate policy filtering
    print("\nPolicy filtering example:")
    entities = [
        EntityDetectionResult("PERSON", 0, 8, 0.85, "John Doe"),  # Above threshold (0.8)
        EntityDetectionResult("PERSON", 10, 18, 0.75, "Jane Smith"),  # Below threshold (0.8)
        EntityDetectionResult("EMAIL_ADDRESS", 20, 37, 0.92, "test@example.com"),  # Above threshold (0.9)
    ]

    for entity in entities:
        should_mask = policy.should_mask_entity(entity.text, entity.entity_type, entity.confidence)
        status = "MASK" if should_mask else "SKIP"
        print(f"  - {entity.entity_type} (conf: {entity.confidence:.2f}): {status}")


def demonstrate_conflict_resolution():
    """Demonstrate entity conflict resolution."""
    print("\n" + "="*60)
    print("3. ENTITY CONFLICT RESOLUTION")
    print("="*60)

    # Create overlapping entities to demonstrate conflict resolution
    conflicting_entities = [
        EntityDetectionResult("PERSON", 0, 10, 0.9, "John Smith"),
        EntityDetectionResult("PERSON", 5, 15, 0.7, "Smith John"),      # Overlaps with first
        EntityDetectionResult("EMAIL_ADDRESS", 20, 37, 0.95, "john@example.com"),
        EntityDetectionResult("URL", 20, 40, 0.6, "http://john@example.com"),  # Overlaps with email
        EntityDetectionResult("PHONE_NUMBER", 50, 62, 0.85, "555-123-4567"),
        EntityDetectionResult("PHONE_NUMBER", 65, 77, 0.80, "555-987-6543"),   # Adjacent (3 char gap)
    ]

    print(f"Starting with {len(conflicting_entities)} entities (some overlapping):")
    for i, entity in enumerate(conflicting_entities):
        print(f"  {i+1}. {entity.entity_type}[{entity.start}-{entity.end}]: '{entity.text}' (conf: {entity.confidence:.2f})")

    # Test different resolution strategies
    strategies = [
        ConflictResolutionStrategy.HIGHEST_CONFIDENCE,
        ConflictResolutionStrategy.LONGEST_ENTITY,
        ConflictResolutionStrategy.MOST_SPECIFIC,
        ConflictResolutionStrategy.MERGE_ADJACENT
    ]

    for strategy in strategies:
        print(f"\n--- Using {strategy.value} strategy ---")

        config = ConflictResolutionConfig(
            strategy=strategy,
            merge_threshold_chars=5  # Allow merging entities within 5 characters
        )
        normalizer = EntityNormalizer(config)

        result = normalizer.normalize_entities(conflicting_entities)

        print(f"Resolved to {len(result.normalized_entities)} entities:")
        for entity in result.normalized_entities:
            print(f"  - {entity.entity_type}[{entity.start}-{entity.end}]: '{entity.text}' (conf: {entity.confidence:.2f})")

        print(f"  Conflicts resolved: {result.conflicts_resolved}")
        print(f"  Entities merged: {result.entities_merged}")


def demonstrate_pipeline_integration():
    """Demonstrate full pipeline integration."""
    print("\n" + "="*60)
    print("4. FULL PIPELINE INTEGRATION")
    print("="*60)

    # Create sample text segments
    segments = create_sample_text_segments()
    print(f"Created {len(segments)} text segments:")
    for i, segment in enumerate(segments):
        print(f"  {i+1}. {segment.node_type}[{segment.start_offset}-{segment.end_offset}]: '{segment.text[:50]}...'")

    # Create detection pipeline with policy
    policy = MaskingPolicy(
        thresholds={"PERSON": 0.8, "EMAIL_ADDRESS": 0.9, "PHONE_NUMBER": 0.7, "US_SSN": 0.9},
        context_rules={"heading": {"enabled": False}}  # Don't detect PII in headings
    )

    pipeline = EntityDetectionPipeline.from_policy(policy)
    print(f"\nCreated detection pipeline with language: {pipeline.analyzer.config.language}")

    # Simulate entity detection (in real usage, this would call Presidio)
    print("\nSimulating entity detection...")

    # Mock detected entities for demonstration
    mock_segment_entities = [
        # Segment 0: Employee info
        [
            EntityDetectionResult("PERSON", 0, 10, 0.9, "John Smith"),
            EntityDetectionResult("EMAIL_ADDRESS", 45, 69, 0.95, "john.smith@company.com"),
            EntityDetectionResult("PHONE_NUMBER", 78, 90, 0.85, "555-123-4567")
        ],
        # Segment 1: Confidential info
        [
            EntityDetectionResult("US_SSN", 54, 66, 0.98, "123-45-6789")
        ],
        # Segment 2: Header (should be filtered out by policy)
        [],
        # Segment 3: More employee info
        [
            EntityDetectionResult("PERSON", 0, 8, 0.85, "Jane Doe"),
            EntityDetectionResult("EMAIL_ADDRESS", 33, 55, 0.92, "jane.doe@company.com"),
            EntityDetectionResult("PHONE_NUMBER", 64, 76, 0.88, "555-987-6543"),
            EntityDetectionResult("PHONE_NUMBER", 80, 92, 0.82, "555-987-6544")
        ]
    ]

    # Create analysis results
    analysis_result = DocumentAnalysisResult("sample_document")

    for segment, entities in zip(segments, mock_segment_entities):
        # Apply policy filtering
        filtered_entities = []
        context = "heading" if segment.node_type == "TitleItem" else None

        for entity in entities:
            if policy.should_mask_entity(entity.text, entity.entity_type, entity.confidence, context):
                filtered_entities.append(entity)

        from cloakpivot.core.detection import SegmentAnalysisResult
        segment_result = SegmentAnalysisResult(segment=segment, entities=filtered_entities)
        analysis_result.add_segment_result(segment_result)

    print("Analysis completed:")
    print(f"  - Total entities detected: {analysis_result.total_entities}")
    print(f"  - Entity breakdown: {analysis_result.entity_breakdown}")
    print(f"  - Success rate: {analysis_result.success_rate:.2%}")

    # Map entities to anchors
    anchors = pipeline.map_entities_to_anchors(analysis_result)
    print(f"\nCreated {len(anchors)} anchor mappings:")
    for anchor in anchors[:3]:  # Show first 3
        print(f"  - Anchor {anchor.anchor_id}: {anchor.entity_type} at {anchor.node_id}[{anchor.start_offset}-{anchor.end_offset}]")

    # Normalize entities
    all_entities = [entity for entity, segment in analysis_result.get_all_entities()]

    normalizer = EntityNormalizer(ConflictResolutionConfig(
        strategy=ConflictResolutionStrategy.HIGHEST_CONFIDENCE
    ))

    normalization_result = normalizer.normalize_entities(all_entities)
    print("\nNormalization result:")
    print(f"  - Original entities: {len(all_entities)}")
    print(f"  - Normalized entities: {len(normalization_result.normalized_entities)}")
    print(f"  - Conflicts resolved: {normalization_result.conflicts_resolved}")


def demonstrate_diagnostic_capabilities():
    """Demonstrate diagnostic and validation capabilities."""
    print("\n" + "="*60)
    print("5. DIAGNOSTIC CAPABILITIES")
    print("="*60)

    # Test analyzer diagnostics
    analyzer = AnalyzerEngineWrapper()
    diagnostics = analyzer.validate_configuration()

    print("Analyzer diagnostics:")
    print(f"  - Configuration valid: {diagnostics['config_valid']}")
    print(f"  - Language: {diagnostics['language']}")
    print(f"  - Enabled recognizers: {len(diagnostics['enabled_recognizers'])}")
    print(f"  - Custom recognizers: {len(diagnostics['custom_recognizers'])}")
    print(f"  - Warnings: {len(diagnostics.get('warnings', []))}")

    if diagnostics.get('errors'):
        print(f"  - Errors: {diagnostics['errors']}")

    # Test normalization validation
    entities = [
        EntityDetectionResult("PERSON", 0, 8, 0.9, "John Doe"),
        EntityDetectionResult("EMAIL_ADDRESS", 10, 27, 0.95, "john@example.com")
    ]

    normalizer = EntityNormalizer()
    result = normalizer.normalize_entities(entities)
    validation = normalizer.validate_normalization(entities, result.normalized_entities)

    print("\nNormalization validation:")
    print(f"  - Original count: {validation['original_count']}")
    print(f"  - Normalized count: {validation['normalized_count']}")
    print(f"  - Sorted correctly: {validation['sorted_correctly']}")
    print(f"  - No overlaps: {validation['no_overlaps']}")
    print(f"  - High confidence preserved: {validation['high_confidence_preserved']}")

    if validation['warnings']:
        print(f"  - Warnings: {validation['warnings']}")


def main():
    """Run all demonstrations."""
    print("PRESIDIO INTEGRATION DEMONSTRATION")
    print("=" * 60)
    print("This example demonstrates the CloakPivot Presidio integration features.")
    print("Note: Some features are mocked for demonstration purposes.")

    try:
        demonstrate_basic_analysis()
        demonstrate_policy_configuration()
        demonstrate_conflict_resolution()
        demonstrate_pipeline_integration()
        demonstrate_diagnostic_capabilities()

        print("\n" + "="*60)
        print("✓ DEMONSTRATION COMPLETED SUCCESSFULLY")
        print("="*60)
        print("\nKey features demonstrated:")
        print("  ✓ Presidio AnalyzerEngine wrapper with configuration")
        print("  ✓ Policy-based entity filtering and thresholds")
        print("  ✓ Entity conflict resolution with multiple strategies")
        print("  ✓ Full detection pipeline with text segments")
        print("  ✓ Entity-to-anchor mapping for document positions")
        print("  ✓ Comprehensive diagnostic and validation tools")

    except Exception as e:
        print(f"\n✗ DEMONSTRATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
