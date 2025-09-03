#!/usr/bin/env python3
"""Debug why non-overlapping entities are being grouped together."""

import sys
from pathlib import Path

from cloakpivot.core.normalization import (
    ConflictResolutionConfig,
    EntityDetectionResult,
    EntityNormalizer,
)

sys.path.insert(0, str(Path(__file__).parent))

# Test the exact scenario
text = "From: Cameron MacIntyre <cameron@example.com>"

# These are the positions in text item 1 (local positions)
entities_info = [
    {
        "type": "PERSON",
        "text": "Cameron MacIntyre",
        "start": 6,
        "end": 23,
        "score": 0.85,
    },
    {
        "type": "EMAIL_ADDRESS",
        "text": "cameron@example.com",
        "start": 25,
        "end": 44,
        "score": 1.00,
    },
    {"type": "URL", "text": "example.com", "start": 33, "end": 44, "score": 0.50},
]

print("Entities in text item 1:")
print(f"Text: '{text}'")
print("-" * 60)

for e in entities_info:
    print(
        f"{e['type']}: [{e['start']}, {e['end']}) = '{e['text']}' (score: {e['score']})"
    )

print("\nChecking overlaps:")
for i in range(len(entities_info)):
    for j in range(i + 1, len(entities_info)):
        e1, e2 = entities_info[i], entities_info[j]
        # Check if they overlap
        if e1["start"] < e2["end"] and e2["start"] < e1["end"]:
            print(f"  OVERLAP: {e1['type']} and {e2['type']}")
        else:
            print(f"  NO OVERLAP: {e1['type']} and {e2['type']}")

# Now test with the actual EntityNormalizer
print("\n" + "=" * 60)
print("Testing with EntityNormalizer:")

# Create EntityDetectionResult objects
detection_results = []
for e in entities_info:
    result = EntityDetectionResult(
        start=e["start"],
        end=e["end"],
        entity_type=e["type"],
        text=e["text"],
        confidence=e["score"],
    )
    detection_results.append(result)

# Create normalizer with default config
config = ConflictResolutionConfig()
print(
    f"Config: strategy={config.strategy.value}, preserve_high_confidence={config.preserve_high_confidence}"
)
print(f"Config: merge_threshold_chars={config.merge_threshold_chars}")

normalizer = EntityNormalizer(config)

# Normalize entities
result = normalizer.normalize_entities(detection_results)

print(f"\nInput: {len(detection_results)} entities")
print(f"Output: {len(result.normalized_entities)} entities")
print(f"Conflicts resolved: {result.conflicts_resolved}")

print("\nNormalized entities:")
for e in result.normalized_entities:
    print(f"  {e.entity_type}: [{e.start}, {e.end}) = '{e.text}'")

print("\nResolution details:")
for detail in result.resolution_details:
    print(
        f"  Group {detail['group_index']}: {detail['original_count']} -> {detail['resolved_count']}"
    )
    print(f"    Strategy: {detail['strategy_used']}")
    print("    Entities involved:")
    for ent in detail["entities_involved"]:
        print(f"      - {ent['type']}: '{ent['text']}' (conf: {ent['confidence']})")
