# Debug Scripts

This directory contains debug scripts that were useful during bug investigation and may be helpful for future debugging.

## debug_entity_grouping.py

Created during investigation of the "unmasked PERSON entities" bug (2025-09-03).

**Purpose**: Demonstrates how the EntityNormalizer groups entities and how the `merge_threshold_chars` parameter affects grouping of adjacent entities.

**Use Case**: Helpful for understanding entity conflict resolution behavior and debugging issues with entity grouping.

**Example Usage**:
```bash
python tests/debug/debug_entity_grouping.py
```

**Key Finding**: Non-overlapping entities that are adjacent (within `merge_threshold_chars` distance) get grouped together, which can cause unexpected entity dropping during conflict resolution.