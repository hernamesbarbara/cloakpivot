"""Common utilities for Presidio integration.

This module contains shared functionality between the masking and unmasking
Presidio adapters to reduce code duplication and maintain consistency.
"""

import logging
from typing import Any

from presidio_analyzer import RecognizerResult
from presidio_anonymizer.entities import OperatorResult

logger = logging.getLogger(__name__)


# Common entity type mappings
ENTITY_TYPE_MAPPING = {
    # Presidio standard types
    "CREDIT_CARD": "CREDIT_CARD",
    "EMAIL_ADDRESS": "EMAIL",
    "EMAIL": "EMAIL",
    "PHONE_NUMBER": "PHONE_NUMBER",
    "PHONE": "PHONE_NUMBER",
    "PERSON": "PERSON",
    "LOCATION": "LOCATION",
    "DATE_TIME": "DATE_TIME",
    "IP_ADDRESS": "IP_ADDRESS",
    "US_SSN": "US_SSN",
    "SSN": "US_SSN",
    "IBAN": "IBAN",
    "URL": "URL",
    "MEDICAL_LICENSE": "MEDICAL_LICENSE",
    "US_DRIVER_LICENSE": "US_DRIVER_LICENSE",
    # Custom types
    "ORGANIZATION": "ORGANIZATION",
    "ADDRESS": "ADDRESS",
    "CURRENCY": "CURRENCY",
    "AGE": "AGE",
}

# Default confidence thresholds by entity type
DEFAULT_CONFIDENCE_THRESHOLDS = {
    "CREDIT_CARD": 0.7,
    "EMAIL": 0.5,
    "PHONE_NUMBER": 0.6,
    "US_SSN": 0.8,
    "PERSON": 0.6,
    "LOCATION": 0.5,
    "DATE_TIME": 0.5,
    "IP_ADDRESS": 0.7,
    "URL": 0.5,
    "IBAN": 0.7,
    "DEFAULT": 0.6,
}


def normalize_entity_type(entity_type: str) -> str:
    """Normalize entity type to standard Presidio format.

    Args:
        entity_type: The entity type to normalize

    Returns:
        Normalized entity type string
    """
    return ENTITY_TYPE_MAPPING.get(entity_type.upper(), entity_type.upper())


def get_confidence_threshold(entity_type: str) -> float:
    """Get the default confidence threshold for an entity type.

    Args:
        entity_type: The entity type

    Returns:
        Default confidence threshold
    """
    normalized = normalize_entity_type(entity_type)
    return DEFAULT_CONFIDENCE_THRESHOLDS.get(normalized, DEFAULT_CONFIDENCE_THRESHOLDS["DEFAULT"])


def validate_presidio_version() -> tuple[str, bool]:
    """Check Presidio version compatibility.

    Returns:
        Tuple of (version_string, is_compatible)
    """
    try:
        import presidio_analyzer
        import presidio_anonymizer

        analyzer_version = getattr(presidio_analyzer, "__version__", "2.0.0")
        anonymizer_version = getattr(presidio_anonymizer, "__version__", "2.0.0")

        # Check major version compatibility (2.x)
        analyzer_major = int(analyzer_version.split(".")[0])
        anonymizer_major = int(anonymizer_version.split(".")[0])

        is_compatible = analyzer_major == 2 and anonymizer_major == 2

        version_info = f"analyzer={analyzer_version}, anonymizer={anonymizer_version}"

        return version_info, is_compatible
    except Exception as e:
        logger.warning(f"Could not determine Presidio version: {e}")
        return "unknown", False


def operator_result_to_dict(result: OperatorResult) -> dict[str, Any]:
    """Convert OperatorResult to dictionary.

    Args:
        result: Presidio OperatorResult object

    Returns:
        Dictionary representation
    """
    return {
        "entity_type": result.entity_type,
        "text": result.text,
        "start": result.start,
        "end": result.end,
        "operator": result.operator if hasattr(result, "operator") else None,
    }


def recognizer_result_to_dict(result: RecognizerResult) -> dict[str, Any]:
    """Convert RecognizerResult to dictionary.

    Args:
        result: Presidio RecognizerResult object

    Returns:
        Dictionary representation
    """
    return {
        "entity_type": result.entity_type,
        "start": result.start,
        "end": result.end,
        "score": result.score,
        "recognition_metadata": (
            result.recognition_metadata if hasattr(result, "recognition_metadata") else {}
        ),
    }


def build_statistics(entities: list[Any], source: str = "unknown") -> dict[str, Any]:
    """Build statistics dictionary from entity list.

    Args:
        entities: List of entities (RecognizerResult or OperatorResult)
        source: Source of entities ("analyzer" or "anonymizer")

    Returns:
        Statistics dictionary
    """
    stats: dict[str, Any] = {
        "total_entities": len(entities),
        "entities_by_type": {},
        "source": source,
    }

    for entity in entities:
        entity_type = entity.entity_type if hasattr(entity, "entity_type") else "UNKNOWN"
        entities_dict = stats["entities_by_type"]
        assert isinstance(entities_dict, dict)
        entities_dict[entity_type] = entities_dict.get(entity_type, 0) + 1

    return stats


def filter_overlapping_entities(entities: list[RecognizerResult]) -> list[RecognizerResult]:
    """Filter overlapping entities, keeping highest confidence.

    Args:
        entities: List of RecognizerResult objects

    Returns:
        Filtered list with no overlaps
    """
    if not entities:
        return []

    # Sort by confidence (descending) then by start position
    sorted_entities = sorted(entities, key=lambda e: (-e.score, e.start))

    filtered = []
    for entity in sorted_entities:
        # Check if this entity overlaps with any already selected
        overlaps = False
        for selected in filtered:
            if entity.start < selected.end and entity.end > selected.start:
                overlaps = True
                break

        if not overlaps:
            filtered.append(entity)

    # Sort by position for processing order
    return sorted(filtered, key=lambda e: e.start)


def validate_entity_boundaries(
    entities: list[RecognizerResult], text: str
) -> list[RecognizerResult]:
    """Validate that entities have valid boundaries within text.

    Args:
        entities: List of RecognizerResult objects
        text: Text to validate against

    Returns:
        List of valid entities
    """
    text_length = len(text)
    valid_entities = []

    for entity in entities:
        if entity.start >= 0 and entity.end <= text_length and entity.start < entity.end:
            valid_entities.append(entity)
        else:
            logger.warning(
                f"Invalid entity boundaries: {entity.entity_type} "
                f"[{entity.start}:{entity.end}] in text of length {text_length}"
            )

    return valid_entities


def create_entity_mapping(
    original_entities: list[RecognizerResult], masked_entities: list[OperatorResult]
) -> dict[tuple[int, int], dict[str, Any]]:
    """Create mapping between original and masked entities.

    Args:
        original_entities: Original recognizer results
        masked_entities: Masked operator results

    Returns:
        Dictionary mapping (start, end) positions to entity info
    """
    mapping = {}

    # Map original entities
    for entity in original_entities:
        key = (entity.start, entity.end)
        mapping[key] = {
            "type": "original",
            "entity_type": entity.entity_type,
            "score": entity.score,
            "text": None,  # Not available in RecognizerResult
        }

    # Map masked entities
    for entity in masked_entities:
        key = (entity.start, entity.end)
        if key in mapping:
            mapping[key]["masked_text"] = entity.text
            mapping[key]["operator"] = entity.operator if hasattr(entity, "operator") else None
        else:
            mapping[key] = {
                "type": "masked",
                "entity_type": entity.entity_type,
                "text": entity.text,
                "operator": entity.operator if hasattr(entity, "operator") else None,
            }

    return mapping
