"""Custom assertion helpers for CloakPivot testing."""

from docling_core.types import DoclingDocument
from presidio_analyzer import RecognizerResult

from cloakpivot.core.cloakmap import CloakMap
from cloakpivot.core.policies import MaskingPolicy
from cloakpivot.masking.engine import MaskingResult


def assert_document_structure_preserved(
    original: DoclingDocument, processed: DoclingDocument
) -> None:
    """Assert that document structure is preserved after processing."""
    assert original.name == processed.name, "Document name should be preserved"
    assert len(original.texts) == len(processed.texts), (
        "Number of text items should be preserved"
    )

    for orig_item, proc_item in zip(original.texts, processed.texts):
        assert orig_item.self_ref == proc_item.self_ref, (
            f"Text item reference should be preserved: {orig_item.self_ref}"
        )
        assert orig_item.label == proc_item.label, (
            f"Text item label should be preserved: {orig_item.label}"
        )


def assert_entities_detected(
    entities: list[RecognizerResult],
    expected_types: list[str],
    min_confidence: float = 0.5,
) -> None:
    """Assert that expected entity types are detected with minimum confidence."""
    detected_types = [
        entity.entity_type for entity in entities if entity.score >= min_confidence
    ]

    for expected_type in expected_types:
        assert expected_type in detected_types, (
            f"Expected entity type '{expected_type}' not detected"
        )


def assert_masking_result_valid(result: MaskingResult) -> None:
    """Assert that a MaskingResult is valid and complete."""
    assert result is not None, "MaskingResult should not be None"
    assert result.masked_document is not None, "Masked document should not be None"
    assert result.cloakmap is not None, "CloakMap should not be None"

    # Check if any entities were actually masked using stats
    entities_masked = 0
    if result.stats and "total_entities_masked" in result.stats:
        entities_masked = result.stats["total_entities_masked"]

    # Only expect anchors if entities were actually masked
    if entities_masked > 0:
        assert len(result.cloakmap.anchors) > 0, (
            f"CloakMap should contain anchors when {entities_masked} entities were masked"
        )
    # If no entities were masked, anchors might be empty and that's valid


def assert_cloakmap_valid(cloakmap: CloakMap) -> None:
    """Assert that a CloakMap is valid and well-formed."""
    assert cloakmap is not None, "CloakMap should not be None"
    assert cloakmap.doc_id is not None, "Document ID should be set"
    assert cloakmap.version is not None, "Version should be set"
    assert isinstance(cloakmap.anchors, list), "Anchors should be a list"

    # Validate anchor structure
    for anchor in cloakmap.anchors:
        assert isinstance(anchor.replacement_id, str), (
            f"Replacement ID should be string: {anchor.replacement_id}"
        )
        assert hasattr(anchor, "masked_value"), (
            f"Anchor should have masked_value: {anchor}"
        )
        assert hasattr(anchor, "entity_type"), (
            f"Anchor should have entity_type: {anchor}"
        )


def assert_round_trip_fidelity(
    original: DoclingDocument,
    masked: DoclingDocument,
    unmasked: DoclingDocument,
    cloakmap: CloakMap,
) -> None:
    """Assert that round-trip masking/unmasking preserves original content."""
    # Document structure should be preserved
    assert_document_structure_preserved(original, masked)
    assert_document_structure_preserved(original, unmasked)

    # Original and unmasked should be identical
    assert len(original.texts) == len(unmasked.texts), (
        "Text count should match after round-trip"
    )

    for orig_item, unmask_item in zip(original.texts, unmasked.texts):
        assert orig_item.text == unmask_item.text, (
            f"Text content should match after round-trip:\n"
            f"Original: '{orig_item.text}'\n"
            f"Unmasked: '{unmask_item.text}'"
        )


def assert_text_contains_no_pii(text: str, entity_types: list[str]) -> None:
    """Assert that text contains no recognizable PII patterns."""
    # Simple pattern checks - in a real implementation, you might use
    # the analyzer to verify no PII is detected
    common_patterns = {
        "PHONE_NUMBER": r"\d{3}-\d{3}-\d{4}|\(\d{3}\)\s*\d{3}-\d{4}",
        "EMAIL_ADDRESS": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
        "US_SSN": r"\d{3}-\d{2}-\d{4}",
        "CREDIT_CARD": r"\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}",
    }

    import re

    for entity_type in entity_types:
        if entity_type in common_patterns:
            pattern = common_patterns[entity_type]
            matches = re.findall(pattern, text)
            assert len(matches) == 0, (
                f"Found potential {entity_type} patterns in text: {matches}\n"
                f"Text: '{text}'"
            )


def assert_policy_applied_correctly(
    policy: MaskingPolicy, entities: list[RecognizerResult], result: MaskingResult
) -> None:
    """Assert that masking policy was applied correctly to detected entities."""
    # Check that entities above threshold were masked
    for entity in entities:
        threshold = policy.get_threshold_for_entity(entity.entity_type)

        if entity.score >= threshold and policy.should_mask_entity(
            original_text="", entity_type=entity.entity_type, confidence=entity.score
        ):
            # Entity should be in the cloakmap anchors
            matching_anchors = [
                anchor
                for anchor in result.cloakmap.anchors
                if anchor.entity_type == entity.entity_type
                and anchor.start <= entity.start <= anchor.end
            ]
            assert len(matching_anchors) > 0, (
                f"Entity {entity.entity_type} with score {entity.score} should be masked"
            )


def assert_performance_acceptable(
    processing_time: float, max_time_seconds: float, document_size_chars: int
) -> None:
    """Assert that processing performance is within acceptable limits."""
    assert processing_time <= max_time_seconds, (
        f"Processing took {processing_time:.2f}s, expected <= {max_time_seconds}s "
        f"for document with {document_size_chars} characters"
    )

    # Performance should scale reasonably with document size
    chars_per_second = (
        document_size_chars / processing_time if processing_time > 0 else float("inf")
    )
    min_chars_per_second = 50  # Realistic minimum performance threshold for testing

    assert chars_per_second >= min_chars_per_second, (
        f"Processing rate {chars_per_second:.0f} chars/sec is below threshold "
        f"of {min_chars_per_second} chars/sec"
    )


def assert_error_handling_robust(exception: Exception, expected_type: type) -> None:
    """Assert that error handling is robust and provides useful information."""
    assert isinstance(exception, expected_type), (
        f"Expected exception of type {expected_type}, got {type(exception)}"
    )
    assert str(exception), "Exception should have a descriptive message"
    # Add more specific error message validation as needed


def assert_memory_usage_reasonable(
    peak_memory_mb: float, max_memory_mb: float, document_size_chars: int
) -> None:
    """Assert that memory usage is within reasonable bounds."""
    assert peak_memory_mb <= max_memory_mb, (
        f"Peak memory usage {peak_memory_mb:.1f}MB exceeds limit of {max_memory_mb}MB "
        f"for document with {document_size_chars} characters"
    )
