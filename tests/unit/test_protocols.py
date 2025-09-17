"""Unit tests for Protocol-based types in masking."""

import pytest
from presidio_anonymizer import OperatorResult

from cloakpivot.masking.protocols import (
    OperatorResultLike,
    SegmentBoundary,
    SyntheticOperatorResult,
)


class TestOperatorResultProtocol:
    """Test the OperatorResultLike protocol."""

    def test_synthetic_result_conforms_to_protocol(self):
        """Test that SyntheticOperatorResult conforms to OperatorResultLike protocol."""
        result = SyntheticOperatorResult(
            entity_type="EMAIL",
            start=0,
            end=10,
            operator="replace",
            text="[MASKED]",
        )

        # Should be instance of protocol
        assert isinstance(result, OperatorResultLike)

        # Should have all required attributes
        assert result.entity_type == "EMAIL"
        assert result.start == 0
        assert result.end == 10
        assert result.operator == "replace"
        assert result.text == "[MASKED]"

    def test_presidio_operator_result_conforms(self):
        """Test that Presidio's OperatorResult conforms to the protocol."""
        # Create a real OperatorResult
        result = OperatorResult(
            start=0, end=10, entity_type="PHONE_NUMBER", text="***", operator="mask"
        )

        # Should be instance of protocol
        assert isinstance(result, OperatorResultLike)

    def test_synthetic_result_validation(self):
        """Test that SyntheticOperatorResult validates inputs."""
        # Valid result
        result = SyntheticOperatorResult(
            entity_type="PERSON", start=0, end=5, operator="redact", text="*****"
        )
        assert result.start == 0
        assert result.end == 5

        # Invalid: negative start
        with pytest.raises(ValueError, match="start must be non-negative"):
            SyntheticOperatorResult(
                entity_type="PERSON", start=-1, end=5, operator="redact", text="*****"
            )

        # Invalid: end < start
        with pytest.raises(ValueError, match="end .* must be >= start"):
            SyntheticOperatorResult(
                entity_type="PERSON", start=10, end=5, operator="redact", text="*****"
            )

        # Invalid: empty entity_type
        with pytest.raises(ValueError, match="entity_type cannot be empty"):
            SyntheticOperatorResult(entity_type="", start=0, end=5, operator="redact", text="*****")

        # Invalid: empty operator
        with pytest.raises(ValueError, match="operator cannot be empty"):
            SyntheticOperatorResult(entity_type="PERSON", start=0, end=5, operator="", text="*****")

    def test_synthetic_result_with_metadata(self):
        """Test SyntheticOperatorResult with optional metadata."""
        metadata = {"original_text": "test@example.com", "confidence": 0.95}
        result = SyntheticOperatorResult(
            entity_type="EMAIL",
            start=0,
            end=16,
            operator="replace",
            text="[EMAIL]",
            operator_metadata=metadata,
        )
        assert result.operator_metadata == metadata


class TestSegmentBoundary:
    """Test the SegmentBoundary dataclass."""

    def test_segment_boundary_creation(self):
        """Test creating a SegmentBoundary."""
        boundary = SegmentBoundary(segment_index=0, start=0, end=100, node_id="#/texts/0")

        assert boundary.segment_index == 0
        assert boundary.start == 0
        assert boundary.end == 100
        assert boundary.node_id == "#/texts/0"

    def test_segment_boundary_contains_position(self):
        """Test the contains_position method."""
        boundary = SegmentBoundary(segment_index=0, start=10, end=50, node_id="#/texts/0")

        # Position within boundary
        assert boundary.contains_position(10)  # At start
        assert boundary.contains_position(30)  # In middle
        assert boundary.contains_position(49)  # Near end

        # Position outside boundary
        assert not boundary.contains_position(9)  # Before start
        assert not boundary.contains_position(50)  # At end (exclusive)
        assert not boundary.contains_position(100)  # After end

    def test_multiple_boundaries(self):
        """Test working with multiple segment boundaries."""
        boundaries = [
            SegmentBoundary(segment_index=0, start=0, end=100, node_id="#/texts/0"),
            SegmentBoundary(segment_index=1, start=102, end=200, node_id="#/texts/1"),  # With gap
            SegmentBoundary(segment_index=2, start=200, end=300, node_id="#/texts/2"),
        ]

        # Find boundary containing position
        def find_boundary(pos):
            for b in boundaries:
                if b.contains_position(pos):
                    return b
            return None

        # Test finding boundaries
        assert find_boundary(50).node_id == "#/texts/0"
        assert find_boundary(150).node_id == "#/texts/1"
        assert find_boundary(250).node_id == "#/texts/2"
        assert find_boundary(101) is None  # In gap
        assert find_boundary(350) is None  # Beyond end


class TestProtocolCompatibility:
    """Test that our protocols work correctly with type checking."""

    def test_list_of_operator_results(self):
        """Test that we can create lists mixing different OperatorResultLike types."""
        results: list[OperatorResultLike] = []

        # Add SyntheticOperatorResult
        results.append(
            SyntheticOperatorResult(
                entity_type="EMAIL", start=0, end=10, operator="replace", text="[EMAIL]"
            )
        )

        # Add actual OperatorResult
        results.append(
            OperatorResult(start=20, end=30, entity_type="PHONE", text="****", operator="mask")
        )

        assert len(results) == 2
        assert all(hasattr(r, "entity_type") for r in results)
        assert all(hasattr(r, "start") for r in results)
        assert all(hasattr(r, "end") for r in results)
        assert all(hasattr(r, "text") for r in results)
        assert all(hasattr(r, "operator") for r in results)

    def test_protocol_duck_typing(self):
        """Test that any object conforming to the protocol works."""

        class CustomResult:
            """Custom class that conforms to OperatorResultLike."""

            def __init__(self):
                self.entity_type = "CUSTOM"
                self.start = 0
                self.end = 10
                self.operator = "custom"
                self.text = "[CUSTOM]"

        result = CustomResult()
        # Should be recognized as OperatorResultLike
        assert isinstance(result, OperatorResultLike)

        # Can be used in lists
        results: list[OperatorResultLike] = [result]
        assert len(results) == 1
