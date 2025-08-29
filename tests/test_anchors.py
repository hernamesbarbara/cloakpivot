"""Tests for the anchor system."""

from datetime import datetime

import pytest

from cloakpivot.core.anchors import AnchorEntry, AnchorIndex


class TestAnchorEntry:
    """Test the AnchorEntry dataclass."""

    def test_basic_creation(self) -> None:
        """Test basic anchor creation."""
        timestamp = datetime.utcnow()
        anchor = AnchorEntry(
            node_id="p1",
            start=10,
            end=20,
            entity_type="PHONE_NUMBER",
            confidence=0.95,
            masked_value="[PHONE]",
            replacement_id="repl_123",
            original_checksum="a" * 64,
            checksum_salt="dGVzdA==",  # base64 encoded "test"
            strategy_used="template",
            timestamp=timestamp
        )

        assert anchor.node_id == "p1"
        assert anchor.start == 10
        assert anchor.end == 20
        assert anchor.entity_type == "PHONE_NUMBER"
        assert anchor.confidence == 0.95
        assert anchor.masked_value == "[PHONE]"
        assert anchor.replacement_id == "repl_123"
        assert anchor.original_checksum == "a" * 64
        assert anchor.strategy_used == "template"
        assert anchor.timestamp == timestamp
        assert anchor.metadata == {}

    def test_creation_with_defaults(self) -> None:
        """Test anchor creation with default timestamp and metadata."""
        before_creation = datetime.utcnow()
        anchor = AnchorEntry(
            node_id="p1",
            start=10,
            end=20,
            entity_type="PHONE_NUMBER",
            confidence=0.95,
            masked_value="[PHONE]",
            replacement_id="repl_123",
            original_checksum="a" * 64,
            checksum_salt="dGVzdA==",  # base64 encoded "test"
            strategy_used="template"
        )
        after_creation = datetime.utcnow()

        # Timestamp should be set automatically
        assert before_creation <= anchor.timestamp <= after_creation
        assert anchor.metadata == {}

    def test_frozen_immutability(self) -> None:
        """Test that anchor instances are immutable."""
        anchor = AnchorEntry(
            node_id="p1",
            start=10,
            end=20,
            entity_type="PHONE_NUMBER",
            confidence=0.95,
            masked_value="[PHONE]",
            replacement_id="repl_123",
            original_checksum="a" * 64,
            checksum_salt="dGVzdA==",  # base64 encoded "test"
            strategy_used="template"
        )

        with pytest.raises((AttributeError, TypeError)):  # FrozenInstanceError
            anchor.start = 15  # type: ignore

    def test_position_validation(self) -> None:
        """Test position validation."""
        # Valid positions
        AnchorEntry(
            node_id="p1", start=0, end=10, entity_type="PHONE", confidence=0.5,
            masked_value="masked", replacement_id="repl", original_checksum="a" * 64,
            checksum_salt="dGVzdA==", strategy_used="template"
        )

        # Invalid start position
        with pytest.raises(ValueError, match="start position must be a non-negative integer"):
            AnchorEntry(
                node_id="p1", start=-1, end=10, entity_type="PHONE", confidence=0.5,
                masked_value="masked", replacement_id="repl", original_checksum="a" * 64,
                checksum_salt="dGVzdA==", strategy_used="template"
            )

        # Invalid end position
        with pytest.raises(ValueError, match="end position must be a non-negative integer"):
            AnchorEntry(
                node_id="p1", start=10, end=-1, entity_type="PHONE", confidence=0.5,
                masked_value="masked", replacement_id="repl", original_checksum="a" * 64,
                checksum_salt="dGVzdA==", strategy_used="template"
            )

        # End <= start
        with pytest.raises(ValueError, match="end position must be greater than start position"):
            AnchorEntry(
                node_id="p1", start=10, end=10, entity_type="PHONE", confidence=0.5,
                masked_value="masked", replacement_id="repl", original_checksum="a" * 64,
                checksum_salt="dGVzdA==", strategy_used="template"
            )

    def test_confidence_validation(self) -> None:
        """Test confidence validation."""
        # Valid confidence values
        for conf in [0.0, 0.5, 1.0]:
            AnchorEntry(
                node_id="p1", start=0, end=10, entity_type="PHONE", confidence=conf,
                masked_value="masked", replacement_id="repl", original_checksum="a" * 64,
                checksum_salt="dGVzdA==", strategy_used="template"
            )

        # Invalid confidence type
        with pytest.raises(ValueError, match="confidence must be a number"):
            AnchorEntry(
                node_id="p1", start=0, end=10, entity_type="PHONE", confidence="high",  # type: ignore
                masked_value="masked", replacement_id="repl", original_checksum="a" * 64,
                checksum_salt="dGVzdA==", strategy_used="template"
            )

        # Out of range confidence
        with pytest.raises(ValueError, match="confidence must be between 0.0 and 1.0"):
            AnchorEntry(
                node_id="p1", start=0, end=10, entity_type="PHONE", confidence=-0.1,
                masked_value="masked", replacement_id="repl", original_checksum="a" * 64,
                checksum_salt="dGVzdA==", strategy_used="template"
            )

        with pytest.raises(ValueError, match="confidence must be between 0.0 and 1.0"):
            AnchorEntry(
                node_id="p1", start=0, end=10, entity_type="PHONE", confidence=1.1,
                masked_value="masked", replacement_id="repl", original_checksum="a" * 64,
                checksum_salt="dGVzdA==", strategy_used="template"
            )

    def test_checksum_validation(self) -> None:
        """Test checksum validation."""
        # Valid checksum (64-char hex)
        valid_checksum = "a1b2c3d4e5f6" + "0" * 52
        AnchorEntry(
            node_id="p1", start=0, end=10, entity_type="PHONE", confidence=0.5,
            masked_value="masked", replacement_id="repl", original_checksum=valid_checksum,
            checksum_salt="dGVzdA==", strategy_used="template"
        )

        # Invalid checksum length
        with pytest.raises(ValueError, match="original_checksum should be a 64-character SHA-256 hex string"):
            AnchorEntry(
                node_id="p1", start=0, end=10, entity_type="PHONE", confidence=0.5,
                masked_value="masked", replacement_id="repl", original_checksum="short",
                checksum_salt="dGVzdA==", strategy_used="template"
            )

        # Invalid hex characters
        with pytest.raises(ValueError, match="original_checksum must contain only hexadecimal characters"):
            AnchorEntry(
                node_id="p1", start=0, end=10, entity_type="PHONE", confidence=0.5,
                masked_value="masked", replacement_id="repl", original_checksum="g" * 64,
                checksum_salt="dGVzdA==", strategy_used="template"
            )

    def test_id_validation(self) -> None:
        """Test ID field validation."""
        # Valid IDs
        AnchorEntry(
            node_id="p1", start=0, end=10, entity_type="PHONE", confidence=0.5,
            masked_value="masked", replacement_id="repl", original_checksum="a" * 64,
            checksum_salt="dGVzdA==", strategy_used="template"
        )

        # Empty node_id
        with pytest.raises(ValueError, match="node_id must be a non-empty string"):
            AnchorEntry(
                node_id="", start=0, end=10, entity_type="PHONE", confidence=0.5,
                masked_value="masked", replacement_id="repl", original_checksum="a" * 64,
                checksum_salt="dGVzdA==", strategy_used="template"
            )

        # Empty replacement_id
        with pytest.raises(ValueError, match="replacement_id must be a non-empty string"):
            AnchorEntry(
                node_id="p1", start=0, end=10, entity_type="PHONE", confidence=0.5,
                masked_value="masked", replacement_id="", original_checksum="a" * 64,
                checksum_salt="dGVzdA==", strategy_used="template"
            )

        # Empty entity_type
        with pytest.raises(ValueError, match="entity_type must be a non-empty string"):
            AnchorEntry(
                node_id="p1", start=0, end=10, entity_type="", confidence=0.5,
                masked_value="masked", replacement_id="repl", original_checksum="a" * 64,
                checksum_salt="dGVzdA==", strategy_used="template"
            )

    def test_properties(self) -> None:
        """Test computed properties."""
        anchor = AnchorEntry(
            node_id="p1", start=10, end=25, entity_type="PHONE", confidence=0.5,
            masked_value="[PHONE]", replacement_id="repl", original_checksum="a" * 64,
            checksum_salt="dGVzdA==", strategy_used="template"
        )

        assert anchor.span_length == 15  # 25 - 10
        assert anchor.replacement_length == 7  # len("[PHONE]")
        assert anchor.length_delta == -8  # 7 - 15

    def test_verify_original_text(self) -> None:
        """Test original text verification."""
        # Create anchor using the factory method which properly creates salted checksums
        original_text = "555-123-4567"
        anchor = AnchorEntry.create_from_detection(
            node_id="p1",
            start=0,
            end=len(original_text),
            entity_type="PHONE",
            confidence=0.5,
            original_text=original_text,
            masked_value="[PHONE]",
            strategy_used="template"
        )

        # Correct text should verify
        assert anchor.verify_original_text(original_text)

        # Incorrect text should not verify
        assert not anchor.verify_original_text("different text")

    def test_overlaps_with(self) -> None:
        """Test overlap detection."""
        anchor1 = AnchorEntry(
            node_id="p1", start=10, end=20, entity_type="PHONE", confidence=0.5,
            masked_value="masked", replacement_id="repl1", original_checksum="a" * 64,
            checksum_salt="dGVzdA==", strategy_used="template"
        )

        # Same node, overlapping ranges
        anchor2_overlap = AnchorEntry(
            node_id="p1", start=15, end=25, entity_type="EMAIL", confidence=0.5,
            masked_value="masked", replacement_id="repl2", original_checksum="b" * 64,
            checksum_salt="dGVzdA==", strategy_used="template"
        )

        # Same node, non-overlapping ranges
        anchor3_no_overlap = AnchorEntry(
            node_id="p1", start=25, end=30, entity_type="NAME", confidence=0.5,
            masked_value="masked", replacement_id="repl3", original_checksum="c" * 64,
            checksum_salt="dGVzdA==", strategy_used="template"
        )

        # Different node
        anchor4_diff_node = AnchorEntry(
            node_id="p2", start=10, end=20, entity_type="PHONE", confidence=0.5,
            masked_value="masked", replacement_id="repl4", original_checksum="d" * 64,
            checksum_salt="dGVzdA==", strategy_used="template"
        )

        assert anchor1.overlaps_with(anchor2_overlap)
        assert not anchor1.overlaps_with(anchor3_no_overlap)
        assert not anchor1.overlaps_with(anchor4_diff_node)

    def test_contains_position(self) -> None:
        """Test position containment check."""
        anchor = AnchorEntry(
            node_id="p1", start=10, end=20, entity_type="PHONE", confidence=0.5,
            masked_value="masked", replacement_id="repl", original_checksum="a" * 64,
            checksum_salt="dGVzdA==", strategy_used="template"
        )

        assert anchor.contains_position(10)  # Start position
        assert anchor.contains_position(15)  # Middle position
        assert anchor.contains_position(19)  # Just before end
        assert not anchor.contains_position(20)  # End position (exclusive)
        assert not anchor.contains_position(5)  # Before start
        assert not anchor.contains_position(25)  # After end

    def test_with_metadata(self) -> None:
        """Test metadata addition."""
        anchor = AnchorEntry(
            node_id="p1", start=10, end=20, entity_type="PHONE", confidence=0.5,
            masked_value="masked", replacement_id="repl", original_checksum="a" * 64,
            checksum_salt="dGVzdA==", strategy_used="template",
            metadata={"existing": "value"}
        )

        updated = anchor.with_metadata(new_key="new_value", another="value")

        # Original unchanged
        assert "new_key" not in anchor.metadata

        # Updated has merged metadata
        expected_metadata = {"existing": "value", "new_key": "new_value", "another": "value"}
        assert updated.metadata == expected_metadata

    def test_serialization(self) -> None:
        """Test to_dict and from_dict serialization."""
        timestamp = datetime(2023, 1, 1, 12, 0, 0)
        anchor = AnchorEntry(
            node_id="p1", start=10, end=20, entity_type="PHONE", confidence=0.95,
            masked_value="[PHONE]", replacement_id="repl_123", original_checksum="a" * 64,
            checksum_salt="dGVzdA==", strategy_used="template", timestamp=timestamp, metadata={"test": "value"}
        )

        # Serialize to dict
        data = anchor.to_dict()
        expected_data = {
            "node_id": "p1",
            "start": 10,
            "end": 20,
            "entity_type": "PHONE",
            "confidence": 0.95,
            "masked_value": "[PHONE]",
            "replacement_id": "repl_123",
            "original_checksum": "a" * 64,
            "checksum_salt": "dGVzdA==",
            "strategy_used": "template",
            "timestamp": "2023-01-01T12:00:00",
            "metadata": {"test": "value"}
        }
        assert data == expected_data

        # Deserialize from dict
        restored = AnchorEntry.from_dict(data)
        assert restored.node_id == anchor.node_id
        assert restored.start == anchor.start
        assert restored.end == anchor.end
        assert restored.entity_type == anchor.entity_type
        assert restored.confidence == anchor.confidence
        assert restored.masked_value == anchor.masked_value
        assert restored.replacement_id == anchor.replacement_id
        assert restored.original_checksum == anchor.original_checksum
        assert restored.strategy_used == anchor.strategy_used
        assert restored.timestamp == anchor.timestamp
        assert restored.metadata == anchor.metadata

    def test_create_replacement_id(self) -> None:
        """Test replacement ID generation."""
        id1 = AnchorEntry.create_replacement_id("PHONE", "p1", 10)
        id2 = AnchorEntry.create_replacement_id("PHONE", "p1", 10)
        id3 = AnchorEntry.create_replacement_id("PHONE", "p1", 20)
        id4 = AnchorEntry.create_replacement_id("EMAIL", "p1", 10)

        # Same inputs produce same ID (deterministic)
        assert id1 == id2

        # Different inputs produce different IDs
        assert id1 != id3
        assert id1 != id4

        # IDs have expected format
        assert id1.startswith("repl_")
        assert len(id1) == 13  # "repl_" + 8 hex chars

    def test_create_from_detection(self) -> None:
        """Test factory method for creating anchors from detection results."""
        original_text = "555-123-4567"
        anchor = AnchorEntry.create_from_detection(
            node_id="p1",
            start=10,
            end=22,
            entity_type="PHONE_NUMBER",
            confidence=0.95,
            original_text=original_text,
            masked_value="[PHONE]",
            strategy_used="template",
            metadata={"source": "presidio"}
        )

        assert anchor.node_id == "p1"
        assert anchor.start == 10
        assert anchor.end == 22
        assert anchor.entity_type == "PHONE_NUMBER"
        assert anchor.confidence == 0.95
        assert anchor.masked_value == "[PHONE]"
        assert anchor.strategy_used == "template"
        assert anchor.metadata == {"source": "presidio"}
        assert anchor.replacement_id.startswith("repl_")
        assert anchor.verify_original_text(original_text)


class TestAnchorIndex:
    """Test the AnchorIndex class."""

    def setup_method(self) -> None:
        """Set up test anchors."""
        self.anchor1 = AnchorEntry.create_from_detection(
            "p1", 10, 20, "PHONE", 0.9, "555-1234", "[PHONE]", "template"
        )
        self.anchor2 = AnchorEntry.create_from_detection(
            "p1", 30, 40, "EMAIL", 0.8, "test@example.com", "[EMAIL]", "template"
        )
        self.anchor3 = AnchorEntry.create_from_detection(
            "p2", 5, 15, "NAME", 0.7, "John Doe", "[NAME]", "template"
        )

    def test_basic_creation(self) -> None:
        """Test basic index creation."""
        # Empty index
        index = AnchorIndex()
        assert len(index.get_all_anchors()) == 0

        # Index with initial anchors
        anchors = [self.anchor1, self.anchor2]
        index = AnchorIndex(anchors)
        assert len(index.get_all_anchors()) == 2

    def test_add_anchor(self) -> None:
        """Test adding anchors to index."""
        index = AnchorIndex()

        index.add_anchor(self.anchor1)
        assert len(index.get_all_anchors()) == 1

        index.add_anchor(self.anchor2)
        assert len(index.get_all_anchors()) == 2

        # Duplicate replacement_id should raise error
        duplicate = AnchorEntry.create_from_detection(
            "p3", 0, 10, "OTHER", 0.5, "text", "masked", "redact",
            replacement_id=self.anchor1.replacement_id
        )

        with pytest.raises(ValueError, match="Duplicate replacement_id"):
            index.add_anchor(duplicate)

    def test_get_by_replacement_id(self) -> None:
        """Test retrieval by replacement ID."""
        index = AnchorIndex([self.anchor1, self.anchor2])

        # Existing anchor
        found = index.get_by_replacement_id(self.anchor1.replacement_id)
        assert found == self.anchor1

        # Non-existing anchor
        not_found = index.get_by_replacement_id("non_existent")
        assert not_found is None

    def test_get_by_node_id(self) -> None:
        """Test retrieval by node ID."""
        index = AnchorIndex([self.anchor1, self.anchor2, self.anchor3])

        # Node with multiple anchors (should be sorted by start position)
        p1_anchors = index.get_by_node_id("p1")
        assert len(p1_anchors) == 2
        assert p1_anchors[0] == self.anchor1  # start=10
        assert p1_anchors[1] == self.anchor2  # start=30

        # Node with single anchor
        p2_anchors = index.get_by_node_id("p2")
        assert len(p2_anchors) == 1
        assert p2_anchors[0] == self.anchor3

        # Non-existing node
        empty_anchors = index.get_by_node_id("non_existent")
        assert len(empty_anchors) == 0

    def test_get_by_entity_type(self) -> None:
        """Test retrieval by entity type."""
        index = AnchorIndex([self.anchor1, self.anchor2, self.anchor3])

        # Existing entity type
        phone_anchors = index.get_by_entity_type("PHONE")
        assert len(phone_anchors) == 1
        assert phone_anchors[0] == self.anchor1

        # Non-existing entity type
        empty_anchors = index.get_by_entity_type("NON_EXISTENT")
        assert len(empty_anchors) == 0

    def test_find_overlapping_anchors(self) -> None:
        """Test finding overlapping anchors."""
        # Create overlapping anchor
        overlapping = AnchorEntry.create_from_detection(
            "p1", 15, 25, "OTHER", 0.5, "overlap", "masked", "redact"
        )

        index = AnchorIndex([self.anchor1, self.anchor2, overlapping])

        # Find overlaps with anchor1 (10-20)
        overlaps = index.find_overlapping_anchors(self.anchor1)
        assert len(overlaps) == 1
        assert overlaps[0] == overlapping

        # Find overlaps with anchor2 (30-40) - should be none
        no_overlaps = index.find_overlapping_anchors(self.anchor2)
        assert len(no_overlaps) == 0

    def test_get_anchors_in_range(self) -> None:
        """Test finding anchors in position range."""
        index = AnchorIndex([self.anchor1, self.anchor2, self.anchor3])

        # Range that covers anchor1 (10-20)
        anchors_in_range = index.get_anchors_in_range("p1", 5, 25)
        assert len(anchors_in_range) == 1
        assert anchors_in_range[0] == self.anchor1

        # Range that covers both p1 anchors
        all_p1_anchors = index.get_anchors_in_range("p1", 0, 50)
        assert len(all_p1_anchors) == 2

        # Range with no intersections
        no_anchors = index.get_anchors_in_range("p1", 50, 60)
        assert len(no_anchors) == 0

    def test_get_stats(self) -> None:
        """Test statistics generation."""
        index = AnchorIndex([self.anchor1, self.anchor2, self.anchor3])
        stats = index.get_stats()

        assert stats["total_anchors"] == 3
        assert stats["unique_nodes"] == 2
        assert stats["entity_type_counts"]["PHONE"] == 1
        assert stats["entity_type_counts"]["EMAIL"] == 1
        assert stats["entity_type_counts"]["NAME"] == 1
        assert stats["strategy_counts"]["template"] == 3

        # Average confidence: (0.9 + 0.8 + 0.7) / 3 = 0.8
        assert abs(stats["average_confidence"] - 0.8) < 0.01
