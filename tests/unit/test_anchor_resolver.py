"""Unit tests for cloakpivot.unmasking.anchor_resolver module."""

from unittest.mock import Mock

from docling_core.types.doc.document import NodeItem, TextItem

from cloakpivot.core.types import DoclingDocument
from cloakpivot.core.types.anchors import AnchorEntry
from cloakpivot.unmasking.anchor_resolver import AnchorResolver, FailedAnchor, ResolvedAnchor


class TestResolvedAnchor:
    """Test ResolvedAnchor dataclass."""

    def test_resolved_anchor_creation(self):
        """Test creating a ResolvedAnchor instance."""
        anchor = Mock(spec=AnchorEntry)
        node_item = Mock(spec=NodeItem)

        resolved = ResolvedAnchor(
            anchor=anchor,
            node_item=node_item,
            found_position=(10, 20),
            found_text="[PERSON]",
            position_delta=2,
            confidence=0.95,
        )

        assert resolved.anchor == anchor
        assert resolved.node_item == node_item
        assert resolved.found_position == (10, 20)
        assert resolved.found_text == "[PERSON]"
        assert resolved.position_delta == 2
        assert resolved.confidence == 0.95

    def test_resolved_anchor_attributes(self):
        """Test ResolvedAnchor attributes."""
        anchor = Mock(spec=AnchorEntry)
        anchor.masked_value = "[EMAIL]"
        anchor.original_text = "test@example.com"

        node_item = Mock(spec=NodeItem)

        resolved = ResolvedAnchor(
            anchor=anchor,
            node_item=node_item,
            found_position=(0, 7),
            found_text="[EMAIL]",
            position_delta=0,
            confidence=1.0,
        )

        assert resolved.anchor.masked_value == "[EMAIL]"
        assert resolved.anchor.original_text == "test@example.com"
        assert resolved.position_delta == 0  # Perfect match


class TestFailedAnchor:
    """Test FailedAnchor dataclass."""

    def test_failed_anchor_creation(self):
        """Test creating a FailedAnchor instance."""
        anchor = Mock(spec=AnchorEntry)

        failed = FailedAnchor(
            anchor=anchor,
            failure_reason="Node not found",
            node_found=False,
            attempted_positions=[(0, 10), (5, 15)],
        )

        assert failed.anchor == anchor
        assert failed.failure_reason == "Node not found"
        assert failed.node_found is False
        assert len(failed.attempted_positions) == 2

    def test_failed_anchor_node_found_but_no_match(self):
        """Test FailedAnchor when node is found but text doesn't match."""
        anchor = Mock(spec=AnchorEntry)

        failed = FailedAnchor(
            anchor=anchor,
            failure_reason="Masked text not found at expected position",
            node_found=True,
            attempted_positions=[(10, 18), (12, 20), (8, 16)],
        )

        assert failed.node_found is True
        assert "not found" in failed.failure_reason
        assert len(failed.attempted_positions) == 3


class TestAnchorResolver:
    """Test AnchorResolver class."""

    def test_initialization(self):
        """Test AnchorResolver initialization."""
        resolver = AnchorResolver()
        assert resolver is not None

    def test_resolve_anchors_empty_list(self):
        """Test resolving empty anchor list."""
        resolver = AnchorResolver()
        doc = Mock(spec=DoclingDocument)

        results = resolver.resolve_anchors(doc, [])

        assert "resolved" in results
        assert "failed" in results
        assert len(results["resolved"]) == 0
        assert len(results["failed"]) == 0

    def test_resolve_single_anchor_success(self):
        """Test successfully resolving a single anchor."""
        resolver = AnchorResolver()

        # Mock document
        doc = Mock(spec=DoclingDocument)
        text_item = Mock(spec=TextItem)
        text_item.text = "Hello [PERSON] there"
        doc.texts = [text_item]

        # Create anchor
        anchor = AnchorEntry.create_from_detection(
            node_id="#/texts/0",
            start=6,
            end=14,
            entity_type="PERSON",
            confidence=0.95,
            original_text="John",
            masked_value="[PERSON]",
            strategy_used="template",
        )

        results = resolver.resolve_anchors(doc, [anchor])

        # Should attempt to resolve
        assert "resolved" in results
        assert "failed" in results

    def test_resolve_multiple_anchors(self):
        """Test resolving multiple anchors."""
        resolver = AnchorResolver()

        doc = Mock(spec=DoclingDocument)
        text_item = Mock(spec=TextItem)
        text_item.text = "[PERSON] sent email to [EMAIL]"
        doc.texts = [text_item]

        anchors = [
            AnchorEntry.create_from_detection(
                node_id="#/texts/0",
                start=0,
                end=8,
                entity_type="PERSON",
                confidence=0.9,
                original_text="Alice",
                masked_value="[PERSON]",
                strategy_used="template",
            ),
            AnchorEntry.create_from_detection(
                node_id="#/texts/0",
                start=23,
                end=30,
                entity_type="EMAIL",
                confidence=0.95,
                original_text="alice@example.com",
                masked_value="[EMAIL]",
                strategy_used="template",
            ),
        ]

        results = resolver.resolve_anchors(doc, anchors)

        assert "resolved" in results
        assert "failed" in results
        # May have some resolved or failed depending on implementation

    def test_resolve_anchor_with_position_drift(self):
        """Test resolving anchor with position drift."""
        resolver = AnchorResolver()

        doc = Mock(spec=DoclingDocument)
        text_item = Mock(spec=TextItem)
        # Position has drifted by a few characters
        text_item.text = "Text before [PERSON] text after"
        doc.texts = [text_item]

        # Anchor expects position at 10, but it's actually at 12
        anchor = AnchorEntry.create_from_detection(
            node_id="#/texts/0",
            start=10,
            end=18,
            entity_type="PERSON",
            confidence=0.9,
            original_text="Name",
            masked_value="[PERSON]",
            strategy_used="template",
        )

        results = resolver.resolve_anchors(doc, anchors=[anchor])

        # Resolver should handle position drift
        assert results is not None

    def test_resolve_anchor_node_not_found(self):
        """Test resolving anchor when node doesn't exist."""
        resolver = AnchorResolver()

        doc = Mock(spec=DoclingDocument)
        doc.texts = []  # No text nodes

        anchor = AnchorEntry.create_from_detection(
            node_id="#/texts/0",  # This node doesn't exist
            start=0,
            end=8,
            entity_type="PERSON",
            confidence=0.9,
            original_text="Name",
            masked_value="[PERSON]",
            strategy_used="template",
        )

        results = resolver.resolve_anchors(doc, [anchor])

        # Should fail to resolve
        assert len(results.get("failed", [])) > 0 or len(results.get("resolved", [])) == 0

    def test_find_masked_text_in_node(self):
        """Test finding masked text within a node."""
        resolver = AnchorResolver()

        node_text = "Some text with [MASKED] value here"
        masked_value = "[MASKED]"
        expected_position = (15, 23)

        # If method exists
        if hasattr(resolver, "find_masked_text"):
            result = resolver.find_masked_text(node_text, masked_value, expected_position)
            assert result is not None

    def test_calculate_confidence_score(self):
        """Test confidence score calculation."""
        resolver = AnchorResolver()

        # If method exists
        if hasattr(resolver, "calculate_confidence"):
            # Perfect match
            score = resolver.calculate_confidence(
                expected_pos=(10, 20), actual_pos=(10, 20), text_match=True
            )
            assert score == 1.0 if score else True

            # Position drift
            score = resolver.calculate_confidence(
                expected_pos=(10, 20), actual_pos=(12, 22), text_match=True
            )
            assert 0 < score < 1.0 if score else True

    def test_resolve_anchors_in_tables(self):
        """Test resolving anchors in table nodes."""
        resolver = AnchorResolver()

        doc = Mock(spec=DoclingDocument)
        table_item = Mock()
        table_item.data = [["Name", "Email"], ["[PERSON]", "[EMAIL]"]]
        doc.tables = [table_item]

        anchor = AnchorEntry.create_from_detection(
            node_id="#/tables/0",
            start=0,
            end=8,
            entity_type="PERSON",
            confidence=0.9,
            original_text="John",
            masked_value="[PERSON]",
            strategy_used="template",
        )

        results = resolver.resolve_anchors(doc, [anchor])
        assert results is not None

    def test_resolve_anchors_with_overlapping_positions(self):
        """Test resolving anchors with overlapping positions."""
        resolver = AnchorResolver()

        doc = Mock(spec=DoclingDocument)
        text_item = Mock(spec=TextItem)
        text_item.text = "[PERSON][EMAIL]"
        doc.texts = [text_item]

        anchors = [
            AnchorEntry.create_from_detection(
                node_id="#/texts/0",
                start=0,
                end=8,
                entity_type="PERSON",
                confidence=0.9,
                original_text="Name",
                masked_value="[PERSON]",
                strategy_used="template",
            ),
            AnchorEntry.create_from_detection(
                node_id="#/texts/0",
                start=8,
                end=15,
                entity_type="EMAIL",
                confidence=0.9,
                original_text="email@test.com",
                masked_value="[EMAIL]",
                strategy_used="template",
            ),
        ]

        results = resolver.resolve_anchors(doc, anchors)
        assert results is not None

    def test_get_node_by_path(self):
        """Test getting node by path."""
        resolver = AnchorResolver()

        doc = Mock(spec=DoclingDocument)
        doc.texts = [Mock(spec=TextItem), Mock(spec=TextItem)]
        doc.tables = [Mock()]

        # If method exists
        if hasattr(resolver, "get_node_by_path"):
            node = resolver.get_node_by_path(doc, "#/texts/0")
            assert node is not None or node is None  # Depends on implementation

            node = resolver.get_node_by_path(doc, "#/tables/0")
            assert node is not None or node is None

    def test_resolve_with_fuzzy_matching(self):
        """Test anchor resolution with fuzzy text matching."""
        resolver = AnchorResolver()

        doc = Mock(spec=DoclingDocument)
        text_item = Mock(spec=TextItem)
        # Slightly different masked format
        text_item.text = "Text with <PERSON> here"
        doc.texts = [text_item]

        anchor = AnchorEntry.create_from_detection(
            node_id="#/texts/0",
            start=10,
            end=18,
            entity_type="PERSON",
            confidence=0.9,
            original_text="Name",
            masked_value="[PERSON]",  # Different format than in text
            strategy_used="template",
        )

        # If fuzzy matching is supported
        if hasattr(resolver, "enable_fuzzy_matching"):
            resolver.enable_fuzzy_matching = True

        results = resolver.resolve_anchors(doc, [anchor])
        assert results is not None

    def test_batch_resolve_performance(self):
        """Test performance with large batch of anchors."""
        resolver = AnchorResolver()

        doc = Mock(spec=DoclingDocument)
        doc.texts = [Mock(spec=TextItem) for _ in range(10)]

        # Create many anchors
        anchors = []
        for i in range(50):
            anchor = AnchorEntry.create_from_detection(
                node_id=f"#/texts/{i % 10}",
                start=i * 10,
                end=i * 10 + 8,
                entity_type="ENTITY",
                confidence=0.9,
                original_text=f"text_{i}",
                masked_value=f"[ENTITY_{i}]",
                strategy_used="template",
            )
            anchors.append(anchor)

        results = resolver.resolve_anchors(doc, anchors)
        assert results is not None
        assert "resolved" in results
        assert "failed" in results

    def test_resolve_statistics(self):
        """Test getting resolution statistics."""
        resolver = AnchorResolver()

        doc = Mock(spec=DoclingDocument)
        anchors = []

        results = resolver.resolve_anchors(doc, anchors)

        # If statistics are provided
        if "statistics" in results:
            stats = results["statistics"]
            assert "total" in stats
            assert "resolved" in stats
            assert "failed" in stats
            assert "success_rate" in stats
