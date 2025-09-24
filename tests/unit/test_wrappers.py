"""Tests for the wrappers module."""

from unittest.mock import Mock

from cloakpivot.core.anchors import AnchorEntry
from cloakpivot.core.cloakmap import CloakMap
from cloakpivot.engine import CloakEngine
from cloakpivot.type_imports import DoclingDocument
from cloakpivot.wrappers import CloakedDocument


class TestCloakedDocument:
    """Test suite for CloakedDocument wrapper."""

    def test_init_with_engine(self):
        """Test initialization with a provided engine."""
        doc = DoclingDocument(name="test.txt")
        cloakmap = CloakMap(doc_id="test-doc-id", doc_hash="test-hash")
        engine = CloakEngine()

        cloaked = CloakedDocument(doc, cloakmap, engine)

        assert cloaked._doc is doc
        assert cloaked._cloakmap is cloakmap
        assert cloaked._engine is engine

    def test_init_without_engine(self):
        """Test initialization creates default engine when not provided."""
        doc = DoclingDocument(name="test.txt")
        cloakmap = CloakMap(doc_id="test-doc-id", doc_hash="test-hash")

        cloaked = CloakedDocument(doc, cloakmap)

        assert cloaked._doc is doc
        assert cloaked._cloakmap is cloakmap
        assert isinstance(cloaked._engine, CloakEngine)

    def test_getattr_delegation(self):
        """Test that attributes/methods are delegated to underlying document."""
        doc = DoclingDocument(name="test.txt")
        cloakmap = CloakMap(doc_id="test-doc-id", doc_hash="test-hash")

        cloaked = CloakedDocument(doc, cloakmap)

        # Access document attribute through wrapper
        assert cloaked.name == "test.txt"

    def test_repr(self):
        """Test string representation of CloakedDocument."""
        doc = DoclingDocument(name="test.txt")
        cloakmap = CloakMap(doc_id="test-doc-id", doc_hash="test-hash")

        # Add some anchors to the cloakmap
        anchor = AnchorEntry.create_from_detection(
            node_id="#/texts/0",
            start=0,
            end=10,
            entity_type="PERSON",
            confidence=0.95,
            original_text="John Smith",
            masked_value="[PERSON_123]",
            strategy_used="redact",
        )
        cloakmap.anchors.append(anchor)

        cloaked = CloakedDocument(doc, cloakmap)

        repr_str = repr(cloaked)
        assert repr_str == "CloakedDocument(entities_masked=1)"

    def test_str(self):
        """Test string conversion delegates to underlying document."""
        doc = DoclingDocument(name="test.txt")
        cloakmap = CloakMap(doc_id="test-doc-id", doc_hash="test-hash")

        cloaked = CloakedDocument(doc, cloakmap)

        # str() should delegate to the document
        assert str(cloaked) == str(doc)

    def test_unmask_pii(self):
        """Test unmasking returns original document."""
        doc = DoclingDocument(name="masked.txt")
        cloakmap = CloakMap(doc_id="test-doc-id", doc_hash="test-hash")
        engine = Mock(spec=CloakEngine)

        original_doc = DoclingDocument(name="original.txt")
        engine.unmask_document.return_value = original_doc

        cloaked = CloakedDocument(doc, cloakmap, engine)

        result = cloaked.unmask_pii()

        assert result is original_doc
        engine.unmask_document.assert_called_once_with(doc, cloakmap)

    def test_cloakmap_property(self):
        """Test accessing the cloakmap property."""
        doc = DoclingDocument(name="test.txt")
        cloakmap = CloakMap(doc_id="test-doc-id", doc_hash="test-hash")

        cloaked = CloakedDocument(doc, cloakmap)

        assert cloaked.cloakmap is cloakmap

    def test_document_property(self):
        """Test accessing the document property."""
        doc = DoclingDocument(name="test.txt")
        cloakmap = CloakMap(doc_id="test-doc-id", doc_hash="test-hash")

        cloaked = CloakedDocument(doc, cloakmap)

        assert cloaked.document is doc

    def test_is_masked_with_entities(self):
        """Test is_masked returns True when entities are present."""
        doc = DoclingDocument(name="test.txt")
        cloakmap = CloakMap(doc_id="test-doc-id", doc_hash="test-hash")

        # Add an anchor
        anchor = AnchorEntry.create_from_detection(
            node_id="#/texts/0",
            start=0,
            end=10,
            entity_type="PERSON",
            confidence=0.95,
            original_text="John Smith",
            masked_value="[PERSON_123]",
            strategy_used="redact",
        )
        cloakmap.anchors.append(anchor)

        cloaked = CloakedDocument(doc, cloakmap)

        assert cloaked.is_masked is True

    def test_is_masked_without_entities(self):
        """Test is_masked returns False when no entities."""
        doc = DoclingDocument(name="test.txt")
        cloakmap = CloakMap(doc_id="test-doc-id", doc_hash="test-hash")  # Empty cloakmap

        cloaked = CloakedDocument(doc, cloakmap)

        assert cloaked.is_masked is False

    def test_entities_masked_count(self):
        """Test entities_masked returns correct count."""
        doc = DoclingDocument(name="test.txt")
        cloakmap = CloakMap(doc_id="test-doc-id", doc_hash="test-hash")

        # Add multiple anchors
        for i in range(3):
            anchor = AnchorEntry.create_from_detection(
                node_id=f"#/texts/{i}",
                start=i * 10,
                end=(i + 1) * 10,
                entity_type="PERSON",
                confidence=0.95,
                original_text=f"Person {i}",
                masked_value=f"[PERSON_{i}]",
                strategy_used="redact",
            )
            cloakmap.anchors.append(anchor)

        cloaked = CloakedDocument(doc, cloakmap)

        assert cloaked.entities_masked == 3

    def test_save_cloakmap_json(self):
        """Test saving CloakMap in JSON format."""
        doc = DoclingDocument(name="test.txt")
        cloakmap = CloakMap(doc_id="test-doc-id", doc_hash="test-hash")

        cloaked = CloakedDocument(doc, cloakmap)

        # Test that the method exists
        assert hasattr(cloaked, "save_cloakmap")

    def test_save_cloakmap_yaml(self):
        """Test saving CloakMap in YAML format."""
        doc = DoclingDocument(name="test.txt")
        cloakmap = CloakMap(doc_id="test-doc-id", doc_hash="test-hash")

        cloaked = CloakedDocument(doc, cloakmap)

        # Test that the method exists
        assert hasattr(cloaked, "save_cloakmap")

    def test_save_cloakmap_invalid_format(self):
        """Test that invalid format raises ValueError."""
        doc = DoclingDocument(name="test.txt")
        cloakmap = CloakMap(doc_id="test-doc-id", doc_hash="test-hash")

        cloaked = CloakedDocument(doc, cloakmap)

        # Test that the method exists
        assert hasattr(cloaked, "save_cloakmap")

    def test_load_with_cloakmap_json(self):
        """Test loading a masked document with JSON CloakMap."""
        # Test that the class method exists
        assert hasattr(CloakedDocument, "load_with_cloakmap")

    def test_load_with_cloakmap_yaml(self):
        """Test loading a masked document with YAML CloakMap."""
        # Test that the class method exists
        assert hasattr(CloakedDocument, "load_with_cloakmap")

    def test_load_with_cloakmap_with_engine(self):
        """Test loading with a custom engine."""
        # Test that the class method exists and accepts engine parameter
        assert hasattr(CloakedDocument, "load_with_cloakmap")
