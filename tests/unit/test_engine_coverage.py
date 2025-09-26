"""Tests to improve engine coverage."""

import json
from unittest.mock import Mock

from cloakpivot.core.policies.policies import MaskingPolicy
from cloakpivot.core.types.cloakmap import CloakMap
from cloakpivot.core.types.strategies import Strategy, StrategyKind
from cloakpivot.engine import CloakEngine
from cloakpivot.type_imports import DoclingDocument


def create_mock_docling_document(text: str, name: str = "test.txt") -> Mock:
    """Create a mock DoclingDocument with all required attributes."""
    doc = Mock(spec=DoclingDocument)
    doc.export_to_markdown.return_value = text
    doc.name = name

    # Add texts attribute with mock text items
    text_item = Mock()
    text_item.text = text
    text_item.self_ref = "#/texts/0"  # Add self_ref attribute
    text_item.label = "text"  # Add label attribute
    text_item.orig = text  # Add orig attribute
    doc.texts = [text_item]
    doc.tables = []  # Empty tables list
    doc.key_value_items = []  # Empty key-value items list

    # Mock the model_dump_json() method for Pydantic serialization
    doc_dict = {
        "name": name,
        "texts": [{"text": text, "self_ref": "#/texts/0", "label": "text", "orig": text}],
        "tables": [],
        "key_value_items": [],
    }
    doc.model_dump_json.return_value = json.dumps(doc_dict)

    return doc


class TestEngineBasicCoverage:
    """Basic tests to improve engine coverage."""

    def test_engine_creation_default(self):
        """Test creating engine with defaults."""
        engine = CloakEngine()
        assert engine is not None
        assert hasattr(engine, "mask_document")
        assert hasattr(engine, "unmask_document")

    def test_engine_creation_with_params(self):
        """Test creating engine with parameters."""
        analyzer_config = {
            "languages": ["en"],
            "entities": ["PERSON", "EMAIL"],
            "confidence_threshold": 0.8,
        }
        engine = CloakEngine(analyzer_config=analyzer_config)
        assert engine is not None

    def test_engine_with_default_policy(self):
        """Test engine with default policy."""
        strategy = Strategy(kind=StrategyKind.REDACT)
        policy = MaskingPolicy(
            default_strategy=strategy,
            per_entity={"EMAIL": strategy},
        )

        engine = CloakEngine(default_policy=policy)
        assert engine is not None

    def test_mask_document_basic(self):
        """Test basic mask_document functionality."""
        engine = CloakEngine()

        # Create a mock DoclingDocument with all required attributes
        doc = create_mock_docling_document("Test document with email@example.com", "test.txt")

        result = engine.mask_document(doc)

        assert result is not None
        assert hasattr(result, "document")
        assert hasattr(result, "cloakmap")
        assert hasattr(result, "entities_found")
        assert hasattr(result, "entities_masked")

    def test_mask_document_with_entities(self):
        """Test mask_document with specific entities."""
        engine = CloakEngine()

        doc = create_mock_docling_document("John Doe sent an email", "test.txt")

        result = engine.mask_document(doc, entities=["PERSON"])

        assert result is not None
        assert hasattr(result, "document")
        assert hasattr(result, "cloakmap")

    def test_mask_document_with_policy(self):
        """Test mask_document with custom policy."""
        engine = CloakEngine()

        doc = create_mock_docling_document("Contact: email@example.com", "test.txt")

        strategy = Strategy(kind=StrategyKind.REDACT)
        policy = MaskingPolicy(
            default_strategy=strategy,
            per_entity={"EMAIL": strategy},
        )

        result = engine.mask_document(doc, policy=policy)

        assert result is not None
        assert hasattr(result, "document")
        assert hasattr(result, "cloakmap")

    def test_unmask_document_basic(self):
        """Test basic unmask_document functionality."""
        engine = CloakEngine()

        doc = create_mock_docling_document("Masked document with [EMAIL]", "masked.txt")

        cloakmap = CloakMap(doc_id="test_doc", doc_hash="test_hash", anchors=[])

        result = engine.unmask_document(doc, cloakmap)

        assert result is not None

    def test_engine_with_custom_config(self):
        """Test engine with custom configuration."""
        conflict_config = {
            "overlap_policy": "resolve_all",
        }

        engine = CloakEngine(
            conflict_resolution_config=conflict_config,
        )
        assert engine is not None

    def test_engine_with_all_params(self):
        """Test engine with all parameters."""
        analyzer_config = {
            "confidence_threshold": 0.9,
        }

        strategy = Strategy(kind=StrategyKind.SURROGATE)
        policy = MaskingPolicy(
            default_strategy=strategy,
            per_entity={"PERSON": strategy},
        )

        conflict_config = {
            "overlap_policy": "allow_partial",
        }

        engine = CloakEngine(
            analyzer_config=analyzer_config,
            default_policy=policy,
            conflict_resolution_config=conflict_config,
        )
        assert engine is not None

    def test_mask_document_with_confidence_threshold(self):
        """Test masking with different confidence threshold."""
        analyzer_config = {"confidence_threshold": 0.5}
        engine = CloakEngine(analyzer_config=analyzer_config)

        doc = create_mock_docling_document("Some text", "test.txt")

        result = engine.mask_document(doc)
        assert result is not None

    def test_unmask_with_empty_cloakmap(self):
        """Test unmasking with empty cloakmap."""
        engine = CloakEngine()

        doc = create_mock_docling_document("No masked content", "test.txt")

        cloakmap = CloakMap(doc_id="test_doc", doc_hash="test_hash", anchors=[])

        result = engine.unmask_document(doc, cloakmap)
        assert result is not None

    def test_mask_document_entities_parameter_types(self):
        """Test entities parameter with different types."""
        engine = CloakEngine()

        doc = create_mock_docling_document("Test text", "test.txt")

        # Test with list of strings
        result1 = engine.mask_document(doc, entities=["PERSON", "EMAIL"])
        assert result1 is not None

        # Test with None
        result2 = engine.mask_document(doc, entities=None)
        assert result2 is not None

        # Test with empty list
        result3 = engine.mask_document(doc, entities=[])
        assert result3 is not None
