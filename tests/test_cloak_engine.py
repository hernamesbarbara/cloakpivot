"""Test suite for simplified CloakEngine API."""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch

from docling_core.types import DoclingDocument
from docling_core.types.doc.document import DocItem, DocItemLabel
from docling_core.types.doc.labels import DocItemLabel

from cloakpivot import CloakEngine, CloakEngineBuilder, CloakedDocument
from cloakpivot.defaults import get_default_policy, get_conservative_policy
from cloakpivot.core.policies import MaskingPolicy
from cloakpivot.core.strategies import Strategy, StrategyKind


class TestCloakEngineBasics:
    """Test basic CloakEngine functionality."""

    def test_default_initialization(self):
        """Test CloakEngine initializes with sensible defaults."""
        engine = CloakEngine()
        assert engine is not None
        assert engine.default_policy is not None
        assert engine.analyzer_config is not None
        assert engine.analyzer_config.languages == ["en"]
        assert engine.analyzer_config.confidence_threshold == 0.7

    def test_custom_initialization(self):
        """Test CloakEngine with custom configuration."""
        custom_policy = get_conservative_policy()
        engine = CloakEngine(
            analyzer_config={'languages': ['es'], 'confidence_threshold': 0.8},
            default_policy=custom_policy
        )
        assert engine.default_policy == custom_policy
        assert engine.analyzer_config.languages == ['es']
        assert engine.analyzer_config.confidence_threshold == 0.8

    def test_mask_document_with_defaults(self, sample_document):
        """Test simple masking workflow with defaults."""
        engine = CloakEngine()
        result = engine.mask_document(sample_document)

        assert result is not None
        assert result.document is not None
        assert result.cloakmap is not None
        assert result.entities_found >= 0
        assert result.entities_masked >= 0
        assert result.entities_masked <= result.entities_found

    def test_mask_document_with_specific_entities(self, sample_document):
        """Test masking with specific entity types."""
        engine = CloakEngine()
        result = engine.mask_document(
            sample_document,
            entities=['EMAIL_ADDRESS', 'PERSON']
        )

        assert result is not None
        # Check that only requested entity types were detected
        for anchor in result.cloakmap.anchors:
            assert anchor.entity_type in ['EMAIL_ADDRESS', 'PERSON']

    def test_mask_document_with_custom_policy(self, sample_document):
        """Test masking with custom policy."""
        engine = CloakEngine()
        custom_policy = MaskingPolicy(
            per_entity={
                "EMAIL_ADDRESS": Strategy(
                    kind=StrategyKind.REDACT,
                    params={"replacement": "[REMOVED]"}
                )
            },
            default_strategy=Strategy(
                kind=StrategyKind.KEEP,
                params={}
            )
        )

        result = engine.mask_document(sample_document, policy=custom_policy)
        assert result is not None

    def test_unmask_document(self, sample_document):
        """Test unmask restores original content."""
        engine = CloakEngine()

        # Mask the document
        mask_result = engine.mask_document(sample_document)
        masked_doc = mask_result.document
        cloakmap = mask_result.cloakmap

        # Unmask the document
        unmasked_doc = engine.unmask_document(masked_doc, cloakmap)

        assert unmasked_doc is not None
        # For a proper test, we'd need to compare specific fields
        # This is a simplified check
        assert isinstance(unmasked_doc, DoclingDocument)


class TestCloakEngineBuilder:
    """Test CloakEngineBuilder functionality."""

    def test_builder_basic(self):
        """Test basic builder usage."""
        engine = CloakEngine.builder().build()
        assert engine is not None
        assert isinstance(engine, CloakEngine)

    def test_builder_with_languages(self):
        """Test builder with custom languages."""
        engine = CloakEngine.builder() \
            .with_languages(['en', 'es', 'fr']) \
            .build()

        assert engine.analyzer_config.languages == ['en', 'es', 'fr']

    def test_builder_with_confidence_threshold(self):
        """Test builder with custom confidence threshold."""
        engine = CloakEngine.builder() \
            .with_confidence_threshold(0.9) \
            .build()

        assert engine.analyzer_config.confidence_threshold == 0.9

    def test_builder_with_invalid_threshold(self):
        """Test builder rejects invalid threshold."""
        with pytest.raises(ValueError):
            CloakEngine.builder() \
                .with_confidence_threshold(1.5) \
                .build()

    def test_builder_with_custom_policy(self):
        """Test builder with custom policy."""
        custom_policy = get_conservative_policy()
        engine = CloakEngine.builder() \
            .with_custom_policy(custom_policy) \
            .build()

        assert engine.default_policy == custom_policy

    def test_builder_chaining(self):
        """Test builder method chaining."""
        engine = CloakEngine.builder() \
            .with_languages(['es']) \
            .with_confidence_threshold(0.8) \
            .with_decision_process(True) \
            .with_custom_policy(get_default_policy()) \
            .build()

        assert engine.analyzer_config.languages == ['es']
        assert engine.analyzer_config.confidence_threshold == 0.8
        assert engine.analyzer_config.return_decision_process == True

    def test_builder_reset(self):
        """Test builder reset functionality."""
        builder = CloakEngine.builder() \
            .with_languages(['es']) \
            .with_confidence_threshold(0.9)

        # Reset should clear settings
        builder.reset()
        engine = builder.build()

        assert engine.analyzer_config.languages == ['en']  # Back to default
        assert engine.analyzer_config.confidence_threshold == 0.7  # Back to default


class TestCloakedDocument:
    """Test CloakedDocument wrapper functionality."""

    def test_cloaked_document_creation(self, sample_document, sample_cloakmap):
        """Test CloakedDocument creation and properties."""
        cloaked = CloakedDocument(sample_document, sample_cloakmap)

        assert cloaked is not None
        assert cloaked.document == sample_document
        assert cloaked.cloakmap == sample_cloakmap
        assert cloaked.is_masked == (len(sample_cloakmap.anchors) > 0)
        assert cloaked.entities_masked == len(sample_cloakmap.anchors)

    def test_cloaked_document_delegation(self, sample_document, sample_cloakmap):
        """Test that CloakedDocument delegates to underlying document."""
        cloaked = CloakedDocument(sample_document, sample_cloakmap)

        # Test delegation of DoclingDocument methods
        # These should work if the underlying document has these methods
        assert hasattr(cloaked, 'export_to_markdown')
        assert hasattr(cloaked, 'export_to_dict')
        assert hasattr(cloaked, 'export_to_text')

    def test_cloaked_document_unmask(self, sample_document, sample_cloakmap):
        """Test unmasking through CloakedDocument."""
        engine = CloakEngine()
        cloaked = CloakedDocument(sample_document, sample_cloakmap, engine)

        unmasked = cloaked.unmask_pii()
        assert unmasked is not None
        assert isinstance(unmasked, DoclingDocument)

    def test_cloaked_document_save_cloakmap(self, tmp_path, sample_document, sample_cloakmap):
        """Test saving CloakMap from CloakedDocument."""
        cloaked = CloakedDocument(sample_document, sample_cloakmap)

        # Test JSON save
        json_path = tmp_path / "test.cloakmap.json"
        cloaked.save_cloakmap(str(json_path), format="json")
        assert json_path.exists()

        # Test YAML save
        yaml_path = tmp_path / "test.cloakmap.yaml"
        cloaked.save_cloakmap(str(yaml_path), format="yaml")
        assert yaml_path.exists()

        # Test invalid format
        with pytest.raises(ValueError):
            cloaked.save_cloakmap(str(tmp_path / "test.txt"), format="invalid")


class TestMethodRegistration:
    """Test method registration on DoclingDocument."""

    def test_register_methods(self, sample_document):
        """Test registering methods on DoclingDocument."""
        from cloakpivot import register_cloak_methods, unregister_cloak_methods, is_registered

        # Clean state
        if is_registered():
            unregister_cloak_methods()

        # Register methods
        register_cloak_methods()
        assert is_registered()

        # Check methods exist
        assert hasattr(sample_document, 'mask_pii')
        assert hasattr(sample_document, 'unmask_pii')

        # Clean up
        unregister_cloak_methods()
        assert not is_registered()

    def test_register_with_custom_engine(self, sample_document):
        """Test registration with custom engine."""
        from cloakpivot import register_cloak_methods, unregister_cloak_methods

        # Create custom engine
        custom_engine = CloakEngine.builder() \
            .with_languages(['es']) \
            .build()

        # Register with custom engine
        register_cloak_methods(custom_engine)

        # The registered engine should be our custom one
        from cloakpivot.registration import get_registered_engine
        assert get_registered_engine() == custom_engine

        # Clean up
        unregister_cloak_methods()


class TestRoundTrip:
    """Test complete mask/unmask round trips."""

    def test_simple_round_trip(self, sample_document_with_pii):
        """Test mask/unmask round trip preserves document."""
        engine = CloakEngine()

        # Get original content for comparison
        original_text = sample_document_with_pii.export_to_text()

        # Mask the document
        masked_result = engine.mask_document(sample_document_with_pii)
        masked_text = masked_result.document.export_to_text()

        # Verify masking changed the content (if PII was found)
        if masked_result.entities_masked > 0:
            assert masked_text != original_text

        # Unmask the document
        unmasked = engine.unmask_document(
            masked_result.document,
            masked_result.cloakmap
        )
        unmasked_text = unmasked.export_to_text()

        # Verify round trip preserved content
        assert unmasked_text == original_text

    def test_round_trip_with_custom_policy(self, sample_document_with_pii):
        """Test round trip with custom masking policy."""
        engine = CloakEngine()
        custom_policy = get_conservative_policy()

        # Mask with custom policy
        masked_result = engine.mask_document(
            sample_document_with_pii,
            policy=custom_policy
        )

        # Unmask
        unmasked = engine.unmask_document(
            masked_result.document,
            masked_result.cloakmap
        )

        # Verify round trip
        original_text = sample_document_with_pii.export_to_text()
        unmasked_text = unmasked.export_to_text()
        assert unmasked_text == original_text


# Fixtures for testing

@pytest.fixture
def sample_document():
    """Create a sample DoclingDocument for testing."""
    doc = DoclingDocument(name="test_doc.pdf")

    # Add some basic content
    item = DocItem(
        label=DocItemLabel.PARAGRAPH,
        text="This is a test document with sample content."
    )
    doc.add_item(item)

    return doc


@pytest.fixture
def sample_document_with_pii():
    """Create a sample document containing PII."""
    doc = DoclingDocument(name="test_pii.pdf")

    # Add content with PII
    item = DocItem(
        label=DocItemLabel.PARAGRAPH,
        text="Contact John Doe at john.doe@example.com or call 555-123-4567."
    )
    doc.add_item(item)

    return doc


@pytest.fixture
def sample_cloakmap():
    """Create a sample CloakMap for testing."""
    from cloakpivot.core.cloakmap import CloakMap
    from cloakpivot.core.anchors import AnchorEntry

    cloakmap = CloakMap()

    # Add a sample anchor
    anchor = AnchorEntry(
        entity_type="EMAIL_ADDRESS",
        start=20,
        end=40,
        original_text="john.doe@example.com",
        masked_text="[EMAIL]",
        strategy=Strategy(kind=StrategyKind.TEMPLATE, params={"template": "[EMAIL]"}),
        confidence=0.95
    )
    cloakmap.add_anchor(anchor)

    return cloakmap