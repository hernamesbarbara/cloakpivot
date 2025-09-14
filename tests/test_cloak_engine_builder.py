"""Test CloakEngineBuilder configuration."""

import pytest
from docling_core.types import DoclingDocument
from docling_core.types.doc.document import TextItem

from cloakpivot.engine import CloakEngine
from cloakpivot.engine_builder import CloakEngineBuilder
from cloakpivot.core.policies import MaskingPolicy
from cloakpivot.core.strategies import Strategy, StrategyKind
from cloakpivot.core.normalization import ConflictResolutionConfig, ConflictResolutionStrategy


class TestCloakEngineBuilder:
    """Test CloakEngineBuilder fluent configuration."""

    @pytest.fixture
    def test_document(self):
        """Create a test document."""
        doc = DoclingDocument(name="test.txt")
        doc.texts = [TextItem(
            text="Email me at test@example.com or call 555-1234",
            self_ref="#/texts/0",
            label="text",
            orig="Email me at test@example.com or call 555-1234"
        )]
        return doc

    @pytest.fixture
    def custom_policy(self):
        """Create a custom masking policy."""
        return MaskingPolicy(
            per_entity={
                "EMAIL_ADDRESS": Strategy(
                    kind=StrategyKind.HASH,
                    parameters={"algorithm": "sha256", "truncate": 8}
                ),
                "PHONE_NUMBER": Strategy(
                    kind=StrategyKind.PARTIAL,
                    parameters={"visible_chars": 4, "position": "end", "mask_char": "*"}
                )
            },
            default_strategy=Strategy(
                kind=StrategyKind.TEMPLATE,
                parameters={"template": "[MASKED]"}
            )
        )

    def test_builder_basic(self):
        """Test basic builder pattern."""
        engine = CloakEngine.builder().build()
        assert isinstance(engine, CloakEngine)
        assert engine is not None

    def test_builder_with_confidence_threshold(self, test_document):
        """Test setting confidence threshold via builder."""
        engine = CloakEngine.builder()\
            .with_confidence_threshold(0.9)\
            .build()

        assert engine is not None
        assert engine.analyzer_config.min_confidence == 0.9

        # Should work for masking
        result = engine.mask_document(test_document)
        assert result is not None

    def test_builder_with_custom_policy(self, test_document, custom_policy):
        """Test setting custom policy via builder."""
        engine = CloakEngine.builder()\
            .with_custom_policy(custom_policy)\
            .build()

        assert engine is not None
        assert engine.default_policy == custom_policy

        # Masking should use the custom policy
        result = engine.mask_document(test_document)
        assert result is not None

    def test_builder_with_languages(self):
        """Test language configuration via builder."""
        # Note: Multi-language support depends on Presidio configuration
        # For now, just test that it accepts the parameter
        engine = CloakEngine.builder()\
            .with_languages(['en', 'es'])\
            .build()

        assert engine is not None
        # First language should be used
        assert engine.analyzer_config.language == 'en'

    def test_builder_with_decision_process(self):
        """Test enabling decision process via builder."""
        engine = CloakEngine.builder()\
            .with_decision_process(True)\
            .build()

        assert engine is not None
        # Decision process flag should be set
        # (Note: This might not be used in simplified version)

    def test_builder_with_analyzer_config(self):
        """Test complete analyzer configuration via builder."""
        config = {
            'languages': ['en'],
            'confidence_threshold': 0.85,
            'return_decision_process': True
        }

        engine = CloakEngine.builder()\
            .with_analyzer_config(config)\
            .build()

        assert engine is not None
        assert engine.analyzer_config.min_confidence == 0.85

    def test_builder_with_conflict_resolution(self):
        """Test conflict resolution configuration via builder."""
        config = ConflictResolutionConfig(
            strategy=ConflictResolutionStrategy.HIGHEST_CONFIDENCE
        )

        engine = CloakEngine.builder()\
            .with_conflict_resolution(config)\
            .build()

        assert engine is not None
        # Conflict resolution should be configured in the engine

    def test_builder_with_presidio_engine(self):
        """Test Presidio engine configuration via builder."""
        engine = CloakEngine.builder()\
            .with_presidio_engine(True)\
            .build()

        assert engine is not None
        # Should use Presidio engine (this is the default anyway)

    def test_builder_with_additional_recognizers(self):
        """Test adding additional recognizers via builder."""
        engine = CloakEngine.builder()\
            .with_additional_recognizers(['CUSTOM_ID'])\
            .build()

        assert engine is not None
        # Additional recognizers should be configured

    def test_builder_exclude_recognizers(self):
        """Test excluding recognizers via builder."""
        engine = CloakEngine.builder()\
            .exclude_recognizers(['DATE_TIME'])\
            .build()

        assert engine is not None
        # Excluded recognizers should be configured

    def test_builder_chaining(self, custom_policy):
        """Test method chaining in builder."""
        engine = CloakEngine.builder()\
            .with_confidence_threshold(0.8)\
            .with_custom_policy(custom_policy)\
            .with_languages(['en'])\
            .with_decision_process(False)\
            .with_presidio_engine(True)\
            .build()

        assert engine is not None
        assert engine.default_policy == custom_policy
        assert engine.analyzer_config.min_confidence == 0.8

    def test_builder_reset(self):
        """Test builder reset functionality."""
        builder = CloakEngineBuilder()

        # Configure builder
        builder.with_confidence_threshold(0.9)
        builder.with_languages(['es'])

        # Reset should clear configuration
        builder.reset()

        # Build with defaults
        engine = builder.build()
        assert engine is not None
        # Should have default values
        assert engine.analyzer_config.min_confidence == 0.7  # default
        assert engine.analyzer_config.language == 'en'  # default

    def test_builder_multiple_builds(self, custom_policy):
        """Test building multiple engines from same builder."""
        builder = CloakEngine.builder()\
            .with_confidence_threshold(0.9)\
            .with_custom_policy(custom_policy)

        # Build first engine
        engine1 = builder.build()

        # Build second engine (should work)
        engine2 = builder.build()

        assert engine1 is not None
        assert engine2 is not None
        assert engine1 is not engine2  # Different instances

    def test_builder_invalid_threshold(self):
        """Test builder with invalid confidence threshold."""
        with pytest.raises(ValueError):
            CloakEngine.builder()\
                .with_confidence_threshold(1.5)\
                .build()

        with pytest.raises(ValueError):
            CloakEngine.builder()\
                .with_confidence_threshold(-0.1)\
                .build()

    def test_builder_end_to_end(self, test_document, custom_policy):
        """Test complete workflow with builder-configured engine."""
        # Build engine with custom configuration
        engine = CloakEngine.builder()\
            .with_confidence_threshold(0.7)\
            .with_custom_policy(custom_policy)\
            .with_languages(['en'])\
            .build()

        # Mask document
        result = engine.mask_document(test_document)
        assert result.entities_found > 0

        # Unmask document
        unmasked = engine.unmask_document(result.document, result.cloakmap)
        assert unmasked.texts[0].text == test_document.texts[0].text

    def test_builder_preserves_defaults(self):
        """Test that builder preserves sensible defaults when not configured."""
        engine = CloakEngine.builder().build()

        # Should have default policy
        assert engine.default_policy is not None

        # Should have default analyzer config
        assert engine.analyzer_config is not None
        assert engine.analyzer_config.language == 'en'
        assert engine.analyzer_config.min_confidence == 0.7

    def test_builder_partial_analyzer_config(self):
        """Test builder with partial analyzer configuration."""
        # Only set some fields
        config = {
            'confidence_threshold': 0.9
            # languages not set, should use default
        }

        engine = CloakEngine.builder()\
            .with_analyzer_config(config)\
            .build()

        assert engine is not None
        assert engine.analyzer_config.min_confidence == 0.9
        # Should still have default language
        assert engine.analyzer_config.language == 'en'