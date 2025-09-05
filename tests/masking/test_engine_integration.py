"""Tests for MaskingEngine integration with Presidio feature flags."""

import os
from unittest import mock

import pytest
from docling_core.types.doc.document import TextItem
from presidio_analyzer import RecognizerResult

from cloakpivot.core.policies import MaskingPolicy
from cloakpivot.core.strategies import Strategy, StrategyKind
from cloakpivot.core.types import DoclingDocument
from cloakpivot.document.extractor import TextSegment
from cloakpivot.masking.engine import MaskingEngine


class TestEngineIntegration:
    """Test MaskingEngine with Presidio integration and feature flags."""

    @pytest.fixture
    def sample_document(self):
        """Create a sample DoclingDocument for testing."""
        doc = DoclingDocument(
            name="test_doc",
            texts=[],
            tables=[],
            key_value_items=[]
        )

        # Add text content
        text_item = TextItem(
            text="John Smith's email is john.smith@example.com and phone is 555-123-4567.",
            self_ref="#/texts/0",
            label="text",
            orig="John Smith's email is john.smith@example.com and phone is 555-123-4567."
        )
        doc.texts = [text_item]
        doc._main_text = text_item.text

        return doc

    @pytest.fixture
    def entities(self):
        """Create sample PII entities."""
        return [
            RecognizerResult(
                entity_type="PERSON",
                start=0,
                end=10,
                score=0.95
            ),
            RecognizerResult(
                entity_type="EMAIL_ADDRESS",
                start=22,
                end=45,
                score=0.98
            ),
            RecognizerResult(
                entity_type="PHONE_NUMBER",
                start=58,
                end=70,
                score=0.92
            )
        ]

    @pytest.fixture
    def text_segments(self):
        """Create text segments."""
        return [
            TextSegment(
                node_id="#/texts/0",
                text="John Smith's email is john.smith@example.com and phone is 555-123-4567.",
                start_offset=0,
                end_offset=71,
                node_type="text"
            )
        ]

    @pytest.fixture
    def masking_policy(self):
        """Create a masking policy."""
        return MaskingPolicy(
            default_strategy=Strategy(StrategyKind.REDACT, {"char": "*"}),
            per_entity={
                "PERSON": Strategy(StrategyKind.TEMPLATE, {"template": "[PERSON]"}),
                "EMAIL_ADDRESS": Strategy(StrategyKind.REDACT, {"char": "X"}),
                "PHONE_NUMBER": Strategy(StrategyKind.PARTIAL, {
                    "visible_chars": 4,
                    "position": "end",
                    "mask_char": "*"
                })
            }
        )

    def test_default_uses_legacy_engine(self, sample_document, entities, masking_policy, text_segments):
        """Test that default configuration uses legacy engine."""
        engine = MaskingEngine()
        assert engine.use_presidio is False
        assert engine.strategy_applicator is not None
        assert engine.presidio_adapter is None

        # Should be able to mask with legacy engine
        result = engine.mask_document(
            document=sample_document,
            entities=entities,
            policy=masking_policy,
            text_segments=text_segments
        )

        assert result is not None
        assert result.masked_document is not None
        assert result.cloakmap is not None

    def test_explicit_flag_enables_presidio(self):
        """Test that explicit flag enables Presidio engine."""
        engine = MaskingEngine(use_presidio_engine=True)
        assert engine.use_presidio is True
        assert engine.strategy_applicator is None
        assert engine.presidio_adapter is not None

    def test_explicit_flag_disables_presidio(self):
        """Test that explicit False flag disables Presidio engine."""
        engine = MaskingEngine(use_presidio_engine=False)
        assert engine.use_presidio is False
        assert engine.strategy_applicator is not None
        assert engine.presidio_adapter is None

    @mock.patch.dict(os.environ, {"CLOAKPIVOT_USE_PRESIDIO_ENGINE": "true"})
    def test_environment_variable_enables_presidio(self):
        """Test that environment variable enables Presidio engine."""
        engine = MaskingEngine()
        assert engine.use_presidio is True
        assert engine.presidio_adapter is not None

    @mock.patch.dict(os.environ, {"CLOAKPIVOT_USE_PRESIDIO_ENGINE": "false"})
    def test_environment_variable_disables_presidio(self):
        """Test that environment variable explicitly disables Presidio engine."""
        engine = MaskingEngine()
        assert engine.use_presidio is False
        assert engine.strategy_applicator is not None

    @mock.patch.dict(os.environ, {"CLOAKPIVOT_USE_PRESIDIO_ENGINE": "yes"})
    def test_environment_variable_yes_enables_presidio(self):
        """Test that 'yes' value enables Presidio engine."""
        engine = MaskingEngine()
        assert engine.use_presidio is True

    @mock.patch.dict(os.environ, {"CLOAKPIVOT_USE_PRESIDIO_ENGINE": "1"})
    def test_environment_variable_one_enables_presidio(self):
        """Test that '1' value enables Presidio engine."""
        engine = MaskingEngine()
        assert engine.use_presidio is True

    @mock.patch.dict(os.environ, {"CLOAKPIVOT_USE_PRESIDIO_ENGINE": "on"})
    def test_environment_variable_on_enables_presidio(self):
        """Test that 'on' value enables Presidio engine."""
        engine = MaskingEngine()
        assert engine.use_presidio is True

    @mock.patch.dict(os.environ, {"CLOAKPIVOT_USE_PRESIDIO_ENGINE": "invalid"})
    def test_environment_variable_invalid_disables_presidio(self):
        """Test that invalid value disables Presidio engine."""
        engine = MaskingEngine()
        assert engine.use_presidio is False

    @mock.patch.dict(os.environ, {"CLOAKPIVOT_USE_PRESIDIO_ENGINE": "true"})
    def test_explicit_flag_overrides_environment(self):
        """Test that explicit flag takes precedence over environment variable."""
        # Explicit False should override environment True
        engine = MaskingEngine(use_presidio_engine=False)
        assert engine.use_presidio is False

        # Explicit True should also work
        engine = MaskingEngine(use_presidio_engine=True)
        assert engine.use_presidio is True

    def test_both_engines_produce_valid_results(
        self, sample_document, entities, masking_policy, text_segments
    ):
        """Test that both engines produce valid MaskingResult objects."""
        # Test legacy engine
        legacy_engine = MaskingEngine(use_presidio_engine=False)
        legacy_result = legacy_engine.mask_document(
            document=sample_document,
            entities=entities,
            policy=masking_policy,
            text_segments=text_segments
        )

        assert legacy_result is not None
        assert legacy_result.masked_document is not None
        assert legacy_result.cloakmap is not None
        assert len(legacy_result.cloakmap.anchors) == len(entities)

        # Test Presidio engine
        presidio_engine = MaskingEngine(use_presidio_engine=True)
        presidio_result = presidio_engine.mask_document(
            document=sample_document,
            entities=entities,
            policy=masking_policy,
            text_segments=text_segments
        )

        assert presidio_result is not None
        assert presidio_result.masked_document is not None
        assert presidio_result.cloakmap is not None
        assert len(presidio_result.cloakmap.anchors) == len(entities)

    def test_api_compatibility_maintained(
        self, sample_document, entities, masking_policy, text_segments
    ):
        """Test that API remains compatible regardless of engine choice."""
        # Both engines should accept the same parameters
        legacy_engine = MaskingEngine(use_presidio_engine=False)
        presidio_engine = MaskingEngine(use_presidio_engine=True)

        # Same method signature
        legacy_result = legacy_engine.mask_document(
            document=sample_document,
            entities=entities,
            policy=masking_policy,
            text_segments=text_segments,
            original_format="pdf"
        )

        presidio_result = presidio_engine.mask_document(
            document=sample_document,
            entities=entities,
            policy=masking_policy,
            text_segments=text_segments,
            original_format="pdf"
        )

        # Both should return MaskingResult
        assert type(legacy_result).__name__ == "MaskingResult"
        assert type(presidio_result).__name__ == "MaskingResult"

        # Both should have the same attributes
        assert hasattr(legacy_result, "masked_document")
        assert hasattr(legacy_result, "cloakmap")
        assert hasattr(legacy_result, "stats")

        assert hasattr(presidio_result, "masked_document")
        assert hasattr(presidio_result, "cloakmap")
        assert hasattr(presidio_result, "stats")

    def test_conflict_resolution_works_with_both_engines(
        self, sample_document, masking_policy, text_segments
    ):
        """Test that conflict resolution works with both engines."""
        # Create overlapping entities
        overlapping_entities = [
            RecognizerResult(entity_type="PERSON", start=0, end=10, score=0.95),
            RecognizerResult(entity_type="NAME", start=5, end=15, score=0.90)
        ]

        # Legacy engine with conflict resolution
        legacy_engine = MaskingEngine(
            use_presidio_engine=False,
            resolve_conflicts=True
        )
        legacy_result = legacy_engine.mask_document(
            document=sample_document,
            entities=overlapping_entities,
            policy=masking_policy,
            text_segments=text_segments
        )
        assert legacy_result is not None

        # Presidio engine with conflict resolution
        presidio_engine = MaskingEngine(
            use_presidio_engine=True,
            resolve_conflicts=True
        )
        presidio_result = presidio_engine.mask_document(
            document=sample_document,
            entities=overlapping_entities,
            policy=masking_policy,
            text_segments=text_segments
        )
        assert presidio_result is not None

    def test_stats_include_engine_info(
        self, sample_document, entities, masking_policy, text_segments
    ):
        """Test that statistics indicate which engine was used."""
        # Presidio engine should indicate its use in stats
        presidio_engine = MaskingEngine(use_presidio_engine=True)
        presidio_result = presidio_engine.mask_document(
            document=sample_document,
            entities=entities,
            policy=masking_policy,
            text_segments=text_segments
        )

        # Check for Presidio-specific stats
        assert presidio_result.stats is not None
        if "presidio_engine_used" in presidio_result.stats:
            assert presidio_result.stats["presidio_engine_used"] is True
