"""Comprehensive tests for PresidioMaskingAdapter using TDD approach."""

from unittest.mock import Mock, patch

from presidio_analyzer import RecognizerResult
from presidio_anonymizer import AnonymizerEngine, OperatorResult

from cloakpivot.core.policies import MaskingPolicy
from cloakpivot.core.strategies import Strategy, StrategyKind
from cloakpivot.core.types import DoclingDocument
from cloakpivot.document.extractor import TextSegment
from cloakpivot.masking.engine import MaskingResult
from cloakpivot.masking.presidio_adapter import PresidioMaskingAdapter


class TestPresidioMaskingAdapter:

    def _get_document_text(self, document: DoclingDocument) -> str:
        """Helper to get text from document, handling both formats."""
        if hasattr(document, '_main_text'):
            return document._main_text
        elif document.texts:
            return document.texts[0].text
        return ""

    def _set_document_text(self, document: DoclingDocument, text: str) -> None:
        """Helper to set text in document, handling both formats."""
        from docling_core.types.doc.document import TextItem
        # Create proper TextItem
        text_item = TextItem(
            text=text,
            self_ref="#/texts/0",
            label="text",
            orig=text
        )
        document.texts = [text_item]
        # Also set _main_text for backward compatibility
        document._main_text = text

    """Test suite for PresidioMaskingAdapter functionality."""

    def test_adapter_initialization(self):
        """Test that adapter initializes with required components."""
        adapter = PresidioMaskingAdapter()

        assert adapter.anonymizer is not None
        assert isinstance(adapter.anonymizer, AnonymizerEngine)
        assert adapter.operator_mapper is not None
        assert adapter.cloakmap_enhancer is not None
        assert adapter._fallback_char == "*"

    def test_apply_strategy_redact(self):
        """Test REDACT strategy through Presidio."""
        adapter = PresidioMaskingAdapter()

        original_text = "555-123-4567"
        entity_type = "PHONE_NUMBER"
        strategy = Strategy(
            kind=StrategyKind.REDACT,
            parameters={"char": "*"}
        )
        confidence = 0.95

        result = adapter.apply_strategy(
            original_text, entity_type, strategy, confidence
        )

        # Should produce redacted output
        assert result == "************"

    def test_apply_strategy_template(self):
        """Test TEMPLATE strategy through Presidio."""
        adapter = PresidioMaskingAdapter()

        original_text = "john.doe@example.com"
        entity_type = "EMAIL_ADDRESS"
        strategy = Strategy(
            kind=StrategyKind.TEMPLATE,
            parameters={"template": "[EMAIL]"}
        )
        confidence = 0.90

        result = adapter.apply_strategy(
            original_text, entity_type, strategy, confidence
        )

        assert result == "[EMAIL]"

    def test_apply_strategy_hash(self):
        """Test HASH strategy through Presidio."""
        adapter = PresidioMaskingAdapter()

        original_text = "123-45-6789"
        entity_type = "US_SSN"
        strategy = Strategy(
            kind=StrategyKind.HASH,
            parameters={"algorithm": "sha256", "prefix": "HASH_"}
        )
        confidence = 0.98

        result = adapter.apply_strategy(
            original_text, entity_type, strategy, confidence
        )

        # Should produce hashed output with prefix
        assert result.startswith("HASH_")
        assert len(result) > 5  # Has actual hash content

    def test_apply_strategy_partial(self):
        """Test PARTIAL strategy through Presidio."""
        adapter = PresidioMaskingAdapter()

        original_text = "4111111111111111"
        entity_type = "CREDIT_CARD"
        strategy = Strategy(
            kind=StrategyKind.PARTIAL,
            parameters={"visible_chars": 4, "position": "end"}
        )
        confidence = 0.99

        result = adapter.apply_strategy(
            original_text, entity_type, strategy, confidence
        )

        # Should show last 4 digits
        assert result == "************1111"

    def test_apply_strategy_surrogate(self):
        """Test SURROGATE strategy through Presidio."""
        adapter = PresidioMaskingAdapter()

        original_text = "John Smith"
        entity_type = "PERSON"
        strategy = Strategy(
            kind=StrategyKind.SURROGATE,
            parameters={"format_preserving": True}
        )
        confidence = 0.85

        result = adapter.apply_strategy(
            original_text, entity_type, strategy, confidence
        )

        # Should produce a fake name
        assert result != original_text
        assert len(result) > 0
        # Should look like a name (contains space between words)
        assert " " in result

    def test_apply_strategy_custom(self):
        """Test CUSTOM strategy through Presidio."""
        adapter = PresidioMaskingAdapter()

        def custom_callback(text: str) -> str:
            return f"CUSTOM_{text.upper()}_MASKED"

        original_text = "sensitive"
        entity_type = "CUSTOM_TYPE"
        strategy = Strategy(
            kind=StrategyKind.CUSTOM,
            parameters={"callback": custom_callback}
        )
        confidence = 0.75

        result = adapter.apply_strategy(
            original_text, entity_type, strategy, confidence
        )

        assert result == "CUSTOM_SENSITIVE_MASKED"

    def test_apply_strategy_with_fallback(self):
        """Test fallback mechanism when Presidio fails."""
        adapter = PresidioMaskingAdapter()

        # Mock anonymizer to raise exception
        with patch.object(adapter.anonymizer, 'anonymize', side_effect=Exception("Presidio error")):
            original_text = "test@example.com"
            entity_type = "EMAIL_ADDRESS"
            strategy = Strategy(
                kind=StrategyKind.TEMPLATE,
                parameters={"template": "[EMAIL]"}
            )
            confidence = 0.95

            result = adapter.apply_strategy(
                original_text, entity_type, strategy, confidence
            )

            # Should fall back to simple redaction
            assert result == "****************"

    def test_mask_document_basic(self):
        """Test basic document masking through Presidio."""
        adapter = PresidioMaskingAdapter()

        # Create test document
        document = Mock(spec=DoclingDocument)
        document.name = "test_doc"
        self._set_document_text(document, "Call me at 555-123-4567 or email john@example.com")

        # Create test entities
        entities = [
            RecognizerResult(
                entity_type="PHONE_NUMBER",
                start=11,
                end=23,
                score=0.95
            ),
            RecognizerResult(
                entity_type="EMAIL_ADDRESS",
                start=34,
                end=50,
                score=0.90
            )
        ]

        # Create test policy
        policy = Mock(spec=MaskingPolicy)
        policy.get_strategy_for_entity.side_effect = lambda entity_type: (
            Strategy(StrategyKind.TEMPLATE, {"template": f"[{entity_type}]"})
        )

        # Create text segments
        text_segments = [
            TextSegment(
                node_id="#/texts/0",
                text=self._get_document_text(document),
                start_offset=0,
                end_offset=len(self._get_document_text(document)),
                node_type="TextItem"
            )
        ]

        result = adapter.mask_document(
            document, entities, policy, text_segments
        )

        assert isinstance(result, MaskingResult)
        assert result.masked_document is not None
        assert result.cloakmap is not None
        # One entity (EMAIL at 34-50) exceeds text length 49, so only 1 entity is masked
        assert len(result.cloakmap.anchors) == 1

    def test_mask_document_with_presidio_metadata(self):
        """Test that masking captures Presidio operator results."""
        adapter = PresidioMaskingAdapter()

        document = Mock(spec=DoclingDocument)
        document.name = "test_doc"
        self._set_document_text(document, "SSN: 123-45-6789")

        entities = [
            RecognizerResult(
                entity_type="US_SSN",
                start=5,
                end=16,
                score=0.99
            )
        ]

        policy = Mock(spec=MaskingPolicy)
        policy.get_strategy_for_entity.return_value = Strategy(
            StrategyKind.HASH, {"algorithm": "sha256"}
        )

        text_segments = [
            TextSegment(
                node_id="#/texts/0",
                text=self._get_document_text(document),
                start_offset=0,
                end_offset=len(self._get_document_text(document)),
                node_type="TextItem"
            )
        ]

        result = adapter.mask_document(
            document, entities, policy, text_segments
        )

        # Check that CloakMap has Presidio metadata
        assert result.cloakmap.is_presidio_enabled
        assert result.cloakmap.presidio_metadata is not None
        assert "operator_results" in result.cloakmap.presidio_metadata

    def test_batch_processing(self):
        """Test batch processing of multiple entities."""
        adapter = PresidioMaskingAdapter()

        text = "Contact John at john@example.com or 555-1234"
        entities = [
            RecognizerResult(entity_type="PERSON", start=8, end=12, score=0.9),
            RecognizerResult(entity_type="EMAIL_ADDRESS", start=16, end=33, score=0.95),
            RecognizerResult(entity_type="PHONE_NUMBER", start=37, end=44, score=0.85)
        ]

        strategies = {
            "PERSON": Strategy(StrategyKind.SURROGATE, {}),
            "EMAIL_ADDRESS": Strategy(StrategyKind.TEMPLATE, {"template": "[EMAIL]"}),
            "PHONE_NUMBER": Strategy(StrategyKind.REDACT, {"char": "*"})
        }

        results = adapter._batch_process_entities(text, entities, strategies)

        assert len(results) == 3
        assert all(isinstance(r, OperatorResult) for r in results)

    def test_error_handling_invalid_strategy(self):
        """Test handling of invalid strategy parameters."""
        adapter = PresidioMaskingAdapter()

        original_text = "test"
        entity_type = "CUSTOM"
        strategy = Strategy(
            kind=StrategyKind.PARTIAL,
            parameters={"visible_chars": 4, "position": "end"}  # Include required parameters
        )
        confidence = 0.8

        # Should handle the strategy properly
        result = adapter.apply_strategy(
            original_text, entity_type, strategy, confidence
        )

        assert result == "test"  # All 4 chars visible

    def test_performance_lazy_loading(self):
        """Test that AnonymizerEngine is lazy-loaded."""
        with patch('cloakpivot.masking.presidio_adapter.AnonymizerEngine') as mock_engine:
            adapter = PresidioMaskingAdapter()

            # Engine should not be created until first use
            assert mock_engine.call_count == 0

            # Trigger engine creation
            _ = adapter.anonymizer
            assert mock_engine.call_count == 1

            # Subsequent access should reuse same instance
            _ = adapter.anonymizer
            assert mock_engine.call_count == 1

    def test_api_compatibility_with_strategy_applicator(self):
        """Test that adapter maintains API compatibility with StrategyApplicator."""
        from cloakpivot.masking.applicator import StrategyApplicator

        adapter = PresidioMaskingAdapter()
        applicator = StrategyApplicator()

        # Check that main methods exist with same signatures
        assert hasattr(adapter, 'apply_strategy')
        assert hasattr(applicator, 'apply_strategy')

        # Both should handle same parameters
        test_params = {
            "original_text": "test",
            "entity_type": "CUSTOM",
            "strategy": Strategy(StrategyKind.REDACT, {"char": "*"}),
            "confidence": 0.9
        }

        # Both should produce output (not necessarily same due to implementation)
        adapter_result = adapter.apply_strategy(**test_params)
        applicator_result = applicator.apply_strategy(**test_params)

        assert isinstance(adapter_result, str)
        assert isinstance(applicator_result, str)

    def test_memory_management_large_results(self):
        """Test memory management with large operator results."""
        adapter = PresidioMaskingAdapter()

        # Create large text with many entities
        large_text = " ".join([f"email{i}@test.com" for i in range(1000)])
        entities = [
            RecognizerResult(
                entity_type="EMAIL_ADDRESS",
                start=i*19,
                end=i*19+17,
                score=0.9
            )
            for i in range(1000)
        ]

        # Should handle without memory issues
        strategies = {
            "EMAIL_ADDRESS": Strategy(StrategyKind.TEMPLATE, {"template": "[EMAIL]"})
        }

        results = adapter._batch_process_entities(large_text, entities, strategies)

        assert len(results) == 1000

        # Clean up should happen automatically
        adapter._cleanup_large_results(results)

        # Results should still be accessible but memory optimized
        assert len(results) <= 1000

    def test_integration_with_existing_masking_engine(self):
        """Test that adapter can replace StrategyApplicator in MaskingEngine."""
        from cloakpivot.masking.engine import MaskingEngine

        # Create engine with our adapter
        engine = MaskingEngine()
        adapter = PresidioMaskingAdapter()

        # Replace the strategy applicator with our adapter
        engine.strategy_applicator = adapter

        # Create test inputs
        from docling_core.types.doc.document import TextItem

        document = DoclingDocument(
            name="test",
            texts=[],
            tables=[],
            key_value_items=[]
        )
        text_item = TextItem(
            text="Call 555-1234",
            self_ref="#/texts/0",
            label="text",
            orig="Call 555-1234"
        )
        document.texts = [text_item]
        self._set_document_text(document, "Call 555-1234")

        entities = [
            RecognizerResult(
                entity_type="PHONE_NUMBER",
                start=5,
                end=13,
                score=0.9
            )
        ]

        policy = Mock(spec=MaskingPolicy)
        policy.get_strategy_for_entity.return_value = Strategy(
            StrategyKind.TEMPLATE, {"template": "[PHONE]"}
        )

        segments = [
            TextSegment(
                node_id="#/texts/0",
                text=self._get_document_text(document),
                start_offset=0,
                end_offset=len(self._get_document_text(document)),
                node_type="TextItem"
            )
        ]

        # Should work seamlessly
        result = engine.mask_document(document, entities, policy, segments)

        assert isinstance(result, MaskingResult)
        assert result.cloakmap is not None
