"""Integration tests for Presidio unmasking workflows."""

from docling_core.types import DoclingDocument
from presidio_analyzer import RecognizerResult

from cloakpivot.core.cloakmap import CloakMap
from cloakpivot.core.cloakmap_enhancer import CloakMapEnhancer
from cloakpivot.core.policies import MaskingPolicy
from cloakpivot.core.strategies import Strategy, StrategyKind
from cloakpivot.masking.presidio_adapter import PresidioMaskingAdapter
from cloakpivot.unmasking.presidio_adapter import PresidioUnmaskingAdapter


class TestPresidioUnmaskingIntegration:
    """Integration tests for Presidio unmasking workflows."""

    def test_round_trip_mask_unmask_simple(self):
        """Test round-trip masking and unmasking with simple replacements."""
        # Original document
        original_doc = DoclingDocument(name="test_doc")
        original_doc._main_text = "Call me at 555-1234 or email john@example.com"

        # Create masking adapter and policy
        masking_adapter = PresidioMaskingAdapter()
        policy = MaskingPolicy(
            per_entity={
                "PHONE_NUMBER": Strategy(StrategyKind.TEMPLATE, {"template": "[PHONE]"}),
                "EMAIL_ADDRESS": Strategy(StrategyKind.TEMPLATE, {"template": "[EMAIL]"})
            }
        )

        # Create mock entities (normally from analyzer)
        entities = [
            RecognizerResult(
                entity_type="PHONE_NUMBER",
                start=11,
                end=19,
                score=0.95
            ),
            RecognizerResult(
                entity_type="EMAIL_ADDRESS",
                start=29,
                end=46,
                score=0.95
            )
        ]

        # Mask the document
        masked_text = original_doc._main_text
        operator_results = []

        for entity in entities:
            original_value = masked_text[entity.start:entity.end]
            strategy = policy.get_strategy_for_entity(entity.entity_type)

            masked_value = masking_adapter.apply_strategy(
                original_value,
                entity.entity_type,
                strategy,
                entity.score
            )

            # Store operator result for unmasking
            operator_results.append({
                "entity_type": entity.entity_type,
                "start": entity.start,
                "end": entity.end,
                "operator": "replace",
                "text": masked_value,
                "original_text": original_value
            })

            # Apply mask to text (in reverse order to maintain positions)
        for entity, result in sorted(zip(entities, operator_results),
                                     key=lambda x: x[0].start, reverse=True):
            masked_text = (
                masked_text[:entity.start] +
                result["text"] +
                masked_text[entity.end:]
            )

        masked_doc = DoclingDocument(name="test_doc")
        masked_doc._main_text = masked_text

        # Create enhanced CloakMap
        enhancer = CloakMapEnhancer()
        cloakmap = CloakMap(
            version="1.0",
            doc_id="test_doc",
            doc_hash="abc123",
            anchors=[]
        )

        enhanced_cloakmap = enhancer.add_presidio_metadata(
            cloakmap,
            operator_results,
            engine_version="2.2.x",
            reversible_operators=["replace"]
        )

        # Unmask the document
        unmasking_adapter = PresidioUnmaskingAdapter()
        result = unmasking_adapter.unmask_document(masked_doc, enhanced_cloakmap)

        # Verify round-trip integrity
        assert result.restored_document._main_text == original_doc._main_text
        assert result.stats["presidio_restored"] == 2
        assert result.stats["presidio_failed"] == 0

    def test_mixed_reversible_non_reversible_operations(self):
        """Test mixed operations with both reversible and non-reversible masking."""
        # Original document
        original_doc = DoclingDocument(name="test_doc")
        original_doc._main_text = "SSN: 123-45-6789, Phone: 555-1234"

        # Create operator results
        operator_results = [
            {
                "entity_type": "SSN",
                "start": 5,
                "end": 16,
                "operator": "redact",  # Non-reversible
                "text": "***********"
            },
            {
                "entity_type": "PHONE_NUMBER",
                "start": 25,
                "end": 33,
                "operator": "replace",  # Reversible
                "text": "[PHONE]",
                "original_text": "555-1234"
            }
        ]

        # Create masked document
        masked_doc = DoclingDocument(name="test_doc")
        masked_doc._main_text = "SSN: ***********, Phone: [PHONE]"

        # Create enhanced CloakMap
        enhancer = CloakMapEnhancer()
        cloakmap = CloakMap(
            version="1.0",
            doc_id="test_doc",
            doc_hash="xyz789",
            anchors=[]  # SSN would need anchors for restoration
        )

        enhanced_cloakmap = enhancer.add_presidio_metadata(
            cloakmap,
            operator_results,
            reversible_operators=["replace"]
        )

        # Unmask the document
        unmasking_adapter = PresidioUnmaskingAdapter()
        result = unmasking_adapter.unmask_document(masked_doc, enhanced_cloakmap)

        # Verify partial restoration
        assert "555-1234" in result.restored_document._main_text  # Phone restored
        assert "***********" in result.restored_document._main_text  # SSN still redacted
        assert result.stats["presidio_restored"] == 1  # Only phone restored

    def test_v1_to_v2_migration_scenario(self):
        """Test migration from v1.0 to v2.0 CloakMap with Presidio metadata."""
        # Create v1.0 CloakMap
        v1_cloakmap = CloakMap(
            version="1.0",
            doc_id="legacy_doc",
            doc_hash="legacy_hash",
            anchors=[]
        )

        # Create masked document
        masked_doc = DoclingDocument(name="legacy_doc")
        masked_doc._main_text = "Data: [MASKED_1], [MASKED_2]"

        # Simulate adding Presidio metadata during migration
        operator_results = [
            {
                "entity_type": "DATA1",
                "start": 6,
                "end": 16,
                "operator": "replace",
                "text": "[MASKED_1]",
                "original_text": "value1"
            },
            {
                "entity_type": "DATA2",
                "start": 18,
                "end": 28,
                "operator": "replace",
                "text": "[MASKED_2]",
                "original_text": "value2"
            }
        ]

        # Migrate to v2.0
        enhancer = CloakMapEnhancer()
        v2_cloakmap = enhancer.migrate_to_v2(
            v1_cloakmap,
            operator_results,
            engine_version="2.2.x"
        )

        # Verify migration
        assert v2_cloakmap.version == "2.0"
        assert enhancer.is_presidio_enabled(v2_cloakmap)

        # Unmask using v2.0 features
        unmasking_adapter = PresidioUnmaskingAdapter()
        result = unmasking_adapter.unmask_document(masked_doc, v2_cloakmap)

        # Verify restoration
        assert "value1" in result.restored_document._main_text
        assert "value2" in result.restored_document._main_text
        assert result.stats["method"] in ["presidio", "hybrid"]

    def test_error_recovery_and_partial_restoration(self):
        """Test error recovery when some operations fail."""
        # Create masked document
        masked_doc = DoclingDocument(name="test_doc")
        masked_doc._main_text = "Valid: [VALID], Invalid: [INVALID], Good: [GOOD]"

        # Create operator results with mixed validity
        operator_results = [
            {
                "entity_type": "VALID",
                "start": 7,
                "end": 14,
                "operator": "replace",
                "text": "[VALID]",
                "original_text": "data1"
            },
            {
                "entity_type": "INVALID",
                "start": 25,
                "end": 34,
                "operator": "custom",  # Custom without reverse function
                "text": "[INVALID]"
                # Missing reverse function - will fail
            },
            {
                "entity_type": "GOOD",
                "start": 42,
                "end": 48,
                "operator": "replace",
                "text": "[GOOD]",
                "original_text": "data3"
            }
        ]

        # Create CloakMap with operator results
        cloakmap = CloakMap(
            version="2.0",
            doc_id="test_doc",
            doc_hash="test_hash",
            anchors=[],
            presidio_metadata={
                "operator_results": operator_results,
                "reversible_operators": ["replace", "custom"]
            }
        )

        # Unmask the document
        unmasking_adapter = PresidioUnmaskingAdapter()
        result = unmasking_adapter.unmask_document(masked_doc, cloakmap)

        # Verify partial restoration
        assert "data1" in result.restored_document._main_text  # Valid restored
        assert "[INVALID]" in result.restored_document._main_text  # Invalid not restored
        assert "data3" in result.restored_document._main_text  # Good restored
        assert result.stats["presidio_restored"] == 2
        assert result.stats["presidio_failed"] == 1

    def test_performance_comparison(self):
        """Test performance of Presidio vs anchor-based restoration."""
        import time

        # Create a document with many entities
        num_entities = 100
        text_parts = ["Original text "]

        for i in range(num_entities):
            text_parts.append(f"entity_{i} ")

        # Create operator results for all entities
        operator_results = []
        current_pos = len("Original text ")

        for i in range(num_entities):
            entity_text = f"entity_{i}"
            operator_results.append({
                "entity_type": f"TYPE_{i}",
                "start": current_pos,
                "end": current_pos + len(entity_text),
                "operator": "replace",
                "text": f"[MASKED_{i}]",
                "original_text": entity_text
            })
            current_pos += len(entity_text) + 1

        # Create masked document
        masked_doc = DoclingDocument(name="perf_test")
        masked_text_parts = ["Original text "]
        for i in range(num_entities):
            masked_text_parts.append(f"[MASKED_{i}] ")
        masked_doc._main_text = "".join(masked_text_parts)

        # Create v2.0 CloakMap
        cloakmap = CloakMap(
            version="2.0",
            doc_id="perf_test",
            doc_hash="perf_hash",
            anchors=[],
            presidio_metadata={
                "operator_results": operator_results,
                "reversible_operators": ["replace"]
            }
        )

        # Measure restoration time
        unmasking_adapter = PresidioUnmaskingAdapter()

        start_time = time.time()
        result = unmasking_adapter.unmask_document(masked_doc, cloakmap)
        end_time = time.time()

        restoration_time = end_time - start_time

        # Verify all entities were restored
        for i in range(num_entities):
            assert f"entity_{i}" in result.restored_document._main_text

        assert result.stats["presidio_restored"] == num_entities

        # Performance should be reasonable (under 1 second for 100 entities)
        assert restoration_time < 1.0, f"Restoration took {restoration_time:.3f}s"
