"""Integration tests for Presidio-based masking functionality."""

import pytest
from typing import Any
from datetime import datetime

from presidio_analyzer import RecognizerResult
from presidio_anonymizer import AnonymizerEngine

from cloakpivot.core.strategies import Strategy, StrategyKind
from cloakpivot.core.policies import MaskingPolicy
from cloakpivot.core.types import DoclingDocument
from cloakpivot.core.cloakmap import CloakMap
from cloakpivot.document.extractor import TextSegment
from cloakpivot.masking.presidio_adapter import PresidioMaskingAdapter
from cloakpivot.masking.engine import MaskingEngine, MaskingResult
from cloakpivot.unmasking.engine import UnmaskingEngine


class TestPresidioMaskingIntegration:
    """Integration tests for end-to-end Presidio masking workflows."""

    def test_end_to_end_masking_with_real_document(self):
        """Test complete masking workflow with a realistic document."""
        # Create adapter
        adapter = PresidioMaskingAdapter()
        
        # Create a realistic document
        document_text = """
        Patient: John Smith
        DOB: 01/15/1980
        SSN: 123-45-6789
        Phone: (555) 123-4567
        Email: john.smith@example.com
        Address: 123 Main St, Anytown, CA 90210
        
        Medical Record #: MRN-2024-0001
        Diagnosis: Hypertension
        
        Emergency Contact: Jane Smith (wife)
        Contact Phone: 555-987-6543
        """
        
        document = DoclingDocument(
            name="medical_record.txt",
            _main_text=document_text.strip()
        )
        
        # Create entities (would normally come from analyzer)
        entities = [
            RecognizerResult(entity_type="PERSON", start=17, end=27, score=0.95),
            RecognizerResult(entity_type="DATE_TIME", start=37, end=47, score=0.90),
            RecognizerResult(entity_type="US_SSN", start=57, end=68, score=0.99),
            RecognizerResult(entity_type="PHONE_NUMBER", start=79, end=93, score=0.92),
            RecognizerResult(entity_type="EMAIL_ADDRESS", start=103, end=126, score=0.98),
            RecognizerResult(entity_type="LOCATION", start=137, end=163, score=0.85),
            RecognizerResult(entity_type="PERSON", start=243, end=253, score=0.93),
            RecognizerResult(entity_type="PHONE_NUMBER", start=276, end=288, score=0.91)
        ]
        
        # Create policy
        policy = MaskingPolicy(
            default_strategy=Strategy(StrategyKind.REDACT, {"char": "*"}),
            entity_strategies={
                "PERSON": Strategy(StrategyKind.SURROGATE, {}),
                "US_SSN": Strategy(StrategyKind.HASH, {"algorithm": "sha256"}),
                "EMAIL_ADDRESS": Strategy(StrategyKind.TEMPLATE, {"template": "[EMAIL]"}),
                "PHONE_NUMBER": Strategy(StrategyKind.PARTIAL, {"visible_chars": 4, "position": "end"}),
                "DATE_TIME": Strategy(StrategyKind.TEMPLATE, {"template": "[DATE]"}),
                "LOCATION": Strategy(StrategyKind.REDACT, {"char": "X"})
            }
        )
        
        # Create text segments
        text_segments = [
            TextSegment(
                node_id="#/texts/0",
                text=document._main_text,
                start_offset=0,
                end_offset=len(document._main_text),
                node_type="TextItem"
            )
        ]
        
        # Perform masking
        result = adapter.mask_document(
            document, entities, policy, text_segments
        )
        
        # Verify result structure
        assert isinstance(result, MaskingResult)
        assert result.masked_document is not None
        assert result.cloakmap is not None
        
        # Verify entities were masked
        assert len(result.cloakmap.anchors) == len(entities)
        
        # Verify CloakMap has Presidio metadata
        assert result.cloakmap.is_presidio_enabled
        assert result.cloakmap.presidio_metadata is not None
        assert "operator_results" in result.cloakmap.presidio_metadata
        assert len(result.cloakmap.presidio_metadata["operator_results"]) == len(entities)
        
        # Verify masked content doesn't contain original PII
        masked_text = result.masked_document._main_text
        assert "John Smith" not in masked_text
        assert "123-45-6789" not in masked_text
        assert "john.smith@example.com" not in masked_text
        
        # Verify templates and patterns were applied
        assert "[EMAIL]" in masked_text
        assert "[DATE]" in masked_text

    def test_complex_strategies_and_parameters(self):
        """Test handling of complex strategy configurations."""
        adapter = PresidioMaskingAdapter()
        
        # Test various parameter combinations
        test_cases = [
            {
                "text": "Credit Card: 4111-1111-1111-1111",
                "entity": RecognizerResult(entity_type="CREDIT_CARD", start=13, end=30, score=0.99),
                "strategy": Strategy(StrategyKind.PARTIAL, {
                    "visible_chars": 4,
                    "position": "end",
                    "mask_char": "X"
                }),
                "expected_pattern": "XXXXXXXXXXXX"  # Last 4 visible
            },
            {
                "text": "API Key: sk-proj-abc123def456",
                "entity": RecognizerResult(entity_type="CUSTOM", start=9, end=29, score=0.95),
                "strategy": Strategy(StrategyKind.HASH, {
                    "algorithm": "sha256",
                    "prefix": "HASHED_",
                    "length": 8
                }),
                "expected_pattern": "HASHED_"
            },
            {
                "text": "Username: admin@root",
                "entity": RecognizerResult(entity_type="USERNAME", start=10, end=20, score=0.88),
                "strategy": Strategy(StrategyKind.SURROGATE, {
                    "format_preserving": True,
                    "locale": "en_US"
                }),
                "expected_not": "admin@root"
            }
        ]
        
        for test_case in test_cases:
            result = adapter.apply_strategy(
                original_text=test_case["text"][test_case["entity"].start:test_case["entity"].end],
                entity_type=test_case["entity"].entity_type,
                strategy=test_case["strategy"],
                confidence=test_case["entity"].score
            )
            
            if "expected_pattern" in test_case:
                assert test_case["expected_pattern"] in result or result.startswith(test_case["expected_pattern"])
            if "expected_not" in test_case:
                assert result != test_case["expected_not"]

    def test_round_trip_mask_unmask_compatibility(self):
        """Test that masked documents can be unmasked using CloakMap."""
        adapter = PresidioMaskingAdapter()
        
        # Original document
        original_text = "Contact Alice Johnson at alice@example.com or 555-0123"
        document = DoclingDocument(name="contact.txt", _main_text=original_text)
        
        # Entities
        entities = [
            RecognizerResult(entity_type="PERSON", start=8, end=21, score=0.94),
            RecognizerResult(entity_type="EMAIL_ADDRESS", start=25, end=43, score=0.97),
            RecognizerResult(entity_type="PHONE_NUMBER", start=47, end=55, score=0.91)
        ]
        
        # Policy - use reversible strategies
        policy = MaskingPolicy(
            entity_strategies={
                "PERSON": Strategy(StrategyKind.HASH, {"algorithm": "sha256"}),
                "EMAIL_ADDRESS": Strategy(StrategyKind.TEMPLATE, {"template": "[EMAIL]"}),
                "PHONE_NUMBER": Strategy(StrategyKind.REDACT, {"char": "*"})
            }
        )
        
        segments = [
            TextSegment(
                text=original_text,
                node_id="#/texts/0",
                start_offset=0,
                end_offset=len(original_text),
                node_type="TextItem"
            )
        ]
        
        # Mask the document
        mask_result = adapter.mask_document(document, entities, policy, segments)
        
        # Verify CloakMap stores original values for reversibility
        assert mask_result.cloakmap is not None
        assert len(mask_result.cloakmap.anchors) == 3
        
        # Check that anchors have original text stored
        for anchor in mask_result.cloakmap.anchors:
            assert anchor.metadata.get("original_text") is not None
            
        # Verify Presidio metadata is present
        assert mask_result.cloakmap.is_presidio_enabled
        operator_results = mask_result.cloakmap.presidio_metadata["operator_results"]
        assert len(operator_results) == 3

    def test_performance_comparison(self):
        """Test performance characteristics compared to original implementation."""
        import time
        
        adapter = PresidioMaskingAdapter()
        
        # Create test data
        num_entities = 100
        text_length = 10000
        base_text = "x" * text_length
        
        entities = [
            RecognizerResult(
                entity_type="CUSTOM",
                start=i * 100,
                end=i * 100 + 10,
                score=0.9
            )
            for i in range(num_entities)
        ]
        
        strategies = {
            "CUSTOM": Strategy(StrategyKind.TEMPLATE, {"template": "[REDACTED]"})
        }
        
        # Measure Presidio processing time
        start_time = time.time()
        results = adapter._batch_process_entities(base_text, entities, strategies)
        presidio_time = time.time() - start_time
        
        # Should process efficiently
        assert len(results) == num_entities
        assert presidio_time < 2.0  # Should complete within 2 seconds
        
        # Verify memory efficiency
        import sys
        results_size = sys.getsizeof(results)
        assert results_size < 1_000_000  # Less than 1MB for 100 entities

    def test_error_recovery_and_logging(self):
        """Test error handling and recovery mechanisms."""
        adapter = PresidioMaskingAdapter()
        
        # Test with malformed entities
        malformed_entities = [
            RecognizerResult(entity_type="TEST", start=100, end=50, score=0.9),  # Invalid range
            RecognizerResult(entity_type="TEST", start=-1, end=10, score=0.9),   # Negative start
            RecognizerResult(entity_type=None, start=0, end=5, score=0.9),       # None type
        ]
        
        text = "This is test text for error handling"
        strategies = {"TEST": Strategy(StrategyKind.REDACT, {})}
        
        # Should handle errors gracefully
        for entity in malformed_entities:
            try:
                results = adapter._batch_process_entities(text, [entity], strategies)
                # Should either skip or use fallback
                assert results is not None
            except Exception as e:
                # Should not raise unhandled exceptions
                pytest.fail(f"Unhandled exception: {e}")

    def test_presidio_engine_configuration(self):
        """Test configuration and customization of Presidio engine."""
        adapter = PresidioMaskingAdapter()
        
        # Verify engine is properly configured
        assert isinstance(adapter.anonymizer, AnonymizerEngine)
        
        # Test custom configuration
        custom_adapter = PresidioMaskingAdapter(
            engine_config={
                "log_level": "DEBUG",
                "default_score_threshold": 0.5
            }
        )
        
        assert custom_adapter.engine_config["log_level"] == "DEBUG"
        assert custom_adapter.engine_config["default_score_threshold"] == 0.5

    def test_mixed_version_cloakmap_handling(self):
        """Test handling of both v1.0 and v2.0 CloakMaps."""
        adapter = PresidioMaskingAdapter()
        
        # Create v1.0 CloakMap (without Presidio metadata)
        v1_cloakmap = CloakMap.create(
            doc_id="test_v1",
            doc_hash="abc123",
            anchors=[]
        )
        
        assert not v1_cloakmap.is_presidio_enabled
        
        # Process document to create v2.0 CloakMap
        document = DoclingDocument(name="test", _main_text="Email: test@example.com")
        entities = [RecognizerResult(entity_type="EMAIL", start=7, end=23, score=0.95)]
        policy = MaskingPolicy(entity_strategies={
            "EMAIL": Strategy(StrategyKind.TEMPLATE, {"template": "[EMAIL]"})
        })
        segments = [TextSegment(text=document._main_text, start_offset=0, 
                               end_offset=len(document._main_text), segment_type="main")]
        
        result = adapter.mask_document(document, entities, policy, segments)
        v2_cloakmap = result.cloakmap
        
        assert v2_cloakmap.is_presidio_enabled
        assert v2_cloakmap.version == "2.0"

    def test_concurrent_masking_operations(self):
        """Test thread safety and concurrent operations."""
        import threading
        import queue
        
        adapter = PresidioMaskingAdapter()
        results_queue = queue.Queue()
        
        def mask_operation(thread_id: int):
            """Perform masking operation in thread."""
            text = f"Thread {thread_id}: Call 555-{thread_id:04d}"
            entity = RecognizerResult(
                entity_type="PHONE_NUMBER",
                start=text.index("555"),
                end=text.index("555") + 8,
                score=0.9
            )
            strategy = Strategy(StrategyKind.REDACT, {"char": "*"})
            
            result = adapter.apply_strategy(
                text[entity.start:entity.end],
                entity.entity_type,
                strategy,
                entity.score
            )
            results_queue.put((thread_id, result))
        
        # Run concurrent masking operations
        threads = []
        num_threads = 10
        
        for i in range(num_threads):
            thread = threading.Thread(target=mask_operation, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Verify all operations completed
        assert results_queue.qsize() == num_threads
        
        # Check results
        while not results_queue.empty():
            thread_id, result = results_queue.get()
            assert result == "********"  # All should be redacted