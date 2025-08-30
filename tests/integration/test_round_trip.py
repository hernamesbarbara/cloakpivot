"""Round-trip fidelity tests for CloakPivot masking/unmasking operations.

These tests ensure that the mask/unmask cycle preserves document content exactly,
verifying data integrity across different document types and masking strategies.
"""

import time

import pytest
from docling_core.types import DoclingDocument

from cloakpivot.core.policies import MaskingPolicy
from cloakpivot.core.strategies import StrategyKind
from cloakpivot.masking.engine import MaskingEngine
from cloakpivot.unmasking.engine import UnmaskingEngine
from tests.utils.assertions import (
    assert_masking_result_valid,
    assert_performance_acceptable,
    assert_round_trip_fidelity,
)
from tests.utils.generators import (
    DocumentGenerator,
    PolicyGenerator,
    TextGenerator,
    generate_test_suite_data,
)
from tests.utils.masking_helpers import mask_document_with_detection


class TestRoundTripFidelity:
    """Test suite for round-trip masking/unmasking fidelity."""

    @pytest.fixture
    def masking_engine(self) -> MaskingEngine:
        """Create masking engine for testing."""
        return MaskingEngine()

    @pytest.fixture
    def unmasking_engine(self) -> UnmaskingEngine:
        """Create unmasking engine for testing."""
        return UnmaskingEngine()

    def perform_round_trip(
        self,
        document: DoclingDocument,
        policy: MaskingPolicy,
        masking_engine: MaskingEngine,
        unmasking_engine: UnmaskingEngine
    ) -> tuple[DoclingDocument, DoclingDocument, float]:
        """Perform complete round-trip and return results with timing."""
        start_time = time.time()

        # Mask document - use helper since we need entity detection
        mask_result = mask_document_with_detection(document, policy)
        assert_masking_result_valid(mask_result)

        # Unmask document
        unmask_result = unmasking_engine.unmask_document(
            mask_result.masked_document,
            mask_result.cloakmap
        )

        processing_time = time.time() - start_time

        return mask_result.masked_document, unmask_result.restored_document, processing_time

    @pytest.mark.integration
    def test_simple_document_round_trip(
        self,
        simple_document: DoclingDocument,
        basic_masking_policy: MaskingPolicy,
        masking_engine: MaskingEngine,
        unmasking_engine: UnmaskingEngine
    ):
        """Test round-trip fidelity with simple document."""
        masked_doc, unmasked_doc, processing_time = self.perform_round_trip(
            simple_document, basic_masking_policy, masking_engine, unmasking_engine
        )

        # Verify fidelity
        assert_round_trip_fidelity(simple_document, masked_doc, unmasked_doc, None)

        # Check performance
        text_length = len(simple_document.texts[0].text)
        assert_performance_acceptable(processing_time, 5.0, text_length)

    @pytest.mark.integration
    def test_complex_document_round_trip(
        self,
        complex_document: DoclingDocument,
        strict_masking_policy: MaskingPolicy,
        masking_engine: MaskingEngine,
        unmasking_engine: UnmaskingEngine
    ):
        """Test round-trip fidelity with complex multi-section document."""
        masked_doc, unmasked_doc, processing_time = self.perform_round_trip(
            complex_document, strict_masking_policy, masking_engine, unmasking_engine
        )

        # Verify fidelity
        assert_round_trip_fidelity(complex_document, masked_doc, unmasked_doc, None)

        # Verify all text sections are preserved
        assert len(complex_document.texts) == len(unmasked_doc.texts)
        for orig, unmask in zip(complex_document.texts, unmasked_doc.texts):
            assert orig.text == unmask.text
            assert orig.label == unmask.label
            assert orig.self_ref == unmask.self_ref

    @pytest.mark.integration
    @pytest.mark.parametrize("privacy_level", [
        "low",
        "medium",
        "high"
    ])
    def test_privacy_level_round_trip(
        self,
        privacy_level: str,
        masking_engine: MaskingEngine,
        unmasking_engine: UnmaskingEngine
    ):
        """Test round-trip fidelity across different privacy levels."""
        # Generate test document with PII
        document, pii_locations = DocumentGenerator.generate_document_with_pii(
            ["PHONE_NUMBER", "EMAIL_ADDRESS", "US_SSN", "PERSON"],
            f"privacy_test_{privacy_level}"
        )

        # Create policy for privacy level
        policy = PolicyGenerator.generate_comprehensive_policy(privacy_level)

        # Perform round-trip
        masked_doc, unmasked_doc, processing_time = self.perform_round_trip(
            document, policy, masking_engine, unmasking_engine
        )

        # Verify fidelity
        assert_round_trip_fidelity(document, masked_doc, unmasked_doc, None)

        # Performance should be reasonable regardless of privacy level
        text_length = len(document.texts[0].text)
        assert_performance_acceptable(processing_time, 10.0, text_length)

    @pytest.mark.integration
    @pytest.mark.parametrize("strategy_kind", [
        StrategyKind.TEMPLATE,
        StrategyKind.REDACT,
        StrategyKind.HASH,
        StrategyKind.SURROGATE,
    ])
    def test_strategy_specific_round_trip(
        self,
        strategy_kind: StrategyKind,
        masking_engine: MaskingEngine,
        unmasking_engine: UnmaskingEngine
    ):
        """Test round-trip fidelity for specific masking strategies."""
        # Create document with specific entity type
        entity_map = {
            StrategyKind.TEMPLATE: "PHONE_NUMBER",
            StrategyKind.REDACT: "PERSON",
            StrategyKind.HASH: "EMAIL_ADDRESS",
            StrategyKind.SURROGATE: "US_SSN"
        }

        entity_type = entity_map[strategy_kind]
        document, _ = DocumentGenerator.generate_document_with_pii(
            [entity_type],
            f"strategy_test_{strategy_kind.value}"
        )

        # Create policy with specific strategy
        policy = PolicyGenerator.generate_custom_policy(
            {entity_type: strategy_kind},
            {entity_type: 0.5}
        )

        # Perform round-trip
        masked_doc, unmasked_doc, processing_time = self.perform_round_trip(
            document, policy, masking_engine, unmasking_engine
        )

        # Verify fidelity
        assert_round_trip_fidelity(document, masked_doc, unmasked_doc, None)

    @pytest.mark.integration
    def test_large_document_round_trip(
        self,
        large_document: DoclingDocument,
        basic_masking_policy: MaskingPolicy,
        masking_engine: MaskingEngine,
        unmasking_engine: UnmaskingEngine
    ):
        """Test round-trip fidelity with large document."""
        masked_doc, unmasked_doc, processing_time = self.perform_round_trip(
            large_document, basic_masking_policy, masking_engine, unmasking_engine
        )

        # Verify fidelity
        assert_round_trip_fidelity(large_document, masked_doc, unmasked_doc, None)

        # Performance should be reasonable for large documents
        total_text_length = sum(len(item.text) for item in large_document.texts)
        assert_performance_acceptable(processing_time, 30.0, total_text_length)

        # Memory usage should be reasonable
        # This is a placeholder - in practice, you'd measure actual memory usage
        assert len(large_document.texts) == len(unmasked_doc.texts)

    @pytest.mark.integration
    def test_structured_content_round_trip(
        self,
        masking_engine: MaskingEngine,
        unmasking_engine: UnmaskingEngine,
        basic_masking_policy: MaskingPolicy
    ):
        """Test round-trip fidelity with structured content like forms and tables."""
        structured_text = TextGenerator.generate_structured_content()
        document = DocumentGenerator.generate_simple_document(
            structured_text,
            "structured_content_test"
        )

        masked_doc, unmasked_doc, processing_time = self.perform_round_trip(
            document, basic_masking_policy, masking_engine, unmasking_engine
        )

        # Verify fidelity
        assert_round_trip_fidelity(document, masked_doc, unmasked_doc, None)

        # Verify structure preservation in detail
        original_lines = structured_text.split('\n')
        unmasked_lines = unmasked_doc.texts[0].text.split('\n')

        assert len(original_lines) == len(unmasked_lines)

        # Check that form structure markers are preserved
        for orig_line, unmask_line in zip(original_lines, unmasked_lines):
            # Empty lines should be preserved
            if not orig_line.strip():
                assert not unmask_line.strip()

            # Lines with colons (field labels) should maintain structure
            if ':' in orig_line and not orig_line.strip().endswith(':'):
                assert ':' in unmask_line

    @pytest.mark.integration
    def test_multiple_documents_batch_round_trip(
        self,
        masking_engine: MaskingEngine,
        unmasking_engine: UnmaskingEngine
    ):
        """Test round-trip fidelity with batch of diverse documents."""
        test_data = generate_test_suite_data(num_documents=5)

        for _i, (document, policy) in enumerate(test_data):
            masked_doc, unmasked_doc, processing_time = self.perform_round_trip(
                document, policy, masking_engine, unmasking_engine
            )

            # Verify fidelity for each document
            assert_round_trip_fidelity(document, masked_doc, unmasked_doc, None)

            # Performance should be reasonable
            text_length = sum(len(item.text) for item in document.texts)
            assert_performance_acceptable(processing_time, 15.0, text_length)

    @pytest.mark.integration
    def test_edge_cases_round_trip(
        self,
        masking_engine: MaskingEngine,
        unmasking_engine: UnmaskingEngine,
        basic_masking_policy: MaskingPolicy
    ):
        """Test round-trip fidelity with edge cases."""
        edge_cases = [
            {
                "name": "empty_document",
                "text": ""
            },
            {
                "name": "whitespace_only",
                "text": "   \n\n   \t\t   \n   "
            },
            {
                "name": "special_characters",
                "text": "Contact: john@example.com 📧 Phone: 555-123-4567 📞 Special chars: !@#$%^&*()_+-=[]{}|;:,.<>?"
            },
            {
                "name": "unicode_content",
                "text": "Nom: François Müller, Téléphone: 555-123-4567, Courriel: françois@exemple.com 🌍"
            },
            {
                "name": "very_long_line",
                "text": "Contact info: " + "john.doe@company.com " * 100 + "Phone: 555-123-4567 " * 50
            },
            {
                "name": "mixed_line_endings",
                "text": "Line 1\nLine 2\r\nLine 3\rLine 4"
            }
        ]

        for edge_case in edge_cases:
            document = DocumentGenerator.generate_simple_document(
                edge_case["text"],
                edge_case["name"]
            )

            try:
                masked_doc, unmasked_doc, processing_time = self.perform_round_trip(
                    document, basic_masking_policy, masking_engine, unmasking_engine
                )

                # Verify fidelity
                assert_round_trip_fidelity(document, masked_doc, unmasked_doc, None)

            except Exception as e:
                pytest.fail(f"Edge case '{edge_case['name']}' failed: {str(e)}")

    @pytest.mark.integration
    def test_round_trip_with_overlapping_entities(
        self,
        masking_engine: MaskingEngine,
        unmasking_engine: UnmaskingEngine,
        basic_masking_policy: MaskingPolicy
    ):
        """Test round-trip fidelity when entities overlap or are adjacent."""
        # Create text with potentially overlapping entities
        overlapping_text = """
        Contact John Smith at john.smith@company.com or call 555-123-4567.
        Alternative contact: johnsmith@personal.com or 555-123-4568.
        Emergency: Jane Smith-Johnson at jane.smith.johnson@emergency.com, 555-999-8888.
        """

        document = DocumentGenerator.generate_simple_document(
            overlapping_text,
            "overlapping_entities_test"
        )

        masked_doc, unmasked_doc, processing_time = self.perform_round_trip(
            document, basic_masking_policy, masking_engine, unmasking_engine
        )

        # Verify fidelity
        assert_round_trip_fidelity(document, masked_doc, unmasked_doc, None)

    @pytest.mark.integration
    @pytest.mark.parametrize("batch_size", [3, 5])
    def test_stress_round_trip_batched(
        self,
        batch_size: int,
        masking_engine: MaskingEngine,
        unmasking_engine: UnmaskingEngine
    ):
        """Test round-trip fidelity with smaller batches for faster execution."""
        # Generate smaller test data batches
        test_data = generate_test_suite_data(num_documents=batch_size)

        success_count = 0
        total_time = 0

        for document, policy in test_data:
            try:
                masked_doc, unmasked_doc, processing_time = self.perform_round_trip(
                    document, policy, masking_engine, unmasking_engine
                )

                assert_round_trip_fidelity(document, masked_doc, unmasked_doc, None)
                success_count += 1
                total_time += processing_time

            except Exception as e:
                pytest.fail(f"Stress test failed on document '{document.name}': {str(e)}")

        # Verify all tests passed
        assert success_count == len(test_data)

        # Average performance should be reasonable for smaller batches
        avg_time = total_time / len(test_data)
        assert avg_time < 5.0, f"Average processing time {avg_time:.2f}s exceeds threshold"

    @pytest.mark.integration
    def test_stress_round_trip_document_variety(
        self,
        masking_engine: MaskingEngine,
        unmasking_engine: UnmaskingEngine
    ):
        """Test round-trip fidelity with diverse document types."""
        # Test specific document variety instead of large batch
        test_cases = [
            # Simple document
            generate_test_suite_data(num_documents=1)[0],
            # Structured document  
            generate_test_suite_data(num_documents=2)[1],
            # Multi-section document
            generate_test_suite_data(num_documents=3)[2]
        ]

        for i, (document, policy) in enumerate(test_cases):
            try:
                masked_doc, unmasked_doc, processing_time = self.perform_round_trip(
                    document, policy, masking_engine, unmasking_engine
                )

                assert_round_trip_fidelity(document, masked_doc, unmasked_doc, None)
                
                # Each individual test should be fast
                assert processing_time < 3.0, f"Document {i} processing time {processing_time:.2f}s too slow"

            except Exception as e:
                pytest.fail(f"Document variety test failed on case {i}: {str(e)}")
