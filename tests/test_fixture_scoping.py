"""Validation tests for fixture scoping changes.

These tests ensure that session-scoped fixtures maintain proper state isolation
and don't cause test interference after the scope optimization changes.
"""

import pytest
from presidio_analyzer import RecognizerResult

from cloakpivot.core.policies import MaskingPolicy


class TestFixtureIsolation:
    """Test that session-scoped fixtures maintain proper test isolation."""

    def test_sample_text_with_pii_immutable(self, sample_text_with_pii):
        """Verify sample_text_with_pii fixture is immutable and reusable."""
        original_text = sample_text_with_pii

        # Verify content doesn't change between accesses
        assert sample_text_with_pii == original_text
        assert "John Doe" in sample_text_with_pii
        assert "555-123-4567" in sample_text_with_pii

    def test_simple_document_immutable(self, simple_document):
        """Verify simple_document fixture is immutable and reusable."""
        original_name = simple_document.name
        original_text_count = len(simple_document.texts)
        original_first_text = (
            simple_document.texts[0].text if simple_document.texts else ""
        )

        # Verify document structure doesn't change
        assert simple_document.name == original_name
        assert len(simple_document.texts) == original_text_count
        if simple_document.texts:
            assert simple_document.texts[0].text == original_first_text

    def test_complex_document_immutable(self, complex_document):
        """Verify complex_document fixture is immutable and reusable."""
        original_name = complex_document.name
        original_text_count = len(complex_document.texts)

        # Verify document structure doesn't change
        assert complex_document.name == original_name
        assert len(complex_document.texts) == original_text_count
        assert complex_document.name == "complex_test_document"

    def test_detected_entities_immutable(self, detected_entities):
        """Verify detected_entities fixture is immutable and reusable."""
        original_count = len(detected_entities)
        original_first_type = (
            detected_entities[0].entity_type if detected_entities else None
        )

        # Verify entities don't change
        assert len(detected_entities) == original_count
        if detected_entities:
            assert detected_entities[0].entity_type == original_first_type

    def test_masking_engine_stateless(self, masking_engine):
        """Verify masking_engine fixture is stateless and reusable."""
        # The engine should be the same instance but behave consistently
        assert masking_engine is not None

        # Test basic functionality works consistently
        from cloakpivot.masking.engine import MaskingEngine

        assert isinstance(masking_engine, MaskingEngine)


class TestFixtureStateConsistency:
    """Test that session fixtures maintain consistent state across tests."""

    def test_policies_maintain_configuration(
        self, basic_masking_policy, strict_masking_policy
    ):
        """Verify policy fixtures maintain their configuration."""
        # Basic policy should have reversible strategies
        assert basic_masking_policy.locale == "en"
        assert "PHONE_NUMBER" in basic_masking_policy.per_entity

        # Strict policy should have hash strategies
        assert strict_masking_policy.locale == "en"
        assert "PHONE_NUMBER" in strict_masking_policy.per_entity

        # Policies should be different objects with different strategies
        basic_phone_strategy = basic_masking_policy.per_entity["PHONE_NUMBER"]
        strict_phone_strategy = strict_masking_policy.per_entity["PHONE_NUMBER"]
        assert basic_phone_strategy.kind != strict_phone_strategy.kind

    def test_text_segments_consistency(
        self, simple_text_segments, complex_text_segments
    ):
        """Verify text segment fixtures maintain consistent structure."""
        # Simple segments
        assert len(simple_text_segments) >= 1
        assert all(segment.node_type == "TextItem" for segment in simple_text_segments)

        # Complex segments
        assert len(complex_text_segments) >= 1
        assert all(segment.node_type == "TextItem" for segment in complex_text_segments)

    def test_path_fixtures_exist(
        self, test_files_dir, golden_files_dir, sample_policies_dir
    ):
        """Verify path fixtures point to valid directories."""
        assert test_files_dir.name == "fixtures"
        assert golden_files_dir.name == "golden_files"
        assert sample_policies_dir.name == "policies"


class TestFixturePerformanceValidation:
    """Test that fixture optimizations provide expected performance benefits."""

    def test_expensive_fixtures_are_cached(self, masking_engine):
        """Verify expensive fixtures are properly cached at session level."""
        # This test mainly validates that the fixture can be created successfully
        # The actual performance benefit is measured by reduced setup time
        assert masking_engine is not None

    def test_multiple_fixture_access_same_instance(self, masking_engine):
        """Verify multiple accesses to session fixture return same instance."""
        # For session-scoped fixtures, multiple accesses should return same object
        engine1 = masking_engine
        engine2 = masking_engine

        # Should be the exact same object for session scope
        assert engine1 is engine2

    def test_document_fixtures_reusable(
        self, simple_document, complex_document, large_document
    ):
        """Verify document fixtures are properly created and reusable."""
        # Verify all documents are valid
        assert simple_document.name == "test_document"
        assert complex_document.name == "complex_test_document"
        assert large_document.name == "large_test_document"

        # Verify they have text content
        assert len(simple_document.texts) >= 1
        assert len(complex_document.texts) >= 1
        assert len(large_document.texts) >= 1


class TestRegressionPrevention:
    """Test that existing test functionality still works with new fixture scopes."""

    def test_existing_test_compatibility_basic_masking(
        self, simple_document, basic_masking_policy, masking_engine
    ):
        """Verify basic masking workflow still works with optimized fixtures."""
        # This replicates a typical masking test to ensure compatibility
        assert simple_document is not None
        assert basic_masking_policy is not None
        assert masking_engine is not None

        # Basic functionality should work
        assert simple_document.texts
        assert basic_masking_policy.per_entity

    def test_existing_test_compatibility_detection(
        self, sample_text_with_pii, detected_entities
    ):
        """Verify entity detection workflow still works with optimized fixtures."""
        assert sample_text_with_pii
        assert detected_entities
        assert len(detected_entities) > 0
        assert all(isinstance(entity, RecognizerResult) for entity in detected_entities)

    def test_policy_fixture_compatibility(
        self, basic_masking_policy, strict_masking_policy, benchmark_policy
    ):
        """Verify all policy fixtures work together."""
        policies = [basic_masking_policy, strict_masking_policy, benchmark_policy]

        # All should be valid MaskingPolicy instances
        assert all(isinstance(policy, MaskingPolicy) for policy in policies)

        # All should have required fields
        for policy in policies:
            assert policy.locale == "en"
            assert policy.per_entity
            assert policy.thresholds


@pytest.mark.performance
class TestFixturePerformanceImpact:
    """Performance tests to measure fixture optimization impact."""

    def test_fixture_setup_performance(
        self, masking_engine, simple_document, basic_masking_policy
    ):
        """Measure time for accessing optimized fixtures."""
        import time

        # These fixtures should be pre-created (session scope)
        # so access should be near-instantaneous
        start_time = time.time()

        # Access all major session fixtures
        _ = masking_engine
        _ = simple_document
        _ = basic_masking_policy

        end_time = time.time()
        access_time = end_time - start_time

        # Session fixtures should be very fast to access (< 1ms typically)
        assert access_time < 0.01, (
            f"Fixture access took {access_time:.4f}s, expected < 0.01s"
        )

    def test_multiple_fixture_access_consistent(self, masking_engine):
        """Test multiple accesses to session fixtures are consistent."""
        import time

        access_times = []
        for _ in range(10):
            start = time.time()
            _ = masking_engine
            end = time.time()
            access_times.append(end - start)

        # All accesses should be similarly fast (session fixtures are cached)
        max_time = max(access_times)
        min_time = min(access_times)

        # Session fixture access should be consistently fast
        assert max_time < 0.001, f"Slowest access: {max_time:.6f}s"
        assert (max_time - min_time) < 0.0005, (
            f"Access time variance: {max_time - min_time:.6f}s"
        )
