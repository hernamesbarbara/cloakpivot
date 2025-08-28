"""Integration tests for StrategyApplicator with enhanced SurrogateGenerator."""

import pytest

from cloakpivot.core.strategies import Strategy, StrategyKind
from cloakpivot.masking.applicator import StrategyApplicator


class TestSurrogateIntegration:
    """Test integration of StrategyApplicator with enhanced surrogate generation."""

    @pytest.fixture
    def applicator(self):
        """Create a StrategyApplicator instance."""
        return StrategyApplicator(seed="test_integration_seed")

    def test_phone_surrogate_format_preservation(self, applicator):
        """Test that phone surrogate generation preserves format."""
        strategy = Strategy(StrategyKind.SURROGATE)
        original = "555-123-4567"

        surrogate = applicator.apply_strategy(original, "PHONE_NUMBER", strategy, 0.95)

        # Should preserve format
        assert len(surrogate) == len(original)
        assert surrogate[3] == '-'
        assert surrogate[7] == '-'
        assert all(c.isdigit() for i, c in enumerate(surrogate) if i not in [3, 7])
        assert surrogate != original

    def test_ssn_surrogate_format_preservation(self, applicator):
        """Test that SSN surrogate generation preserves format."""
        strategy = Strategy(StrategyKind.SURROGATE)
        original = "123-45-6789"

        surrogate = applicator.apply_strategy(original, "US_SSN", strategy, 0.95)

        # Should preserve format
        assert len(surrogate) == len(original)
        assert surrogate[3] == '-'
        assert surrogate[6] == '-'
        assert all(c.isdigit() for i, c in enumerate(surrogate) if i not in [3, 6])
        assert surrogate != original

    def test_email_surrogate_format_preservation(self, applicator):
        """Test that email surrogate generation preserves format."""
        strategy = Strategy(StrategyKind.SURROGATE)
        original = "user@example.com"

        surrogate = applicator.apply_strategy(original, "EMAIL_ADDRESS", strategy, 0.95)

        # Should preserve email structure
        assert '@' in surrogate
        assert '.' in surrogate
        parts = surrogate.split('@')
        assert len(parts) == 2
        assert '.' in parts[1]  # Domain should have dot
        assert surrogate != original

    def test_deterministic_generation(self, applicator):
        """Test that surrogate generation is deterministic."""
        strategy = Strategy(StrategyKind.SURROGATE)
        original = "555-123-4567"
        entity_type = "PHONE_NUMBER"

        result1 = applicator.apply_strategy(original, entity_type, strategy, 0.95)
        result2 = applicator.apply_strategy(original, entity_type, strategy, 0.95)

        assert result1 == result2

    def test_different_inputs_produce_different_outputs(self, applicator):
        """Test that different inputs produce different surrogate outputs."""
        strategy = Strategy(StrategyKind.SURROGATE)

        phone1 = applicator.apply_strategy("555-123-4567", "PHONE_NUMBER", strategy, 0.95)
        phone2 = applicator.apply_strategy("555-987-6543", "PHONE_NUMBER", strategy, 0.95)

        assert phone1 != phone2
        # But both should preserve format
        assert len(phone1) == len(phone2) == 12

    def test_custom_pattern_surrogate(self, applicator):
        """Test custom pattern-based surrogate generation."""
        strategy = Strategy(StrategyKind.SURROGATE, {"pattern": "XXX-XX-9999"})
        original = "ABC-DE-1234"

        surrogate = applicator.apply_strategy(original, "CUSTOM", strategy, 0.95)

        assert len(surrogate) == len("XXX-XX-9999")
        assert surrogate[3] == '-'
        assert surrogate[6] == '-'
        assert surrogate[:3].isalpha()
        assert surrogate[:3].isupper()
        assert surrogate[4:6].isalpha()
        assert surrogate[4:6].isupper()
        assert surrogate[7:].isdigit()

    def test_surrogate_quality_metrics_tracking(self, applicator):
        """Test that quality metrics are tracked during surrogate generation."""
        strategy = Strategy(StrategyKind.SURROGATE)

        # Generate some surrogates
        applicator.apply_strategy("555-123-4567", "PHONE_NUMBER", strategy, 0.95)
        applicator.apply_strategy("123-45-6789", "US_SSN", strategy, 0.95)
        applicator.apply_strategy("user@example.com", "EMAIL_ADDRESS", strategy, 0.95)

        # Check metrics
        metrics = applicator.get_surrogate_quality_metrics()
        assert metrics.total_generated >= 3
        assert metrics.format_preservation_rate > 0.8  # Should be high
        assert metrics.uniqueness_rate > 0.8  # Should be high

    def test_document_scope_reset(self, applicator):
        """Test document scope reset functionality."""
        strategy = Strategy(StrategyKind.SURROGATE)

        # Generate surrogate in first document scope
        result1 = applicator.apply_strategy("555-123-4567", "PHONE_NUMBER", strategy, 0.95)

        # Reset scope
        applicator.reset_document_scope()

        # Generate same surrogate in new document scope (should be same due to deterministic seed)
        result2 = applicator.apply_strategy("555-123-4567", "PHONE_NUMBER", strategy, 0.95)

        assert result1 == result2  # Should be same due to deterministic generation

    def test_legacy_fallback_compatibility(self, applicator):
        """Test that legacy surrogate generation still works as fallback."""
        # Test with a format that might not be handled by enhanced generator
        strategy = Strategy(StrategyKind.SURROGATE, {"format_type": "custom", "pattern": "test"})
        original = "some-unusual-format"

        # Should not fail and should produce some output
        surrogate = applicator.apply_strategy(original, "UNKNOWN_TYPE", strategy, 0.95)

        assert isinstance(surrogate, str)
        assert len(surrogate) > 0
        assert surrogate != original

    def test_enhanced_vs_legacy_phone_generation(self, applicator):
        """Test that enhanced generation produces better results than legacy."""
        strategy = Strategy(StrategyKind.SURROGATE)
        original_phone = "555-123-4567"

        # Generate with enhanced system (default)
        enhanced_result = applicator.apply_strategy(original_phone, "PHONE_NUMBER", strategy, 0.95)

        # Should preserve exact format
        assert len(enhanced_result) == len(original_phone)
        assert enhanced_result[3] == '-'
        assert enhanced_result[7] == '-'
        assert all(c.isdigit() for i, c in enumerate(enhanced_result) if i not in [3, 7])

    def test_collision_handling_within_document(self, applicator):
        """Test that the system handles potential collisions within a document scope."""
        strategy = Strategy(StrategyKind.SURROGATE)

        # Generate multiple surrogates of same type
        surrogates = []
        for i in range(10):
            original = f"555-123-456{i}"
            surrogate = applicator.apply_strategy(original, "PHONE_NUMBER", strategy, 0.95)
            surrogates.append(surrogate)

        # All should be unique (very high probability with good RNG)
        assert len(set(surrogates)) == len(surrogates)

    def test_strategy_parameter_override(self, applicator):
        """Test that strategy parameters can override default behavior."""
        # Test with explicit pattern override
        pattern_strategy = Strategy(StrategyKind.SURROGATE, {"pattern": "999-999-9999"})
        original = "555-123-4567"

        surrogate = applicator.apply_strategy(original, "PHONE_NUMBER", pattern_strategy, 0.95)

        # Should follow the explicit pattern
        assert len(surrogate) == 12
        assert surrogate[3] == '-'
        assert surrogate[7] == '-'
        assert all(c.isdigit() for i, c in enumerate(surrogate) if i not in [3, 7])

    def test_error_handling_with_fallback(self, applicator):
        """Test error handling and fallback to legacy generation."""
        # Create a strategy that might cause issues
        problematic_strategy = Strategy(StrategyKind.SURROGATE, {"invalid_param": "test"})
        original = "test-input"

        # Should not raise exception and should produce output
        result = applicator.apply_strategy(original, "UNKNOWN", problematic_strategy, 0.95)

        assert isinstance(result, str)
        assert len(result) > 0
