"""Tests for surrogate generation functionality."""

import pytest

from cloakpivot.core.surrogate import (
    FormatPattern,
    SurrogateGenerator,
    SurrogateQualityMetrics,
)


class TestFormatPattern:
    """Test format pattern analysis and preservation."""

    def test_analyze_phone_pattern(self):
        """Test phone number pattern analysis."""
        pattern = FormatPattern.analyze("555-123-4567")
        assert pattern.original_length == 12
        assert pattern.digit_positions == [0, 1, 2, 4, 5, 6, 8, 9, 10, 11]
        assert pattern.separator_positions == {3: '-', 7: '-'}
        assert pattern.character_classes == "DDDSDDDSDDDD"  # D=digit, S=separator
        assert pattern.detected_format == "phone"

    def test_analyze_ssn_pattern(self):
        """Test SSN pattern analysis."""
        pattern = FormatPattern.analyze("123-45-6789")
        assert pattern.original_length == 11
        assert pattern.digit_positions == [0, 1, 2, 4, 5, 7, 8, 9, 10]
        assert pattern.separator_positions == {3: '-', 6: '-'}
        assert pattern.character_classes == "DDDSDDSDDDD"
        assert pattern.detected_format == "ssn"

    def test_analyze_email_pattern(self):
        """Test email pattern analysis."""
        pattern = FormatPattern.analyze("user@example.com")
        assert pattern.original_length == 16
        assert '@' in pattern.special_chars
        assert '.' in pattern.special_chars
        assert pattern.detected_format == "email"

    def test_analyze_credit_card_pattern(self):
        """Test credit card pattern analysis."""
        pattern = FormatPattern.analyze("4532-1234-5678-9012")
        assert pattern.original_length == 19
        assert len(pattern.digit_positions) == 16
        assert pattern.detected_format == "credit_card"


class TestSurrogateGenerator:
    """Test surrogate generation functionality."""

    @pytest.fixture
    def generator(self):
        """Create a SurrogateGenerator instance."""
        return SurrogateGenerator(seed="test_seed_123")

    def test_deterministic_generation(self, generator):
        """Test that generation is deterministic with same input and seed."""
        original = "555-123-4567"
        entity_type = "PHONE_NUMBER"

        result1 = generator.generate_surrogate(original, entity_type)
        result2 = generator.generate_surrogate(original, entity_type)

        assert result1 == result2
        assert result1 != original
        assert len(result1) == len(original)

    def test_format_preservation_phone(self, generator):
        """Test format preservation for phone numbers."""
        original = "555-123-4567"
        surrogate = generator.generate_surrogate(original, "PHONE_NUMBER")

        # Should preserve format
        assert len(surrogate) == len(original)
        assert surrogate[3] == '-'
        assert surrogate[7] == '-'
        assert all(c.isdigit() for i, c in enumerate(surrogate) if i not in [3, 7])

    def test_format_preservation_ssn(self, generator):
        """Test format preservation for SSN."""
        original = "123-45-6789"
        surrogate = generator.generate_surrogate(original, "US_SSN")

        # Should preserve format
        assert len(surrogate) == len(original)
        assert surrogate[3] == '-'
        assert surrogate[6] == '-'
        assert all(c.isdigit() for i, c in enumerate(surrogate) if i not in [3, 6])

    def test_format_preservation_email(self, generator):
        """Test format preservation for email addresses."""
        original = "user@example.com"
        surrogate = generator.generate_surrogate(original, "EMAIL_ADDRESS")

        # Should preserve email structure
        assert '@' in surrogate
        assert '.' in surrogate
        parts = surrogate.split('@')
        assert len(parts) == 2
        assert '.' in parts[1]  # Domain should have dot

    def test_different_seeds_produce_different_results(self):
        """Test that different seeds produce different results."""
        original = "555-123-4567"
        entity_type = "PHONE_NUMBER"

        gen1 = SurrogateGenerator(seed="seed1")
        gen2 = SurrogateGenerator(seed="seed2")

        result1 = gen1.generate_surrogate(original, entity_type)
        result2 = gen2.generate_surrogate(original, entity_type)

        assert result1 != result2
        # But both should preserve format
        assert len(result1) == len(result2) == len(original)

    def test_collision_detection(self, generator):
        """Test collision detection and resolution."""
        original1 = "555-123-4567"
        original2 = "555-987-6543"

        # Generate multiple surrogates
        surrogate1 = generator.generate_surrogate(original1, "PHONE_NUMBER")
        surrogate2 = generator.generate_surrogate(original2, "PHONE_NUMBER")

        # Should be different (very low probability of collision with good RNG)
        assert surrogate1 != surrogate2

    def test_surrogate_uniqueness_within_scope(self, generator):
        """Test that surrogates are unique within a document scope."""
        originals = [f"555-123-456{i}" for i in range(10)]
        surrogates = []

        for original in originals:
            surrogate = generator.generate_surrogate(original, "PHONE_NUMBER")
            surrogates.append(surrogate)

        # All should be unique
        assert len(set(surrogates)) == len(surrogates)

    def test_validation_no_real_data_leakage(self, generator):
        """Test that surrogates don't accidentally contain real data."""
        original = "555-123-4567"
        surrogate = generator.generate_surrogate(original, "PHONE_NUMBER")

        # Should not contain original digits in same positions
        assert surrogate != original

        # Get quality metrics
        metrics = generator.get_quality_metrics()
        assert metrics.format_preservation_rate > 0.9
        assert metrics.uniqueness_rate > 0.9

    def test_custom_pattern_generation(self, generator):
        """Test custom pattern-based generation."""
        pattern = "XXX-XX-9999"  # X=letter, 9=digit
        surrogate = generator.generate_from_pattern(pattern)

        assert len(surrogate) == len(pattern)
        assert surrogate[3] == '-'
        assert surrogate[6] == '-'
        assert surrogate[:3].isalpha()
        assert surrogate[4:6].isalpha()
        assert surrogate[7:].isdigit()

    def test_entity_specific_rules(self, generator):
        """Test entity-specific generation rules."""
        # Test different entity types produce appropriate formats
        phone_surrogate = generator.generate_surrogate("555-123-4567", "PHONE_NUMBER")
        ssn_surrogate = generator.generate_surrogate("123-45-6789", "US_SSN")
        email_surrogate = generator.generate_surrogate("user@example.com", "EMAIL_ADDRESS")

        # Each should follow appropriate patterns
        assert '-' in phone_surrogate
        assert phone_surrogate.replace('-', '').isdigit()

        assert '-' in ssn_surrogate
        assert ssn_surrogate.replace('-', '').isdigit()

        assert '@' in email_surrogate and '.' in email_surrogate


class TestSurrogateQualityMetrics:
    """Test quality metrics and validation."""

    def test_format_preservation_measurement(self):
        """Test format preservation measurement."""
        metrics = SurrogateQualityMetrics()

        # Record successful format preservation
        metrics.record_generation("555-123-4567", "555-987-6543", True, True, True)
        metrics.record_generation("123-45-6789", "987-65-4321", True, True, True)

        assert metrics.total_generated == 2
        assert metrics.format_preservation_rate == 1.0
        assert metrics.uniqueness_rate == 1.0
        assert metrics.validation_success_rate == 1.0

    def test_quality_degradation_tracking(self):
        """Test tracking when quality degrades."""
        metrics = SurrogateQualityMetrics()

        # Mixed results
        metrics.record_generation("555-123-4567", "555-987-6543", True, True, True)
        metrics.record_generation("bad-input", "fallback", False, True, False)

        assert metrics.total_generated == 2
        assert metrics.format_preservation_rate == 0.5
        assert metrics.validation_success_rate == 0.5

    def test_collision_tracking(self):
        """Test collision detection and tracking."""
        metrics = SurrogateQualityMetrics()

        # Simulate collision
        metrics.record_collision("555-123-4567", "duplicate_value")

        assert metrics.collision_count == 1
        assert "duplicate_value" in metrics.collision_examples
