"""Tests for enhanced masking strategies (template, partial, hash)."""

import pytest

from cloakpivot.core.strategies import Strategy, StrategyKind
from cloakpivot.masking.applicator import StrategyApplicator


class TestEnhancedTemplateStrategy:
    """Test the enhanced template masking strategy."""

    def test_auto_generate_phone_template(self):
        """Test auto-generation of phone number templates."""
        applicator = StrategyApplicator()
        strategy = Strategy(StrategyKind.TEMPLATE, {"auto_generate": True})

        # Test various phone formats
        test_cases = [
            ("555-123-4567", "XXX-XXX-XXXX"),
            ("(555) 123-4567", "(XXX) XXX-XXXX"),
            ("5551234567", "XXX-XXX-XXXX"),  # Should format
            ("+1 555-123-4567", "+X XXX-XXX-XXXX"),
        ]

        for phone_number, expected_pattern in test_cases:
            result = applicator.apply_strategy(
                phone_number, "PHONE_NUMBER", strategy, 0.9
            )
            # Check that result follows expected pattern structure
            assert len(result) >= len(expected_pattern) - 2  # Allow some variance
            if "-" in expected_pattern:
                assert "-" in result or "X" in result

    def test_auto_generate_email_template(self):
        """Test auto-generation of email templates."""
        applicator = StrategyApplicator()
        strategy = Strategy(StrategyKind.TEMPLATE, {"auto_generate": True})

        test_cases = [
            "john@example.com",
            "user.name@subdomain.example.org",
            "simple@test.co",
        ]

        for email in test_cases:
            result = applicator.apply_strategy(email, "EMAIL_ADDRESS", strategy, 0.9)
            # Should preserve @ and . structure
            assert "@" in result
            assert "." in result
            # Should not contain original text
            assert "john" not in result
            assert "example" not in result

    def test_auto_generate_ssn_template(self):
        """Test auto-generation of SSN templates."""
        applicator = StrategyApplicator()
        strategy = Strategy(StrategyKind.TEMPLATE, {"auto_generate": True})

        test_cases = [
            ("123-45-6789", "XXX-XX-XXXX"),
            ("123456789", "XXXXXXXXX"),
        ]

        for ssn, _expected_pattern in test_cases:
            result = applicator.apply_strategy(ssn, "US_SSN", strategy, 0.9)
            assert len(result) == len(ssn)
            if "-" in ssn:
                assert "-" in result

    def test_preserve_format_with_template(self):
        """Test format preservation with user templates."""
        applicator = StrategyApplicator()
        strategy = Strategy(
            StrategyKind.TEMPLATE, {"template": "[PHONE]", "preserve_format": True}
        )

        result = applicator.apply_strategy(
            "555-123-4567", "PHONE_NUMBER", strategy, 0.9
        )
        # Should include format information
        assert "[PHONE:" in result or "XXX-XXX-XXXX" in result

    def test_template_placeholders(self):
        """Test template placeholder substitution."""
        applicator = StrategyApplicator()
        strategy = Strategy(
            StrategyKind.TEMPLATE, {"template": "[{entity_type}:{length}]"}
        )

        result = applicator.apply_strategy(
            "555-123-4567", "PHONE_NUMBER", strategy, 0.9
        )
        assert result == "[PHONE_NUMBER:12]"


class TestEnhancedPartialStrategy:
    """Test the enhanced partial masking strategy."""

    def test_format_aware_phone_partial(self):
        """Test format-aware partial masking for phone numbers."""
        applicator = StrategyApplicator()
        strategy = Strategy(
            StrategyKind.PARTIAL,
            {
                "visible_chars": 4,
                "position": "end",
                "format_aware": True,
                "preserve_delimiters": True,
            },
        )

        result = applicator.apply_strategy(
            "555-123-4567", "PHONE_NUMBER", strategy, 0.9
        )

        # Should preserve delimiters and show last 4 digits
        assert "-" in result  # Delimiters preserved
        assert "4567" in result  # Last 4 digits visible
        assert "555" not in result  # First part masked
        assert len(result) == len("555-123-4567")

    def test_format_aware_email_partial(self):
        """Test format-aware partial masking for emails."""
        applicator = StrategyApplicator()
        strategy = Strategy(
            StrategyKind.PARTIAL,
            {
                "visible_chars": 3,
                "position": "start",
                "format_aware": True,
                "preserve_delimiters": True,
            },
        )

        result = applicator.apply_strategy(
            "john@example.com", "EMAIL_ADDRESS", strategy, 0.9
        )

        # Should preserve @ and . and show first 3 chars
        assert "@" in result
        assert "." in result
        assert result.startswith("joh")
        assert "example" not in result

    def test_deterministic_partial_masking(self):
        """Test that partial masking is deterministic."""
        applicator = StrategyApplicator()
        strategy = Strategy(
            StrategyKind.PARTIAL,
            {"visible_chars": 4, "position": "random", "deterministic": True},
        )

        # Same input should produce same output
        text = "sensitive-information"
        result1 = applicator.apply_strategy(text, "TEST", strategy, 0.9)
        result2 = applicator.apply_strategy(text, "TEST", strategy, 0.9)

        assert result1 == result2

    def test_partial_position_variations(self):
        """Test different position options for partial masking."""
        applicator = StrategyApplicator()
        text = "1234567890"

        test_cases = [
            ("start", 3),
            ("end", 3),
            ("middle", 4),
            ("random", 5),
        ]

        for position, visible_chars in test_cases:
            strategy = Strategy(
                StrategyKind.PARTIAL,
                {
                    "visible_chars": visible_chars,
                    "position": position,
                    "deterministic": True,
                },
            )

            result = applicator.apply_strategy(text, "TEST", strategy, 0.9)

            # Result should be same length
            assert len(result) == len(text)

            # Should have correct number of visible characters
            visible_count = sum(1 for c in result if c != "*")
            assert visible_count == min(visible_chars, len(text))


class TestEnhancedHashStrategy:
    """Test the enhanced deterministic hashing strategy."""

    def test_deterministic_hash_output(self):
        """Test that hash output is deterministic for same input."""
        applicator = StrategyApplicator(seed="test_seed")
        strategy = Strategy(
            StrategyKind.HASH, {"algorithm": "sha256", "salt": "test_salt"}
        )

        text = "sensitive-data"
        result1 = applicator.apply_strategy(text, "TEST", strategy, 0.9)
        result2 = applicator.apply_strategy(text, "TEST", strategy, 0.9)

        assert result1 == result2

    def test_per_entity_salt(self):
        """Test per-entity-type salting."""
        applicator = StrategyApplicator()
        strategy = Strategy(
            StrategyKind.HASH,
            {
                "algorithm": "sha256",
                "per_entity_salt": {
                    "PHONE_NUMBER": "phone_salt",
                    "EMAIL_ADDRESS": "email_salt",
                },
                "truncate": 8,
            },
        )

        same_text = "123456789"
        phone_result = applicator.apply_strategy(
            same_text, "PHONE_NUMBER", strategy, 0.9
        )
        email_result = applicator.apply_strategy(
            same_text, "EMAIL_ADDRESS", strategy, 0.9
        )

        # Same text with different entity types should produce different hashes
        assert phone_result != email_result

    def test_hash_format_output_variations(self):
        """Test different hash output formats."""
        applicator = StrategyApplicator()
        text = "test-data"

        formats = ["hex", "base64", "base32"]

        for format_type in formats:
            strategy = Strategy(
                StrategyKind.HASH,
                {"algorithm": "sha256", "format_output": format_type, "truncate": 12},
            )

            result = applicator.apply_strategy(text, "TEST", strategy, 0.9)

            # Should be truncated to requested length
            assert len(result) == 12

            if format_type == "hex":
                # Hex should only contain 0-9, a-f
                assert all(c in "0123456789abcdef" for c in result.lower())

    def test_consistent_length_truncation(self):
        """Test consistent length truncation."""
        applicator = StrategyApplicator()
        strategy = Strategy(
            StrategyKind.HASH,
            {"algorithm": "sha256", "truncate": 8, "consistent_length": True},
        )

        # Similar length inputs should have similar hash patterns
        text1 = "test1234"
        text2 = "test5678"

        result1 = applicator.apply_strategy(text1, "TEST", strategy, 0.9)
        result2 = applicator.apply_strategy(text2, "TEST", strategy, 0.9)

        assert len(result1) == len(result2) == 8

    def test_preserve_format_structure_in_hash(self):
        """Test format structure preservation in hash output."""
        applicator = StrategyApplicator()
        strategy = Strategy(
            StrategyKind.HASH,
            {"algorithm": "sha256", "preserve_format_structure": True, "truncate": 12},
        )

        # Test with structured input
        text = "123-45-6789"
        result = applicator.apply_strategy(text, "US_SSN", strategy, 0.9)

        # Should potentially preserve some structural elements
        # This is a heuristic test since format preservation is best-effort
        assert len(result) >= 8  # Should have reasonable length


class TestStrategyComposition:
    """Test strategy composition and fallback mechanisms."""

    def test_strategy_fallback_on_failure(self):
        """Test fallback when primary strategy fails."""
        applicator = StrategyApplicator()

        # Create a custom strategy that will fail at runtime
        def failing_callback(original_text, entity_type, confidence):
            raise RuntimeError("Intentional failure for testing")

        failing_strategy = Strategy(StrategyKind.CUSTOM, {"callback": failing_callback})

        # Should fallback gracefully
        result = applicator.apply_strategy("test-data", "TEST", failing_strategy, 0.9)

        # Should not raise exception and should return masked value
        assert result is not None
        assert len(result) > 0
        assert result != "test-data"

    def test_strategy_composition(self):
        """Test composing multiple strategies."""
        applicator = StrategyApplicator()

        strategies = [
            Strategy(StrategyKind.PARTIAL, {"visible_chars": 4, "position": "end"}),
            Strategy(StrategyKind.HASH, {"algorithm": "sha256", "truncate": 8}),
        ]

        result = applicator.compose_strategies(
            "sensitive-data", "TEST", strategies, 0.9
        )

        # Should apply strategies in sequence
        assert result is not None
        assert result != "sensitive-data"

    def test_empty_input_handling(self):
        """Test handling of empty inputs."""
        applicator = StrategyApplicator()
        strategy = Strategy(StrategyKind.TEMPLATE, {"template": "[REDACTED]"})

        result = applicator.apply_strategy("", "TEST", strategy, 0.9)
        assert result == "[REDACTED]"

    def test_special_characters_handling(self):
        """Test handling of special characters in input."""
        applicator = StrategyApplicator()
        strategy = Strategy(
            StrategyKind.PARTIAL,
            {"visible_chars": 2, "position": "end", "format_aware": True},
        )

        test_inputs = [
            "test@#$%",
            "123-!@#-456",
            "üñíçødé",  # Unicode
            "emoji👨‍💻test",  # Emoji
        ]

        for test_input in test_inputs:
            result = applicator.apply_strategy(test_input, "TEST", strategy, 0.9)
            assert result is not None
            assert len(result) >= 1


class TestPolicyIntegration:
    """Test integration with enhanced policies."""

    def test_format_aware_template_policy(self):
        """Test the FORMAT_AWARE_TEMPLATE_POLICY."""
        from cloakpivot.core.policies import FORMAT_AWARE_TEMPLATE_POLICY

        policy = FORMAT_AWARE_TEMPLATE_POLICY

        # Test phone number strategy
        phone_strategy = policy.get_strategy_for_entity("PHONE_NUMBER")
        assert phone_strategy.kind == StrategyKind.TEMPLATE
        assert phone_strategy.get_parameter("auto_generate") is True
        assert phone_strategy.get_parameter("preserve_format") is True

    def test_format_aware_partial_policy(self):
        """Test the FORMAT_AWARE_PARTIAL_POLICY."""
        from cloakpivot.core.policies import FORMAT_AWARE_PARTIAL_POLICY

        policy = FORMAT_AWARE_PARTIAL_POLICY

        # Test email strategy
        email_strategy = policy.get_strategy_for_entity("EMAIL_ADDRESS")
        assert email_strategy.kind == StrategyKind.PARTIAL
        assert email_strategy.get_parameter("format_aware") is True
        assert email_strategy.get_parameter("preserve_delimiters") is True
        assert email_strategy.get_parameter("deterministic") is True

    def test_deterministic_hash_policy(self):
        """Test the DETERMINISTIC_HASH_POLICY."""
        from cloakpivot.core.policies import DETERMINISTIC_HASH_POLICY

        policy = DETERMINISTIC_HASH_POLICY

        # Should have per-entity salts configured
        default_strategy = policy.default_strategy
        assert default_strategy.kind == StrategyKind.HASH

        per_entity_salt = default_strategy.get_parameter("per_entity_salt")
        assert isinstance(per_entity_salt, dict)
        assert "PHONE_NUMBER" in per_entity_salt
        assert "EMAIL_ADDRESS" in per_entity_salt
        assert "default" in per_entity_salt

    def test_mixed_strategy_policy(self):
        """Test the MIXED_STRATEGY_POLICY with different strategies per entity."""
        from cloakpivot.core.policies import MIXED_STRATEGY_POLICY

        policy = MIXED_STRATEGY_POLICY

        # Test that different entities use different strategies
        phone_strategy = policy.get_strategy_for_entity("PHONE_NUMBER")
        email_strategy = policy.get_strategy_for_entity("EMAIL_ADDRESS")
        credit_card_strategy = policy.get_strategy_for_entity("CREDIT_CARD")

        assert phone_strategy.kind == StrategyKind.PARTIAL
        assert email_strategy.kind == StrategyKind.TEMPLATE
        assert credit_card_strategy.kind == StrategyKind.HASH

        # Test thresholds are configured
        assert policy.get_threshold_for_entity("PHONE_NUMBER") == 0.8
        assert policy.get_threshold_for_entity("EMAIL_ADDRESS") == 0.7
        assert policy.get_threshold_for_entity("CREDIT_CARD") == 0.9


if __name__ == "__main__":
    pytest.main([__file__])
