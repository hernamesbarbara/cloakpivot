"""Tests for conflict resolution in masking operations."""

import pytest
from cloakpivot.masking.applicator import StrategyApplicator
from cloakpivot.core.strategies import Strategy


class TestConflictResolution:
    """Test conflict resolution and strategy application."""

    @pytest.fixture
    def applicator(self):
        """Create a StrategyApplicator instance."""
        return StrategyApplicator()

    def test_apply_hash_strategy(self, applicator):
        """Test hash strategy application."""
        strategy = Strategy(name="hash", params={"algorithm": "sha256"})
        result = applicator.apply_strategy("test@example.com", strategy)

        # Should return a hashed value
        assert result != "test@example.com"
        assert len(result) > 0

    def test_apply_redact_strategy(self, applicator):
        """Test redact strategy application."""
        strategy = Strategy(name="redact", params={})
        result = applicator.apply_strategy("sensitive data", strategy)

        # Should return redacted
        assert result == "[REDACTED]"

    def test_apply_partial_strategy(self, applicator):
        """Test partial masking strategy."""
        strategy = Strategy(name="partial", params={"visible_ratio": 0.5})
        result = applicator.apply_strategy("test@example.com", strategy)

        # Should partially mask
        assert result != "test@example.com"
        assert "*" in result

    def test_apply_template_strategy(self, applicator):
        """Test template-based masking."""
        strategy = Strategy(name="template", params={})
        result = applicator.apply_strategy("john@example.com", strategy, entity_type="EMAIL")

        # Should apply template
        assert result != "john@example.com"
        assert "EMAIL" in result or "@" in result

    def test_apply_surrogate_strategy(self, applicator):
        """Test surrogate value generation."""
        strategy = Strategy(name="surrogate", params={"seed": 42})
        result = applicator.apply_strategy("555-1234", strategy, entity_type="PHONE_NUMBER")

        # Should generate surrogate
        assert result != "555-1234"
        assert len(result) > 0

    def test_fallback_on_unknown_strategy(self, applicator):
        """Test fallback when strategy is unknown."""
        strategy = Strategy(name="unknown_strategy", params={})
        result = applicator.apply_strategy("test data", strategy)

        # Should fall back to redaction
        assert result == "[REDACTED]"

    def test_hash_with_different_algorithms(self, applicator):
        """Test hash strategy with different algorithms."""
        text = "test@example.com"

        sha256 = Strategy(name="hash", params={"algorithm": "sha256"})
        sha512 = Strategy(name="hash", params={"algorithm": "sha512"})

        result256 = applicator.apply_strategy(text, sha256)
        result512 = applicator.apply_strategy(text, sha512)

        # Different algorithms should produce different results
        assert result256 != result512
        assert result256 != text
        assert result512 != text

    def test_partial_masking_ratios(self, applicator):
        """Test different partial masking ratios."""
        text = "1234567890"

        # Test different visibility ratios
        strategy_25 = Strategy(name="partial", params={"visible_ratio": 0.25})
        strategy_50 = Strategy(name="partial", params={"visible_ratio": 0.5})
        strategy_75 = Strategy(name="partial", params={"visible_ratio": 0.75})

        result_25 = applicator.apply_strategy(text, strategy_25)
        result_50 = applicator.apply_strategy(text, strategy_50)
        result_75 = applicator.apply_strategy(text, strategy_75)

        # More visible ratio should show more characters
        visible_25 = sum(1 for c in result_25 if c != '*')
        visible_50 = sum(1 for c in result_50 if c != '*')
        visible_75 = sum(1 for c in result_75 if c != '*')

        assert visible_25 <= visible_50 <= visible_75

    def test_custom_strategy_application(self, applicator):
        """Test custom strategy with parameters."""
        strategy = Strategy(
            name="custom",
            params={
                "pattern": "[CUSTOM:{}]",
                "include_type": True
            }
        )

        result = applicator.apply_strategy(
            "sensitive",
            strategy,
            entity_type="CUSTOM_TYPE"
        )

        # Should apply custom pattern
        assert result != "sensitive"

    def test_strategy_with_format_preservation(self, applicator):
        """Test strategies that preserve format."""
        phone = "555-123-4567"
        strategy = Strategy(name="partial", params={"preserve_format": True})

        result = applicator.apply_strategy(phone, strategy, entity_type="PHONE_NUMBER")

        # Should preserve dashes
        if "-" in phone:
            # Format might be preserved
            assert len(result) == len(phone)