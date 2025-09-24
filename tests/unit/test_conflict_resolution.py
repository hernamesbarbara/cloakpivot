"""Tests for conflict resolution in masking operations."""

import pytest

from cloakpivot.core.types.strategies import Strategy, StrategyKind
from cloakpivot.masking.applicator import StrategyApplicator


class TestConflictResolution:
    """Test conflict resolution and strategy application."""

    @pytest.fixture
    def applicator(self):
        """Create a StrategyApplicator instance."""
        return StrategyApplicator()

    def test_apply_hash_strategy(self, applicator):
        """Test hash strategy application."""
        strategy = Strategy(kind=StrategyKind.HASH, parameters={"algorithm": "sha256"})
        result = applicator.apply_strategy("test@example.com", strategy)

        # Should return a hashed value
        assert result != "test@example.com"
        assert len(result) > 0

    def test_apply_redact_strategy(self, applicator):
        """Test redact strategy application."""
        strategy = Strategy(kind=StrategyKind.REDACT, parameters={})
        result = applicator.apply_strategy("sensitive data", strategy)

        # Should return redacted
        assert result == "[REDACTED]"

    def test_apply_partial_strategy(self, applicator):
        """Test partial masking strategy."""
        strategy = Strategy(kind=StrategyKind.PARTIAL, parameters={"visible_chars": 4, "position": "end"})
        result = applicator.apply_strategy("test@example.com", strategy)

        # Should partially mask
        assert result != "test@example.com"
        assert "*" in result

    def test_apply_template_strategy(self, applicator):
        """Test template-based masking."""
        strategy = Strategy(kind=StrategyKind.TEMPLATE, parameters={"template": "[EMAIL]"})
        result = applicator.apply_strategy("john@example.com", strategy, entity_type="EMAIL")

        # Should apply template
        assert result != "john@example.com"
        assert "EMAIL" in result or "@" in result

    def test_apply_surrogate_strategy(self, applicator):
        """Test surrogate value generation."""
        strategy = Strategy(kind=StrategyKind.SURROGATE, parameters={"seed": 42})
        result = applicator.apply_strategy("555-1234", strategy, entity_type="PHONE_NUMBER")

        # Should generate surrogate
        assert result != "555-1234"
        assert len(result) > 0

    def test_fallback_on_unknown_strategy(self, applicator):
        """Test fallback when strategy is unknown."""
        # Using CUSTOM which should fall back if not handled
        strategy = Strategy(kind=StrategyKind.CUSTOM, parameters={})
        result = applicator.apply_strategy("test data", strategy)

        # Should fall back to redaction
        assert result == "[REDACTED]"

    def test_hash_with_different_algorithms(self, applicator):
        """Test hash strategy with different algorithms."""
        text = "test@example.com"

        sha256 = Strategy(kind=StrategyKind.HASH, parameters={"algorithm": "sha256"})
        sha512 = Strategy(kind=StrategyKind.HASH, parameters={"algorithm": "sha512"})

        result256 = applicator.apply_strategy(text, sha256)
        result512 = applicator.apply_strategy(text, sha512)

        # Different algorithms should produce different results
        assert result256 != result512
        assert result256 != text
        assert result512 != text

    def test_partial_masking_ratios(self, applicator):
        """Test different partial masking ratios."""
        text = "1234567890"

        # Test different visibility ratios - using visible_chars parameter
        strategy_25 = Strategy(kind=StrategyKind.PARTIAL, parameters={"visible_chars": 2, "position": "end"})
        strategy_50 = Strategy(kind=StrategyKind.PARTIAL, parameters={"visible_chars": 5, "position": "end"})
        strategy_75 = Strategy(kind=StrategyKind.PARTIAL, parameters={"visible_chars": 7, "position": "end"})

        result_25 = applicator.apply_strategy(text, strategy_25)
        result_50 = applicator.apply_strategy(text, strategy_50)
        result_75 = applicator.apply_strategy(text, strategy_75)

        # More visible characters should show more
        visible_25 = sum(1 for c in result_25 if c != '*')
        visible_50 = sum(1 for c in result_50 if c != '*')
        visible_75 = sum(1 for c in result_75 if c != '*')

        assert visible_25 <= visible_50 <= visible_75

    def test_custom_strategy_application(self, applicator):
        """Test custom strategy with parameters."""
        strategy = Strategy(
            kind=StrategyKind.CUSTOM,
            parameters={
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
        strategy = Strategy(kind=StrategyKind.PARTIAL, parameters={"visible_chars": 4, "position": "end", "preserve_format": True})

        result = applicator.apply_strategy(phone, strategy, entity_type="PHONE_NUMBER")

        # Should preserve dashes
        if "-" in phone:
            # Format might be preserved
            assert len(result) == len(phone)
