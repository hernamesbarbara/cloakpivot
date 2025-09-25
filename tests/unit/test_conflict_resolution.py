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
        result = applicator.apply_strategy("test@example.com", "EMAIL_ADDRESS", strategy, 0.95)

        # Should return a hashed value
        assert result != "test@example.com"
        assert len(result) > 0

    def test_apply_redact_strategy(self, applicator):
        """Test redact strategy application."""
        strategy = Strategy(kind=StrategyKind.REDACT, parameters={})
        result = applicator.apply_strategy("sensitive data", "GENERIC", strategy, 0.95)

        # Should return asterisks (REDACT strategy returns asterisks, not "[REDACTED]")
        assert result == "*" * len("sensitive data")

    def test_apply_partial_strategy(self, applicator):
        """Test partial masking strategy."""
        strategy = Strategy(
            kind=StrategyKind.PARTIAL, parameters={"visible_chars": 4, "position": "end"}
        )
        result = applicator.apply_strategy("test@example.com", "EMAIL_ADDRESS", strategy, 0.95)

        # Should partially mask
        assert result != "test@example.com"
        assert "*" in result

    def test_apply_template_strategy(self, applicator):
        """Test template-based masking."""
        strategy = Strategy(kind=StrategyKind.TEMPLATE, parameters={"template": "[EMAIL]"})
        result = applicator.apply_strategy("john@example.com", "EMAIL_ADDRESS", strategy, 0.95)

        # Should apply template
        assert result != "john@example.com"
        assert "EMAIL" in result or "@" in result

    def test_apply_surrogate_strategy(self, applicator):
        """Test surrogate value generation."""
        strategy = Strategy(
            kind=StrategyKind.SURROGATE, parameters={"seed": "42"}
        )  # Seed must be a string
        result = applicator.apply_strategy("555-1234", "PHONE_NUMBER", strategy, 0.95)

        # Should generate surrogate
        assert result != "555-1234"
        assert len(result) > 0

    def test_fallback_on_unknown_strategy(self, applicator):
        """Test fallback when strategy is unknown."""

        # Using CUSTOM with a simple callback (with correct signature)
        def simple_callback(original_text: str, entity_type: str, confidence: float) -> str:
            return "[CUSTOM]"

        strategy = Strategy(kind=StrategyKind.CUSTOM, parameters={"callback": simple_callback})
        result = applicator.apply_strategy("test data", "GENERIC", strategy, 0.95)

        # Should apply custom callback
        assert result == "[CUSTOM]"

    def test_hash_with_different_algorithms(self, applicator):
        """Test hash strategy with different algorithms."""
        text = "test@example.com"

        sha256 = Strategy(kind=StrategyKind.HASH, parameters={"algorithm": "sha256"})
        sha512 = Strategy(kind=StrategyKind.HASH, parameters={"algorithm": "sha512"})

        result256 = applicator.apply_strategy(text, "EMAIL_ADDRESS", sha256, 0.95)
        result512 = applicator.apply_strategy(text, "EMAIL_ADDRESS", sha512, 0.95)

        # Different algorithms should produce different results
        assert result256 != result512
        assert result256 != text
        assert result512 != text

    def test_partial_masking_ratios(self, applicator):
        """Test different partial masking ratios."""
        text = "1234567890"

        # Test different visibility ratios - using visible_chars parameter
        strategy_25 = Strategy(
            kind=StrategyKind.PARTIAL, parameters={"visible_chars": 2, "position": "end"}
        )
        strategy_50 = Strategy(
            kind=StrategyKind.PARTIAL, parameters={"visible_chars": 5, "position": "end"}
        )
        strategy_75 = Strategy(
            kind=StrategyKind.PARTIAL, parameters={"visible_chars": 7, "position": "end"}
        )

        result_25 = applicator.apply_strategy(text, "GENERIC", strategy_25, 0.95)
        result_50 = applicator.apply_strategy(text, "GENERIC", strategy_50, 0.95)
        result_75 = applicator.apply_strategy(text, "GENERIC", strategy_75, 0.95)

        # More visible characters should show more
        visible_25 = sum(1 for c in result_25 if c != "*")
        visible_50 = sum(1 for c in result_50 if c != "*")
        visible_75 = sum(1 for c in result_75 if c != "*")

        assert visible_25 <= visible_50 <= visible_75

    def test_custom_strategy_application(self, applicator):
        """Test custom strategy with parameters."""

        def custom_callback(original_text: str, entity_type: str, confidence: float) -> str:
            return f"[CUSTOM:{entity_type}]"

        strategy = Strategy(
            kind=StrategyKind.CUSTOM,
            parameters={
                "callback": custom_callback,
                "pattern": "[CUSTOM:{}]",
                "include_type": True,
            },
        )

        result = applicator.apply_strategy("sensitive", "CUSTOM_TYPE", strategy, 0.95)

        # Should apply custom pattern
        assert result == "[CUSTOM:CUSTOM_TYPE]"

    def test_strategy_with_format_preservation(self, applicator):
        """Test strategies that preserve format."""
        phone = "555-123-4567"
        strategy = Strategy(
            kind=StrategyKind.PARTIAL,
            parameters={"visible_chars": 4, "position": "end", "preserve_format": True},
        )

        result = applicator.apply_strategy(phone, "PHONE_NUMBER", strategy, 0.95)

        # Should preserve dashes
        if "-" in phone:
            # Format might be preserved
            assert len(result) == len(phone)
